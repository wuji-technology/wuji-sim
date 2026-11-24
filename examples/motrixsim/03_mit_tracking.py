# SPDX-License-Identifier: Apache-2.0
import argparse
import csv
import math
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from wuji_sim import assets, conventions, framing, mit, profiles, records, scene
from wuji_sim.platforms import motrixsim as platform

sys.path.insert(0, str(Path(__file__).resolve().parent / "_shared"))

import rendering  # noqa: E402

DEMO = "03_mit_tracking"
RENDERING = "MotrixSim physics / MotrixSim rendering"
VIDEO_FPS = 25
VIDEO_CAMERA = scene.PREVIEW_CAMERA
TRAJECTORY_COLUMNS_NOTE = (
    "t_s then q_des:<joint> for every joint in results.control.joint_order, then q:<joint> for every joint in the "
    "same order; a --reference-trajectory file must share this header, joint order and an identical t_s grid"
)


def steps_per_frame(timestep: float) -> int:
    ratio = 1.0 / (VIDEO_FPS * timestep)
    steps = round(ratio)
    if steps < 1 or not math.isclose(ratio, steps, rel_tol=1e-9):
        raise ValueError(f"video frame period 1/{VIDEO_FPS} s is not a whole number of {timestep} s physics steps")
    return steps


def write_trajectory_csv(path: Path, joint_names: Sequence[str], t: np.ndarray, q_des: np.ndarray,
                         q: np.ndarray) -> None:
    with open(path, "x", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", *(f"q_des:{name}" for name in joint_names), *(f"q:{name}" for name in joint_names)])
        for k in range(len(t)):
            writer.writerow([f"{t[k]:.9g}", *(f"{value:.9g}" for value in q_des[k]),
                             *(f"{value:.9g}" for value in q[k])])


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Track the preset pose sequence with MIT torque control.")
    assets.add_cli_args(parser)
    records.add_cli_args(parser)
    scene.add_cli_args(parser)
    parser.add_argument(
        "--write-trajectory", action="store_true",
        help="also record q_des and q at every control step and write results.trajectory.csv with the schema "
             "in results.trajectory.columns_note (t_s then q_des:<joint>, then q:<joint>, in joint order) for "
             "use as another platform's --reference-trajectory; disabled by default, so a plain run keeps "
             "writing only its video; headless only",
    )
    args = parser.parse_args(argv[1:])

    asset = assets.resolve(args.hand, args.asset_root, strict=args.strict)
    print(asset.describe())
    run = records.RunRecord(platform=platform.PLATFORM, demo=DEMO, hand=args.hand,
                            output_dir=args.output_dir, argv=argv)
    view = framing.view(asset.mjcf, args.hand, platform.PLATFORM, "tracking")
    # The video uses only the preview camera, so the system camera is built with the view's own fovy.
    loaded = platform.load(asset.mjcf, scene=lambda spec: scene.add_scene(spec, args.hand, view),
                           description_version=asset.expected_version, render_resolution=view.render_size,
                           render_fovy=view.fovy_deg)

    timestep = float(loaded.model.opt.timestep)
    frame_steps = steps_per_frame(timestep)
    duration = scene.tracking_duration()
    n_steps = round(duration / timestep)
    joint_names = loaded.joint_names
    n = len(joint_names)
    tau_limit = np.array(loaded.tau_limit)
    kp = np.full(n, scene.DEMO_KP)
    kd = np.full(n, scene.DEMO_KD)
    tau_ff = np.zeros(n)
    stability = platform.mit_stability(loaded, kp, kd)
    platform.set_mit_gains(loaded, kp, kd)
    print(f"MIT stability ({stability['control_path']}): max ratio {stability['max_ratio']:.3f} "
          f"at {stability['max_joint']} (dt {timestep:g} s, limit {stability['limit']:g})")

    control = {
        "law": "tau = clip(kp*(q_des-q) + kd*(qd_des-qd) + tau_ff, -tau_limit, tau_limit)",
        "kp": scene.DEMO_KP,
        "kd": scene.DEMO_KD,
        "tau_ff": 0.0,
        "units": {"kp": "N*m/rad", "kd": "N*m*s/rad", "tau_ff": "N*m", "tau_limit": "N*m",
                  "control_period": "s", "q": "rad", "qd": "rad/s"},
        "control_period": timestep,
        "joint_order": list(joint_names),
        "actuator_ids": list(loaded.actuator_ids),
        "tau_limit": tau_limit.tolist(),
        "stability": stability,
    }

    if args.headless:
        errors = np.empty((n_steps, n))
        saturated = np.zeros(n, dtype=np.int64)
        if args.write_trajectory:
            t = np.arange(n_steps, dtype=np.float64) * timestep
            q_des_series = np.empty((n_steps, n))
            q_series = np.empty((n_steps, n))
        with (
            rendering.OffscreenRenderer(loaded, view, args.hand, view.fovy_deg) as renderer,
            run.open_video(VIDEO_CAMERA, VIDEO_FPS) as video,
        ):
            video.append(renderer.render(VIDEO_CAMERA))
            for step in range(n_steps):
                q_des, qd_des = scene.tracking_reference(joint_names, step * timestep)
                cmd = mit.MitCommand(joint_names, q_des, qd_des, kp, kd, tau_ff)
                q = np.array(loaded.scene_data.dof_pos, dtype=float)
                qd = np.array(loaded.scene_data.dof_vel, dtype=float)
                tau = mit.mit_torque(cmd, q, qd, tau_limit)
                errors[step] = q_des - q
                if args.write_trajectory:
                    q_des_series[step] = q_des
                    q_series[step] = q
                saturated += np.abs(tau) >= tau_limit
                platform.apply_mit(loaded, q_des, qd_des, tau_ff)
                platform.step(loaded)
                if (step + 1) % frame_steps == 0:
                    video.append(renderer.render(VIDEO_CAMERA))
            frames = video.frames
            video_path = video.path

        csv_path = None
        if args.write_trajectory:
            csv_path = run.artifact_path("trajectory", ".csv")
            write_trajectory_csv(csv_path, joint_names, t, q_des_series, q_series)

        rms = np.sqrt(np.mean(errors**2, axis=0))
        max_abs = np.max(np.abs(errors), axis=0)
        results = {
            "scene": scene.scene_record(args.hand, view),
            "control": control,
            "tracking": {
                "reference": {"sequence": list(scene.TRACKING_SEQUENCE), "move_s": scene.MOVE_S,
                              "hold_s": scene.HOLD_S, "profile": "min-jerk"},
                "duration_s": n_steps * timestep,
                "steps": n_steps,
                "error_definition": "q_des(t) - q(t), sampled before each physics step",
                "max_rms_error_rad": float(rms.max()),
                "max_abs_error_rad": float(max_abs.max()),
                "joints": {
                    name: {"rms_error_rad": float(rms[i]), "max_abs_error_rad": float(max_abs[i]),
                           "saturation_fraction": float(saturated[i] / n_steps)}
                    for i, name in enumerate(joint_names)
                },
            },
            "video": {"path": str(video_path), "camera": VIDEO_CAMERA, "fps": VIDEO_FPS, "frames": frames,
                      "steps_per_frame": frame_steps},
        }
        if args.write_trajectory:
            results["trajectory"] = {
                "csv": str(csv_path.resolve()),
                "columns_note": TRAJECTORY_COLUMNS_NOTE,
                "rows": n_steps,
            }
        record_path = run.write(asset=asset.to_record(), load_report=loaded.report, results=results,
                                rendering=RENDERING)
        print(f"tracking: max RMS error {rms.max():.4f} rad, max abs error {max_abs.max():.4f} rad")
        print(f"record: {record_path}")
        print(f"artifacts: {2 if args.write_trajectory else 1}")
    else:
        record_path = run.write(asset=asset.to_record(), load_report=loaded.report,
                                results={"scene": scene.scene_record(args.hand, view), "control": control},
                                rendering=RENDERING)
        print(f"record: {record_path}")
        print("artifacts: 0")

        def frame(t: float) -> None:
            q_des, qd_des = scene.tracking_reference(joint_names, t % duration)
            platform.apply_mit(loaded, q_des, qd_des, tau_ff)
            platform.step(loaded)

        with rendering.LiveDisplay(loaded, view, view.fovy_deg) as display:
            rendering.run_live(display, timestep, frame)


if __name__ == "__main__":
    try:
        main(sys.argv)
    except (assets.AssetError, profiles.ProfileError, mit.MitStabilityError, records.RecordError,
            conventions.ViewerError, FileExistsError) as exc:
        sys.exit(f"error: {exc}")
