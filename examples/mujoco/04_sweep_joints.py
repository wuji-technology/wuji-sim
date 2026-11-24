# SPDX-License-Identifier: Apache-2.0
import argparse
import math
import sys
import time

import mujoco
import numpy as np

from wuji_sim import assets, conventions, framing, mit, profiles, records, scene
from wuji_sim.platforms import mujoco as platform

DEMO = "04_sweep_joints"
RENDERING = "MuJoCo physics / MuJoCo rendering"
VIDEO_FPS = 25
VIDEO_CAMERA = scene.PREVIEW_CAMERA
CONTROL_LAW = "tau = clip(kp*(q_des-q) + kd*(qd_des-qd) + tau_ff, -tau_limit, tau_limit)"
TAU_APPLIED = (
    "data.qfrc_actuator at the joint dof after mj_step: the actuator torque that step applied after the general-"
    "actuator forcerange and joint actuatorfrcrange clamps; frictionloss, joint-limit and contact forces are not "
    "included"
)


def steps_per_frame(timestep: float) -> int:
    ratio = 1.0 / (VIDEO_FPS * timestep)
    steps = round(ratio)
    if steps < 1 or not math.isclose(ratio, steps, rel_tol=1e-9):
        raise ValueError(f"video frame period 1/{VIDEO_FPS} s is not a whole number of {timestep} s physics steps")
    return steps


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Sweep each joint between its inner limits with MIT torque control, one joint at a time."
    )
    assets.add_cli_args(parser)
    records.add_cli_args(parser)
    scene.add_cli_args(parser)
    args = parser.parse_args(argv[1:])

    asset = assets.resolve(args.hand, args.asset_root, strict=args.strict)
    print(asset.describe())
    run = records.RunRecord(platform=platform.PLATFORM, demo=DEMO, hand=args.hand,
                            output_dir=args.output_dir, argv=argv)
    view = framing.view(asset.mjcf, args.hand, platform.PLATFORM, "sweep")
    loaded = platform.load(asset.mjcf, scene=lambda spec: scene.add_scene(spec, args.hand, view),
                           description_version=asset.expected_version)
    model, data = loaded.model, loaded.data
    joint_names = loaded.joint_names
    description = scene.description_joint_ranges(asset.mjcf)
    effective = {name: loaded.report["effective_values"]["joints"][name]["range"] for name in joint_names}
    scene.check_joint_table(args.hand, description, effective)
    print(f"joint table: {len(joint_names)} joints; names and compiled ranges match {asset.mjcf}")
    ranges = np.array([effective[name] for name in joint_names])

    timestep = float(model.opt.timestep)
    frame_steps = steps_per_frame(timestep)
    n = len(joint_names)
    n_steps = scene.whole_steps(scene.sweep_duration(joint_names), timestep)
    qpos_adr = list(loaded.qpos_adr)
    dof_adr = list(loaded.dof_adr)
    actuator_ids = list(loaded.actuator_ids)
    tau_limit = np.array(loaded.tau_limit)
    kp = np.full(n, scene.DEMO_KP)
    kd = np.full(n, scene.DEMO_KD)
    tau_ff = np.zeros(n)
    stability = platform.mit_stability(loaded, kp, kd)
    platform.set_mit_gains(loaded, kp, kd)
    print(f"MIT stability ({stability['control_path']}): max ratio {stability['max_ratio']:.3f} "
          f"at {stability['max_joint']} (dt {timestep:g} s, limit {stability['limit']:g})")

    control = {
        "law": CONTROL_LAW,
        "kp": scene.DEMO_KP,
        "kd": scene.DEMO_KD,
        "tau_ff": 0.0,
        "units": {"kp": "N*m/rad", "kd": "N*m*s/rad", "tau_ff": "N*m", "tau_limit": "N*m",
                  "control_period": "s", "q": "rad", "qd": "rad/s"},
        "control_period": timestep,
        "joint_order": list(joint_names),
        "actuator_ids": actuator_ids,
        "tau_limit": tau_limit.tolist(),
        "stability": stability,
    }
    joint_table_check = {
        "description_mjcf": str(asset.mjcf),
        "joints": n,
        "compared": "hinge joint count, names and order; description MJCF range vs compiled model range",
        "range_atol_rad": scene.JOINT_RANGE_ATOL,
        "passed": True,
    }

    if args.headless:
        t = np.arange(n_steps) * timestep
        active = np.empty(n_steps, dtype=np.int64)
        series = {field: np.empty((n_steps, n)) for field in scene.SWEEP_SERIES}
        with (
            scene.SceneRenderer(model, *view.render_size) as renderer,
            scene.Display(model, data, view, headless=True) as display,
            run.open_video(VIDEO_CAMERA, VIDEO_FPS) as video,
        ):
            video.append(renderer.render(data, VIDEO_CAMERA))
            frame_started = time.perf_counter()
            for step in range(n_steps):
                q_des, qd_des, active[step] = scene.sweep_reference(joint_names, ranges, t[step])
                cmd = mit.MitCommand(joint_names, q_des, qd_des, kp, kd, tau_ff)
                q, qd = data.qpos[qpos_adr], data.qvel[dof_adr]
                tau = mit.mit_torque(cmd, q, qd, tau_limit)
                series["q_des"][step], series["q"][step], series["qd"][step] = q_des, q, qd
                series["tau_command"][step] = tau
                data.ctrl[actuator_ids] = kp * q_des + kd * qd_des + tau_ff
                mujoco.mj_step(model, data)
                series["tau_applied"][step] = data.qfrc_actuator[dof_adr]
                if (step + 1) % frame_steps == 0:
                    video.append(renderer.render(data, VIDEO_CAMERA))
                    display.sync()
                    if display.is_running():
                        time.sleep(max(0.0, 1.0 / VIDEO_FPS - (time.perf_counter() - frame_started)))
                        frame_started = time.perf_counter()
            frames = video.frames
            video_path = video.path

        error = series["q_des"] - series["q"]
        saturated = np.abs(series["tau_command"]) >= tau_limit
        rows = scene.sweep_joint_table(joint_names, description, effective, active, series["q"], error, saturated)
        flagged = scene.sweep_flags(rows)
        csv_path = run.artifact_path("timeseries", ".csv")
        scene.write_sweep_csv(csv_path, joint_names, t, active, series)
        title = f"{asset.product} {args.hand} {DEMO} ({RENDERING}; kp {scene.DEMO_KP}, kd {scene.DEMO_KD})"
        figure_path = run.save_figure("joint_curves", scene.sweep_figure(rows, t, series, title))

        results = {
            "scene": scene.scene_record(args.hand, view),
            "control": control,
            "joint_table_check": joint_table_check,
            "sweep": {
                "reference": scene.sweep_reference_record(joint_names),
                "duration_s": n_steps * timestep,
                "steps": n_steps,
                "error_definition": "q_des(t) - q(t), sampled before each physics step",
                "saturation_definition": "|tau_command| >= tau_limit",
                "active_window": "the sweep window of the joint's group (see reference.windows); run = the whole "
                                 "sweep, including holding open while other windows move",
                "error_flag_rad": scene.SWEEP_ERROR_FLAG_RAD,
                "flagged_joints": flagged,
                "joints": rows,
            },
            "proprioception": {
                "csv": str(csv_path),
                "columns": ["t_s", "active_window", *(f"{field}:<joint>" for field in scene.SWEEP_SERIES)],
                "units": {"t_s": "s", **scene.SWEEP_SERIES_UNITS},
                "sample_period_s": timestep,
                "q_qd": "data.qpos / data.qvel at the joint, sampled before each physics step",
                "tau_command": "clipped MIT torque from mit.mit_torque on the host, using the pre-step q and qd; "
                               "data.ctrl holds the affine actuator input kp*q_des + kd*qd_des + tau_ff",
                "tau_applied": TAU_APPLIED,
            },
            "figure": str(figure_path),
            "video": {"path": str(video_path), "camera": VIDEO_CAMERA, "fps": VIDEO_FPS, "frames": frames,
                      "steps_per_frame": frame_steps},
        }
        record_path = run.write(asset=asset.to_record(), load_report=loaded.report, results=results,
                                rendering=RENDERING)
        for row in rows:
            print(f"{row['joint']}: target {row['sweep_target_rad'][0]:+.3f}..{row['sweep_target_rad'][1]:+.3f} "
                  f"reached {row['reached_min_rad']:+.3f}..{row['reached_max_rad']:+.3f} rad, "
                  f"max |error| {row['max_abs_error_run_rad']:.4f} rad, "
                  f"saturation {row['saturation_fraction_run']:.4f}")
        print(f"flagged (saturation > 0 or max |error| > {scene.SWEEP_ERROR_FLAG_RAD} rad): {flagged}")
        print(f"record: {record_path}")
        print("artifacts: 1")
    else:
        results = {"scene": scene.scene_record(args.hand, view), "control": control,
                   "joint_table_check": joint_table_check}
        record_path = run.write(asset=asset.to_record(), load_report=loaded.report, results=results,
                                rendering=RENDERING)
        print(f"record: {record_path}")
        print("artifacts: 0")

        def frame(t: float) -> None:
            q_des, qd_des, _ = scene.sweep_reference(joint_names, ranges, t)
            data.ctrl[actuator_ids] = kp * q_des + kd * qd_des + tau_ff
            mujoco.mj_step(model, data)

        with scene.Display(model, data, view, headless=False) as display:
            scene.run_live(display, timestep, frame)


if __name__ == "__main__":
    try:
        main(sys.argv)
    except (assets.AssetError, profiles.ProfileError, mit.MitStabilityError, records.RecordError,
            scene.JointTableError, conventions.ViewerError, FileExistsError) as exc:
        sys.exit(f"error: {exc}")
