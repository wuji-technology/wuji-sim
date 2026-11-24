# SPDX-License-Identifier: Apache-2.0
import argparse
import csv
import math
import os
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from wuji_sim import assets, conventions, framing, mit, profiles, records, scene, scene_isaaclab
from wuji_sim.platforms import isaaclab as platform

DEMO = "03_mit_tracking"
RENDERING = "Isaac Lab physics / Isaac Sim RTX camera rendering"
VIDEO_FPS = 25
VIDEO_CAMERA = scene.PREVIEW_CAMERA
PARITY_THRESHOLD_FORMULA = "round_up_to_2_sig_figs(2 * observed_rad)"
# Measured independently (PARITY_THRESHOLDS_SOURCE): a threshold derived from the run it gates always passes.
PARITY_THRESHOLDS_RAD: dict[str, dict[str, float]] | None = {
    "l_thumb_cmc_flex": {"max_abs_rad": 0.031, "rms_rad": 0.013},
    "l_thumb_cmc_abd": {"max_abs_rad": 0.031, "rms_rad": 0.011},
    "l_thumb_mcp": {"max_abs_rad": 0.034, "rms_rad": 0.011},
    "l_thumb_ip": {"max_abs_rad": 0.034, "rms_rad": 0.011},
    "l_index_finger_mcp_flex": {"max_abs_rad": 0.042, "rms_rad": 0.015},
    "l_index_finger_mcp_abd": {"max_abs_rad": 0.0045, "rms_rad": 0.0014},
    "l_index_finger_pip": {"max_abs_rad": 0.059, "rms_rad": 0.021},
    "l_index_finger_dip": {"max_abs_rad": 0.034, "rms_rad": 0.013},
    "l_middle_finger_mcp_flex": {"max_abs_rad": 0.042, "rms_rad": 0.013},
    "l_middle_finger_mcp_abd": {"max_abs_rad": 0.000032, "rms_rad": 0.000012},
    "l_middle_finger_pip": {"max_abs_rad": 0.059, "rms_rad": 0.018},
    "l_middle_finger_dip": {"max_abs_rad": 0.034, "rms_rad": 0.011},
    "l_ring_finger_mcp_flex": {"max_abs_rad": 0.042, "rms_rad": 0.013},
    "l_ring_finger_mcp_abd": {"max_abs_rad": 0.0022, "rms_rad": 0.00070},
    "l_ring_finger_pip": {"max_abs_rad": 0.059, "rms_rad": 0.018},
    "l_ring_finger_dip": {"max_abs_rad": 0.034, "rms_rad": 0.011},
    "l_pinky_mcp_flex": {"max_abs_rad": 0.041, "rms_rad": 0.013},
    "l_pinky_mcp_abd": {"max_abs_rad": 0.0044, "rms_rad": 0.0014},
    "l_pinky_pip": {"max_abs_rad": 0.059, "rms_rad": 0.018},
    "l_pinky_dip": {"max_abs_rad": 0.034, "rms_rad": 0.011},
}
PARITY_THRESHOLDS_SOURCE: str | None = (
    "independent MuJoCo vs Isaac Lab measurement, 2 deterministic runs: MuJoCo 03/05 run with its trajectory "
    "output enabled, then Isaac Lab 03/05 run with --reference-trajectory pointed at that file; threshold = "
    "observed x 2 rounded up to 2 significant figures"
)
TRAJECTORY_COLUMNS_NOTE = (
    "t_s then q_des:<joint> for every joint in results.control.joint_order, then q:<joint> for every joint in the "
    "same order; a --reference-trajectory file must share this header, joint order and an identical t_s grid"
)


class ReferenceTrajectoryError(ValueError):
    pass


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


def read_reference_trajectory(path: Path, joint_names: Sequence[str], t_expected: np.ndarray) -> np.ndarray:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReferenceTrajectoryError(f"cannot read reference trajectory {path}: {exc}") from exc
    rows = list(csv.reader(text.splitlines()))
    if not rows:
        raise ReferenceTrajectoryError(f"reference trajectory {path} is empty")
    header, data_rows = rows[0], rows[1:]
    expected_header = ["t_s", *(f"q_des:{name}" for name in joint_names), *(f"q:{name}" for name in joint_names)]
    if header != expected_header:
        raise ReferenceTrajectoryError(
            f"reference trajectory {path} header {header} does not match this run's joint order; expected "
            f"{expected_header}"
        )
    if len(data_rows) != len(t_expected):
        raise ReferenceTrajectoryError(
            f"reference trajectory {path} has {len(data_rows)} rows; this run has {len(t_expected)} steps"
        )
    values = np.empty((len(data_rows), len(header)), dtype=np.float64)
    for i, row in enumerate(data_rows):
        line = i + 2
        if len(row) != len(header):
            raise ReferenceTrajectoryError(
                f"reference trajectory {path} line {line} has {len(row)} cells; the header has {len(header)}"
            )
        for j, cell in enumerate(row):
            try:
                values[i, j] = float(cell)
            except ValueError:
                raise ReferenceTrajectoryError(
                    f"reference trajectory {path} line {line} column {header[j]!r}: {cell!r} is not a number"
                ) from None
    if not np.allclose(values[:, 0], t_expected, rtol=0.0, atol=1e-6):
        raise ReferenceTrajectoryError(f"reference trajectory {path} time grid differs from this run's grid by more "
                                       "than 1e-6 s")
    n = len(joint_names)
    q_reference = values[:, 1 + n:1 + 2 * n]
    nonfinite = np.argwhere(~np.isfinite(q_reference))
    if nonfinite.size:
        i, j = nonfinite[0]
        raise ReferenceTrajectoryError(
            f"reference trajectory {path} line {i + 2} column {header[1 + n + j]!r}: measured position "
            f"{q_reference[i, j]} is not finite"
        )
    return q_reference


def parity_results(joint_names: Sequence[str], q_isaac: np.ndarray, q_reference: np.ndarray,
                   thresholds: dict[str, dict[str, float]] | None, source: str | None) -> dict:
    diff = np.abs(q_isaac - q_reference)
    max_abs = np.max(diff, axis=0)
    rms = np.sqrt(np.mean(diff**2, axis=0))
    has_fixed_threshold = thresholds is not None and any(name in thresholds for name in joint_names)
    joints = {}
    for j, name in enumerate(joint_names):
        observed = {"max_abs_rad": float(max_abs[j]), "rms_rad": float(rms[j])}
        fixed = None if thresholds is None else thresholds.get(name)
        threshold = None if fixed is None else {"max_abs_rad": float(fixed["max_abs_rad"]),
                                                 "rms_rad": float(fixed["rms_rad"])}
        joints[name] = {
            "joint": name,
            "max_abs_rad": observed["max_abs_rad"],
            "rms_rad": observed["rms_rad"],
            "observed": observed,
            "threshold": threshold,
            "threshold_formula": PARITY_THRESHOLD_FORMULA,
            "threshold_source": None if threshold is None else source,
            "pass": None if threshold is None else {key: observed[key] <= threshold[key] for key in observed},
        }
    return {
        "max_abs_rad": float(max_abs.max()),
        "rms_rad": float(rms.max()),
        "threshold_source": source if has_fixed_threshold else None,
        "joints": joints,
    }


def comparison_figure(joint_names: Sequence[str], t: np.ndarray, q_isaac: np.ndarray, q_reference: np.ndarray,
                      reference_label: str):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    cols = 4
    rows = math.ceil(len(joint_names) / cols)
    fig = Figure(figsize=(5.0 * cols, 3.0 * rows), dpi=80, layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(rows, cols, squeeze=False).ravel()
    handles = []
    for j, (ax, name) in enumerate(zip(axes, joint_names)):
        ax.plot(t, q_isaac[:, j], color="C0", label="isaaclab q")
        ax.plot(t, q_reference[:, j], color="C1", linestyle="--", label=f"{reference_label} q")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("q [rad]")
        ax.grid(True, alpha=0.3)
        if j == 0:
            handles = ax.get_legend_handles_labels()[0]
    for ax in axes[len(joint_names):]:
        ax.set_visible(False)
    fig.legend(handles=handles, loc="outside lower center", ncols=len(handles))
    fig.suptitle(f"MIT tracking parity: isaaclab vs {reference_label}")
    return fig


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Track the preset pose sequence with MIT torque control on Isaac Lab."
    )
    assets.add_cli_args(parser)
    records.add_cli_args(parser)
    scene.add_cli_args(parser)
    parser.add_argument(
        "--reference-trajectory", type=Path, default=None,
        help="trajectory CSV from another platform's run of this demo (see results.trajectory.columns_note), with "
             "the same joint order and an identical time grid; when given, results.parity reports the real "
             "per-joint max/RMS |q_isaac - q_reference|, gated only against PARITY_THRESHOLDS_RAD if that fixed, "
             "independently measured constant is configured for the joint (never computed from this comparison); "
             "headless only",
    )
    args = parser.parse_args(argv[1:])
    if args.reference_trajectory is not None and not args.headless:
        parser.error("--reference-trajectory needs --headless: a live run records no trajectory to compare")

    asset = assets.resolve(args.hand, args.asset_root, strict=args.strict)
    print(asset.describe())
    run = records.RunRecord(platform=platform.PLATFORM, demo=DEMO, hand=args.hand,
                            output_dir=args.output_dir, argv=argv)
    view = framing.view(asset.mjcf, args.hand, platform.PLATFORM, "tracking")
    with platform.launch(headless=args.headless, enable_cameras=True):
        try:
            loaded = platform.load(asset.usd, hand=args.hand, description_version=asset.expected_version,
                                   floor_z=view.floor_z, scene=scene_isaaclab.Scene())

            joint_names = loaded.joint_names
            n = len(joint_names)
            timestep = float(loaded.dt)
            duration = scene.tracking_duration()
            tau_limit = np.asarray(loaded.tau_limit, dtype=np.float64)

            if args.headless:
                frame_steps = steps_per_frame(timestep)
                n_steps = round(duration / timestep)
                width, height = view.render_size
                camera = scene_isaaclab.add_camera(loaded, VIDEO_CAMERA, width=width, height=height,
                                                   fovy_deg=view.fovy_deg)

            # Cameras need this reset; it replays physics from USD, discarding MIT gains and joint state set before it.
            loaded.sim.reset()

            kp = np.full(n, scene.DEMO_KP)
            kd = np.full(n, scene.DEMO_KD)
            tau_ff = np.zeros(n)
            stability = platform.mit_stability(loaded, kp, kd)
            platform.set_mit_gains(loaded, kp, kd)
            print(f"MIT stability ({stability['control_path']}): max ratio {stability['max_ratio']:.3f} "
                  f"at {stability['max_joint']} (dt {timestep:g} s, limit {stability['limit']:g})")

            zeros = [0.0] * n
            platform.set_joint_state(loaded, zeros, zeros)

            control = {
                "law": "tau = clip(kp*(q_des-q) + kd*(qd_des-qd) + tau_ff, -tau_limit, tau_limit)",
                "kp": scene.DEMO_KP,
                "kd": scene.DEMO_KD,
                "tau_ff": 0.0,
                "units": {"kp": "N*m/rad", "kd": "N*m*s/rad", "tau_ff": "N*m", "tau_limit": "N*m",
                          "control_period": "s", "q": "rad", "qd": "rad/s"},
                "control_period": timestep,
                "joint_order": list(joint_names),
                "actuator_ids": list(loaded.joint_ids),
                "tau_limit": tau_limit.tolist(),
                "stability": stability,
            }

            if args.headless:
                scene_isaaclab.aim_camera(loaded, camera, view.camera_position, view.camera_target)
                scene_isaaclab.use_path_tracing()

                t = np.arange(n_steps, dtype=np.float64) * timestep
                # Read before the run so a bad reference fails before any artifact is written.
                q_reference = (None if args.reference_trajectory is None
                               else read_reference_trajectory(args.reference_trajectory, joint_names, t))
                q_des_series = np.empty((n_steps, n))
                q_series = np.empty((n_steps, n))
                errors = np.empty((n_steps, n))
                saturated = np.zeros(n, dtype=np.int64)

                with run.open_video(VIDEO_CAMERA, VIDEO_FPS) as video:
                    for step in range(n_steps):
                        q_des, qd_des = scene.tracking_reference(joint_names, step * timestep)
                        q, qd = (np.asarray(values, dtype=np.float64) for values in platform.joint_state(loaded))
                        cmd = mit.MitCommand(joint_names, q_des, qd_des, kp, kd, tau_ff)
                        tau = mit.mit_torque(cmd, q, qd, tau_limit)
                        q_des_series[step] = q_des
                        q_series[step] = q
                        errors[step] = q_des - q
                        saturated += np.abs(tau) >= tau_limit
                        platform.set_command(loaded, q_des, qd_des, tau_ff)
                        platform.step(loaded)
                        # Only a step makes a pose visible (docs/platforms.md), so frame 0 follows step 0.
                        if step == 0 or (step + 1) % frame_steps == 0:
                            video.append(
                                scene_isaaclab.capture_rgb(loaded, camera, renders=scene_isaaclab.CAPTURE_RENDERS).numpy()
                            )
                    frames = video.frames
                    video_path = video.path

                csv_path = run.artifact_path("trajectory", ".csv")
                write_trajectory_csv(csv_path, joint_names, t, q_des_series, q_series)

                rms = np.sqrt(np.mean(errors**2, axis=0))
                max_abs = np.max(np.abs(errors), axis=0)

                parity = {"max_abs_rad": None, "rms_rad": None, "joints": {}, "figure": None,
                         "threshold_formula": PARITY_THRESHOLD_FORMULA, "threshold_source": None,
                         "reference": {"trajectory_csv": None, "label": None},
                         "note": "pass --reference-trajectory <csv written by this script's own results.trajectory.csv schema "
                                 "from another platform's run> to populate per-joint max_abs_rad/rms_rad; threshold/pass/"
                                 "threshold_source stay null here because no reference trajectory was supplied this run "
                                 "(PARITY_THRESHOLDS_RAD is a fixed table from an independent cross-platform measurement)"}
                if q_reference is not None:
                    reference_label = Path(args.reference_trajectory).stem
                    parity.update(parity_results(joint_names, q_series, q_reference, PARITY_THRESHOLDS_RAD,
                                                  PARITY_THRESHOLDS_SOURCE))
                    parity["reference"] = {"trajectory_csv": str(Path(args.reference_trajectory).resolve()),
                                           "label": reference_label}
                    parity["note"] = (
                        f"threshold/pass come from a fixed prior measurement (source: {PARITY_THRESHOLDS_SOURCE}); any "
                        "joint missing from that table still reads back as threshold/pass/threshold_source null"
                        if parity["threshold_source"] is not None else
                        "observed per-joint max_abs_rad/rms_rad above are real measurements of this run; threshold/pass/"
                        "threshold_source are null because no joint in this run has a fixed threshold in "
                        "PARITY_THRESHOLDS_RAD (table unset, or none of this run's joints are in its independently "
                        "measured key set) -- never fabricated here"
                    )
                    fig = comparison_figure(joint_names, t, q_series, q_reference, reference_label)
                    parity["figure"] = str(run.save_figure("parity", fig).resolve())
                    print(f"parity vs {reference_label}: max abs diff {parity['max_abs_rad']:.4f} rad, "
                          f"max RMS diff {parity['rms_rad']:.4f} rad")

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
                    "trajectory": {
                        "csv": str(csv_path.resolve()),
                        "columns_note": TRAJECTORY_COLUMNS_NOTE,
                        "rows": n_steps,
                    },
                    "parity": parity,
                    "video": {"path": str(video_path), "camera": VIDEO_CAMERA, "fps": VIDEO_FPS, "frames": frames,
                              "steps_per_frame": frame_steps},
                }
                record_path = run.write(asset=asset.to_record(), load_report=loaded.report, results=results,
                                        rendering=RENDERING)
                print(f"tracking: max RMS error {rms.max():.4f} rad, max abs error {max_abs.max():.4f} rad")
                print(f"record: {record_path}")
                print(f"artifacts: {3 if args.reference_trajectory is not None else 2}")
            else:
                record_path = run.write(asset=asset.to_record(), load_report=loaded.report,
                                        results={"scene": scene.scene_record(args.hand, view), "control": control},
                                        rendering=RENDERING)
                print(f"record: {record_path}")
                print("artifacts: 0")

                display = scene_isaaclab.Display(loaded, view)

                def frame(t: float) -> None:
                    q_des, qd_des = scene.tracking_reference(joint_names, t % duration)
                    platform.set_command(loaded, q_des, qd_des, tau_ff)
                    platform.step(loaded)

                scene.run_live(display, timestep, frame)
        except (assets.AssetError, profiles.ProfileError, mit.MitStabilityError, records.RecordError,
                scene.JointTableError, ReferenceTrajectoryError, conventions.ViewerError, FileExistsError) as exc:
            # launch_simulation prints a full traceback for anything but SystemExit; exiting keeps the one-line error.
            print(f"error: {exc}", file=sys.stderr)
            sys.stdout.flush()
            sys.stderr.flush()
            sys.exit(1)


if __name__ == "__main__":
    try:
        main(sys.argv)
    except (assets.AssetError, profiles.ProfileError, mit.MitStabilityError, records.RecordError,
            scene.JointTableError, ReferenceTrajectoryError, conventions.ViewerError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
