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

DEMO = "04_sweep_joints"
RENDERING = "Isaac Lab physics / Isaac Sim RTX camera rendering"
VIDEO_FPS = 25
VIDEO_CAMERA = scene.PREVIEW_CAMERA
CONTROL_LAW = "tau = clip(kp*(q_des-q) + kd*(qd_des-qd) + tau_ff, -tau_limit, tau_limit)"
TAU_APPLIED = (
    "Isaac Lab ImplicitActuator applied_effort read after sim.step: approximate actuator effort telemetry that "
    "Isaac Lab computes from the pre-step state, not an effort reported by PhysX; it is not the unclipped MIT "
    "reference and does not include external contact forces"
)
# Applied offline by PARITY_THRESHOLDS_SOURCE, never to this run: 2*x rounded up >= x would always pass.
PARITY_THRESHOLD_FORMULA = "round_up_to_2_sig_figs(2 * observed_rad)"
PARITY_THRESHOLDS_SOURCE: str | None = (
    "independent MuJoCo vs Isaac Lab measurement, 2 deterministic runs: MuJoCo 03/04 run with its trajectory "
    "output enabled, then Isaac Lab 03/04 run with --reference-trajectory pointed at that file; threshold = "
    "observed x 2 rounded up to 2 significant figures"
)
PARITY_THRESHOLDS: dict[str, dict[str, float]] | None = {
    "l_thumb_cmc_flex": {"max_abs_rad": 0.052, "rms_rad": 0.0074},
    "l_thumb_cmc_abd": {"max_abs_rad": 0.047, "rms_rad": 0.0068},
    "l_thumb_mcp": {"max_abs_rad": 0.056, "rms_rad": 0.008},
    "l_thumb_ip": {"max_abs_rad": 0.055, "rms_rad": 0.0079},
    "l_index_finger_mcp_flex": {"max_abs_rad": 0.058, "rms_rad": 0.0082},
    "l_index_finger_mcp_abd": {"max_abs_rad": 0.030, "rms_rad": 0.0043},
    "l_index_finger_pip": {"max_abs_rad": 0.067, "rms_rad": 0.0097},
    "l_index_finger_dip": {"max_abs_rad": 0.055, "rms_rad": 0.0079},
    "l_middle_finger_mcp_flex": {"max_abs_rad": 0.058, "rms_rad": 0.0082},
    "l_middle_finger_mcp_abd": {"max_abs_rad": 0.031, "rms_rad": 0.0043},
    "l_middle_finger_pip": {"max_abs_rad": 0.067, "rms_rad": 0.0097},
    "l_middle_finger_dip": {"max_abs_rad": 0.055, "rms_rad": 0.0079},
    "l_ring_finger_mcp_flex": {"max_abs_rad": 0.058, "rms_rad": 0.0082},
    "l_ring_finger_mcp_abd": {"max_abs_rad": 0.031, "rms_rad": 0.0044},
    "l_ring_finger_pip": {"max_abs_rad": 0.067, "rms_rad": 0.0097},
    "l_ring_finger_dip": {"max_abs_rad": 0.055, "rms_rad": 0.0079},
    "l_pinky_mcp_flex": {"max_abs_rad": 0.057, "rms_rad": 0.0082},
    "l_pinky_mcp_abd": {"max_abs_rad": 0.033, "rms_rad": 0.0059},
    "l_pinky_pip": {"max_abs_rad": 0.067, "rms_rad": 0.0097},
    "l_pinky_dip": {"max_abs_rad": 0.055, "rms_rad": 0.0079},
}


class ReferenceTrajectoryError(ValueError):
    pass


def steps_per_frame(timestep: float) -> int:
    ratio = 1.0 / (VIDEO_FPS * timestep)
    steps = round(ratio)
    if steps < 1 or not math.isclose(ratio, steps, rel_tol=1e-9):
        raise ValueError(f"video frame period 1/{VIDEO_FPS} s is not a whole number of {timestep} s physics steps")
    return steps


def read_reference_trajectory(path: Path, joint_names: Sequence[str], t_expected: np.ndarray) -> np.ndarray:
    path = Path(path)
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except OSError as exc:
        raise ReferenceTrajectoryError(f"cannot read reference trajectory {path}: {exc}") from exc
    if not rows:
        raise ReferenceTrajectoryError(f"reference trajectory {path} is empty")
    q_columns = [f"q:{name}" for name in joint_names]
    missing = [col for col in ("t_s", *q_columns) if col not in rows[0]]
    if missing:
        raise ReferenceTrajectoryError(
            f"reference trajectory {path} is missing columns {missing}; expected the same joint order as this run "
            "(the schema results.proprioception.csv of this script's own output)"
        )
    if len(rows) != len(t_expected):
        raise ReferenceTrajectoryError(
            f"reference trajectory {path} has {len(rows)} rows; this run has {len(t_expected)} steps"
        )
    columns = ["t_s", *q_columns]
    values = np.empty((len(rows), len(columns)), dtype=np.float64)
    for i, row in enumerate(rows):
        line = i + 2
        for j, col in enumerate(columns):
            cell = row[col]
            if cell is None:
                raise ReferenceTrajectoryError(f"reference trajectory {path} line {line} ends before column {col!r}")
            try:
                values[i, j] = float(cell)
            except ValueError:
                raise ReferenceTrajectoryError(
                    f"reference trajectory {path} line {line} column {col!r}: {cell!r} is not a number"
                ) from None
    if not np.allclose(values[:, 0], t_expected, rtol=0.0, atol=1e-6):
        raise ReferenceTrajectoryError(
            f"reference trajectory {path} time grid differs from this run's grid by more than 1e-6 s"
        )
    q_reference = values[:, 1:]
    nonfinite = np.argwhere(~np.isfinite(q_reference))
    if nonfinite.size:
        i, j = nonfinite[0]
        raise ReferenceTrajectoryError(
            f"reference trajectory {path} line {i + 2} column {q_columns[j]!r}: measured position "
            f"{q_reference[i, j]} is not finite"
        )
    return q_reference


def parity_results(joint_names: Sequence[str], q_isaac: np.ndarray, q_reference: np.ndarray) -> dict:
    diff = np.abs(q_isaac - q_reference)
    max_abs = np.max(diff, axis=0)
    rms = np.sqrt(np.mean(diff**2, axis=0))
    joints = {
        name: {"joint": name, "max_abs_rad": float(max_abs[j]), "rms_rad": float(rms[j])}
        for j, name in enumerate(joint_names)
    }
    return {"max_abs_rad": float(max_abs.max()), "rms_rad": float(rms.max()), "joints": joints}


def apply_parity_thresholds(joints: dict[str, dict], thresholds: dict[str, dict[str, float]] | None,
                            source: str | None) -> tuple[str, str | None]:
    # Thresholds come only from the fixed table; deriving them from `joints` would make every check pass.
    has_fixed_threshold = thresholds is not None and any(name in thresholds for name in joints)
    for name, entry in joints.items():
        threshold = None if thresholds is None else thresholds.get(name)
        entry["threshold"] = threshold
        entry["pass"] = (None if threshold is None else
                         {key: entry[key] <= threshold[key] for key in ("max_abs_rad", "rms_rad")})
    if has_fixed_threshold:
        return "configured", source
    return "no fixed threshold for this joint set", None


def comparison_figure(joint_names: Sequence[str], t: np.ndarray, q_isaac: np.ndarray, q_reference: np.ndarray,
                      reference_label: str):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    cols = 4
    grid_rows = math.ceil(len(joint_names) / cols)
    fig = Figure(figsize=(5.0 * cols, 3.0 * grid_rows), dpi=80, layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(grid_rows, cols, squeeze=False).ravel()
    handles = []
    for j, (ax, name) in enumerate(zip(axes, joint_names)):
        ax.plot(t, q_reference[:, j], color="gray", linestyle="--", label=reference_label)
        ax.plot(t, q_isaac[:, j], color="C0", label="isaaclab")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("q [rad]")
        ax.grid(True, alpha=0.3)
        if j == 0:
            handles = ax.get_legend_handles_labels()[0]
    for ax in axes[len(joint_names):]:
        ax.set_visible(False)
    fig.legend(handles=handles, loc="outside lower center", ncols=len(handles))
    fig.suptitle(f"isaaclab {DEMO} vs {reference_label}: per-joint q(t)")
    return fig


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Sweep each joint between its inner limits with MIT torque control, one joint at a time."
    )
    assets.add_cli_args(parser)
    records.add_cli_args(parser)
    scene.add_cli_args(parser)
    parser.add_argument(
        "--reference-trajectory", type=Path, default=None,
        help="proprioception CSV from another platform's run of this demo (same schema as "
             "results.proprioception.csv, written by wuji_sim.scene.write_sweep_csv), with the same joint "
             "order and an identical time grid; when given, results.parity reports the per-joint observed "
             "max/RMS |q_isaac - q_reference|. The pass/fail threshold is a fixed constant "
             "(PARITY_THRESHOLDS) filled in only after an independent prior measurement; it is never "
             "computed from this run; headless only",
    )
    args = parser.parse_args(argv[1:])
    if args.reference_trajectory is not None and not args.headless:
        parser.error("--reference-trajectory needs --headless: a live run records no trajectory to compare")

    asset = assets.resolve(args.hand, args.asset_root, strict=args.strict)
    print(asset.describe())
    run = records.RunRecord(platform=platform.PLATFORM, demo=DEMO, hand=args.hand,
                            output_dir=args.output_dir, argv=argv)
    view = framing.view(asset.mjcf, args.hand, platform.PLATFORM, "sweep")
    with platform.launch(headless=args.headless, enable_cameras=True):
        try:
            loaded = platform.load(asset.usd, hand=args.hand, description_version=asset.expected_version,
                                   floor_z=view.floor_z, scene=scene_isaaclab.Scene())
            joint_names = loaded.joint_names
            description = scene.description_joint_ranges(asset.mjcf)
            effective = {name: loaded.report["effective_values"]["joints"][name]["range"] for name in joint_names}
            scene.check_joint_table(args.hand, description, effective)
            print(f"joint table: {len(joint_names)} joints; names and compiled ranges match {asset.mjcf}")
            ranges = np.array([effective[name] for name in joint_names])

            timestep = float(loaded.dt)
            n = len(joint_names)
            tau_limit = np.asarray(loaded.tau_limit, dtype=np.float64)

            if args.headless:
                frame_steps = steps_per_frame(timestep)
                n_steps = scene.whole_steps(scene.sweep_duration(joint_names), timestep)
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
                "law": CONTROL_LAW,
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
            joint_table_check = {
                "description_mjcf": str(asset.mjcf),
                "joints": n,
                "compared": "hinge joint count, names and order; description MJCF range vs compiled model range",
                "range_atol_rad": scene.JOINT_RANGE_ATOL,
                "passed": True,
            }

            if args.headless:
                scene_isaaclab.aim_camera(loaded, camera, view.camera_position, view.camera_target)
                scene_isaaclab.use_path_tracing()

                t = np.arange(n_steps) * timestep
                # Read before the run so a bad reference fails before any artifact is written.
                q_reference = (None if args.reference_trajectory is None
                               else read_reference_trajectory(args.reference_trajectory, joint_names, t))
                active = np.empty(n_steps, dtype=np.int64)
                series = {field: np.empty((n_steps, n)) for field in scene.SWEEP_SERIES}
                with run.open_video(VIDEO_CAMERA, VIDEO_FPS) as video:
                    for step in range(n_steps):
                        q_des, qd_des, active[step] = scene.sweep_reference(joint_names, ranges, t[step])
                        cmd = mit.MitCommand(joint_names, q_des, qd_des, kp, kd, tau_ff)
                        q, qd = (np.asarray(values, dtype=np.float64) for values in platform.joint_state(loaded))
                        tau = mit.mit_torque(cmd, q, qd, tau_limit)
                        series["q_des"][step], series["q"][step], series["qd"][step] = q_des, q, qd
                        series["tau_command"][step] = tau
                        platform.set_command(loaded, q_des, qd_des, tau_ff)
                        platform.step(loaded)
                        series["tau_applied"][step] = platform.applied_effort(loaded)
                        # Only a step makes a pose visible (docs/platforms.md), so frame 0 follows step 0.
                        if step == 0 or (step + 1) % frame_steps == 0:
                            video.append(
                                scene_isaaclab.capture_rgb(loaded, camera, renders=scene_isaaclab.CAPTURE_RENDERS).numpy()
                            )
                    frames = video.frames
                    video_path = video.path

                error = series["q_des"] - series["q"]
                saturated = np.abs(series["tau_command"]) >= tau_limit
                rows = scene.sweep_joint_table(joint_names, description, effective, active, series["q"], error,
                                               saturated)
                flagged = scene.sweep_flags(rows)
                csv_path = run.artifact_path("timeseries", ".csv")
                scene.write_sweep_csv(csv_path, joint_names, t, active, series)
                title = f"{asset.product} {args.hand} {DEMO} ({RENDERING}; kp {scene.DEMO_KP}, kd {scene.DEMO_KD})"
                figure_path = run.save_figure("joint_curves", scene.sweep_figure(rows, t, series, title))

                parity = {
                    "max_abs_rad": None, "rms_rad": None, "joints": {}, "figure": None,
                    "threshold_formula": PARITY_THRESHOLD_FORMULA,
                    "threshold_status": "threshold not configured",
                    "threshold_source": None,
                    "reference": {"trajectory_csv": None, "label": None},
                    "note": "pass --reference-trajectory <csv written by this script's own results.proprioception.csv "
                            "schema from another run> to populate per-joint observed max_abs_rad/rms_rad",
                }
                if q_reference is not None:
                    reference_label = Path(args.reference_trajectory).stem
                    parity.update(parity_results(joint_names, series["q"], q_reference))
                    parity["reference"] = {"trajectory_csv": str(Path(args.reference_trajectory).resolve()),
                                           "label": reference_label}
                    parity["note"] = None
                    fig = comparison_figure(joint_names, t, series["q"], q_reference, reference_label)
                    parity["figure"] = str(run.save_figure("parity", fig).resolve())
                    parity["threshold_status"], parity["threshold_source"] = apply_parity_thresholds(
                        parity["joints"], PARITY_THRESHOLDS, PARITY_THRESHOLDS_SOURCE)
                    print(f"parity vs {reference_label}: max abs diff {parity['max_abs_rad']:.4f} rad, "
                          f"max RMS diff {parity['rms_rad']:.4f} rad ({parity['threshold_status']})")

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
                        "q_qd": "articulation joint_pos / joint_vel, sampled before each physics step",
                        "tau_command": "MIT torque reference computed from the pre-step state for recording; the integrated "
                                       "actuator applies the same law internally",
                        "tau_applied": TAU_APPLIED,
                    },
                    "parity": parity,
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
                print(f"artifacts: {4 if args.reference_trajectory is not None else 3}")
            else:
                results = {"scene": scene.scene_record(args.hand, view), "control": control,
                          "joint_table_check": joint_table_check}
                record_path = run.write(asset=asset.to_record(), load_report=loaded.report, results=results,
                                        rendering=RENDERING)
                print(f"record: {record_path}")
                print("artifacts: 0")

                display = scene_isaaclab.Display(loaded, view)

                def frame(t: float) -> None:
                    q_des, qd_des, _ = scene.sweep_reference(joint_names, ranges, t)
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
