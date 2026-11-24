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

DEMO = "05_parallel_envs"
RENDERING = "Isaac Lab physics / Isaac Sim RTX camera rendering"
VIDEO_FPS = 25
VIDEO_CAMERA = scene.PREVIEW_CAMERA
MOTION_FRACTION = 0.5
# Minimum visited-pose spread for an env's motion check to mean anything; a run below it aborts.
MOTION_EPSILON_RAD = 1e-3
ROOT_TOLERANCE_M = 1e-3
CONTROL_LAW = "Isaac Lab ImplicitActuator PD (PhysX-integrated): tau = clip(kp*(q_des-q)+kd*(qd_des-qd), -tau_limit, tau_limit)"


def steps_per_frame(timestep: float) -> int:
    ratio = 1.0 / (VIDEO_FPS * timestep)
    steps = round(ratio)
    if steps < 1 or not math.isclose(ratio, steps, rel_tol=1e-9):
        raise ValueError(f"video frame period 1/{VIDEO_FPS} s is not a whole number of {timestep} s physics steps")
    return steps


def batch_joint_state(loaded: platform.Loaded) -> np.ndarray:
    # platform.joint_state reads env 0 only; this reads every env, shaped (num_envs, len(joint_ids)).
    q = loaded.articulation.data.joint_pos.torch[:, list(loaded.joint_ids)]
    if hasattr(q, "detach"):
        q = q.detach().cpu().numpy()
    return np.asarray(q, dtype=np.float64)


def write_batch_joint_state(loaded: platform.Loaded, q: np.ndarray) -> None:
    # platform.set_joint_state writes one pose to every environment; here each environment starts at its own pose.
    import torch

    position = torch.as_tensor(q, dtype=torch.float32, device=loaded.device)
    loaded.articulation.write_joint_state_to_sim_index(
        position=position, velocity=torch.zeros_like(position), joint_ids=list(loaded.joint_ids)
    )
    loaded.sim.forward()


def apply_batch_command(loaded: platform.Loaded, q_des: np.ndarray, qd_des: np.ndarray) -> None:
    # Each env tracks its own reference; platform.set_command would broadcast one (n_joints,) target to all envs.
    import torch

    joint_ids = list(loaded.joint_ids)
    target_command = loaded.articulation.actuators.target_command
    q_tensor = torch.as_tensor(q_des, dtype=torch.float32, device=loaded.device)
    qd_tensor = torch.as_tensor(qd_des, dtype=torch.float32, device=loaded.device)
    target_command.set_position_index(value=q_tensor, joint_ids=joint_ids)
    target_command.set_velocity_index(value=qd_tensor, joint_ids=joint_ids)
    target_command.set_effort_index(value=torch.zeros_like(q_tensor), joint_ids=joint_ids)


def batch_reference(joint_names: Sequence[str], num_envs: int, t: float) -> tuple[np.ndarray, np.ndarray]:
    references = [scene.parallel_reference(joint_names, env, t) for env in range(num_envs)]
    return np.stack([q for q, _ in references]), np.stack([qd for _, qd in references])


def write_batch_csv(path: Path, joint_names: Sequence[str], timestep: float, trace: np.ndarray) -> None:
    with open(path, "x", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "env", *joint_names])
        for step, rows in enumerate(trace):
            t = step * timestep
            for env, q in enumerate(rows):
                writer.writerow([f"{t:.4f}", env, *(f"{value:.9g}" for value in q)])


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Clone the hand model into N parallel Isaac Lab environments via InteractiveSceneCfg env "
        "cloning; the environments cycle through the preset poses out of phase to prove the clones run "
        "independently."
    )
    assets.add_cli_args(parser)
    records.add_cli_args(parser)
    scene.add_cli_args(parser)
    parser.add_argument(
        "--num-envs", type=int, default=4,
        help=f"number of cloned Isaac Lab environments (>= 1); env i starts at {scene.POSES}[i %% {len(scene.POSES)}]",
    )
    args = parser.parse_args(argv[1:])
    if args.num_envs < 1:
        parser.error(f"--num-envs must be >= 1, got {args.num_envs}")

    asset = assets.resolve(args.hand, args.asset_root, strict=args.strict)
    print(asset.describe())
    run = records.RunRecord(
        platform=platform.PLATFORM, demo=DEMO, hand=args.hand, output_dir=args.output_dir, argv=argv
    )
    view = framing.view(asset.mjcf, args.hand, platform.PLATFORM, "grid", worlds=args.num_envs)
    with platform.launch(headless=args.headless, enable_cameras=True):
        try:
            loaded = platform.load(
                asset.usd,
                hand=args.hand,
                description_version=asset.expected_version,
                num_envs=args.num_envs,
                env_spacing=scene.PARALLEL_SPACING_M,
                floor_z=view.floor_z,
                scene=scene_isaaclab.Scene(),
            )
            if args.headless:
                width, height = view.render_size
                camera = scene_isaaclab.add_camera(loaded, VIDEO_CAMERA, width=width, height=height,
                                                   fovy_deg=view.fovy_deg)
            # Cameras need this reset; it replays physics from USD, discarding MIT gains and joint state set before it.
            loaded.sim.reset()

            joint_names = loaded.joint_names
            n = len(joint_names)
            observed_shape = tuple(batch_joint_state(loaded).shape)
            if observed_shape != (args.num_envs, n):
                raise RuntimeError(
                    f"InteractiveScene reports joint_pos shape {observed_shape}; expected ({args.num_envs}, {n}) "
                    f"for --num-envs {args.num_envs}"
                )
            root_pos_w_m = np.asarray(loaded.articulation.data.root_pos_w.torch.detach().cpu().tolist())
            root_error_m = float(np.abs(root_pos_w_m - np.asarray(view.hand_roots)).max())
            if not root_error_m < ROOT_TOLERANCE_M:
                raise RuntimeError(
                    f"env roots {root_pos_w_m.tolist()} differ from the preview layout {list(view.hand_roots)} by "
                    f"{root_error_m:.3g} m (>= {ROOT_TOLERANCE_M:g} m)"
                )

            gains = {"kp": np.full(n, scene.DEMO_KP), "kd": np.full(n, scene.DEMO_KD)}
            stability = platform.mit_stability(loaded, gains["kp"], gains["kd"])
            platform.set_mit_gains(loaded, gains["kp"], gains["kd"])
            print(
                f"MIT stability ({stability['control_path']}): max ratio {stability['max_ratio']:.3f} "
                f"at {stability['max_joint']} (dt {loaded.dt:g} s, limit {stability['limit']:g})"
            )

            timestep = float(loaded.dt)
            start_poses = [scene.POSES[env % len(scene.POSES)] for env in range(args.num_envs)]
            write_batch_joint_state(loaded, batch_reference(joint_names, args.num_envs, 0.0)[0])
            batch = {
                "num_envs": args.num_envs,
                "env_spacing_m": scene.PARALLEL_SPACING_M,
                "root_pos_w_m": root_pos_w_m.tolist(),
                "hand_roots_m": [list(root) for root in view.hand_roots],
                "root_error_max_m": root_error_m,
            }

            if args.headless:
                steps = scene.whole_steps(scene.parallel_duration(), timestep)
                frame_steps = steps_per_frame(timestep)
                trace = np.empty((steps + 1, args.num_envs, n), dtype=np.float64)
                reference = np.empty((steps, args.num_envs, n), dtype=np.float64)
                trace[0] = batch_joint_state(loaded)
                scene_isaaclab.aim_camera(loaded, camera, view.camera_position, view.camera_target)
                scene_isaaclab.use_path_tracing()
                with run.open_video(VIDEO_CAMERA, VIDEO_FPS) as video:
                    for k in range(steps):
                        q_des, qd_des = batch_reference(joint_names, args.num_envs, k * timestep)
                        reference[k] = q_des
                        apply_batch_command(loaded, q_des, qd_des)
                        platform.step(loaded)
                        trace[k + 1] = batch_joint_state(loaded)
                        # Only a step makes a pose visible (see 02), so frame 0 follows step 0, still at the start pose.
                        if k == 0 or (k + 1) % frame_steps == 0:
                            video.append(
                                scene_isaaclab.capture_rgb(loaded, camera, renders=scene_isaaclab.CAPTURE_RENDERS).numpy()
                            )
                    frames = video.frames
                    video_path = video.path

                final = trace[-1]
                final_error_rad = np.abs(final - batch_reference(joint_names, args.num_envs, steps * timestep)[0])
                trace_range_rad = trace.max(axis=0) - trace.min(axis=0)
                # The reference holds each visited pose and moves monotonically between them: its range = their spread.
                pose_delta_rad = (reference.max(axis=0) - reference.min(axis=0)).max(axis=1)
                insufficient = [env for env in range(args.num_envs) if pose_delta_rad[env] <= MOTION_EPSILON_RAD]
                if insufficient:
                    raise RuntimeError(
                        f"env(s) {insufficient} visit poses within {MOTION_EPSILON_RAD:g} rad of each other; every "
                        "env's reference must move it"
                    )
                still = [
                    f"env{env} moved {trace_range_rad[env].max():.3g} rad < "
                    f"{MOTION_FRACTION:g} * {pose_delta_rad[env]:.3g} rad"
                    for env in range(args.num_envs)
                    if trace_range_rad[env].max() < MOTION_FRACTION * pose_delta_rad[env]
                ]
                if still:
                    raise RuntimeError(f"env(s) did not track their reference: {'; '.join(still)}")

                # Distinct final vectors catch a loop that stepped one env and broadcast it; nothing above would.
                final_vectors = {tuple(np.round(row, 6)) for row in final}
                if len(final_vectors) < min(args.num_envs, len(scene.POSES)):
                    raise RuntimeError(
                        f"{args.num_envs} environments converged to only {len(final_vectors)} distinct joint "
                        f"vector(s); expected independent per-environment trajectories, not a shared/broadcast result"
                    )

                envs = [
                    {
                        "env": env,
                        "start_pose": start_poses[env],
                        "initial_q_rad": trace[0, env].tolist(),
                        "final_q_rad": final[env].tolist(),
                        "final_error_max_rad": float(final_error_rad[env].max()),
                        "trace_range_max_rad": float(trace_range_rad[env].max()),
                        "visited_pose_delta_max_rad": float(pose_delta_rad[env]),
                        "motion_threshold_rad": MOTION_FRACTION * float(pose_delta_rad[env]),
                    }
                    for env in range(args.num_envs)
                ]

                csv_path = run.artifact_path("batch", ".csv")
                write_batch_csv(csv_path, joint_names, timestep, trace)

                results = {
                    "scene": scene.scene_record(args.hand, view),
                    "batch": {
                        **batch,
                        "mechanism": "isaaclab.scene.InteractiveSceneCfg(num_envs=N, replicate_physics=False) env "
                        "cloning inside platform.load(num_envs=N, env_spacing=scene.PARALLEL_SPACING_M)",
                        "joint_pos_shape": list(observed_shape),
                        "reference": "wuji_sim.scene.parallel_reference: env i starts at scene.POSES[i % 4] and from "
                        "there cycles through scene.TRACKING_SEQUENCE closed on itself, min-jerk over move_s then "
                        "holding for hold_s; one cycle of scene.parallel_duration()",
                        "move_s": scene.MOVE_S,
                        "hold_s": scene.HOLD_S,
                        "steps": steps,
                        "joint_order": list(joint_names),
                        "envs": envs,
                        "csv": str(csv_path.resolve()),
                    },
                    "control": {
                        "law": CONTROL_LAW,
                        "kp": scene.DEMO_KP,
                        "kd": scene.DEMO_KD,
                        "units": {"kp": "N*m/rad", "kd": "N*m*s/rad", "period": "s"},
                        "period_s": timestep,
                        "stability": stability,
                    },
                    "video": {"path": str(video_path), "camera": VIDEO_CAMERA, "fps": VIDEO_FPS, "frames": frames,
                              "steps_per_frame": frame_steps, "worlds": args.num_envs},
                }
                record_path = run.write(asset=asset.to_record(), load_report=loaded.report, results=results,
                                        rendering=RENDERING)
                if not record_path.is_file() or not csv_path.is_file() or not Path(video_path).is_file():
                    raise records.RecordError(f"run.write reported success but output is missing: {record_path}")
                print(f"record: {record_path}")
                print("artifacts: 2")
            else:
                results = {
                    "scene": scene.scene_record(args.hand, view),
                    "batch": batch,
                    "control": {"law": CONTROL_LAW, "kp": scene.DEMO_KP, "kd": scene.DEMO_KD, "stability": stability},
                }
                record_path = run.write(asset=asset.to_record(), load_report=loaded.report, results=results,
                                        rendering=RENDERING)
                if not record_path.is_file():
                    raise records.RecordError(f"run.write reported success but output is missing: {record_path}")
                print(f"record: {record_path}")
                print("artifacts: 0")

                display = scene_isaaclab.Display(loaded, view)

                def frame(t: float) -> None:
                    q_des, qd_des = batch_reference(joint_names, args.num_envs, t)
                    apply_batch_command(loaded, q_des, qd_des)
                    platform.step(loaded)

                scene.run_live(display, timestep, frame)
        except (
            assets.AssetError,
            profiles.ProfileError,
            mit.MitStabilityError,
            records.RecordError,
            scene.JointTableError,
            RuntimeError,
            FileExistsError,
        ) as exc:
            # launch_simulation prints a full traceback for anything but SystemExit; exiting keeps the one-line error.
            print(f"error: {exc}", file=sys.stderr)
            sys.stdout.flush()
            sys.stderr.flush()
            sys.exit(1)


if __name__ == "__main__":
    try:
        main(sys.argv)
    except (
        assets.AssetError,
        profiles.ProfileError,
        mit.MitStabilityError,
        records.RecordError,
        scene.JointTableError,
        RuntimeError,
        conventions.ViewerError,
        FileExistsError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()
        # os._exit skips interpreter shutdown, where Kit's own cleanup would reset this nonzero code to 0
        os._exit(1)
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
