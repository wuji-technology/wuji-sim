# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import sys
import traceback
from pathlib import Path

import numpy as np

from wuji_sim import assets, conventions, framing, profiles, records, scene, scene_isaaclab
from wuji_sim.platforms import isaaclab as platform

DEMO = "02_preset_poses"
RENDERING = "Isaac Lab physics / Isaac Sim RTX camera rendering"
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
FOVY_DEG = 45.0  # matches wuji_sim.scene's MuJoCo camera fovy
LIVE_HOLD_S = 1.5


def target_position(loaded: platform.Loaded, hand: str) -> np.ndarray:
    name = conventions.HAND_PREFIXES[hand] + "middle_finger_proximal"
    body_ids, body_names = loaded.articulation.find_bodies([name], preserve_order=True)
    if tuple(body_names) != (name,):
        raise scene.JointTableError(f"USD body names do not contain {name!r}: {body_names!r}")
    position = loaded.articulation.data.body_pos_w.torch[0, int(body_ids[0])]
    if hasattr(position, "detach"):
        position = position.detach().cpu().numpy()
    return np.asarray(position, dtype=np.float64)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Render the four preset hand poses from every scene camera.")
    assets.add_cli_args(parser)
    records.add_cli_args(parser)
    scene.add_cli_args(parser)
    args = parser.parse_args(argv[1:])

    asset = assets.resolve(args.hand, args.asset_root, strict=args.strict)
    print(asset.describe())
    run = records.RunRecord(platform=platform.PLATFORM, demo=DEMO, hand=args.hand,
                            output_dir=args.output_dir, argv=argv)
    # Framed like 03: its tracking motion passes through every preset pose.
    view = framing.view(asset.mjcf, args.hand, platform.PLATFORM, "tracking")
    with platform.launch(headless=args.headless, enable_cameras=True):
        try:
            loaded = platform.load(asset.usd, hand=args.hand, description_version=asset.expected_version,
                                   floor_z=view.floor_z, scene=scene_isaaclab.Scene())

            if args.headless:
                camera_positions = {
                    "front": scene.FRONT_CAMERA_POS[args.hand],
                    "side": scene.SIDE_CAMERA_POS,
                    "back": scene.BACK_CAMERA_POS[args.hand],
                }
                cameras = {
                    label: scene_isaaclab.add_camera(loaded, label, width=IMAGE_WIDTH, height=IMAGE_HEIGHT, fovy_deg=FOVY_DEG)
                    for label in scene.CAMERAS
                }
                width, height = view.render_size
                cameras[scene.PREVIEW_CAMERA] = scene_isaaclab.add_camera(
                    loaded, scene.PREVIEW_CAMERA, width=width, height=height, fovy_deg=view.fovy_deg
                )

            # Cameras built after load() get data buffers only on the next PHYSICS_READY, which sim.reset() dispatches.
            loaded.sim.reset()
            zeros = [0.0] * len(loaded.joint_names)
            platform.set_joint_state(loaded, zeros, zeros)

            def pose_q(pose: str) -> np.ndarray:
                q = scene.pose_vector(loaded.joint_names, pose)
                for name, value in zip(loaded.joint_names, q):
                    lo, hi = loaded.report["effective_values"]["joints"][name]["range"]
                    if not lo <= value <= hi:
                        raise ValueError(f"pose {pose!r}: {name} = {value} rad is outside range [{lo}, {hi}]")
                return q

            if args.headless:
                target = target_position(loaded, args.hand)
                for label in scene.CAMERAS:
                    scene_isaaclab.aim_camera(loaded, cameras[label], camera_positions[label], target)
                scene_isaaclab.aim_camera(loaded, cameras[scene.PREVIEW_CAMERA], view.camera_position, view.camera_target)
                scene_isaaclab.use_path_tracing()

                poses = {}
                pose_targets = {}
                views = {}
                for pose in scene.POSES:
                    q = pose_q(pose)
                    platform.set_joint_state(loaded, q, zeros)
                    # RTX reads poses from Fabric, which only a physics step flushes to Hydra; target q keeps the pose.
                    platform.set_command(loaded, q, zeros, zeros)
                    q_before_step, _ = platform.joint_state(loaded)
                    platform.step(loaded)
                    q_after_step, _ = platform.joint_state(loaded)
                    step_drift = float(np.abs(np.asarray(q_after_step) - np.asarray(q_before_step)).max())
                    print(f"{pose}: max|dq| over the Fabric-sync step = {step_drift:.3e} rad")
                    # The images show the post-step state, so the record keeps that and the requested pose apart.
                    poses[pose] = {name: float(value) for name, value in zip(loaded.joint_names, q_after_step)}
                    pose_targets[pose] = {name: float(value) for name, value in zip(loaded.joint_names, q)}
                    views[pose] = {
                        label: str(run.write_png(
                            f"{pose}_{label}",
                            scene_isaaclab.capture_rgb(loaded, camera, renders=scene_isaaclab.CAPTURE_RENDERS).numpy(),
                        ))
                        for label, camera in cameras.items()
                    }

                record_path = run.write(
                    asset=asset.to_record(),
                    load_report=loaded.report,
                    results={"scene": scene.scene_record(args.hand, view), "poses": poses, "pose_targets": pose_targets,
                             "views": views},
                    rendering=RENDERING,
                )
                if not record_path.is_file() or not all(
                    Path(path).is_file() for pose_views in views.values() for path in pose_views.values()
                ):
                    raise records.RecordError(f"run.write reported success but output is missing: {record_path}")
                print(f"record: {record_path}")
                print(f"artifacts: {sum(len(v) for v in views.values())}")
            else:
                for pose in scene.POSES:
                    pose_q(pose)

                record_path = run.write(
                    asset=asset.to_record(),
                    load_report=loaded.report,
                    results={"scene": scene.scene_record(args.hand, view)},
                    rendering=RENDERING,
                )
                if not record_path.is_file():
                    raise records.RecordError(f"run.write reported success but output is missing: {record_path}")
                print(f"record: {record_path}")
                print("artifacts: 0")

                display = scene_isaaclab.Display(loaded, view)

                def frame(t: float) -> None:
                    pose = scene.POSES[int(t // LIVE_HOLD_S) % len(scene.POSES)]
                    q = pose_q(pose)
                    platform.set_joint_state(loaded, q, zeros)
                    platform.set_command(loaded, q, zeros, zeros)
                    platform.step(loaded)

                scene.run_live(display, loaded.dt, frame)
        except (assets.AssetError, profiles.ProfileError, records.RecordError, scene.JointTableError,
                conventions.ViewerError, FileExistsError) as exc:
            # launch_simulation prints a full traceback for anything but SystemExit; exiting keeps the one-line error.
            print(f"error: {exc}", file=sys.stderr)
            sys.stdout.flush()
            sys.stderr.flush()
            sys.exit(1)


if __name__ == "__main__":
    try:
        main(sys.argv)
    except (assets.AssetError, profiles.ProfileError, records.RecordError, scene.JointTableError,
            conventions.ViewerError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
