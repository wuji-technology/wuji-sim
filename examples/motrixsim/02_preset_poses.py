# SPDX-License-Identifier: Apache-2.0
import argparse
import sys
from pathlib import Path

import numpy as np

from wuji_sim import assets, conventions, framing, profiles, records, scene
from wuji_sim.platforms import motrixsim as platform

sys.path.insert(0, str(Path(__file__).resolve().parent / "_shared"))

import rendering  # noqa: E402

DEMO = "02_preset_poses"
RENDERING = "MotrixSim physics / MotrixSim rendering"
LIVE_HOLD_S = 1.5
# The system camera is built with the fixed cameras' fovy, so front/side/back are exact and only the
# preview camera's distance is scaled (see examples/motrixsim/_shared/rendering.py).
BUILT_FOVY_DEG = rendering.FIXED_CAMERA_FOVY_DEG


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Render the preset hand poses from every scene camera.")
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
    loaded = platform.load(asset.mjcf, scene=lambda spec: scene.add_scene(spec, args.hand, view),
                           description_version=asset.expected_version, render_resolution=view.render_size,
                           render_fovy=BUILT_FOVY_DEG)
    model = loaded.model

    def pose_q(pose: str):
        q = scene.pose_vector(loaded.joint_names, pose)
        for name, value in zip(loaded.joint_names, q):
            lo, hi = model.jnt_range[model.joint(name).id]
            if not lo <= value <= hi:
                raise ValueError(f"pose {pose!r}: {name} = {value} rad is outside jnt_range [{lo}, {hi}]")
        return q

    def set_pose(q) -> None:
        platform.set_state(loaded, q)

    poses = {}
    views = {}
    if args.headless:
        with rendering.OffscreenRenderer(loaded, view, args.hand, BUILT_FOVY_DEG) as renderer:
            for pose in scene.POSES:
                q = pose_q(pose)
                set_pose(q)
                poses[pose] = {name: float(value) for name, value in zip(loaded.joint_names, q)}
                # The mujoco front/side/back cameras track the middle-finger proximal body.
                target = platform.link_pose(loaded, conventions.HAND_PREFIXES[args.hand]
                                            + "middle_finger_proximal")[:3]
                views[pose] = {camera: str(run.write_png(f"{pose}_{camera}",
                                                         renderer.render(camera, target=target)))
                               for camera in scene.CAMERAS}
                views[pose][scene.PREVIEW_CAMERA] = str(run.write_png(
                    f"{pose}_{scene.PREVIEW_CAMERA}", renderer.render(scene.PREVIEW_CAMERA)
                ))
        results = {"scene": scene.scene_record(args.hand, view), "poses": poses, "views": views}
    else:
        for pose in scene.POSES:
            pose_q(pose)
        results = {"scene": scene.scene_record(args.hand, view)}

    record_path = run.write(asset=asset.to_record(), load_report=loaded.report, results=results, rendering=RENDERING)
    print(f"record: {record_path}")
    print(f"artifacts: {sum(len(v) for v in views.values())}")

    if not args.headless:
        def frame(t: float) -> None:
            pose = scene.POSES[round(t / LIVE_HOLD_S) % len(scene.POSES)]
            set_pose(pose_q(pose))

        with rendering.LiveDisplay(loaded, view, BUILT_FOVY_DEG) as display:
            rendering.run_live(display, LIVE_HOLD_S, frame)


if __name__ == "__main__":
    try:
        main(sys.argv)
    except (assets.AssetError, profiles.ProfileError, records.RecordError, conventions.ViewerError,
            FileExistsError) as exc:
        sys.exit(f"error: {exc}")
