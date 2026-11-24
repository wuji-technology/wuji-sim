# SPDX-License-Identifier: Apache-2.0
import argparse
import sys
from pathlib import Path

import numpy as np

from wuji_sim import assets, conventions, framing, profiles, records, scene
from wuji_sim.platforms import motrixsim as platform

sys.path.insert(0, str(Path(__file__).resolve().parent / "_shared"))

import rendering  # noqa: E402

DEMO = "01_load_and_view"
RENDERING = "MotrixSim physics / MotrixSim rendering"
# The system camera is built with the fixed cameras' fovy, so front/side/back are exact and only the
# preview camera's distance is scaled (see examples/motrixsim/_shared/rendering.py).
BUILT_FOVY_DEG = rendering.FIXED_CAMERA_FOVY_DEG


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Load the hand model into the demo scene and render it at qpos = 0.")
    assets.add_cli_args(parser)
    records.add_cli_args(parser)
    scene.add_cli_args(parser)
    args = parser.parse_args(argv[1:])

    asset = assets.resolve(args.hand, args.asset_root, strict=args.strict)
    print(asset.describe())
    run = records.RunRecord(platform=platform.PLATFORM, demo=DEMO, hand=args.hand,
                            output_dir=args.output_dir, argv=argv)
    # Framed like 03: its tracking motion passes through every pose this demo shows.
    view = framing.view(asset.mjcf, args.hand, platform.PLATFORM, "tracking")
    loaded = platform.load(asset.mjcf, scene=lambda spec: scene.add_scene(spec, args.hand, view),
                           description_version=asset.expected_version, render_resolution=view.render_size,
                           render_fovy=BUILT_FOVY_DEG)
    zeros = np.zeros(len(loaded.joint_names))
    platform.set_state(loaded, zeros)
    results = {"scene": scene.scene_record(args.hand, view), "qpos": "zeros"}
    if args.headless:
        views = {}
        # The mujoco front/side/back cameras track the middle-finger proximal body; the look-at point follows it.
        target = platform.link_pose(
            loaded, conventions.HAND_PREFIXES[args.hand] + "middle_finger_proximal")[:3]
        with rendering.OffscreenRenderer(loaded, view, args.hand, BUILT_FOVY_DEG) as renderer:
            renderer.sync()
            for camera in scene.CAMERAS:
                views[camera] = str(run.write_png(camera, renderer.render(camera, target=target)))
            views[scene.PREVIEW_CAMERA] = str(run.write_png(scene.PREVIEW_CAMERA,
                                                            renderer.render(scene.PREVIEW_CAMERA)))
        results["views"] = views

    record_path = run.write(asset=asset.to_record(), load_report=loaded.report, results=results, rendering=RENDERING)
    print(f"record: {record_path}")
    print(f"artifacts: {len(results.get('views', {}))}")

    if not args.headless:
        def frame(t: float) -> None:
            platform.apply_mit(loaded, zeros, zeros)
            platform.step(loaded)

        with rendering.LiveDisplay(loaded, view, BUILT_FOVY_DEG) as display:
            rendering.run_live(display, float(loaded.model.opt.timestep), frame)


if __name__ == "__main__":
    try:
        main(sys.argv)
    except (assets.AssetError, profiles.ProfileError, records.RecordError, conventions.ViewerError,
            FileExistsError) as exc:
        sys.exit(f"error: {exc}")
