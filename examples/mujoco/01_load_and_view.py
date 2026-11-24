# SPDX-License-Identifier: Apache-2.0
import argparse
import sys

import mujoco

from wuji_sim import assets, conventions, framing, profiles, records, scene
from wuji_sim.platforms import mujoco as platform

DEMO = "01_load_and_view"
RENDERING = "MuJoCo physics / MuJoCo rendering"


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
                           description_version=asset.expected_version)
    model, data = loaded.model, loaded.data

    data.qpos[:] = 0.0
    mujoco.mj_forward(model, data)
    results = {"scene": scene.scene_record(args.hand, view), "qpos": "zeros"}
    if args.headless:
        views = {}
        with scene.SceneRenderer(model, world_frame=True) as renderer:
            for camera in scene.CAMERAS:
                views[camera] = str(run.write_png(camera, renderer.render(data, camera)))
        with scene.SceneRenderer(model, *view.render_size) as renderer:
            views[scene.PREVIEW_CAMERA] = str(
                run.write_png(scene.PREVIEW_CAMERA, renderer.render(data, scene.PREVIEW_CAMERA))
            )
        results["views"] = views

    record_path = run.write(asset=asset.to_record(), load_report=loaded.report, results=results, rendering=RENDERING)
    print(f"record: {record_path}")
    print(f"artifacts: {len(results.get('views', {}))}")

    def frame(t: float) -> None:
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)

    with scene.Display(model, data, view, headless=args.headless) as display:
        scene.run_live(display, model.opt.timestep, frame)


if __name__ == "__main__":
    try:
        main(sys.argv)
    except (assets.AssetError, profiles.ProfileError, records.RecordError, conventions.ViewerError,
            FileExistsError) as exc:
        sys.exit(f"error: {exc}")
