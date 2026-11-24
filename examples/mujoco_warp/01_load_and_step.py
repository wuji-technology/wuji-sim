# SPDX-License-Identifier: Apache-2.0
import argparse
import sys
import time

import mujoco_warp as mjw
import numpy as np

from wuji_sim import assets, conventions, framing, profiles, records, scene
from wuji_sim.platforms import mujoco_warp as platform

DEMO = "01_load_and_step"
NWORLD = 1
DURATION_S = 1.0
CAMERA = "front"
RENDERING = "Warp physics / MuJoCo rendering"
FINITE_FIELDS = ("qpos", "qvel", "qacc", "ctrl", "qfrc_actuator")


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Load a hand model into MuJoCo Warp, step it with zero torque and render the first and last frame."
    )
    assets.add_cli_args(parser)
    records.add_cli_args(parser)
    scene.add_cli_args(parser)
    args = parser.parse_args(argv)

    asset = assets.resolve(args.hand, args.asset_root, strict=args.strict)
    print(asset.describe())
    run = records.RunRecord(
        platform=platform.PLATFORM, demo=DEMO, hand=args.hand, output_dir=args.output_dir, argv=sys.argv
    )
    # Framed like 03: its tracking motion passes through the q = 0 pose this demo starts from.
    view = framing.view(asset.mjcf, args.hand, platform.PLATFORM, "tracking")
    loaded = platform.load(asset.mjcf, nworld=NWORLD, scene=lambda spec: scene.add_scene(spec, args.hand, view),
                           description_version=asset.expected_version)

    timestep = float(loaded.model.opt.timestep)

    if args.headless:
        steps = scene.whole_steps(DURATION_S, timestep)
        qpos0 = loaded.d.qpos.numpy()[0].copy()
        peak = {}
        step_s = []
        frames = {}
        with (
            scene.SceneRenderer(loaded.model) as renderer,
            scene.SceneRenderer(loaded.model, *view.render_size) as preview,
            scene.Display(loaded.model, loaded.data, view, headless=True) as display,
        ):
            data = platform.host_data(loaded)
            for camera, camera_renderer in ((CAMERA, renderer), (scene.PREVIEW_CAMERA, preview)):
                label = f"{camera}_step{0:04d}"
                path = run.write_png(label, camera_renderer.render(data, camera))
                frames[label] = {"path": str(path), "step": 0, "t_s": 0.0, "world": 0, "camera": camera}
            for _ in range(steps):
                start = time.perf_counter()
                mjw.step(loaded.m, loaded.d)
                platform.check_capacity(loaded, peak)
                step_s.append(time.perf_counter() - start)
                if display.is_running():
                    platform.host_data(loaded)
                    display.sync()
            label = f"{CAMERA}_step{steps:04d}"
            path = run.write_png(label, renderer.render(platform.host_data(loaded), CAMERA))
            frames[label] = {"path": str(path), "step": steps, "t_s": steps * timestep, "world": 0, "camera": CAMERA}

        finite = {name: bool(np.isfinite(getattr(loaded.d, name).numpy()).all()) for name in FINITE_FIELDS}
        results = {
            "scene": scene.scene_record(args.hand, view),
            "nworld": NWORLD,
            "duration_s": steps * timestep,
            "steps": steps,
            "timestep_s": timestep,
            "control": "zero torque: ctrl = 0 on every motor",
            "finite": finite,
            "max_abs_qpos_change_rad": float(np.abs(loaded.d.qpos.numpy()[0] - qpos0).max()),
            "first_step_s": step_s[0],
            "step_time_ms": 1000.0 * float(np.mean(step_s[1:])),
            "step_time_note": (
                "mean wall time of steps 2..N, each mjw.step plus the check_capacity device readback; "
                "first_step_s includes loading or compiling the Warp kernels"
            ),
            "capacity_usage": peak,
            "frames": frames,
        }
        record = run.write(asset=asset.to_record(), load_report=loaded.report, results=results, rendering=RENDERING)
        print(
            f"{steps} zero-torque steps ({steps * timestep:g} s at dt {timestep:g} s): finite {finite}, "
            f"max |dq| {results['max_abs_qpos_change_rad']:.4f} rad, "
            f"{results['step_time_ms']:.3f} ms/step, capacity usage {peak}"
        )
        print(f"record: {record.resolve()}")
        print(f"artifacts: {len(frames)}")
    else:
        results = {"scene": scene.scene_record(args.hand, view), "nworld": NWORLD,
                  "control": "zero torque: ctrl = 0 on every motor"}
        record = run.write(asset=asset.to_record(), load_report=loaded.report, results=results, rendering=RENDERING)
        print(f"record: {record.resolve()}")
        print("artifacts: 0")

        peak = {}

        def frame(t: float) -> None:
            mjw.step(loaded.m, loaded.d)
            platform.check_capacity(loaded, peak)
            platform.host_data(loaded)

        with scene.Display(loaded.model, loaded.data, view, headless=False) as display:
            scene.run_live(display, timestep, frame)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except (assets.AssetError, profiles.ProfileError, records.RecordError, conventions.ViewerError,
            FileExistsError) as exc:
        sys.exit(f"error: {exc}")
