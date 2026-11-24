# SPDX-License-Identifier: Apache-2.0
import argparse
import math
import sys

import mujoco_warp as mjw
import numpy as np

from wuji_sim import assets, conventions, framing, mit, profiles, records, scene
from wuji_sim.platforms import mujoco_warp as platform

DEMO = "03_mit_tracking"
NWORLD = 1
WORLD = 0
CAMERA = scene.PREVIEW_CAMERA
VIDEO_FPS = 25
RENDERING = "Warp physics / MuJoCo rendering"
CONTROL_LAW = "tau = clip(kp*(q_des-q) + kd*(qd_des-qd) + tau_ff, -tau_limit, tau_limit)"


def steps_per_frame(timestep: float) -> int:
    ratio = 1.0 / (VIDEO_FPS * timestep)
    steps = round(ratio)
    if steps < 1 or not math.isclose(ratio, steps, rel_tol=1e-9):
        raise ValueError(f"video frame period 1/{VIDEO_FPS} s is not a whole number of {timestep} s physics steps")
    return steps


def upload(buffers: platform.MitBuffers, cmd: mit.MitCommand, nworld: int) -> None:
    shape = (nworld, len(cmd.joint_names))
    for name in platform.MIT_FIELDS:
        value = np.broadcast_to(getattr(cmd, name), shape)
        getattr(buffers, name).assign(np.ascontiguousarray(value, dtype=np.float32))


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="MIT tracking of the preset-pose sequence on MuJoCo Warp; records a video of world 0."
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
    view = framing.view(asset.mjcf, args.hand, platform.PLATFORM, "tracking")
    loaded = platform.load(asset.mjcf, nworld=NWORLD, scene=lambda spec: scene.add_scene(spec, args.hand, view),
                           description_version=asset.expected_version)

    timestep = float(loaded.model.opt.timestep)
    duration = scene.tracking_duration()
    steps = round(duration / timestep)
    frame_steps = steps_per_frame(timestep)
    n = len(loaded.joint_names)
    gains = {"kp": np.full(n, scene.DEMO_KP), "kd": np.full(n, scene.DEMO_KD), "tau_ff": np.zeros(n)}
    stability = platform.mit_stability(loaded, gains["kp"], gains["kd"])
    platform.set_mit_gains(loaded, gains["kp"], gains["kd"])
    print(
        f"MIT stability ({stability['control_path']}): max ratio {stability['max_ratio']:.3f} "
        f"at {stability['max_joint']} (dt {timestep:g} s, limit {stability['limit']:g})"
    )
    qpos_adr, dof_adr, act_ids = list(loaded.qpos_adr), list(loaded.dof_adr), list(loaded.actuator_ids)
    tau_limit = np.array(loaded.tau_limit, dtype=np.float32)
    buffers = platform.mit_buffers(loaded)

    control = {
        "law": CONTROL_LAW,
        "computed_by": "wuji_sim.platforms.mujoco_warp.mit_ctrl_kernel on the GPU (float32)",
        "kp": scene.DEMO_KP,
        "kd": scene.DEMO_KD,
        "tau_ff": 0.0,
        "units": {
            "kp": "N*m/rad",
            "kd": "N*m*s/rad",
            "tau_ff": "N*m",
            "tau_limit": "N*m",
            "control_period": "s",
            "q": "rad",
            "qd": "rad/s",
        },
        "control_period": timestep,
        "updates_per_physics_step": 1,
        "joint_order": list(loaded.joint_names),
        "actuator_ids": act_ids,
        "tau_limit": list(loaded.tau_limit),
        "stability": stability,
    }

    if args.headless:
        error = np.empty((steps, n))
        saturated = np.empty((steps, n), dtype=bool)
        peak = {}
        label = f"{CAMERA}_world{WORLD}"
        with (
            scene.SceneRenderer(loaded.model, *view.render_size) as renderer,
            scene.Display(loaded.model, loaded.data, view, headless=True) as display,
            run.open_video(label, VIDEO_FPS) as video,
        ):
            video.append(renderer.render(platform.host_data(loaded, WORLD), CAMERA))
            for k in range(steps):
                q_des, qd_des = scene.tracking_reference(loaded.joint_names, k * timestep)
                cmd = mit.MitCommand(loaded.joint_names, q_des, qd_des, **gains)
                upload(buffers, cmd, NWORLD)
                platform.apply_mit(loaded, buffers)
                error[k] = q_des - loaded.d.qpos.numpy()[WORLD, qpos_adr]
                mjw.step(loaded.m, loaded.d)
                saturated[k] = np.abs(loaded.d.qfrc_actuator.numpy()[WORLD, dof_adr]) >= tau_limit
                platform.check_capacity(loaded, peak)
                if (k + 1) % frame_steps == 0:
                    video.append(renderer.render(platform.host_data(loaded, WORLD), CAMERA))
                if display.is_running():
                    platform.host_data(loaded, WORLD)
                    display.sync()

        rms = np.sqrt(np.mean(error**2, axis=0))
        max_abs = np.max(np.abs(error), axis=0)
        saturation = saturated.mean(axis=0)
        results = {
            "scene": scene.scene_record(args.hand, view),
            "nworld": NWORLD,
            "control": control,
            "tracking": {
                "reference": {
                    "sequence": list(scene.TRACKING_SEQUENCE),
                    "move_s": scene.MOVE_S,
                    "hold_s": scene.HOLD_S,
                    "profile": "min-jerk",
                },
                "duration_s": steps * timestep,
                "steps": steps,
                "error_definition": f"q_des(t) - q(t) of world {WORLD}, sampled before each physics step",
                "max_rms_error_rad": float(rms.max()),
                "max_abs_error_rad": float(max_abs.max()),
                "joints": {
                    name: {
                        "rms_error_rad": float(rms[i]),
                        "max_abs_error_rad": float(max_abs[i]),
                        "saturation_fraction": float(saturation[i]),
                    }
                    for i, name in enumerate(loaded.joint_names)
                },
            },
            "video": {
                "path": str(video.path),
                "camera": CAMERA,
                "fps": VIDEO_FPS,
                "frames": video.frames,
                "steps_per_frame": frame_steps,
                "world": WORLD,
                "label": label,
            },
            "capacity_usage": peak,
        }
        record = run.write(asset=asset.to_record(), load_report=loaded.report, results=results, rendering=RENDERING)
        print(
            f"{steps} steps: max RMS error {rms.max():.4f} rad, max |error| {max_abs.max():.4f} rad, "
            f"max saturation {saturation.max():.3f}, {video.frames} video frames, capacity usage {peak}"
        )
        print(f"record: {record.resolve()}")
        print("artifacts: 1")
    else:
        results = {"scene": scene.scene_record(args.hand, view), "nworld": NWORLD, "control": control}
        record = run.write(asset=asset.to_record(), load_report=loaded.report, results=results, rendering=RENDERING)
        print(f"record: {record.resolve()}")
        print("artifacts: 0")

        peak = {}

        def frame(t: float) -> None:
            q_des, qd_des = scene.tracking_reference(loaded.joint_names, t % duration)
            cmd = mit.MitCommand(loaded.joint_names, q_des, qd_des, **gains)
            upload(buffers, cmd, NWORLD)
            platform.apply_mit(loaded, buffers)
            mjw.step(loaded.m, loaded.d)
            platform.check_capacity(loaded, peak)
            platform.host_data(loaded, WORLD)

        with scene.Display(loaded.model, loaded.data, view, headless=False) as display:
            scene.run_live(display, timestep, frame)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except (
        assets.AssetError,
        profiles.ProfileError,
        mit.MitStabilityError,
        records.RecordError,
        conventions.ViewerError,
        FileExistsError,
    ) as exc:
        sys.exit(f"error: {exc}")
