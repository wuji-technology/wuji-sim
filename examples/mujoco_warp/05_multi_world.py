# SPDX-License-Identifier: Apache-2.0
import argparse
import math
import sys

import mujoco
import mujoco_warp as mjw
import numpy as np

from wuji_sim import assets, conventions, framing, mit, profiles, records, scene
from wuji_sim.platforms import mujoco_warp as platform

DEMO = "05_multi_world"
VIDEO_FPS = 25
VIDEO_CAMERA = scene.PREVIEW_CAMERA
COMPARISON_CAMERA = "front"
RENDERING = "Warp physics / MuJoCo rendering"
CONTROL_LAW = "tau = clip(kp*(q_des-q) + kd*(qd_des-qd) + tau_ff, -tau_limit, tau_limit)"
REFERENCE = (
    f"wuji_sim.scene.parallel_reference: world w starts on POSES[w mod {len(scene.POSES)}] and from there cycles "
    f"through TRACKING_SEQUENCE {list(scene.TRACKING_SEQUENCE)} closed on itself, min-jerk to the next pose in "
    "move_s, then holding it for hold_s; one cycle lasts len(TRACKING_SEQUENCE) * (move_s + hold_s) and ends on the "
    "pose it started from"
)


def steps_per_frame(timestep: float) -> int:
    ratio = 1.0 / (VIDEO_FPS * timestep)
    steps = round(ratio)
    if steps < 1 or not math.isclose(ratio, steps, rel_tol=1e-9):
        raise ValueError(f"video frame period 1/{VIDEO_FPS} s is not a whole number of {timestep} s physics steps")
    return steps


def held_pose(world: int, segment: int) -> str:
    # Mirrors scene.parallel_reference: world w starts at POSES[w mod 4]'s first slot in TRACKING_SEQUENCE.
    sequence = scene.TRACKING_SEQUENCE
    start = sequence.index(scene.POSES[world % len(scene.POSES)])
    return sequence[(start + segment + 1) % len(sequence)]


def mit_command(joint_names: tuple[str, ...], nworld: int, t: float, gains: dict) -> mit.MitCommand:
    references = [scene.parallel_reference(joint_names, world, t) for world in range(nworld)]
    return mit.MitCommand(
        joint_names, np.stack([q for q, _ in references]), np.stack([qd for _, qd in references]), **gains
    )


def upload(buffers: platform.MitBuffers, cmd: mit.MitCommand, nworld: int) -> None:
    shape = (nworld, len(cmd.joint_names))
    for name in platform.MIT_FIELDS:
        value = np.broadcast_to(getattr(cmd, name), shape)
        getattr(buffers, name).assign(np.ascontiguousarray(value, dtype=np.float32))


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Parallel MuJoCo Warp worlds cycle through the preset poses, each starting from a different "
        "one; every world is replayed on CPU MuJoCo with the same model and controller for comparison."
    )
    assets.add_cli_args(parser)
    records.add_cli_args(parser)
    scene.add_cli_args(parser)
    parser.add_argument(
        "--nworld", type=int, default=4, help=f"number of Warp worlds (>= 1); world i starts on {scene.POSES}[i %% 4]"
    )
    args = parser.parse_args(argv)
    if args.nworld < 1:
        parser.error(f"--nworld must be >= 1, got {args.nworld}")

    asset = assets.resolve(args.hand, args.asset_root, strict=args.strict)
    print(asset.describe())
    run = records.RunRecord(
        platform=platform.PLATFORM, demo=DEMO, hand=args.hand, output_dir=args.output_dir, argv=sys.argv
    )
    view = framing.view(asset.mjcf, args.hand, platform.PLATFORM, "grid", worlds=args.nworld)
    loaded = platform.load(asset.mjcf, nworld=args.nworld, scene=lambda spec: scene.add_scene(spec, args.hand, view),
                           description_version=asset.expected_version)

    model = loaded.model
    joint_names = loaded.joint_names
    timestep = float(model.opt.timestep)
    steps = scene.whole_steps(scene.parallel_duration(), timestep)
    segment_steps = scene.whole_steps(scene.MOVE_S + scene.HOLD_S, timestep)
    frame_steps = steps_per_frame(timestep)
    poses = [scene.POSES[w % len(scene.POSES)] for w in range(args.nworld)]
    n = len(joint_names)
    gains = {"kp": np.full(n, scene.DEMO_KP), "kd": np.full(n, scene.DEMO_KD), "tau_ff": np.zeros(n)}
    stability = platform.mit_stability(loaded, gains["kp"], gains["kd"])
    platform.set_mit_gains(loaded, gains["kp"], gains["kd"])
    print(
        f"MIT stability ({stability['control_path']}): max ratio {stability['max_ratio']:.3f} "
        f"at {stability['max_joint']} (dt {timestep:g} s, limit {stability['limit']:g})"
    )
    qpos_adr, act_ids = list(loaded.qpos_adr), list(loaded.actuator_ids)
    buffers = platform.mit_buffers(loaded)

    qpos = loaded.d.qpos.numpy()
    qpos[:, qpos_adr] = np.stack([scene.parallel_reference(joint_names, w, 0.0)[0] for w in range(args.nworld)])
    loaded.d.qpos.assign(qpos)
    platform.host_data(loaded, 0)

    # add_scene put world 0 at view.hand_roots[0]; the other worlds are drawn shifted relative to it.
    offsets = np.array(view.hand_roots) - view.hand_roots[0]
    pert = mujoco.MjvPerturb()

    def add_other_worlds(option: mujoco.MjvOption, scn: mujoco.MjvScene) -> None:
        for w in range(1, args.nworld):
            platform.add_world_geoms(loaded, w, offsets[w], option, pert, scn)
        # add_world_geoms leaves loaded.data on the last world it drew.
        platform.host_data(loaded, 0)

    if args.headless:
        warp_q = np.empty((steps + 1, args.nworld, n))
        warp_q[0] = loaded.d.qpos.numpy()[:, qpos_adr]
        peak = {}
        worlds = []
        with scene.SceneRenderer(model, *view.render_size) as renderer, scene.SceneRenderer(model) as comparison:

            def preview_frame() -> np.ndarray:
                return renderer.render(platform.host_data(loaded, 0), VIDEO_CAMERA,
                                       extra=lambda scn: add_other_worlds(renderer.option, scn))

            # Enter SceneRenderer (EGL) before Display (GLFW): the reverse order breaks later EGL context creation.
            with (
                scene.Display(model, loaded.data, view, headless=True) as display,
                run.open_video(VIDEO_CAMERA, VIDEO_FPS) as video,
            ):
                video.append(preview_frame())
                for k in range(steps):
                    upload(buffers, mit_command(joint_names, args.nworld, k * timestep, gains), args.nworld)
                    platform.apply_mit(loaded, buffers)
                    mjw.step(loaded.m, loaded.d)
                    platform.check_capacity(loaded, peak)
                    warp_q[k + 1] = loaded.d.qpos.numpy()[:, qpos_adr]
                    if (k + 1) % frame_steps == 0:
                        video.append(preview_frame())
                    if display.is_running():
                        platform.host_data(loaded, 0)
                        display.sync()

            for w, pose in enumerate(poses):
                warp_image = comparison.render(platform.host_data(loaded, w), COMPARISON_CAMERA)
                cpu = mujoco.MjData(model)
                cpu.qpos[qpos_adr] = warp_q[0, w]
                mujoco.mj_forward(model, cpu)
                cpu_q = np.empty((steps + 1, n))
                cpu_q[0] = cpu.qpos[qpos_adr]
                for k in range(steps):
                    q_des, qd_des = scene.parallel_reference(joint_names, w, k * timestep)
                    cpu.ctrl[act_ids] = gains["kp"] * q_des + gains["kd"] * qd_des + gains["tau_ff"]
                    mujoco.mj_step(model, cpu)
                    cpu_q[k + 1] = cpu.qpos[qpos_adr]
                mujoco.mj_forward(model, cpu)
                cpu_image = comparison.render(cpu, COMPARISON_CAMERA)
                path = run.write_png(f"world{w}_{pose}", np.concatenate([warp_image, cpu_image], axis=1))
                goal = scene.pose_vector(joint_names, pose)
                hold_end = []
                for segment in range(len(scene.TRACKING_SEQUENCE)):
                    step = (segment + 1) * segment_steps
                    held = held_pose(w, segment)
                    target = scene.pose_vector(joint_names, held)
                    hold_end.append(
                        {
                            "t_s": step * timestep,
                            "pose": held,
                            "warp_error_max_rad": float(np.abs(warp_q[step, w] - target).max()),
                            "mujoco_error_max_rad": float(np.abs(cpu_q[step] - target).max()),
                        }
                    )
                worlds.append(
                    {
                        "world": w,
                        "pose": pose,
                        "warp_final_error_max_rad": float(np.abs(warp_q[-1, w] - goal).max()),
                        "mujoco_final_error_max_rad": float(np.abs(cpu_q[-1] - goal).max()),
                        "max_abs_qpos_diff_rad": float(np.abs(warp_q[:, w] - cpu_q).max()),
                        "final_max_abs_qpos_diff_rad": float(np.abs(warp_q[-1, w] - cpu_q[-1]).max()),
                        "warp_final_q_rad": warp_q[-1, w].tolist(),
                        "hold_end": hold_end,
                        "image": str(path),
                    }
                )
                print(
                    f"world {w} ({pose}): final max |q - pose| Warp {worlds[-1]['warp_final_error_max_rad']:.4f} rad, "
                    f"MuJoCo {worlds[-1]['mujoco_final_error_max_rad']:.4f} rad; hold-end max |q - pose| Warp "
                    f"{max(entry['warp_error_max_rad'] for entry in hold_end):.4f} rad; max |q_warp - q_mujoco| "
                    f"{worlds[-1]['max_abs_qpos_diff_rad']:.2e} rad"
                )
        results = {
            "scene": scene.scene_record(args.hand, view),
            "nworld": args.nworld,
            "reference": REFERENCE,
            "move_s": scene.MOVE_S,
            "hold_s": scene.HOLD_S,
            "steps": steps,
            "control": {
                "law": CONTROL_LAW,
                "warp": "wuji_sim.platforms.mujoco_warp.mit_ctrl_kernel on the GPU (float32), all worlds in one launch",
                "mujoco": "MuJoCo general actuators with affine kp/kd bias (float64), one world at a time",
                "kp": scene.DEMO_KP,
                "kd": scene.DEMO_KD,
                "tau_ff": 0.0,
                "units": {"kp": "N*m/rad", "kd": "N*m*s/rad", "tau_ff": "N*m", "tau_limit": "N*m", "period": "s"},
                "period_s": timestep,
                "updates_per_physics_step": 1,
                "joint_order": list(joint_names),
                "actuator_ids": act_ids,
                "tau_limit": list(loaded.tau_limit),
                "stability": stability,
            },
            "comparison": {
                "mujoco_reference": "same compiled MjModel as the Warp worlds, a new MjData per world started from "
                                    "that world's initial joint positions, mujoco.mj_step",
                "final_error": "max over the joints of |q(t_end) - pose|, rad; pose is the preset the world starts "
                               "and ends on",
                "hold_end_error": "max over the joints of |q(t) - pose| at the end of every hold, "
                                  "t = (k + 1) * (move_s + hold_s), rad; pose is the preset held there",
                "max_abs_qpos_diff_rad": "max over all steps and joints of |q_warp - q_mujoco|, rad",
                "final_max_abs_qpos_diff_rad": "max over the joints of |q_warp - q_mujoco| at t_end, rad",
                "warp_final_q_rad": "q_warp(t_end) in control.joint_order, rad",
            },
            "images": (
                f"per world {COMPARISON_CAMERA} camera, 640x480 halves: left Warp final frame (Warp physics / MuJoCo "
                "rendering via host_data), right MuJoCo CPU reference final frame (MuJoCo physics / MuJoCo rendering)"
            ),
            "worlds": worlds,
            "video": {
                "path": str(video.path),
                "camera": VIDEO_CAMERA,
                "fps": VIDEO_FPS,
                "frames": video.frames,
                "steps_per_frame": frame_steps,
                "worlds": args.nworld,
            },
            "capacity_usage": peak,
        }
        record = run.write(asset=asset.to_record(), load_report=loaded.report, results=results, rendering=RENDERING)
        print(f"{steps} steps, {video.frames} preview frames of every world, capacity usage {peak}")
        print(f"record: {record.resolve()}")
        print(f"artifacts: {len(worlds) + 1}")
    else:
        results = {"scene": scene.scene_record(args.hand, view), "nworld": args.nworld, "reference": REFERENCE}
        record = run.write(asset=asset.to_record(), load_report=loaded.report, results=results, rendering=RENDERING)
        print(f"record: {record.resolve()}")
        print("artifacts: 0")

        option = mujoco.MjvOption()
        # Same visual-only groups as scene.Display.
        option.geomgroup[2] = 0
        option.sitegroup[:] = 0

        peak = {}

        def frame(t: float) -> None:
            upload(buffers, mit_command(joint_names, args.nworld, t, gains), args.nworld)
            platform.apply_mit(loaded, buffers)
            mjw.step(loaded.m, loaded.d)
            platform.check_capacity(loaded, peak)
            display.user_scn.ngeom = 0
            add_other_worlds(option, display.user_scn)

        with scene.Display(model, loaded.data, view, headless=False) as display:
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
