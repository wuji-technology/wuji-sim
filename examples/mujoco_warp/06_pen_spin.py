# SPDX-License-Identifier: Apache-2.0
import argparse
import math
import sys
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np

from wuji_sim import assets, conventions, framing, mit, profiles, records, scene
from wuji_sim.platforms import mujoco_warp as platform

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))

import pen_spin  # noqa: E402

DEMO = "06_pen_spin"
NWORLD = 1
WORLD = 0
CAMERA = scene.PREVIEW_CAMERA
VIDEO_FPS = 25
RENDERING = "Warp physics / MuJoCo rendering"
CONTROL_LAW = "tau = clip(kp*(q_des-q) + kd*(qd_des-qd) + tau_ff, -tau_limit, tau_limit)"


def steps_per_frame(period: float) -> int:
    ratio = 1.0 / (VIDEO_FPS * period)
    steps = round(ratio)
    if steps < 1 or not math.isclose(ratio, steps, rel_tol=1e-9):
        raise ValueError(f"video frame period 1/{VIDEO_FPS} s is not a whole number of {period} s control steps")
    return steps


def upload(buffers: platform.MitBuffers, cmd: mit.MitCommand, nworld: int) -> None:
    shape = (nworld, len(cmd.joint_names))
    for name in platform.MIT_FIELDS:
        value = np.broadcast_to(getattr(cmd, name), shape)
        getattr(buffers, name).assign(np.ascontiguousarray(value, dtype=np.float32))


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Replay a published pen-spinning policy on the right hand in one MuJoCo Warp world."
    )
    assets.add_cli_args(parser)
    records.add_cli_args(parser)
    scene.add_cli_args(parser)
    pen_spin.add_cli_args(parser)
    args = parser.parse_args(argv)
    if args.hand != pen_spin.HAND:
        parser.error(f"the published policy drives the {pen_spin.HAND} hand; rerun with --hand {pen_spin.HAND}")

    asset = assets.resolve(args.hand, args.asset_root, strict=args.strict)
    print(asset.describe())
    task = pen_spin.load(args.group, args.clip, args.cache_dir)
    print(task.describe())
    reference = task.reference
    policy = pen_spin.Policy(task.model)
    run = records.RunRecord(
        platform=platform.PLATFORM, demo=DEMO, hand=args.hand, output_dir=args.output_dir, argv=sys.argv
    )
    view = framing.view(asset.mjcf, args.hand, platform.PLATFORM, "pen_spin")
    wrist = pen_spin.wrist_pose(view.hand_roots[0])

    def build(model_source: mujoco.MjSpec) -> None:
        scene.add_scene(model_source, args.hand, view)
        pen_spin.add_pen_scene(model_source, args.hand, wrist, pen_spin.to_world(reference.pen_pose(0), wrist))

    loaded = platform.load(asset.mjcf, nworld=NWORLD, scene=build, description_version=asset.expected_version)
    joint_names = loaded.joint_names
    pen_spin.check_joint_order(joint_names, args.hand)
    n = len(joint_names)
    timestep = float(loaded.model.opt.timestep)
    substeps = scene.whole_steps(pen_spin.CONTROL_DT, timestep)
    qpos_adr, dof_adr, act_ids = list(loaded.qpos_adr), list(loaded.dof_adr), list(loaded.actuator_ids)
    pen_adr = pen_spin.pen_qpos_adr(loaded.model)
    kp, kd = pen_spin.gains()
    gains = {"kp": kp, "kd": kd, "tau_ff": np.zeros(n)}
    # The policy was trained on position targets alone.
    qd_des = np.zeros(n)
    stability = platform.mit_stability(loaded, kp, kd)
    platform.set_mit_gains(loaded, kp, kd)
    print(
        f"MIT stability ({stability['control_path']}): max ratio {stability['max_ratio']:.3f} "
        f"at {stability['max_joint']} (dt {timestep:g} s, limit {stability['limit']:g})"
    )
    buffers = platform.mit_buffers(loaded)
    residual = pen_spin.ResidualAction(task.config)
    last_action = np.zeros(pen_spin.ACTION_DIM, dtype=np.float32)

    control = {
        "law": CONTROL_LAW,
        "computed_by": "wuji_sim.platforms.mujoco_warp.mit_ctrl_kernel on the GPU (float32)",
        "targets_computed_by": "the policy on the CPU, from the joint state and pen pose read back every control step",
        "kp": kp.tolist(),
        "kd": kd.tolist(),
        "qd_des": 0.0,
        "tau_ff": 0.0,
        "gains_source": "the gains the policy was trained with",
        "units": {"kp": "N*m/rad", "kd": "N*m*s/rad", "tau_ff": "N*m", "tau_limit": "N*m",
                  "control_period": "s", "physics_timestep": "s", "q": "rad", "qd": "rad/s"},
        "control_period": pen_spin.CONTROL_DT,
        "physics_timestep": timestep,
        "physics_steps_per_control_step": substeps,
        "joint_order": list(joint_names),
        "actuator_ids": act_ids,
        "tau_limit": list(loaded.tau_limit),
        "stability": stability,
    }

    def pen_pose() -> np.ndarray:
        return loaded.d.qpos.numpy()[WORLD, pen_adr:pen_adr + 7]

    def reset(step: int) -> None:
        nonlocal last_action
        mjw.reset_data(loaded.m, loaded.d)
        qpos = loaded.d.qpos.numpy()
        qpos[WORLD, qpos_adr] = reference.joint_pos(step)
        qpos[WORLD, pen_adr:pen_adr + 7] = pen_spin.to_world(reference.pen_pose(step), wrist)
        loaded.d.qpos.assign(qpos)
        residual.reset()
        last_action = np.zeros(pen_spin.ACTION_DIM, dtype=np.float32)

    def advance(step: int, peak: dict) -> None:
        nonlocal last_action
        if step > 0 and reference.restart(step):
            reset(step)
        qpos, qvel = loaded.d.qpos.numpy()[WORLD], loaded.d.qvel.numpy()[WORLD]
        pen_local = pen_spin.to_wrist_local(qpos[pen_adr:pen_adr + 7], wrist)
        obs = pen_spin.observation(reference.command(step), qpos[qpos_adr], qvel[dof_adr], pen_local, last_action)
        last_action = policy(obs)
        q_des = residual.step(last_action, reference.joint_pos(step))
        upload(buffers, mit.MitCommand(joint_names, q_des, qd_des, **gains), NWORLD)
        platform.apply_mit(loaded, buffers)
        for _ in range(substeps):
            mjw.step(loaded.m, loaded.d)
            platform.check_capacity(loaded, peak)

    reset(0)
    scene_results = pen_spin.scene_record(args.hand, view, wrist)
    if args.headless:
        frame_steps = steps_per_frame(pen_spin.CONTROL_DT)
        log = pen_spin.PenLog(reference, wrist)
        log.start(pen_pose())
        peak = {}
        label = f"{CAMERA}_world{WORLD}"
        with (
            scene.SceneRenderer(loaded.model, *view.render_size) as renderer,
            run.open_video(label, VIDEO_FPS) as video,
        ):
            video.append(renderer.render(platform.host_data(loaded, WORLD), CAMERA))
            for step in range(reference.num_steps):
                advance(step, peak)
                log.update(step, pen_pose())
                if (step + 1) % frame_steps == 0:
                    video.append(renderer.render(platform.host_data(loaded, WORLD), CAMERA))

        replay = {"steps": reference.num_steps, "duration_s": reference.num_steps * pen_spin.CONTROL_DT, **log.record()}
        results = {
            "scene": scene_results,
            "nworld": NWORLD,
            "policy": {**task.record(), "inference": policy.record()},
            "control": control,
            "pen": pen_spin.pen_record(),
            "replay": replay,
            "video": {"path": str(video.path), "camera": CAMERA, "fps": VIDEO_FPS, "frames": video.frames,
                      "control_steps_per_frame": frame_steps, "world": WORLD, "label": label},
            "capacity_usage": peak,
        }
        record = run.write(asset=asset.to_record(), load_report=loaded.report, results=results, rendering=RENDERING)
        print(f"replay: {pen_spin.describe_replay(replay)}, capacity usage {peak}")
        print(f"record: {record.resolve()}")
        print("artifacts: 1")
    else:
        results = {"scene": scene_results, "nworld": NWORLD,
                   "policy": {**task.record(), "inference": policy.record()}, "control": control,
                   "pen": pen_spin.pen_record()}
        record = run.write(asset=asset.to_record(), load_report=loaded.report, results=results, rendering=RENDERING)
        print(f"record: {record.resolve()}")
        print("artifacts: 0")

        peak = {}

        def frame(t: float) -> None:
            advance(round(t / pen_spin.CONTROL_DT), peak)
            platform.host_data(loaded, WORLD)

        with scene.Display(loaded.model, loaded.data, view, headless=False) as display:
            scene.run_live(display, pen_spin.CONTROL_DT, frame)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except (
        assets.AssetError,
        pen_spin.PenSpinError,
        profiles.ProfileError,
        mit.MitStabilityError,
        records.RecordError,
        conventions.ViewerError,
        FileExistsError,
    ) as exc:
        sys.exit(f"error: {exc}")
