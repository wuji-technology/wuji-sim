# SPDX-License-Identifier: Apache-2.0
import argparse
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

from wuji_sim import assets, conventions, framing, mit, profiles, records, scene
from wuji_sim.platforms import mujoco as platform

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))

import pen_spin  # noqa: E402

DEMO = "06_pen_spin"
RENDERING = "MuJoCo physics / MuJoCo rendering"
VIDEO_FPS = 25
VIDEO_CAMERA = scene.PREVIEW_CAMERA
CONTROL_LAW = "tau = clip(kp*(q_des-q) + kd*(qd_des-qd) + tau_ff, -tau_limit, tau_limit)"


def steps_per_frame(period: float) -> int:
    ratio = 1.0 / (VIDEO_FPS * period)
    steps = round(ratio)
    if steps < 1 or not math.isclose(ratio, steps, rel_tol=1e-9):
        raise ValueError(f"video frame period 1/{VIDEO_FPS} s is not a whole number of {period} s control steps")
    return steps


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Replay a published pen-spinning policy on the right hand under MIT torque control."
    )
    assets.add_cli_args(parser)
    records.add_cli_args(parser)
    scene.add_cli_args(parser)
    pen_spin.add_cli_args(parser)
    args = parser.parse_args(argv[1:])
    if args.hand != pen_spin.HAND:
        parser.error(f"the published policy drives the {pen_spin.HAND} hand; rerun with --hand {pen_spin.HAND}")

    asset = assets.resolve(args.hand, args.asset_root, strict=args.strict)
    print(asset.describe())
    task = pen_spin.load(args.group, args.clip, args.cache_dir)
    print(task.describe())
    reference = task.reference
    policy = pen_spin.Policy(task.model)
    run = records.RunRecord(platform=platform.PLATFORM, demo=DEMO, hand=args.hand,
                            output_dir=args.output_dir, argv=argv)
    view = framing.view(asset.mjcf, args.hand, platform.PLATFORM, "pen_spin")
    wrist = pen_spin.wrist_pose(view.hand_roots[0])

    def build(model_source: mujoco.MjSpec) -> None:
        scene.add_scene(model_source, args.hand, view)
        pen_spin.add_pen_scene(model_source, args.hand, wrist, pen_spin.to_world(reference.pen_pose(0), wrist))

    loaded = platform.load(asset.mjcf, scene=build, description_version=asset.expected_version)
    model, data = loaded.model, loaded.data
    joint_names = loaded.joint_names
    pen_spin.check_joint_order(joint_names, args.hand)
    timestep = float(model.opt.timestep)
    substeps = scene.whole_steps(pen_spin.CONTROL_DT, timestep)
    qpos_adr, dof_adr, actuator_ids = list(loaded.qpos_adr), list(loaded.dof_adr), list(loaded.actuator_ids)
    pen_adr = pen_spin.pen_qpos_adr(model)
    kp, kd = pen_spin.gains()
    # The policy was trained on position targets alone.
    qd_des = np.zeros(len(joint_names))
    tau_ff = np.zeros(len(joint_names))
    stability = platform.mit_stability(loaded, kp, kd)
    platform.set_mit_gains(loaded, kp, kd)
    print(f"MIT stability ({stability['control_path']}): max ratio {stability['max_ratio']:.3f} "
          f"at {stability['max_joint']} (dt {timestep:g} s, limit {stability['limit']:g})")
    residual = pen_spin.ResidualAction(task.config)
    last_action = np.zeros(pen_spin.ACTION_DIM, dtype=np.float32)

    control = {
        "law": CONTROL_LAW,
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
        "actuator_ids": actuator_ids,
        "tau_limit": list(loaded.tau_limit),
        "stability": stability,
    }

    def reset(step: int) -> None:
        nonlocal last_action
        mujoco.mj_resetData(model, data)
        data.qpos[qpos_adr] = reference.joint_pos(step)
        data.qpos[pen_adr:pen_adr + 7] = pen_spin.to_world(reference.pen_pose(step), wrist)
        mujoco.mj_forward(model, data)
        residual.reset()
        last_action = np.zeros(pen_spin.ACTION_DIM, dtype=np.float32)

    def advance(step: int) -> None:
        nonlocal last_action
        if step > 0 and reference.restart(step):
            reset(step)
        pen_local = pen_spin.to_wrist_local(data.qpos[pen_adr:pen_adr + 7], wrist)
        obs = pen_spin.observation(reference.command(step), data.qpos[qpos_adr], data.qvel[dof_adr], pen_local,
                                   last_action)
        last_action = policy(obs)
        q_des = residual.step(last_action, reference.joint_pos(step))
        data.ctrl[actuator_ids] = kp * q_des + kd * qd_des + tau_ff
        for _ in range(substeps):
            mujoco.mj_step(model, data)

    reset(0)
    scene_results = pen_spin.scene_record(args.hand, view, wrist)
    if args.headless:
        frame_steps = steps_per_frame(pen_spin.CONTROL_DT)
        log = pen_spin.PenLog(reference, wrist)
        log.start(data.qpos[pen_adr:pen_adr + 7])
        with (
            scene.SceneRenderer(model, *view.render_size) as renderer,
            run.open_video(VIDEO_CAMERA, VIDEO_FPS) as video,
        ):
            video.append(renderer.render(data, VIDEO_CAMERA))
            for step in range(reference.num_steps):
                advance(step)
                log.update(step, data.qpos[pen_adr:pen_adr + 7])
                if (step + 1) % frame_steps == 0:
                    video.append(renderer.render(data, VIDEO_CAMERA))
            frames = video.frames
            video_path = video.path

        replay = {"steps": reference.num_steps, "duration_s": reference.num_steps * pen_spin.CONTROL_DT, **log.record()}
        results = {
            "scene": scene_results,
            "policy": {**task.record(), "inference": policy.record()},
            "control": control,
            "pen": pen_spin.pen_record(),
            "replay": replay,
            "video": {"path": str(video_path), "camera": VIDEO_CAMERA, "fps": VIDEO_FPS, "frames": frames,
                      "control_steps_per_frame": frame_steps},
        }
        record_path = run.write(asset=asset.to_record(), load_report=loaded.report, results=results,
                                rendering=RENDERING)
        print(f"replay: {pen_spin.describe_replay(replay)}")
        print(f"record: {record_path}")
        print("artifacts: 1")
    else:
        results = {"scene": scene_results, "policy": {**task.record(), "inference": policy.record()},
                   "control": control, "pen": pen_spin.pen_record()}
        record_path = run.write(asset=asset.to_record(), load_report=loaded.report, results=results,
                                rendering=RENDERING)
        print(f"record: {record_path}")
        print("artifacts: 0")

        def frame(t: float) -> None:
            advance(round(t / pen_spin.CONTROL_DT))

        with scene.Display(model, data, view, headless=False) as display:
            scene.run_live(display, pen_spin.CONTROL_DT, frame)


if __name__ == "__main__":
    try:
        main(sys.argv)
    except (assets.AssetError, pen_spin.PenSpinError, profiles.ProfileError, mit.MitStabilityError,
            records.RecordError, conventions.ViewerError, FileExistsError) as exc:
        sys.exit(f"error: {exc}")
