# SPDX-License-Identifier: Apache-2.0
import argparse
import math
import sys
from pathlib import Path

import numpy as np

from wuji_sim import assets, conventions, framing, mit, profiles, records, scene
from wuji_sim.platforms import motrixsim as platform

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "_shared"))

import pen_spin  # noqa: E402
import pen  # noqa: E402
import rendering  # noqa: E402

DEMO = "06_pen_spin"
RENDERING = "MotrixSim physics / MotrixSim rendering"
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

    def build(model_source) -> None:
        scene.add_scene(model_source, args.hand, view)
        pen.move_wrist_mount(model_source, args.hand, wrist)

    # The pen is authored natively on the imported msd world (isaaclab's USD authoring is the precedent); the
    # profile's contact parameters travel the XML route, so they are passed to the native geoms explicitly.
    contact = {name: entry["value"] for name, entry in platform.load_profile()["params"]["contact"].items()}

    def pen_world(world) -> None:
        pen.add_pen_world(world, pen_spin.to_world(reference.pen_pose(0), wrist), contact)

    loaded = platform.load(asset.mjcf, scene=build, description_version=asset.expected_version,
                           render_resolution=view.render_size, render_fovy=view.fovy_deg, world_scene=pen_world)
    joint_names = loaded.joint_names
    pen_spin.check_joint_order(joint_names, args.hand)
    model, scene_model, data = loaded.model, loaded.scene_model, loaded.scene_data
    timestep = float(model.opt.timestep)
    substeps = scene.whole_steps(pen_spin.CONTROL_DT, timestep)
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

    # DoF layout: the 20 hand hinges take dof_pos 0..19 and dof_vel 0..19; the pen freejoint is the model's
    # floating base and adds 7 dof_pos ([x, y, z, qx, qy, qz, qw]; mujoco's qpos order is [x, y, z, qw, ...])
    # and 6 dof_vel at the end.
    joints = dict(zip(scene_model.joint_names, scene_model.joints))
    hand_pos_adr = [int(joints[name].dof_pos_index) for name in joint_names]
    hand_vel_adr = [int(joints[name].dof_vel_index) for name in joint_names]
    n_dof, n_vel = int(scene_model.num_dof_pos), int(scene_model.num_dof_vel)
    pen_adr = n_dof - 7
    if n_dof != len(joint_names) + 7 or n_vel != len(joint_names) + 6 or hand_pos_adr != list(range(
            len(joint_names))):
        raise pen_spin.PenSpinError(
            f"the model must have {len(joint_names)} hinge DoFs plus the free-floating pen, "
            f"got dof_pos {n_dof} / dof_vel {n_vel} with hinge pos indices {hand_pos_adr}"
        )

    def pen_pose_w() -> np.ndarray:
        return platform.link_pose(loaded, pen_spin.PEN_BODY)

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
        "actuator_ids": list(loaded.actuator_ids),
        "tau_limit": list(loaded.tau_limit),
        "stability": stability,
    }

    def reset(step: int) -> None:
        nonlocal last_action
        dof_pos = np.zeros(n_dof)
        dof_pos[hand_pos_adr] = reference.joint_pos(step)
        pose_w = pen_spin.to_world(reference.pen_pose(step), wrist)
        dof_pos[pen_adr:pen_adr + 7] = [*pose_w[:3], *pose_w[4:7], pose_w[3]]
        data.reset(scene_model, dof_pos=list(dof_pos), dof_vel=[0.0] * n_vel, forward_kinematic=True)
        residual.reset()
        last_action = np.zeros(pen_spin.ACTION_DIM, dtype=np.float32)

    def advance(step: int) -> None:
        nonlocal last_action
        if step > 0 and reference.restart(step):
            reset(step)
        pose_w = pen_pose_w()
        obs = pen_spin.observation(reference.command(step), np.asarray(data.dof_pos)[hand_pos_adr],
                                   np.asarray(data.dof_vel)[hand_vel_adr], pen_spin.to_wrist_local(pose_w, wrist),
                                   last_action)
        last_action = policy(obs)
        q_des = residual.step(last_action, reference.joint_pos(step))
        platform.apply_mit(loaded, q_des, qd_des, tau_ff)
        for _ in range(substeps):
            platform.step(loaded)
        return pose_w

    reset(0)
    scene_results = pen_spin.scene_record(args.hand, view, wrist)
    if args.headless:
        frame_steps = steps_per_frame(pen_spin.CONTROL_DT)
        log = pen_spin.PenLog(reference, wrist)
        log.start(pen_pose_w())
        with (
            rendering.OffscreenRenderer(loaded, view, args.hand, view.fovy_deg) as renderer,
            run.open_video(VIDEO_CAMERA, VIDEO_FPS) as video,
        ):
            video.append(renderer.render(VIDEO_CAMERA))
            for step in range(reference.num_steps):
                pose_w = advance(step)
                log.update(step, pose_w)
                if (step + 1) % frame_steps == 0:
                    video.append(renderer.render(VIDEO_CAMERA))
            frames = video.frames
            video_path = video.path

        replay = {"steps": reference.num_steps, "duration_s": reference.num_steps * pen_spin.CONTROL_DT, **log.record()}
        results = {
            "scene": scene_results,
            "policy": {**task.record(), "inference": policy.record()},
            "control": control,
            "pen": pen.pen_record(),
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
                   "control": control, "pen": pen.pen_record()}
        record_path = run.write(asset=asset.to_record(), load_report=loaded.report, results=results,
                                rendering=RENDERING)
        print(f"record: {record_path}")
        print("artifacts: 0")

        def frame(t: float) -> None:
            advance(round(t / pen_spin.CONTROL_DT))

        with rendering.LiveDisplay(loaded, view, view.fovy_deg) as display:
            rendering.run_live(display, pen_spin.CONTROL_DT, frame)


if __name__ == "__main__":
    try:
        main(sys.argv)
    except (assets.AssetError, pen_spin.PenSpinError, profiles.ProfileError, mit.MitStabilityError,
            records.RecordError, conventions.ViewerError, FileExistsError) as exc:
        sys.exit(f"error: {exc}")
