# SPDX-License-Identifier: Apache-2.0
import argparse
import math
import sys
from pathlib import Path

import motrixsim
import numpy as np
from motrixsim.render import RenderApp

from wuji_sim import assets, conventions, framing, mit, profiles, records, scene
from wuji_sim.platforms import motrixsim as platform

sys.path.insert(0, str(Path(__file__).resolve().parent / "_shared"))

import rendering  # noqa: E402

DEMO = "05_multi_world"
VIDEO_FPS = 25
VIDEO_CAMERA = scene.PREVIEW_CAMERA
RENDERING = "MotrixSim physics / MotrixSim rendering"
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


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Batched MotrixSim worlds cycle through the preset poses, each starting from a different "
                    "one; every world is stepped in one SceneData and rendered in one grid view."
    )
    assets.add_cli_args(parser)
    records.add_cli_args(parser)
    scene.add_cli_args(parser)
    parser.add_argument(
        "--nworld", type=int, default=4, help=f"number of batched worlds (>= 1); world i starts on "
                                              f"{scene.POSES}[i %% 4]"
    )
    args = parser.parse_args(argv[1:])
    if args.nworld < 1:
        parser.error(f"--nworld must be >= 1, got {args.nworld}")

    asset = assets.resolve(args.hand, args.asset_root, strict=args.strict)
    print(asset.describe())
    run = records.RunRecord(
        platform=platform.PLATFORM, demo=DEMO, hand=args.hand, output_dir=args.output_dir, argv=argv
    )
    nworld = args.nworld
    view = framing.view(asset.mjcf, args.hand, platform.PLATFORM, "grid", worlds=nworld)
    loaded = platform.load(asset.mjcf, scene=lambda spec: scene.add_scene(spec, args.hand, view),
                           description_version=asset.expected_version, render_resolution=view.render_size,
                           render_fovy=view.fovy_deg)
    # The whole demo runs on one batched SceneData; pointing loaded.scene_data at it routes the adapter's
    # set_mit_gains overrides into the batch the renderer syncs.
    data = motrixsim.SceneData(loaded.scene_model, batch=(nworld,))
    loaded.scene_data = data

    joint_names = loaded.joint_names
    timestep = float(loaded.model.opt.timestep)
    steps = scene.whole_steps(scene.parallel_duration(), timestep)
    segment_steps = scene.whole_steps(scene.MOVE_S + scene.HOLD_S, timestep)
    frame_steps = steps_per_frame(timestep)
    poses = [scene.POSES[w % len(scene.POSES)] for w in range(nworld)]
    n = len(joint_names)
    kp = np.full(n, scene.DEMO_KP)
    kd = np.full(n, scene.DEMO_KD)
    tau_ff = np.zeros(n)
    stability = platform.mit_stability(loaded, kp, kd)
    platform.set_mit_gains(loaded, kp, kd)
    print(
        f"MIT stability ({stability['control_path']}): max ratio {stability['max_ratio']:.3f} "
        f"at {stability['max_joint']} (dt {timestep:g} s, limit {stability['limit']:g})"
    )

    q0 = np.stack([scene.parallel_reference(joint_names, w, 0.0)[0] for w in range(nworld)])
    data.reset(loaded.scene_model, dof_pos=q0.astype(np.float32),
               dof_vel=np.zeros((nworld, n), dtype=np.float32), forward_kinematic=True)

    # add_scene put world 0 at view.hand_roots[0]; the other worlds are drawn shifted relative to it, as in
    # the mujoco_warp 05 demo (scene.env_grid_offsets layout, scene.PARALLEL_SPACING_M apart). The floor is a
    # true infinite plane on this platform (see scene.add_scene), which MotrixSim's renderer deduplicates
    # across batch instances, so no per-world plane overlap remains.
    offsets = (np.array(view.hand_roots) - view.hand_roots[0]).tolist()

    def apply_ctrl(t: float) -> None:
        references = [scene.parallel_reference(joint_names, w, t) for w in range(nworld)]
        for j, actuator in enumerate(loaded.actuators):
            actuator.set_ctrl(data, [float(kp[j] * q_des[j] + kd[j] * qd_des[j] + tau_ff[j])
                                     for q_des, qd_des in references])

    if args.headless:
        q_all = np.empty((steps + 1, nworld, n))
        q_all[0] = data.dof_pos
        with (
            RenderApp(headless=True) as app,
            run.open_video(VIDEO_CAMERA, VIDEO_FPS) as video,
        ):
            app.launch(loaded.scene_model, batch=nworld, render_offset=offsets)
            rendering.hide_fixture_groups(app)
            distance, azimuth, elevation = rendering.orbit(view.camera_position, view.camera_target)
            app.system_camera.set_view(lookat=[float(v) for v in view.camera_target], distance=distance,
                                       elevation=elevation, azimuth=azimuth)
            for _ in range(rendering.WARMUP_SYNCS):
                app.sync(data, wait=True)
            for _ in range(rendering.WARMUP_CAPTURES):
                rendering.capture_rgb(app, data)
            video.append(rendering.capture_rgb(app, data))
            for k in range(steps):
                apply_ctrl(k * timestep)
                motrixsim.step(loaded.scene_model, data)
                q_all[k + 1] = data.dof_pos
                if (k + 1) % frame_steps == 0:
                    video.append(rendering.capture_rgb(app, data))
            final_frame = rendering.capture_rgb(app, data)
            frames, video_path = video.frames, video.path

        final_path = run.write_png("worlds_final", final_frame)
        worlds = []
        for w, pose in enumerate(poses):
            goal = scene.pose_vector(joint_names, pose)
            hold_end = []
            for segment in range(len(scene.TRACKING_SEQUENCE)):
                step = (segment + 1) * segment_steps
                held = held_pose(w, segment)
                hold_end.append(
                    {
                        "t_s": step * timestep,
                        "pose": held,
                        "error_max_rad": float(np.abs(q_all[step, w]
                                                      - scene.pose_vector(joint_names, held)).max()),
                    }
                )
            worlds.append(
                {
                    "world": w,
                    "pose": pose,
                    "final_error_max_rad": float(np.abs(q_all[-1, w] - goal).max()),
                    "hold_end_error_max_rad": max(entry["error_max_rad"] for entry in hold_end),
                    "final_q_rad": q_all[-1, w].tolist(),
                    "hold_end": hold_end,
                }
            )
            print(
                f"world {w} ({pose}): final max |q - pose| {worlds[-1]['final_error_max_rad']:.4f} rad; hold-end "
                f"max |q - pose| {worlds[-1]['hold_end_error_max_rad']:.4f} rad"
            )
        results = {
            "scene": scene.scene_record(args.hand, view),
            "nworld": nworld,
            "reference": REFERENCE,
            "move_s": scene.MOVE_S,
            "hold_s": scene.HOLD_S,
            "steps": steps,
            "control": {
                "law": CONTROL_LAW,
                "backend": "MotrixSim general actuators with affine kp/kd bias, one batched SceneData of "
                           f"{nworld} worlds stepped in one motrixsim.step call; ctrl written per world through "
                           "actuator.set_ctrl",
                "kp": scene.DEMO_KP,
                "kd": scene.DEMO_KD,
                "tau_ff": 0.0,
                "units": {"kp": "N*m/rad", "kd": "N*m*s/rad", "tau_ff": "N*m", "tau_limit": "N*m", "period": "s"},
                "period_s": timestep,
                "updates_per_physics_step": 1,
                "joint_order": list(joint_names),
                "actuator_ids": list(loaded.actuator_ids),
                "tau_limit": list(loaded.tau_limit),
                "stability": stability,
            },
            "final_error": "max over the joints of |q(t_end) - pose|, rad; pose is the preset the world starts "
                           "and ends on",
            "hold_end_error": "max over the joints of |q(t) - pose| at the end of every hold, "
                              "t = (k + 1) * (move_s + hold_s), rad; pose is the preset held there",
            "images": f"one {VIDEO_CAMERA} camera frame of every world in the grid at t_end, plus the video",
            "worlds": worlds,
            "video": {
                "path": str(video_path),
                "camera": VIDEO_CAMERA,
                "fps": VIDEO_FPS,
                "frames": frames,
                "steps_per_frame": frame_steps,
                "worlds": nworld,
            },
        }
        record = run.write(asset=asset.to_record(), load_report=loaded.report, results=results, rendering=RENDERING)
        print(f"{steps} steps, {frames} preview frames of every world")
        print(f"record: {record.resolve()}")
        print("artifacts: 2")
    else:
        results = {"scene": scene.scene_record(args.hand, view), "nworld": nworld, "reference": REFERENCE}
        record = run.write(asset=asset.to_record(), load_report=loaded.report, results=results, rendering=RENDERING)
        print(f"record: {record.resolve()}")
        print("artifacts: 0")

        with rendering.LiveDisplay(loaded, view, view.fovy_deg, batch=nworld,
                                   render_offset=offsets) as display:
            # loaded.scene_data is the batched SceneData, so the display syncs the whole world set.
            def frame(t: float) -> None:
                apply_ctrl(t)
                motrixsim.step(loaded.scene_model, data)

            rendering.run_live(display, timestep, frame)


if __name__ == "__main__":
    try:
        main(sys.argv)
    except (
        assets.AssetError,
        profiles.ProfileError,
        mit.MitStabilityError,
        records.RecordError,
        conventions.ViewerError,
        FileExistsError,
    ) as exc:
        sys.exit(f"error: {exc}")
