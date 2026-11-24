# SPDX-License-Identifier: Apache-2.0
import argparse
import math
import sys

import mujoco_warp as mjw
import numpy as np

from wuji_sim import assets, conventions, framing, mit, profiles, records, scene
from wuji_sim.platforms import mujoco_warp as platform

DEMO = "04_sweep_joints"
NWORLD = 1
WORLD = 0
CAMERA = scene.PREVIEW_CAMERA
VIDEO_FPS = 25
RENDERING = "Warp physics / MuJoCo rendering"
CONTROL_LAW = "tau = clip(kp*(q_des-q) + kd*(qd_des-qd) + tau_ff, -tau_limit, tau_limit)"
TAU_APPLIED = (
    "d.qfrc_actuator of world 0 at the joint dof after mjw.step: the actuator torque that step applied after the "
    "general-actuator forcerange and joint actuatorfrcrange clamps; frictionloss, joint-limit and contact forces "
    "are not included"
)


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
        description="Sweep each joint between its inner limits with MIT torque control on MuJoCo Warp, one joint "
        "at a time; records a video of world 0."
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
    view = framing.view(asset.mjcf, args.hand, platform.PLATFORM, "sweep")
    loaded = platform.load(asset.mjcf, nworld=NWORLD, scene=lambda spec: scene.add_scene(spec, args.hand, view),
                           description_version=asset.expected_version)
    joint_names = loaded.joint_names
    description = scene.description_joint_ranges(asset.mjcf)
    # The device model's ranges (float32, world 0) are the limits the Warp solver enforces.
    effective = {name: loaded.report["effective_values"]["joints"][name]["range"] for name in joint_names}
    scene.check_joint_table(args.hand, description, effective)
    print(f"joint table: {len(joint_names)} joints; names and device ranges match {asset.mjcf}")
    ranges = np.array([effective[name] for name in joint_names])

    timestep = float(loaded.model.opt.timestep)
    frame_steps = steps_per_frame(timestep)
    n = len(joint_names)
    steps = scene.whole_steps(scene.sweep_duration(joint_names), timestep)
    gains = {"kp": np.full(n, scene.DEMO_KP), "kd": np.full(n, scene.DEMO_KD), "tau_ff": np.zeros(n)}
    stability = platform.mit_stability(loaded, gains["kp"], gains["kd"])
    platform.set_mit_gains(loaded, gains["kp"], gains["kd"])
    print(f"MIT stability ({stability['control_path']}): max ratio {stability['max_ratio']:.3f} "
          f"at {stability['max_joint']} (dt {timestep:g} s, limit {stability['limit']:g})")
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
        "joint_order": list(joint_names),
        "actuator_ids": act_ids,
        "tau_limit": list(loaded.tau_limit),
        "stability": stability,
    }
    joint_table_check = {
        "description_mjcf": str(asset.mjcf),
        "joints": n,
        "compared": "hinge joint count, names and order; description MJCF range vs MuJoCo Warp device range (world 0)",
        "range_atol_rad": scene.JOINT_RANGE_ATOL,
        "passed": True,
    }

    if args.headless:
        t = np.arange(steps) * timestep
        active = np.empty(steps, dtype=np.int64)
        series = {field: np.empty((steps, n)) for field in scene.SWEEP_SERIES}
        peak = {}
        label = f"{CAMERA}_world{WORLD}"
        with (
            scene.SceneRenderer(loaded.model, *view.render_size) as renderer,
            scene.Display(loaded.model, loaded.data, view, headless=True) as display,
            run.open_video(label, VIDEO_FPS) as video,
        ):
            video.append(renderer.render(platform.host_data(loaded, WORLD), CAMERA))
            for k in range(steps):
                q_des, qd_des, active[k] = scene.sweep_reference(joint_names, ranges, t[k])
                cmd = mit.MitCommand(joint_names, q_des, qd_des, **gains)
                upload(buffers, cmd, NWORLD)
                platform.apply_mit(loaded, buffers)
                series["q_des"][k] = q_des
                series["q"][k] = loaded.d.qpos.numpy()[WORLD, qpos_adr]
                series["qd"][k] = loaded.d.qvel.numpy()[WORLD, dof_adr]
                series["tau_command"][k] = mit.mit_torque(
                    cmd, loaded.d.qpos.numpy()[WORLD, qpos_adr], loaded.d.qvel.numpy()[WORLD, dof_adr], tau_limit
                )
                mjw.step(loaded.m, loaded.d)
                series["tau_applied"][k] = loaded.d.qfrc_actuator.numpy()[WORLD, dof_adr]
                platform.check_capacity(loaded, peak)
                if (k + 1) % frame_steps == 0:
                    video.append(renderer.render(platform.host_data(loaded, WORLD), CAMERA))
                if display.is_running():
                    platform.host_data(loaded, WORLD)
                    display.sync()

        error = series["q_des"] - series["q"]
        saturated = np.abs(series["tau_command"]) >= tau_limit
        rows = scene.sweep_joint_table(joint_names, description, effective, active, series["q"], error, saturated)
        flagged = scene.sweep_flags(rows)
        csv_path = run.artifact_path("timeseries", ".csv")
        scene.write_sweep_csv(csv_path, joint_names, t, active, series)
        title = (f"{asset.product} {args.hand} {DEMO} ({RENDERING}, world {WORLD} of {NWORLD}; "
                 f"kp {scene.DEMO_KP}, kd {scene.DEMO_KD})")
        figure_path = run.save_figure("joint_curves", scene.sweep_figure(rows, t, series, title))

        results = {
            "scene": scene.scene_record(args.hand, view),
            "nworld": NWORLD,
            "control": control,
            "joint_table_check": joint_table_check,
            "sweep": {
                "reference": scene.sweep_reference_record(joint_names),
                "duration_s": steps * timestep,
                "steps": steps,
                "error_definition": f"q_des(t) - q(t) of world {WORLD}, sampled before each physics step",
                "saturation_definition": "|tau_command| >= float32 tau_limit",
                "active_window": "the sweep window of the joint's group (see reference.windows); run = the whole "
                                 "sweep, including holding open while other windows move",
                "error_flag_rad": scene.SWEEP_ERROR_FLAG_RAD,
                "flagged_joints": flagged,
                "joints": rows,
            },
            "proprioception": {
                "csv": str(csv_path),
                "columns": ["t_s", "active_window", *(f"{field}:<joint>" for field in scene.SWEEP_SERIES)],
                "units": {"t_s": "s", **scene.SWEEP_SERIES_UNITS},
                "sample_period_s": timestep,
                "q_qd": f"d.qpos / d.qvel of world {WORLD} at the joint, sampled before each physics step",
                "tau_command": "clipped MIT torque from mit.mit_torque on the host, using the pre-step q and qd of "
                               f"world {WORLD}; d.ctrl holds the affine actuator input kp*q_des + kd*qd_des + tau_ff",
                "tau_applied": TAU_APPLIED,
            },
            "figure": str(figure_path),
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
        for row in rows:
            print(f"{row['joint']}: target {row['sweep_target_rad'][0]:+.3f}..{row['sweep_target_rad'][1]:+.3f} "
                  f"reached {row['reached_min_rad']:+.3f}..{row['reached_max_rad']:+.3f} rad, "
                  f"max |error| {row['max_abs_error_run_rad']:.4f} rad, "
                  f"saturation {row['saturation_fraction_run']:.4f}")
        print(f"flagged (saturation > 0 or max |error| > {scene.SWEEP_ERROR_FLAG_RAD} rad): {flagged}")
        print(f"{steps} steps, {video.frames} video frames, capacity usage {peak}")
        print(f"record: {record.resolve()}")
    else:
        results = {"scene": scene.scene_record(args.hand, view), "nworld": NWORLD, "control": control,
                  "joint_table_check": joint_table_check}
        record = run.write(asset=asset.to_record(), load_report=loaded.report, results=results, rendering=RENDERING)
        print(f"record: {record.resolve()}")
        print("artifacts: 0")

        peak = {}

        def frame(t: float) -> None:
            q_des, qd_des, _ = scene.sweep_reference(joint_names, ranges, t)
            cmd = mit.MitCommand(joint_names, q_des, qd_des, **gains)
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
    except (assets.AssetError, profiles.ProfileError, mit.MitStabilityError, records.RecordError,
            scene.JointTableError, conventions.ViewerError, FileExistsError) as exc:
        sys.exit(f"error: {exc}")
