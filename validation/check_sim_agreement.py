# SPDX-License-Identifier: Apache-2.0
import argparse
import io
import sys
from collections.abc import Sequence
from dataclasses import dataclass

import mujoco
import numpy as np

from validation import common
from wuji_sim import assets, framing, mit, profiles, scene, scene_isaaclab
from wuji_sim.platforms import isaaclab as isaac_platform
from wuji_sim.platforms import mujoco as mujoco_platform

CHECK = "sim_agreement"
MOTIONS = ("03_mit_tracking", "04_sweep_joints")
MOTION_VIEWS = {"03_mit_tracking": "tracking", "04_sweep_joints": "sweep"}
REPEATED_MOTION = MOTIONS[0]
ISAAC_VIEW_KIND = "tracking"
SERIES = ("q_des", "qd_des", "q")
CONTROL_LAW = "tau = clip(kp*(q_des-q) + kd*(qd_des-qd) + tau_ff, -tau_limit, tau_limit)"
EXPECTED_ERRORS = (*common.EXPECTED_ERRORS, scene.JointTableError, profiles.ProfileError, mit.MitStabilityError)
SIMULATION = {
    "mujoco": "a fresh load per motion with that motion's example scene; data.ctrl = kp*q_des + kd*qd_des at the "
              "actuators, then mujoco.mj_step",
    "mujoco_warp": "a fresh load per motion with that motion's example scene and one world; the command is uploaded "
                   "in float32 and applied by wuji_sim.platforms.mujoco_warp.apply_mit, then mujoco_warp.step",
    "isaaclab": "one load with the tracking scene for every motion; before each motion set_joint_state(0, 0); "
                "each step set_command(q_des, qd_des, 0) on the PhysX implicit joint drive, then step; "
                f"{REPEATED_MOTION} runs once more after the other motions and is compared with its first run",
}


@dataclass
class Run:
    args: argparse.Namespace
    asset: assets.ResolvedAsset
    record: common.ValidationRecord
    thresholds: dict
    gate: float | None
    reference: dict | None
    ref: dict | None


def agreement_gate(thresholds: dict, simulator: str) -> float | None:
    table = common.lookup(thresholds, CHECK, simulator)
    if not isinstance(table, dict):
        raise common.ValidationError(f"thresholds {thresholds['file']}: {CHECK}.{simulator} must be a table")
    if table.get("report_only") is True:
        return None
    return common.criterion(thresholds, CHECK, simulator, "max_joint_rms_error", unit="rad")


def commands(motion: str, joint_names: Sequence[str], dt: float, ranges) -> tuple[np.ndarray, np.ndarray]:
    if motion == "03_mit_tracking":
        steps = round(scene.tracking_duration() / dt)
        pairs = [scene.tracking_reference(joint_names, k * dt) for k in range(steps)]
    elif motion == "04_sweep_joints":
        steps = scene.whole_steps(scene.sweep_duration(joint_names), dt)
        pairs = [scene.sweep_reference(joint_names, ranges, k * dt)[:2] for k in range(steps)]
    else:
        raise common.ValidationError(f"motion must be one of {MOTIONS}, got {motion!r}")
    return np.array([q_des for q_des, _ in pairs]), np.array([qd_des for _, qd_des in pairs])


def motion_errors(joint_names: Sequence[str], q_ref: np.ndarray, q_test: np.ndarray) -> dict:
    errors = common.joint_errors(q_ref, q_test)
    rms_joint, max_joint = int(np.argmax(errors["rms"])), int(np.argmax(errors["max_abs"]))
    return {
        "steps": len(q_ref),
        "worst_joint_rms_rad": float(errors["rms"][rms_joint]),
        "worst_joint_rms_joint": joint_names[rms_joint],
        "worst_joint_max_abs_rad": float(errors["max_abs"][max_joint]),
        "worst_joint_max_abs_joint": joint_names[max_joint],
        "joints": {name: {"rms_rad": float(errors["rms"][index]), "max_abs_rad": float(errors["max_abs"][index])}
                   for index, name in enumerate(joint_names)},
    }


def read_reference_data(data: bytes, path) -> dict:
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
            motions = tuple(str(motion) for motion in arrays["motions"])
            ref = {"joint_names": [str(name) for name in arrays["joint_names"]], "dt": float(arrays["dt"]),
                   "kp": float(arrays["kp"]), "kd": float(arrays["kd"])}
            for motion in MOTIONS:
                ref[motion] = {name: np.asarray(arrays[f"{motion}.{name}"], dtype=np.float64) for name in SERIES}
    except (OSError, KeyError, ValueError) as exc:
        raise common.ValidationError(f"cannot read reference data {path}: {exc}") from exc
    if motions != MOTIONS:
        raise common.ValidationError(f"reference data {path} holds the motions {motions}, expected {MOTIONS}")
    for motion in MOTIONS:
        shapes = {name: ref[motion][name].shape for name in SERIES}
        if len(set(shapes.values())) != 1 or shapes["q"][1:] != (len(ref["joint_names"]),):
            raise common.ValidationError(f"reference data {path}: {motion} series shapes {shapes} do not match "
                                         f"{len(ref['joint_names'])} joints")
    return ref


def check_reference_match(joint_names: Sequence[str], dt: float, ref: dict) -> None:
    if list(joint_names) != ref["joint_names"]:
        raise common.ValidationError(f"joint order {list(joint_names)} differs from the reference {ref['joint_names']}")
    if dt != ref["dt"]:
        raise common.ValidationError(f"timestep {dt} s differs from the reference timestep {ref['dt']} s")


def protocol(run: Run, joint_names: Sequence[str], dt: float, kp: float, kd: float, steps: dict) -> dict:
    compared = run.args.simulator != common.REFERENCE_SIMULATOR
    isaac = run.args.simulator == "isaaclab"
    return {
        "motions": {motion: {"steps": steps[motion], "duration_s": steps[motion] * dt,
                             "view": ISAAC_VIEW_KIND if isaac else MOTION_VIEWS[motion]}
                    for motion in MOTIONS},
        "commands": ("q_des and qd_des of the MuJoCo run, read from the reference data file" if compared else
                     "wuji_sim.scene.tracking_reference for 03_mit_tracking; wuji_sim.scene.sweep_reference over "
                     "the compiled joint ranges for 04_sweep_joints"),
        "control_law": CONTROL_LAW,
        "kp": kp,
        "kd": kd,
        "tau_ff": 0.0,
        "units": {"kp": "N*m/rad", "kd": "N*m*s/rad", "tau_ff": "N*m", "dt": "s", "q": "rad"},
        "dt": dt,
        "joint_order": list(joint_names),
        "sampling": "q of every joint before each physics step; q[0] is the initial state with every joint at 0",
        "error": "q_simulator - q_mujoco per joint and step; RMS and max |error| over the steps of each motion",
        "simulation": SIMULATION[run.args.simulator],
        "reference_simulation": SIMULATION[common.REFERENCE_SIMULATOR],
    }


def mujoco_motion(run: Run, motion: str) -> dict:
    hand = run.args.hand
    view = framing.view(run.asset.mjcf, hand, mujoco_platform.PLATFORM, MOTION_VIEWS[motion])
    loaded = mujoco_platform.load(run.asset.mjcf, scene=lambda spec: scene.add_scene(spec, hand, view),
                                  description_version=run.asset.expected_version)
    model, data = loaded.model, loaded.data
    joint_names = list(loaded.joint_names)
    dt = float(model.opt.timestep)
    ranges = None
    if motion == "04_sweep_joints":
        effective = {name: loaded.report["effective_values"]["joints"][name]["range"] for name in joint_names}
        scene.check_joint_table(hand, scene.description_joint_ranges(run.asset.mjcf), effective)
        ranges = np.array([effective[name] for name in joint_names])
    q_des, qd_des = commands(motion, joint_names, dt, ranges)
    kp = np.full(len(joint_names), scene.DEMO_KP)
    kd = np.full(len(joint_names), scene.DEMO_KD)
    stability = mujoco_platform.mit_stability(loaded, kp, kd)
    mujoco_platform.set_mit_gains(loaded, kp, kd)
    qpos_adr, actuator_ids = list(loaded.qpos_adr), list(loaded.actuator_ids)
    q = np.empty_like(q_des)
    for k in range(len(q_des)):
        q[k] = data.qpos[qpos_adr]
        data.ctrl[actuator_ids] = kp * q_des[k] + kd * qd_des[k]
        mujoco.mj_step(model, data)
    return {"joint_names": joint_names, "dt": dt, "q_des": q_des, "qd_des": qd_des, "q": q,
            "stability_max_ratio": stability["max_ratio"], "scene": view.record()}


def run_mujoco(run: Run) -> int:
    results = {motion: mujoco_motion(run, motion) for motion in MOTIONS}
    joint_names, dt = results[MOTIONS[0]]["joint_names"], results[MOTIONS[0]]["dt"]
    if any(result["joint_names"] != joint_names or result["dt"] != dt for result in results.values()):
        raise common.ValidationError("the motion scenes compiled to different joint orders or timesteps")
    arrays = {"joint_names": np.array(joint_names), "dt": np.float64(dt), "kp": np.float64(scene.DEMO_KP),
              "kd": np.float64(scene.DEMO_KD), "motions": np.array(MOTIONS)}
    for motion, result in results.items():
        arrays.update({f"{motion}.{name}": result[name] for name in SERIES})
    data_path = run.record.artifact_path("reference", ".npz", data=True)
    with open(data_path, "xb") as f:
        np.savez(f, **arrays)
    steps = {motion: len(result["q"]) for motion, result in results.items()}
    values = {"motions": {motion: {"steps": steps[motion], "stability_max_ratio": result["stability_max_ratio"],
                                   "scene": result["scene"]}
                          for motion, result in results.items()}}
    verdicts = {f"{motion}.max_joint_rms_error": {"value": None, "threshold": None, "unit": "rad",
                                                   "verdict": "reference"}
                for motion in MOTIONS}
    path = run.record.write(asset=run.asset, thresholds=run.thresholds, reference=None,
                            protocol=protocol(run, joint_names, dt, scene.DEMO_KP, scene.DEMO_KD, steps),
                            values=values, verdicts=verdicts)
    print(f"reference: {', '.join(f'{motion} {count} steps' for motion, count in steps.items())} in {data_path}")
    return common.print_result(path, verdicts)


def error_figure(ref: dict, runs: dict, title: str):
    from matplotlib import colormaps
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    joint_names = ref["joint_names"]
    colors = colormaps["tab20"]
    fig = Figure(figsize=(13.0, 4.0 * len(MOTIONS)), dpi=100, layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(len(MOTIONS), 1, squeeze=False)[:, 0]
    for ax, motion in zip(axes, MOTIONS):
        error = np.abs(runs[motion] - ref[motion]["q"])
        t = np.arange(len(error)) * ref["dt"]
        for index, name in enumerate(joint_names):
            ax.plot(t, error[:, index], color=colors(index), linewidth=0.8, label=name)
        ax.set_yscale("log")
        ax.set_title(motion)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("|q - q_mujoco| [rad]")
        ax.grid(True, alpha=0.3)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside right center", fontsize=7)
    fig.suptitle(title)
    return fig


def finish(run: Run, runs: dict, extra: dict) -> int:
    ref = run.ref
    joint_names = ref["joint_names"]
    motions = {motion: motion_errors(joint_names, ref[motion]["q"], runs[motion]) for motion in MOTIONS}
    values = {"motions": motions, **extra}
    verdicts = {}
    for motion in MOTIONS:
        value = motions[motion]["worst_joint_rms_rad"]
        verdicts[f"{motion}.max_joint_rms_error"] = {
            "value": value, "threshold": run.gate, "unit": "rad",
            "verdict": "report_only" if run.gate is None else common.judge(value, run.gate),
        }
    if "reset_repeat_max_abs_rad" in extra:
        verdicts["reset_repeat_max_abs_error"] = {"value": extra["reset_repeat_max_abs_rad"], "threshold": None,
                                                  "unit": "rad", "verdict": "report_only"}
    figure_path = run.record.artifact_path("errors", ".png")
    title = f"{run.asset.product} {run.args.hand} joint trajectories: {run.args.simulator} vs mujoco"
    error_figure(ref, runs, title).savefig(figure_path, format="png")
    steps = {motion: len(ref[motion]["q"]) for motion in MOTIONS}
    path = run.record.write(asset=run.asset, thresholds=run.thresholds, reference=run.reference,
                            protocol=protocol(run, joint_names, ref["dt"], ref["kp"], ref["kd"], steps),
                            values=values, verdicts=verdicts)
    for motion, entry in motions.items():
        print(f"{motion}: worst joint RMS {entry['worst_joint_rms_rad']:.3g} rad ({entry['worst_joint_rms_joint']}), "
              f"worst joint max |error| {entry['worst_joint_max_abs_rad']:.3g} rad "
              f"({entry['worst_joint_max_abs_joint']})")
    if "reset_repeat_max_abs_rad" in extra:
        print(f"{REPEATED_MOTION} run again after reset: max |q - q_first| {extra['reset_repeat_max_abs_rad']:.3g} rad "
              f"({extra['reset_repeat_joint']})")
    return common.print_result(path, verdicts)


def run_warp(run: Run) -> int:
    import mujoco_warp as mjw
    from wuji_sim.platforms import mujoco_warp as platform

    ref = run.ref
    hand = run.args.hand
    runs, stability, capacity = {}, {}, {}
    for motion in MOTIONS:
        view = framing.view(run.asset.mjcf, hand, platform.PLATFORM, MOTION_VIEWS[motion])
        loaded = platform.load(run.asset.mjcf, nworld=1, scene=lambda spec: scene.add_scene(spec, hand, view),
                               description_version=run.asset.expected_version)
        check_reference_match(loaded.joint_names, float(loaded.model.opt.timestep), ref)
        n = len(loaded.joint_names)
        gains = {"kp": np.full(n, ref["kp"]), "kd": np.full(n, ref["kd"]), "tau_ff": np.zeros(n)}
        stability[motion] = platform.mit_stability(loaded, gains["kp"], gains["kd"])["max_ratio"]
        platform.set_mit_gains(loaded, gains["kp"], gains["kd"])
        buffers = platform.mit_buffers(loaded)
        qpos_adr = list(loaded.qpos_adr)
        q_des, qd_des = ref[motion]["q_des"], ref[motion]["qd_des"]
        q = np.empty_like(q_des)
        peak = {}
        for k in range(len(q_des)):
            command = {"q_des": q_des[k], "qd_des": qd_des[k], **gains}
            for name in platform.MIT_FIELDS:
                getattr(buffers, name).assign(np.ascontiguousarray(command[name][None, :], dtype=np.float32))
            platform.apply_mit(loaded, buffers)
            q[k] = loaded.d.qpos.numpy()[0, qpos_adr]
            mjw.step(loaded.m, loaded.d)
            platform.check_capacity(loaded, peak)
        runs[motion] = q
        capacity[motion] = peak
    return finish(run, runs, {"stability_max_ratio": stability, "capacity_usage": capacity})


def isaac_motion(loaded, motion_ref: dict, zeros: list[float]) -> np.ndarray:
    isaac_platform.set_joint_state(loaded, zeros, zeros)
    q_des, qd_des = motion_ref["q_des"], motion_ref["qd_des"]
    q = np.empty_like(q_des)
    for k in range(len(q_des)):
        q[k] = isaac_platform.joint_state(loaded)[0]
        isaac_platform.set_command(loaded, q_des[k], qd_des[k], zeros)
        isaac_platform.step(loaded)
    return q


def isaac_motions(run: Run) -> int:
    ref = run.ref
    view = framing.view(run.asset.mjcf, run.args.hand, isaac_platform.PLATFORM, ISAAC_VIEW_KIND)
    loaded = isaac_platform.load(run.asset.usd, hand=run.args.hand, floor_z=view.floor_z,
                                 description_version=run.asset.expected_version, scene=scene_isaaclab.Scene())
    check_reference_match(loaded.joint_names, float(loaded.dt), ref)
    n = len(loaded.joint_names)
    kp, kd = np.full(n, ref["kp"]), np.full(n, ref["kd"])
    stability = isaac_platform.mit_stability(loaded, kp, kd)
    isaac_platform.set_mit_gains(loaded, kp, kd)
    zeros = [0.0] * n
    runs = {motion: isaac_motion(loaded, ref[motion], zeros) for motion in MOTIONS}
    repeat = isaac_motion(loaded, ref[REPEATED_MOTION], zeros)
    difference = np.abs(repeat - runs[REPEATED_MOTION])
    if not np.all(np.isfinite(difference)):
        raise common.ValidationError(f"the repeated {REPEATED_MOTION} run returned non-finite joint positions")
    joint = int(np.argmax(difference.max(axis=0)))
    return finish(run, runs, {"stability_max_ratio": stability["max_ratio"], "scene": view.record(),
                              "reset_repeat_max_abs_rad": float(difference.max()),
                              "reset_repeat_joint": ref["joint_names"][joint]})


def run_isaac(run: Run) -> int:
    with isaac_platform.launch(headless=True):
        # launch_simulation turns any exception but SystemExit into exit 1 (a failed check), so always sys.exit here.
        try:
            code = isaac_motions(run)
        except Exception as exc:
            code = common.report_error(exc, EXPECTED_ERRORS)
        sys.exit(code)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Compare a simulator's joint trajectories with MuJoCo's under the MIT tracking and joint sweep "
                    "commands of examples 03 and 04."
    )
    common.add_cli_args(parser)
    args = common.parse_args(parser, argv[1:])

    thresholds = common.load_thresholds(args.thresholds)
    gate = None if args.simulator == common.REFERENCE_SIMULATOR else agreement_gate(thresholds, args.simulator)
    asset = assets.resolve(args.hand, args.asset_root, strict=args.strict)
    print(asset.describe())
    reference = ref = None
    if args.reference is not None:
        reference = common.load_reference(args.reference, check=CHECK, hand=args.hand, asset=asset)
        if reference["data_path"] is None:
            raise common.ValidationError(f"reference record {reference['path']} lists no data file")
        ref = read_reference_data(reference["data_bytes"], reference["data_path"])
    record = common.ValidationRecord(check=CHECK, simulator=args.simulator, hand=args.hand,
                                     output_dir=args.output_dir, argv=argv)
    run = Run(args=args, asset=asset, record=record, thresholds=thresholds, gate=gate, reference=reference, ref=ref)
    if args.simulator == "mujoco":
        return run_mujoco(run)
    if args.simulator == "mujoco_warp":
        return run_warp(run)
    return run_isaac(run)


if __name__ == "__main__":
    try:
        code = main(sys.argv)
    except Exception as exc:
        code = common.report_error(exc, EXPECTED_ERRORS)
    sys.exit(code)
