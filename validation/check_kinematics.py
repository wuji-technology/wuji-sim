# SPDX-License-Identifier: Apache-2.0
import argparse
import io
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from validation import common
from wuji_sim import assets, framing, profiles, scene, scene_isaaclab
from wuji_sim.platforms import isaaclab as isaac_platform
from wuji_sim.platforms import mujoco as mujoco_platform

CHECK = "kinematics"
VIEW_KIND = "tracking"
DEFAULT_SAMPLES = 200
DEFAULT_SEED = 7
SCENE_BODY_PREFIX = "scene_"
# The root link's rotation must match MuJoCo's first; a mismatch means misread (x, y, z, w) poses, not kinematics.
ROOT_ROTATION_TOLERANCE_RAD = 1e-5
REFERENCE_ARRAYS = ("joint_names", "config_labels", "q", "body_names", "root_body", "position", "rotation",
                    "body_has_joint")
EXPECTED_ERRORS = (*common.EXPECTED_ERRORS, profiles.ProfileError)
CONFIGURATIONS = (
    "the presets of wuji_sim.scene.POSES, then --samples joint vectors drawn uniformly within the description MJCF "
    "joint ranges by numpy.random.default_rng(--seed); every value is rounded to float32, so each simulator is "
    "given identical joint values"
)
FORWARD_KINEMATICS = {
    "mujoco": "data.qpos set at the joints, then mujoco.mj_kinematics; data.xpos and data.xmat (float64)",
    "mujoco_warp": "one world per configuration with Data.qpos assigned in float32, then mujoco_warp.kinematics; "
                   "Data.xpos and Data.xmat read back from the device (float32)",
    "isaaclab": "wuji_sim.platforms.isaaclab.set_joint_state (write_joint_state_to_sim with zero velocity, then "
                "sim.forward()), then articulation.data.body_link_pose_w (float32) with (x, y, z, w) quaternions",
}


@dataclass
class Run:
    args: argparse.Namespace
    asset: assets.ResolvedAsset
    record: common.ValidationRecord
    thresholds: dict
    position_limit: float
    angle_limit: float
    view: framing.View
    reference: dict | None
    ref: dict | None


def non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {value}")
    return value


def configurations(joint_names: Sequence[str], mjcf: Path, samples: int, seed: int) -> tuple[list[str], np.ndarray]:
    description = scene.description_joint_ranges(mjcf)
    ranges = np.array([description[name] for name in joint_names])
    presets = [scene.pose_vector(joint_names, pose) for pose in scene.POSES]
    drawn = np.random.default_rng(seed).uniform(ranges[:, 0], ranges[:, 1], size=(samples, len(joint_names)))
    q = np.vstack([*presets, drawn]).astype(np.float32).astype(np.float64)
    return [*scene.POSES, *(f"random_{index:03d}" for index in range(samples))], q


def hand_bodies(model: mujoco.MjModel) -> tuple[list[int], list[str], str]:
    ids = [body for body in range(1, model.nbody) if not model.body(body).name.startswith(SCENE_BODY_PREFIX)]
    names = [model.body(body).name for body in ids]
    if "" in names or len(set(names)) != len(names):
        raise common.ValidationError(f"hand bodies need unique, non-empty names, got {names}")
    roots = [model.body(body).name for body in ids if model.body_parentid[body] == 0]
    if len(roots) != 1:
        raise common.ValidationError(f"the hand needs exactly one body attached to the world, found {roots}")
    return ids, names, roots[0]


def read_reference_data(data: bytes, path: Path) -> dict:
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as arrays_file:
            arrays = {name: arrays_file[name] for name in REFERENCE_ARRAYS}
    except (OSError, KeyError, ValueError) as exc:
        raise common.ValidationError(f"cannot read reference data {path}: {exc}") from exc
    ref = {
        "joint_names": [str(name) for name in arrays["joint_names"]],
        "labels": [str(label) for label in arrays["config_labels"]],
        "q": np.asarray(arrays["q"], dtype=np.float64),
        "body_names": [str(name) for name in arrays["body_names"]],
        "root_body": str(arrays["root_body"]),
        "position": np.asarray(arrays["position"], dtype=np.float64),
        "rotation": np.asarray(arrays["rotation"], dtype=np.float64),
        "body_has_joint": np.asarray(arrays["body_has_joint"], dtype=bool),
    }
    count, links = len(ref["labels"]), len(ref["body_names"])
    shapes = {"q": (count, len(ref["joint_names"])), "position": (count, links, 3), "rotation": (count, links, 3, 3),
              "body_has_joint": (links,)}
    wrong = {key: ref[key].shape for key, shape in shapes.items() if ref[key].shape != shape}
    if wrong or ref["root_body"] not in ref["body_names"]:
        raise common.ValidationError(f"reference data {path} is inconsistent: wrong shapes {wrong}, root link "
                                     f"{ref['root_body']!r} of links {ref['body_names']}")
    return ref


def check_joint_order(joint_names: Sequence[str], ref: dict) -> None:
    if list(joint_names) != ref["joint_names"]:
        raise common.ValidationError(f"joint order {list(joint_names)} differs from the reference {ref['joint_names']}")


def compare(ref: dict, names: Sequence[str], position: np.ndarray, rotation: np.ndarray) -> dict:
    names = list(names)
    labels, ref_names, root = ref["labels"], ref["body_names"], ref["root_body"]
    if len(set(names)) != len(names):
        raise common.ValidationError(f"link names are not unique: {sorted(names)}")
    if root not in names:
        raise common.ValidationError(f"the reference root link {root!r} is not among the compared links {names}")
    position = np.asarray(position, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    if position.shape != (len(labels), len(names), 3) or rotation.shape != (*position.shape, 3):
        raise common.ValidationError(f"poses of {len(labels)} configurations x {len(names)} links expected, got "
                                     f"positions {position.shape} and rotations {rotation.shape}")
    if not (np.all(np.isfinite(position)) and np.all(np.isfinite(rotation))):
        raise common.ValidationError("the compared simulator returned non-finite link poses")
    shared = [name for name in ref_names if name in names]
    ref_position, ref_rotation = common.root_relative(ref["position"], ref["rotation"], ref_names.index(root))
    test_position, test_rotation = common.root_relative(position, rotation, names.index(root))
    ref_index = [ref_names.index(name) for name in shared]
    test_index = [names.index(name) for name in shared]
    position_error = np.linalg.norm(test_position[:, test_index] - ref_position[:, ref_index], axis=-1)
    angle_error = common.rotation_angle(ref_rotation[:, ref_index], test_rotation[:, test_index])

    def worst(errors: np.ndarray) -> tuple[float, dict]:
        config, link = np.unravel_index(np.argmax(errors), errors.shape)
        return float(errors[config, link]), {"configuration": labels[config], "link": shared[link]}

    max_position, position_at = worst(position_error)
    max_angle, angle_at = worst(angle_error)
    return {
        "compared_links": shared,
        "compared_link_count": len(shared),
        "mujoco_only_links": [name for name in ref_names if name not in names],
        "other_only_links": [name for name in names if name not in ref_names],
        "missing_jointed_links": [name for name, jointed in zip(ref_names, ref["body_has_joint"])
                                  if jointed and name not in names],
        "max_position_error_m": max_position,
        "max_position_error_at": position_at,
        "max_angle_error_rad": max_angle,
        "max_angle_error_at": angle_at,
        "rms_position_error_m": float(np.sqrt(np.mean(position_error**2))),
        "links": {name: {"max_position_error_m": float(position_error[:, index].max()),
                         "max_angle_error_rad": float(angle_error[:, index].max())}
                  for index, name in enumerate(shared)},
    }


def error_figure(result: dict, position_limit: float, angle_limit: float, title: str):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    links = result["compared_links"]
    x = np.arange(len(links))
    fig = Figure(figsize=(max(10.0, 0.25 * len(links)), 8.0), dpi=100, layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 1, sharex=True)
    for ax, key, limit, label in ((axes[0], "max_position_error_m", position_limit, "max position error [m]"),
                                  (axes[1], "max_angle_error_rad", angle_limit, "max angle error [rad]")):
        ax.bar(x, [result["links"][name][key] for name in links], color="C0", label="max over configurations")
        # A log axis cannot show a zero threshold; zero errors (the root link) draw no bar either.
        if limit > 0.0:
            ax.axhline(limit, color="C3", linestyle="--", label=f"threshold {limit:g}")
        ax.set_yscale("log")
        ax.set_ylabel(label)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="upper right")
    axes[1].set_xticks(x, links, rotation=90, fontsize=7)
    fig.suptitle(title)
    return fig


def verdict_item(value, threshold: float, unit: str) -> dict:
    return {"value": value, "threshold": threshold, "unit": unit, "verdict": common.judge(value, threshold)}


def criteria(run: Run) -> tuple[tuple[str, float, str], ...]:
    return (("max_position_error", run.position_limit, "m"), ("max_angle_error", run.angle_limit, "rad"),
            ("missing_jointed_links", 0, "links"))


def protocol(run: Run, joint_names: Sequence[str]) -> dict:
    return {
        "configurations": CONFIGURATIONS,
        "joint_order": list(joint_names),
        "compared": "pose of every link both simulators have, relative to the root link: link-origin position and "
                    "orientation in the root link frame, so where each simulator places the hand root does not count",
        "position_error": "|p_simulator - p_mujoco| per link and configuration, in m",
        "angle_error": "rotation angle of R_mujoco^T R_simulator per link and configuration, in rad",
        "forward_kinematics": FORWARD_KINEMATICS[run.args.simulator],
        "reference_forward_kinematics": FORWARD_KINEMATICS[common.REFERENCE_SIMULATOR],
        "scene": run.view.record(),
    }


def run_mujoco(run: Run) -> int:
    args = run.args
    samples = DEFAULT_SAMPLES if args.samples is None else args.samples
    seed = DEFAULT_SEED if args.seed is None else args.seed
    loaded = mujoco_platform.load(run.asset.mjcf, scene=lambda spec: scene.add_scene(spec, args.hand, run.view),
                                  description_version=run.asset.expected_version)
    model, data = loaded.model, loaded.data
    joint_names = list(loaded.joint_names)
    labels, q = configurations(joint_names, run.asset.mjcf, samples, seed)
    ids, names, root = hand_bodies(model)
    position = np.empty((len(q), len(ids), 3))
    rotation = np.empty((len(q), len(ids), 3, 3))
    qpos_adr = list(loaded.qpos_adr)
    for index, values in enumerate(q):
        data.qpos[qpos_adr] = values
        mujoco.mj_kinematics(model, data)
        position[index] = data.xpos[ids]
        rotation[index] = data.xmat[ids].reshape(-1, 3, 3)
    has_joint = model.body_jntnum[ids] > 0
    data_path = run.record.artifact_path("reference", ".npz", data=True)
    with open(data_path, "xb") as f:
        np.savez(f, joint_names=np.array(joint_names), config_labels=np.array(labels), q=q,
                 body_names=np.array(names), root_body=np.array(root), position=position, rotation=rotation,
                 body_has_joint=has_joint)
    values = {
        "configurations": len(labels),
        "random_samples": samples,
        "seed": seed,
        "root_link": root,
        "links": names,
        "link_count": len(names),
        "jointed_links": [name for name, jointed in zip(names, has_joint) if jointed],
    }
    verdicts = {name: {"value": None, "threshold": threshold, "unit": unit, "verdict": "reference"}
                for name, threshold, unit in criteria(run)}
    path = run.record.write(asset=run.asset, thresholds=run.thresholds, reference=None,
                            protocol=protocol(run, joint_names), values=values, verdicts=verdicts)
    print(f"reference: {len(labels)} configurations x {len(names)} links (root {root}) in {data_path}")
    return common.print_result(path, verdicts)


def finish(run: Run, names: Sequence[str], position: np.ndarray, rotation: np.ndarray, readback: np.ndarray,
           extra: dict) -> int:
    ref = run.ref
    result = compare(ref, names, position, rotation)
    readback = np.asarray(readback, dtype=np.float64)
    if readback.shape != ref["q"].shape or not np.all(np.isfinite(readback)):
        raise common.ValidationError(f"joint readback of shape {readback.shape} with finite values expected "
                                     f"{ref['q'].shape}")
    readback_error = np.abs(readback - ref["q"])
    config, joint = np.unravel_index(np.argmax(readback_error), readback_error.shape)
    values = {
        "configurations": len(ref["labels"]),
        "root_link": ref["root_body"],
        **result,
        "max_joint_readback_error_rad": float(readback_error[config, joint]),
        "max_joint_readback_error_at": {"configuration": ref["labels"][config], "joint": ref["joint_names"][joint]},
        **extra,
    }
    measured = {"max_position_error": result["max_position_error_m"], "max_angle_error": result["max_angle_error_rad"],
                "missing_jointed_links": len(result["missing_jointed_links"])}
    verdicts = {name: verdict_item(measured[name], threshold, unit) for name, threshold, unit in criteria(run)}
    figure_path = run.record.artifact_path("errors", ".png")
    title = (f"{run.asset.product} {run.args.hand} forward kinematics: {run.args.simulator} vs mujoco, "
             f"{values['configurations']} configurations")
    error_figure(result, run.position_limit, run.angle_limit, title).savefig(figure_path, format="png")
    path = run.record.write(asset=run.asset, thresholds=run.thresholds, reference=run.reference,
                            protocol=protocol(run, ref["joint_names"]), values=values, verdicts=verdicts)
    position_at, angle_at = result["max_position_error_at"], result["max_angle_error_at"]
    print(f"{run.args.simulator} vs mujoco: {result['compared_link_count']} links x {values['configurations']} "
          f"configurations; max position error {result['max_position_error_m']:.3g} m at {position_at['link']} "
          f"({position_at['configuration']}), max angle error {result['max_angle_error_rad']:.3g} rad at "
          f"{angle_at['link']} ({angle_at['configuration']}), max joint readback error "
          f"{values['max_joint_readback_error_rad']:.3g} rad")
    return common.print_result(path, verdicts)


def run_warp(run: Run) -> int:
    import mujoco_warp as mjw
    from wuji_sim.platforms import mujoco_warp as platform

    ref = run.ref
    count = len(ref["labels"])
    loaded = platform.load(run.asset.mjcf, nworld=count,
                           scene=lambda spec: scene.add_scene(spec, run.args.hand, run.view),
                           description_version=run.asset.expected_version)
    check_joint_order(loaded.joint_names, ref)
    qpos_adr = list(loaded.qpos_adr)
    qpos = loaded.d.qpos.numpy()
    qpos[:, qpos_adr] = ref["q"]
    loaded.d.qpos.assign(qpos)
    mjw.kinematics(loaded.m, loaded.d)
    ids, names, _ = hand_bodies(loaded.model)
    position = loaded.d.xpos.numpy()[:, ids].astype(np.float64)
    rotation = loaded.d.xmat.numpy()[:, ids].reshape(count, len(ids), 3, 3).astype(np.float64)
    readback = loaded.d.qpos.numpy()[:, qpos_adr]
    return finish(run, names, position, rotation, readback, {"nworld": count})


def root_rotation_check(ref: dict, names: Sequence[str], rotation: np.ndarray) -> dict:
    root = ref["root_body"]
    if root not in names:
        raise common.ValidationError(f"the reference root link {root!r} is not among the Isaac Lab links {list(names)}")
    angle = float(common.rotation_angle(ref["rotation"][0, ref["body_names"].index(root)],
                                        rotation[0, list(names).index(root)]))
    if not angle <= ROOT_ROTATION_TOLERANCE_RAD:
        raise common.ValidationError(
            f"root link {root}: the Isaac Lab world rotation, read with (x, y, z, w) quaternions, differs from the "
            f"MuJoCo one by {angle:.6g} rad in configuration {ref['labels'][0]} (tolerance "
            f"{ROOT_ROTATION_TOLERANCE_RAD:g} rad): the pose quaternion order or the hand root placement differs"
        )
    return {"link": root, "configuration": ref["labels"][0], "angle_rad": angle,
            "tolerance_rad": ROOT_ROTATION_TOLERANCE_RAD, "quaternion_order": "xyzw"}


def isaac_poses(run: Run) -> int:
    ref = run.ref
    loaded = isaac_platform.load(run.asset.usd, hand=run.args.hand, floor_z=run.view.floor_z,
                                 description_version=run.asset.expected_version, scene=scene_isaaclab.Scene())
    check_joint_order(loaded.joint_names, ref)
    zeros = [0.0] * len(loaded.joint_names)
    isaac_platform.set_mit_gains(loaded, zeros, zeros)
    names = [str(name) for name in loaded.articulation.body_names]
    count = len(ref["labels"])
    position = np.empty((count, len(names), 3))
    rotation = np.empty((count, len(names), 3, 3))
    readback = np.empty(ref["q"].shape)
    for index, q in enumerate(ref["q"]):
        isaac_platform.set_joint_state(loaded, q, zeros)
        readback[index] = isaac_platform.joint_state(loaded)[0]
        pose = loaded.articulation.data.body_link_pose_w.torch[0].detach().cpu().numpy().astype(np.float64)
        position[index] = pose[:, :3]
        rotation[index] = common.quat_xyzw_to_matrix(pose[:, 3:7])
    return finish(run, names, position, rotation, readback,
                  {"root_rotation_check": root_rotation_check(ref, names, rotation)})


def run_isaac(run: Run) -> int:
    with isaac_platform.launch(headless=True):
        # launch_simulation turns any exception but SystemExit into exit 1 (a failed check), so always sys.exit here.
        try:
            code = isaac_poses(run)
        except Exception as exc:
            code = common.report_error(exc, EXPECTED_ERRORS)
        sys.exit(code)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Compare a simulator's forward kinematics with MuJoCo's: the pose of every link relative to the "
                    "root link over fixed joint configurations."
    )
    common.add_cli_args(parser)
    parser.add_argument("--samples", type=non_negative_int, default=None,
                        help=f"random configurations after the presets (--simulator mujoco only; default "
                             f"{DEFAULT_SAMPLES})")
    parser.add_argument("--seed", type=non_negative_int, default=None,
                        help=f"seed of the random configurations (--simulator mujoco only; default {DEFAULT_SEED})")
    args = common.parse_args(parser, argv[1:])
    if args.simulator != common.REFERENCE_SIMULATOR and (args.samples is not None or args.seed is not None):
        parser.error(f"--samples and --seed only apply to --simulator {common.REFERENCE_SIMULATOR}; the other "
                     "simulators take the configurations from --reference")

    thresholds = common.load_thresholds(args.thresholds)
    position_limit = common.criterion(thresholds, CHECK, "max_position_error", unit="m")
    angle_limit = common.criterion(thresholds, CHECK, "max_angle_error", unit="rad")
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
    run = Run(args=args, asset=asset, record=record, thresholds=thresholds, position_limit=position_limit,
              angle_limit=angle_limit, view=framing.view(asset.mjcf, args.hand, args.simulator, VIEW_KIND),
              reference=reference, ref=ref)
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
