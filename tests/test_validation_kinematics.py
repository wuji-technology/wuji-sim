# SPDX-License-Identifier: Apache-2.0
import datetime
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

from validation import check_kinematics, common
from wuji_sim import assets, conventions, framing, scene
from wuji_sim.platforms import mujoco as mujoco_platform

PUBLIC = Path(__file__).resolve().parents[1]
MODULE = "validation.check_kinematics"
SCRIPT = PUBLIC / "validation" / "check_kinematics.py"
HAND = "left"
SAMPLES = 6
CHECK_TIMEOUT_S = 300.0
# Kit startup dominates an Isaac Lab run.
ISAAC_TIMEOUT_S = 900.0
ZEROED = {
    'max_position_error = { value = 1e-6, unit = "m" }': 'max_position_error = { value = 0.0, unit = "m" }',
    'max_angle_error = { value = 1e-5, unit = "rad" }': 'max_angle_error = { value = 0.0, unit = "rad" }',
}
VERDICT_ITEMS = {"max_position_error", "max_angle_error", "missing_jointed_links"}
FLIPPED_JOINT = conventions.HAND_PREFIXES[HAND] + "index_finger_mcp_flex"


def run_check(*flags, timeout: float = CHECK_TIMEOUT_S) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", MODULE, *map(str, flags)], cwd=PUBLIC, capture_output=True, text=True,
                          timeout=timeout)


def base_flags(description_root: Path, output_dir: Path) -> list:
    return ["--hand", HAND, "--asset-root", description_root, "--strict", "--output-dir", output_dir]


def only_record(output_dir: Path) -> tuple[Path, dict]:
    [path] = Path(output_dir).glob("*.json")
    return path, json.loads(path.read_text(encoding="utf-8"))


def make_reference(description_root: Path, output_dir: Path) -> tuple[Path, dict]:
    result = run_check(*base_flags(description_root, output_dir), "--simulator", "mujoco", "--samples", SAMPLES)
    assert result.returncode == common.EXIT_PASS, result.stderr
    return only_record(output_dir)


def zero_thresholds(tmp_path: Path) -> Path:
    text = common.THRESHOLDS_PATH.read_text(encoding="utf-8")
    for old, new in ZEROED.items():
        assert text.count(old) == 1
        text = text.replace(old, new)
    path = tmp_path / "zero-thresholds.toml"
    path.write_text(text, encoding="utf-8")
    return path


def assert_compared_record(result: subprocess.CompletedProcess, path: Path, record: dict, simulator: str) -> None:
    assert result.returncode in (common.EXIT_PASS, common.EXIT_FAIL), result.stderr
    assert result.returncode == common.exit_code(record["verdict"])
    assert (record["check"], record["simulator"], record["hand"]) == ("kinematics", simulator, HAND)
    assert set(record["verdicts"]) == VERDICT_ITEMS
    values = record["values"]
    for key in ("max_position_error_m", "max_angle_error_rad", "rms_position_error_m", "max_joint_readback_error_rad"):
        assert math.isfinite(values[key]), key
    assert values["configurations"] == len(scene.POSES) + SAMPLES
    assert values["compared_link_count"] == len(values["compared_links"]) > 0
    assert values["missing_jointed_links"] == []
    assert {"configuration", "link"} == set(values["max_position_error_at"]) == set(values["max_angle_error_at"])
    assert record["reference"]["data"] is not None
    assert f"record: {path}" in result.stdout
    assert [Path(entry["path"]).name for entry in record["artifacts"]] == [f"{path.stem}-errors.png"]
    assert all(Path(entry["path"]).is_file() for entry in record["artifacts"])


def random_rotations(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    q = rng.normal(size=(*shape, 4))
    return common.quat_wxyz_to_matrix(q / np.linalg.norm(q, axis=-1, keepdims=True))


def synthetic_reference(rng: np.random.Generator) -> dict:
    return {
        "joint_names": ["l_a_joint", "l_b_joint"],
        "labels": ["open", "random_000", "random_001"],
        "q": np.zeros((3, 2)),
        "body_names": ["l_mount", "l_wrist", "l_a", "l_b"],
        "root_body": "l_mount",
        "position": rng.normal(size=(3, 4, 3)),
        "rotation": random_rotations(rng, (3, 4)),
        "body_has_joint": np.array([False, False, True, True]),
    }


def rotate(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    skew = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * skew @ skew


def test_compare_locates_the_worst_link_and_configuration_after_a_rigid_offset():
    ref = synthetic_reference(np.random.default_rng(11))
    order = [3, 0, 2, 1]
    names = [ref["body_names"][i] for i in order] + ["l_extra"]
    position = np.concatenate([ref["position"][:, order], np.zeros((3, 1, 3))], axis=1)
    rotation = np.concatenate([ref["rotation"][:, order], np.broadcast_to(np.eye(3), (3, 1, 3, 3))], axis=1)
    position[1, 0] += [2e-6, 0.0, 0.0]
    rotation[2, 2] = rotation[2, 2] @ rotate([0.0, 1.0, 1.0], 3e-6)
    motion = rotate([1.0, 0.0, 0.0], math.pi)
    position = position @ motion.T + [0.0, 0.0, -0.08]
    rotation = motion @ rotation

    result = check_kinematics.compare(ref, names, position, rotation)
    assert result["max_position_error_m"] == pytest.approx(2e-6, rel=1e-6)
    assert result["max_position_error_at"] == {"configuration": "random_000", "link": "l_b"}
    assert result["max_angle_error_rad"] == pytest.approx(3e-6, rel=1e-6)
    assert result["max_angle_error_at"] == {"configuration": "random_001", "link": "l_a"}
    assert result["compared_links"] == ["l_mount", "l_wrist", "l_a", "l_b"]
    assert result["compared_link_count"] == 4
    assert result["other_only_links"] == ["l_extra"]
    assert result["mujoco_only_links"] == result["missing_jointed_links"] == []
    assert result["links"]["l_mount"]["max_position_error_m"] == 0.0
    assert result["links"]["l_mount"]["max_angle_error_rad"] < 1e-12


def test_compare_lists_jointed_links_missing_from_the_other_simulator():
    ref = synthetic_reference(np.random.default_rng(12))
    result = check_kinematics.compare(ref, ref["body_names"][:3], ref["position"][:, :3], ref["rotation"][:, :3])
    assert result["mujoco_only_links"] == result["missing_jointed_links"] == ["l_b"]
    assert result["max_position_error_m"] < 1e-12


def test_compare_needs_the_root_link_and_finite_poses():
    ref = synthetic_reference(np.random.default_rng(13))
    with pytest.raises(common.ValidationError, match="root link 'l_mount'"):
        check_kinematics.compare(ref, ref["body_names"][1:], ref["position"][:, 1:], ref["rotation"][:, 1:])
    position = ref["position"].copy()
    position[0, 2, 1] = math.nan
    with pytest.raises(common.ValidationError, match="non-finite"):
        check_kinematics.compare(ref, ref["body_names"], position, ref["rotation"])


def test_configurations_start_with_the_presets_and_are_exact_in_float32(description_root):
    mjcf = description_root / assets.MJCF_ENTRIES[HAND]
    names = [conventions.HAND_PREFIXES[HAND] + suffix for suffix in conventions.JOINT_SUFFIXES]
    labels, q = check_kinematics.configurations(names, mjcf, samples=5, seed=3)
    assert labels == [*scene.POSES, "random_000", "random_001", "random_002", "random_003", "random_004"]
    presets = np.array([scene.pose_vector(names, pose) for pose in scene.POSES])
    np.testing.assert_array_equal(q[:4], presets.astype(np.float32).astype(np.float64))
    np.testing.assert_array_equal(q, q.astype(np.float32).astype(np.float64))
    ranges = scene.description_joint_ranges(mjcf)
    low, high = np.array([ranges[name] for name in names]).T
    assert np.all(q[4:] >= low - 1e-6) and np.all(q[4:] <= high + 1e-6)
    np.testing.assert_array_equal(check_kinematics.configurations(names, mjcf, samples=5, seed=3)[1], q)


def test_samples_and_seed_only_apply_to_the_mujoco_reference(tmp_path, description_root):
    output_dir = tmp_path / "out"
    result = run_check(*base_flags(description_root, output_dir), "--simulator", "mujoco_warp",
                       "--reference", tmp_path / "reference.json", "--samples", 3)
    assert result.returncode == common.EXIT_ERROR
    assert "--samples" in result.stderr
    assert not output_dir.exists()


def test_a_negative_seed_is_rejected_before_anything_runs(tmp_path, description_root):
    output_dir = tmp_path / "out"
    result = run_check(*base_flags(description_root, output_dir), "--simulator", "mujoco", "--seed", -1)
    assert result.returncode == common.EXIT_ERROR
    assert "--seed: must be >= 0, got -1" in result.stderr
    assert not output_dir.exists()


def test_mujoco_run_writes_the_reference_record_and_poses(tmp_path, description_root):
    path, record = make_reference(description_root, tmp_path / "reference")
    started = datetime.datetime.fromisoformat(record["created"])
    assert path.name == f"hand2_2-{HAND}-mujoco-check_kinematics-{started:%Y%m%d-%H%M%S}.json"
    assert record["verdict"] == "reference"
    assert set(record["verdicts"]) == VERDICT_ITEMS
    assert {item["verdict"] for item in record["verdicts"].values()} == {"reference"}
    assert Path(record["data"]["path"]).name == f"{path.stem}-reference.npz"
    count = len(scene.POSES) + SAMPLES
    with np.load(record["data"]["path"]) as data:
        names = [str(name) for name in data["body_names"]]
        root = names.index(str(data["root_body"]))
        assert data["q"].shape == (count, len(conventions.JOINT_SUFFIXES))
        assert [str(label) for label in data["config_labels"]][:4] == list(scene.POSES)
        assert data["position"].shape == (count, len(names), 3)
        assert data["rotation"].shape == (count, len(names), 3, 3)
        assert int(data["body_has_joint"].sum()) == len(conventions.JOINT_SUFFIXES)
        assert not any(name.startswith("scene_") for name in names)
        np.testing.assert_allclose(data["position"][:, root], [record["protocol"]["scene"]["hand_roots"][0]] * count)
        tip = names.index(f"{conventions.HAND_PREFIXES[HAND]}index_finger_distal")
        assert np.linalg.norm(data["position"][0, tip] - data["position"][2, tip]) > 0.01


@pytest.mark.gpu
def test_mujoco_warp_run_compares_against_the_reference_and_fails_on_zero_thresholds(tmp_path, description_root):
    pytest.importorskip("mujoco_warp")
    reference, _ = make_reference(description_root, tmp_path / "reference")
    output_dir = tmp_path / "warp"
    result = run_check(*base_flags(description_root, output_dir), "--simulator", "mujoco_warp",
                       "--reference", reference)
    path, record = only_record(output_dir)
    assert_compared_record(result, path, record, "mujoco_warp")
    assert record["values"]["max_joint_readback_error_rad"] == 0.0
    assert record["values"]["mujoco_only_links"] == record["values"]["other_only_links"] == []

    output_dir = tmp_path / "warp-zero"
    result = run_check(*base_flags(description_root, output_dir), "--simulator", "mujoco_warp",
                       "--reference", reference, "--thresholds", zero_thresholds(tmp_path))
    assert result.returncode == common.EXIT_FAIL, result.stderr
    assert only_record(output_dir)[1]["verdict"] == "fail"


@pytest.mark.isaac
def test_isaaclab_run_compares_against_the_reference_and_exits_one_on_zero_thresholds(tmp_path, description_root):
    pytest.importorskip("isaaclab")
    reference, _ = make_reference(description_root, tmp_path / "reference")
    output_dir = tmp_path / "isaaclab"
    result = run_check(*base_flags(description_root, output_dir), "--simulator", "isaaclab",
                       "--reference", reference, timeout=ISAAC_TIMEOUT_S)
    path, record = only_record(output_dir)
    assert_compared_record(result, path, record, "isaaclab")
    check = record["values"]["root_rotation_check"]
    assert check["angle_rad"] <= check["tolerance_rad"]

    output_dir = tmp_path / "isaaclab-zero"
    result = run_check(*base_flags(description_root, output_dir), "--simulator", "isaaclab",
                       "--reference", reference, "--thresholds", zero_thresholds(tmp_path), timeout=ISAAC_TIMEOUT_S)
    assert result.returncode == common.EXIT_FAIL, result.stderr
    assert only_record(output_dir)[1]["verdict"] == "fail"


def downstream_links(mjcf: Path, joint: str) -> set[str]:
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    top = int(model.jnt_bodyid[model.joint(joint).id])

    def moves_with_joint(body: int) -> bool:
        while body > 0:
            if body == top:
                return True
            body = int(model.body_parentid[body])
        return False

    return {model.body(body).name for body in range(1, model.nbody) if moves_with_joint(body)}


def flip_joint_axis(spec: mujoco.MjSpec, joint: str) -> None:
    [found] = [candidate for candidate in spec.joints if candidate.name == joint]
    found.axis = -np.asarray(found.axis)


def flipping_load(original_load):
    def load(mjcf, **kwargs):
        add_scene = kwargs.pop("scene")

        def broken_scene(spec: mujoco.MjSpec) -> None:
            add_scene(spec)
            flip_joint_axis(spec, FLIPPED_JOINT)

        return original_load(mjcf, scene=broken_scene, **kwargs)

    return load


def over_threshold(links: dict) -> set[str]:
    thresholds = common.load_thresholds()
    position_limit = common.criterion(thresholds, "kinematics", "max_position_error", unit="m")
    angle_limit = common.criterion(thresholds, "kinematics", "max_angle_error", unit="rad")
    return {name for name, errors in links.items()
            if errors["max_position_error_m"] > position_limit or errors["max_angle_error_rad"] > angle_limit}


def assert_failed_downstream(result: dict, downstream: set[str]) -> None:
    assert result["max_position_error_at"]["link"] in downstream
    assert result["max_angle_error_at"]["link"] in downstream
    assert over_threshold(result["links"]) == downstream


def flip_usd_joint(usd: Path, joint: str) -> None:
    from pxr import Gf, Usd, UsdPhysics

    def matrix(q) -> np.ndarray:
        return common.quat_wxyz_to_matrix([q.GetReal(), *q.GetImaginary()])

    stage = Usd.Stage.Open(str(usd))
    [prim] = [prim for prim in stage.Traverse() if prim.GetName() == joint and prim.IsA(UsdPhysics.RevoluteJoint)]
    revolute = UsdPhysics.RevoluteJoint(prim)
    perpendicular = {"X": (0.0, 1.0, 0.0), "Y": (0.0, 0.0, 1.0), "Z": (1.0, 0.0, 0.0)}[revolute.GetAxisAttr().Get()]
    # A half turn about a perpendicular axis on both joint frames keeps the zero pose and reverses the joint direction.
    half_turn = Gf.Quatf(0.0, Gf.Vec3f(*perpendicular))
    for attribute in (revolute.GetLocalRot0Attr(), revolute.GetLocalRot1Attr()):
        before = attribute.Get()
        attribute.Set(before * half_turn)
        np.testing.assert_allclose(matrix(attribute.Get()), matrix(before) @ matrix(half_turn), atol=1e-6)
    assert stage.GetRootLayer().Save()


def test_flipped_joint_axis_fails_exactly_the_links_downstream_of_it(tmp_path, description_root):
    _, record = make_reference(description_root, tmp_path / "reference")
    data_path = Path(record["data"]["path"])
    ref = check_kinematics.read_reference_data(data_path.read_bytes(), data_path)
    asset = assets.resolve(HAND, description_root, strict=True)
    view = framing.view(asset.mjcf, HAND, mujoco_platform.PLATFORM, check_kinematics.VIEW_KIND)
    loaded = flipping_load(mujoco_platform.load)(asset.mjcf, scene=lambda spec: scene.add_scene(spec, HAND, view),
                                                 description_version=asset.expected_version)
    ids, names, _ = check_kinematics.hand_bodies(loaded.model)
    position = np.empty((len(ref["q"]), len(ids), 3))
    rotation = np.empty((len(ref["q"]), len(ids), 3, 3))
    for index, q in enumerate(ref["q"]):
        loaded.data.qpos[list(loaded.qpos_adr)] = q
        mujoco.mj_kinematics(loaded.model, loaded.data)
        position[index] = loaded.data.xpos[ids]
        rotation[index] = loaded.data.xmat[ids].reshape(-1, 3, 3)

    result = check_kinematics.compare(ref, names, position, rotation)
    thresholds = common.load_thresholds()
    assert common.judge(result["max_position_error_m"],
                        common.criterion(thresholds, "kinematics", "max_position_error", unit="m")) == "fail"
    assert common.judge(result["max_angle_error_rad"],
                        common.criterion(thresholds, "kinematics", "max_angle_error", unit="rad")) == "fail"
    assert_failed_downstream(result, downstream_links(asset.mjcf, FLIPPED_JOINT))


@pytest.mark.gpu
def test_mujoco_warp_run_fails_at_the_links_downstream_of_a_flipped_joint(tmp_path, description_root, monkeypatch):
    pytest.importorskip("mujoco_warp")
    from wuji_sim.platforms import mujoco_warp as warp_platform

    reference, _ = make_reference(description_root, tmp_path / "reference")
    monkeypatch.setattr(warp_platform, "load", flipping_load(warp_platform.load))
    output_dir = tmp_path / "warp-flipped"
    code = check_kinematics.main([str(SCRIPT), *map(str, base_flags(description_root, output_dir)),
                                  "--simulator", "mujoco_warp", "--reference", str(reference)])
    assert code == common.EXIT_FAIL
    _, record = only_record(output_dir)
    assert record["verdict"] == "fail"
    assert_failed_downstream(record["values"],
                             downstream_links(description_root / assets.MJCF_ENTRIES[HAND], FLIPPED_JOINT))


@pytest.mark.isaac
def test_isaaclab_run_fails_at_the_links_downstream_of_a_flipped_usd_joint(tmp_path, description_root):
    pytest.importorskip("isaaclab")
    hand_model = tmp_path / "hand-model"
    shutil.copytree(description_root, hand_model, ignore=shutil.ignore_patterns("__pycache__"))
    # Without manifest.json the edited copy resolves as a modified checkout instead of a mismatched package.
    (hand_model / "manifest.json").unlink()
    flip_usd_joint(hand_model / assets.USD_ENTRIES[HAND], FLIPPED_JOINT)
    flags = ["--hand", HAND, "--asset-root", hand_model]
    result = run_check(*flags, "--output-dir", tmp_path / "reference", "--simulator", "mujoco", "--samples", SAMPLES)
    assert result.returncode == common.EXIT_PASS, result.stderr
    reference, _ = only_record(tmp_path / "reference")
    output_dir = tmp_path / "isaaclab-flipped"
    result = run_check(*flags, "--output-dir", output_dir, "--simulator", "isaaclab", "--reference", reference,
                       timeout=ISAAC_TIMEOUT_S)
    assert result.returncode == common.EXIT_FAIL, result.stderr
    _, record = only_record(output_dir)
    assert record["verdict"] == "fail"
    assert record["values"]["root_rotation_check"]["angle_rad"] <= check_kinematics.ROOT_ROTATION_TOLERANCE_RAD
    assert_failed_downstream(record["values"], downstream_links(hand_model / assets.MJCF_ENTRIES[HAND], FLIPPED_JOINT))
