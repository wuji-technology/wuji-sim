# SPDX-License-Identifier: Apache-2.0
import argparse
import dataclasses
import datetime
import hashlib
import json
import math
import shutil

import mujoco
import numpy as np
import pytest

from validation import common
from wuji_sim import assets

NOW = datetime.datetime(2026, 11, 19, 12, 0, 0)


def axis_rotation(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    skew = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * skew @ skew


def random_rotations(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    q = rng.normal(size=(*shape, 4))
    return common.quat_wxyz_to_matrix(q / np.linalg.norm(q, axis=-1, keepdims=True))


@pytest.fixture(scope="module")
def left_asset(description_root) -> assets.ResolvedAsset:
    return assets.resolve("left", description_root)


def new_record(tmp_path, *, check="kinematics", simulator="mujoco", hand="left", now=NOW):
    return common.ValidationRecord(check=check, simulator=simulator, hand=hand, output_dir=tmp_path,
                                   argv=["check.py", "--hand", hand], now=now)


def write_reference(record: common.ValidationRecord, asset, *, data_bytes: bytes | None = b"reference data"):
    if data_bytes is not None:
        record.artifact_path("reference", ".npz", data=True).write_bytes(data_bytes)
    return record.write(asset=asset, thresholds=common.load_thresholds(), reference=None, protocol={"kind": "test"},
                        values={"configurations": 1},
                        verdicts={"item": {"value": None, "threshold": 1.0, "unit": "m", "verdict": "reference"}})


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_quaternions_in_both_orders_match_mujoco(seed):
    rng = np.random.default_rng(seed)
    wxyz = rng.normal(size=4)
    wxyz /= np.linalg.norm(wxyz)
    expected = np.empty(9)
    mujoco.mju_quat2Mat(expected, wxyz)
    np.testing.assert_allclose(common.quat_wxyz_to_matrix(wxyz), expected.reshape(3, 3), rtol=0.0, atol=1e-15)
    xyzw = np.roll(wxyz, -1)
    np.testing.assert_allclose(common.quat_xyzw_to_matrix(xyzw), expected.reshape(3, 3), rtol=0.0, atol=1e-15)


def test_quaternion_batches_keep_their_leading_shape_and_are_normalized():
    q = np.random.default_rng(4).normal(size=(2, 5, 4)) * 3.0
    matrices = common.quat_xyzw_to_matrix(q)
    assert matrices.shape == (2, 5, 3, 3)
    np.testing.assert_allclose(np.swapaxes(matrices, -1, -2) @ matrices, np.broadcast_to(np.eye(3), (2, 5, 3, 3)),
                               atol=1e-14)


@pytest.mark.parametrize("q", [[0.0, 0.0, 0.0, 0.0], [math.nan, 0.0, 0.0, 1.0], [math.inf, 0.0, 0.0, 1.0],
                               [1.0, 0.0, 0.0]])
def test_quaternion_conversion_rejects_invalid_input(q):
    with pytest.raises(common.ValidationError):
        common.quat_wxyz_to_matrix(q)


@pytest.mark.parametrize("component", [1.5e308, 1e-320])
def test_quaternions_with_huge_or_tiny_components_still_convert(component):
    np.testing.assert_allclose(common.quat_wxyz_to_matrix([component, component, 0.0, 0.0]),
                               axis_rotation([1.0, 0.0, 0.0], math.pi / 2), rtol=0.0, atol=1e-15)


@pytest.mark.parametrize("angle", [0.0, 1e-7, math.pi / 2, math.pi - 1e-6])
def test_rotation_angle_recovers_the_relative_angle(angle):
    r_ref = random_rotations(np.random.default_rng(5), ())
    r_test = r_ref @ axis_rotation([0.3, -0.5, 0.8], angle)
    got = common.rotation_angle(r_ref, r_test)
    assert got == pytest.approx(angle, rel=1e-9, abs=1e-15)


def test_rotation_angle_broadcasts_over_batches():
    rng = np.random.default_rng(6)
    r_ref = random_rotations(rng, (3, 4))
    angles = rng.uniform(0.0, math.pi, size=(3, 4))
    r_test = np.stack([np.stack([r_ref[i, j] @ axis_rotation([1.0, 2.0, 3.0], angles[i, j]) for j in range(4)])
                       for i in range(3)])
    np.testing.assert_allclose(common.rotation_angle(r_ref, r_test), angles, rtol=1e-9, atol=1e-12)


def test_root_relative_poses_ignore_a_rigid_motion_of_the_whole_hand():
    rng = np.random.default_rng(7)
    position = rng.normal(size=(3, 5, 3))
    rotation = random_rotations(rng, (3, 5))
    motion = axis_rotation([0.2, 1.0, -0.4], 2.1)
    moved_position = position @ motion.T + np.array([0.5, -1.0, 2.0])
    moved_rotation = motion @ rotation
    for root in (0, 3):
        pos, rot = common.root_relative(position, rotation, root)
        moved_pos, moved_rot = common.root_relative(moved_position, moved_rotation, root)
        np.testing.assert_allclose(moved_pos, pos, atol=1e-12)
        np.testing.assert_allclose(moved_rot, rot, atol=1e-12)
        np.testing.assert_array_equal(pos[:, root], 0.0)


def test_root_relative_locates_an_offset_of_one_link():
    rng = np.random.default_rng(8)
    position = rng.normal(size=(4, 6, 3))
    rotation = random_rotations(rng, (4, 6))
    shifted = position.copy()
    shifted[2, 4] += [3e-6, -4e-6, 0.0]
    pos, _ = common.root_relative(position, rotation, 0)
    shifted_pos, _ = common.root_relative(shifted, rotation, 0)
    error = np.linalg.norm(shifted_pos - pos, axis=-1)
    assert np.unravel_index(np.argmax(error), error.shape) == (2, 4)
    assert error.max() == pytest.approx(5e-6, rel=1e-9)
    assert np.count_nonzero(error > 1e-12) == 1


def test_root_relative_rejects_mismatched_shapes_and_a_bad_root():
    position = np.zeros((2, 3, 3))
    rotation = np.broadcast_to(np.eye(3), (2, 3, 3, 3))
    with pytest.raises(common.ValidationError):
        common.root_relative(position[:, :2], rotation, 0)
    with pytest.raises(common.ValidationError):
        common.root_relative(position, rotation, 3)


def test_joint_errors_report_per_joint_rms_and_max():
    q_ref = np.zeros((4, 3))
    q_test = np.array([[0.0, 1.0, 3.0], [0.0, -1.0, -4.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    errors = common.joint_errors(q_ref, q_test)
    np.testing.assert_allclose(errors["rms"], [0.0, 1.0, 2.5])
    np.testing.assert_allclose(errors["max_abs"], [0.0, 1.0, 4.0])


@pytest.mark.parametrize("q_test", [np.zeros((4, 2)), np.zeros((3, 3)), np.array([[0.0, math.nan, 0.0]] * 4),
                                    np.zeros(4)])
def test_joint_errors_reject_mismatched_or_non_finite_series(q_test):
    with pytest.raises(common.ValidationError):
        common.joint_errors(np.zeros((4, 3)), q_test)


def test_judge_passes_a_value_equal_to_the_threshold():
    assert common.judge(1e-6, 1e-6) == "pass"
    assert common.judge(0.0, 0.0) == "pass"
    assert common.judge(float(np.nextafter(1e-6, 1.0)), 1e-6) == "fail"
    assert common.judge(math.nan, 1.0) == "fail"


@pytest.mark.parametrize("verdicts, expected", [
    ({"a": "pass", "b": "fail", "c": "reference"}, "fail"),
    ({"a": "reference", "b": "reference"}, "reference"),
    ({"a": "reference", "b": "pass"}, "pass"),
    ({"a": "report_only", "b": "report_only"}, "pass"),
    ({"a": "not_applicable", "b": "pass"}, "pass"),
])
def test_record_verdict_rule(verdicts, expected):
    items = {name: {"value": 0.0, "threshold": 1.0, "unit": "m", "verdict": verdict}
             for name, verdict in verdicts.items()}
    assert common.record_verdict(items) == expected


@pytest.mark.parametrize("items", [
    {},
    {"a": {"value": 0.0, "threshold": 1.0, "unit": "m", "verdict": "maybe"}},
    {"a": {"value": 0.0, "threshold": 1.0, "verdict": "pass"}},
])
def test_record_verdict_rejects_malformed_items(items):
    with pytest.raises(common.ValidationError):
        common.record_verdict(items)


def test_exit_code_is_one_only_for_fail():
    assert common.exit_code("fail") == common.EXIT_FAIL == 1
    assert common.exit_code("pass") == common.exit_code("reference") == common.EXIT_PASS == 0
    with pytest.raises(common.ValidationError):
        common.exit_code("error")


def test_criterion_returns_the_value_in_the_expected_unit():
    thresholds = common.load_thresholds()
    assert common.criterion(thresholds, "kinematics", "max_angle_error", unit="rad") == 1e-5
    with pytest.raises(common.ValidationError, match="max_angle_error must be given in m"):
        common.criterion(thresholds, "kinematics", "max_angle_error", unit="m")
    with pytest.raises(common.ValidationError, match="sim_agreement.isaaclab is not a"):
        common.criterion(thresholds, "sim_agreement", "isaaclab", unit="rad")


def test_print_result_lists_every_item_and_returns_the_exit_code(capsys, tmp_path):
    verdicts = {"a": {"value": 2.0, "threshold": 1.0, "unit": "m", "verdict": "fail"},
                "b": {"value": None, "threshold": None, "unit": "rad", "verdict": "report_only"}}
    assert common.print_result(tmp_path / "r.json", verdicts) == common.EXIT_FAIL
    assert capsys.readouterr().out.splitlines() == [
        "a: fail (value 2, threshold 1, unit m)",
        "b: report_only (value n/a, threshold n/a, unit rad)",
        "verdict: fail",
        f"record: {tmp_path / 'r.json'}",
    ]


def test_report_error_prints_one_line_for_expected_errors_and_a_traceback_otherwise(capsys):
    assert common.report_error(common.ValidationError("bad input")) == common.EXIT_ERROR
    assert capsys.readouterr().err == "error: bad input\n"
    try:
        raise KeyError("missing key")
    except KeyError as exc:
        assert common.report_error(exc) == common.EXIT_ERROR
    err = capsys.readouterr().err
    assert err.startswith("Traceback") and "KeyError: 'missing key'" in err


def test_record_names_its_outputs_by_check_simulator_hand_and_start_time(tmp_path):
    record = new_record(tmp_path, check="sim_agreement", simulator="mujoco_warp")
    assert record.stem == "hand2_2-left-mujoco_warp-check_sim_agreement-20261119-120000"
    assert record.artifact_path("errors", ".png") == tmp_path / f"{record.stem}-errors.png"
    with pytest.raises(common.ValidationError):
        record.artifact_path("Errors", ".png")
    with pytest.raises(common.ValidationError):
        record.artifact_path("errors", ".png")
    with pytest.raises(common.ValidationError):
        new_record(tmp_path, simulator="genesis")


def test_record_takes_the_next_free_stem_instead_of_overwriting(tmp_path, left_asset):
    first = write_reference(new_record(tmp_path), left_asset)
    written = first.read_bytes()
    second = new_record(tmp_path)
    assert second.stem == f"{first.stem}-2"
    assert first.read_bytes() == written
    other = tmp_path / "other"
    other.mkdir()
    leftover = other / f"{first.stem}-errors.png"
    leftover.write_bytes(b"left by a check that stopped early")
    assert new_record(other).stem == f"{first.stem}-2"
    assert leftover.read_bytes() == b"left by a check that stopped early"


def test_record_write_holds_identity_verdicts_and_artifact_digests(tmp_path, left_asset):
    record = new_record(tmp_path, check="sim_agreement")
    figure = record.artifact_path("errors", ".png")
    figure.write_bytes(b"figure")
    verdicts = {"a": {"value": 2.0, "threshold": 1.0, "unit": "rad", "verdict": "fail"},
                "b": {"value": 0.5, "threshold": 1.0, "unit": "rad", "verdict": "pass"}}
    path = record.write(asset=left_asset, thresholds=common.load_thresholds(), reference=None,
                        protocol={"steps": 3}, values={"worst": 2.0}, verdicts=verdicts)
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["record_format"] == common.RECORD_FORMAT
    assert (written["check"], written["simulator"], written["hand"]) == ("sim_agreement", "mujoco", "left")
    assert written["verdict"] == "fail"
    assert written["verdicts"] == verdicts
    assert written["asset"]["digest"] == left_asset.digest
    assert written["thresholds"]["sha256"] == hashlib.sha256(common.THRESHOLDS_PATH.read_bytes()).hexdigest()
    assert written["artifacts"] == [{"path": str(figure), "sha256": hashlib.sha256(b"figure").hexdigest()}]
    assert written["data"] is None
    assert {"created", "product", "command", "cwd", "pixi_environment", "sim_commit", "backend_versions",
            "reference", "protocol", "values"} <= set(written)


def test_record_write_requires_registered_artifacts_on_disk(tmp_path, left_asset):
    record = new_record(tmp_path)
    record.artifact_path("errors", ".png")
    with pytest.raises(common.ValidationError, match="not written"):
        write_reference(record, left_asset, data_bytes=None)


def test_load_reference_accepts_a_copy_next_to_its_data_file(tmp_path, left_asset):
    path = write_reference(new_record(tmp_path / "made"), left_asset)
    moved = tmp_path / "moved"
    moved.mkdir()
    for source in (tmp_path / "made").iterdir():
        shutil.copy(source, moved / source.name)
    reference = common.load_reference(moved / path.name, check="kinematics", hand="left", asset=left_asset)
    assert reference["path"] == moved / path.name
    assert reference["data_path"] == moved / "hand2_2-left-mujoco-check_kinematics-20261119-120000-reference.npz"
    assert reference["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert reference["record"]["check"] == "kinematics"


def test_load_reference_hands_over_the_bytes_it_verified(tmp_path, left_asset):
    path = write_reference(new_record(tmp_path), left_asset)
    reference = common.load_reference(path, check="kinematics", hand="left", asset=left_asset)
    reference["data_path"].write_bytes(b"replaced after the check")
    assert reference["data_bytes"] == b"reference data"
    assert reference["record"]["data"]["sha256"] == hashlib.sha256(b"reference data").hexdigest()


@pytest.mark.parametrize("wanted, message", [
    ({"check": "kinematics", "hand": "right"}, "hand is 'left', expected 'right'"),
    ({"check": "sim_agreement", "hand": "left"}, "check is 'kinematics', expected 'sim_agreement'"),
])
def test_load_reference_rejects_another_hand_or_check(tmp_path, left_asset, wanted, message):
    path = write_reference(new_record(tmp_path), left_asset)
    with pytest.raises(common.ValidationError, match=message):
        common.load_reference(path, asset=left_asset, **wanted)


def test_load_reference_rejects_a_record_of_another_simulator(tmp_path, left_asset):
    path = write_reference(new_record(tmp_path), left_asset)
    written = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**written, "simulator": "mujoco_warp"}), encoding="utf-8")
    with pytest.raises(common.ValidationError, match="simulator is 'mujoco_warp', expected 'mujoco'"):
        common.load_reference(path, check="kinematics", hand="left", asset=left_asset)


def test_load_reference_rejects_other_hand_model_assets(tmp_path, left_asset):
    path = write_reference(new_record(tmp_path), left_asset)
    other = dataclasses.replace(left_asset, digest="0" * 64)
    with pytest.raises(common.ValidationError, match=f"assets {left_asset.digest}, this run uses {'0' * 64}"):
        common.load_reference(path, check="kinematics", hand="left", asset=other)


def test_load_reference_rejects_changed_or_missing_data(tmp_path, left_asset):
    path = write_reference(new_record(tmp_path), left_asset)
    data = tmp_path / "hand2_2-left-mujoco-check_kinematics-20261119-120000-reference.npz"
    data.write_bytes(b"changed")
    with pytest.raises(common.ValidationError, match="sha256"):
        common.load_reference(path, check="kinematics", hand="left", asset=left_asset)
    data.unlink()
    with pytest.raises(common.ValidationError, match="not found"):
        common.load_reference(path, check="kinematics", hand="left", asset=left_asset)


def test_load_thresholds_reads_the_shipped_criteria():
    thresholds = common.load_thresholds()
    assert thresholds["file"] == str(common.THRESHOLDS_PATH)
    assert thresholds["sha256"] == hashlib.sha256(common.THRESHOLDS_PATH.read_bytes()).hexdigest()
    assert set(common.CHECKS) <= set(thresholds["tables"])
    assert common.lookup(thresholds, "kinematics", "max_position_error") == {"value": 1e-6, "unit": "m"}
    assert common.lookup(thresholds, "sim_agreement", "isaaclab", "report_only") is True
    with pytest.raises(common.ValidationError, match="sim_agreement.genesis"):
        common.lookup(thresholds, "sim_agreement", "genesis")


@pytest.mark.parametrize("old, new, message", [
    ("[self_collision.push]", "[other.push]", r"no table for \['self_collision'\]"),
    ('max_position_error = { value = 1e-6, unit = "m" }', 'max_position_error = { unit = "m" }',
     r"kinematics.max_position_error is missing \['value'\]"),
    ('max_angle_error = { value = 1e-5, unit = "rad" }', "max_angle_error = { value = 1e-5 }",
     r"kinematics.max_angle_error is missing \['unit'\]"),
    ('max_angle_error = { value = 1e-5, unit = "rad" }', 'max_angle_error = { value = nan, unit = "rad" }',
     r"kinematics.max_angle_error.value must be a finite number"),
    ('max_angle_error = { value = 1e-5, unit = "rad" }', "max_angle_error = 1e-5",
     r"kinematics.max_angle_error must be a table"),
])
def test_load_thresholds_rejects_missing_tables_and_fields(tmp_path, old, new, message):
    text = common.THRESHOLDS_PATH.read_text(encoding="utf-8")
    assert text.count(old) == 1
    path = tmp_path / "thresholds.toml"
    path.write_text(text.replace(old, new), encoding="utf-8")
    with pytest.raises(common.ValidationError, match=message):
        common.load_thresholds(path)


def cli(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    common.add_cli_args(parser)
    return common.parse_args(parser, argv)


def test_cli_requires_a_reference_except_for_mujoco(tmp_path):
    args = cli(["--hand", "left", "--simulator", "mujoco"])
    assert (args.reference, args.thresholds, args.output_dir) == (None, common.THRESHOLDS_PATH, None)
    args = cli(["--hand", "right", "--simulator", "isaaclab", "--reference", str(tmp_path / "r.json")])
    assert args.reference == tmp_path / "r.json"
    for argv in (["--hand", "left", "--simulator", "mujoco_warp"],
                 ["--hand", "left", "--simulator", "mujoco", "--reference", "r.json"]):
        with pytest.raises(SystemExit) as raised:
            cli(argv)
        assert raised.value.code == common.EXIT_ERROR
