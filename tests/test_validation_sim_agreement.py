# SPDX-License-Identifier: Apache-2.0
import datetime
import json
import math
import subprocess
import sys
import types
from pathlib import Path

import mujoco
import numpy as np
import pytest

from validation import check_sim_agreement, common
from wuji_sim import assets, conventions, framing, scene
from wuji_sim.platforms import mujoco as mujoco_platform

PUBLIC = Path(__file__).resolve().parents[1]
MODULE = "validation.check_sim_agreement"
SCRIPT = PUBLIC / "validation" / "check_sim_agreement.py"
HAND = "left"
JOINT_NAMES = [conventions.HAND_PREFIXES[HAND] + suffix for suffix in conventions.JOINT_SUFFIXES]
DT = 0.005
STEPS = {"03_mit_tracking": 800, "04_sweep_joints": 4800}
CHECK_TIMEOUT_S = 600.0
# Kit startup dominates an Isaac Lab run.
ISAAC_TIMEOUT_S = 900.0
WARP_GATE = 'max_joint_rms_error = { value = 1e-6, unit = "rad" }'
ISAAC_REPORT_ONLY = "[sim_agreement.isaaclab]\nreport_only = true"
GAIN_JOINT = conventions.HAND_PREFIXES[HAND] + "index_finger_pip"


def run_check(*flags, timeout: float = CHECK_TIMEOUT_S) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", MODULE, *map(str, flags)], cwd=PUBLIC, capture_output=True, text=True,
                          timeout=timeout)


def base_flags(description_root: Path, output_dir: Path) -> list:
    return ["--hand", HAND, "--asset-root", description_root, "--strict", "--output-dir", output_dir]


def only_record(output_dir: Path) -> tuple[Path, dict]:
    [path] = Path(output_dir).glob("*.json")
    return path, json.loads(path.read_text(encoding="utf-8"))


def make_reference(description_root: Path, output_dir: Path) -> tuple[Path, dict]:
    result = run_check(*base_flags(description_root, output_dir), "--simulator", "mujoco")
    assert result.returncode == common.EXIT_PASS, result.stderr
    return only_record(output_dir)


def thresholds_file(tmp_path: Path, old: str, new: str) -> Path:
    text = common.THRESHOLDS_PATH.read_text(encoding="utf-8")
    assert text.count(old) == 1
    path = tmp_path / "zero-thresholds.toml"
    path.write_text(text.replace(old, new), encoding="utf-8")
    return path


def assert_compared_record(result: subprocess.CompletedProcess, path: Path, record: dict, simulator: str) -> None:
    assert result.returncode in (common.EXIT_PASS, common.EXIT_FAIL), result.stderr
    assert result.returncode == common.exit_code(record["verdict"])
    assert (record["check"], record["simulator"], record["hand"]) == ("sim_agreement", simulator, HAND)
    motions = record["values"]["motions"]
    assert set(motions) == set(STEPS)
    for motion, steps in STEPS.items():
        entry = motions[motion]
        assert entry["steps"] == steps
        assert set(entry["joints"]) == set(JOINT_NAMES)
        assert entry["worst_joint_rms_joint"] in JOINT_NAMES and entry["worst_joint_max_abs_joint"] in JOINT_NAMES
        assert math.isfinite(entry["worst_joint_rms_rad"]) and math.isfinite(entry["worst_joint_max_abs_rad"])
        assert entry["worst_joint_rms_rad"] <= entry["worst_joint_max_abs_rad"]
        assert record["verdicts"][f"{motion}.max_joint_rms_error"]["value"] == entry["worst_joint_rms_rad"]
    assert record["reference"]["data"] is not None
    assert f"record: {path}" in result.stdout
    assert [Path(entry["path"]).name for entry in record["artifacts"]] == [f"{path.stem}-errors.png"]
    assert all(Path(entry["path"]).is_file() for entry in record["artifacts"])


def test_motion_errors_name_the_worst_joints():
    q_ref = np.zeros((4, 3))
    q_test = np.array([[0.0, 0.1, 0.0], [0.0, -0.1, 0.0], [0.0, 0.1, 0.0], [0.3, -0.1, 0.0]])
    errors = check_sim_agreement.motion_errors(["a", "b", "c"], q_ref, q_test)
    assert errors["steps"] == 4
    assert (errors["worst_joint_rms_joint"], errors["worst_joint_rms_rad"]) == ("a", pytest.approx(0.15))
    assert (errors["worst_joint_max_abs_joint"], errors["worst_joint_max_abs_rad"]) == ("a", pytest.approx(0.3))
    assert errors["joints"]["b"] == {"rms_rad": pytest.approx(0.1), "max_abs_rad": pytest.approx(0.1)}
    assert errors["joints"]["c"] == {"rms_rad": 0.0, "max_abs_rad": 0.0}


def test_commands_follow_the_example_references(description_root):
    q_des, qd_des = check_sim_agreement.commands("03_mit_tracking", JOINT_NAMES, DT, None)
    assert q_des.shape == qd_des.shape == (STEPS["03_mit_tracking"], len(JOINT_NAMES))
    np.testing.assert_array_equal(q_des[0], scene.pose_vector(JOINT_NAMES, "open"))
    for k in (1, 333, 799):
        expected = scene.tracking_reference(JOINT_NAMES, k * DT)
        np.testing.assert_array_equal(q_des[k], expected[0])
        np.testing.assert_array_equal(qd_des[k], expected[1])
    ranges = scene.description_joint_ranges(description_root / assets.MJCF_ENTRIES[HAND])
    ranges = np.array([ranges[name] for name in JOINT_NAMES])
    q_des, qd_des = check_sim_agreement.commands("04_sweep_joints", JOINT_NAMES, DT, ranges)
    assert q_des.shape == (STEPS["04_sweep_joints"], len(JOINT_NAMES))
    np.testing.assert_array_equal(q_des[2500], scene.sweep_reference(JOINT_NAMES, ranges, 2500 * DT)[0])


def test_agreement_gate_reads_report_only_or_the_rms_criterion(tmp_path):
    thresholds = common.load_thresholds()
    assert check_sim_agreement.agreement_gate(thresholds, "mujoco_warp") == 1e-6
    assert check_sim_agreement.agreement_gate(thresholds, "isaaclab") is None
    gated = common.load_thresholds(thresholds_file(tmp_path, ISAAC_REPORT_ONLY,
                                                   "[sim_agreement.isaaclab]\n" + WARP_GATE.replace("1e-6", "0.0")))
    assert check_sim_agreement.agreement_gate(gated, "isaaclab") == 0.0


@pytest.mark.parametrize("simulator, views", [
    ("mujoco", {"03_mit_tracking": "tracking", "04_sweep_joints": "sweep"}),
    ("mujoco_warp", {"03_mit_tracking": "tracking", "04_sweep_joints": "sweep"}),
    ("isaaclab", {"03_mit_tracking": "tracking", "04_sweep_joints": "tracking"}),
])
def test_protocol_records_the_view_each_simulator_loads_for_a_motion(simulator, views):
    run = types.SimpleNamespace(args=types.SimpleNamespace(simulator=simulator))
    recorded = check_sim_agreement.protocol(run, JOINT_NAMES, DT, 1.0, 0.1, STEPS)
    assert {motion: entry["view"] for motion, entry in recorded["motions"].items()} == views


def test_mujoco_run_writes_the_reference_record_and_trajectories(tmp_path, description_root):
    path, record = make_reference(description_root, tmp_path / "reference")
    started = datetime.datetime.fromisoformat(record["created"])
    assert path.name == f"hand2_2-{HAND}-mujoco-check_sim_agreement-{started:%Y%m%d-%H%M%S}.json"
    assert record["verdict"] == "reference"
    assert set(record["verdicts"]) == {f"{motion}.max_joint_rms_error" for motion in STEPS}
    assert {item["verdict"] for item in record["verdicts"].values()} == {"reference"}
    with np.load(record["data"]["path"]) as data:
        assert [str(name) for name in data["joint_names"]] == JOINT_NAMES
        assert float(data["dt"]) == DT
        assert (float(data["kp"]), float(data["kd"])) == (scene.DEMO_KP, scene.DEMO_KD)
        for motion, steps in STEPS.items():
            q = data[f"{motion}.q"]
            assert q.shape == data[f"{motion}.q_des"].shape == data[f"{motion}.qd_des"].shape == (steps, 20)
            np.testing.assert_array_equal(q[0], 0.0)
            assert np.ptp(q, axis=0).max() > 0.1
            assert np.abs(q - data[f"{motion}.q_des"]).max() < 0.5


@pytest.mark.gpu
def test_mujoco_warp_run_compares_against_the_reference_and_fails_on_zero_thresholds(tmp_path, description_root):
    pytest.importorskip("mujoco_warp")
    reference, _ = make_reference(description_root, tmp_path / "reference")
    output_dir = tmp_path / "warp"
    result = run_check(*base_flags(description_root, output_dir), "--simulator", "mujoco_warp",
                       "--reference", reference)
    path, record = only_record(output_dir)
    assert_compared_record(result, path, record, "mujoco_warp")
    assert {item["threshold"] for item in record["verdicts"].values()} == {1e-6}
    for motion, entry in record["values"]["motions"].items():
        verdict = "pass" if entry["worst_joint_rms_rad"] <= 1e-6 else "fail"
        assert record["verdicts"][f"{motion}.max_joint_rms_error"]["verdict"] == verdict, motion

    output_dir = tmp_path / "warp-zero"
    zero = thresholds_file(tmp_path, WARP_GATE, WARP_GATE.replace("1e-6", "0.0"))
    result = run_check(*base_flags(description_root, output_dir), "--simulator", "mujoco_warp",
                       "--reference", reference, "--thresholds", zero)
    assert result.returncode == common.EXIT_FAIL, result.stderr
    assert only_record(output_dir)[1]["verdict"] == "fail"


@pytest.mark.isaac
def test_isaaclab_run_reports_agreement_and_exits_one_when_gated_at_zero(tmp_path, description_root):
    pytest.importorskip("isaaclab")
    reference, _ = make_reference(description_root, tmp_path / "reference")
    output_dir = tmp_path / "isaaclab"
    result = run_check(*base_flags(description_root, output_dir), "--simulator", "isaaclab",
                       "--reference", reference, timeout=ISAAC_TIMEOUT_S)
    path, record = only_record(output_dir)
    assert_compared_record(result, path, record, "isaaclab")
    assert result.returncode == common.EXIT_PASS
    assert {item["verdict"] for item in record["verdicts"].values()} == {"report_only"}
    assert math.isfinite(record["values"]["reset_repeat_max_abs_rad"])

    output_dir = tmp_path / "isaaclab-zero"
    gated = thresholds_file(tmp_path, ISAAC_REPORT_ONLY,
                            "[sim_agreement.isaaclab]\n" + WARP_GATE.replace("1e-6", "0.0"))
    result = run_check(*base_flags(description_root, output_dir), "--simulator", "isaaclab",
                       "--reference", reference, "--thresholds", gated, timeout=ISAAC_TIMEOUT_S)
    assert result.returncode == common.EXIT_FAIL, result.stderr
    assert only_record(output_dir)[1]["verdict"] == "fail"


def mujoco_runs(description_root: Path, ref: dict, kp_scale: dict[str, float]) -> dict[str, np.ndarray]:
    asset = assets.resolve(HAND, description_root, strict=True)
    runs = {}
    for motion in check_sim_agreement.MOTIONS:
        view = framing.view(asset.mjcf, HAND, mujoco_platform.PLATFORM, check_sim_agreement.MOTION_VIEWS[motion])
        loaded = mujoco_platform.load(asset.mjcf, scene=lambda spec: scene.add_scene(spec, HAND, view),
                                      description_version=asset.expected_version)
        assert list(loaded.joint_names) == ref["joint_names"]
        kp = np.array([ref["kp"] * kp_scale.get(name, 1.0) for name in loaded.joint_names])
        kd = np.full(len(kp), ref["kd"])
        mujoco_platform.set_mit_gains(loaded, kp, kd)
        qpos_adr, actuator_ids = list(loaded.qpos_adr), list(loaded.actuator_ids)
        q_des, qd_des = ref[motion]["q_des"], ref[motion]["qd_des"]
        q = np.empty_like(q_des)
        for k in range(len(q_des)):
            q[k] = loaded.data.qpos[qpos_adr]
            loaded.data.ctrl[actuator_ids] = kp * q_des[k] + kd * qd_des[k]
            mujoco.mj_step(loaded.model, loaded.data)
        runs[motion] = q
    return runs


def test_doubled_joint_gain_fails_the_warp_gate_at_that_joint(tmp_path, description_root):
    _, record = make_reference(description_root, tmp_path / "reference")
    data_path = Path(record["data"]["path"])
    ref = check_sim_agreement.read_reference_data(data_path.read_bytes(), data_path)
    for motion, q in mujoco_runs(description_root, ref, {}).items():
        np.testing.assert_array_equal(q, ref[motion]["q"])
    gate = check_sim_agreement.agreement_gate(common.load_thresholds(), "mujoco_warp")
    for motion, q in mujoco_runs(description_root, ref, {GAIN_JOINT: 2.0}).items():
        errors = check_sim_agreement.motion_errors(ref["joint_names"], ref[motion]["q"], q)
        assert common.judge(errors["worst_joint_rms_rad"], gate) == "fail", motion
        assert errors["worst_joint_rms_joint"] == GAIN_JOINT, (motion, errors["worst_joint_rms_joint"])


@pytest.mark.gpu
def test_mujoco_warp_run_fails_at_a_joint_with_doubled_gain(tmp_path, description_root, monkeypatch):
    pytest.importorskip("mujoco_warp")
    from wuji_sim.platforms import mujoco_warp as warp_platform

    reference, _ = make_reference(description_root, tmp_path / "reference")
    set_mit_gains = warp_platform.set_mit_gains

    def set_doubled_gain(loaded, kp, kd):
        kp = np.array(kp, dtype=np.float64)
        kp[loaded.joint_names.index(GAIN_JOINT)] *= 2.0
        set_mit_gains(loaded, kp, kd)

    monkeypatch.setattr(warp_platform, "set_mit_gains", set_doubled_gain)
    output_dir = tmp_path / "warp-doubled-gain"
    code = check_sim_agreement.main([str(SCRIPT), *map(str, base_flags(description_root, output_dir)),
                                     "--simulator", "mujoco_warp", "--reference", str(reference)])
    assert code == common.EXIT_FAIL
    _, record = only_record(output_dir)
    motions = record["values"]["motions"]
    caught = [motion for motion, entry in motions.items() if entry["worst_joint_rms_joint"] == GAIN_JOINT
              and record["verdicts"][f"{motion}.max_joint_rms_error"]["verdict"] == "fail"]
    assert caught, {motion: entry["worst_joint_rms_joint"] for motion, entry in motions.items()}
