# SPDX-License-Identifier: Apache-2.0
import csv
import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("isaaclab")

import imageio.v3 as iio
from isaaclab_example_support import assert_artifacts, assert_moving_trace, base_flags, read_record, run_example

from wuji_sim import conventions, scene


pytestmark = pytest.mark.isaac
PUBLIC = Path(__file__).resolve().parents[1]
SWEEP_EXAMPLE = PUBLIC / "examples" / "isaaclab" / "04_sweep_joints.py"
MUJOCO_SWEEP_EXAMPLE = PUBLIC / "examples" / "mujoco" / "04_sweep_joints.py"
HAND = "left"
JOINT_NAMES = tuple(conventions.HAND_PREFIXES[HAND] + suffix for suffix in conventions.JOINT_SUFFIXES)
# No shortened sweep exists (as in MuJoCo 04); the ceiling covers a slow host or a cold Kit start.
SIM_TIMEOUT_S = 1800.0
RUN_MUJOCO_PARITY_ENV = "WUJI_RUN_MUJOCO_PARITY"
# Must match PARITY_THRESHOLDS_SOURCE in 04_sweep_joints.py.
EXPECTED_PARITY_THRESHOLDS_SOURCE = (
    "independent MuJoCo vs Isaac Lab measurement, 2 deterministic runs: MuJoCo 03/04 run with its trajectory "
    "output enabled, then Isaac Lab 03/04 run with --reference-trajectory pointed at that file; threshold = "
    "observed x 2 rounded up to 2 significant figures"
)


def read_csv_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def column(rows: list[dict], field: str, joint: str) -> list[float]:
    return [float(row[f"{field}:{joint}"]) for row in rows]


def test_sweep_writes_artifacts_and_moving_trajectory(tmp_path, description_root):
    output_dir = tmp_path / "sweep"
    result = run_example(SWEEP_EXAMPLE, *base_flags(description_root, output_dir, hand=HAND), timeout=SIM_TIMEOUT_S)
    assert result.returncode == 0, result.stderr

    record = read_record(output_dir)
    assert record["demo"] == "04_sweep_joints"
    assert record["platform"] == "isaaclab"

    rows = record["results"]["sweep"]["joints"]
    assert len(rows) == len(JOINT_NAMES)
    assert {row["joint"] for row in rows} == set(JOINT_NAMES)

    video = record["results"]["video"]
    assert video["camera"] == scene.PREVIEW_CAMERA
    assert video["frames"] == record["results"]["sweep"]["steps"] // video["steps_per_frame"] + 1
    assert Path(video["path"]).is_file()
    width, height = record["results"]["scene"]["preview"]["render_size"]
    assert iio.imread(video["path"], index=0).shape == (height, width, 3)
    assert Path(record["results"]["figure"]).is_file()

    csv_path = Path(record["results"]["proprioception"]["csv"])
    assert csv_path.is_file()
    csv_rows = read_csv_rows(csv_path)
    assert len(csv_rows) == record["results"]["sweep"]["steps"]
    windows_seen = {row["active_window"] for row in csv_rows}
    expected_windows = {label for label, _ in scene.sweep_groups(JOINT_NAMES)}
    assert windows_seen == expected_windows

    # Shape checks above pass for a frozen hand; the recorded q(t) traces do not.
    for name in JOINT_NAMES:
        assert_moving_trace(column(csv_rows, "q", name), label=name)

    parity = record["results"]["parity"]
    assert parity["threshold_status"] == "threshold not configured"
    assert parity["joints"] == {}
    assert parity["reference"] == {"trajectory_csv": None, "label": None}

    assert_artifacts(record)


def test_sweep_missing_asset_root_exits_nonzero(tmp_path):
    output_dir = tmp_path / "sweep-missing"
    missing_root = tmp_path / "does-not-exist"
    result = run_example(SWEEP_EXAMPLE, *base_flags(missing_root, output_dir, hand=HAND))
    assert result.returncode != 0
    assert "does not exist" in result.stderr, result.stderr
    assert not output_dir.exists() or not list(output_dir.glob("*.json"))


def test_sweep_parity_reports_observed_without_fabricating_threshold(tmp_path, description_root):
    """Self-parity of two identical deterministic sweeps checks the reference reader and gate end to end."""
    baseline_dir = tmp_path / "sweep-baseline"
    baseline = run_example(SWEEP_EXAMPLE, *base_flags(description_root, baseline_dir, hand=HAND), timeout=SIM_TIMEOUT_S)
    assert baseline.returncode == 0, baseline.stderr
    baseline_record = read_record(baseline_dir)
    baseline_csv = Path(baseline_record["results"]["proprioception"]["csv"])
    assert baseline_csv.is_file()

    output_dir = tmp_path / "sweep-self-parity"
    result = run_example(
        SWEEP_EXAMPLE, *base_flags(description_root, output_dir, hand=HAND), "--reference-trajectory", str(baseline_csv),
        timeout=SIM_TIMEOUT_S,
    )
    assert result.returncode == 0, result.stderr

    record = read_record(output_dir)
    parity = record["results"]["parity"]
    assert parity["reference"]["trajectory_csv"] == str(baseline_csv.resolve())
    assert parity["reference"]["label"] == baseline_csv.stem
    assert Path(parity["figure"]).is_file()
    assert parity["threshold_status"] == "configured"
    assert parity["threshold_source"] == EXPECTED_PARITY_THRESHOLDS_SOURCE

    joint_order = record["results"]["control"]["joint_order"]
    assert set(parity["joints"]) == set(joint_order)
    for name in joint_order:
        entry = parity["joints"][name]
        assert entry["joint"] == name
        assert entry["threshold"] is not None
        assert entry["pass"]["max_abs_rad"] is True, (name, entry)
        assert entry["pass"]["rms_rad"] is True, (name, entry)
        for metric in ("max_abs_rad", "rms_rad"):
            # Same deterministic law and start pose: any divergence is numerical (e.g. GPU reduction order).
            assert entry[metric] < 1e-3, (name, metric, entry[metric])

    assert_artifacts(record)


@pytest.mark.skipif(
    os.environ.get(RUN_MUJOCO_PARITY_ENV) != "1",
    reason=(
        f"set {RUN_MUJOCO_PARITY_ENV}=1 to run the real cross-platform MuJoCo-vs-Isaac parity measurement "
        "(invokes a separate MuJoCo environment and the full sweep on both platforms); skipped by default so "
        "the routine Isaac Lab test selection stays fast and does not require that environment"
    ),
)
def test_sweep_mujoco_parity(tmp_path, description_root):
    """Run MuJoCo 04 in its own pixi env (never inside Kit) and gate Isaac 04 against it by the fixed thresholds."""
    if shutil.which("pixi") is None:
        pytest.fail("pixi is required to run the MuJoCo reference example in its own pixi environment")

    mujoco_dir = tmp_path / "mujoco-sweep-parity"
    mujoco_result = subprocess.run(
        ["pixi", "run", "-e", "mujoco", "python", str(MUJOCO_SWEEP_EXAMPLE),
         *base_flags(description_root, mujoco_dir, hand=HAND)],
        cwd=PUBLIC, capture_output=True, text=True, timeout=SIM_TIMEOUT_S,
    )
    assert mujoco_result.returncode == 0, mujoco_result.stderr
    mujoco_record = read_record(mujoco_dir)
    mujoco_csv = Path(mujoco_record["results"]["proprioception"]["csv"])
    assert mujoco_csv.is_file()

    isaac_dir = tmp_path / "isaac-sweep-parity"
    isaac_result = run_example(
        SWEEP_EXAMPLE, *base_flags(description_root, isaac_dir, hand=HAND), "--reference-trajectory", str(mujoco_csv),
        timeout=SIM_TIMEOUT_S,
    )
    assert isaac_result.returncode == 0, isaac_result.stderr

    record = read_record(isaac_dir)
    parity = record["results"]["parity"]
    assert parity["reference"]["trajectory_csv"] == str(mujoco_csv.resolve())
    assert Path(parity["figure"]).is_file()
    joint_order = record["results"]["control"]["joint_order"]
    assert set(parity["joints"]) == set(joint_order)

    summary = "; ".join(
        f"{name}: max {entry['max_abs_rad']:.6f} rad, RMS {entry['rms_rad']:.6f} rad"
        for name, entry in parity["joints"].items()
    )
    print(f"observed Isaac-vs-MuJoCo sweep q error per joint: {summary}")

    if parity["threshold_status"] != "configured":
        pytest.skip(
            f"parity thresholds not yet configured ({parity['threshold_status']}); "
            f"observed per-joint max/RMS error: {summary}"
        )
    for name, entry in parity["joints"].items():
        assert entry["pass"]["max_abs_rad"], (name, entry)
        assert entry["pass"]["rms_rad"], (name, entry)

    assert_artifacts(record)


def _load_sweep_module():
    spec = importlib.util.spec_from_file_location("sweep_joints_04_under_test", SWEEP_EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_apply_parity_thresholds_right_hand_has_no_fixed_threshold():
    """Joints absent from PARITY_THRESHOLDS read back a null threshold/pass, not a pass or a KeyError."""
    module = _load_sweep_module()
    right_joint_names = [conventions.HAND_PREFIXES["right"] + suffix for suffix in conventions.JOINT_SUFFIXES]
    joints = {name: {"joint": name, "max_abs_rad": 0.01, "rms_rad": 0.002} for name in right_joint_names}

    status, source = module.apply_parity_thresholds(joints, module.PARITY_THRESHOLDS,
                                                     module.PARITY_THRESHOLDS_SOURCE)

    assert status == "no fixed threshold for this joint set"
    assert source is None
    for name in right_joint_names:
        assert joints[name]["threshold"] is None
        assert joints[name]["pass"] is None


def test_apply_parity_thresholds_left_hand_is_configured_and_never_fabricates_a_pass():
    """A covered joint reads back the table's threshold and a pass/fail from it, not from the observed value."""
    module = _load_sweep_module()
    name = conventions.HAND_PREFIXES["left"] + conventions.JOINT_SUFFIXES[0]
    fixed = module.PARITY_THRESHOLDS[name]
    joints = {name: {"joint": name, "max_abs_rad": fixed["max_abs_rad"], "rms_rad": fixed["rms_rad"] * 2}}

    status, source = module.apply_parity_thresholds(joints, module.PARITY_THRESHOLDS,
                                                     module.PARITY_THRESHOLDS_SOURCE)

    assert status == "configured"
    assert source == module.PARITY_THRESHOLDS_SOURCE
    assert joints[name]["threshold"] == fixed
    assert joints[name]["pass"] == {"max_abs_rad": True, "rms_rad": False}


def _write_sweep_reference(path: Path, t, q) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", *(f"q:{name}" for name in JOINT_NAMES)])
        for k, time in enumerate(t):
            writer.writerow([f"{time:.9g}", *(f"{value:.9g}" for value in q[k])])
    return path


def test_sweep_reference_reader_names_the_line_and_column_of_a_missing_cell(tmp_path):
    module = _load_sweep_module()
    t = [0.0, 0.005, 0.01]
    path = _write_sweep_reference(tmp_path / "ref.csv", t, [[0.0] * len(JOINT_NAMES)] * 3)
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[3] = ",".join(lines[3].split(",")[:5])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(module.ReferenceTrajectoryError, match=f"line 4 ends before column 'q:{JOINT_NAMES[4]}'"):
        module.read_reference_trajectory(path, JOINT_NAMES, t)


def test_sweep_reference_reader_rejects_a_non_numeric_or_non_finite_position(tmp_path):
    module = _load_sweep_module()
    t = [0.0, 0.005]
    path = _write_sweep_reference(tmp_path / "text.csv", t, [[0.0] * len(JOINT_NAMES)] * 2)
    path.write_text(path.read_text(encoding="utf-8").replace("\n0.005,0,", "\n0.005,x,", 1), encoding="utf-8")
    with pytest.raises(module.ReferenceTrajectoryError, match=f"line 3 column 'q:{JOINT_NAMES[0]}': 'x' is not a number"):
        module.read_reference_trajectory(path, JOINT_NAMES, t)

    q = [[0.0] * len(JOINT_NAMES), [0.0] * (len(JOINT_NAMES) - 1) + [float("inf")]]
    path = _write_sweep_reference(tmp_path / "inf.csv", t, q)
    with pytest.raises(module.ReferenceTrajectoryError, match=f"line 3 column 'q:{JOINT_NAMES[-1]}'.*not finite"):
        module.read_reference_trajectory(path, JOINT_NAMES, t)


def test_sweep_reference_reader_holds_the_time_grid_to_an_absolute_1e6_s(tmp_path):
    module = _load_sweep_module()
    path = _write_sweep_reference(tmp_path / "ref.csv", [0.0, 1.00001], [[0.0] * len(JOINT_NAMES)] * 2)
    with pytest.raises(module.ReferenceTrajectoryError, match="time grid"):
        module.read_reference_trajectory(path, JOINT_NAMES, [0.0, 1.0])


def test_sweep_rejects_a_reference_without_headless(tmp_path, description_root):
    output_dir = tmp_path / "live-reference"
    flags = [flag for flag in base_flags(description_root, output_dir, hand=HAND) if flag != "--headless"]
    result = run_example(SWEEP_EXAMPLE, *flags, "--reference-trajectory", str(tmp_path / "ref.csv"))
    assert result.returncode == 2
    assert "--reference-trajectory needs --headless" in result.stderr
    assert not output_dir.exists()
