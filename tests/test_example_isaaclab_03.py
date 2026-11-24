# SPDX-License-Identifier: Apache-2.0
import csv
import importlib.util
import os
import shutil
import subprocess
import tomllib
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("isaaclab")

import imageio.v3 as iio
from isaaclab_example_support import assert_artifacts, assert_moving_trace, base_flags, read_record, run_example

from wuji_sim import conventions, scene

pytestmark = pytest.mark.isaac
PUBLIC = Path(__file__).resolve().parents[1]
MIT_TRACKING_EXAMPLE = PUBLIC / "examples" / "isaaclab" / "03_mit_tracking.py"
# index_finger_mcp_flex takes four distinct angles over TRACKING_SEQUENCE, so a frozen hand reads back flat.
TRACE_JOINT = conventions.HAND_PREFIXES["left"] + "index_finger_mcp_flex"
SIM_TIMEOUT_S = 600.0  # Kit startup plus MIT tracking with a path-traced camera exceeds run_example's default timeout
# Must match PARITY_THRESHOLDS_SOURCE in 03_mit_tracking.py.
EXPECTED_PARITY_THRESHOLDS_SOURCE = (
    "independent MuJoCo vs Isaac Lab measurement, 2 deterministic runs: MuJoCo 03/05 run with its trajectory "
    "output enabled, then Isaac Lab 03/05 run with --reference-trajectory pointed at that file; threshold = "
    "observed x 2 rounded up to 2 significant figures"
)
RUN_MUJOCO_PARITY_ENV = "WUJI_RUN_MUJOCO_PARITY"
MUJOCO_MIT_TRACKING_EXAMPLE = PUBLIC / "examples" / "mujoco" / "03_mit_tracking.py"


def read_trajectory_column(csv_path: Path, joint_name: str) -> list[float]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, data_rows = rows[0], rows[1:]
    column = header.index(f"q:{joint_name}")
    return [float(row[column]) for row in data_rows]


def test_mit_tracking_writes_video_and_tracks_reference(tmp_path, description_root):
    output_dir = tmp_path / "mit"
    result = run_example(MIT_TRACKING_EXAMPLE, *base_flags(description_root, output_dir), timeout=SIM_TIMEOUT_S)
    assert result.returncode == 0, result.stderr

    record = read_record(output_dir)
    assert record["demo"] == "03_mit_tracking"
    assert record["results"]["scene"]["cameras"] == [*scene.CAMERAS, scene.PREVIEW_CAMERA]

    joint_order = record["results"]["control"]["joint_order"]
    assert joint_order
    assert record["results"]["tracking"]["reference"]["sequence"] == list(scene.TRACKING_SEQUENCE)

    tracking_joints = record["results"]["tracking"]["joints"]
    assert set(tracking_joints) == set(joint_order)
    for name in joint_order:
        entry = tracking_joints[name]
        assert {"rms_error_rad", "max_abs_error_rad", "saturation_fraction"} <= set(entry)

    video = record["results"]["video"]
    assert video["camera"] == scene.PREVIEW_CAMERA
    assert video["frames"] == record["results"]["tracking"]["steps"] // video["steps_per_frame"] + 1
    assert Path(video["path"]).is_file()
    width, height = record["results"]["scene"]["preview"]["render_size"]
    assert iio.imread(video["path"], index=0).shape == (height, width, 3)

    trajectory = record["results"]["trajectory"]
    csv_path = Path(trajectory["csv"])
    assert csv_path.is_file()
    assert trajectory["rows"] > 0

    # Shape checks above pass for a frozen hand; the recorded joint trace does not.
    assert_moving_trace(read_trajectory_column(csv_path, TRACE_JOINT), label=TRACE_JOINT)

    # The parity slot exists and stays empty until a --reference-trajectory is supplied.
    parity = record["results"]["parity"]
    assert parity["joints"] == {}
    assert parity["max_abs_rad"] is None
    assert parity["rms_rad"] is None
    assert parity["reference"] == {"trajectory_csv": None, "label": None}
    assert parity["threshold_source"] is None

    assert_artifacts(record)


def test_mit_tracking_missing_asset_root_exits_nonzero(tmp_path):
    output_dir = tmp_path / "missing"
    missing_root = tmp_path / "does-not-exist"
    result = run_example(MIT_TRACKING_EXAMPLE, *base_flags(missing_root, output_dir))
    assert result.returncode != 0
    assert result.stderr.strip()
    assert not output_dir.exists() or not list(output_dir.glob("*.json"))


def test_mit_tracking_parity_reports_per_joint_max_and_rms(tmp_path, description_root):
    # The reference is a second run of the same deterministic law; every configured threshold must pass.
    baseline_dir = tmp_path / "baseline"
    baseline = run_example(MIT_TRACKING_EXAMPLE, *base_flags(description_root, baseline_dir), timeout=SIM_TIMEOUT_S)
    assert baseline.returncode == 0, baseline.stderr
    baseline_record = read_record(baseline_dir)
    baseline_csv = Path(baseline_record["results"]["trajectory"]["csv"])
    assert baseline_csv.is_file()

    output_dir = tmp_path / "parity"
    result = run_example(
        MIT_TRACKING_EXAMPLE, *base_flags(description_root, output_dir), "--reference-trajectory", str(baseline_csv),
        timeout=SIM_TIMEOUT_S,
    )
    assert result.returncode == 0, result.stderr

    record = read_record(output_dir)
    joint_order = record["results"]["control"]["joint_order"]
    parity = record["results"]["parity"]
    assert parity["reference"]["trajectory_csv"] == str(baseline_csv.resolve())
    assert parity["reference"]["label"] == baseline_csv.stem
    assert Path(parity["figure"]).is_file()
    assert set(parity["joints"]) == set(joint_order)

    assert parity["threshold_source"] == EXPECTED_PARITY_THRESHOLDS_SOURCE
    for name in joint_order:
        entry = parity["joints"][name]
        assert entry["joint"] == name
        assert entry["threshold_formula"] == "round_up_to_2_sig_figs(2 * observed_rad)"
        assert entry["threshold"] is not None
        assert entry["threshold_source"] == EXPECTED_PARITY_THRESHOLDS_SOURCE
        assert entry["pass"]["max_abs_rad"] is True, (name, entry)
        assert entry["pass"]["rms_rad"] is True, (name, entry)
        for metric in ("max_abs_rad", "rms_rad"):
            observed = entry["observed"][metric]
            # Same deterministic law and start pose: any divergence is numerical (e.g. GPU reduction order).
            assert observed < 1e-3, (name, metric, observed)

    assert_artifacts(record)


def test_mit_tracking_parity_has_no_fixed_threshold_for_a_joint_set_the_table_does_not_cover():
    """Joints absent from PARITY_THRESHOLDS_RAD read back null threshold/pass/source, not a pass or a KeyError."""
    spec = importlib.util.spec_from_file_location("mit_tracking_03_under_test", MIT_TRACKING_EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    right_joint_names = [conventions.HAND_PREFIXES["right"] + suffix for suffix in conventions.JOINT_SUFFIXES]
    n_steps, n = 5, len(right_joint_names)
    q_isaac = np.zeros((n_steps, n))
    q_reference = np.full((n_steps, n), 0.01)

    result = module.parity_results(right_joint_names, q_isaac, q_reference, module.PARITY_THRESHOLDS_RAD,
                                    module.PARITY_THRESHOLDS_SOURCE)
    assert result["threshold_source"] is None
    for name in right_joint_names:
        entry = result["joints"][name]
        assert entry["threshold"] is None
        assert entry["threshold_source"] is None
        assert entry["pass"] is None
        assert entry["observed"]["max_abs_rad"] == pytest.approx(0.01)


def _load_tracking_module():
    spec = importlib.util.spec_from_file_location("mit_tracking_03_reader_under_test", MIT_TRACKING_EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_reference(path: Path, joint_names, t, q) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", *(f"q_des:{name}" for name in joint_names), *(f"q:{name}" for name in joint_names)])
        for k, time in enumerate(t):
            writer.writerow([f"{time:.9g}", *(f"{value + 1.0:.9g}" for value in q[k]), *(f"{value:.9g}" for value in q[k])])
    return path


LEFT_JOINTS = [conventions.HAND_PREFIXES["left"] + suffix for suffix in conventions.JOINT_SUFFIXES]


def test_reference_reader_returns_the_measured_columns(tmp_path):
    module = _load_tracking_module()
    t = np.arange(3) * 0.005
    q = np.arange(3 * len(LEFT_JOINTS), dtype=np.float64).reshape(3, -1) / 100.0
    path = _write_reference(tmp_path / "ref.csv", LEFT_JOINTS, t, q)
    assert module.read_reference_trajectory(path, LEFT_JOINTS, t) == pytest.approx(q)


def test_reference_reader_rejects_a_truncated_row(tmp_path):
    module = _load_tracking_module()
    t = np.arange(3) * 0.005
    path = _write_reference(tmp_path / "ref.csv", LEFT_JOINTS, t, np.zeros((3, len(LEFT_JOINTS))))
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[2] = ",".join(lines[2].split(",")[:2])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(module.ReferenceTrajectoryError, match="line 3 has 2 cells"):
        module.read_reference_trajectory(path, LEFT_JOINTS, t)


def test_reference_reader_rejects_a_non_finite_measured_position(tmp_path):
    module = _load_tracking_module()
    t = np.arange(3) * 0.005
    q = np.zeros((3, len(LEFT_JOINTS)))
    q[1, 4] = np.nan
    path = _write_reference(tmp_path / "ref.csv", LEFT_JOINTS, t, q)
    with pytest.raises(module.ReferenceTrajectoryError, match=f"line 3 column 'q:{LEFT_JOINTS[4]}'.*not finite"):
        module.read_reference_trajectory(path, LEFT_JOINTS, t)


def test_reference_reader_holds_the_time_grid_to_an_absolute_1e6_s(tmp_path):
    module = _load_tracking_module()
    path = _write_reference(tmp_path / "ref.csv", LEFT_JOINTS, [0.0, 1.00001], np.zeros((2, len(LEFT_JOINTS))))
    with pytest.raises(module.ReferenceTrajectoryError, match="time grid"):
        module.read_reference_trajectory(path, LEFT_JOINTS, np.array([0.0, 1.0]))


def test_mit_tracking_rejects_a_reference_without_headless(tmp_path, description_root):
    output_dir = tmp_path / "live-reference"
    flags = [flag for flag in base_flags(description_root, output_dir) if flag != "--headless"]
    result = run_example(MIT_TRACKING_EXAMPLE, *flags, "--reference-trajectory", str(tmp_path / "ref.csv"))
    assert result.returncode == 2
    assert "--reference-trajectory needs --headless" in result.stderr
    assert not output_dir.exists()


def test_mit_tracking_bad_reference_exits_1_inside_kit_before_any_artifact(tmp_path, description_root):
    # The error is raised after Kit starts, so this checks Kit shutdown keeps the handled failure's exit status.
    timestep = tomllib.loads(
        (PUBLIC / "wuji_sim" / "profiles" / "isaaclab.toml").read_text(encoding="utf-8")
    )["option"]["timestep"]["value"]
    n_steps = round(scene.tracking_duration() / timestep)
    q = np.zeros((n_steps, len(LEFT_JOINTS)))
    q[1, 0] = np.nan
    reference = _write_reference(tmp_path / "ref.csv", LEFT_JOINTS, np.arange(n_steps) * timestep, q)
    output_dir = tmp_path / "bad-reference"
    result = run_example(MIT_TRACKING_EXAMPLE, *base_flags(description_root, output_dir),
                         "--reference-trajectory", str(reference), timeout=SIM_TIMEOUT_S)
    assert result.returncode == 1, result.stderr
    assert f"line 3 column 'q:{LEFT_JOINTS[0]}'" in result.stderr, result.stderr
    written = [path.name for path in output_dir.iterdir() if path.stat().st_size] if output_dir.exists() else []
    assert written == [], written


@pytest.mark.skipif(
    os.environ.get(RUN_MUJOCO_PARITY_ENV) != "1",
    reason=(
        f"set {RUN_MUJOCO_PARITY_ENV}=1 to run the real cross-platform MuJoCo-vs-Isaac parity measurement "
        "(invokes a separate MuJoCo environment and this demo on both platforms); skipped by default so the "
        "routine Isaac Lab test selection stays fast and does not require that environment"
    ),
)
def test_mit_tracking_mujoco_parity(tmp_path, description_root):
    """Run MuJoCo 03 in its own pixi env (never inside Kit) and gate Isaac 03 against it by the fixed thresholds."""
    if shutil.which("pixi") is None:
        pytest.fail("pixi is required to run the MuJoCo reference example in its own pixi environment")

    mujoco_dir = tmp_path / "mujoco-mit-tracking-parity"
    mujoco_result = subprocess.run(
        ["pixi", "run", "-e", "mujoco", "python", str(MUJOCO_MIT_TRACKING_EXAMPLE),
         *base_flags(description_root, mujoco_dir), "--write-trajectory"],
        cwd=PUBLIC, capture_output=True, text=True, timeout=SIM_TIMEOUT_S,
    )
    assert mujoco_result.returncode == 0, mujoco_result.stderr
    mujoco_record = read_record(mujoco_dir)
    mujoco_csv = Path(mujoco_record["results"]["trajectory"]["csv"])
    assert mujoco_csv.is_file()

    isaac_dir = tmp_path / "isaac-mit-tracking-parity"
    isaac_result = run_example(
        MIT_TRACKING_EXAMPLE, *base_flags(description_root, isaac_dir), "--reference-trajectory", str(mujoco_csv),
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
        f"{name}: max {entry['observed']['max_abs_rad']:.6f} rad, RMS {entry['observed']['rms_rad']:.6f} rad"
        for name, entry in parity["joints"].items()
    )
    print(f"observed Isaac-vs-MuJoCo MIT tracking q error per joint: {summary}")

    if parity["threshold_source"] is None:
        pytest.skip(f"no fixed threshold for this run's joints; observed per-joint max/RMS error: {summary}")
    for name, entry in parity["joints"].items():
        assert entry["pass"]["max_abs_rad"], (name, entry)
        assert entry["pass"]["rms_rad"], (name, entry)

    assert_artifacts(record)
