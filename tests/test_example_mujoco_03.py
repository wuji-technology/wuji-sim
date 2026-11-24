# SPDX-License-Identifier: Apache-2.0
import csv
import json
import subprocess
import sys
from pathlib import Path

import imageio.v2

from wuji_sim import assets, conventions, framing, scene

PUBLIC = Path(__file__).resolve().parents[1]
EXAMPLE = PUBLIC / "examples" / "mujoco" / "03_mit_tracking.py"
HAND = "left"
JOINT_NAMES = tuple(conventions.HAND_PREFIXES[HAND] + suffix for suffix in conventions.JOINT_SUFFIXES)


def base_flags(description_root: Path, output_dir: Path) -> list[str]:
    return ["--hand", HAND, "--asset-root", str(description_root), "--strict", "--headless", "--output-dir", str(output_dir)]


def run_example(*flags: str, timeout: float = 120.0) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(EXAMPLE), *flags], cwd=PUBLIC, capture_output=True, text=True,
                          timeout=timeout)


def read_record(output_dir: Path) -> dict:
    [path] = output_dir.glob("*.json")
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_header_and_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def test_default_run_writes_only_the_preview_video_and_no_trajectory_field(tmp_path, description_root):
    output_dir = tmp_path / "mit-default"
    result = run_example(*base_flags(description_root, output_dir))
    assert result.returncode == 0, result.stderr

    record = read_record(output_dir)
    assert "trajectory" not in record["results"]
    artifacts = record["artifacts"]
    assert len(artifacts) == 1
    for entry in artifacts:
        assert Path(entry["path"]).is_file()
    video_path = Path(artifacts[0]["path"])
    assert video_path.suffix == ".mp4"
    assert video_path.stem.endswith(f"-{scene.PREVIEW_CAMERA}")

    view = framing.view(description_root / assets.MJCF_ENTRIES[HAND], HAND, "mujoco", "tracking")
    assert record["results"]["scene"]["preview"] == view.record()
    video = record["results"]["video"]
    assert (video["path"], video["camera"], video["fps"]) == (str(video_path), scene.PREVIEW_CAMERA, 25)
    reader = imageio.v2.get_reader(str(video_path))
    try:
        shapes = [frame.shape for frame in reader]
    finally:
        reader.close()
    assert shapes == [(view.render_size[1], view.render_size[0], 3)] * video["frames"]


def test_write_trajectory_flag_adds_an_isaac_compatible_reference_csv(tmp_path, description_root):
    output_dir = tmp_path / "mit-trajectory"
    result = run_example(*base_flags(description_root, output_dir), "--write-trajectory")
    assert result.returncode == 0, result.stderr

    record = read_record(output_dir)
    joint_order = record["results"]["control"]["joint_order"]
    assert joint_order == list(JOINT_NAMES)

    trajectory = record["results"]["trajectory"]
    csv_path = Path(trajectory["csv"])
    assert csv_path.is_file()
    assert trajectory["rows"] == record["results"]["tracking"]["steps"]
    assert trajectory["columns_note"]

    header, rows = read_csv_header_and_rows(csv_path)
    expected_header = ["t_s", *(f"q_des:{name}" for name in joint_order), *(f"q:{name}" for name in joint_order)]
    assert header == expected_header
    assert len(rows) == trajectory["rows"]
    for row in (*rows[:3], rows[-1]):
        values = [float(cell) for cell in row]
        assert len(values) == len(expected_header)

    # A frozen q column would pass every shape/header assertion above; reject a constant trace.
    trace_joint = JOINT_NAMES[0]
    q_values = [float(row[header.index(f"q:{trace_joint}")]) for row in rows]
    assert max(q_values) - min(q_values) > 1e-6, f"{trace_joint} q trace is constant in the written CSV"

    artifacts = record["artifacts"]
    assert len(artifacts) == 2
    for entry in artifacts:
        assert Path(entry["path"]).is_file()
    assert any(Path(entry["path"]) == csv_path for entry in artifacts)
