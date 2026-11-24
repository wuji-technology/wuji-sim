# SPDX-License-Identifier: Apache-2.0
"""Each platform's default (non-headless) example run writes only the JSON run record, no PNG/MP4/CSV."""
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from wuji_sim import assets, framing

PUBLIC = Path(__file__).resolve().parents[1]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_only_json_record(output_dir: Path) -> dict:
    written = sorted(p for p in output_dir.iterdir() if p.is_file())
    assert [p.suffix for p in written] == [".json"], written
    record = json.loads(written[0].read_text(encoding="utf-8"))
    assert record["artifacts"] == []
    return record


def test_mujoco_default_run_writes_only_the_json_record(tmp_path, description_root):
    module = _load_module(PUBLIC / "examples" / "mujoco" / "01_load_and_view.py", "mujoco_01_default_live_test")
    output_dir = tmp_path / "live"
    fake_viewer = MagicMock()
    fake_viewer.is_running.return_value = False
    with patch("wuji_sim.scene._require_display", lambda: None), \
         patch("mujoco.viewer.launch_passive", return_value=fake_viewer) as launch:
        module.main([
            "01_load_and_view.py", "--hand", "left", "--asset-root", str(description_root), "--strict",
            "--output-dir", str(output_dir),
        ])
    record = _assert_only_json_record(output_dir)
    fake_viewer.is_running.assert_called()
    assert launch.call_args.kwargs == {"show_left_ui": True, "show_right_ui": True}
    view = framing.view(description_root / assets.MJCF_ENTRIES["left"], "left", "mujoco", "tracking")
    assert record["results"]["scene"]["preview"] == view.record()
    fake_viewer.cam.lookat.__setitem__.assert_called_once_with(slice(None), view.camera_target)
    assert (fake_viewer.cam.distance, fake_viewer.cam.azimuth, fake_viewer.cam.elevation) == (
        view.distance_m, view.azimuth_deg, view.elevation_deg
    )


@pytest.mark.gpu
def test_mujoco_warp_default_run_writes_only_the_json_record(tmp_path, description_root):
    pytest.importorskip("mujoco_warp")
    module = _load_module(
        PUBLIC / "examples" / "mujoco_warp" / "01_load_and_step.py", "mujoco_warp_01_default_live_test"
    )
    output_dir = tmp_path / "live"
    fake_viewer = MagicMock()
    fake_viewer.is_running.return_value = False
    with patch("wuji_sim.scene._require_display", lambda: None), \
         patch("mujoco.viewer.launch_passive", return_value=fake_viewer) as launch:
        module.main([
            "--hand", "left", "--asset-root", str(description_root), "--strict", "--output-dir", str(output_dir),
        ])
    _assert_only_json_record(output_dir)
    fake_viewer.is_running.assert_called()
    assert launch.call_args.kwargs == {"show_left_ui": True, "show_right_ui": True}


@pytest.mark.isaac
def test_isaaclab_default_run_writes_only_the_json_record(tmp_path, description_root):
    pytest.importorskip("isaaclab")
    example = PUBLIC / "examples" / "isaaclab" / "01_load_and_view.py"
    output_dir = tmp_path / "live"
    # Kit segfaults on in-process teardown unless the process exits, so run a subprocess on an X display (Xvfb :99).
    env = dict(os.environ)
    env.setdefault("DISPLAY", ":99")
    # The loop below never reads the child's output; a pipe would block Kit once full, a file never does.
    log_path = tmp_path / "isaaclab-01-live.log"
    with log_path.open("wb") as log:
        proc = subprocess.Popen(
            [sys.executable, str(example), "--hand", "left", "--asset-root", str(description_root), "--strict",
             "--output-dir", str(output_dir)],
            cwd=PUBLIC, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
    record_path = None
    try:
        deadline = time.time() + 120.0
        while time.time() < deadline:
            if output_dir.is_dir():
                # The record file is created empty when the run starts and filled once the record is written.
                candidates = [path for path in output_dir.glob("*.json") if path.stat().st_size]
                if candidates:
                    record_path = candidates[0]
                    break
            if proc.poll() is not None:
                break
            time.sleep(0.5)
        if record_path is not None:
            # Let the live loop start so a file written after the record would still be caught.
            time.sleep(2.0)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    if record_path is None:
        output = log_path.read_text(encoding="utf-8", errors="replace")
        raise AssertionError(f"no run record appeared within 120 s; process output:\n{output}")
    _assert_only_json_record(output_dir)
