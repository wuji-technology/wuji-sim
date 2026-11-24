# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import imageio.v3
import pytest

pytest.importorskip("mujoco_warp")

from wuji_sim import assets, conventions, framing, scene

pytestmark = pytest.mark.gpu
PUBLIC = Path(__file__).resolve().parents[1]
EXAMPLE = PUBLIC / "examples" / "mujoco_warp" / "02_preset_poses.py"
HAND = "left"
# index_finger_mcp_flex differs between every two presets, so it shows whether the rendered poses differ.
TRACE_JOINT = conventions.HAND_PREFIXES[HAND] + "index_finger_mcp_flex"


def base_flags(description_root: Path, output_dir: Path) -> list[str]:
    return ["--hand", HAND, "--asset-root", str(description_root), "--strict", "--headless", "--output-dir", str(output_dir)]


def run_example(*flags: str, timeout: float = 120.0) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(EXAMPLE), *flags], cwd=PUBLIC, capture_output=True, text=True,
                          timeout=timeout)


def read_record(output_dir: Path) -> dict:
    [path] = output_dir.glob("*.json")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_preset_poses_writes_every_camera_and_the_preview_per_pose(tmp_path, description_root):
    output_dir = tmp_path / "poses"
    result = run_example(*base_flags(description_root, output_dir))
    assert result.returncode == 0, result.stderr

    record = read_record(output_dir)
    assert record["demo"] == "02_preset_poses"
    cameras = [*scene.CAMERAS, scene.PREVIEW_CAMERA]
    assert record["results"]["scene"]["cameras"] == cameras
    view = framing.view(description_root / assets.MJCF_ENTRIES[HAND], HAND, "mujoco_warp", "tracking")
    assert record["results"]["scene"]["preview"] == view.record()

    poses = record["results"]["poses"]
    views = record["results"]["views"]
    assert set(poses) == set(scene.POSES)
    assert set(views) == set(scene.POSES)

    width, height = view.render_size
    png_count = 0
    for pose in scene.POSES:
        assert set(views[pose]) == set(cameras)
        for camera in cameras:
            path = views[pose][camera]
            assert Path(path).is_file(), path
            png_count += 1
        assert imageio.v3.imread(views[pose][scene.PREVIEW_CAMERA]).shape == (height, width, 3)
    assert png_count == 16
    assert len(list(output_dir.glob("*.png"))) == 16

    # A loop that kept rendering one pose would pass the shape assertions above; compare the joint vectors.
    distinct_vectors = {tuple(sorted(poses[pose].items())) for pose in scene.POSES}
    assert len(distinct_vectors) > 1, f"all preset poses produced the same joint vector: {poses}"

    trace = [poses[pose][TRACE_JOINT] for pose in scene.POSES]
    assert max(trace) - min(trace) > 1e-6, f"{TRACE_JOINT} trace is constant across poses: {trace}"

    # A stale render reusing the first pose's frame would pass the path/shape assertions above; hash the PNGs.
    for camera in cameras:
        camera_hashes = {sha256(views[pose][camera]) for pose in scene.POSES}
        assert len(camera_hashes) > 1, f"camera {camera!r} produced the same image hash for every pose: {camera_hashes}"

    artifacts = record["artifacts"]
    assert len(artifacts) == 16
    for entry in artifacts:
        assert Path(entry["path"]).is_file()


def test_preset_poses_missing_asset_root_exits_nonzero(tmp_path):
    output_dir = tmp_path / "missing"
    missing_root = tmp_path / "does-not-exist"
    result = run_example(*base_flags(missing_root, output_dir))
    assert result.returncode != 0
    assert result.stderr.strip()
    assert not output_dir.exists() or not list(output_dir.glob("*.json"))
