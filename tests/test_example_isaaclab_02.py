# SPDX-License-Identifier: Apache-2.0
import hashlib
from pathlib import Path

import pytest

pytest.importorskip("isaaclab")

import imageio.v3 as iio
from isaaclab_example_support import assert_artifacts, assert_moving_trace, base_flags, read_record, run_example

from wuji_sim import assets, conventions, framing, scene

pytestmark = pytest.mark.isaac
PUBLIC = Path(__file__).resolve().parents[1]
PRESET_POSES_EXAMPLE = PUBLIC / "examples" / "isaaclab" / "02_preset_poses.py"
# index_finger_mcp_flex differs between every two presets, so its trace shows whether the rendered poses differ.
TRACE_JOINT = conventions.HAND_PREFIXES["left"] + "index_finger_mcp_flex"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_preset_poses_writes_camera_and_preview_images(tmp_path, description_root):
    output_dir = tmp_path / "poses"
    result = run_example(PRESET_POSES_EXAMPLE, *base_flags(description_root, output_dir))
    assert result.returncode == 0, result.stderr

    record = read_record(output_dir)
    assert record["demo"] == "02_preset_poses"
    cameras = [*scene.CAMERAS, scene.PREVIEW_CAMERA]
    assert record["results"]["scene"]["cameras"] == cameras
    view = framing.view(description_root / assets.MJCF_ENTRIES["left"], "left", "isaaclab", "tracking")
    assert record["results"]["scene"]["preview"] == view.record()

    poses = record["results"]["poses"]
    views = record["results"]["views"]
    assert set(poses) == set(scene.POSES)
    assert set(views) == set(scene.POSES)
    # poses are the measured post-step states the PNGs show; pose_targets are the requested preset vectors.
    pose_targets = record["results"]["pose_targets"]
    assert set(pose_targets) == set(scene.POSES)
    for pose in scene.POSES:
        names = list(pose_targets[pose])
        assert list(poses[pose]) == names
        assert list(pose_targets[pose].values()) == scene.pose_vector(names, pose).tolist()

    png_count = 0
    width, height = view.render_size
    for pose in scene.POSES:
        assert set(views[pose]) == set(cameras)
        for camera in cameras:
            path = views[pose][camera]
            assert Path(path).is_file(), path
            png_count += 1
        assert iio.imread(views[pose][scene.PREVIEW_CAMERA]).shape == (height, width, 3)
    assert png_count == 16
    assert len(list(output_dir.glob("*.png"))) == 16

    # Catches a loop that kept rendering "open"; the shape checks above would not.
    distinct_vectors = {tuple(sorted(poses[pose].items())) for pose in scene.POSES}
    assert len(distinct_vectors) > 1, f"all preset poses produced the same joint vector: {poses}"

    assert_moving_trace([poses[pose][TRACE_JOINT] for pose in scene.POSES], label=TRACE_JOINT)

    # A stale RTX render product reuses the first pose's frame; only per-camera PNG hashes catch that.
    for camera in cameras:
        camera_hashes = {sha256(views[pose][camera]) for pose in scene.POSES}
        assert len(camera_hashes) > 1, f"camera {camera!r} produced the same image hash for every pose: {camera_hashes}"

    assert_artifacts(record)


def test_preset_poses_missing_asset_root_exits_nonzero(tmp_path):
    output_dir = tmp_path / "missing"
    missing_root = tmp_path / "does-not-exist"
    result = run_example(PRESET_POSES_EXAMPLE, *base_flags(missing_root, output_dir))
    assert result.returncode != 0
    assert result.stderr.strip()
    assert not output_dir.exists() or not list(output_dir.glob("*.json"))
