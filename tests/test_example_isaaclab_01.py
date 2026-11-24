# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

pytest.importorskip("isaaclab")

import imageio.v3 as iio
from isaaclab_example_support import assert_artifacts, base_flags, read_record, run_example

from wuji_sim import assets, conventions, framing, scene

pytestmark = pytest.mark.isaac
PUBLIC = Path(__file__).resolve().parents[1]
LOAD_EXAMPLE = PUBLIC / "examples" / "isaaclab" / "01_load_and_view.py"


def test_load_and_view_writes_camera_and_preview_images(tmp_path, description_root):
    output_dir = tmp_path / "load"
    result = run_example(LOAD_EXAMPLE, *base_flags(description_root, output_dir))
    assert result.returncode == 0, result.stderr

    record = read_record(output_dir)
    assert record["demo"] == "01_load_and_view"
    assert record["results"]["scene"]["cameras"] == [*scene.CAMERAS, scene.PREVIEW_CAMERA]
    assert record["results"]["qpos"] == "zeros"
    # The images are captured after one step holding q = 0; the record keeps the state they show.
    rendered = record["results"]["q_rendered_rad"]
    assert list(rendered) == [conventions.HAND_PREFIXES["left"] + suffix for suffix in conventions.JOINT_SUFFIXES]
    assert max(abs(value) for value in rendered.values()) < 0.05, rendered
    view = framing.view(description_root / assets.MJCF_ENTRIES["left"], "left", "isaaclab", "tracking")
    assert record["results"]["scene"]["preview"] == view.record()

    views = record["results"]["views"]
    assert set(views) == {*scene.CAMERAS, scene.PREVIEW_CAMERA}
    for path in views.values():
        assert Path(path).is_file(), path
    assert len(list(output_dir.glob("*.png"))) == 4
    width, height = view.render_size
    assert iio.imread(views[scene.PREVIEW_CAMERA]).shape == (height, width, 3)

    assert_artifacts(record)


def test_load_and_view_missing_asset_root_exits_nonzero(tmp_path):
    output_dir = tmp_path / "missing"
    missing_root = tmp_path / "does-not-exist"
    result = run_example(LOAD_EXAMPLE, *base_flags(missing_root, output_dir))
    assert result.returncode != 0
    assert result.stderr.strip()
    assert not output_dir.exists() or not list(output_dir.glob("*.json"))
