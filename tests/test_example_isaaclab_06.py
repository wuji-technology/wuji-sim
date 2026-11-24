# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

pytest.importorskip("isaaclab")

import imageio.v3 as iio
import pen_spin
from isaaclab_example_support import assert_artifacts, base_flags, read_record, run_example

from wuji_sim import assets, framing, scene

pytestmark = pytest.mark.isaac
PUBLIC = Path(__file__).resolve().parents[1]
PEN_SPIN_EXAMPLE = PUBLIC / "examples" / "isaaclab" / "06_pen_spin.py"
HAND = "right"
GROUP = "01"
CLIP = "clip_001"
CLIP_STEPS = 369
REFERENCE_DISTANCE_LIMIT_M = 0.1
SIM_TIMEOUT_S = 900.0


def test_pen_spin_replays_a_clip_with_the_pen_held_near_the_reference(tmp_path, description_root):
    archive = pen_spin.default_cache_dir() / f"{pen_spin.RELEASE_NAME}.zip"
    if not archive.is_file():
        pytest.skip(f"{archive} is not cached; run example 06 once to download it")
    output_dir = tmp_path / "pen_spin"
    result = run_example(PEN_SPIN_EXAMPLE, *base_flags(description_root, output_dir, hand=HAND),
                         "--group", GROUP, "--clip", CLIP, timeout=SIM_TIMEOUT_S)
    assert result.returncode == 0, result.stderr

    record = read_record(output_dir)
    assert record["demo"] == "06_pen_spin"
    view = framing.view(description_root / assets.MJCF_ENTRIES[HAND], HAND, "isaaclab", "pen_spin")
    assert record["results"]["scene"]["preview"] == view.record()
    replay = record["results"]["replay"]
    assert replay["steps"] == CLIP_STEPS
    assert replay["reference_distance_max_m"] < REFERENCE_DISTANCE_LIMIT_M
    assert record["results"]["policy"]["inference"]["session_runs"] == CLIP_STEPS

    video = record["results"]["video"]
    assert video["camera"] == scene.PREVIEW_CAMERA
    assert video["frames"] == CLIP_STEPS // video["control_steps_per_frame"] + 1
    width, height = view.render_size
    assert iio.imread(video["path"], index=0).shape == (height, width, 3)

    assert_artifacts(record)
