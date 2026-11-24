# SPDX-License-Identifier: Apache-2.0
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

import imageio.v2
import imageio.v3
import numpy as np
import pytest

pytest.importorskip("mujoco_warp")

from wuji_sim import assets, framing, scene

pytestmark = pytest.mark.gpu
PUBLIC = Path(__file__).resolve().parents[1]
EXAMPLE = PUBLIC / "examples" / "mujoco_warp" / "05_multi_world.py"
HAND = "right"
NWORLD = 4


def run_example(*flags: str, timeout: float = 600.0) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(EXAMPLE), *flags], cwd=PUBLIC, capture_output=True, text=True,
                          timeout=timeout)


def read_record(output_dir: Path) -> dict:
    [path] = output_dir.glob("*.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_four_worlds_cycle_through_the_presets_in_one_preview_video(tmp_path, description_root):
    output_dir = tmp_path / "grid"
    result = run_example("--hand", HAND, "--nworld", str(NWORLD), "--asset-root", str(description_root), "--strict",
                         "--headless", "--output-dir", str(output_dir))
    assert result.returncode == 0, result.stderr

    record = read_record(output_dir)
    results = record["results"]
    view = framing.view(description_root / assets.MJCF_ENTRIES[HAND], HAND, "mujoco_warp", "grid", worlds=NWORLD)
    assert results["scene"]["preview"] == view.record()
    assert results["steps"] == scene.whole_steps(scene.parallel_duration(), results["control"]["period_s"])

    video = results["video"]
    assert (video["camera"], video["fps"], video["worlds"]) == (scene.PREVIEW_CAMERA, 25, NWORLD)
    assert video["frames"] == results["steps"] // video["steps_per_frame"] + 1
    reader = imageio.v2.get_reader(video["path"])
    try:
        shapes = [frame.shape for frame in reader]
    finally:
        reader.close()
    assert shapes == [(960, 1286, 3)] * video["frames"]

    joint_names = results["control"]["joint_order"]
    presets = {pose: scene.pose_vector(joint_names, pose) for pose in scene.POSES}
    # Within half the smallest max-joint distance between two presets, a hand is nearer its own preset than any other.
    margin = 0.5 * min(np.abs(presets[a] - presets[b]).max() for a, b in itertools.combinations(scene.POSES, 2))
    worlds = results["worlds"]
    assert [(entry["world"], entry["pose"]) for entry in worlds] == list(enumerate(scene.POSES))
    finals = [tuple(entry["warp_final_q_rad"]) for entry in worlds]
    assert len(set(finals)) == NWORLD, finals
    period = scene.MOVE_S + scene.HOLD_S
    sequence = scene.TRACKING_SEQUENCE
    for entry, final in zip(worlds, finals):
        assert np.abs(np.array(final) - presets[entry["pose"]]).max() < margin, entry
        assert entry["warp_final_error_max_rad"] < margin, entry
        # A world that never moved would sit on its start preset at every hold end and fail here.
        hold_end = entry["hold_end"]
        assert [hold["t_s"] for hold in hold_end] == pytest.approx([(k + 1) * period for k in range(len(sequence))])
        start = sequence.index(entry["pose"])
        assert [hold["pose"] for hold in hold_end] == [sequence[(start + k + 1) % len(sequence)]
                                                       for k in range(len(sequence))]
        assert hold_end[-1]["pose"] == entry["pose"]
        for hold in hold_end:
            assert hold["warp_error_max_rad"] < margin, (entry["world"], hold)
            assert math.isfinite(hold["mujoco_error_max_rad"])
        # The CPU MuJoCo replay is recorded for comparison only.
        for key in ("mujoco_final_error_max_rad", "max_abs_qpos_diff_rad", "final_max_abs_qpos_diff_rad"):
            assert math.isfinite(entry[key]), (entry["world"], key)
        assert imageio.v3.imread(entry["image"]).shape == (480, 1280, 3)

    assert len(record["artifacts"]) == NWORLD + 1
    for artifact in record["artifacts"]:
        assert Path(artifact["path"]).is_file()
