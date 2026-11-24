# SPDX-License-Identifier: Apache-2.0
import csv
import itertools
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("isaaclab")

import imageio.v3 as iio
from isaaclab_example_support import assert_artifacts, base_flags, read_record, run_example

from wuji_sim import assets, framing, scene

pytestmark = pytest.mark.isaac
PUBLIC = Path(__file__).resolve().parents[1]
PARALLEL_ENVS_EXAMPLE = PUBLIC / "examples" / "isaaclab" / "05_parallel_envs.py"
HAND = "right"
NUM_ENVS = 4
ENV_SPACING_M = 0.3
MOTION_FRACTION = 0.5
MOTION_EPSILON_RAD = 1e-3
SIM_TIMEOUT_S = 600.0  # Kit startup plus one cycle with a path-traced camera exceeds run_example's default timeout


def _read_batch_csv(path: Path) -> dict[int, dict[str, list[float]]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        joint_columns = [name for name in reader.fieldnames if name not in ("t_s", "env")]
        traces: dict[int, dict[str, list[float]]] = {}
        for row in reader:
            env = int(row["env"])
            env_trace = traces.setdefault(env, {name: [] for name in joint_columns})
            for name in joint_columns:
                env_trace[name].append(float(row[name]))
        return traces


def test_parallel_envs_runs_four_independent_environments(tmp_path, description_root):
    output_dir = tmp_path / "parallel"
    result = run_example(PARALLEL_ENVS_EXAMPLE, *base_flags(description_root, output_dir, hand=HAND),
                         "--num-envs", str(NUM_ENVS), timeout=SIM_TIMEOUT_S)
    assert result.returncode == 0, result.stderr

    record = read_record(output_dir)
    assert record["demo"] == "05_parallel_envs"
    view = framing.view(description_root / assets.MJCF_ENTRIES[HAND], HAND, "isaaclab", "grid", worlds=NUM_ENVS)
    assert record["results"]["scene"]["preview"] == view.record()
    batch = record["results"]["batch"]
    assert batch["num_envs"] == NUM_ENVS
    assert len(batch["envs"]) == NUM_ENVS
    assert batch["env_spacing_m"] == ENV_SPACING_M
    roots = [tuple(position) for position in batch["root_pos_w_m"]]
    assert len(roots) == NUM_ENVS
    nearest = [min(math.dist(root, other) for j, other in enumerate(roots) if j != i) for i, root in enumerate(roots)]
    assert nearest == pytest.approx([ENV_SPACING_M] * NUM_ENVS, abs=1e-4)
    # The preview camera frames the hands where framing.view lays them out; the env roots must sit there.
    assert np.abs(np.asarray(roots) - np.asarray(view.hand_roots)).max() < 1e-3
    assert batch["root_error_max_m"] < 1e-3
    assert [entry["start_pose"] for entry in batch["envs"]] == list(scene.POSES)
    assert (batch["move_s"], batch["hold_s"]) == (scene.MOVE_S, scene.HOLD_S)
    assert batch["steps"] == round(scene.parallel_duration() / record["results"]["control"]["period_s"])

    video = record["results"]["video"]
    assert video["camera"] == scene.PREVIEW_CAMERA
    assert video["worlds"] == NUM_ENVS
    assert video["frames"] == batch["steps"] // video["steps_per_frame"] + 1
    width, height = view.render_size
    assert iio.imread(video["path"], index=0).shape == (height, width, 3)

    csv_paths = list(output_dir.glob("*.csv"))
    assert len(csv_paths) == 1
    traces = _read_batch_csv(csv_paths[0])
    assert set(traces) == set(range(NUM_ENVS))

    # A run that simulated one env and copied its rows would pass every check above; distinct final vectors do not.
    final_vectors = {env: tuple(round(values[-1], 6) for values in joints.values()) for env, joints in traces.items()}
    assert len(set(final_vectors.values())) == NUM_ENVS, (
        f"environments did not converge to {NUM_ENVS} distinct joint vectors: {final_vectors}"
    )

    # One parallel_duration() cycle takes every env through every preset pose.
    joint_names = record["results"]["batch"]["joint_order"]
    pose_delta_rad = max(
        float(np.abs(scene.pose_vector(joint_names, a) - scene.pose_vector(joint_names, b)).max())
        for a, b in itertools.combinations(scene.POSES, 2)
    )
    for entry in batch["envs"]:
        env = entry["env"]
        joints = traces[env]
        # Same as the example's trace_range_max_rad: the largest single-joint excursion, not a cross-joint span.
        trace_range_rad = max(max(values) - min(values) for values in joints.values())
        assert trace_range_rad == pytest.approx(entry["trace_range_max_rad"], abs=1e-6), (
            f"env{env} CSV trace range {trace_range_rad:.6g} does not match reported "
            f"trace_range_max_rad {entry['trace_range_max_rad']:.6g}"
        )
        assert entry["visited_pose_delta_max_rad"] == pytest.approx(pose_delta_rad, abs=1e-9)
        assert entry["visited_pose_delta_max_rad"] > MOTION_EPSILON_RAD
        required = MOTION_FRACTION * entry["visited_pose_delta_max_rad"]
        assert trace_range_rad >= required, (
            f"env{env} q moved only {trace_range_rad:.3g} rad; expected >= {required:.3g} rad "
            f"({MOTION_FRACTION:g} * visited_pose_delta_max_rad={entry['visited_pose_delta_max_rad']:.3g})"
        )

    assert_artifacts(record)


def test_parallel_envs_rejects_non_positive_num_envs(tmp_path, description_root):
    output_dir = tmp_path / "invalid"
    result = run_example(PARALLEL_ENVS_EXAMPLE, *base_flags(description_root, output_dir), "--num-envs", "0")
    assert result.returncode != 0
    assert result.stderr.strip()
    assert not output_dir.exists() or not list(output_dir.glob("*.json"))
