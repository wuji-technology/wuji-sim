# SPDX-License-Identifier: Apache-2.0
import argparse
import datetime
import errno
import gc
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path

import imageio.v2
import imageio.v3
import numpy as np
import pytest
from matplotlib.figure import Figure

import wuji_sim
from wuji_sim import records
from wuji_sim.records import RecordError, RunRecord

NOW = datetime.datetime(2026, 11, 10, 23, 30, 0)
STEM = "hand2_2-right-mujoco-01_load_and_view-20261110-233000"
TOP_LEVEL_FIELDS = {
    "record_format", "created", "product", "hand", "platform", "demo", "command", "cwd", "environment",
    "sim_commit", "backend_versions", "asset", "profile", "overrides", "option", "effective_values",
    "backend_imposed", "backend_checked", "import_options", "labels", "rendering", "results", "artifacts",
}


def make_run(output_dir: Path | None, **kwargs) -> RunRecord:
    params = dict(platform="mujoco", demo="01_load_and_view", hand="right", output_dir=output_dir,
                  argv=["examples/mujoco/01_load_and_view.py", "--hand", "right"], now=NOW)
    params.update(kwargs)
    return RunRecord(**params)


def asset_record(labels=()):
    return {"product": "hand2_2", "hand": "right", "digest": "a" * 64, "labels": list(labels)}


def load_report(labels=("with override: position actuators replaced by motors (pending hand-model fix)",)):
    return {
        "platform": "mujoco",
        "mjcf": "/x/mjcf/right.xml",
        "profile": {"platform": "mujoco", "file": "/x/mujoco.toml", "sha256": "b" * 64},
        "labels": list(labels),
        "overrides": [{"kind": "actuator_replacement"}],
        "option": {"timestep": {"unit": "s", "model_source": 0.002, "profile": 0.002, "effective": 0.002}},
        "effective_values": {"units": {}},
        "backend_imposed": [],
        "backend_checked": [],
        "import_options": {},
    }


def write_record(run: RunRecord, **kwargs) -> dict:
    params = dict(asset=asset_record(), load_report=load_report(), results={}, rendering=None)
    params.update(kwargs)
    path = run.write(**params)
    assert path == run.record_path
    return json.loads(path.read_text(encoding="utf-8"))


def test_stem_and_paths(tmp_path):
    run = make_run(tmp_path)
    assert run.stem == STEM
    assert run.output_dir == tmp_path
    assert run.record_path == tmp_path / f"{STEM}.json"
    assert run.artifact_path("front", ".png") == tmp_path / f"{STEM}-front.png"


def test_output_dir_is_created(tmp_path):
    target = tmp_path / "a" / "b"
    run = make_run(target)
    assert target.is_dir()
    assert run.output_dir == target


def test_relative_output_dir_is_made_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = make_run(Path("rel") / "runs")
    assert run.output_dir == tmp_path / "rel" / "runs"
    assert run.output_dir.is_absolute()
    assert run.record_path.is_absolute()
    assert run.artifact_path("front", ".png").is_absolute()


@pytest.mark.parametrize(
    "kwargs",
    [{"platform": "isaac"}, {"hand": "both"}, {"demo": "load"}, {"demo": "1_load"}, {"demo": "01-load"},
     {"demo": "01_Load"}],
)
def test_invalid_identity_rejected(tmp_path, kwargs):
    with pytest.raises(RecordError):
        make_run(tmp_path, **kwargs)


def test_existing_records_move_the_run_to_the_next_free_stem(tmp_path):
    (tmp_path / f"{STEM}.json").write_text("{}")
    (tmp_path / f"{STEM}-2.json").write_text("{}")
    run = make_run(tmp_path)
    assert run.stem == f"{STEM}-3"
    assert run.record_path == tmp_path / f"{STEM}-3.json"
    assert run.artifact_path("front", ".png") == tmp_path / f"{STEM}-3-front.png"
    assert (tmp_path / f"{STEM}.json").read_text() == (tmp_path / f"{STEM}-2.json").read_text() == "{}"


def test_artifacts_left_without_a_record_move_the_run_to_the_next_free_stem(tmp_path):
    leftover = tmp_path / f"{STEM}-front.png"
    leftover.write_bytes(b"left by a run that stopped early")
    run = make_run(tmp_path)
    assert run.stem == f"{STEM}-2"
    assert not (tmp_path / f"{STEM}.json").exists()
    assert leftover.read_bytes() == b"left by a run that stopped early"


def test_other_runs_do_not_change_the_stem(tmp_path):
    (tmp_path / "hand2_2-left-mujoco-01_load_and_view-20261110-233000.json").write_text("{}")
    (tmp_path / "hand2_2-right-mujoco-01_load_and_view-20261110-232959-front.png").write_bytes(b"")
    (tmp_path / "hand2_2-right-mujoco-01_load_and_view_more-20261110-233000.json").write_text("{}")
    (tmp_path / "hand2_2-right-mujoco-01_load_and_view-20261110.json").write_text("{}")
    (tmp_path / "hand2_2-right-mujoco-01_load_and_view-20261110-front.png").write_bytes(b"")
    (tmp_path / f"{STEM}-2.json").write_text("{}")
    assert make_run(tmp_path).stem == STEM


def test_runs_started_in_the_same_second_write_separate_files(tmp_path):
    first, second = make_run(tmp_path), make_run(tmp_path)
    assert (first.stem, second.stem) == (STEM, f"{STEM}-2")
    black = first.write_png("front", np.zeros((4, 4, 3), dtype=np.uint8))
    white = second.write_png("front", np.full((4, 4, 3), 255, dtype=np.uint8))
    assert black != white
    assert write_record(first)["artifacts"][0]["path"] == str(black.resolve())
    assert write_record(second)["artifacts"][0]["path"] == str(white.resolve())
    assert imageio.v3.imread(black).max() == 0 and imageio.v3.imread(white).min() == 255


def test_artifact_path_rejects_duplicate_label(tmp_path):
    run = make_run(tmp_path)
    run.artifact_path("front", ".png")
    run.artifact_path("front", ".mp4")
    with pytest.raises(RecordError):
        run.artifact_path("front", ".png")


@pytest.mark.parametrize("label", ["Front", "front-view", "", "a/b", "2", "2nd"])
def test_artifact_path_rejects_bad_label(tmp_path, label):
    run = make_run(tmp_path)
    with pytest.raises(RecordError):
        run.artifact_path(label, ".png")


def test_write_contains_all_top_level_fields(tmp_path):
    run = make_run(tmp_path)
    record = write_record(run, rendering="MuJoCo physics / MuJoCo rendering")
    assert set(record) == TOP_LEVEL_FIELDS
    assert record["record_format"] == "wuji-sim-run-record/1"
    assert datetime.datetime.fromisoformat(record["created"]).utcoffset() is not None
    assert (record["product"], record["hand"], record["platform"], record["demo"]) == (
        "hand2_2", "right", "mujoco", "01_load_and_view")
    assert record["command"] == ["examples/mujoco/01_load_and_view.py", "--hand", "right"]
    assert set(record["environment"]) == {"pixi_environment", "MUJOCO_GL"}
    assert set(record["sim_commit"]) == {"head", "dirty", "error"}
    assert record["sim_commit"]["error"] is None and len(record["sim_commit"]["head"]) == 40
    assert {"python", "mujoco", "numpy"} <= set(record["backend_versions"])
    assert record["backend_versions"]["wuji-sim"] == wuji_sim.__version__
    assert record["asset"] == asset_record()
    for key in ("profile", "overrides", "option", "effective_values", "backend_imposed", "backend_checked",
                "import_options"):
        assert record[key] == load_report()[key]
    assert record["rendering"]["pipeline"] == "MuJoCo physics / MuJoCo rendering"
    assert "MUJOCO_GL" in record["rendering"]
    assert record["artifacts"] == []


def test_version_is_the_pyproject_version():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    assert wuji_sim.__version__ == tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]


def test_write_without_rendering(tmp_path):
    record = write_record(make_run(tmp_path), rendering=None)
    assert record["rendering"] is None


def test_labels_are_ordered_and_deduplicated(tmp_path):
    record = write_record(
        make_run(tmp_path),
        asset=asset_record(["assets modified"]),
        load_report=load_report(["replaced", "assets modified", "with override: x.y", "replaced"]),
    )
    assert record["labels"] == ["assets modified", "replaced", "with override: x.y"]


def test_artifact_hashes_and_sizes(tmp_path):
    run = make_run(tmp_path)
    path = run.artifact_path("data", ".bin")
    payload = bytes(range(256)) * 3
    path.write_bytes(payload)
    record = write_record(run)
    assert record["artifacts"] == [
        {"path": str(path.resolve()), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    ]


def test_registered_but_missing_artifact_is_an_error(tmp_path):
    run = make_run(tmp_path)
    run.artifact_path("front", ".png")
    with pytest.raises(RecordError):
        run.write(asset=asset_record(), load_report=load_report(), results={}, rendering=None)
    assert run.record_path.read_bytes() == b""


def test_numpy_values_serialize(tmp_path):
    results = {"scalar": np.float32(0.5), "int": np.int64(3), "array": np.arange(3, dtype=np.float64),
               "bool": np.bool_(True)}
    record = write_record(make_run(tmp_path), results=results)
    assert record["results"] == {"scalar": 0.5, "int": 3, "array": [0.0, 1.0, 2.0], "bool": True}


@pytest.mark.parametrize("value", [float("nan"), np.float32("nan"), np.array([1.0, np.inf])])
def test_non_finite_values_are_an_error(tmp_path, value):
    run = make_run(tmp_path)
    with pytest.raises(RecordError):
        run.write(asset=asset_record(), load_report=load_report(), results={"x": value}, rendering=None)
    assert run.record_path.read_bytes() == b""


def test_a_written_record_is_never_overwritten(tmp_path):
    run = make_run(tmp_path)
    write_record(run)
    text = run.record_path.read_text(encoding="utf-8")
    with pytest.raises(RecordError, match="already holds a written record"):
        run.write(asset=asset_record(), load_report=load_report(), results={"again": True}, rendering=None)
    assert run.record_path.read_text(encoding="utf-8") == text


class _DiskFullHalfway:
    def __init__(self, f):
        self._f = f

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self._f.close()

    def write(self, text):
        self._f.write(text[:len(text) // 2])
        self._f.flush()
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))


def _replace_on_full_disk(src, dst):
    raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))


@pytest.mark.parametrize("failing", ["write", "replace"])
def test_a_failed_record_write_leaves_the_run_retryable(tmp_path, monkeypatch, failing):
    run = make_run(tmp_path)
    mode = run.record_path.stat().st_mode
    if failing == "write":
        fdopen = os.fdopen
        monkeypatch.setattr(os, "fdopen", lambda *args, **kwargs: _DiskFullHalfway(fdopen(*args, **kwargs)))
    else:
        monkeypatch.setattr(os, "replace", _replace_on_full_disk)
    with pytest.raises(OSError, match=os.strerror(errno.ENOSPC)):
        write_record(run)
    assert run.record_path.read_bytes() == b""
    assert list(tmp_path.iterdir()) == [run.record_path]

    monkeypatch.undo()
    assert write_record(run)["demo"] == "01_load_and_view"
    assert list(tmp_path.iterdir()) == [run.record_path]
    assert run.record_path.stat().st_mode == mode
    with pytest.raises(RecordError, match="already holds a written record"):
        write_record(run)


def test_run_stem_matches_record_stem(tmp_path):
    stem = records.run_stem(platform="mujoco", demo="01_load_and_view", hand="right", created=NOW)
    assert stem == STEM
    assert make_run(tmp_path).stem == stem


def test_missing_required_distribution_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setitem(records.REQUIRED_DISTRIBUTIONS, "mujoco", ("mujoco", "wuji-sim-no-such-distribution"))
    run = make_run(tmp_path)
    with pytest.raises(RecordError, match="wuji-sim-no-such-distribution"):
        run.write(asset=asset_record(), load_report=load_report(), results={}, rendering=None)


@pytest.mark.parametrize(
    "kwargs", [{"platform": "isaac"}, {"hand": "both"}, {"demo": "load_and_view"}]
)
def test_run_stem_rejects_invalid_identity(kwargs):
    params = dict(platform="mujoco", demo="01_load_and_view", hand="left", created=NOW)
    params.update(kwargs)
    with pytest.raises(RecordError):
        records.run_stem(**params)



def test_missing_load_report_key_is_an_error(tmp_path):
    report = load_report()
    del report["backend_checked"]
    run = make_run(tmp_path)
    with pytest.raises(RecordError, match="backend_checked"):
        run.write(asset=asset_record(), load_report=report, results={}, rendering=None)


def test_default_output_dir_in_this_repo():
    top = next(parent for parent in Path(records.__file__).resolve().parent.parents if (parent / ".git").exists())
    expected = top / "artifacts" / "runs-20261110"
    existed = expected.exists()
    assert records.default_output_dir(datetime.date(2026, 11, 10)) == expected
    assert expected.exists() == existed


@pytest.mark.parametrize("env", [{"GIT_DIR": "not-a-repo"}, {"PATH": ""}], ids=["outside_a_checkout", "without_git"])
def test_without_git_outputs_go_to_the_current_directory(tmp_path, monkeypatch, env):
    monkeypatch.chdir(tmp_path)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert records.default_output_dir(datetime.date(2026, 11, 10)) == tmp_path / "artifacts" / "runs-20261110"
    run = make_run(None)
    assert run.record_path == tmp_path / "artifacts" / "runs-20261110" / f"{STEM}.json"
    commit = write_record(run)["sim_commit"]
    assert commit["head"] is None and commit["dirty"] is None
    assert commit["error"]


def test_cli_args():
    parser = argparse.ArgumentParser()
    records.add_cli_args(parser)
    assert parser.parse_args([]).output_dir is None
    assert parser.parse_args(["--output-dir", "/x"]).output_dir == Path("/x")


def test_write_png_is_readable(tmp_path):
    run = make_run(tmp_path)
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    rgb[:, :32] = 255
    path = run.write_png("front", rgb)
    assert path == tmp_path / f"{STEM}-front.png"
    np.testing.assert_array_equal(imageio.v3.imread(path), rgb)
    record = write_record(run)
    assert record["artifacts"][0]["path"] == str(path.resolve())


def test_open_video_is_readable(tmp_path):
    run = make_run(tmp_path)
    with run.open_video("tracking", fps=25) as video:
        for i in range(10):
            video.append(np.full((48, 64, 3), i * 20, dtype=np.uint8))
    path = tmp_path / f"{STEM}-tracking.mp4"
    reader = imageio.v2.get_reader(str(path))
    try:
        assert sum(1 for _ in reader) == 10
    finally:
        reader.close()
    record = write_record(run)
    assert record["artifacts"][0]["path"] == str(path.resolve())


def test_open_video_keeps_the_frame_size(tmp_path):
    run = make_run(tmp_path)
    with run.open_video("preview", fps=25) as video:
        for i in range(3):
            video.append(np.full((50, 70, 3), 40 * i, dtype=np.uint8))
    reader = imageio.v2.get_reader(str(tmp_path / f"{STEM}-preview.mp4"))
    try:
        assert [frame.shape for frame in reader] == [(50, 70, 3)] * 3
    finally:
        reader.close()


def test_open_video_rejects_bad_frames(tmp_path):
    run = make_run(tmp_path)
    with run.open_video("tracking", fps=25) as video:
        with pytest.raises(ValueError):
            video.append(np.zeros((48, 64, 3), dtype=np.float32))
        with pytest.raises(ValueError):
            video.append(np.zeros((48, 64), dtype=np.uint8))
        with pytest.raises(ValueError, match="even"):
            video.append(np.zeros((49, 64, 3), dtype=np.uint8))
        with pytest.raises(ValueError, match="even"):
            video.append(np.zeros((48, 65, 3), dtype=np.uint8))
        video.append(np.zeros((48, 64, 3), dtype=np.uint8))


def test_save_figure(tmp_path):
    run = make_run(tmp_path)
    fig = Figure(figsize=(2, 2))
    fig.add_subplot().plot([0, 1], [0, 1])
    path = run.save_figure("sweep", fig)
    assert path == tmp_path / f"{STEM}-sweep.png"
    assert imageio.v3.imread(path).ndim == 3


def test_a_run_that_writes_no_record_releases_its_stem(tmp_path):
    unwritten, written = make_run(tmp_path), make_run(tmp_path)
    write_record(written)
    unwritten_path, written_path = unwritten.record_path, written.record_path
    assert unwritten_path.read_bytes() == b""
    del unwritten, written
    gc.collect()
    assert not unwritten_path.exists()
    assert json.loads(written_path.read_text(encoding="utf-8"))["demo"] == "01_load_and_view"
    assert make_run(tmp_path).stem == STEM


RESERVE_IN_CHILD = """
import datetime, sys
from pathlib import Path
from wuji_sim.records import RunRecord
print("ready", flush=True)
sys.stdin.readline()
run = RunRecord(platform="mujoco", demo="01_load_and_view", hand="right", output_dir=Path(sys.argv[1]), argv=[],
                now=datetime.datetime.fromisoformat(sys.argv[2]))
print(run.stem, flush=True)
sys.stdin.readline()
"""


HANDSHAKE_TIMEOUT_S = 60


def _read_lines(stream, count: int) -> queue.Queue:
    # A daemon thread does the blocking reads, so the test can give up on a stalled child; "" marks EOF.
    lines = queue.Queue()

    def read():
        for _ in range(count):
            line = stream.readline()
            lines.put(line)
            if not line:
                return

    threading.Thread(target=read, daemon=True).start()
    return lines


def _handshake(line_queues: list[queue.Queue], phase: str) -> list[str]:
    deadline = time.monotonic() + HANDSHAKE_TIMEOUT_S
    received = []
    for index, lines in enumerate(line_queues):
        try:
            received.append(lines.get(timeout=max(0.0, deadline - time.monotonic())))
        except queue.Empty:
            pytest.fail(f"child {index} printed no {phase} line within {HANDSHAKE_TIMEOUT_S} s")
    return received


def test_processes_started_in_the_same_second_get_distinct_stems(tmp_path):
    children, line_queues = [], []
    try:
        for _ in range(8):
            child = subprocess.Popen([sys.executable, "-c", RESERVE_IN_CHILD, str(tmp_path), NOW.isoformat()],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            children.append(child)
            # The readers stop after the stem line, so communicate() below is the only reader left.
            line_queues.append(_read_lines(child.stdout, 2))
        assert _handshake(line_queues, "ready") == ["ready\n"] * 8
        # Every child holds its reservation until all have printed a stem, so all eight race for the same names.
        for child in children:
            child.stdin.write("\n")
            child.stdin.flush()
        stems = [line.strip() for line in _handshake(line_queues, "stem")]
        assert sorted(stems) == sorted([STEM, *(f"{STEM}-{n}" for n in range(2, 9))])
        assert sorted(path.name for path in tmp_path.iterdir()) == sorted(f"{stem}.json" for stem in stems)
        for child in children:
            child.communicate("\n", timeout=60)
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait()
    assert [child.returncode for child in children] == [0] * 8
    assert list(tmp_path.iterdir()) == []


def test_example_runs_outside_a_git_checkout(tmp_path, description_root):
    public = Path(records.__file__).resolve().parents[1]
    copy = tmp_path / "unpacked"
    shutil.copytree(public / "wuji_sim", copy / "wuji_sim", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(public / "examples" / "mujoco", copy / "examples" / "mujoco")
    shutil.copy(public / "assets-expected.toml", copy)
    work = tmp_path / "work"
    work.mkdir()
    env = {**os.environ, "PYTHONPATH": str(copy), "GIT_CEILING_DIRECTORIES": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, str(copy / "examples" / "mujoco" / "01_load_and_view.py"), "--hand", "right",
         "--asset-root", str(description_root), "--strict", "--headless"],
        cwd=work, env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    [path] = [Path(line.removeprefix("record: ")) for line in result.stdout.splitlines() if line.startswith("record: ")]
    record = json.loads(path.read_text(encoding="utf-8"))
    created = datetime.datetime.fromisoformat(record["created"])
    assert path.parent == work / "artifacts" / f"runs-{created:%Y%m%d}"
    assert path.name == f"hand2_2-right-mujoco-01_load_and_view-{created:%Y%m%d-%H%M%S}.json"
    assert Path(record["profile"]["file"]).is_relative_to(copy)
    assert record["sim_commit"]["head"] is None and record["sim_commit"]["error"]
    assert {Path(entry["path"]).parent for entry in record["artifacts"]} == {path.parent}
