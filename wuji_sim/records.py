# SPDX-License-Identifier: Apache-2.0
import argparse
import contextlib
import datetime
import hashlib
import importlib.metadata
import itertools
import json
import os
import platform as _platform
import re
import stat
import subprocess
import tempfile
import weakref
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import wuji_sim

PRODUCT = "hand2_2"
PLATFORMS = ("mujoco", "mujoco_warp", "isaaclab")
HANDS = ("left", "right")
REQUIRED_DISTRIBUTIONS = {
    "mujoco": ("mujoco", "numpy"),
    "mujoco_warp": ("mujoco", "mujoco-warp", "warp-lang", "numpy"),
    "isaaclab": ("isaaclab", "isaacsim", "torch", "numpy"),
}
OPTIONAL_DISTRIBUTIONS = ("imageio", "imageio-ffmpeg", "matplotlib", "onnxruntime")
RECORD_FORMAT = "wuji-sim-run-record/1"
STEM_TIME_FORMAT = "%Y%m%d-%H%M%S"
LOAD_REPORT_FIELDS = ("profile", "overrides", "option", "effective_values", "backend_imposed",
                      "backend_checked", "import_options")

_DEMO_RE = re.compile(r"^\d\d_[a-z0-9_]+$")
# A label starts with a letter, so "<stem>-<label>" never reads as "<stem>-<n>", the stem of a later run.
_LABEL_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_SUFFIX_RE = re.compile(r"^\.[a-z0-9]+$")
_HERE = Path(__file__).resolve().parent


class RecordError(RuntimeError):
    pass


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="directory for the run record and artifacts (default: artifacts/runs-<YYYYMMDD> at "
                             "the root of the git checkout, or in the current directory outside a git checkout)")


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(_HERE), *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def default_output_dir(day: datetime.date) -> Path:
    try:
        top = Path(_git("rev-parse", "--show-toplevel"))
    except (OSError, subprocess.CalledProcessError):
        # No git, or this package is not inside a git checkout (for example an unpacked release archive).
        top = Path.cwd()
    return top / "artifacts" / f"runs-{day:%Y%m%d}"


def _sim_commit() -> dict:
    try:
        head = _git("rev-parse", "HEAD")
        dirty = bool(_git("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        return {"head": None, "dirty": None, "error": f"{exc} {stderr.strip()}".strip()}
    return {"head": head, "dirty": dirty, "error": None}


def _backend_versions(platform: str) -> dict:
    versions = {"python": _platform.python_version(), "wuji-sim": wuji_sim.__version__}
    missing = []
    for dist in REQUIRED_DISTRIBUTIONS[platform]:
        try:
            versions[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            missing.append(dist)
    if missing:
        raise RecordError(f"required distributions for platform {platform!r} are not installed: {missing}")
    for dist in OPTIONAL_DISTRIBUTIONS:
        try:
            versions[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def run_stem(*, platform: str, demo: str, hand: str, created: datetime.datetime) -> str:
    if platform not in PLATFORMS:
        raise RecordError(f"platform must be one of {PLATFORMS}, got {platform!r}")
    if hand not in HANDS:
        raise RecordError(f"hand must be one of {HANDS}, got {hand!r}")
    if not _DEMO_RE.match(demo):
        raise RecordError(f"demo must match {_DEMO_RE.pattern}, got {demo!r}")
    return f"{PRODUCT}-{hand}-{platform}-{demo}-{created.strftime(STEM_TIME_FORMAT)}"


def _release(path: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        if path.stat().st_size == 0:
            path.unlink()


def reserve_record(owner: object, output_dir: Path, base: str) -> Path:
    # Exclusive creation claims the stem, so concurrent runs get base, base-2, ... and never share a file.
    for n in itertools.count(1):
        stem = base if n == 1 else f"{base}-{n}"
        path = output_dir / f"{stem}.json"
        try:
            path.touch(exist_ok=False)
        except FileExistsError:
            continue
        # Artifacts left by a run that never wrote its record still hold the stem.
        if any(output_dir.glob(f"{stem}-[a-z]*")):
            path.unlink()
            continue
        # A killed or os._exit-ed process (as Isaac Sim's shutdown) leaves the empty file, still holding the stem.
        weakref.finalize(owner, _release, path)
        return path


def fill_record(path: Path, text: str) -> None:
    reserved = path.stat()
    # Anything but the empty file reserve_record created is a written record, which is never overwritten.
    if reserved.st_size:
        raise RecordError(f"{path} already holds a written record")
    # The record appears only by rename: a failed write leaves the reservation empty, so the run can retry.
    # The leading dot keeps the temporary file out of the stem's "<stem>-*" and "*.json" names.
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".part")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as f:
            f.write(text)
        # mkstemp creates the file 0600; the record keeps the umask mode of the reservation.
        os.chmod(temporary, stat.S_IMODE(reserved.st_mode))
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class VideoWriter:
    def __init__(self, path: Path, fps: int):
        import imageio.v2

        self.path = path
        self.frames = 0
        # macro_block_size 1 avoids rescaling; quality 8 (crf 10) stays within 1 grey level on average, 7 does not.
        self._writer = imageio.v2.get_writer(str(path), fps=fps, codec="libx264", macro_block_size=1,
                                             pixelformat="yuv420p", quality=8)

    def append(self, rgb: np.ndarray) -> None:
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"video frames must be uint8 HxWx3, got {rgb.dtype} {rgb.shape}")
        # yuv420p needs even sides; ffmpeg fails on odd ones without imageio raising, leaving an empty video.
        if rgb.shape[0] % 2 or rgb.shape[1] % 2:
            raise ValueError(f"video frames need an even width and height, got {rgb.shape[1]}x{rgb.shape[0]}")
        self._writer.append_data(rgb)
        self.frames += 1

    def close(self) -> None:
        self._writer.close()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class RunRecord:
    def __init__(self, *, platform: str, demo: str, hand: str, output_dir: Path | None,
                 argv: Sequence[str], now: datetime.datetime | None = None):
        self.created = (now or datetime.datetime.now()).astimezone()
        base = run_stem(platform=platform, demo=demo, hand=hand, created=self.created)
        self.platform = platform
        self.demo = demo
        self.hand = hand
        self.argv = [str(arg) for arg in argv]
        self.output_dir = (Path(output_dir) if output_dir is not None
                           else default_output_dir(self.created.date())).absolute()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.record_path = reserve_record(self, self.output_dir, base)
        self.stem = self.record_path.stem
        self._artifacts: dict[tuple[str, str], Path] = {}

    def artifact_path(self, label: str, suffix: str) -> Path:
        if not _LABEL_RE.match(label):
            raise RecordError(f"artifact label must match {_LABEL_RE.pattern}, got {label!r}")
        if not _SUFFIX_RE.match(suffix):
            raise RecordError(f"artifact suffix must match {_SUFFIX_RE.pattern}, got {suffix!r}")
        if (label, suffix) in self._artifacts:
            raise RecordError(f"artifact {label}{suffix} is already registered for run {self.stem}")
        path = self.output_dir / f"{self.stem}-{label}{suffix}"
        self._artifacts[(label, suffix)] = path
        return path

    def write_png(self, label: str, rgb: np.ndarray) -> Path:
        import imageio.v3

        path = self.artifact_path(label, ".png")
        imageio.v3.imwrite(path, rgb, extension=".png")
        return path

    def open_video(self, label: str, fps: int) -> VideoWriter:
        return VideoWriter(self.artifact_path(label, ".mp4"), fps)

    def save_figure(self, label: str, fig) -> Path:
        path = self.artifact_path(label, ".png")
        fig.savefig(path, format="png")
        return path

    def write(self, *, asset: dict, load_report: dict, results: dict, rendering: str | None) -> Path:
        self._check_artifacts()
        missing_fields = [key for key in (*LOAD_REPORT_FIELDS, "labels") if key not in load_report]
        if missing_fields:
            raise RecordError(f"load_report is missing {missing_fields}")

        labels = list(dict.fromkeys([*asset["labels"], *load_report["labels"]]))
        return self._dump({
            **self._identity(),
            "asset": asset,
            **{key: load_report[key] for key in LOAD_REPORT_FIELDS},
            "labels": labels,
            "rendering": None if rendering is None else {"pipeline": rendering,
                                                         "MUJOCO_GL": os.environ.get("MUJOCO_GL")},
            "results": results,
            "artifacts": self._artifact_entries(),
        })


    def _check_artifacts(self) -> None:
        missing_artifacts = [str(p) for p in self._artifacts.values() if not p.is_file()]
        if missing_artifacts:
            raise RecordError(f"registered artifacts were not written: {missing_artifacts}")

    def _identity(self) -> dict:
        return {
            "record_format": RECORD_FORMAT,
            "created": self.created.isoformat(),
            "product": PRODUCT,
            "hand": self.hand,
            "platform": self.platform,
            "demo": self.demo,
            "command": self.argv,
            "cwd": os.getcwd(),
            "environment": {"pixi_environment": os.environ.get("PIXI_ENVIRONMENT_NAME"),
                            "MUJOCO_GL": os.environ.get("MUJOCO_GL")},
            "sim_commit": _sim_commit(),
            "backend_versions": _backend_versions(self.platform),
        }

    def _artifact_entries(self) -> list[dict]:
        return [{"path": str(p.resolve()), "bytes": p.stat().st_size, "sha256": _sha256(p)}
                for p in self._artifacts.values()]

    def _dump(self, record: dict) -> Path:
        try:
            text = json.dumps(record, indent=2, allow_nan=False, default=_json_default)
        except (TypeError, ValueError) as exc:
            raise RecordError(f"run record for {self.stem} is not valid JSON: {exc}") from exc
        fill_record(self.record_path, text + "\n")
        return self.record_path
