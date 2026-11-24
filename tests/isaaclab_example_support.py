# SPDX-License-Identifier: Apache-2.0
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

PUBLIC = Path(__file__).resolve().parent.parent


def run_example(example: Path, *flags: str, timeout: float = 180.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(example), *flags],
        cwd=PUBLIC,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def base_flags(description_root: Path, output_dir: Path, *, hand: str = "left") -> list[str]:
    return [
        "--hand", hand,
        "--asset-root", str(description_root),
        "--strict",
        "--headless",
        "--output-dir", str(output_dir),
    ]


def read_record(output_dir: Path) -> dict:
    paths = sorted(Path(output_dir).glob("*.json"))
    if len(paths) != 1:
        raise AssertionError(f"expected exactly one run record JSON in {output_dir}, found {[p.name for p in paths]}")
    return json.loads(paths[0].read_text(encoding="utf-8"))


def assert_artifacts(record: Mapping) -> None:
    artifacts = record.get("artifacts") or []
    if not artifacts:
        raise AssertionError(f"record for demo {record.get('demo')!r} registered no artifacts")
    missing = [entry["path"] for entry in artifacts if not Path(entry["path"]).is_file()]
    if missing:
        raise AssertionError(f"registered artifacts missing from disk: {missing}")


def assert_moving_trace(values: Sequence[float], *, label: str = "trace", atol: float = 1e-9) -> None:
    array = np.asarray(values, dtype=np.float64)
    non_finite = np.flatnonzero(~np.isfinite(array))
    if non_finite.size:
        raise AssertionError(f"{label} has {non_finite.size} non-finite sample(s), first at index {non_finite[0]}")
    spread = float(np.max(array) - np.min(array))
    if spread <= atol:
        raise AssertionError(f"{label} is constant (max-min={spread:.3g} <= {atol:.3g}); expected it to move")
