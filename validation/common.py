# SPDX-License-Identifier: Apache-2.0
import argparse
import datetime
import hashlib
import json
import math
import os
import sys
import tomllib
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from wuji_sim import assets, records

SIMULATORS = ("mujoco", "mujoco_warp", "isaaclab")
REFERENCE_SIMULATOR = "mujoco"
CHECKS = ("kinematics", "self_collision", "sim_agreement")
RECORD_FORMAT = "wuji-sim-validation-record/1"
THRESHOLDS_PATH = Path(__file__).resolve().parent / "thresholds.toml"
EXIT_PASS, EXIT_FAIL, EXIT_ERROR = 0, 1, 2
ITEM_VERDICTS = ("pass", "fail", "report_only", "reference", "not_applicable")
RECORD_VERDICTS = ("pass", "fail", "reference")
VERDICT_FIELDS = frozenset({"value", "threshold", "unit", "verdict"})
CRITERION_FIELDS = frozenset({"value", "unit"})


class ValidationError(RuntimeError):
    pass


EXPECTED_ERRORS = (ValidationError, assets.AssetError, records.RecordError, FileExistsError)


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    assets.add_cli_args(parser)
    records.add_cli_args(parser)
    parser.add_argument("--simulator", required=True, choices=SIMULATORS)
    parser.add_argument("--reference", type=Path, default=None,
                        help=f"JSON record written by the --simulator {REFERENCE_SIMULATOR} run of this check, with "
                             "its data file next to it; required for the other simulators")
    parser.add_argument("--thresholds", type=Path, default=THRESHOLDS_PATH,
                        help="pass criteria (default: validation/thresholds.toml)")


def parse_args(parser: argparse.ArgumentParser, argv: Sequence[str]) -> argparse.Namespace:
    args = parser.parse_args(argv)
    if args.simulator == REFERENCE_SIMULATOR and args.reference is not None:
        parser.error(f"--reference does not apply to --simulator {REFERENCE_SIMULATOR}, which writes the reference")
    if args.simulator != REFERENCE_SIMULATOR and args.reference is None:
        parser.error(f"--simulator {args.simulator} needs --reference, the JSON record of the "
                     f"--simulator {REFERENCE_SIMULATOR} run of this check")
    return args


def load_thresholds(path: Path = THRESHOLDS_PATH) -> dict:
    path = Path(path).resolve()
    try:
        raw = path.read_bytes()
        tables = tomllib.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ValidationError(f"cannot read thresholds {path}: {exc}") from exc
    missing = [check for check in CHECKS if not isinstance(tables.get(check), dict)]
    if missing:
        raise ValidationError(f"thresholds {path}: no table for {missing}")
    _check_table(tables, path, ())
    return {"file": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "tables": tables}


def _check_table(table: Mapping, path: Path, keys: tuple[str, ...]) -> None:
    for key, value in table.items():
        name = ".".join((*keys, key))
        if isinstance(value, bool):
            continue
        if not isinstance(value, dict):
            raise ValidationError(f"thresholds {path}: {name} must be a table, a true/false flag or "
                                  f"{{ value, unit }}, got {value!r}")
        if value.keys() & CRITERION_FIELDS:
            _check_criterion(value, path, name)
        else:
            _check_table(value, path, (*keys, key))


def _check_criterion(entry: Mapping, path: Path, name: str) -> None:
    missing = sorted(CRITERION_FIELDS - entry.keys())
    if missing:
        raise ValidationError(f"thresholds {path}: {name} is missing {missing}")
    unknown = sorted(entry.keys() - CRITERION_FIELDS)
    if unknown:
        raise ValidationError(f"thresholds {path}: {name} has unknown keys {unknown}")
    value, unit = entry["value"], entry["unit"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValidationError(f"thresholds {path}: {name}.value must be a finite number, got {value!r}")
    if not isinstance(unit, str) or not unit.strip():
        raise ValidationError(f"thresholds {path}: {name}.unit must be a non-empty string, got {unit!r}")


def lookup(thresholds: Mapping, *keys: str):
    node = thresholds["tables"]
    for depth, key in enumerate(keys):
        if not isinstance(node, dict) or key not in node:
            raise ValidationError(f"thresholds {thresholds['file']} has no {'.'.join(keys[:depth + 1])}")
        node = node[key]
    return node


def criterion(thresholds: Mapping, *keys: str, unit: str) -> float:
    entry = lookup(thresholds, *keys)
    name = ".".join(keys)
    if not (isinstance(entry, dict) and entry.keys() == CRITERION_FIELDS):
        raise ValidationError(f"thresholds {thresholds['file']}: {name} is not a {{ value, unit }} criterion")
    if entry["unit"] != unit:
        raise ValidationError(f"thresholds {thresholds['file']}: {name} must be given in {unit}, "
                              f"got {entry['unit']!r}")
    return float(entry["value"])


def load_reference(path: Path, *, check: str, hand: str, asset: assets.ResolvedAsset) -> dict:
    path = Path(path).resolve()
    try:
        raw = path.read_bytes()
        record = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read reference record {path}: {exc}") from exc
    if not isinstance(record, dict):
        raise ValidationError(f"reference record {path} is not a JSON object")
    expected = {"record_format": RECORD_FORMAT, "check": check, "hand": hand, "simulator": REFERENCE_SIMULATOR}
    for key, want in expected.items():
        if record.get(key) != want:
            raise ValidationError(f"reference record {path}: {key} is {record.get(key)!r}, expected {want!r}")
    recorded_asset = record.get("asset")
    recorded = recorded_asset.get("digest") if isinstance(recorded_asset, dict) else None
    if recorded != asset.digest:
        raise ValidationError(f"reference record {path} was made with hand model assets {recorded}, "
                              f"this run uses {asset.digest}")
    data_path = data_bytes = None
    data = record.get("data")
    if data is not None:
        if not (isinstance(data, dict) and isinstance(data.get("path"), str) and isinstance(data.get("sha256"), str)):
            raise ValidationError(f"reference record {path}: data must be {{path, sha256}}, got {data!r}")
        # Only the file name counts: a record and its data file are copied between machines as a pair.
        data_path = path.parent / Path(data["path"]).name
        if not data_path.is_file():
            raise ValidationError(f"reference data {data_path} not found; it must sit next to {path}")
        try:
            data_bytes = data_path.read_bytes()
        except OSError as exc:
            raise ValidationError(f"cannot read reference data {data_path}: {exc}") from exc
        digest = hashlib.sha256(data_bytes).hexdigest()
        if digest != data["sha256"]:
            raise ValidationError(f"reference data {data_path} has sha256 {digest}; {path} lists {data['sha256']}")
    return {"record": record, "path": path, "sha256": hashlib.sha256(raw).hexdigest(), "data_path": data_path,
            "data_bytes": data_bytes}


def record_verdict(verdicts: Mapping) -> str:
    if not verdicts:
        raise ValidationError("a record needs at least one verdict item")
    for name, item in verdicts.items():
        if not isinstance(item, Mapping) or set(item) != VERDICT_FIELDS:
            raise ValidationError(f"verdict item {name!r} must have exactly the fields {sorted(VERDICT_FIELDS)}, "
                                  f"got {item!r}")
        if item["verdict"] not in ITEM_VERDICTS:
            raise ValidationError(f"verdict item {name!r} must be one of {ITEM_VERDICTS}, got {item['verdict']!r}")
    found = [item["verdict"] for item in verdicts.values()]
    if "fail" in found:
        return "fail"
    if all(verdict == "reference" for verdict in found):
        return "reference"
    return "pass"


def exit_code(verdict: str) -> int:
    if verdict not in RECORD_VERDICTS:
        raise ValidationError(f"record verdict must be one of {RECORD_VERDICTS}, got {verdict!r}")
    return EXIT_FAIL if verdict == "fail" else EXIT_PASS


def _file_entry(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": records._sha256(path)}


def _shown(value) -> str:
    return "n/a" if value is None else f"{value:.6g}" if isinstance(value, float) else str(value)


def print_result(record_path: Path, verdicts: Mapping) -> int:
    verdict = record_verdict(verdicts)
    for name, item in verdicts.items():
        print(f"{name}: {item['verdict']} (value {_shown(item['value'])}, threshold {_shown(item['threshold'])}, "
              f"unit {item['unit']})")
    print(f"verdict: {verdict}")
    print(f"record: {record_path}")
    return exit_code(verdict)


def report_error(exc: BaseException, expected: tuple[type[BaseException], ...] = EXPECTED_ERRORS) -> int:
    if isinstance(exc, expected):
        print(f"error: {exc}", file=sys.stderr)
    else:
        traceback.print_exception(exc)
    sys.stdout.flush()
    sys.stderr.flush()
    return EXIT_ERROR


def _reference_entry(reference: Mapping) -> dict:
    record = reference["record"]
    data_path = reference["data_path"]
    return {
        "path": str(reference["path"]),
        "sha256": reference["sha256"],
        "data": None if data_path is None else {"path": str(data_path), "sha256": record["data"]["sha256"]},
        "created": record.get("created"),
        "sim_commit": record.get("sim_commit"),
        "backend_versions": record.get("backend_versions"),
    }


class ValidationRecord:
    def __init__(self, *, check: str, simulator: str, hand: str, output_dir: Path | None, argv: Sequence[str],
                 now: datetime.datetime | None = None):
        if check not in CHECKS:
            raise ValidationError(f"check must be one of {CHECKS}, got {check!r}")
        if simulator not in SIMULATORS:
            raise ValidationError(f"simulator must be one of {SIMULATORS}, got {simulator!r}")
        if hand not in records.HANDS:
            raise ValidationError(f"hand must be one of {records.HANDS}, got {hand!r}")
        self.check = check
        self.simulator = simulator
        self.hand = hand
        self.created = (now or datetime.datetime.now()).astimezone()
        self.argv = [str(arg) for arg in argv]
        self.output_dir = (Path(output_dir) if output_dir is not None
                           else records.default_output_dir(self.created.date())).absolute()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = self.created.strftime(records.STEM_TIME_FORMAT)
        self.record_path = records.reserve_record(self, self.output_dir,
                                                  f"{records.PRODUCT}-{hand}-{simulator}-check_{check}-{stamp}")
        self.stem = self.record_path.stem
        self._artifacts: dict[tuple[str, str], Path] = {}
        self._data: Path | None = None

    def artifact_path(self, label: str, suffix: str, *, data: bool = False) -> Path:
        if not records._LABEL_RE.match(label):
            raise ValidationError(f"artifact label must match {records._LABEL_RE.pattern}, got {label!r}")
        if not records._SUFFIX_RE.match(suffix):
            raise ValidationError(f"artifact suffix must match {records._SUFFIX_RE.pattern}, got {suffix!r}")
        if (label, suffix) in self._artifacts:
            raise ValidationError(f"artifact {label}{suffix} is already registered for {self.stem}")
        if data and self._data is not None:
            raise ValidationError(f"{self.stem} already has the data file {self._data.name}")
        path = self.output_dir / f"{self.stem}-{label}{suffix}"
        self._artifacts[(label, suffix)] = path
        if data:
            self._data = path
        return path

    def write(self, *, asset: assets.ResolvedAsset, thresholds: Mapping, reference: Mapping | None,
              protocol: Mapping, values: Mapping, verdicts: Mapping) -> Path:
        missing = [str(path) for path in self._artifacts.values() if not path.is_file()]
        if missing:
            raise ValidationError(f"registered artifacts were not written: {missing}")
        record = {
            "record_format": RECORD_FORMAT,
            "created": self.created.isoformat(),
            "product": records.PRODUCT,
            "check": self.check,
            "simulator": self.simulator,
            "hand": self.hand,
            "command": self.argv,
            "cwd": os.getcwd(),
            "pixi_environment": os.environ.get("PIXI_ENVIRONMENT_NAME"),
            "sim_commit": records._sim_commit(),
            "backend_versions": records._backend_versions(self.simulator),
            "asset": asset.to_record(),
            "thresholds": dict(thresholds),
            "reference": None if reference is None else _reference_entry(reference),
            "protocol": protocol,
            "values": values,
            "verdicts": verdicts,
            "verdict": record_verdict(verdicts),
            "data": None if self._data is None else _file_entry(self._data),
            "artifacts": [_file_entry(path) for path in self._artifacts.values()],
        }
        try:
            text = json.dumps(record, indent=2, allow_nan=False, default=records._json_default)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"record {self.stem} is not valid JSON: {exc}") from exc
        records.fill_record(self.record_path, text + "\n")
        return self.record_path


def _quaternions(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if q.shape[-1:] != (4,):
        raise ValidationError(f"quaternions need a last axis of length 4, got shape {q.shape}")
    if not np.all(np.isfinite(q)):
        raise ValidationError("quaternions must be finite and non-zero")
    # Scaling by the largest component first keeps the norm finite for huge components and nonzero for tiny ones.
    scale = np.max(np.abs(q), axis=-1, keepdims=True)
    if not np.all(scale > 0.0):
        raise ValidationError("quaternions must be finite and non-zero")
    q = q / scale
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _quaternion_matrix(w, x, y, z) -> np.ndarray:
    return np.stack([
        np.stack([w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
        np.stack([2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)], axis=-1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z], axis=-1),
    ], axis=-2)


def quat_wxyz_to_matrix(q) -> np.ndarray:
    q = _quaternions(q)
    return _quaternion_matrix(q[..., 0], q[..., 1], q[..., 2], q[..., 3])


def quat_xyzw_to_matrix(q) -> np.ndarray:
    q = _quaternions(q)
    return _quaternion_matrix(q[..., 3], q[..., 0], q[..., 1], q[..., 2])


def rotation_angle(r_ref, r_test) -> np.ndarray:
    r_ref = np.asarray(r_ref, dtype=np.float64)
    r_test = np.asarray(r_test, dtype=np.float64)
    if r_ref.shape[-2:] != (3, 3) or r_test.shape[-2:] != (3, 3):
        raise ValidationError(f"rotations must be 3x3 matrices, got shapes {r_ref.shape} and {r_test.shape}")
    r = np.swapaxes(r_ref, -1, -2) @ r_test
    # atan2 of the sine and cosine parts stays accurate for angles near 0 and near pi, where arccos does not.
    vee = np.stack([r[..., 2, 1] - r[..., 1, 2], r[..., 0, 2] - r[..., 2, 0], r[..., 1, 0] - r[..., 0, 1]], axis=-1)
    return np.arctan2(0.5 * np.linalg.norm(vee, axis=-1), 0.5 * (np.trace(r, axis1=-2, axis2=-1) - 1.0))


def root_relative(position, rotation, root_index: int) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(position, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    if position.ndim != 3 or position.shape[2] != 3 or rotation.shape != (*position.shape, 3):
        raise ValidationError("positions (configurations, links, 3) and rotations (configurations, links, 3, 3) "
                              f"do not match: {position.shape} and {rotation.shape}")
    if not 0 <= root_index < position.shape[1]:
        raise ValidationError(f"root link index {root_index} is outside the {position.shape[1]} links")
    root_inverse = np.swapaxes(rotation[:, root_index:root_index + 1], -1, -2)
    offset = position - position[:, root_index:root_index + 1]
    return (root_inverse @ offset[..., None])[..., 0], root_inverse @ rotation


def joint_errors(q_ref, q_test) -> dict:
    q_ref = np.asarray(q_ref, dtype=np.float64)
    q_test = np.asarray(q_test, dtype=np.float64)
    if q_ref.ndim != 2 or q_ref.shape != q_test.shape or q_ref.shape[0] == 0:
        raise ValidationError(f"joint series must share one non-empty (samples, joints) shape, got {q_ref.shape} "
                              f"and {q_test.shape}")
    if not (np.all(np.isfinite(q_ref)) and np.all(np.isfinite(q_test))):
        raise ValidationError("joint series hold non-finite values")
    error = q_test - q_ref
    return {"rms": np.sqrt(np.mean(error**2, axis=0)), "max_abs": np.max(np.abs(error), axis=0)}


def judge(value: float, threshold: float) -> str:
    return "pass" if float(value) <= float(threshold) else "fail"
