# SPDX-License-Identifier: Apache-2.0
import hashlib
import math
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from wuji_sim import assets

MODEL_OVERRIDES_PATH = Path(__file__).resolve().parent.parent / "model_overrides.toml"
MODEL_OVERRIDE_UNITS = {"joint_armature": "kg*m^2", "joint_torque_limit": "N*m"}
PARAM_REQUIRED_KEYS = ("value", "unit")
PARAM_KEYS = frozenset({*PARAM_REQUIRED_KEYS, "source"})
OVERRIDE_KEYS = frozenset({"value", "reason"})
TAU_LIMIT_UNIT = "N*m"
TAU_LIMIT_FILE_KEYS = frozenset({"source", "unit", "tau_limit"})


class ProfileError(ValueError):
    pass


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def is_count(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def is_numbers(count: int) -> Callable[[object], bool]:
    return lambda value: isinstance(value, (list, tuple)) and len(value) == count and all(map(is_number, value))


def is_text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


@dataclass(frozen=True)
class Schema:
    platform: str
    # None: the parameter names are model joint names, checked against the model when the profile is applied.
    groups: Mapping[str, tuple[str, ...] | None]
    # (group, parameter name; None for a joint-named group) -> (what the value must be, its check).
    rules: Mapping[tuple[str, str | None], tuple[str, Callable[[object], bool]]]


def _check_value(schema: Schema, key: str, group: str, name: str, value, where: str) -> None:
    description, accepts = schema.rules[(group, name if schema.groups[group] else None)]
    if not accepts(value):
        raise ProfileError(f"{key}: value {value!r} is not {description} ({where})")


def _plain(value):
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float):
        return float(value)
    return value


def _check_param(schema: Schema, group: str, name: str, entry, group_source: str | None, where: str) -> dict:
    key = f"{group}.{name}"
    if not isinstance(entry, Mapping):
        raise ProfileError(f"{key}: expected a table with keys {list(PARAM_REQUIRED_KEYS)} ({where})")
    for field in PARAM_REQUIRED_KEYS:
        if field not in entry:
            raise ProfileError(f"{key}: missing '{field}' ({where})")
    unknown = sorted(set(entry) - PARAM_KEYS)
    if unknown:
        raise ProfileError(f"{key}: unknown key(s) {unknown}, allowed {sorted(PARAM_KEYS)} ({where})")
    for field in ("unit", "source"):
        if field in entry and not is_text(entry[field]):
            raise ProfileError(f"{key}: '{field}' must be a non-empty string ({where})")
    _check_value(schema, key, group, name, entry["value"], where)
    return {"source": group_source, **entry}


def check_overrides(schema: Schema, overrides, params: dict, where: str) -> list[tuple[str, str, dict]]:
    if not isinstance(overrides, Mapping):
        raise ProfileError(f"overrides: expected a table of groups ({where})")
    checked = []
    for group, table in overrides.items():
        if group not in params:
            raise ProfileError(f"overrides.{group}: unknown group, allowed {list(params)} ({where})")
        if not isinstance(table, Mapping):
            raise ProfileError(f"overrides.{group}: expected a table of parameters ({where})")
        for name, entry in table.items():
            key = f"overrides.{group}.{name}"
            if name not in params[group]:
                raise ProfileError(f"{key}: target {group}.{name} does not exist in the profile ({where})")
            if not isinstance(entry, Mapping) or set(entry) != OVERRIDE_KEYS:
                got = sorted(entry) if isinstance(entry, Mapping) else type(entry).__name__
                raise ProfileError(f"{key}: keys must be exactly {sorted(OVERRIDE_KEYS)}, got {got} ({where})")
            if not (isinstance(entry["reason"], str) and entry["reason"].strip()):
                raise ProfileError(f"{key}: 'reason' must be a non-empty string ({where})")
            _check_value(schema, key, group, name, entry["value"], where)
            checked.append((group, name, {"value": _plain(entry["value"]), "reason": entry["reason"]}))
    return checked


def load_profile(path: Path, schema: Schema) -> dict:
    path = Path(path).resolve()
    where = f"profile {path}"
    try:
        raw = path.read_bytes()
        doc = tomllib.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ProfileError(f"cannot load {where}: {exc}") from exc
    unknown = sorted(set(doc) - set(schema.groups) - {"overrides"})
    if unknown:
        raise ProfileError(f"unknown group(s) {unknown}, allowed {[*schema.groups, 'overrides']} ({where})")
    params = {}
    for group, fields in schema.groups.items():
        if group not in doc:
            raise ProfileError(f"{group}: missing group ({where})")
        table = doc[group]
        if not isinstance(table, Mapping):
            raise ProfileError(f"{group}: expected a table of parameters ({where})")
        entries = {name: entry for name, entry in table.items() if name != "source"}
        if "source" in table:
            if not is_text(table["source"]):
                raise ProfileError(f"{group}: 'source' must be a non-empty string ({where})")
        else:
            # The group's source stands in for every parameter that sets none of its own.
            relying = [
                name for name, entry in entries.items() if not (isinstance(entry, Mapping) and "source" in entry)
            ]
            if relying:
                raise ProfileError(f"{group}: missing 'source' for parameter(s) {relying} that set none ({where})")
        if fields is not None:
            for name in fields:
                if name not in entries:
                    raise ProfileError(f"{group}.{name}: missing parameter ({where})")
            for name in entries:
                if name not in fields:
                    raise ProfileError(f"{group}.{name}: unknown parameter, allowed {list(fields)} ({where})")
        params[group] = {
            name: _check_param(schema, group, name, entry, table.get("source"), where)
            for name, entry in entries.items()
        }
    overrides = doc.get("overrides", {})
    check_overrides(schema, overrides, params, where)
    return {
        "platform": schema.platform,
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "params": params,
        "overrides": overrides,
    }


def load_tau_limits(path: Path) -> dict:
    path = Path(path).resolve()
    where = f"tau limit file {path}"
    try:
        raw = path.read_bytes()
        document = tomllib.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ProfileError(f"cannot load tau limit file {path}: {exc}") from exc
    missing, unknown = sorted(TAU_LIMIT_FILE_KEYS - set(document)), sorted(set(document) - TAU_LIMIT_FILE_KEYS)
    if missing or unknown:
        raise ProfileError(
            f"top-level keys must be exactly {sorted(TAU_LIMIT_FILE_KEYS)}: missing {missing}, unknown {unknown} "
            f"({where})"
        )
    source = document["source"]
    if not is_text(source):
        raise ProfileError(f"'source' must be a non-empty string ({where})")
    if document["unit"] != TAU_LIMIT_UNIT:
        raise ProfileError(f"'unit' must be {TAU_LIMIT_UNIT!r}, got {document['unit']!r} ({where})")
    table = document["tau_limit"]
    if not (isinstance(table, Mapping) and table):
        raise ProfileError(f"'tau_limit' must be a non-empty table of joint name = limit ({where})")
    for name, value in table.items():
        if not (is_number(value) and value > 0):
            raise ProfileError(f"tau_limit.{name}: value {value!r} is not a positive number ({where})")
    return {
        "file": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "source": source,
        "unit": TAU_LIMIT_UNIT,
        "values": {name: float(value) for name, value in table.items()},
    }


def load_model_overrides(path: Path, expected_version: str | None) -> dict:
    path = Path(path).resolve()
    where = f"model override file {path}"
    try:
        raw = path.read_bytes()
        document = tomllib.loads(raw.decode("utf-8"))
        with assets.EXPECTED_PATH.open("rb") as stream:
            expected = tomllib.load(stream)["hand2_2"]["description_version"]
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, KeyError) as exc:
        raise ProfileError(f"cannot load {where}: {exc}") from exc
    expected = expected if expected_version is None else expected_version
    if document.get("product") != "hand2_2" or document.get("description_version") != expected:
        raise ProfileError(
            f"{where}: product must be 'hand2_2' and description_version must match loaded asset {expected!r}"
        )
    unknown = sorted(set(document) - {"product", "description_version", *MODEL_OVERRIDE_UNITS})
    if unknown:
        raise ProfileError(f"{where}: unknown top-level keys {unknown}")
    groups = {}
    for group, unit in MODEL_OVERRIDE_UNITS.items():
        table = document.get(group, {})
        if not isinstance(table, Mapping) or not is_text(table.get("source")):
            raise ProfileError(f"{where}: {group} requires a non-empty 'source'")
        values = {}
        for name, entry in table.items():
            if name == "source":
                continue
            if not isinstance(entry, Mapping) or set(entry) != {"value", "unit"}:
                raise ProfileError(f"{where}: {group}.{name} must have exactly value and unit")
            if entry["unit"] != unit:
                raise ProfileError(f"{where}: {group}.{name} unit must be {unit!r}, got {entry['unit']!r}")
            value = entry["value"]
            valid = is_number(value) and (value >= 0 if group == "joint_armature" else value > 0)
            if not valid:
                requirement = "non-negative" if group == "joint_armature" else "positive"
                raise ProfileError(f"{where}: {group}.{name} value must be {requirement}")
            values[name] = {"value": float(value), "unit": unit}
        groups[group] = {"source": table["source"], "values": values}
    return {"groups": groups}
