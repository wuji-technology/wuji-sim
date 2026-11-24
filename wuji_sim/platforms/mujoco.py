# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import mujoco

from wuji_sim import profiles
from wuji_sim.platforms import _mjcf

PLATFORM = "mujoco"
PROFILE_PATH = Path(__file__).resolve().parent.parent / "profiles" / "mujoco.toml"
SCHEMA = profiles.Schema(platform=PLATFORM, groups=_mjcf.PROFILE_GROUPS, rules=_mjcf.VALUE_RULES)


def load_profile(path: Path = PROFILE_PATH) -> dict:
    return profiles.load_profile(path, SCHEMA)


def load(
    mjcf: Path,
    *,
    scene: Callable[[mujoco.MjSpec], None] | None = None,
    overrides: Mapping | None = None,
    profile_path: Path = PROFILE_PATH,
    tau_limits: Mapping | None = None,
    model_overrides_path: Path = profiles.MODEL_OVERRIDES_PATH,
    description_version: str | None = None,
) -> _mjcf.Loaded:
    build = _mjcf.build(
        mjcf,
        SCHEMA,
        scene=scene,
        overrides=overrides,
        profile_path=profile_path,
        tau_limits=tau_limits,
        model_overrides_path=model_overrides_path,
        description_version=description_version,
    )
    report = _mjcf.report(build, PLATFORM, _mjcf.host_arrays(build.model))
    return _mjcf.Loaded(**build.loaded_fields(), report=report)


def mit_stability(loaded: _mjcf.Loaded, kp: Sequence[float], kd: Sequence[float]) -> dict:
    return _mjcf.mit_stability(loaded, kp, kd)


def set_mit_gains(loaded: _mjcf.Loaded, kp: Sequence[float], kd: Sequence[float]) -> None:
    _mjcf.set_host_gains(loaded, kp, kd)
