# SPDX-License-Identifier: Apache-2.0
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path

import pytest

from wuji_sim import profiles

PROFILE_FILES = sorted(Path(profiles.__file__).resolve().parent.glob("*.toml"))
SOURCE_FORMAT = re.compile(r"^(default: \S.*|choice|estimate(: \S.*)?|measured: \S.*)$")
SCHEMA = profiles.Schema(
    platform="test",
    groups={"option": ("timestep", "iterations")},
    rules={
        ("option", "timestep"): ("a positive number", lambda value: profiles.is_number(value) and value > 0),
        ("option", "iterations"): ("a positive integer", profiles.is_count),
    },
)
OWN_SOURCES = """
[option.timestep]
value = 0.005
unit = "s"
source = "choice"

[option.iterations]
value = 100
unit = "count"
source = "default: MuJoCo 3.11.0"
"""


def sources(table: Mapping, prefix: str = "") -> list[tuple[str, str]]:
    found = []
    for key, value in table.items():
        if key == "source":
            found.append((prefix or "<top>", value))
        elif isinstance(value, Mapping):
            found += sources(value, f"{prefix}.{key}" if prefix else key)
    return found


def write_profile(tmp_path, text):
    path = tmp_path / "profile.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_the_platform_profiles_are_checked():
    assert [path.name for path in PROFILE_FILES] == ["isaaclab.toml", "motrixsim.toml", "mujoco.toml", "mujoco_warp.toml"]


@pytest.mark.parametrize("path", [*PROFILE_FILES, profiles.MODEL_OVERRIDES_PATH], ids=lambda path: path.name)
def test_every_source_says_only_where_the_value_comes_from(path):
    found = sources(tomllib.loads(path.read_text(encoding="utf-8")))
    assert found
    assert [(where, text) for where, text in found if not SOURCE_FORMAT.fullmatch(text)] == []


@pytest.mark.parametrize("path", PROFILE_FILES, ids=lambda path: path.name)
def test_a_group_source_is_present_exactly_when_a_parameter_relies_on_it(path):
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    for group, table in document.items():
        if group == "overrides":
            continue
        relying = [name for name, entry in table.items() if name != "source" and "source" not in entry]
        assert ("source" in table) == bool(relying), group


def test_group_source_may_be_omitted_when_every_parameter_has_its_own(tmp_path):
    profile = profiles.load_profile(write_profile(tmp_path, OWN_SOURCES), SCHEMA)
    assert {name: entry["source"] for name, entry in profile["params"]["option"].items()} == {
        "timestep": "choice",
        "iterations": "default: MuJoCo 3.11.0",
    }


def test_group_source_is_required_when_a_parameter_has_none(tmp_path):
    text = OWN_SOURCES.replace('source = "default: MuJoCo 3.11.0"\n', "")
    message = "option: missing 'source' for parameter(s) ['iterations']"
    with pytest.raises(profiles.ProfileError, match=re.escape(message)):
        profiles.load_profile(write_profile(tmp_path, text), SCHEMA)
