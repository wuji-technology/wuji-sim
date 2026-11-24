# SPDX-License-Identifier: Apache-2.0
import ast
import sys
from pathlib import Path, PurePosixPath

import pytest

PLATFORMS_DIR = Path(__file__).resolve().parent.parent / "wuji_sim" / "platforms"
SIMULATOR_IMPORTS = {
    "mujoco": {"mujoco"},
    "mujoco_warp": {"mujoco", "mujoco_warp", "warp"},
    "isaaclab": {"isaaclab", "isaaclab_physx", "isaacsim", "omni", "pxr", "torch", "carb"},
    "_mjcf": {"mujoco"},
}
# Imported as modules (`from wuji_sim import profiles`) so no shared name is reachable through a platform module.
SHARED_IMPORTS = {
    "mujoco": {"wuji_sim.mit", "wuji_sim.profiles", "wuji_sim.platforms._mjcf"},
    "mujoco_warp": {"wuji_sim.mit", "wuji_sim.profiles", "wuji_sim.platforms._mjcf"},
    "isaaclab": {"wuji_sim.conventions", "wuji_sim.mit", "wuji_sim.profiles"},
    "_mjcf": {"wuji_sim.mit", "wuji_sim.profiles"},
}
FORBIDDEN_IMPORTS = {"numpy"}
MODULE_FILES = sorted(path for path in PLATFORMS_DIR.glob("*.py") if path.name != "__init__.py")
PLATFORM_FILES = [path for path in MODULE_FILES if not path.name.startswith("_")]


def import_violations(source, simulator, shared):
    permitted = (set(simulator) | set(sys.stdlib_module_names)) - FORBIDDEN_IMPORTS
    violations = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                violations.append(f"line {node.lineno}: relative import")
                continue
            if node.module.split(".")[0] == "wuji_sim":
                modules = [f"{node.module}.{alias.name}" for alias in node.names]
            else:
                modules = [node.module]
        elif isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        else:
            continue
        for module in modules:
            if module.split(".")[0] == "wuji_sim":
                allowed = module in shared
            else:
                allowed = module.split(".")[0] in permitted
            if not allowed:
                violations.append(f"line {node.lineno}: import {module}")
    return violations


def profile_path_constants(source):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if any(isinstance(target, ast.Name) and target.id == "PROFILE_PATH" for target in targets):
            return [n.value for n in ast.walk(node.value) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    return None


def test_the_platforms_are_the_three_simulators():
    assert [path.stem for path in PLATFORM_FILES] == ["isaaclab", "mujoco", "mujoco_warp"]


def test_every_module_is_registered():
    assert [path.name for path in MODULE_FILES if path.stem not in SIMULATOR_IMPORTS] == []


@pytest.mark.parametrize("path", MODULE_FILES, ids=lambda path: path.name)
def test_module_imports_only_stdlib_its_simulator_and_the_shared_modules(path):
    assert path.stem in SIMULATOR_IMPORTS, f"{path.name} is not a registered module"
    source = path.read_text(encoding="utf-8")
    assert import_violations(source, SIMULATOR_IMPORTS[path.stem], SHARED_IMPORTS[path.stem]) == []


@pytest.mark.parametrize("path", PLATFORM_FILES, ids=lambda path: path.name)
def test_profile_path_names_the_same_toml(path):
    constants = profile_path_constants(path.read_text(encoding="utf-8"))
    assert constants is not None, f"{path.name} has no module-level PROFILE_PATH assignment"
    assert f"{path.stem}.toml" in [PurePosixPath(c).name for c in constants], constants


@pytest.mark.parametrize(
    "line, violation",
    [
        ("import numpy as np", "import numpy"),
        ("from numpy import ndarray", "import numpy"),
        ("from wuji_sim import scene", "import wuji_sim.scene"),
        ("import wuji_sim.framing", "import wuji_sim.framing"),
        ("from wuji_sim.profiles import ProfileError", "import wuji_sim.profiles.ProfileError"),
        ("from wuji_sim.platforms import mujoco_warp", "import wuji_sim.platforms.mujoco_warp"),
        ("from . import mujoco_warp", "relative import"),
        ("from .mujoco import load", "relative import"),
        ("import mujoco_warp", "import mujoco_warp"),
        ("import warp as wp", "import warp"),
        ("import torch", "import torch"),
    ],
)
def test_checker_flags_forbidden_imports(line, violation):
    source = f"import json\nimport mujoco\n\n\ndef f():\n    {line}\n"
    found = import_violations(source, SIMULATOR_IMPORTS["mujoco"], SHARED_IMPORTS["mujoco"])
    assert found == [f"line 6: {violation}"]


def test_checker_accepts_stdlib_simulator_and_shared_imports():
    source = (
        "import tomllib\nfrom pathlib import Path\nimport mujoco\nimport mujoco.viewer\nimport warp\n"
        "from wuji_sim import mit, profiles\nfrom wuji_sim.platforms import _mjcf\n"
    )
    assert import_violations(source, SIMULATOR_IMPORTS["mujoco_warp"], SHARED_IMPORTS["mujoco_warp"]) == []
