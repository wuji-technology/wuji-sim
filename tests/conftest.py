# SPDX-License-Identifier: Apache-2.0
import os
import sys
from pathlib import Path

import pytest

from wuji_sim import assets


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA device and the mujoco-warp pixi environment")
    config.addinivalue_line("markers", "isaac: needs the isaaclab pixi environment and a GPU (Isaac Sim)")


def pytest_sessionfinish(session, exitstatus):
    session.config._isaac_exit_status = int(exitstatus)


def pytest_unconfigure(config):
    stack = getattr(config, "_isaac_kit_stack", None)
    if stack is None:
        return
    status = getattr(config, "_isaac_exit_status", 1)
    sys.stdout.flush()
    sys.stderr.flush()
    stack.__exit__(SystemExit, SystemExit(status), None)
    os._exit(status)


@pytest.fixture(scope="session")
def description_root() -> Path:
    try:
        asset = assets.resolve("right")
    except assets.AssetError as exc:
        pytest.fail(str(exc))
    return asset.root
