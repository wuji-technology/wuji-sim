# SPDX-License-Identifier: Apache-2.0
import importlib.metadata


def __getattr__(name: str) -> str:
    if name == "__version__":
        return importlib.metadata.version("wuji-sim")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
