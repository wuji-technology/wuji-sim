# SPDX-License-Identifier: Apache-2.0
import subprocess
import sys
from pathlib import Path

import pytest

from dependency_sources import find_local_path_dependencies

PUBLIC = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve().with_name("dependency_sources.py")

PIXI_TOML = """\
[workspace]
channels = ["https://prefix.dev/conda-forge"]

[dependencies]
python = "3.12.*"
local-conda = { path = "./vendor/local-conda" }

[pypi-dependencies]
numpy = ">=2.4,<3"
remote-wheel = { url = "https://example.com/remote_wheel-1.0-py3-none-any.whl" }
inline-path = { path = "./vendor/inline_path-1.0-py3-none-any.whl" }
file-url = { url = "file:///opt/wheels/file_url-1.0-py3-none-any.whl" }
dot-string = "./vendor/dot_string"
parent-string = "../dot_parent"
absolute-string = "/opt/wheels/absolute_string"
home-string = "~/wheels/home_string"

[feature.extra.pypi-dependencies.table-path]
path = "/opt/wheels/table_path-1.0-py3-none-any.whl"
"""

PYPROJECT_TOML = """\
[project]
name = "sample"
dependencies = [
    "requests>=2",
    "remote @ https://example.com/remote-1.0-py3-none-any.whl",
    "file-ref @ file:///opt/wheels/file_ref-1.0-py3-none-any.whl",
    "dot-ref @ ./vendor/dot_ref-1.0-py3-none-any.whl",
]

[tool.uv.sources]
source-path = { path = "../source_path" }
"""

PIXI_LOCK = """\
version: 6
environments:
  default:
    packages:
      linux-64:
      - conda: https://prefix.dev/conda-forge/linux-64/python-3.12.0-0.conda
      - conda: /opt/channel/linux-64/local-1.0-0.conda
      - pypi: https://files.pythonhosted.org/packages/aa/bb/numpy-2.4.0-cp312-cp312-manylinux_2_28_x86_64.whl
      - pypi: ./vendor/inline_path-1.0-py3-none-any.whl
packages:
- pypi: ./vendor/inline_path-1.0-py3-none-any.whl
  name: inline-path
  version: '1.0'
- pypi: https://files.pythonhosted.org/packages/aa/bb/numpy-2.4.0-cp312-cp312-manylinux_2_28_x86_64.whl
  name: numpy
  version: 2.4.0
  sha256: 0000000000000000000000000000000000000000000000000000000000000000
- pypi: editable-pkg
  name: editable-pkg
  path: ./vendor/editable_pkg
"""

REQUIREMENTS_TXT = """\
# comment ./vendor/not_a_dependency
numpy==2.4.0
https://example.com/remote-1.0-py3-none-any.whl
-r requirements-base.txt
-e ./vendor/editable_pkg
--editable /opt/src/editable_long
file-ref @ file:///opt/wheels/file_ref-1.0-py3-none-any.whl
./vendor/dot_ref-1.0-py3-none-any.whl
vendor/bare_ref-1.0-py3-none-any.whl
~/wheels/home_ref-1.0-py3-none-any.whl
"""

REQUIREMENTS_OPTIONS_TXT = """\
numpy==2.4.0 \\
    --hash=sha256:0000000000000000000000000000000000000000000000000000000000000000
--find-links file:///opt/wheels
-f ./wheels
-i file:///opt/index
-r requirements-base.txt
"""

REPEATED_PATH_TOML = """\
[feature.first.pypi-dependencies]
shared = { path = "./vendor/shared-1.0-py3-none-any.whl" }

[feature.second.pypi-dependencies]
shared = { path = "./vendor/shared-1.0-py3-none-any.whl" }
"""

PROJECT_FOLDER_TOML = """\
[pypi-dependencies]
sample = { path = ".", editable = true }
same-folder = { path = "./", editable = true }
sibling = { path = "./sibling", editable = true }
parent = { path = "..", editable = true }
"""

PROJECT_FOLDER_LOCK = """\
version: 6
environments:
  default:
    packages:
      linux-64:
      - pypi: ./
      - pypi: ./sibling
packages:
- pypi: ./
  name: sample
- pypi: ./sibling
  name: sibling
"""


def git_tree(root: Path, files: dict[str, str]) -> Path:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    for relpath, text in files.items():
        path = root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def test_public_tree_has_no_local_path_dependencies():
    assert find_local_path_dependencies(PUBLIC) == []


@pytest.mark.parametrize(
    ("relpath", "text", "expected_lines"),
    [
        ("pixi.toml", PIXI_TOML, [6, 11, 12, 13, 14, 15, 16, 19]),
        ("sub/pyproject.toml", PYPROJECT_TOML, [6, 7, 11]),
        ("pixi.lock", PIXI_LOCK, [7, 9, 11, 18, 20]),
        ("requirements-dev.txt", REQUIREMENTS_TXT, [5, 6, 7, 8, 9, 10]),
        ("requirements.txt", REQUIREMENTS_OPTIONS_TXT, [3, 4, 5]),
        ("pixi.toml", REPEATED_PATH_TOML, [2, 5]),
        ("pixi.toml", PROJECT_FOLDER_TOML, [4, 5]),
        ("pixi.lock", PROJECT_FOLDER_LOCK, [7, 11]),
    ],
)
def test_catches_each_local_dependency_form(tmp_path, relpath, text, expected_lines):
    root = git_tree(tmp_path, {relpath: text})
    lines = text.splitlines()
    assert find_local_path_dependencies(root) == [
        f"{relpath}:{number}: {lines[number - 1].strip()}" for number in expected_lines
    ]


def test_ignored_files_are_not_scanned(tmp_path):
    root = git_tree(tmp_path, {".gitignore": "env/\n", "env/pixi.toml": PIXI_TOML})
    assert find_local_path_dependencies(root) == []


def test_cli_exit_status(tmp_path):
    clean = git_tree(tmp_path / "clean", {"pixi.toml": '[pypi-dependencies]\nnumpy = ">=2.4,<3"\n'})
    dirty = git_tree(tmp_path / "dirty", {"requirements.txt": "-e ./vendor/editable_pkg\n"})
    passed = subprocess.run([sys.executable, str(SCRIPT), str(clean)], capture_output=True, text=True)
    failed = subprocess.run([sys.executable, str(SCRIPT), str(dirty)], capture_output=True, text=True)
    assert (passed.returncode, passed.stdout) == (0, "")
    assert (failed.returncode, failed.stdout) == (1, "requirements.txt:1: -e ./vendor/editable_pkg\n")


def test_bare_archive_requirement_is_local_but_remote_ones_are_not(tmp_path):
    text = "pkg-1.0-py3-none-any.whl\npkg==1.0\npkg @ https://host/pkg-1.0-py3-none-any.whl\nhttps://host/x.whl\n"
    root = git_tree(tmp_path, {"requirements.txt": text})
    assert find_local_path_dependencies(root) == ["requirements.txt:1: pkg-1.0-py3-none-any.whl"]


def test_archive_requirement_with_per_requirement_options_is_local(tmp_path):
    text = "pkg-1.0-py3-none-any.whl --hash=sha256:abc\nhttps://host/x.whl --hash=sha256:abc\n"
    root = git_tree(tmp_path, {"requirements.txt": text})
    assert find_local_path_dependencies(root) == ["requirements.txt:1: pkg-1.0-py3-none-any.whl --hash=sha256:abc"]


def test_slashes_in_requirement_options_do_not_make_it_local(tmp_path):
    text = (
        "pkg==1.0 --config-settings=build-dir=/tmp/build\n"
        "https://host/x.whl --config-settings=build-dir=/tmp/build\n"
        "vendor/pkg-1.0-py3-none-any.whl --config-settings=build-dir=/tmp/build\n"
    )
    root = git_tree(tmp_path, {"requirements.txt": text})
    assert find_local_path_dependencies(root) == [
        "requirements.txt:3: vendor/pkg-1.0-py3-none-any.whl --config-settings=build-dir=/tmp/build"
    ]


def test_local_archive_path_with_spaces_is_local(tmp_path):
    text = (
        "vendor packages/pkg-1.0-py3-none-any.whl\n"
        "vendor packages/pkg-1.0-py3-none-any.whl --hash=sha256:abc\n"
        "pkg-1.0-py3-none-any.whl  --hash=sha256:abc\n"
        "pkg==1.0\n"
    )
    root = git_tree(tmp_path, {"requirements.txt": text})
    assert find_local_path_dependencies(root) == [
        "requirements.txt:1: vendor packages/pkg-1.0-py3-none-any.whl",
        "requirements.txt:2: vendor packages/pkg-1.0-py3-none-any.whl --hash=sha256:abc",
        "requirements.txt:3: pkg-1.0-py3-none-any.whl  --hash=sha256:abc",
    ]
