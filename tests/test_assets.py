# SPDX-License-Identifier: Apache-2.0
import argparse
import hashlib
import json
import os
import tomllib
from pathlib import Path

import pytest

from wuji_sim import assets
from wuji_sim.assets import AssetError, AssetsModifiedWarning

SYNTHETIC_FILES = {
    "package.xml": b"<package>root, excluded</package>\n",
    "CMakeLists.txt": b"project(excluded)\n",
    ".hidden": b"hidden\n",
    "a.txt": b"lower a\n",
    "B.txt": b"upper B\n",
    "sub/package.xml": b"<package>nested, included</package>\n",
    "sub/z.bin": bytes(range(256)),
    "\u00e9.txt": "e acute \u00e9\n".encode("utf-8"),
    "mjcf/right_with_glove.xml": b"<mujoco/>\n",
}

# Expected values computed outside Python with find, LC_ALL=C sort and sha256sum on the tree make_product_tree writes.
SYNTHETIC_DIGEST = "0c205c9d9cc320d4b74ffb4098a44871c9761c791a92c144e4f8dd482a9c2acb"
SYNTHETIC_FILE_COUNT = 7
SYNTHETIC_TOTAL_BYTES = 336

# hand2/hand2_2/body of wuji-description v2026.10.12.
GOLDEN_ENV_VAR = "WUJI_DIGEST_GOLDEN_ROOT"
GOLDEN_DIGEST = "d7182d0d48608db595d899227d4de6dd7562234ce93330b8d04ff276d6816b7a"

OTHER_DIGEST = "0" * 64


def make_product_tree(root: Path) -> Path:
    for rel, content in SYNTHETIC_FILES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root


def write_expected(path: Path, digest: str) -> Path:
    path.write_text(
        "[hand2_2]\n"
        'description_version = "v-test"\n'
        f'digest = "{digest}"\n'
        f"files = {SYNTHETIC_FILE_COUNT}\n"
        f"bytes = {SYNTHETIC_TOTAL_BYTES}\n",
        encoding="utf-8",
    )
    return path


def append_package_expected(path: Path, package_digest: str, package_files: int, package_bytes: int) -> Path:
    with path.open("a", encoding="utf-8") as f:
        f.write(f'package = "{assets.PACKAGE_NAME}"\n')
        f.write(f"package_files = {package_files}\n")
        f.write(f"package_bytes = {package_bytes}\n")
        f.write(f'package_digest = "{package_digest}"\n')
    return path


def make_package_tree(root: Path) -> tuple[Path, str, int, int]:
    files = {
        "mjcf/right_with_glove.xml": b"<mujoco/>\n",
        "mjcf/left_with_glove.xml": b"<mujoco/>\n",
        "urdf/right.urdf": b"<robot/>\n",
        "meshes/x.stl": bytes(range(16)),
    }
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (root / "__init__.py").write_bytes(b"not asset content\n")
    (root / "_version.py").write_bytes(b'__version__ = "v-test"\n')
    entries = [{"path": rel, "sha256": hashlib.sha256(content).hexdigest()} for rel, content in sorted(files.items())]
    digest = hashlib.sha256("".join(f"{e['sha256']}  {e['path']}\n" for e in entries).encode()).hexdigest()
    manifest = {"package": assets.PACKAGE_NAME, "version": "v-test", "files": entries, "digest": digest}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root, digest, len(entries), sum(len(content) for content in files.values())


@pytest.fixture
def package(tmp_path: Path, expected_match: Path, monkeypatch) -> tuple[Path, str, int, int]:
    package_tree = make_package_tree(tmp_path / "package")
    append_package_expected(expected_match, *package_tree[1:])
    monkeypatch.setattr(assets, "_package_root", lambda: package_tree[0])
    return package_tree


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    return make_product_tree(tmp_path / "body")


@pytest.fixture
def expected_match(tmp_path: Path) -> Path:
    return write_expected(tmp_path / "expected-match.toml", SYNTHETIC_DIGEST)


@pytest.fixture
def expected_other(tmp_path: Path) -> Path:
    return write_expected(tmp_path / "expected-other.toml", OTHER_DIGEST)


def test_synthetic_digest_matches_coreutils_pipeline(tree):
    assert assets.product_digest(tree) == SYNTHETIC_DIGEST


def test_manifest_format_order_and_exclusions(tree):
    lines = assets.product_manifest(tree).split("\n")
    assert lines[-1] == ""
    paths = [line.split("  ", 1)[1] for line in lines[:-1]]
    assert paths == [".hidden", "B.txt", "a.txt", "mjcf/right_with_glove.xml", "sub/package.xml", "sub/z.bin", "\u00e9.txt"]
    for line in lines[:-1]:
        digest, _ = line.split("  ", 1)
        assert len(digest) == 64 and digest == digest.lower()


def test_root_level_package_xml_change_keeps_digest(tree):
    (tree / "package.xml").write_bytes(b"<package>changed</package>\n")
    (tree / "CMakeLists.txt").write_bytes(b"project(changed)\n")
    assert assets.product_digest(tree) == SYNTHETIC_DIGEST


def test_nested_package_xml_change_changes_digest(tree):
    (tree / "sub" / "package.xml").write_bytes(b"<package>changed</package>\n")
    assert assets.product_digest(tree) != SYNTHETIC_DIGEST


def test_symlink_in_tree_is_rejected(tree):
    (tree / "link.txt").symlink_to(tree / "a.txt")
    with pytest.raises(AssetError, match="link.txt"):
        assets.product_digest(tree)


def test_symlinked_directory_in_tree_is_rejected(tree):
    (tree / "sub_link").symlink_to(tree / "sub", target_is_directory=True)
    with pytest.raises(AssetError, match="sub_link"):
        assets.product_digest(tree)


def test_non_regular_file_in_tree_is_rejected(tree):
    os.mkfifo(tree / "sub" / "pipe")
    with pytest.raises(AssetError, match="pipe"):
        assets.product_digest(tree)


def test_argument_takes_precedence_over_env(tree, tmp_path, expected_match):
    other = tmp_path / "other"
    other.mkdir()
    asset = assets.resolve("right", tree, env={assets.ENV_VAR: str(other)}, expected_path=expected_match)
    assert asset.source == "argument"
    assert asset.root == tree.resolve()
    assert asset.mjcf == tree.resolve() / "mjcf" / "right_with_glove.xml"
    assert asset.usd == tree.resolve() / "usd" / "right_with_glove" / "wujihand2_2_glove.usd"


def test_env_used_without_argument(tree, expected_match):
    asset = assets.resolve("right", env={assets.ENV_VAR: str(tree)}, expected_path=expected_match)
    assert asset.source == "env"
    assert asset.root == tree.resolve()


def test_env_defaults_to_os_environ(tree, expected_match, monkeypatch):
    monkeypatch.setenv(assets.ENV_VAR, str(tree))
    asset = assets.resolve("right", expected_path=expected_match)
    assert asset.source == "env"


@pytest.mark.parametrize("env", [{}, {"WUJI_DESCRIPTION_ROOT": ""}])
def test_no_argument_and_no_env_is_an_error(env, expected_match, monkeypatch):
    monkeypatch.setattr(assets, "_package_root", lambda: None)
    with pytest.raises(AssetError) as excinfo:
        assets.resolve("right", env=env, expected_path=expected_match)
    message = str(excinfo.value)
    assert "WUJI_DESCRIPTION_ROOT" in message
    assert "--asset-root" in message
    assert assets.PACKAGE_NAME in message


def test_package_used_without_argument_or_env(package, expected_match):
    package_root, digest, file_count, total_bytes = package
    asset = assets.resolve("right", env={}, expected_path=expected_match)
    assert asset.source == "package"
    assert asset.root == package_root.resolve()
    assert asset.mjcf == package_root.resolve() / "mjcf" / "right_with_glove.xml"
    assert asset.digest == digest
    assert asset.expected_digest == digest
    assert asset.file_count == file_count
    assert asset.total_bytes == total_bytes
    assert asset.status == "matches-expected"


@pytest.mark.parametrize("given, want", [("argument", "argument"), ("env", "env"), (None, "package")])
def test_argument_then_env_then_package(tree, package, expected_match, given, want):
    asset_root = tree if given == "argument" else None
    env = {assets.ENV_VAR: str(tree)} if given in ("argument", "env") else {}
    asset = assets.resolve("right", asset_root, env=env, expected_path=expected_match)
    assert asset.source == want
    assert asset.root == (package[0] if want == "package" else tree).resolve()


def test_package_root_as_argument_uses_package_digest(package, expected_match):
    package_root, digest, _, _ = package
    asset = assets.resolve("right", package_root, env={}, expected_path=expected_match)
    assert asset.source == "argument"
    assert asset.digest == digest
    assert asset.status == "matches-expected"


def test_package_pycache_is_skipped(package, expected_match):
    package_root, digest, file_count, _ = package
    (package_root / "__pycache__").mkdir()
    (package_root / "__pycache__" / "__init__.cpython-311.pyc").write_bytes(b"bytecode\n")
    asset = assets.resolve("right", env={}, expected_path=expected_match)
    assert asset.digest == digest
    assert asset.file_count == file_count


def test_package_mismatch_with_expected_is_an_error(tmp_path, expected_match, monkeypatch):
    package_root, _, _, _ = make_package_tree(tmp_path / "package")
    append_package_expected(expected_match, OTHER_DIGEST, 1, 1)
    monkeypatch.setattr(assets, "_package_root", lambda: package_root)
    with pytest.raises(AssetError, match=OTHER_DIGEST):
        assets.resolve("right", strict=True, env={}, expected_path=expected_match)


def test_package_content_differing_from_its_manifest_is_an_error(package, expected_match):
    package_root = package[0]
    (package_root / "mjcf" / "right_with_glove.xml").write_bytes(b"<mujoco>tampered</mujoco>\n")
    with pytest.raises(AssetError, match="manifest.json"):
        assets.resolve("right", env={}, expected_path=expected_match)


def test_tampered_package_manifest_is_an_error(package, expected_match):
    package_root = package[0]
    manifest_path = package_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["digest"] = OTHER_DIGEST
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(AssetError, match=OTHER_DIGEST):
        assets.resolve("right", env={}, expected_path=expected_match)


def test_missing_entry_mjcf_is_an_error(tree, expected_match):
    with pytest.raises(AssetError, match="hand2/hand2_2/body"):
        assets.resolve("left", tree, env={}, expected_path=expected_match)


def test_missing_root_is_an_error(tmp_path, expected_match):
    with pytest.raises(AssetError, match="hand2/hand2_2/body"):
        assets.resolve("right", tmp_path / "missing", env={}, expected_path=expected_match)


def test_root_that_is_a_file_is_an_error(tree, expected_match):
    with pytest.raises(AssetError, match="hand2/hand2_2/body"):
        assets.resolve("right", tree / "a.txt", env={}, expected_path=expected_match)


def test_invalid_hand_is_an_error(tree, expected_match):
    with pytest.raises(AssetError, match="hand"):
        assets.resolve("middle", tree, env={}, expected_path=expected_match)


def test_matching_digest(tree, expected_match):
    asset = assets.resolve("right", tree, env={}, expected_path=expected_match)
    assert asset.product == "hand2_2"
    assert asset.hand == "right"
    assert asset.digest == SYNTHETIC_DIGEST
    assert asset.expected_digest == SYNTHETIC_DIGEST
    assert asset.expected_version == "v-test"
    assert asset.file_count == SYNTHETIC_FILE_COUNT
    assert asset.total_bytes == SYNTHETIC_TOTAL_BYTES
    assert asset.status == "matches-expected"
    assert asset.labels == []


def test_mismatch_warns_and_labels(tree, expected_other):
    with pytest.warns(AssetsModifiedWarning):
        asset = assets.resolve("right", tree, env={}, expected_path=expected_other)
    assert asset.status == "assets-modified"
    assert asset.labels == ["assets modified"]
    assert asset.digest == SYNTHETIC_DIGEST
    assert asset.expected_digest == OTHER_DIGEST
    assert asset.describe() == "Wuji Hand 2 right, hand model v-test, assets modified"


def test_mismatch_strict_is_an_error(tree, expected_other):
    with pytest.raises(AssetError) as excinfo:
        assets.resolve("right", tree, strict=True, env={}, expected_path=expected_other)
    message = str(excinfo.value)
    assert SYNTHETIC_DIGEST in message
    assert OTHER_DIGEST in message
    assert str(SYNTHETIC_FILE_COUNT) in message
    assert str(SYNTHETIC_TOTAL_BYTES) in message


@pytest.mark.parametrize("field", ["files", "bytes"])
def test_expected_missing_files_or_bytes_is_an_error(tree, tmp_path, field):
    path = tmp_path / "expected-incomplete.toml"
    lines = write_expected(path, SYNTHETIC_DIGEST).read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text("".join(line for line in lines if not line.startswith(f"{field} =")), encoding="utf-8")
    with pytest.raises(AssetError, match=field) as excinfo:
        assets.resolve("right", tree, env={}, expected_path=path)
    for required in ("digest", "description_version", "files", "bytes"):
        assert required in str(excinfo.value)


def test_describe_and_record(tree, expected_match):
    asset = assets.resolve("right", tree, env={}, expected_path=expected_match)
    assert asset.describe() == "Wuji Hand 2 right, hand model v-test"
    record = asset.to_record()
    assert record["root"] == str(tree.resolve())
    assert record["mjcf"] == str(tree.resolve() / "mjcf" / "right_with_glove.xml")
    assert record["usd"] == str(tree.resolve() / "usd" / "right_with_glove" / "wujihand2_2_glove.usd")
    assert record["labels"] == []
    assert record["status"] == "matches-expected"
    assert set(record) == {
        "product", "hand", "source", "root", "mjcf", "usd", "digest", "file_count", "total_bytes",
        "expected_digest", "expected_version", "status", "labels",
    }


def test_cli_args():
    parser = argparse.ArgumentParser()
    assets.add_cli_args(parser)
    args = parser.parse_args(["--hand", "left", "--asset-root", "/x", "--strict"])
    assert args.hand == "left"
    assert args.asset_root == Path("/x")
    assert args.strict is True
    args = parser.parse_args(["--hand", "right"])
    assert args.asset_root is None
    assert args.strict is False
    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--hand", "middle"])


def test_description_release_matches_expected():
    expected = tomllib.loads(assets.EXPECTED_PATH.read_text(encoding="utf-8"))["hand2_2"]
    asset = assets.resolve("right", env={})
    assert asset.source == "package"
    assert asset.status == "matches-expected"
    assert asset.digest == expected["package_digest"]
    assert asset.expected_version == expected["description_version"] == "v2026.11.18"
    assert asset.file_count == expected["package_files"]
    assert asset.total_bytes == expected["package_bytes"]


def test_installed_package_digest_matches_its_manifest_and_expected():
    root = assets._package_root()
    assert root is not None, f"{assets.PACKAGE_NAME} is not installed"
    declared = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["digest"]
    manifest, _, _ = assets._manifest(root, assets.PACKAGE_DIGEST_EXCLUDED)
    expected = tomllib.loads(assets.EXPECTED_PATH.read_text(encoding="utf-8"))["hand2_2"]
    assert hashlib.sha256(manifest.encode("utf-8")).hexdigest() == declared == expected["package_digest"]


def test_product_digest_golden_vector():
    value = os.environ.get(GOLDEN_ENV_VAR, "")
    if not value:
        pytest.skip(
            f"{GOLDEN_ENV_VAR} is not set; point it at hand2/hand2_2/body of a wuji-description "
            "v2026.10.12 checkout to check the product digest golden vector"
        )
    assert assets.product_digest(Path(value)) == GOLDEN_DIGEST
