# SPDX-License-Identifier: Apache-2.0
import re
import subprocess
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

TOML_NAMES = frozenset({"pixi.toml", "pyproject.toml"})
LOCK_NAME = "pixi.lock"
REQUIREMENTS_NAME = re.compile(r"requirements[^/]*\.txt")

# Matches pixi [*dependencies] and pyproject dependency tables, incl. build-system.requires and tool.uv.sources.
DEPENDENCY_KEY = re.compile(r"(?:^|-)dependencies$|^dependency-groups$|^requires$|^sources$")
LOCAL_PREFIX = re.compile(r"(?:\.{1,2}/|/|~)")
LOCK_SOURCE = re.compile(r"^\s*-\s+(?:pypi|conda):\s*(\S+)")
LOCK_PATH = re.compile(r"^\s*path:\s*(\S+)")
REMOTE = re.compile(r"https?://")
LOCAL_SOURCE_OPTION = re.compile(r"(?:-f|--find-links|-i|--index-url|--extra-index-url)[\s=]+(?:\.{1,2}/|/|~)")
# pip installs a bare archive file name from the working directory.
ARCHIVE_SUFFIXES = (".whl", ".zip", ".tar.gz", ".tgz", ".tar.bz2", ".tbz", ".tar.xz", ".txz", ".tar")


# The project installs itself editable from its own folder; that path is the same on every machine.
def is_project_folder(path: str) -> bool:
    return path != "" and PurePosixPath(path) == PurePosixPath(".")


def is_local_string(value: str) -> bool:
    if "file:" in value:
        return True
    target = value.split(" @ ", 1)[1] if " @ " in value else value
    return LOCAL_PREFIX.match(target.strip()) is not None


def entry_findings(name: str, value) -> Iterator[tuple[str, str]]:
    if isinstance(value, str):
        if is_local_string(value):
            yield name, value
    elif isinstance(value, dict):
        path = value.get("path")
        if isinstance(path, str) and not is_project_folder(path):
            yield name, path
        for inner in value.values():
            if isinstance(inner, str) and "file:" in inner:
                yield name, inner
    elif isinstance(value, list):
        for item in value:
            yield from entry_findings(name, item)


def toml_findings(node) -> Iterator[tuple[str, str]]:
    if isinstance(node, dict):
        for key, value in node.items():
            if DEPENDENCY_KEY.search(key) is None:
                yield from toml_findings(value)
            elif isinstance(value, dict):
                for name, spec in value.items():
                    yield from entry_findings(name, spec)
            else:
                yield from entry_findings("", value)
    elif isinstance(node, list):
        for item in node:
            yield from toml_findings(item)


def line_of(lines: list[str], needle: str, after: int = 0) -> int | None:
    if not needle:
        return None
    for number, line in enumerate(lines[after:], after + 1):
        if needle in line and not line.lstrip().startswith("#"):
            return number
    return None


def scan_toml(text: str) -> Iterator[int]:
    lines = text.splitlines()
    previous: dict[str, int] = {}
    for key, value in toml_findings(tomllib.loads(text)):
        number = line_of(lines, value, previous.get(value, 0)) or line_of(lines, value) or line_of(lines, key) or 1
        previous[value] = number
        yield number


def scan_lock(text: str) -> Iterator[int]:
    for number, line in enumerate(text.splitlines(), 1):
        local = LOCK_SOURCE.match(line) or LOCK_PATH.match(line)
        if local and not REMOTE.match(local.group(1)) and not is_project_folder(local.group(1)):
            yield number


def requirement_target(entry: str) -> str:
    # Mirrors pip's req_file.break_args_options: options start at the first "-"-prefixed token.
    args = []
    for token in entry.split(" "):
        if token.startswith("-"):
            break
        args.append(token)
    return " ".join(args).strip()


def scan_requirements(text: str) -> Iterator[int]:
    for number, line in enumerate(text.splitlines(), 1):
        entry = re.split(r"(?:^|\s)#", line, maxsplit=1)[0].strip().removesuffix("\\").rstrip()
        if not entry:
            continue
        if entry.startswith(("-e", "--editable")):
            yield number
        elif entry.startswith("-"):
            if "file:" in entry or LOCAL_SOURCE_OPTION.match(entry):
                yield number
        elif is_local_string(entry):
            yield number
        elif "://" not in (target := requirement_target(entry)) and ("/" in target or target.endswith(ARCHIVE_SUFFIXES)):
            yield number


def find_local_path_dependencies(root: Path) -> list[str]:
    root = Path(root)
    listed = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    found = []
    for relpath in sorted({name for name in listed.split("\0") if name}):
        name = Path(relpath).name
        if name in TOML_NAMES:
            scanner = scan_toml
        elif name == LOCK_NAME:
            scanner = scan_lock
        elif REQUIREMENTS_NAME.fullmatch(name):
            scanner = scan_requirements
        else:
            continue
        path = root / relpath
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        for number in sorted(set(scanner(text))):
            found.append(f"{relpath}:{number}: {lines[number - 1].strip()}")
    return found


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} ROOT")
    findings = find_local_path_dependencies(Path(sys.argv[1]))
    for finding in findings:
        print(finding)
    sys.exit(1 if findings else 0)
