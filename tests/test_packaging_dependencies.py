# test_packaging_dependencies.py
# Created: 2026-10-14
# Author: Isaac Travers
#
# Keeps the two lists of this worker's dependencies from drifting apart.
#
# There are two, and that is not an accident: `pyproject.toml` is what a
# developer installs from, and `packaging/requirements-*.in` is what a
# volunteer's installer resolves from. The second deliberately omits the worker
# itself, because a bare `.` would resolve against PowerShell's launch
# directory on somebody else's machine.
#
# They drift silently and in the worst possible direction. Adding pystray for
# the tray (#57) to pyproject alone produced an installer that built, installed
# and ran -- with no icon and no error, because the launcher swallows a missing
# tray on purpose so a headless machine still works (R9). The thing that makes
# a missing tray survivable is the same thing that makes a missing dependency
# invisible.
#
# What belongs here: anything about what the packaged artifact contains.

# tomllib reads pyproject without a dependency of its own (3.11+).
import tomllib

# Path finds the repository root from this file.
from pathlib import Path

import pytest


# The repository root, two levels up from this file.
_ROOT = Path(__file__).resolve().parent.parent


# requirement_name()
# Reduces a requirement line to the package name.
# Inputs: a line such as `uvicorn[standard]` or `torch==2.11.0+cu126`.
# Output: the lower-case name, or "" for a comment or blank line.
# Extras, pins and local versions are all stripped: this compares what is
# present, not which version, because the variants pin torch differently by
# design.
def requirement_name(line: str) -> str:

    line = line.strip()
    if not line or line.startswith("#"):
        return ""

    for separator in ("==", ">=", "<=", "~=", ">", "<", "["):
        if separator in line:
            line = line.split(separator, 1)[0]
    return line.strip().lower()


# declared_dependencies()
# What pyproject says this worker needs at runtime.
# Inputs: none.
# Output: a set of package names.
def declared_dependencies() -> set[str]:

    data = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    names = {requirement_name(entry) for entry in data["project"]["dependencies"]}
    return names - {""}


# Every packaging input, so a new variant is covered the day it is added rather
# than the day somebody remembers to add it here.
_INPUTS = sorted((_ROOT / "packaging").glob("requirements-*.in"))


# test_there_is_something_to_check()
# Guards the parametrised test below from passing vacuously.
# Inputs: none. Output: pytest pass/fail.
# A glob that matches nothing turns the next test into zero tests, which looks
# exactly like green. The doctrine's "a skipped suite looks green", one tier up.
def test_there_is_something_to_check() -> None:

    assert _INPUTS, "no packaging/requirements-*.in files were found"


# test_the_installer_resolves_every_runtime_dependency()
# Verifies a volunteer's install gets what the code imports.
# Inputs: one packaging input file. Output: pytest pass/fail.
#
# The failure this catches is silent by construction, so it has to be caught
# here: the worker installs, starts and works, and only the feature that needed
# the missing package is quietly absent.
@pytest.mark.parametrize("path", _INPUTS, ids=lambda p: p.stem)
def test_the_installer_resolves_every_runtime_dependency(path: Path) -> None:

    packaged = {
        requirement_name(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    } - {""}

    missing = declared_dependencies() - packaged

    assert not missing, (
        f"{path.name} does not install {sorted(missing)}, which pyproject.toml "
        "declares. A volunteer's worker would run without it, and the feature "
        "that needs it would be missing with no error."
    )


# test_the_locks_were_regenerated_after_the_inputs_changed()
# Verifies the compiled lock reflects its own input.
# Inputs: one packaging input file. Output: pytest pass/fail.
#
# Editing a `.in` without re-running `uv pip compile` is the same silent
# failure one step later: the build reads the lock, not the input. This
# compares only the names, so a transitive package in the lock is fine and an
# input that never made it in is not.
@pytest.mark.parametrize("path", _INPUTS, ids=lambda p: p.stem)
def test_the_locks_were_regenerated_after_the_inputs_changed(path: Path) -> None:

    lock = path.with_suffix(".lock.txt")
    if not lock.is_file():
        pytest.fail(f"{path.name} has no compiled lock beside it")

    locked = {
        requirement_name(line)
        for line in lock.read_text(encoding="utf-8").splitlines()
        if not line.startswith((" ", "\t", "-"))
    } - {""}

    wanted = {
        requirement_name(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    } - {""}

    missing = wanted - locked

    assert not missing, (
        f"{lock.name} is older than {path.name}: it does not carry "
        f"{sorted(missing)}. Re-run the `uv pip compile` command recorded at "
        "the top of the lock file."
    )
