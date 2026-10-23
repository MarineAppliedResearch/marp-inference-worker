# provenance.py
# Created: 2026-10-22
# Author: Isaac Travers
#
# Code provenance for the MARP Inference Worker.
# Answers one question: which build of this worker produced a result. A training
# run is a scientific artefact that outlives the checkout that made it, and
# MARP's `training_runs.train_script_commit` is where that answer is kept, so the
# worker has to be able to say it rather than leave the column empty.
#
# What belongs here: identifying this build. What does not: anything about the
# machine it runs on -- that is `device.py` and `resource_monitor.py` -- and
# anything about the coordinator.

# subprocess asks git for the commit, when there is a git to ask.
import subprocess

# Path finds the checkout this module was loaded out of.
from pathlib import Path

# The installed package's version, which is the answer when git cannot be.
from marp_inference_worker.jobs.worker_state import worker_version


# The resolved revision, worked out once per process.
#
# A job's settings report asks for this once, but the resolution shells out to
# git, and a cache means a future caller on a hotter path costs nothing. None
# means "not yet asked", which is distinguishable from a failed lookup because a
# failure still caches a string.
_CACHED_REVISION: str | None = None


# worker_revision()
# Returns an identifier for the code this worker is running.
# Inputs: none.
# Output: a git commit sha when this is a checkout, otherwise the installed
#   package version, otherwise "unknown".
# Use this wherever a result has to be traceable back to the code that made it.
#
# Two forms on purpose, and a reader can tell them apart: a checkout answers
# with 40 hex characters, possibly suffixed `+dirty`, and an installed worker
# answers with its package version. A volunteer running a wheel has no git
# history to name and the version is the most specific true thing about it, so
# the fallback is a real answer rather than a shrug.
def worker_revision() -> str:

    global _CACHED_REVISION

    if _CACHED_REVISION is not None:
        return _CACHED_REVISION

    _CACHED_REVISION = _resolve_revision()
    return _CACHED_REVISION


# _resolve_revision()
# Works out the revision string once.
# Inputs: none.
# Output: the revision string.
# Separate from the cache so a test can exercise the resolution directly.
def _resolve_revision() -> str:

    commit = _git_commit()
    if commit:
        return commit

    return worker_version()


# _git_commit()
# Asks git for this checkout's HEAD, and whether it is dirty.
# Inputs: none.
# Output: the sha, with `+dirty` appended when the tree has changes, or None
#   when this is not a checkout or git is unavailable.
#
# `+dirty` matters more here than it would elsewhere. A run started from a
# working tree with uncommitted changes cannot be reproduced from the sha alone,
# and a provenance record that does not say so is worse than one that is empty,
# because it looks authoritative.
def _git_commit() -> str | None:

    # src/marp_inference_worker/system/provenance.py -> the repository root.
    # Checked before shelling out so an installed worker, which has no `.git`,
    # never pays for a subprocess it is certain to lose.
    root = Path(__file__).resolve().parents[3]
    if not (root / ".git").exists():
        return None

    sha = _git(root, "rev-parse", "HEAD")
    if not sha:
        return None

    # An empty status is a clean tree. A failed status is reported as neither
    # clean nor dirty rather than guessed, so the sha stands on its own.
    status = _git(root, "status", "--porcelain")
    return f"{sha}+dirty" if status else sha


# _git()
# Runs one git command in the checkout and returns its stripped output.
# Inputs: the checkout root and the git arguments.
# Output: stdout with whitespace stripped, or None on any failure.
# Use this so a missing git, a broken repository or a timeout all come back the
# same way: as "cannot say", never as an exception out of a training run.
def _git(root: Path, *args: str) -> str | None:

    try:
        finished = subprocess.run(
            ["git", *args],
            cwd=str(root),
            capture_output=True,
            text=True,
            # A git call against a local checkout is milliseconds. The timeout
            # is there for the pathological case -- a repository on a network
            # share that has gone away -- where blocking a job's first epoch on
            # a bookkeeping lookup would be the wrong trade.
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if finished.returncode != 0:
        return None

    return finished.stdout.strip() or None
