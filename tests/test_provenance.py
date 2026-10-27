# test_provenance.py
# Created: 2026-10-22
# Author: Isaac Travers
#
# Code-provenance tests for the MARP Inference Worker.
# These cover the one thing worker_revision() has to get right: it always
# answers, and what it answers says which of its two forms it is. A training run
# outlives the checkout that made it, so an empty or ambiguous answer is the
# failure mode that matters -- a wrong commit recorded confidently is worse than
# no commit at all.

# The module under test.
from marp_inference_worker.system import provenance


# clear_the_cache()
# Drops the process-wide cached revision between tests.
# Inputs: none.
# Output: none.
# Needed because worker_revision() memoizes on first call, so a test that did
# not clear it would assert against whatever the previous test resolved.
def clear_the_cache() -> None:

    provenance._CACHED_REVISION = None


# test_a_checkout_answers_with_its_commit()
# Verifies the normal case on a development machine.
# Inputs: monkeypatch.
# Output: pytest pass/fail result.
def test_a_checkout_answers_with_its_commit(monkeypatch) -> None:

    clear_the_cache()
    monkeypatch.setattr(provenance, "_git_commit", lambda: "a" * 40)

    assert provenance.worker_revision() == "a" * 40


# test_a_worker_with_no_checkout_answers_with_its_version()
# Verifies an installed wheel still says something specific about itself.
# Inputs: monkeypatch.
# Output: pytest pass/fail result.
#
# This is the volunteer's case and it is the common one: a worker installed from
# a wheel has no git history to name, so the package version is the most
# specific true thing about it. Falling back is a real answer rather than a
# shrug, and the alternative -- leaving the column empty for every volunteer
# run -- is what this work exists to stop.
def test_a_worker_with_no_checkout_answers_with_its_version(monkeypatch) -> None:

    clear_the_cache()
    monkeypatch.setattr(provenance, "_git_commit", lambda: None)
    monkeypatch.setattr(provenance, "worker_version", lambda: "0.4.2")

    assert provenance.worker_revision() == "0.4.2"


# test_an_uncommitted_tree_says_so()
# Verifies a run from a dirty checkout is not recorded as its last commit.
# Inputs: monkeypatch.
# Output: pytest pass/fail result.
#
# A sha alone claims the run is reproducible from that commit. If the tree had
# edits in it, that claim is false and nothing downstream could tell -- which is
# worse than an empty column, because it looks authoritative.
def test_an_uncommitted_tree_says_so(monkeypatch) -> None:

    monkeypatch.setattr(
        provenance, "_git", lambda root, *args: "b" * 40 if args[0] == "rev-parse" else " M file.py"
    )

    assert provenance._git_commit() == "b" * 40 + "+dirty"


# test_a_clean_tree_is_reported_bare()
# The control for the test above.
# Inputs: monkeypatch.
# Output: pytest pass/fail result.
#
# Without it, `_git_commit` returning "+dirty" unconditionally would pass.
def test_a_clean_tree_is_reported_bare(monkeypatch) -> None:

    monkeypatch.setattr(
        provenance, "_git", lambda root, *args: "c" * 40 if args[0] == "rev-parse" else None
    )

    assert provenance._git_commit() == "c" * 40


# test_the_revision_is_resolved_once()
# Verifies a job's settings report does not shell out to git on every call.
# Inputs: monkeypatch.
# Output: pytest pass/fail result.
def test_the_revision_is_resolved_once(monkeypatch) -> None:

    clear_the_cache()
    calls: list[int] = []

    def counted() -> str:
        calls.append(1)
        return "d" * 40

    monkeypatch.setattr(provenance, "_git_commit", counted)

    provenance.worker_revision()
    provenance.worker_revision()

    assert len(calls) == 1


# test_this_checkout_can_say_what_it_is()
# The end-to-end control: no mocks, the real resolution, on this machine.
# Inputs: none.
# Output: pytest pass/fail result.
#
# Every test above replaces part of the resolution, so all of them would pass
# against a worker_revision() that could never produce an answer at all. This
# one runs it for real and only insists the answer is a non-empty string, which
# is what a record needs and is true in both of its forms.
def test_this_checkout_can_say_what_it_is() -> None:

    clear_the_cache()
    revision = provenance.worker_revision()

    assert isinstance(revision, str)
    assert revision.strip()


# test_an_installed_build_names_its_commit(tmp_path, monkeypatch)
# Verifies a worker with no git still records which build it is.
# Inputs: tmp_path and monkeypatch.
# Output: pytest pass/fail result.
#
# This is the volunteer's case and the common one. Until the installer carried
# the commit through, an installed worker had no .git to ask and fell back to
# its package version -- so every run of every 0.1.8 build recorded the same
# `train_script_commit`, and the column could not tell two builds apart. That
# is the whole job of that column.
def test_an_installed_build_names_its_commit(tmp_path, monkeypatch) -> None:

    clear_the_cache()
    monkeypatch.setattr(provenance, "_git_commit", lambda: None)
    monkeypatch.setattr(provenance, "worker_version", lambda: "0.1.8")

    # The layout an install produces: the manifest sits several directories
    # above the package, so it is found by searching upward.
    version_root = tmp_path / "versions" / "0.1.8-cu128"
    package = version_root / "runtime" / "Lib" / "site-packages" / "marp" / "system"
    package.mkdir(parents=True)
    (version_root / "manifest.json").write_text(
        '{"version": "0.1.8", "worker_commit": "' + "a" * 40 + '"}', encoding="utf-8"
    )

    monkeypatch.setattr(provenance, "__file__", str(package / "provenance.py"))

    assert provenance.worker_revision() == "a" * 40


# test_a_manifest_without_a_commit_is_not_mistaken_for_one(tmp_path, monkeypatch)
# Verifies the search matches on the key, not the filename.
# Inputs: tmp_path and monkeypatch.
# Output: pytest pass/fail result.
#
# `manifest.json` is a common enough name that finding one proves nothing.
# Every install before this change wrote one WITHOUT a commit, so those must
# fall through to the version rather than being treated as an answer.
def test_a_manifest_without_a_commit_is_not_mistaken_for_one(tmp_path, monkeypatch) -> None:

    clear_the_cache()
    monkeypatch.setattr(provenance, "_git_commit", lambda: None)
    monkeypatch.setattr(provenance, "worker_version", lambda: "0.1.7")

    version_root = tmp_path / "versions" / "0.1.7-cu128"
    package = version_root / "runtime" / "Lib" / "site-packages" / "marp" / "system"
    package.mkdir(parents=True)

    # Exactly what 0.1.4 through 0.1.8 wrote: no worker_commit.
    (version_root / "manifest.json").write_text(
        '{"version": "0.1.7", "platform": "windows"}', encoding="utf-8"
    )

    monkeypatch.setattr(provenance, "__file__", str(package / "provenance.py"))

    assert provenance.worker_revision() == "0.1.7"


# test_a_checkout_still_prefers_its_own_git(tmp_path, monkeypatch)
# Verifies the order of preference did not invert.
# Inputs: tmp_path and monkeypatch.
# Output: pytest pass/fail result.
#
# A developer's checkout can say `+dirty`, which an installed manifest never
# can, so git stays first. Without this test, a manifest left lying around in a
# source tree would quietly outrank the real commit.
def test_a_checkout_still_prefers_its_own_git(tmp_path, monkeypatch) -> None:

    clear_the_cache()
    monkeypatch.setattr(provenance, "_git_commit", lambda: "b" * 40 + "+dirty")

    package = tmp_path / "pkg"
    package.mkdir()
    (tmp_path / "manifest.json").write_text(
        '{"worker_commit": "' + "c" * 40 + '"}', encoding="utf-8"
    )
    monkeypatch.setattr(provenance, "__file__", str(package / "provenance.py"))

    assert provenance.worker_revision() == "b" * 40 + "+dirty"


# test_an_unreadable_manifest_is_skipped(tmp_path, monkeypatch)
# Verifies a broken file does not cost the answer.
# Inputs: tmp_path and monkeypatch.
# Output: pytest pass/fail result.
def test_an_unreadable_manifest_is_skipped(tmp_path, monkeypatch) -> None:

    clear_the_cache()
    monkeypatch.setattr(provenance, "_git_commit", lambda: None)
    monkeypatch.setattr(provenance, "worker_version", lambda: "0.1.8")

    package = tmp_path / "a" / "b"
    package.mkdir(parents=True)
    (tmp_path / "a" / "manifest.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(provenance, "__file__", str(package / "provenance.py"))

    assert provenance.worker_revision() == "0.1.8"
