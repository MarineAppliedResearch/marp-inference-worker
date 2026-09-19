# test_device.py
# Created: 2026-09-09
# Author: Isaac Travers
#
# Tests for compute device resolution in the MARP Inference Worker.
# One rule here matters more than the rest, and it changed on 2026-09-18: "auto"
# now falls back to "cpu" on a machine with no GPU, and must announce it every
# time.
#
# R14 previously refused instead, for a good reason -- "results nobody knows were
# degraded, which is worse than an error". But refusing made a machine without a
# graphics card useless: the coordinator sends specs with no device, which means
# "auto", so such a worker enrolled, polled and declined every job while reporting
# itself healthy. The volunteer programme accepts any computer, with a GPU or
# without. So the danger R14 named is answered by the warning rather than by the
# refusal, and these tests now assert the warning as strictly as they once
# asserted the exception.
#
# These run on a machine with no GPU, which is the case that makes them useful:
# the fallback these guard against is exactly what a GPU-less machine would do.

import pytest

from marp_inference_worker.engines.base_engine import JobUnrunnable
from marp_inference_worker.system import device as device_module


# test_auto_falls_back_to_cpu_and_says_so(monkeypatch, caplog)
# Verifies the central rule, as it now stands.
# Inputs: pytest monkeypatch.
# Output: pytest pass/fail result.
#
# A machine with no CUDA device runs the job on the processor rather than refusing
# it -- and warns, because a silent twenty-fold slowdown is the thing R14 was right
# to object to. Replaces test_auto_refuses_rather_than_falling_back_to_cpu.
def test_auto_falls_back_to_cpu_and_says_so(monkeypatch) -> None:

    # Stand in for a machine with no usable GPU.
    monkeypatch.setattr(device_module, "cuda_device_count", lambda: 0)

    warnings = []
    monkeypatch.setattr(
        device_module.logger, "warning", lambda message, *a, **k: warnings.append(message)
    )

    assert device_module.resolve_device("auto", slot_index=0) == "cpu"

    # The fallback must not be silent. Asserting the warning is the whole point:
    # without it this test would be licensing exactly what R14 forbade.
    assert len(warnings) == 1
    assert "CPU" in warnings[0]
    assert "slower" in warnings[0]


# test_missing_device_is_treated_as_auto_and_also_refuses(monkeypatch)
# Verifies the default.
# Inputs: pytest monkeypatch.
# Output: pytest pass/fail result.
#
# Proves an omitted device behaves exactly like "auto". This is the common case
# rather than an edge one: the coordinator sends specs with no device at all, so
# whatever happens here is what happens to every job on a machine without a GPU.
def test_missing_device_is_treated_as_auto(monkeypatch) -> None:

    monkeypatch.setattr(device_module, "cuda_device_count", lambda: 0)
    monkeypatch.setattr(device_module.logger, "warning", lambda *a, **k: None)

    assert device_module.resolve_device(None, slot_index=0) == "cpu"


# test_explicit_cpu_is_honoured()
# Verifies that CPU is available when asked for outright.
# Inputs: none.
# Output: pytest pass/fail result.
# Proves the escape hatch exists: running without a GPU is possible, but it has
# to be a decision somebody made rather than one that happened.
def test_explicit_cpu_is_honoured() -> None:

    assert device_module.resolve_device("cpu", slot_index=0) == "cpu"

    # Casing and surrounding whitespace do not change the meaning.
    assert device_module.resolve_device(" CPU ", slot_index=3) == "cpu"


# test_auto_resolves_to_the_slots_own_gpu(monkeypatch)
# Verifies slot pinning.
# Inputs: pytest monkeypatch.
# Output: pytest pass/fail result.
# Proves R6's device half: a job pinned to slot 1 runs on cuda:1, so two jobs on
# one host do not both land on GPU 0 and exhaust its memory.
def test_auto_resolves_to_the_slots_own_gpu(monkeypatch) -> None:

    monkeypatch.setattr(device_module, "cuda_device_count", lambda: 4)

    assert device_module.resolve_device("auto", slot_index=0) == "cuda:0"
    assert device_module.resolve_device("auto", slot_index=1) == "cuda:1"
    assert device_module.resolve_device("auto", slot_index=3) == "cuda:3"


# test_auto_shares_the_cards_when_there_are_more_slots_than_gpus(monkeypatch)
# Verifies that a slot is a unit of concurrency rather than a device index.
# Inputs: pytest monkeypatch.
# Output: pytest pass/fail result.
#
# This test used to assert the opposite -- that `auto` refuses a slot past the
# last GPU -- and that assertion was the bug. A slot *was* a device, so one card
# meant one job however large the card was. Raising the slot count then failed
# three leases out of four with "job was pinned to slot 3 but this worker has 1
# CUDA device(s)", and each failure spent one of the job's attempts; seven jobs
# were exhausted before it was caught.
#
# A card holds several jobs comfortably -- under 1 GiB each measured on a 16 GiB
# card -- so slots now wrap over the devices. The refusal that matters is kept
# below: a job naming a device outright is still checked.
def test_auto_shares_the_cards_when_there_are_more_slots_than_gpus(monkeypatch) -> None:

    monkeypatch.setattr(device_module, "cuda_device_count", lambda: 2)

    # Two cards, six slots: round robin, so the load spreads rather than
    # piling onto the first card or refusing.
    assert device_module.resolve_device("auto", slot_index=0) == "cuda:0"
    assert device_module.resolve_device("auto", slot_index=1) == "cuda:1"
    assert device_module.resolve_device("auto", slot_index=5) == "cuda:1"

    # One card, four slots: the case that failed in the field.
    monkeypatch.setattr(device_module, "cuda_device_count", lambda: 1)

    assert [
        device_module.resolve_device("auto", slot_index=slot) for slot in range(4)
    ] == ["cuda:0"] * 4


# test_auto_never_hands_back_a_card_that_does_not_exist(monkeypatch)
# Verifies the failure `auto` must still not have.
# Inputs: pytest monkeypatch.
# Output: pytest pass/fail result.
# Sharing cards is only meaningful when there is a card. With none, the answer is
# "cpu" -- never `cuda:0`, which would fail later and further from the cause.
def test_auto_never_hands_back_a_card_that_does_not_exist(monkeypatch) -> None:

    monkeypatch.setattr(device_module, "cuda_device_count", lambda: 0)
    monkeypatch.setattr(device_module.logger, "warning", lambda *a, **k: None)

    resolved = device_module.resolve_device("auto", slot_index=0)

    assert resolved == "cpu"
    assert "cuda" not in resolved


# test_explicit_cuda_index_is_checked_against_what_exists(monkeypatch)
# Verifies an explicit device request.
# Inputs: pytest monkeypatch.
# Output: pytest pass/fail result.
# Proves R14's first sentence for the explicit case: a job asking for cuda:3 on
# a two-GPU machine fails now, with the reason, rather than when it starts.
def test_explicit_cuda_index_is_checked_against_what_exists(monkeypatch) -> None:

    monkeypatch.setattr(device_module, "cuda_device_count", lambda: 2)

    # Within range, honoured exactly -- not remapped to the slot.
    assert device_module.resolve_device("cuda:1", slot_index=0) == "cuda:1"

    # Out of range, refused.
    with pytest.raises(JobUnrunnable) as raised:
        device_module.resolve_device("cuda:3", slot_index=0)
    assert "cuda:3" in str(raised.value)


# test_bare_cuda_means_the_slots_gpu(monkeypatch)
# Verifies the shorthand.
# Inputs: pytest monkeypatch.
# Output: pytest pass/fail result.
# Proves "cuda" without an index is the slot's device, so a coordinator need not
# know how the worker numbers its slots.
def test_bare_cuda_means_the_slots_gpu(monkeypatch) -> None:

    monkeypatch.setattr(device_module, "cuda_device_count", lambda: 2)

    assert device_module.resolve_device("cuda", slot_index=1) == "cuda:1"


# test_cuda_request_refuses_when_there_is_no_cuda(monkeypatch)
# Verifies that an explicit GPU request also refuses.
# Inputs: pytest monkeypatch.
# Output: pytest pass/fail result.
# Proves the rule is about the fallback, not about the word "auto": asking for
# cuda:0 on a GPU-less machine is refused too.
def test_cuda_request_refuses_when_there_is_no_cuda(monkeypatch) -> None:

    monkeypatch.setattr(device_module, "cuda_device_count", lambda: 0)

    with pytest.raises(JobUnrunnable):
        device_module.resolve_device("cuda:0", slot_index=0)


# test_unparseable_and_unsupported_devices_are_refused(monkeypatch)
# Verifies that unknown values are not guessed at.
# Inputs: pytest monkeypatch.
# Output: pytest pass/fail result.
# Proves an unrecognized device is a refusal rather than a silent default -- the
# same principle as the auto rule, applied to typos and to backends this worker
# has never heard of.
def test_unparseable_and_unsupported_devices_are_refused(monkeypatch) -> None:

    monkeypatch.setattr(device_module, "cuda_device_count", lambda: 2)

    # A malformed cuda spec.
    with pytest.raises(JobUnrunnable):
        device_module.resolve_device("cuda:not-a-number", slot_index=0)

    # A backend this worker does not support.
    with pytest.raises(JobUnrunnable) as raised:
        device_module.resolve_device("mps", slot_index=0)
    assert "mps" in str(raised.value)


# test_cuda_device_count_survives_a_broken_driver(monkeypatch)
# Verifies the discovery path's error handling.
# Inputs: pytest monkeypatch.
# Output: pytest pass/fail result.
#
# Proves that a machine whose driver will not initialize reports as having no
# GPU rather than taking the worker down. It then refuses GPU jobs, by the rule
# above, which is the correct outcome: the machine genuinely cannot run them.
def test_cuda_device_count_survives_a_broken_driver(monkeypatch) -> None:

    class BrokenTorch:
        class cuda:
            @staticmethod
            def is_available():
                raise RuntimeError("CUDA driver version is insufficient")

    monkeypatch.setattr(device_module, "_load_torch", lambda: BrokenTorch)

    assert device_module.cuda_device_count() == 0


# test_cuda_device_count_is_zero_without_torch(monkeypatch)
# Verifies the no-ML-stack case.
# Inputs: pytest monkeypatch.
# Output: pytest pass/fail result.
# Proves the worker's API and enrolment work on a machine with no torch, which
# is how a status check on a broken deployment stays possible.
def test_cuda_device_count_is_zero_without_torch(monkeypatch) -> None:

    monkeypatch.setattr(device_module, "_load_torch", lambda: None)

    assert device_module.cuda_device_count() == 0
    assert device_module.describe_cuda_devices() == []
