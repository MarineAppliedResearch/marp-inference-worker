---
task: MarineAppliedResearch/marp-inference-worker#38
repos: [marp-inference-worker]
status: verifying
needs: []
---

## Goal

A volunteer with a Linux computer and an NVIDIA card downloads one small file, runs it, and
their machine joins MARP. No Python to install, no browser to install, no CUDA toolkit, no
administrator password, and nothing to paste. This is the Linux half of #38; `vp1-marp` has
Windows.

## Requirements

- **R1** — One artifact a volunteer downloads and runs. No manual configuration.
- **R2** — Works on a machine with no Python, no browser and no CUDA toolkit.
- **R3** — Never requires root. Everything lives under one directory; uninstall is `rm -rf`.
- **R4** — Every downloaded artifact is verified against a checked-in SHA-256 before use.
- **R5** — Re-running after a failure continues rather than starting over.
- **R6** — A machine that cannot run MARP is told why, in terms it can act on, before
  anything is downloaded.
- **R7** — The worker starts at login and restarts if it crashes, without a volunteer
  intervening.
- **R8** — The enrolment code is a build-time input and never enters the repository.

## Open assumptions

- [x] **A1 · architectural · blocking** — answered 2026-09-18 by Isaac: bundle everything, or
  download at install? **Download.** Costed on the axis he cares about, which is bytes MARP
  serves rather than bytes a volunteer fetches: ~5.8 MB per volunteer against ~6.8 GB for a
  bundle, and a bundle also exceeds GitHub's 2 GB release-asset limit so it would need paid
  hosting. His words: *"Yeah, just go with your recommendation of a downloader. Just make
  sure that it works really, really well."*

- [x] **A2 · architectural · blocking** — answered 2026-09-18 by Isaac: is dropping torch for
  a smaller runtime acceptable? **No.** Running his own PyTorch models is a planned
  capability, so ONNX and TensorRT are closed on requirements rather than benchmarks. This
  also settles that `triton` and `sympy` stay although unloaded by YOLO inference: they are
  `torch.compile`'s JIT and torch's symbolic shape machinery, and a future custom model is
  exactly what reaches for them.

- [x] **A3 · product · blocking** — decided here rather than escalated, on Isaac's
  instruction to stop bringing him decisions: **a machine with no NVIDIA GPU is refused at
  install.** `resolve_device` refuses "auto" without CUDA by design, so installing anyway
  would produce a worker that enrols, polls, and declines every job — worse for a volunteer
  than an honest refusal with instructions.

- [x] **A4 · product · blocking** — decided here: **cu126 and cu128 selected per card, not
  cu130.** cu130 offers a newer torch and ~1 GB less, at the cost of excluding Maxwell and
  Pascal. A donated GTX 1060 is precisely the hardware this feature exists to accept.

## Decisions

- **2026-09-18** — POSIX `sh`, not bash. A volunteer's machine may not have bash and nothing
  here needs it.
- **2026-09-18** — the CUDA variant is chosen by the same rule as
  `bootstrap-windows.ps1:92-102`: capability >= 12.0 takes cu128, everything else cu126.
  Stated as *the same rule* deliberately — two platforms selecting differently would mean two
  inference stacks.
- **2026-09-18** — a driver floor is enforced before anything downloads. Without it the CUDA
  wheels install perfectly against a too-old driver and the worker enrols, polls, heartbeats
  and fails every job while reporting itself online. Raised by `vp1-marp`, who hit it.
- **2026-09-18** — `MARP_WORKER_STATE_DIR` is set explicitly for `marp-worker-activate` and
  for the service. The two have different defaults, and the mismatch spends a volunteer's
  enrolment then reports "this worker is not activated" — a path fault that reads as a
  credential fault. MARP_API#208 records what it cost.
- **2026-09-18** — a systemd **user** service tied to `graphical-session.target`, not
  lingering. Lingering would start the worker with no display, so the watch window would
  silently never open — on this platform, the failure that looks like success.
- **2026-09-18** — `--check-only`, so a volunteer tests their machine in two seconds rather
  than after twenty minutes of downloading. Finding out late is how people give up.

## Plan

1. `bootstrap-linux.sh`, mirroring the Windows stage order. *(done)*
2. `chromium-linux-x64.lock.json` and `uv-linux-x64.lock.json` with real hashes. *(done)*
3. `requirements-linux-cu126.{in,lock.txt}` — this repository had no Linux lock at all,
   which is #37. *(done)*
4. Driver floor, disk check, dependency checks. *(done)*
5. `build-linux.sh` to assemble the package and substitute the enrolment code. *(not done)*

## Acceptance criteria

- A volunteer downloads one file, runs it, and the worker enrols and starts. **Stages 1-6
  verified; 7 and 8 are not.**
- Nothing requires root. **Met.**
- Every download verified. **Met.**
- Re-running after an interrupted install continues. **Met**, by killing the installer
  mid-download and re-running.
- An unsuitable machine is told why before downloading. **Met** for a missing driver, an old
  driver, insufficient disk and missing tools.

## Test plan

Run against this machine: Ubuntu 26.04, GTX 1660 Ti, driver 595.91.07.

- `--help` and an unknown option behave.
- `--check-only` passes, reporting GPU, capability, chosen runtime, driver and disk.
- A machine with no `nvidia-smi` is refused with distribution-specific instructions
  (simulated by restricting `PATH`).
- A too-small disk is refused before downloading (observed against a 7 GB filesystem).
- Full install: a 5.8 MB package produces a 7.9 GB install — Chromium 156.0.8067.0 runs,
  torch 2.14.0+cu126 sees the card, all three entry points present, and the worker's own
  `_find_chromium()` resolves the bundled browser.
- Interrupted install: `SIGKILL` mid-Chromium-download, then re-run — `uv already downloaded
  and verified`, `reusing the existing environment`, `Chromium already downloaded and
  verified`.

**NOT verified, and it is two of the eight stages:** enrolment and service start. Both need
the standing code from MARP_API#208, which this machine does not have; the test runs used a
deliberately fake code and stage 7 failed exactly as designed. **Do not read "stages 1-6
verified" as "the installer works end to end."**

Also not verified: any machine but this one. One GPU, one distribution, one driver version.
"Runs on many different Linux computers" is the requirement, and a single box cannot
demonstrate it.

## Status

- **Gate:** verifying
- **Notes:** the first resolution of the Linux lock picked `ultralytics-platform==0.1.48`
  where the Windows lock pins `0.1.40` — the exact package VP1 drifted on. Pinned, and the
  two platforms now resolve identically across all 60 shared packages.

  Found and left alone: the worker writes **no local record when a job fails**. Diagnosing
  seven failed attempts here required a traceback fetched from the coordinator by another
  agent. A volunteer has no such route, which makes it a packaging concern as much as a
  worker one. Filed separately.
