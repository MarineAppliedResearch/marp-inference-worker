---
task: MarineAppliedResearch/marp-inference-worker#38
repos: [marp-inference-worker, marp-video-player]
status: verifying
needs: []
---

# A Windows installer that runs on any volunteer's computer

Refs MarineAppliedResearch/marp-inference-worker#38

## The task

A volunteer downloads one small file, runs it, and their computer joins the pool —
whatever they happen to have. No Python, no browser, no CUDA toolkit, no Visual Studio
beforehand, no questions asked during setup, and **no computer refused**.

The installer stays small and fetches what *that machine* needs, decided from what it
actually has. It does not carry every runtime for every machine, and it does not hand the
volunteer a choice of downloads.

## Requirements

- **R1** The installer reads the machine — GPU present, compute capability, driver
  version — and selects a runtime from that, without asking the volunteer anything.
- **R2** No computer is refused. A machine with no NVIDIA GPU, or one no CUDA build can
  serve, installs the CPU runtime and works slowly rather than failing.
- **R3** Every worker in the pool runs the same inference stack: same torch version, same
  ultralytics, same ByteTrack. Only the CUDA build differs.
- **R4** `MARP_WORKER_STATE_DIR` is set explicitly for both `marp-worker-activate` and the
  job loop, so activation and the worker agree where the credential lives.
- **R5** The enrolment code is a build-time input, baked into the artifact, never committed.
- **R6** Setup succeeds with no Visual Studio and no MSVC redistributable newer than a
  stock Windows install.
- **R7** A runtime the driver cannot serve is never installed. Refusing is not enough:
  the machine falls back to something that works.

## The variants, and what decides between them

```
cu128   sm_70-sm_120    Volta through Blackwell
cu126   sm_50-sm_90     Maxwell through Hopper -- the donated GTX 1060
cpu     no GPU, or a GPU no CUDA build can serve
```

Newest CUDA the machine can run wins, so every current card takes cu128 and cu126 exists
for older hardware. All three pin **torch 2.11.0 / torchvision 0.26.0**, which is the
newest version present on all three indexes — cu128's ceiling is what sets it.

**The capability test is a floor, never list membership.** CUDA minor-version
compatibility runs a binary on later minor versions of the same major family: an RTX 4080
SUPER reports 8.9, which appears in no arch list we ship, and runs on sm_86 kernels. A
membership test would refuse a card that has worked for weeks.

## Why torch is pinned across variants

cu126 pinned `torch==2.14.0` and cu128 pinned `2.10.0`, so the pool ran three torch
versions against the same job specs — ABYSS 2.14.0, VP1 2.14.0, the laptop 2.11.0 — with
nothing announcing it. Same class of problem as R16's exact `ultralytics` pin.

Moving every variant to 2.11.0 is a **downgrade for cu126 machines**, taken deliberately:
one version across the whole pool is worth more than a newer one on two thirds of it.

## Not in scope: making the runtime smaller

Torch stays whole. Running Isaac's own PyTorch models is a planned capability, so what
gets installed is a general PyTorch runtime rather than "what YOLO needs" — anything torch
might `dlopen` for a model nobody has written yet stays. Measured: every DLL in Windows
`torch/lib`, all 3,856 MB, maps at plain `import torch`, and the module count is identical
before and after real CUDA work. There is no lazy loading to exploit.

## Decided without escalating

- **VC++ redistributable, not a vendored CRT**, and installed only when the machine's is
  older than the one we ship. Microsoft services it for security; vendoring would make us
  reship for every CRT fix. **Load-bearing, not hygiene:** VP1 carried MSVC 14.28 from
  2020 and `import torch` failed outright with `OSError WinError 1114` naming `c10.dll` —
  an error pointing nowhere near the runtime, which a volunteer would report as a broken
  package. Checking first means most machines never see the elevation prompt.

- **cu130 dropped.** Measured on real hardware: newest torch, 1.07 GB smaller, reaches
  Blackwell — but it drops Maxwell and Pascal and needs driver 580. Narrower coverage is
  the wrong trade when the goal is any volunteer.

## Open assumptions

- [x] **A1 · cross-repository** — answered 2026-09-19: **ship the CPU variant anyway and
  track the worker-side fix separately.** `resolve_device` refuses `auto` with no CUDA by
  design (R14), so a CPU machine installs correctly, enrols, takes a job and fails it with
  `no usable CUDA device`. marp-laptop's fix is PR #42 on `cpu-only-workers`. This does not
  block the installer: R2 is delivered on the installer's side, and what is missing lives
  in another repository and another PR. Recorded as non-blocking rather than left open,
  because leaving it open would have stopped work that was not actually blocked. The
  consequence — the end-to-end CPU path stays unverified until #42 merges, and testing it
  against `develop` shows a failure that is not the installer's — is carried in
  Verification, which is where an untested path belongs.

- [x] **A2 · environment** — answered 2026-09-19: **ship the published minimums and record
  that they are unmeasured.** `runtime-variants.json` carries `driver_floor_verified: false`
  on both cu126 and cu128 so the uncertainty travels with the data rather than living in
  somebody's memory. Nobody in the fleet has a machine old enough to find where either
  floor actually stops, so measuring it is not available to us; the choice was between
  guessing lower, guessing higher, or shipping the vendor's number and saying so. The one
  measured datapoint is cu128 working at driver 573.13 on an RTX 5060, well below its
  published floor, which suggests the floor is conservative rather than wrong. A decision
  taken with its uncertainty recorded is an answer; it is not the same as a verified floor,
  and Verification says so.

## Record which torch build produced an observation

`gpu_job_attempts.capabilities_snapshot` records `cuda_devices`, `cuda_device_count` and
`ultralytics_version` per attempt, but **not the torch build** — so cu126 against cu128
cannot be recovered for a given observation.

This cuts both ways and both halves belong here. Pinning one torch version is partly a way
of making the question stop mattering: one build in the pool is a variable nobody has to
record. But two CUDA builds are deliberate and right, because a donated GTX 1060 is the
volunteer this is for. **So both builds stay and the snapshot should gain the field** —
the pinning is not an argument for collapsing to one.

Cross-repository: the recording half is `MarineAppliedResearch/MARP_API`.

## Verification

`packaging/test-select-variant.ps1` drives the selection rule with fabricated machine
readings, so it covers hardware nobody in the fleet has. 11 cases, all passing: every
fleet machine, a Pascal and a Maxwell volunteer, no GPU at all, `nvidia-smi` answering
nothing, a GPU on a driver too old for any runtime, and Blackwell never being handed
cu126.

Fabricated readings prove the arithmetic, not the reading of real hardware, which
marp-laptop-install-test pointed out and then closed by running the real thing.

### What has now been run end to end

**marp-laptop-install-test, RTX 5060 Laptop, Windows 11 26200 — clean machine.**
Selected cu128 live off `nvidia-smi` at compute capability 12.0, driver 573.13. Enrolled
as worker 1187, torch 2.11.0+cu128, CUDA 12.8, 10.21 GB in 49,789 files on disk. The
`cython-bbox` wheel installed from the payload rather than being built. Critically, the
CUDA check **executed and synchronized a real kernel on sm_120** — which is what separates
"torch imported" from "this card can run our work", and is the check that catches a wheel
built without this GPU's kernels. That retires the main risk in the cu128 arch-list
decision. It is one allocation: it says nothing about a full job, sustained load, or
memory pressure on 8 GB.

**VP1, GTX 1660, Windows 10 19045 — upgrade over an existing install.** Selected cu128 at
capability 7.5, reused the protected credential, and came up as worker 1185 (separate from
the hand-built dev worker 1079, which has its own state directory and is untouched).
Answered `/health` and `/status` in about 10 seconds, idle, enrolled, 1 slot free,
accepting jobs, on Python 3.12.14.

### Four defects found by running it, and fixed

marp-laptop-install-test read its own transcript and could not tell a healthy install from
a dead one without reading the source. A volunteer cannot do that, so this was a real
defect even though the install had worked.

- **Stage 7 was silent on success** — a machine that enrolled correctly and one that did
  not produced identical transcripts. It now names the worker id and fails loudly if
  activation claims success but writes no identity file.
- **Stage 8 leaked three `TerminatingError(Invoke-RestMethod)` lines** directly above the
  success banner. `-ErrorAction Stop` inside a `try` does **not** fix this: Start-Transcript
  records a terminating error whether or not a `catch` handles it. Proven on two machines
  and two Windows builds. The fix is a TCP socket gate — do not make the HTTP call until
  something is listening. The deadline also became 180s wall-clock instead of 60 iterations
  of a probe whose length was set by its own timeout, which was a loop count, not a deadline.
- **Stage 4 leaked `TerminatingError(): "The pipeline has been stopped."`** from
  `Select-Object -First 1` ending a pipeline early — cosmetic, and fixed by collecting
  into an array and indexing it.

- **Stage 4 built the runtime on the wrong Python.** `Sort-Object Name -Descending` is a
  **string** sort, so it ranked `cpython-3.12.9` above `cpython-3.12.14` and picked the
  older interpreter on any machine holding both. Reproduced on two machines with real
  directories rather than argued:

      OLD  string sort + Select-Object -First 1  ->  cpython-3.12.9   WRONG
      NEW  numeric sort on parsed patch, index   ->  cpython-3.12.14  correct

  It now sorts on the parsed patch number, and rejects a directory that matches the name
  but holds no `python.exe`. Same trap that nearly picked the wrong CUDA runtime when
  `2.9.1` sorted above `2.14.0` — twice in this codebase now, so `Sort-Object` on anything
  version-shaped is worth checking on sight.

**How that last one was found is the part worth keeping.** It is the only defect here that
fails silently: the three reporting defects produced a confusing transcript, while this one
produces a wrong runtime and reports success. Nobody was looking for it. It was found
because the cosmetic noise line immediately above it dragged somebody's eye to the pipeline
that emitted it. **The cosmetic bug was the symptom that led to the correctness bug** — which
is an argument for fixing transcript noise on sight rather than filing it as polish, since
noise is where a reader's attention goes and quiet wrong answers are what survive review.

The banner now names what was checked rather than asserting readiness, and says "a CUDA
kernel ran" rather than claiming the card is proven for production.

### The stage 8 deadline is sized against disk, not GPU

Both machines report the worker answering in **about 10 seconds** — VP1 on a GTX 1660 and
marp-laptop-install-test on an RTX 5060 Laptop. The faster card did not help, which it
would have if the wait were compute. Loading PyTorch is dominated by reading roughly 2.5 GB
of DLLs and initialising the runtime, and the GPU contributes nothing to that.

So the 180s deadline has headroom against *these two machines' SSDs* and nothing is known
about its headroom on a spinning disk, which is the volunteer most likely to strain it.
Neither of us can measure that, and neither machine's result should be read as though we
had. The prediction going in was that the faster card would answer sooner; it did not, and
the identical number on very different GPUs is stronger evidence for a disk bound than a
merely slower one would have been.

### A machine cannot test the branch its own configuration makes unreachable

Four separate gaps in this task turned out to be one rule, and stating it once is more
useful than listing them four times, because the rule predicts the next blind spot instead
of merely recording the last one.

- VP1 cannot exercise stage 2, because its VC++ runtime is already current, so the step
  is skipped.
- Neither machine can exercise the Inno GUI, because both run `/VERYSILENT`.
- marp-laptop-install-test cannot exercise the Python sort fix, because it holds exactly
  one managed 3.12 and the bug needs two.
- **VP1 cannot exercise the `Resolve-BuildPython` fallback, because it has a `py` launcher
  and the resolver never reaches the fallback.** This one is the sharpest, because the bug
  being fixed *lived in the fallback*: a green build on VP1 proves the resolver still works
  on the path that was never broken. The machine that needs the fix is the only machine that
  can test it.

The general form: **a green result from a machine whose configuration skips the code under
test is not evidence about that code.** It is worth asking, of every passing run, which
branch this machine's configuration makes unreachable — and routing that branch to a
machine that reaches it, or recording it as untested. Two green runs on machines that both
skip a step is not coverage of the step, however many times it is repeated.

And the reason it matters more than an ordinary gap: **running a test that cannot fail for
the reason you are testing is worse than not running one**, because it produces a green
result that then gets reported as verification. The `Resolve-BuildPython` fix was reported
here as verified on the strength of a VP1 build that never entered the repaired branch. The
gap was closed by running it on the machine with no `py` launcher, where the fallback is
the only path available:

    py absent - fallback WILL be exercised
    Building the wheel with ...\MARP\Worker\python\cpython-3.12.14-...\python.exe (Python 3.12.14).

That host confirmed the launcher's absence in the same run rather than asserting it, so the
branch that executed is demonstrable rather than assumed — which is the standard the rule
above is asking for.

### What is still not verified, and by whom

- **The VC++ install-and-elevate path has never executed.** VP1 has MSVC 14.44 and
  marp-laptop-install-test has 14.51, both at or above the 14.44 the installer ships, so
  stage 2 is skipped on every machine in the fleet. `WinError 1114` remains unreproduced.
  Two green runs on machines that both skip the step are not coverage of the step. This
  needs a clean Windows box and is Isaac's to place.
- **The Inno GUI wizard has never been seen.** Both agents have only run `/VERYSILENT`.
- **The driver floors remain published minimums**, recorded as `driver_floor_verified:
  false`. The one measured datapoint is cu128 working at driver 573.13.
- **No genuinely clean volunteer run** — double-clicked from Explorer on a machine that has
  never held a worker — has happened. Both runs were agent-driven.
- **The wrong-interpreter path is covered by the three-directory test only.** It misbehaves
  only on a host holding more than one managed 3.12, and neither machine does, so neither
  install exercises it. Cite the fabricated-directory comparison for that defect, not either
  run. What the second machine proves is that the transcript is clean and the upgrade path
  is intact on a different GPU architecture — which is what it was for.
- **The stage 8 deadline against slow storage**, per the section above.
- **The CPU path** cannot be meaningfully tested until marp-laptop's PR #42 merges; until
  then R2 is delivered by the installer and defeated by the worker.
- **Repeated clean installs accumulate orphaned workers** coordinator-side, because a fresh
  install on a machine whose state directory was removed enrols a new id. Fleet hygiene,
  not an installer bug, but it should be someone's decision rather than a surprise.

The pattern worth keeping: twice in this session the thing that caught an error was
somebody running it, not somebody reasoning about it. Both the stage 8 misreading and my
own `-ErrorAction Stop` non-fix were settled by a five-line script, not by argument.
