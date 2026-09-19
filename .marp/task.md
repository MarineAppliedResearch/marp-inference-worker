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

- [ ] **The CPU path needs a worker change that is not merged.** (cross-repository,
  `blocking`) `resolve_device` refuses `auto` with no CUDA by design (R14), so a CPU
  machine installs correctly, enrols, takes a job and fails it with `no usable CUDA
  device`. marp-laptop's fix is PR #42 on `cpu-only-workers`. Until it merges, R2 is
  delivered by the installer and defeated by the worker, and testing the CPU path against
  `develop` will show a failure that is not the installer's.

- [ ] **The cu126 and cu128 driver floors are published minimums, not measured ones.**
  (environment) Recorded as `driver_floor_verified: false`. Nobody in the fleet has a
  machine old enough to find where either actually stops. The one measured datapoint is
  cu128 working at driver 573.13, from marp-laptop.

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

**What is not yet verified, and by whom:**

- No installer has been built. `build-windows.ps1` needs Node for the player build, which
  is not installed on VP1.
- Nothing downstream of selection has been executed anywhere: the download path, the VC++
  step, activation, or the worker starting from an installed runtime.
- The CPU path cannot be meaningfully tested until PR #42 merges.
- Real-hardware runs are split by machine: VP1 has sm_75, ABYSS sm_89, marp-laptop sm_120
  on the old driver. Nobody has a Maxwell or Pascal card, or a machine with no GPU.
