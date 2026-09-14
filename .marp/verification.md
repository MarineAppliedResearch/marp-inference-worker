---
task: MarineAppliedResearch/marp-inference-worker#18
status: awaiting-plan-review
---

## What this verifies

This is the installer and worker half of the second-computer pilot. It verifies a person can
open one small setup application on a supported Windows NVIDIA computer and reach a running,
separately enrolled worker without installing Python, Node, Git, a browser, CUDA toolkit, or
a compiler themselves.

## Focused automated checks

1. Run `pytest tests/test_installed_worker.py tests/test_coordinator_contract.py
   tests/test_job_runner.py`. These checks prove DPAPI protection, verified side-by-side
   update staging, path-traversal rejection, worker-bound artifact requests, and that the
   installer integration did not break job execution.
2. Parse both PowerShell scripts and compile the Inno Setup definition. Inspect the resulting
   build manifest and require the setup executable to remain below 20 MiB. Inspect its file
   table to confirm it does not contain Python, PyTorch, Chromium, model weights, or a completed
   worker runtime.
3. Corrupt one downloaded-component hash in an isolated copied payload and run only the
   download verifier. Setup must stop before extracting or executing those bytes. Restore the
   committed lock afterward.

## Real clean-machine installation

1. Uninstall any earlier MARP worker from the second Windows computer and ensure no worker,
   Python, Node, Git, Chrome/Chromium, CUDA toolkit, or C++ build tool from this workspace is
   available to the installer.
2. Create a one-time activation code through the issue #189 API, open the unsigned development
   setup application once, enter the API address and code, and make no other installation
   action. Record setup's component progress and any failure verbatim.
3. Confirm setup does not finish until local `/health` is OK, `/status` says enrolled, and the
   reported capabilities name the real NVIDIA GPU. Confirm the machine has its own worker row,
   its protected credential file does not contain the bearer text, and the sign-in startup
   shortcut launches visible watch mode by default.
4. Restart the installed worker through its stable launcher. It must reuse the same machine
   identity and credential and must update the existing worker row rather than create another.
5. Assign one real inference job through the API. The second computer must download and verify
   the registered model through MARP_API, reuse it from cache on a second request, run on CUDA,
   report progress/results, and open the existing player watch window with live boxes.
6. Revoke only this machine's token. Its next request must fail while the first computer's
   worker remains able to poll. Then issue a fresh activation code before further pilot use.

## Not covered

- The development setup is unsigned. A public volunteer build remains blocked on purchasing
  and configuring a Windows code-signing certificate.
- Automatic fleet rollout and rollback are not exercised here; the stable layout and release
  metadata are prepared for that later milestone.
- No production `mare_v1` mutation is performed.

## Results

Not run. Awaiting human review of this plan.
