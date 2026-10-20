# verify_training_run.py
# Created: 2026-10-20
# Author: Isaac Travers
#
# Manual verification for the training engine, on real hardware and real data.
# Everything else about training is covered by tests that run anywhere; this is
# the part that cannot be. `TrainingEngine.run()` needs a CUDA device, a real
# dataset archive and real starting weights, so what proves it is a run, and this
# is the script that performs one.
#
# It is deliberately not a test. It takes minutes, it needs a GPU and several
# gigabytes of disk, and it writes a model. Run it by hand and paste what it
# prints into .marp/verification.md.
#
#   python scripts/verify_training_run.py \
#       --archive <packaged .tar> --weights <starting .pt> [--epochs 2]
#
# What it checks, which is everything between the archive and the weights:
#   - the archive unpacks and its data.yaml is made true for this machine
#   - Ultralytics trains from it without being told anything else
#   - every epoch reports progress and metrics through the job context
#   - a best.pt appears and is published

# argparse takes the paths this has to be told.
import argparse

# json prints the collected events in a form worth pasting into a record.
import json

# time measures the run, since "did it train" and "how long" are one question.
import time

# Path types every location below.
from pathlib import Path

# Any types the free-form event payloads.
from typing import Any


# RecordingContext
# A JobContext that keeps what the engine reports instead of sending it.
# The real one writes newline-delimited JSON to stdout for the parent to read;
# this keeps the same calls so the engine cannot tell the difference, and prints
# a summary at the end.
class RecordingContext:

    # __init__()
    # Builds a context around a scratch directory.
    # Inputs: the directory the engine may write into.
    # Output: initialized RecordingContext.
    def __init__(self, checkpoint_dir: Path) -> None:

        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.params: dict[str, Any] = {}
        self.progress: list[dict[str, Any]] = []
        self.metrics: list[dict[str, Any]] = []
        self.logs: list[str] = []
        self.artifacts: list[dict[str, Any]] = []
        self.settings: dict[str, Any] = {}

    @property
    def checkpoint_dir(self) -> Path:
        return self._checkpoint_dir

    @property
    def resume_from(self) -> Path | None:
        return None

    def log(self, message: str, level: str = "info") -> None:
        self.logs.append(f"[{level}] {message}")
        print(f"    [{level}] {message}", flush=True)

    def report_progress(self, done, total, unit, phase=None, tracks=None) -> None:
        self.progress.append({"done": done, "total": total, "unit": unit, "phase": phase})
        print(f"    progress: {done}/{total} {unit} ({phase})", flush=True)

    def report_metrics(self, step: int, phase: str, metrics) -> None:
        self.metrics.append({"step": step, "phase": phase, "metrics": dict(metrics)})
        interesting = {k: v for k, v in metrics.items() if "precision" in k or "recall" in k or "loss" in k}
        print(f"    epoch {step}: {json.dumps(interesting, default=str)}", flush=True)

    def report_settings(self, engine, settings, ignored) -> None:
        self.settings = {"engine": engine, "settings": dict(settings), "ignored": dict(ignored)}

    def publish_artifact(self, path: Path, kind: str) -> str:
        # Hashed exactly as the real context does, so the figure printed here is
        # the one MARP would have been handed.
        from marp_inference_worker.jobs.context import hash_file

        digest = hash_file(Path(path))
        self.artifacts.append({
            "role": kind, "path": str(path), "sha256": digest,
            "size_bytes": Path(path).stat().st_size,
        })
        print(f"    published {kind}: {Path(path).name} sha256={digest[:16]}…", flush=True)
        return digest

    def should_stop(self) -> bool:
        return False

    def wait_while_paused(self, while_waiting=None) -> bool:
        return False


# main()
# Unpacks the dataset, runs the engine, and prints what happened.
# Inputs: command line arguments.
# Output: process exit code.
def main() -> int:

    parser = argparse.ArgumentParser(description="Run the training engine on real data.")
    parser.add_argument("--archive", required=True, help="a packaged dataset .tar")
    parser.add_argument("--weights", required=True, help="starting weights, a .pt file")
    parser.add_argument("--work", default=".marp/local/training-verification", help="scratch directory")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    from marp_inference_worker.engines.training_engine import TrainingEngine
    from marp_inference_worker.system.device import cuda_device_count
    from marp_inference_worker.training import ensure_dataset

    archive = Path(args.archive).resolve()
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)

    print(f"==> CUDA devices: {cuda_device_count()}", flush=True)
    if cuda_device_count() <= 0:
        print("    no GPU; the engine would refuse this job, which is the point of preflight")
        return 1

    # The archive is already on this machine, so the fetch is a copy. The unpack,
    # the hash check and the YAML rewrite are the real thing, which is what this
    # part is here to exercise.
    import hashlib
    print(f"==> hashing {archive.name}", flush=True)
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    sha256 = digest.hexdigest()
    print(f"    sha256 {sha256}", flush=True)

    def copy_download(url, destination, headers=None, on_progress=None):
        import shutil
        shutil.copyfile(archive, destination)

    print("==> unpacking into the shared dataset cache", flush=True)
    started = time.monotonic()
    dataset = ensure_dataset(
        url=str(archive),
        sha256=sha256,
        cache_root=work / "datasets",
        download=copy_download,
    )
    print(f"    {'unpacked' if dataset['fetched'] else 'already cached'} "
          f"in {time.monotonic() - started:.1f}s at {dataset['path']}", flush=True)
    print(f"    data.yaml:\n{Path(dataset['data_yaml']).read_text(encoding='utf-8')}", flush=True)

    ctx = RecordingContext(work / "attempt" / "work")
    spec = {
        "engine": "marp_training",
        "params": {
            "data_yaml": str(dataset["data_yaml"]),
            "model_path": str(Path(args.weights).resolve()),
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "slot_index": 0,
        },
    }

    engine = TrainingEngine()
    print("==> preflight", flush=True)
    engine.preflight(spec)
    print("    accepted", flush=True)

    print(f"==> training for {args.epochs} epochs", flush=True)
    summary = engine.run(ctx, spec)

    print("\n==> summary", flush=True)
    print(json.dumps(summary, indent=2, default=str), flush=True)

    print("\n==> what the engine reported", flush=True)
    print(f"    progress events: {len(ctx.progress)}", flush=True)
    print(f"    metric events:   {len(ctx.metrics)} (one per epoch)", flush=True)
    print(f"    artifacts:       {[a['role'] for a in ctx.artifacts]}", flush=True)
    print(f"    ignored params:  {ctx.settings.get('ignored')}", flush=True)

    ok = (
        summary.get("epochs_completed") == args.epochs
        and len(ctx.metrics) == args.epochs
        and any(a["role"] == "weights" for a in ctx.artifacts)
    )
    print(f"\n==> {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
