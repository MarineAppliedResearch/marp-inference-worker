# verify_habitat_training.py
# Created: 2026-11-10
# Author: Isaac Travers
#
# Manual verification for the habitat training engine, on real hardware and real data
# (MarineAppliedResearch/MARP_API#287). The tests prove the recipe and the engine on a
# stand-in network on a CPU; this proves the real thing: DINOv2 ViT-B/14 built from the
# weights MARP serves, trained on a real habitat dataset on a GPU, with its test scores
# set beside the feasibility runs'.
#
# It is deliberately not a test. It takes minutes to hours and needs a CUDA GPU. Run it
# by hand and paste what it prints into .marp/verification.md.
#
#   python scripts/verify_habitat_training.py --archive <unpacked habitat archive> \
#       --weights <dinov2 model.safetensors> [--epochs 10] [--head-epochs 3] [--out <dir>]
#
# What it checks, everything between the archive and the three published files:
#   - the archive is one the engine accepts (preflight)
#   - DINOv2 is built from the weights given, offline
#   - every pass reports through the job context
#   - weights, decision_rules and final_metrics are published, and say what they should

# argparse takes the paths this has to be told.
import argparse

# json prints what was published.
import json

# os keeps Hugging Face offline, as the job's child process does.
import os

# time measures the run.
import time

# Path types every location below.
from pathlib import Path

# Any types the free-form payloads.
from typing import Any


# RecordingContext
# A JobContext that prints what the engine reports and keeps what it publishes.
class RecordingContext:

    # A scratch directory for the engine, and empty records.
    def __init__(self, checkpoint_dir: Path) -> None:

        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: list[dict[str, Any]] = []
        self.published: dict[str, str] = {}

    @property
    def params(self) -> dict[str, Any]:

        return {}

    @property
    def checkpoint_dir(self) -> Path:

        return self._checkpoint_dir

    @property
    def resume_from(self) -> None:

        return None

    # Every line the engine logs, printed as it comes.
    def log(self, message: str, level: str = "info") -> None:

        print(f"[{level}] {message}", flush=True)

    def report_progress(self, done, total, unit, phase=None) -> None:

        print(f"progress {done}/{total} {unit} ({phase})", flush=True)

    def report_metrics(self, step, phase, metrics) -> None:

        self.metrics.append({"step": step, "phase": phase, **dict(metrics)})

    def report_settings(self, engine, settings, ignored) -> None:

        print(f"settings {engine}: {json.dumps(dict(settings))}; ignored {json.dumps(dict(ignored))}", flush=True)

    # A published file is kept where it was written, and named here.
    def publish_artifact(self, path, kind) -> str:

        self.published[kind] = str(path)
        return f"local:{kind}"

    def should_stop(self) -> bool:

        return False

    def wait_while_paused(self, while_waiting=None) -> bool:

        return False


# main()
# Runs one habitat training on this machine's GPU and prints the record.
def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, help="an unpacked habitat dataset archive")
    parser.add_argument("--weights", required=True, help="DINOv2 ViT-B/14 model.safetensors, as MARP serves it")
    parser.add_argument("--epochs", type=int, default=None, help="fine-tune passes (default: the recipe's 10)")
    parser.add_argument("--head-epochs", type=int, default=None, help="frozen-head passes (default: the recipe's 3)")
    parser.add_argument("--out", default="habitat-training-run", help="where the run's files go")
    args = parser.parse_args()

    # Offline, as the worker's job process is.
    os.environ["HF_HUB_OFFLINE"] = "1"

    # Imported after the environment is set, so timm never reaches the network.
    from marp_inference_worker.engines.habitat_training_engine import HabitatTrainingEngine

    params: dict[str, Any] = {"dataset_path": args.archive, "model_path": args.weights}
    if args.epochs is not None:
        params["epochs"] = args.epochs
    if args.head_epochs is not None:
        params["head_epochs"] = args.head_epochs
    spec = {"engine": "marp_habitat_training", "params": params}

    engine = HabitatTrainingEngine()
    engine.preflight(spec)
    ctx = RecordingContext(Path(args.out))
    started = time.monotonic()
    summary = engine.run(ctx, spec)

    # The record: every pass, the summary, and the three published files' contents.
    print("\n== passes")
    for row in ctx.metrics:
        print(json.dumps(row))
    print("\n== summary")
    print(json.dumps(summary, indent=1))
    for role in ("decision_rules", "final_metrics"):
        if role in ctx.published:
            print(f"\n== {role}")
            print(Path(ctx.published[role]).read_text(encoding="utf-8"))
    print(f"\nweights: {ctx.published.get('weights')}")
    print(f"took {time.monotonic() - started:.0f} s")


if __name__ == "__main__":
    main()
