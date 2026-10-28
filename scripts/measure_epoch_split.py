# measure_epoch_split.py
# Created: 2026-10-28
# Author: Isaac Travers
#
# A manual measurement, NOT part of the job path and NOT run by the test suite.
# It answers one question: where does a training epoch's wall time go?
#
# **Never run. Parked 2026-10-28 with the question open**, because the live
# progress work had drifted from what was actually being asked for. The reasoning
# is in .marp/local/epoch-time-hypothesis.md; this is the instrument that would
# settle it.
#
# Takes a data.yaml and runs a real GPU training job, so it costs real GPU time
# and belongs to whoever is at the keyboard.

# Where does an epoch's wall time actually go?
#
# Job 21799 ran ~150s an epoch with 144 batches, yet every batch callback
# landed inside one 2.5s window. Either the batch loop is genuinely that fast
# -- implying 57 it/s while validation measures 5.8 on the same card, which is
# not believable -- or the Python loop races ahead of an async CUDA queue and
# the wall time sits somewhere with no callback.
#
# This times every boundary Ultralytics offers and prints the gaps, so the
# answer is measured rather than argued. It changes nothing about the engine
# and adds no synchronisation: a torch.cuda.synchronize() would cost real
# throughput and that is Isaac's decision, not a probe's.
import json
import sys
import time
from pathlib import Path


DATASET = sys.argv[1] if len(sys.argv) > 1 else None
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
IMGSZ = int(sys.argv[3]) if len(sys.argv) > 3 else 640
BATCH = int(sys.argv[4]) if len(sys.argv) > 4 else 16

if not DATASET:
    raise SystemExit("usage: probe_epoch_split.py <data.yaml> [epochs] [imgsz] [batch]")

# Every boundary, with the wall clock at which it happened.
marks: list[tuple[str, float]] = []
batch_times: list[float] = []


def mark(name):
    def hook(_obj):
        marks.append((name, time.time()))
    return hook


def on_batch(_trainer):
    batch_times.append(time.time())


from ultralytics import YOLO

model = YOLO("yolo11n.pt" if Path("yolo11n.pt").exists() else "yolo11n.yaml")
for event in ("on_train_epoch_start", "on_train_epoch_end", "on_val_start",
              "on_val_end", "on_model_save", "on_fit_epoch_end", "on_train_end"):
    model.add_callback(event, mark(event))
model.add_callback("on_train_batch_end", on_batch)

started = time.time()
model.train(
    data=DATASET, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=0,
    workers=8, verbose=False, plots=False, val=True,
    project=str(Path(__file__).parent / "splitrun"), name="run", exist_ok=True,
)

print("SPLIT_START")
print(f"# total wall time: {time.time() - started:.1f}s, {len(batch_times)} batch callbacks")

# Walk the marks in order and report each gap, attributing the batch loop to
# the span between the first and last batch callback inside each epoch.
previous_name, previous_at = "run start", started
for name, at in marks:
    gap = at - previous_at
    note = ""
    if name == "on_train_epoch_end":
        inside = [t for t in batch_times if previous_at <= t <= at]
        if inside:
            loop = inside[-1] - inside[0]
            note = (f"  [{len(inside)} batches spanning {loop:.1f}s"
                    f" -> {len(inside)/loop:.1f}/s if loop is real;"
                    f" {gap - loop:.1f}s AFTER the last callback]")
    print(f"{previous_name:>22s} -> {name:<22s} {gap:7.1f}s{note}")
    previous_name, previous_at = name, at

print("SPLIT_END")
