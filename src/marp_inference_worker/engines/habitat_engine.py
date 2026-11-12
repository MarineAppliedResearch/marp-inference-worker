# habitat_engine.py
# Created: 2026-11-10
# Author: Isaac Travers
#
# Inference engine for habitat substrate models (MarineAppliedResearch/MARP_API#287).
# Reads a job's range of video, looks at it a few times a second, scores the habitat crop of
# each frame with the model, and turns the scores into habitat stretches by the decision rules
# MARP sends with the model. Publishes the stretches, which MARP writes as an automated Habitat
# session, and every frame's scores beside them, so different cut-offs can be tried later
# without the GPU (settled with Isaac on 2026-11-10).
# Decoding, scoring and publishing belong here; the stretch rules are reduction/habitat_stretches.

# json writes the result files.
import json

# time measures the run.
import time

# Path locates files in the job's own directory.
from pathlib import Path

# Types for the spec and the summary.
from typing import Any, Mapping

# numpy holds the scores.
import numpy as np

# The contract every engine is invoked through.
from marp_inference_worker.engines.base_engine import BaseEngine, JobContext, JobUnrunnable

# The architecture habitat models are built on.
from marp_inference_worker.engines.habitat_training_engine import DINOV2_ARCHITECTURE

# Reading the job's range of frames, and picking and cropping habitat frames.
from marp_inference_worker.media.frame_range_reader import iter_frame_range, open_video
from marp_inference_worker.media.habitat_frames import CropBox, crop_fraction, sample_by_time

# The stretch rules.
from marp_inference_worker.reduction.habitat_stretches import HABITAT_REDUCTION, stretches_for

# Device selection is the worker's decision, not the engine's.
from marp_inference_worker.system.device import cuda_device_count, resolve_device


# BATCH
# Frames scored together; 16 fits DINOv2 ViT-B at 518 px on an 8 GB card.
BATCH = 16

# RULE_FIELDS
# What a run's decision rules must carry for the stretches to be made.
RULE_FIELDS = ("substrates", "cut_offs", "crop", "input_size", "samples_per_second",
               "average_s", "min_stretch_s", "bridge_s")


# HabitatEngine
# Runs one habitat model over one range of one video.
class HabitatEngine(BaseEngine):

    # engine_name
    # The name a job spec selects this engine by; MARP's HABITAT_ENGINE.
    @property
    def engine_name(self) -> str:

        return "marp_habitat"

    # describe()
    # Says what this engine is and what it needs, for /status and MARP.
    def describe(self) -> dict[str, Any]:

        return {"engine": self.engine_name, "kind": "inference", "requires_gpu": True, "tasks": ["habitat"],
                "reduction": {"name": HABITAT_REDUCTION[0], "version": HABITAT_REDUCTION[1]}}

    # preflight()
    # Decides whether this machine can run this job at all.
    # Inputs: the job spec mapping.
    # Output: none on success; raises JobUnrunnable with a reason otherwise.
    def preflight(self, spec: Mapping[str, Any]) -> None:

        # The stretch rule the spec names must be this one; another would mislabel the output.
        reduction = spec.get("reduction") or {}
        if (str(reduction.get("name")), str(reduction.get("version"))) != HABITAT_REDUCTION:
            raise JobUnrunnable(f"the habitat engine applies {HABITAT_REDUCTION[0]} version {HABITAT_REDUCTION[1]}, "
                                f"not {reduction.get('name')} version {reduction.get('version')}")

        # The model's decision rules, which MARP sends with it.
        rules = (spec.get("model") or {}).get("decision_rules") or {}
        missing = [field for field in RULE_FIELDS if field not in rules]
        if missing:
            raise JobUnrunnable(f"the model's decision rules are missing {', '.join(missing)}")

        if not (spec.get("video") or {}).get("url"):
            raise JobUnrunnable("job spec carried no video.url")

        # A GPU, unless the job asked for the processor by name.
        params = spec.get("params") or {}
        if str(params.get("device", "auto")).lower() != "cpu" and cuda_device_count() == 0:
            raise JobUnrunnable("this worker has no usable CUDA device and the job did not request 'cpu'")

    # build_model()
    # The habitat model from its trained weights.
    # Inputs: the number of substrates, the input size and the weights file.
    # Output: the model, ready to score.
    # A method, so a test can substitute a small network for the real one.
    def build_model(self, classes: int, input_size: int, weights: Path) -> Any:

        # The trained weights are a whole state dict, so nothing pretrained is fetched.
        import timm
        import torch

        model = timm.create_model(DINOV2_ARCHITECTURE, pretrained=False, num_classes=classes, img_size=input_size,
                                  dynamic_img_pad=True, dynamic_img_size=True)
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        return model

    # run()
    # Scores one range of video and makes its stretches.
    # Inputs: the job context and the job spec.
    # Output: a JSON-safe summary; the stretches and scores go out through publish_artifact.
    def run(self, ctx: JobContext, spec: Mapping[str, Any]) -> dict[str, Any]:

        # torch and torchvision are needed from here on.
        import torch
        from PIL import Image
        from torchvision import transforms

        started = time.monotonic()
        params = dict(spec.get("params") or {})
        rules = dict(spec["model"]["decision_rules"])
        substrates = list(rules["substrates"])
        crop = CropBox(**{k: float(rules["crop"][k]) for k in ("x0", "y0", "x1", "y1")})
        video = dict(spec.get("video") or {})
        start_frame = int((spec.get("range") or {}).get("start_frame", 0))
        end_frame = int((spec.get("range") or {}).get("end_frame", 0))
        expected = max(end_frame - start_frame, 0)
        device = torch.device(resolve_device(params.get("device"), int(params.get("slot_index") or 0)))
        ctx.report_settings(self.engine_name, {
            "samples_per_second": rules["samples_per_second"], "cut_offs": rules["cut_offs"],
            "average_s": rules["average_s"], "min_stretch_s": rules["min_stretch_s"],
            "bridge_s": rules["bridge_s"], "exclusive": rules.get("exclusive") or [],
            "crop": dict(crop._asdict()), "input_size": rules["input_size"], "device": str(device),
        }, {})

        # The video, numbered on its nominal clock when the job gives it.
        ctx.report_progress(0, expected, "frames", phase="opening_video")
        capture, geometry = open_video(str(video["url"]))
        if video.get("frame_rate"):
            geometry = geometry._replace(frame_rate=float(video["frame_rate"]))
        ctx.log(f"opened video {geometry.width}x{geometry.height} at {geometry.frame_rate:.3f} fps")

        ctx.report_progress(0, expected, "frames", phase="loading_model")
        model = self.build_model(len(substrates), int(rules["input_size"]), Path(str(params["model_path"])))
        model.to(device).eval()

        # The preparation the model was trained with, without the training-only changes.
        size = int(rules["input_size"])
        prepare = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        frames: list[int] = []
        times: list[float] = []
        scores: list[np.ndarray] = []
        batch_frames: list[Any] = []
        stopped_early = False
        last_index = start_frame - 1

        # The last frame read, scored or not, for how far the range was covered.
        read_upto = [start_frame - 1]

        # Passes the reader's frames through, noting each one's number.
        def noting(frames_in):

            for f in frames_in:
                read_upto[0] = f.index
                yield f

        # Scores a batch and keeps its numbers.
        def score(batch: list[Any]) -> None:

            images = torch.stack([prepare(Image.fromarray(crop_fraction(f.image, crop)[:, :, ::-1].copy())) for f in batch])
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                out = torch.sigmoid(model(images.to(device))).float().cpu().numpy()
            for f, row in zip(batch, out):
                frames.append(f.index)
                times.append(f.time_s * 1000.0)
                scores.append(row)

        try:
            ctx.report_progress(0, expected, "frames", phase="inferring")
            for frame in sample_by_time(noting(iter_frame_range(capture, geometry, start_frame, end_frame)),
                                        float(rules["samples_per_second"])):
                batch_frames.append(frame)
                last_index = frame.index
                if len(batch_frames) < BATCH:
                    continue
                score(batch_frames)
                batch_frames = []
                ctx.report_progress(last_index - start_frame + 1, expected, "frames", phase="inferring")

                # A pause holds the model loaded; a stop finishes with what was scored.
                ctx.wait_while_paused()
                if ctx.should_stop():
                    ctx.log("stop requested; making stretches from what was scored", level="warning")
                    stopped_early = True
                    break
            if batch_frames:
                score(batch_frames)
        finally:
            capture.release()

        # The stretches, and every frame's scores beside them.
        ctx.report_progress(last_index - start_frame + 1, expected, "frames", phase="reducing")
        matrix = np.stack(scores) if scores else np.zeros((0, len(substrates)))
        stretches = stretches_for(frames, times, matrix, rules)
        workdir = Path(ctx.checkpoint_dir)
        source = {"video_source": video.get("source_name"), "jellyfin_item_id": video.get("jellyfin_item_id")}
        stretch_path = workdir / "habitat_stretches.jsonl"
        stretch_path.write_text("".join(json.dumps({**s, **source}) + "\n" for s in stretches), encoding="utf-8")
        score_path = workdir / "habitat_scores.jsonl"
        score_path.write_text("".join(
            json.dumps({"frame": f, "ms": round(t, 1), "scores": [round(float(v), 4) for v in row]}) + "\n"
            for f, t, row in zip(frames, times, matrix)), encoding="utf-8")

        ctx.report_progress(last_index - start_frame + 1, expected, "frames", phase="publishing")
        stretch_sha = ctx.publish_artifact(stretch_path, "habitat_stretches")
        score_sha = ctx.publish_artifact(score_path, "habitat_scores")

        return {
            "engine": self.engine_name,
            "device": str(device),
            "range": {"start_frame": start_frame, "end_frame": end_frame},
            # Half-open, as the species engine reports it: one past the last frame read.
            "processed_range": {"start_frame": start_frame, "end_frame": read_upto[0] + 1},
            "frames_scored": len(frames),
            "stretches": len(stretches),
            "results": {"kind": "habitat_stretches", "sha256": stretch_sha, "path": str(stretch_path),
                        "line_count": len(stretches)},
            "scores": {"sha256": score_sha, "line_count": len(frames)},
            "reduction": {"name": HABITAT_REDUCTION[0], "version": HABITAT_REDUCTION[1]},
            "substrates": substrates,
            "elapsed_s": round(time.monotonic() - started, 3),
            "stopped_early": stopped_early,
        }
