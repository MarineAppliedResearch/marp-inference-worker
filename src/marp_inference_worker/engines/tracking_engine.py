# tracking_engine.py
# Created: 2026-09-09
# Author: Isaac Travers
#
# MARP tracking engine for the MARP Inference Worker.
# This is the engine that runs MARP's actual pipeline: YOLO detection per frame,
# ByteTrack assigning track ids, and each finished track reduced to keyframes
# labelled start/middle/end (R10). It is a port of the live path in
# object_tracking_live.py, behind the run(ctx, spec) contract.
#
# Everything MARP-specific about the *shape* of the output is here or in
# tracking/observations.py. Nothing about MARP's transport is: this engine has no
# idea what a coordinator, a token or an attempt is. Results are written to a
# file and handed over by hash through ctx.publish_artifact() (R11).

# json writes the results file, one observation per line.
import json

# Path types the results file and the model artifact.
from pathlib import Path

# Any and Mapping type the spec and the free-form params.
from typing import Any, Mapping

# The engine contract, and the two ways a job can end other than success.
from marp_inference_worker.engines.base_engine import (
    BaseEngine,
    JobContext,
    JobUnrunnable,
)

# The Ultralytics adapter supplies detection and the frame generator.
from marp_inference_worker.engines.ultralytics_engine import UltralyticsEngine

# Frame-range decoding, one frame at a time.
from marp_inference_worker.media.frame_range_reader import iter_frame_range, open_video

# The keyframe reduction registry, so the spec chooses the rule.
from marp_inference_worker.reduction import keyframes as keyframe_reduction

# ByteTrack construction and the IoU used to attach a class to a track.
from marp_inference_worker.tracking.byte_tracker_adapter import (
    calculate_iou,
    create_tracker,
    detections_to_array,
)

# Observation shaping, in the terms MARP already stores.
from marp_inference_worker.tracking.observations import build_observation

# Track gathering, ageing out, and the range seam.
from marp_inference_worker.tracking.track_accumulator import TrackAccumulator

# Device resolution, which refuses rather than falling back to CPU.
from marp_inference_worker.system.device import cuda_device_count, resolve_device


# How often, in frames, progress is reported and the stop flag is consulted.
# Per-frame reporting would be one event per frame, which for an hour of video at
# 30fps is 108,000 events. Every 30 frames is about one per second of video.
_PROGRESS_EVERY_FRAMES = 30


# IoU above which a track's box is considered to be the same thing as a
# detection's box, so the detection's class can be attached to the track.
# From the live script, which used 0.4.
_CLASS_MATCH_IOU = 0.4


# Ultralytics predict() settings a job may set, with the type each is coerced to.
#
# **An allow list rather than a pass-through.** A job spec is submitted data, and
# handing it straight to predict() as keyword arguments would let a typo become a
# TypeError on the volunteer's machine halfway through a video, or let something
# unrelated -- `save`, `project`, `show` -- write files on it.
#
# These are the settings `object_tracking_live.py` actually varied, which is the
# reason for the list rather than a guess at what might be useful:
#
#     rockfish:  conf 0.60,  imgsz 1280, iou 0.2, augment True, agnostic_nms True
#     deepsea:   conf 0.001, imgsz 1280, iou 0.2, augment True, agnostic_nms True
#
# `conf` is deliberately absent: confidence has its own argument and its own
# default, and having two ways to set it is how they end up disagreeing.
_INFERENCE_OPTIONS: dict[str, Any] = {
    "imgsz": int,
    "iou": float,
    "augment": bool,
    "agnostic_nms": bool,
    "max_det": int,
    "half": bool,
}


# inference_options_from_params()
# Picks the Ultralytics predict() settings a job asked for out of its params.
# Inputs: the job's params mapping.
# Output: a dict of recognised settings, coerced, ready to pass to predict().
#
# Anything unrecognised is dropped silently here and reported by the caller, so
# a job that sets a key the worker does not honour is visible rather than
# quietly ignored -- which is the failure this whole path had before.
def inference_options_from_params(params: Mapping[str, Any]) -> dict[str, Any]:

    options: dict[str, Any] = {}

    for name, coerce in _INFERENCE_OPTIONS.items():
        if name not in params or params[name] is None:
            continue

        # A bad value is worth failing on rather than silently dropping: it means
        # the job asked for something specific and would otherwise run as though
        # it had not.
        try:
            options[name] = coerce(params[name])
        except (TypeError, ValueError) as error:
            raise JobUnrunnable(
                f"job spec params.{name} is not a valid {coerce.__name__}: {params[name]!r}"
            ) from error

    return options


# The type each reported setting is recorded under -- the same types MARP's
# settings catalogue declares for this engine (MARP_API#232). A report whose type
# disagrees with the catalogue is refused, so these are not free to drift.
_SETTING_TYPES: dict[str, str] = {
    "confidence": "real",
    "imgsz": "int",
    "iou": "real",
    "augment": "bool",
    "agnostic_nms": "bool",
    "max_det": "int",
    "half": "bool",
    "track_thresh": "real",
    "match_thresh": "real",
    "track_buffer": "int",
    "mot20": "bool",
    "class_match_iou": "real",
}


# _typed()
# Puts a value into the type its setting is recorded under.
# Inputs: the value that ran, and one of real, int, bool, text.
# Output: the value in that type, or unchanged when it cannot honestly convert.
#
# Converting rather than reporting the raw JSON type, because what is recorded
# is what ran: ByteTrack reads `mot20` by truthiness, and a whole `track_buffer`
# of 300.0 behaves as 300. A value that cannot convert is left as it is, so the
# coordinator refuses it loudly rather than this guessing.
def _typed(value: Any, value_type: str) -> Any:

    if value_type == "bool":
        return bool(value)

    if value_type == "int":
        # A checkpoint can carry imgsz as [h, w]; a square one is one number.
        if isinstance(value, (list, tuple)) and value and len(set(value)) == 1:
            value = value[0]
        if isinstance(value, bool):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value

    if value_type == "real":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return value

    return str(value)


# inference_settings_report()
# Says exactly how this attempt will run, one setting at a time (MARP_API#232).
# Inputs: the job's params, the confidence resolved from them, the predict
#   options it set, the tracker's effective settings, the loaded model's own
#   overrides, and the param keys nothing honours.
# Output: `(settings, ignored)` -- settings as {name: {value, type, source}},
#   ignored as {key: the value the job asked for}.
#
# **A default is resolved the way Ultralytics resolves it, not assumed.**
# `Model.predict()` merges `{**model.overrides, **custom, **kwargs}`, and a loaded
# `.pt` keeps its training `imgsz` in those overrides. So a job that sets no
# imgsz runs at the model's size -- often 1280 here -- not DEFAULT_CFG's 640, and
# recording 640 would be exactly the false record this exists to prevent. Order:
# what the job set, then the model's overrides, then DEFAULT_CFG.
def inference_settings_report(
    params: Mapping[str, Any],
    confidence: float,
    predict_options: Mapping[str, Any],
    tracker_settings: Mapping[str, Any],
    model_overrides: Mapping[str, Any],
    unknown: list[str],
    ultralytics_defaults: Mapping[str, Any] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:

    # DEFAULT_CFG is what predict() falls back to for anything neither the job
    # nor the model supplied. Injectable so a test need not import Ultralytics.
    if ultralytics_defaults is None:
        from ultralytics.cfg import DEFAULT_CFG
        ultralytics_defaults = vars(DEFAULT_CFG)

    # The report, one entry per setting, in the shape MARP validates.
    settings: dict[str, dict[str, Any]] = {}

    # record()
    # Adds one setting to the report under its catalogue type.
    # Inputs: the setting's name, the value that ran, and where it came from.
    # Output: none.
    def record(name: str, value: Any, source: str) -> None:

        value_type = _SETTING_TYPES[name]
        settings[name] = {"value": _typed(value, value_type), "type": value_type, "source": source}

    # Confidence has its own argument and its own engine default of 0.15.
    set_by_job = params.get("confidence") is not None or params.get("conf") is not None
    record("confidence", confidence, "job" if set_by_job else "default")

    # The predict() settings a job may set.
    for name in _INFERENCE_OPTIONS:
        if name in predict_options:
            record(name, predict_options[name], "job")
        elif name in model_overrides:
            record(name, model_overrides[name], "default")
        else:
            record(name, ultralytics_defaults.get(name), "default")

    # ByteTrack's four: effective values, which already merge the job over the
    # engine's defaults. A key the job sent is the job's, whatever it was. Only
    # what the tracker actually holds is recorded -- TrackerArgs always holds all
    # four, and a setting it lacked did not run.
    for name in ("track_thresh", "match_thresh", "track_buffer", "mot20"):
        if name not in tracker_settings:
            continue
        source = "job" if params.get(name) is not None else "default"
        record(name, tracker_settings[name], source)

    # Not settable by a job, but it decides every track's species.
    record("class_match_iou", _CLASS_MATCH_IOU, "engine")

    # What the job asked for that nothing here honours, with what it asked for.
    ignored = {key: params[key] for key in unknown}

    return settings, ignored


# TrackingEngine
# Runs MARP's detect -> track -> reduce pipeline over one frame range.
# One instance per job, in the job's own child process, so its tracker and model
# state belong to that job alone and need no locking or unload coordination.
class TrackingEngine(BaseEngine):

    # __init__()
    # Builds the engine and its detection adapter.
    # Inputs: none.
    # Output: initialized TrackingEngine.
    def __init__(self) -> None:

        # Detection is delegated so all Ultralytics API usage stays in one file.
        self._detector = UltralyticsEngine()

    # engine_name
    # Returns the public engine name used in a job spec's `engine` field.
    @property
    def engine_name(self) -> str:

        # Named for the pipeline, not the library, because the pipeline is the
        # thing MARP depends on and the library underneath it may change.
        return "marp_tracking"

    # describe()
    # Returns what this engine is and what it needs.
    # Inputs: none.
    # Output: JSON-safe mapping for the worker's /status and capabilities.
    def describe(self) -> dict[str, Any]:

        # Versions are read from what is installed, not written in as literals.
        return {
            "engine": self.engine_name,
            "pipeline": "detect -> track -> reduce",
            "detector": "ultralytics",
            "ultralytics_version": UltralyticsEngine.ultralytics_version(),
            "tracker": "bytetrack",
            "requires_gpu": True,
            "reductions": keyframe_reduction.available_reductions(),
        }

    # preflight()
    # Decides whether this job can run on this machine, before any work starts.
    # Inputs: the job spec mapping.
    # Output: none on success; raises JobUnrunnable with a reason otherwise.
    # Use this to turn "would have died half way through" into "refused, and
    # here is why" (R14). Everything checked here is cheap and local.
    def preflight(self, spec: Mapping[str, Any]) -> None:

        # The reduction the spec names must be one this worker actually has.
        # Falling back to another rule would mislabel the output.
        reduction = spec.get("reduction") or {}
        try:
            keyframe_reduction.get_reduction(
                str(reduction.get("name")), str(reduction.get("version"))
            )
        except KeyError as error:
            raise JobUnrunnable(str(error)) from error

        # Ultralytics has to be installed, or nothing can be detected.
        if UltralyticsEngine.ultralytics_version() is None:
            raise JobUnrunnable("ultralytics is not installed on this worker")

        # ByteTrack has to be importable. It is a vendored source tree rather
        # than a wheel, so its absence is a deployment mistake worth naming.
        try:
            create_tracker({})
        except Exception as error:
            raise JobUnrunnable(f"bytetrack is unavailable on this worker: {error}") from error

        # A GPU is required unless the job explicitly asked for CPU.
        params = spec.get("params") or {}
        requested_device = params.get("device", "auto")
        if str(requested_device).lower() != "cpu" and cuda_device_count() == 0:
            raise JobUnrunnable(
                "this worker has no usable CUDA device and the job did not request 'cpu'"
            )

    # run()
    # Runs the pipeline over the job's frame range.
    # Inputs: a JobContext and the job spec mapping.
    # Output: a JSON-safe summary. The observations themselves go out as a
    # published artifact, never in this return value (R11).
    def run(self, ctx: JobContext, spec: Mapping[str, Any]) -> dict[str, Any]:

        # Everything the job needs is in the spec; nothing is read from config.
        params = dict(spec.get("params") or {})
        frame_range = spec["range"]
        start_frame = int(frame_range["start_frame"])
        end_frame = int(frame_range["end_frame"])
        expected_frames = end_frame - start_frame

        # The slot the runner pinned this job to, and the device it resolves to.
        #
        # `auto` on a machine with no CUDA device now resolves to the processor
        # rather than refusing, so a volunteer with integrated graphics can
        # still donate. A job that asked for a GPU by name is still refused.
        slot_index = int(params.get("slot_index", 0))
        device = resolve_device(params.get("device"), slot_index)

        if device == "cpu" and str(params.get("device") or "auto").lower() == "auto":
            # Said as a warning, not a log line, because this reaches `/status`
            # where a volunteer can see it. A machine quietly running ten or
            # twenty times slower than the person expects is the sort of thing
            # they conclude is broken.
            ctx.log(
                f"no usable CUDA device on this worker, so slot {slot_index} is "
                "running on the processor; this is much slower than a GPU",
                level="warning",
            )
        else:
            ctx.log(f"resolved device {device} for slot {slot_index}")

        # The reduction rule, chosen by the spec and recorded with every row.
        reduction_ref = spec["reduction"]
        reduce = keyframe_reduction.get_reduction(
            str(reduction_ref["name"]), str(reduction_ref["version"])
        )

        # Where the frames come from. The coordinator resolved the video and
        # the spec carries a source this engine can open, so the engine never
        # learns what that source is nor how it was arrived at (A8).
        video_source = str((spec.get("video") or {}).get("url") or "")
        if not video_source:
            raise JobUnrunnable("job spec carried no video.url")

        # The model artifact, already fetched and hash-verified by the runner.
        model_path = Path(str(params["model_path"]))

        # Detection confidence. The live pipeline ran low on purpose: MARP's
        # reviewers reject false positives cheaply, and a false negative costs
        # somebody re-reviewing a great deal of video.
        #
        # **`conf` is accepted as well as `confidence`.** Every note and script
        # MARP's settings come from writes `conf`, because that is what
        # Ultralytics calls it, and a job spec that used it was silently running
        # at the 0.15 default instead -- which looks exactly like the settings
        # having no effect.
        confidence = float(params.get("confidence", params.get("conf", 0.15)))

        # The rest of what the job asked predict() for: imgsz, iou, augment and
        # so on. Filtered to an allow list, so a key the worker does not honour
        # is dropped here and reported below rather than ignored in silence.
        predict_options = inference_options_from_params(params)

        # Say what is being applied, including the settings nothing honours.
        # Until this existed there was no way -- in the logs or in the database --
        # to tell a job that ran with the MBARI deep-sea settings from one that
        # asked for them and ran at the defaults. See MARP_API#232.
        ignored = sorted(
            key for key in params
            if key in {"conf", "confidence", "track_thresh", "match_thresh", "track_buffer",
                       "mot20", "data_type", "slot_index", "slot_count", "device", "watch"}
            or key.startswith("_") or key.startswith("model_")
        )
        unknown = sorted(set(params) - set(ignored) - set(predict_options))

        ctx.log(
            "inference settings: confidence=%s %s%s"
            % (
                confidence,
                " ".join(f"{k}={v}" for k, v in sorted(predict_options.items())) or "(ultralytics defaults)",
                f"  IGNORED: {', '.join(unknown)}" if unknown else "",
            )
        )

        # Dataset type drives which survey rule picks the observation frame.
        data_type = str(params.get("data_type", "Fish"))

        watch = None
        # `_watch_screen_mode` is the whole decision. The runner sets it only
        # when the machine is in a display mode, so a second check against the
        # job's own `watch` flag would put the veto back where it does not
        # belong -- with whoever queued the work rather than with the volunteer.
        screen_mode = str(params.get("_watch_screen_mode", "off"))
        wants_watch = screen_mode in {"window", "fullscreen"}

        # Open the video and read its geometry once.
        ctx.report_progress(0, expected_frames, "frames", phase="opening_video")
        capture, geometry = open_video(str(video_source))
        ctx.log(
            f"opened video {geometry.width}x{geometry.height} at {geometry.frame_rate:.3f} fps, "
            f"container reports {geometry.total_frames} frames"
        )

        # Number frames on the video's nominal clock when the job gives it. The
        # decoder's rate is an average, which a timestamp gap pulls below it.
        nominal_rate = (spec.get("video") or {}).get("frame_rate")
        if nominal_rate:
            geometry = geometry._replace(frame_rate=float(nominal_rate))
            ctx.log(f"numbering frames at the nominal {geometry.frame_rate:.3f} fps")

        # Build the model, the tracker and the accumulator for this range only.
        # A fresh tracker per range is what makes the boundary a seam (R10a).
        ctx.report_progress(0, expected_frames, "frames", phase="loading_model")

        # Hash the file that is about to be loaded, not the one that was
        # fetched.
        #
        # The cache verifies an artifact when it puts it there and on every hit,
        # which is real -- but it verifies *a path*, and loading is a separate
        # step afterwards. Nothing proved the engine opened the file that was
        # checked. Hashing here closes that gap for a few seconds on a 136 MB
        # file, which is nothing against publishing a run made by the wrong
        # weights.
        # Imported here rather than at module scope: `models` pulls in the
        # engine registry, which imports this module, and the cycle breaks the
        # package at import time.
        from marp_inference_worker.models import model_cache

        expected_sha = str((spec.get("model") or {}).get("sha256") or "")
        loaded_sha = model_cache.verify_sha256(Path(model_path), expected_sha)             if expected_sha else None

        yolo_model = self._detector.load_weights(model_path, device)

        # And check the vocabulary against what MARP registered for this model.
        #
        # This is the one thing that catches weights which are not the weights
        # the job asked for. A hash proves the *file* is right; this proves what
        # came out of it is. They are not the same claim, and the evening this
        # was written the two disagreed on screen with nothing to notice it.
        loaded_names = sorted(str(name) for name in (yolo_model.names or {}).values())
        declared = (spec.get("model") or {}).get("class_names")

        if declared:
            expected_names = sorted(str(name) for name in declared)

            if loaded_names != expected_names:
                raise JobUnrunnable(
                    "the loaded model is not the model this job asked for: "
                    f"weights contain {len(loaded_names)} classes {loaded_names}, "
                    f"but MARP registered {len(expected_names)} for "
                    f"{(spec.get('model') or {}).get('name')!r}: {expected_names}"
                )
        else:
            # A model with no registered species is a seeding gap, not a wrong
            # model. Say so plainly rather than refusing work over it.
            ctx.log(
                "MARP declared no class names for this model, so the loaded "
                "weights could not be checked against it",
                level="warning",
            )

        # Record what actually ran, so "which weights made this?" is a query
        # rather than somebody's recollection.
        ctx.log(
            f"loaded model {(spec.get('model') or {}).get('name')!r} "
            f"sha256={loaded_sha or 'unverified'} "
            f"classes={loaded_names}",
            level="info",
        )
        if wants_watch:
            from marp_inference_worker.watch import WatchDisplay

            model_names = yolo_model.names
            species_names = [str(model_names[key]) for key in sorted(model_names)]
            watch = WatchDisplay(
                screen_mode=screen_mode,
                workspace=ctx.checkpoint_dir.parent,
                warn=lambda message: ctx.log(message, level="warning"),
                job_id=str(params.get("_job_id") or "") or None,
                attempt_id=str(params.get("_attempt_id") or "") or None,
                model_name=str((spec.get("model") or {}).get("name") or "") or None,
                species_names=species_names,
                # So the window can be tiled beside its siblings rather than
                # opened on top of them. Both come from the runner, which is the
                # only thing that knows how many jobs this machine runs at once.
                slot_index=int(params.get("slot_index") or 0),
                slot_count=int(params.get("slot_count") or 1),
            )
            if not watch.start():
                watch = None
        tracker, tracker_args = create_tracker(params)
        accumulator = TrackAccumulator(track_buffer=tracker_args.as_dict["track_buffer"])

        # Say exactly how this attempt runs, before the first frame, so an
        # attempt that fails later still has its record (MARP_API#232). The log
        # line above is for people; this is the record, one setting at a time.
        #
        # A fault in building the record is a warning, never a failed job: the
        # record is bookkeeping, and losing a volunteer's GPU run over it would
        # cost more than the record is worth.
        try:
            settings, ignored_params = inference_settings_report(
                params,
                confidence,
                predict_options,
                tracker_args.as_dict,
                getattr(yolo_model, "overrides", None) or {},
                unknown,
            )
            ctx.report_settings(self.engine_name, settings, ignored_params)
        except Exception as error:
            ctx.log(
                f"could not report this attempt's settings: {type(error).__name__}: {error}",
                level="warning",
            )

        # Results go to a file in this attempt's own workspace.
        results_path = ctx.checkpoint_dir / "observations.jsonl"

        # Counters for the summary and for progress.
        frames_processed = 0
        observations_written = 0
        detections_seen = 0
        stopped_early = False

        # Distinct tracks this job has seen -- one per animal, however many
        # frames it was visible for.
        #
        # Ids rather than a running sum: a track lives across frames, so adding
        # up per-frame counts would count one fish once for every frame it
        # appeared on. `set` because the tracker hands back the same id each
        # frame it holds the animal, and ids are unique within a job because the
        # tracker is built fresh for each range.
        track_ids_seen: set[Any] = set()

        try:
            # One line per observation, written as tracks finish, so a long job
            # never holds all its results in memory either.
            with results_path.open("w", encoding="utf-8") as results_file:

                # Detection is a generator over a generator: the frame reader
                # yields one decoded frame, the detector yields its boxes, and
                # neither retains what came before (R8).
                ctx.report_progress(0, expected_frames, "frames", phase="seeking")
                frame_stream = iter_frame_range(
                    capture,
                    geometry,
                    start_frame,
                    end_frame,
                    on_seek_complete=lambda: ctx.report_progress(
                        0, expected_frames, "frames", phase="inferring"
                    ),
                )
                for frame_detections in self._detector.infer_stream(
                    yolo_model, frame_stream, confidence, predict_options
                ):
                    frame = frame_detections.frame
                    detections = frame_detections.detections
                    detections_seen += len(detections)

                    # Hand this frame's boxes to ByteTrack, which returns the
                    # tracks it believes are present now.
                    tracked = tracker.update(
                        detections_to_array(detections),
                        [geometry.height, geometry.width],
                        (geometry.height, geometry.width),
                    )

                    # Record each track's box on this frame.
                    live_tracks = self._observe_tracks(
                        accumulator=accumulator,
                        tracked=tracked,
                        detections=detections,
                        yolo_model=yolo_model,
                        frame_index=frame.index,
                        frame_time_s=frame.time_s,
                        frame_width=geometry.width,
                        frame_height=geometry.height,
                    )

                    track_ids_seen.update(track["track_id"] for track in live_tracks)

                    if watch is not None and not watch.present(
                        frame,
                        frame.index,
                        live_tracks,
                        range_start=start_frame,
                        range_end=end_frame,
                    ):
                        watch.close()
                        watch = None

                    # Close and write out any track the tracker has lost.
                    for ended in accumulator.take_aged_out(frame.index):
                        observations_written += self._write_observation(
                            results_file=results_file,
                            ended=ended,
                            reduce=reduce,
                            reduction_ref=reduction_ref,
                            spec=spec,
                            frame_rate=geometry.frame_rate,
                            data_type=data_type,
                        )

                    frames_processed += 1

                    # Report progress and check for a stop on the same cadence.
                    if frames_processed % _PROGRESS_EVERY_FRAMES == 0:
                        ctx.report_progress(frames_processed, expected_frames, "frames",
                                            tracks=len(track_ids_seen))
                        ctx.report_metrics(
                            step=frame.index,
                            phase="inference",
                            metrics={
                                "active_tracks": accumulator.active_track_count,
                                "observations": observations_written,
                                "detections": detections_seen,
                            },
                        )

                        # Cancellation arrives here and nowhere else. Breaking
                        # rather than raising means the tracks gathered so far
                        # are still closed and written below, so a cancelled job
                        # reports partial work honestly instead of losing it.
                        if ctx.should_stop():
                            ctx.log("stop requested; closing tracks and finishing", level="warning")
                            stopped_early = True
                            break

                # The range has ended -- either at end_frame, because the video
                # was shorter, or because a stop was requested. Every track
                # still open ends here. It is not carried forward and it is not
                # stitched to whatever the next range finds (R10a).
                ctx.report_progress(
                    frames_processed, expected_frames, "frames", phase="reducing"
                )
                for ended in accumulator.finish_range():
                    observations_written += self._write_observation(
                        results_file=results_file,
                        ended=ended,
                        reduce=reduce,
                        reduction_ref=reduction_ref,
                        spec=spec,
                        frame_rate=geometry.frame_rate,
                        data_type=data_type,
                    )

            # Report the final frame count, which the cadence above may have
            # skipped past, before handing the file over.
            ctx.report_progress(
                frames_processed, expected_frames, "frames", phase="publishing"
            )

            # Hand the results file over by hash. The coordinator is told the
            # hash and asks for the bytes only if it does not already have them.
            results_sha256 = ctx.publish_artifact(results_path, "observations")

            # The summary is small and safe to send inline; the observations are
            # not, and are not in it.
            return {
                "engine": self.engine_name,
                "device": device,
                "range": {"start_frame": start_frame, "end_frame": end_frame},
                "frames_expected": expected_frames,
                "frames_processed": frames_processed,
                "frames_short_of_range": expected_frames - frames_processed,
                "detections_seen": detections_seen,
                "observations": observations_written,
                "tracks_discarded_too_short": accumulator.discarded_track_count,
                "results": {
                    "kind": "observations",
                    "sha256": results_sha256,
                    "path": str(results_path),
                    "line_count": observations_written,
                },
                "reduction": dict(reduction_ref),
                "tracker": tracker_args.as_dict,
                "confidence": confidence,
                "stopped_early": stopped_early,
            }

        finally:
            # Release the capture and the model whatever happened, so a failed
            # job does not leave a stream open or GPU memory held until the
            # child process is reaped.
            capture.release()
            self._detector.unload_all()
            if watch is not None:
                watch.close()

    # _observe_tracks()
    # Records this frame's tracked boxes into the accumulator.
    # Inputs: the accumulator, ByteTrack's tracks, this frame's detections, the
    # model handle for its class names, and the frame's index, time and size.
    # Output: none.
    # Use this once per frame. ByteTrack tracks boxes and does not carry a class
    # through, so each track's class is recovered by matching its box back to
    # the detection that produced it -- the same IoU match the live script used.
    def _observe_tracks(
        self,
        accumulator: TrackAccumulator,
        tracked: Any,
        detections: list[dict[str, Any]],
        yolo_model: Any,
        frame_index: int,
        frame_time_s: float,
        frame_width: int,
        frame_height: int,
    ) -> list[dict[str, Any]]:

        live_tracks = []
        # Walk the tracks the tracker believes are present on this frame.
        for track in tracked:
            x1, y1, x2, y2 = (float(value) for value in track.tlbr)

            # Find the detection this track's box overlaps most, above the
            # threshold, and take its class. Matching per frame rather than
            # caching per track means a track that drifts onto a different
            # animal is labelled by what is actually there.
            best_iou = 0.0
            class_id = -1
            # None, not 0.0: no detection behind this track on this frame is a
            # different fact from a detection that scored zero, and the two must
            # not arrive at the observation looking the same.
            confidence: float | None = None
            for detection in detections:
                iou = calculate_iou((x1, y1, x2, y2), detection["bbox_xyxy"])
                if iou > best_iou:
                    best_iou = iou
                    class_id = detection["class_id"]
                    confidence = detection["confidence"]

            # Below the threshold the track is a prediction with no detection
            # behind it on this frame, so its class is not known.
            if best_iou <= _CLASS_MATCH_IOU:
                class_id = -1
                confidence = None

            # Resolve the class name through the model's own names.
            class_name = "Unknown"
            if class_id != -1:
                class_name = yolo_model.names.get(class_id, str(class_id))

            track_id = int(track.track_id)

            # Normalize to centre form in 0..1, which is what the keyframe
            # reduction's thresholds are calibrated against.
            accumulator.observe(
                track_id=track_id,
                class_name=class_name,
                frame_index=frame_index,
                frame_time_s=frame_time_s,
                bbox_normalized=(
                    (x1 + x2) / 2 / frame_width,
                    (y1 + y2) / 2 / frame_height,
                    (x2 - x1) / frame_width,
                    (y2 - y1) / frame_height,
                ),
                confidence=confidence,
            )

            live_tracks.append(
                {
                    "track_id": track_id,
                    # The species this track will be recorded as, decided from
                    # every frame so far, so the label does not flip with one
                    # frame's opinion (#49).
                    "class_name": accumulator.current_species(track_id),
                    # ByteTrack carries its score on every returned track,
                    # including frames propagated by the tracker.
                    "confidence": float(track.score),
                    "bbox_normalized": [
                        (x1 + x2) / 2 / frame_width,
                        (y1 + y2) / 2 / frame_height,
                        (x2 - x1) / frame_width,
                        (y2 - y1) / frame_height,
                    ],
                }
            )

        return live_tracks

    # _write_observation()
    # Reduces one finished track and writes its observation to the results file.
    # Inputs: the open results file, the ended track, the reduction callable and
    # its reference, the job spec, the frame rate and the dataset type.
    # Output: 1 when a row was written, 0 when the reduction produced nothing.
    # Use this from both closers so the reduce-and-write path exists once.
    def _write_observation(
        self,
        results_file: Any,
        ended: Any,
        reduce: Any,
        reduction_ref: Mapping[str, Any],
        spec: Mapping[str, Any],
        frame_rate: float,
        data_type: str,
    ) -> int:

        # Reduce the dense track to keyframes labelled start/middle/end.
        reduced = reduce(ended.track)

        # A reduction that produced nothing has nothing to record. It should not
        # happen for a track that passed the minimum-span rule, but writing a
        # row with no keyframes would put an unreviewable observation in MARP.
        if not reduced:
            return 0

        # Shape the row in the terms MARP already stores.
        observation = build_observation(
            frames=ended.track["frames"],
            keyframes=reduced,
            video_source_name=str(spec["video"]["source_name"]),
            # Opaque provenance, passed through exactly as received. A job given
            # a bare url carries none, and the observation then records null.
            jellyfin_item_id=spec["video"].get("jellyfin_item_id"),
            frame_rate=frame_rate,
            data_type=data_type,
            reduction_name=str(reduction_ref["name"]),
            reduction_version=str(reduction_ref["version"]),
            track_id=ended.track_id,
            end_reason=ended.reason,
        )

        # One JSON object per line, so the file streams both ways.
        results_file.write(json.dumps(observation) + "\n")
        return 1
