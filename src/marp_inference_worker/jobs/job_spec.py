# job_spec.py
# Created: 2026-09-09
# Author: Isaac Travers
#
# Job spec schemas for the MARP Inference Worker.
# This file validates the job the coordinator hands down, and the attempt envelope
# that identifies the lease it belongs to. It is the worker side of the contract
# recorded as A1 in .marp/task.md.
# Only validation belongs here -- no HTTP, no execution, no engine knowledge.

# Pydantic validates the coordinator payload and gives a JSON-safe round trip.
from pydantic import BaseModel, Field, field_validator, model_validator

# Any types the deliberately free-form params mapping.
from typing import Any


# ModelRef
# Names the model a job needs and the hash it must have.
# The worker resolves the name to a local artifact and verifies the hash before
# use, so a wrong or truncated download cannot reach an engine (R13).
class ModelRef(BaseModel):

    # Coordinator-side model name, used as the cache identity.
    name: str

    # Expected sha256 of the model artifact, lower-case hex.
    sha256: str

    # Optional locator the worker may fetch from when the artifact is not cached.
    url: str | None = None


# VideoRef
# Names the video a job runs over: how to open it, and what MARP calls it.
# The coordinator resolves the video and hands over a source the worker can
# open, so the worker knows nothing about MARP's library or about Jellyfin (A8).
# It does not search, does not score a filename match, and holds no media
# credential -- which also means a worker can process any reachable source.
class VideoRef(BaseModel):

    # The only thing the worker uses to open the video. Required, and refused
    # empty: a spec that cannot name its source must be rejected at the boundary
    # rather than by a child process that has already been launched for it.
    url: str = Field(min_length=1)

    # MARP `video_source` value, the filename as the database records it. This
    # is what appears as `video_source` on every observation.
    source_name: str

    # Opaque provenance, optional. Echoed into the observation output exactly as
    # received and never resolved, parsed or acted on -- the worker cannot tell
    # what kind of identifier it is, and a job given a bare url carries none.
    jellyfin_item_id: str | None = None


# FrameRange
# The frame span this job covers, half-open: [start_frame, end_frame).
# A range is always present, even for a whole video, so nothing in the worker has
# to special-case "the entire thing" (R9).
#
# **Half-open is the agreed convention on both sides**, settled 2026-09-09 with
# MARP_API. Two reasons it is worth being exact about. The count is a plain
# subtraction, `end - start`, with no off-by-one to get wrong; and consecutive
# pieces share a bound, so a video split into pieces is just
# `[0,300) [300,600) [600,900)` with nothing to add or subtract between them.
#
# Inclusive-on-both-ends was the alternative and it drops one frame at every
# piece boundary -- silently, because each piece looks complete on its own. A
# ten-hour video split into ten pieces would lose nine frames and no test on
# either side would have noticed.
#
# A human-facing label may well read "frames 0-999" for `[0, 1000)`. That
# display is inclusive; the contract is not. Do not read a label back into a
# spec without subtracting.
class FrameRange(BaseModel):

    # First frame index this job is responsible for, inclusive.
    start_frame: int = Field(ge=0)

    # One past the last frame index this job is responsible for, exclusive.
    # This frame belongs to the NEXT piece, and this job must not process it.
    end_frame: int = Field(gt=0)

    # Reject an inverted or empty range at the boundary rather than mid-decode.
    @model_validator(mode="after")
    def _check_order(self) -> "FrameRange":

        # An empty or backwards range is a coordinator bug and must not run.
        if self.end_frame <= self.start_frame:
            raise ValueError("end_frame must be greater than start_frame")
        return self

    # frame_count
    # How many frames this job covers.
    # Inputs: none.
    # Output: frame count, used as the progress total.
    @property
    def frame_count(self) -> int:

        # Half-open, so the difference is the count.
        return self.end_frame - self.start_frame


# ReductionRef
# Names the keyframe reduction rule this job must use, and its version.
# Recorded with every observation the job produces, because the reduction is
# expected to change and a stored observation has to stay attributable to the
# rule that made it (R10b).
class ReductionRef(BaseModel):

    # Registered reduction name, such as "v3_dirpad".
    name: str

    # Version of that named reduction.
    version: str

    # Accept the version as a number as well as a string.
    #
    # The reduction registry keys on strings, but MARP's published job spec
    # documents `version` as an integer -- so a submission written against the
    # coordinator's own schema arrived as `1` and pydantic refused it, taking
    # the whole job with it. Normalized here, at the one place the two
    # vocabularies meet, rather than loosened everywhere downstream.
    @field_validator("version", mode="before")
    @classmethod
    def _version_as_text(cls, value: object) -> object:

        # Only integers are folded in; a float version would be a real mistake.
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return value


# JobSpec
# One unit of work as the coordinator describes it.
# Everything an engine needs is here; nothing about MARP's transport is.
class JobSpec(BaseModel):

    # Engine that must run this job.
    engine: str

    # Model to run, with the hash the worker verifies.
    model: ModelRef

    # Video the frames come from.
    video: VideoRef

    # Frame span this job owns.
    range: FrameRange

    # Free-form engine parameters. Deliberately unschema'd: confidence
    # thresholds, tracker settings and Ultralytics arguments differ per engine.
    params: dict[str, Any] = Field(default_factory=dict)

    # Keyframe reduction rule to apply to finished tracks.
    reduction: ReductionRef


# AttemptEnvelope
# The lease identity every state-changing coordinator call must carry.
# A mismatch on any of the three is answered `abandon`, which is how the worker
# learns its lease was taken away while it was busy (A1).
class AttemptEnvelope(BaseModel):

    # Coordinator's identifier for this attempt at this job.
    attempt_id: str

    # This worker's persistent identity.
    worker_id: str

    # Lease generation, bumped by the coordinator whenever the lease moves.
    lease_epoch: int

    # The work itself.
    spec: JobSpec

    # Accept the coordinator's integer identifiers.
    #
    # MARP issues `attempt_id` and `worker_id` as integers; the worker keeps
    # them as text because it also uses them as a workspace directory name and
    # a URL segment. Coerced on the way in and converted back on the way out
    # (see `coordinator_client.coordinator_id`), so the wire type is the
    # coordinator's everywhere and the local type is the worker's everywhere.
    # Without this the lease itself failed to validate and every offered job was
    # reported back as an invalid spec.
    @field_validator("attempt_id", "worker_id", mode="before")
    @classmethod
    def _identifier_as_text(cls, value: object) -> object:

        # Only integers; anything else is passed through to normal validation.
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return value


# HeartbeatAction
# The actions a coordinator may return in a heartbeat response.
# Cancel, pause and abandon all arrive this way, so the worker never needs to be
# reachable from outside (R5).
_HEARTBEAT_ACTIONS = ("continue", "cancel", "pause", "abandon")


# is_stop_action()
# Says whether a heartbeat action means "stop running this job".
# Inputs: the action string from a heartbeat response.
# Output: True for cancel and abandon, False otherwise.
# Use this so the runner treats an unknown action as `continue` rather than
# guessing -- an old worker against a newer coordinator keeps working.
def is_stop_action(action: str) -> bool:

    # Pause stops the worker taking new work but leaves a running job alone (A5).
    return action in ("cancel", "abandon")
