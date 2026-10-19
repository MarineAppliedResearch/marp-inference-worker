# __init__.py
# Created: 2026-10-19
# Author: Isaac Travers
#
# Training package for the MARP Inference Worker.
# Everything a training job needs that is not the engine itself: getting the
# dataset onto this machine, and making it something Ultralytics can read.
#
# Nothing here talks to the coordinator. A training job reaches the outside
# world through the same JobContext every other engine gets, and through the
# locator its job spec carried -- which is resolved and authorized by the
# coordinator client, not by this package.

# The dataset cache is the package's whole public surface for now.
from marp_inference_worker.training.dataset import (
    DatasetHashMismatch,
    ensure_dataset,
)

__all__ = ["DatasetHashMismatch", "ensure_dataset"]
