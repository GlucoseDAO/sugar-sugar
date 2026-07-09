"""Kick off model inference in the background as soon as a round's data is
known, so the AI's guess is ready (or nearly ready) by the time the human
hits submit, instead of only starting once they do.

See GlucoseDAO/sugar-sugar issue #49: inference should run "at the same
time while the user is playing" - loading a Chronos checkpoint and running
it isn't instant, so starting it the moment the round's history/events are
available (e.g. when the chart is drawn) hides that latency behind the
human's own thinking time.

Dash's dcc.Store is client-side JSON, so a running Future can't be kept
there - this module keeps a small server-side registry instead, keyed by
whatever id the caller already uses to track a round (e.g.
f"{run_id}:{round_number}" from user-info-store).
"""
from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional, Sequence

from sugar_sugar.models.base import PredictionRequest
from sugar_sugar.models.multi import ModelPrediction, predict_many

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="model-inflight")
_inflight: dict[str, Future] = {}
_lock = threading.Lock()


def start_prediction(
    round_key: str,
    request: PredictionRequest,
    model_ids: Optional[Sequence[str]] = None,
) -> None:
    """Fire off model inference for this round in the background.

    Call this as soon as the round's history/events are known (e.g. when
    the chart for that round is first drawn), not at submit time.

    Safe to call again with the same `round_key` before the first job
    finishes (e.g. the round's data changed) - the new job simply replaces
    the old one in the registry; the old job is left to finish and is
    garbage collected, it just won't be waited on.
    """
    future = _executor.submit(predict_many, request, model_ids)
    with _lock:
        _inflight[round_key] = future


def collect_predictions(
    round_key: str,
    request: Optional[PredictionRequest] = None,
    model_ids: Optional[Sequence[str]] = None,
    timeout: Optional[float] = None,
) -> dict[str, ModelPrediction]:
    """Called at submit time. Blocks until the background job for this
    round finishes and returns its results.

    If `start_prediction` was never called for this `round_key` (or it was
    already collected), falls back to running `predict_many` synchronously
    right here - pass `request` (and optionally `model_ids`) so that
    fallback has what it needs. This means submit never comes back empty
    just because the early-start step was skipped or raced.
    """
    with _lock:
        future = _inflight.pop(round_key, None)

    if future is not None:
        return future.result(timeout=timeout)

    if request is None:
        raise ValueError(
            f"No in-flight prediction for round {round_key!r}, and no "
            "`request` was given to fall back to a synchronous run."
        )
    return predict_many(request, model_ids)


def discard_prediction(round_key: str) -> None:
    """Drop a round's in-flight job without waiting on it - call this if a
    round is abandoned (e.g. the user navigates away) so the registry
    doesn't hold onto it forever."""
    with _lock:
        _inflight.pop(round_key, None)