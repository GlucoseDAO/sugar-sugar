"""Chronos (Amazon) pretrained time-series forecasters, wrapped to the
GlucosePredictor interface.

Requires the `chronos-forecasting` package:
    uv add chronos-forecasting torch

Chronos-Bolt checkpoints are used by default - they're regression-based
(much faster than the original T5 sampling models), which matters for a
forecast triggered live from a button click in the browser.

Docs / model cards: https://github.com/amazon-science/chronos-forecasting
"""
from __future__ import annotations

import threading

import numpy as np
import torch

from sugar_sugar.models.base import GlucosePredictor, PredictionRequest
from sugar_sugar.models.events import apply_event_adjustment

# One pipeline per model_name, loaded lazily and cached for the life of the
# server process - loading downloads + places weights on device, so we do
# NOT want to repeat that on every single prediction request.
_pipeline_cache: dict[str, object] = {}
# Guards first-load of each pipeline. Needed once predictions can run
# concurrently (see models/multi.py) - two threads racing to load the same
# model_name for the first time would otherwise double-download/double-init.
_cache_lock = threading.Lock()


def _get_pipeline(model_name: str, device_map: str):
    if model_name in _pipeline_cache:
        return _pipeline_cache[model_name]
    with _cache_lock:
        if model_name not in _pipeline_cache:
            from chronos import BaseChronosPipeline  # local import: optional heavy dep

            _pipeline_cache[model_name] = BaseChronosPipeline.from_pretrained(
                model_name,
                device_map=device_map,
                torch_dtype="auto",
            )
        return _pipeline_cache[model_name]


class ChronosPredictor(GlucosePredictor):
    """One Chronos checkpoint (e.g. bolt-tiny/mini/small) exposed as a model."""

    def __init__(self, id: str, label: str, model_name: str, device_map: str = "cpu") -> None:
        self.id = id
        self.label = label
        self.model_name = model_name
        self.device_map = device_map

    def predict(self, request: PredictionRequest) -> np.ndarray:
        pipeline = _get_pipeline(self.model_name, self.device_map)

        # Chronos requires a torch.Tensor as context (1D, list of 1D, or a
        # left-padded 2D batch) - a raw numpy array is not accepted. The
        # keyword is `context`, NOT `inputs` (that name only exists as an
        # internal local inside the library).
        context_np = request.history.get_column("gl").to_numpy().astype("float32")


        context = torch.tensor(context_np)

        with torch.no_grad():
            # `predict_quantiles` is the BaseChronosPipeline interface and
            # works for BOTH Chronos-Bolt (regression) and the original T5
            # (sampling) checkpoints, so we don't have to branch on model
            # type or guess the raw output's axis order.
            #
            # IMPORTANT: pass the context POSITIONALLY. The first parameter's
            # *name* differs between chronos-forecasting versions - older
            # builds call it `inputs`, newer ones `context` - so naming it as
            # a keyword breaks on whichever version you don't have installed.
            #   quantiles: (batch, prediction_length, num_quantile_levels)
            #   mean:      (batch, prediction_length)
            _quantiles, mean = pipeline.predict_quantiles(
                context,
                prediction_length=request.horizon,
                quantile_levels=[0.5],
            )

        # `mean` is the point forecast, already shape (1, horizon) - unambiguous.
        point_forecast = mean[0].detach().cpu().numpy().astype(np.float64)

        adjusted = apply_event_adjustment(
            point_forecast, request.history, request.events, request.horizon
        )


        return adjusted