"""Run several registered predictors against the same `PredictionRequest`.

This is the entry point for anything that wants more than one model's
opinion at once - e.g. a "Play vs AI" view that shows multiple Chronos
flavors alongside the human's guess, or an offline comparison of which
checkpoint forecasts best on recent data. Single-model call sites (the
current default opponent, say) can keep using `registry.get_model()`
directly; this module is purely additive.

A model that raises, times out on load, or returns malformed output must
never take the other models down with it - each one is isolated and
reported independently via `ModelPrediction.ok` / `.error`.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from sugar_sugar.models.base import GlucosePredictor, PredictionRequest
from sugar_sugar.models.registry import MODEL_REGISTRY, get_model


@dataclass
class ModelPrediction:
    """Result of running one model against one `PredictionRequest`."""

    model_id: str
    label: str
    prediction: Optional[np.ndarray] = None
    error: Optional[str] = None
    elapsed_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None and self.prediction is not None


def _run_one(model: GlucosePredictor, request: PredictionRequest) -> ModelPrediction:
    start = time.perf_counter()
    try:
        prediction = model.predict(request)
        return ModelPrediction(
            model_id=model.id,
            label=model.label,
            prediction=np.asarray(prediction, dtype=np.float64),
            elapsed_seconds=time.perf_counter() - start,
        )
    except Exception as exc:  # noqa: BLE001 - one bad model must not sink the rest
        return ModelPrediction(
            model_id=model.id,
            label=model.label,
            error=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=time.perf_counter() - start,
        )


def predict_many(
    request: PredictionRequest,
    model_ids: Optional[Sequence[str]] = None,
    *,
    parallel: bool = True,
    max_workers: int = 4,
) -> dict[str, ModelPrediction]:
    """Run every requested model (default: ALL registered models) against
    the same `request`.

    Returns {model_id: ModelPrediction}, in registry/requested order
    regardless of which model finished first. Check `.ok` before using a
    result - a failed model is reported, not raised.

    Example:
        results = predict_many(request)
        for model_id, result in results.items():
            if result.ok:
                print(result.label, result.prediction)
            else:
                print(result.label, "failed:", result.error)
    """
    ids = list(model_ids) if model_ids is not None else list(MODEL_REGISTRY.keys())
    models = [get_model(model_id) for model_id in ids]

    if not parallel or len(models) <= 1:
        ordered = {m.id: _run_one(m, request) for m in models}
    else:
        results: dict[str, ModelPrediction] = {}
        with ThreadPoolExecutor(max_workers=min(max_workers, len(models))) as pool:
            futures = {pool.submit(_run_one, m, request): m for m in models}
            for future in as_completed(futures):
                result = future.result()
                results[result.model_id] = result
        # Preserve requested/registry order regardless of completion order.
        ordered = {model_id: results[model_id] for model_id in ids if model_id in results}

    return ordered


def best_prediction(results: dict[str, ModelPrediction]) -> Optional[ModelPrediction]:
    """Convenience: first successful result in registry/requested order, or
    None if every model failed."""
    for result in results.values():
        if result.ok:
            return result
    return None


def successful_predictions(results: dict[str, ModelPrediction]) -> dict[str, np.ndarray]:
    """Convenience: just the {model_id: prediction} pairs that succeeded -
    handy for plotting several forecast lines on one chart."""
    return {
        model_id: result.prediction
        for model_id, result in results.items()
        if result.ok and result.prediction is not None
    }