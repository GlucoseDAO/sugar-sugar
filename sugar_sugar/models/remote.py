"""Client-side predictor that runs inference on the remote Chronos service.

This is the whole point of the CPU/GPU split: the app never imports torch or
Chronos. It implements the SAME `GlucosePredictor` interface every other model
uses, but `.predict()` just POSTs the window to the inference endpoint and
reads the forecast back. As far as `multi.py`, `inflight.py` and the app
callbacks are concerned, nothing changed - they still call `.predict()`.

The service base URL comes from `CHRONOS_SERVICE_URL` (see registry.py).

Failure handling: any network / HTTP / decode error is raised here. That is
deliberate - `models.multi._run_one` catches per-model exceptions and records
them as `ModelPrediction.error`, so a down endpoint degrades to "the AI made
no guess this round" instead of taking down a real human submit.
"""
from __future__ import annotations

import httpx
import numpy as np

from sugar_sugar.models.base import GlucosePredictor, PredictionRequest
from sugar_sugar.models import serialization as wire


class RemoteChronosPredictor(GlucosePredictor):
    """A single remote Chronos checkpoint, addressed by its service-side id."""

    def __init__(
        self,
        id: str,
        label: str,
        base_url: str,
        *,
        timeout: float = 30.0,
    ) -> None:
        self.id = id
        self.label = label
        # Normalise so we can safely append the path regardless of trailing /.
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def predict(self, request: PredictionRequest) -> np.ndarray:
        payload = wire.request_to_payload(
            request.history,
            request.events,
            request.horizon,
            model_id=self.id,
        )

        try:
            response = httpx.post(
                f"{self.base_url}/predict",
                json=payload,
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"chronos service unreachable at {self.base_url}: {exc}"
            ) from exc

        if response.status_code != 200:
            # Surface the service's error detail so it lands in ModelPrediction.error.
            detail = _safe_detail(response)
            raise RuntimeError(
                f"chronos service returned {response.status_code} for "
                f"model {self.id!r}: {detail}"
            )

        body = response.json()
        values = body.get("prediction")
        if values is None:
            raise RuntimeError(
                f"chronos service response for {self.id!r} had no 'prediction'"
            )

        prediction = wire.payload_to_prediction(values)
        if prediction.shape[0] != request.horizon:
            raise RuntimeError(
                f"chronos service returned {prediction.shape[0]} points for "
                f"{self.id!r}, expected horizon={request.horizon}"
            )
        return prediction


def _safe_detail(response: "httpx.Response") -> str:
    try:
        return str(response.json().get("detail", response.text))
    except Exception:  # noqa: BLE001 - error reporting must not itself raise
        return response.text[:300]
