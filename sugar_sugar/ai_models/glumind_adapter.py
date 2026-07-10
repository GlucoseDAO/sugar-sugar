from __future__ import annotations

from .base import BaseGlucoseModel


class GluMindAdapter(BaseGlucoseModel):
    """Placeholder adapter. Replace predict() with the real GluMind model call
    once the checkpoint / real inference code is available.
    """

    def predict(self, history: list[float], prediction_steps: int) -> list[float]:
        if not history:
            return [100.0] * prediction_steps
        last_value = history[-1]
        # Naive placeholder: flat continuation of the last known value.
        return [round(last_value, 1)] * prediction_steps