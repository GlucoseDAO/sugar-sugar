from __future__ import annotations

GLUCOSE_MIN = 40.0
GLUCOSE_MAX = 400.0


def normalize_glucose(value: float) -> float:
    """Scale a raw mg/dL glucose value to the model's [0, 1] input range."""
    span = GLUCOSE_MAX - GLUCOSE_MIN
    return (value - GLUCOSE_MIN) / span


def denormalize_glucose(value: float) -> float:
    """Reverse of normalize_glucose: scale a [0, 1] model output back to mg/dL."""
    span = GLUCOSE_MAX - GLUCOSE_MIN
    return value * span + GLUCOSE_MIN