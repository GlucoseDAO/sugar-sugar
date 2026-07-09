"""Central place to register every pluggable prediction model.

To add a new model:
  1. Implement (or reuse) a `GlucosePredictor` subclass.
  2. Add one entry to the list below.
That's it - the model picker UI and callbacks read from `MODEL_REGISTRY`
and `model_dropdown_options()`, and never need to change.
"""
from __future__ import annotations

from typing import Optional

from sugar_sugar.models.base import GlucosePredictor
from sugar_sugar.models.chronos_model import ChronosPredictor

_MODELS: list[GlucosePredictor] = [
    ChronosPredictor(
        id="chronos-bolt-tiny",
        label="Chronos Bolt Tiny",
        model_name="amazon/chronos-bolt-tiny",
    ),
    ChronosPredictor(
        id="chronos-bolt-mini",
        label="Chronos Bolt Mini",
        model_name="amazon/chronos-bolt-mini",
    ),
    ChronosPredictor(
        id="chronos-bolt-small",
        label="Chronos Bolt Small",
        model_name="amazon/chronos-bolt-small",
    ),
]

MODEL_REGISTRY: dict[str, GlucosePredictor] = {m.id: m for m in _MODELS}

DEFAULT_MODEL_ID: str = "chronos-bolt-mini"


def get_model(model_id: Optional[str]) -> GlucosePredictor:
    """Look up a model by id, falling back to the default if unknown/None."""
    return MODEL_REGISTRY.get(model_id or DEFAULT_MODEL_ID, MODEL_REGISTRY[DEFAULT_MODEL_ID])


def model_dropdown_options() -> list[dict[str, str]]:
    """Options list ready to hand straight to a dcc.Dropdown."""
    return [{"label": m.label, "value": m.id} for m in _MODELS]