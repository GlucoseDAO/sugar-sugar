"""Central place to register the prediction models the app can play against.

CPU/GPU split: these are `RemoteChronosPredictor`s. The actual Chronos
checkpoints live in the separate `chronos-service` project on the GPU box;
here we just register thin HTTP clients that call it. torch / chronos are NOT
importable from the app, by design.

The model ids/labels below MUST match the service's registry (see
chronos_service/registry.py). They're intentionally kept as a static list so
the app starts fine even when the endpoint is momentarily down - it only talks
to the service when a prediction is actually requested. If you'd rather
discover them dynamically, GET {CHRONOS_SERVICE_URL}/models returns the same
{id, label} list.

To add a checkpoint: add it in the service registry, then mirror its id+label
here. Nothing else in the app (callbacks, UI, multi.py, inflight.py) changes.
"""
from __future__ import annotations

import os
from typing import Optional

from sugar_sugar.models.base import GlucosePredictor
from sugar_sugar.models.remote import RemoteChronosPredictor

# Where the GPU inference endpoint lives. Override per-environment, e.g.
#   CHRONOS_SERVICE_URL=http://gpu-box.internal:8500
_SERVICE_URL = os.getenv("CHRONOS_SERVICE_URL", "http://localhost:8500")
# Per-request HTTP timeout (seconds). collect_predictions() also applies its
# own outer timeout on the background job; keep this comfortably below that.
_SERVICE_TIMEOUT = float(os.getenv("CHRONOS_SERVICE_TIMEOUT", "30"))

_MODELS: list[GlucosePredictor] = [
    RemoteChronosPredictor(
        id="chronos-bolt-tiny",
        label="Chronos Bolt Tiny",
        base_url=_SERVICE_URL,
        timeout=_SERVICE_TIMEOUT,
    ),
    RemoteChronosPredictor(
        id="chronos-bolt-mini",
        label="Chronos Bolt Mini",
        base_url=_SERVICE_URL,
        timeout=_SERVICE_TIMEOUT,
    ),
    RemoteChronosPredictor(
        id="chronos-bolt-small",
        label="Chronos Bolt Small",
        base_url=_SERVICE_URL,
        timeout=_SERVICE_TIMEOUT,
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