from __future__ import annotations

from .base import BaseGlucoseModel
from .glumind_adapter import GluMindAdapter

_MODEL_INSTANCES: dict[str, BaseGlucoseModel] = {}


def get_model(name: str) -> BaseGlucoseModel:
    if name not in _MODEL_INSTANCES:
        if name == "glumind":
            _MODEL_INSTANCES[name] = GluMindAdapter()
        else:
            raise ValueError(f"Unknown model: {name}")
    return _MODEL_INSTANCES[name]


def list_available_models() -> list[str]:
    return ["glumind"]