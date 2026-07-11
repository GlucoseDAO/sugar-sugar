from __future__ import annotations

from typing import Callable

from .base import BaseGlucoseModel
from .glumind_adapter import GluMindAdapter

_MODEL_FACTORIES: dict[str, Callable[[], BaseGlucoseModel]] = {
    "glumind": GluMindAdapter,
}
_MODEL_INSTANCES: dict[str, BaseGlucoseModel] = {}


def get_model(name: str) -> BaseGlucoseModel:
    if name not in _MODEL_INSTANCES:
        factory = _MODEL_FACTORIES.get(name)
        if factory is None:
            raise ValueError(f"Unknown model: {name}")
        _MODEL_INSTANCES[name] = factory()
    return _MODEL_INSTANCES[name]


def list_available_models() -> list[str]:
    return list(_MODEL_FACTORIES.keys())