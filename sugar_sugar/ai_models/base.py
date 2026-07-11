from __future__ import annotations

from abc import ABC, abstractmethod


class BaseGlucoseModel(ABC):
    """Common interface every glucose-forecasting model adapter must implement."""

    @abstractmethod
    def predict(self, history: list[float], prediction_steps: int) -> list[float]:
        """Given past glucose values, predict the next `prediction_steps` values."""
        raise NotImplementedError