"""Common interface every glucose-prediction model must implement.

Add a new model by subclassing `GlucosePredictor` and registering an
instance in `registry.py`. The rest of the app (callbacks, UI) only ever
talks to this interface, so plugging in a new flavour never touches
app.py or the components.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class PredictionRequest:
    """Everything a model is given in order to make a forecast.

    history: chronologically sorted glucose points the player has *seen*
        so far (columns: time, gl). This is only the visible window - it
        never includes the hidden/held-out area we're trying to predict.
    events: carb / insulin / exercise events with time <= history's last
        timestamp. Columns: time, event_type, event_subtype, insulin_value
        (see data.py::_adapt_events_df for the exact schema).
    horizon: number of future 5-minute points to forecast. This matches
        PREDICTION_HOUR_OFFSET from config.py.
    """

    history: pl.DataFrame
    events: pl.DataFrame
    horizon: int


class GlucosePredictor(ABC):
    """Base class every pluggable model implements."""

    #: stable machine id - used as the registry key and stored in result CSVs
    id: str
    #: human-readable label shown in the model-picker dropdown
    label: str

    @abstractmethod
    def predict(self, request: PredictionRequest) -> np.ndarray:
        """Return an array of length `request.horizon` with predicted gl values."""
        raise NotImplementedError