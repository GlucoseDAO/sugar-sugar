"""Model-agnostic adjustment layer that nudges a raw glucose forecast using
recent carb / insulin / exercise events.

Pretrained time-series models like Chronos only ever see the glucose curve
itself - they have no idea a meal or insulin dose just happened. This module
adds a lightweight, explainable correction on top of *any* model's raw
forecast, so event-awareness only has to be implemented once, not per model.

CURRENT DATA LIMITATION: the app's events dataframe (see
data.py::_adapt_events_df) does not carry carb grams - only event_type /
event_subtype, plus insulin_value for insulin events. Until carb quantity is
exposed, every carbohydrate event is treated as a fixed nominal size
(_CARB_NOMINAL_GRAMS below). Swap that out - or thread a real carb_grams
column through from data.py - once that data is available upstream.

All mg/dL-per-unit constants below are rough starting points, not clinically
validated values. Tune them against real prediction-vs-outcome data once you
have enough recorded rounds to compare against.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import polars as pl

_MINUTES_PER_POINT = 5

# The event-adjustment heuristic below is EXPERIMENTAL and OFF by default.
#
# Why off: on real data a single forecast window can carry many recent events
# (observed: 26 in one 2-hour window of example.csv). This layer sums them ALL
# at full magnitude with no cap, which craters an otherwise-good forecast into
# the floor clamp - i.e. it makes the AI opponent worse, not better. The raw
# Chronos output already tracks the curve well, so that is the better default.
#
# Before flipping this back to True, two things need doing: (1) find out why so
# many events land in one window (likely duplication/over-generation upstream in
# data.py, since ~1 event per reading is not realistic), and (2) bound the total
# adjustment so no plausible stack of events can slam the forecast into the floor.
ENABLE_EVENT_ADJUSTMENT = False

# Physiological clamp for the *adjusted* forecast. Glucose can't be zero or
# negative, and a raw event adjustment (especially a large insulin dose with
# the rough constants below) can otherwise drag the forecast straight through
# the chart's axis floor - which renders as a vertical dive to 0. These bounds
# are deliberately wide (well outside any real CGM reading) so they only ever
# catch runaway adjustments, never trim a plausible forecast.
_GL_FLOOR_MGDL = 40.0
_GL_CEIL_MGDL = 400.0

# --- Carbohydrates ---
_CARB_NOMINAL_GRAMS = 40.0
_CARB_MGDL_PER_40G = 40.0
_CARB_PEAK_MIN = 45.0
_CARB_DURATION_MIN = 180.0

# --- Insulin ---
# mg/dL lowered per unit, as realized *within the ~1h forecast window*. This is
# NOT a full insulin-sensitivity factor (a correction dose lowers glucose over
# ~4h): the gamma curve below already peaks at ~75 min, so it applies most of
# its magnitude inside the one-hour horizon. Setting this to a full ISF (~30-50)
# compresses insulin's entire multi-hour effect into one hour and slams the
# forecast into the floor clamp. 15 is a provisional value - tune it against
# recorded dose-vs-outcome data, ideally per-user, once that data exists.
_INSULIN_MGDL_PER_UNIT = 15.0
_FAST_PEAK_MIN = 75.0
_FAST_DURATION_MIN = 240.0
_LONG_PEAK_MIN = 180.0
_LONG_DURATION_MIN = 480.0

# --- Exercise ---
_EXERCISE_MGDL = {"Light": -5.0, "Medium": -10.0, "Heavy": -18.0}
_EXERCISE_PEAK_MIN = 60.0
_EXERCISE_DURATION_MIN = 180.0


def _gamma_curve(minutes_since: np.ndarray, peak_min: float, duration_min: float) -> np.ndarray:
    """Rise-then-decay curve: 0 before the event, peaks at `peak_min`, back to
    0 by `duration_min`. Normalised so the curve's own peak value is 1.0."""
    x = np.clip(minutes_since, 0.0, duration_min)
    shape = 2.0
    safe_peak = max(peak_min, 1e-6)
    curve = (x / safe_peak) ** shape * np.exp(shape * (1.0 - x / safe_peak))
    curve = np.where(minutes_since < 0, 0.0, curve)
    curve = np.where(minutes_since > duration_min, 0.0, curve)
    return curve


def apply_event_adjustment(
    raw_forecast: np.ndarray,
    history: pl.DataFrame,
    events: pl.DataFrame,
    horizon: int,
) -> np.ndarray:
    """Nudge `raw_forecast` (length == horizon) using events near the forecast window."""
    if not ENABLE_EVENT_ADJUSTMENT:
        return raw_forecast
    if events is None or events.is_empty() or history.is_empty():
        return raw_forecast

    last_time = history.get_column("time").max()
    future_times = [
        last_time + timedelta(minutes=_MINUTES_PER_POINT * (i + 1)) for i in range(horizon)
    ]
    adjustment = np.zeros(horizon, dtype=np.float64)

    for row in events.iter_rows(named=True):
        event_type = row.get("event_type")
        event_subtype = row.get("event_subtype") or ""
        event_time = row.get("time")
        if event_time is None:
            continue

        minutes_since = np.array(
            [(future_time - event_time).total_seconds() / 60.0 for future_time in future_times]
        )

        if event_type == "Carbohydrates":
            curve = _gamma_curve(minutes_since, _CARB_PEAK_MIN, _CARB_DURATION_MIN)
            adjustment += curve * (_CARB_NOMINAL_GRAMS / 40.0 * _CARB_MGDL_PER_40G)

        elif event_type == "Insulin":
            units = float(row.get("insulin_value") or 0.0)
            if units <= 0:
                continue
            peak_min, duration_min = (
                (_LONG_PEAK_MIN, _LONG_DURATION_MIN)
                if event_subtype == "Long Acting"
                else (_FAST_PEAK_MIN, _FAST_DURATION_MIN)
            )
            curve = _gamma_curve(minutes_since, peak_min, duration_min)
            adjustment -= curve * (units * _INSULIN_MGDL_PER_UNIT)

        elif event_type == "Exercise":
            magnitude = _EXERCISE_MGDL.get(event_subtype, -8.0)
            curve = _gamma_curve(minutes_since, _EXERCISE_PEAK_MIN, _EXERCISE_DURATION_MIN)
            adjustment += curve * magnitude

    adjusted = raw_forecast + adjustment
    # Never let an event adjustment render an impossible glucose value.
    return np.clip(adjusted, _GL_FLOOR_MGDL, _GL_CEIL_MGDL)