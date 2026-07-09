"""Wire format shared by the app (client) and the Chronos service (server).

This module is the CONTRACT between the two projects. Keep it byte-for-byte
identical in both repos - the app serialises a `PredictionRequest` with
`request_to_payload`, the service rebuilds it with `payload_to_request`, and
predictions come back through `prediction_to_payload` / `payload_to_prediction`.

Everything is plain JSON-able Python (dicts / lists / str / float / int) so it
survives an HTTP round trip untouched. Times are naive ISO-8601 strings at
second precision - the same shape app.py already uses for its dcc.Store data
(see convert_df_to_dict), so nothing upstream has to change.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import polars as pl

# Naive, second-precision timestamps. Matches app.py's existing store format.
_TIME_FMT = "%Y-%m-%dT%H:%M:%S"


def _df_to_records(df: Optional[pl.DataFrame]) -> list[dict[str, Any]]:
    """Serialise a polars DataFrame to a list of JSON-safe row dicts.

    Datetime columns are emitted as ISO strings; everything else is passed
    through as-is. An empty / column-less frame serialises to []."""
    if df is None or df.height == 0:
        return []
    out = df
    # Stringify any temporal column so json.dumps won't choke on datetime objs.
    temporal = [name for name, dtype in zip(df.columns, df.dtypes) if dtype in (pl.Datetime, pl.Date)]
    if temporal:
        out = out.with_columns(
            [pl.col(name).dt.strftime(_TIME_FMT).alias(name) for name in temporal]
        )
    return out.to_dicts()


def _records_to_df(records: Sequence[dict[str, Any]], time_columns: Sequence[str]) -> pl.DataFrame:
    """Rebuild a polars DataFrame from row dicts, re-parsing the named time
    columns back into naive Datetimes. Empty input -> empty DataFrame."""
    if not records:
        return pl.DataFrame()
    df = pl.DataFrame(records)
    for name in time_columns:
        if name in df.columns:
            df = df.with_columns(pl.col(name).str.to_datetime(_TIME_FMT).alias(name))
    return df


# --- PredictionRequest <-> JSON -------------------------------------------------

def request_to_payload(
    history: pl.DataFrame,
    events: pl.DataFrame,
    horizon: int,
    *,
    model_id: Optional[str] = None,
    model_ids: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Build the JSON body for a /predict or /predict_batch call.

    `model_id` targets a single checkpoint (/predict); `model_ids` (or None ->
    all) is for the batch endpoint (/predict_batch)."""
    payload: dict[str, Any] = {
        "horizon": int(horizon),
        "history": _df_to_records(history),
        "events": _df_to_records(events),
    }
    if model_id is not None:
        payload["model_id"] = model_id
    if model_ids is not None:
        payload["model_ids"] = list(model_ids)
    return payload


def payload_to_request(payload: dict[str, Any]):
    """Rebuild (history_df, events_df, horizon) on the service side.

    Returns a tuple rather than a PredictionRequest so this module stays free
    of a base.py import; the server wraps it into PredictionRequest itself."""
    history = _records_to_df(payload.get("history") or [], time_columns=["time"])
    events = _records_to_df(payload.get("events") or [], time_columns=["time"])
    horizon = int(payload["horizon"])
    return history, events, horizon


# --- prediction array <-> JSON --------------------------------------------------

def prediction_to_payload(prediction: np.ndarray) -> list[float]:
    """A forecast array -> a plain list of floats."""
    return [float(v) for v in np.asarray(prediction, dtype=np.float64).ravel()]


def payload_to_prediction(values: Sequence[float]) -> np.ndarray:
    """A list of floats -> a float64 forecast array."""
    return np.asarray(list(values), dtype=np.float64)
