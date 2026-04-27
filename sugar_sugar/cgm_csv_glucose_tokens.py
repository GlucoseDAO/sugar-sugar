"""
CGM CSV exports sometimes use the literal tokens 'High' and 'Low' in glucose
cells instead of a numeric value (e.g. Dexcom for out-of-range readings).

The app stores glucose in mg/dL. This module is the only place that maps those
tokens to numeric stand-ins for charting and metrics. To change or remove this
behaviour, edit the constants and/or this file — callers only invoke
`utf8_token_column_to_mgdl_float` (and the mmol variant) from the CSV loaders
in `sugar_sugar.data`.
"""
from __future__ import annotations

import polars as pl

# Stand-ins when the file contains text instead of a number (mg/dL basis).
_CGM_CSV_TOKEN_HIGH_MGDL: float = 399.0
_CGM_CSV_TOKEN_LOW_MGDL: float = 39.0
_MGDL_PER_MMOL: float = 18.0


def utf8_token_column_to_mgdl_float(col: pl.Expr, *, source_is_mmol: bool) -> pl.Expr:
    """
    Build a Float64 column in mg/dL from a column that may be numeric or the
    strings 'High' / 'Low' (case-insensitive, surrounding spaces ignored).
    If ``source_is_mmol`` is True, non-token values are treated as mmol/L and
    converted to mg/dL. Tokens always map to the same mg/dL stand-ins.
    """
    s = col.cast(pl.Utf8, strict=False).str.strip_chars()
    u = s.str.to_uppercase()
    num = s.str.replace_all(",", ".").cast(pl.Float64, strict=False)
    high = pl.lit(_CGM_CSV_TOKEN_HIGH_MGDL, dtype=pl.Float64)
    low = pl.lit(_CGM_CSV_TOKEN_LOW_MGDL, dtype=pl.Float64)
    if source_is_mmol:
        return (
            pl.when(u.eq("HIGH"))
            .then(high)
            .when(u.eq("LOW"))
            .then(low)
            .otherwise(num * pl.lit(_MGDL_PER_MMOL))
        )
    return pl.when(u.eq("HIGH")).then(high).when(u.eq("LOW")).then(low).otherwise(num)
