"""Unit tests for CGM CSV 'High' / 'Low' token mapping (mg/dL stand-ins)."""

import polars as pl
import pytest

from sugar_sugar.cgm_csv_glucose_tokens import utf8_token_column_to_mgdl_float


@pytest.mark.parametrize(
    ("raw", "expected_mgdl", "is_mmol"),
    [
        ("High", 399.0, False),
        ("Low", 39.0, False),
        ("HIGH", 399.0, False),
        ("120", 120.0, False),
        ("5.5", 99.0, True),  # 5.5 mmol/L * 18
    ],
)
def test_utf8_token_column_to_mgdl_float(
    raw: str, expected_mgdl: float, is_mmol: bool
) -> None:
    df = pl.DataFrame({"v": [raw]})
    out = df.select(
        utf8_token_column_to_mgdl_float(pl.col("v"), source_is_mmol=is_mmol).alias("gl")
    )
    assert out.get_column("gl")[0] == pytest.approx(expected_mgdl)
