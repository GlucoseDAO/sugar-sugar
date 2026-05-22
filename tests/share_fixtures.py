"""Shared fixtures for share-page integration tests."""
from __future__ import annotations

from datetime import datetime
from typing import Any


def make_test_share_record() -> dict[str, Any]:
    """Synthetic share record with multiple formats (mirrors scripts/_smoke_share.py)."""

    def _fake_round(round_number: int, window_size: int = 36) -> dict[str, Any]:
        actual: dict[str, str] = {"metric": "Actual Glucose"}
        predicted: dict[str, str] = {"metric": "Predicted"}
        for i in range(window_size):
            g: float = 140.0 + 25.0 * ((-1) ** (i % 4)) + 2.0 * i
            actual[f"t{i}"] = f"{g:.1f}"
            if i >= window_size - 12:
                predicted[f"t{i}"] = f"{g + (round_number * 4):.1f}"
            else:
                predicted[f"t{i}"] = "-"
        abs_err: dict[str, str] = {"metric": "Absolute Error"}
        rel_err: dict[str, str] = {"metric": "Relative Error (%)"}
        for i in range(window_size):
            if predicted.get(f"t{i}", "-") == "-":
                abs_err[f"t{i}"] = "-"
                rel_err[f"t{i}"] = "-"
            else:
                a: float = float(actual[f"t{i}"])
                p: float = float(predicted[f"t{i}"])
                abs_err[f"t{i}"] = f"{abs(a - p):.1f}"
                rel_err[f"t{i}"] = f"{abs(a - p) / a * 100:.1f}"
        return {
            "round_number": round_number,
            "prediction_window_start": 100 + round_number * window_size,
            "prediction_window_size": window_size,
            "format": "A",
            "is_example_data": True,
            "data_source_name": "example.csv",
            "prediction_table_data": [actual, predicted, abs_err, rel_err],
        }

    rounds: list[dict[str, Any]] = []
    r1 = _fake_round(1)
    r1["format"] = "A"
    rounds.append(r1)
    r2 = _fake_round(2)
    r2["format"] = "A"
    rounds.append(r2)
    r3 = _fake_round(3)
    r3["format"] = "B"
    rounds.append(r3)

    return {
        "schema_version": 2,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "locale": "en",
        "rounds": rounds,
        "played_formats": ["B", "A"],
        "rankings": {
            "per_format": [
                {"format": "B", "rank": 11, "total": 42},
                {"format": "A", "rank": 7, "total": 63},
            ],
            "overall": {"rank": 9, "total": 70},
        },
        "user_info": {
            "name": "Test Player",
            "study_id": "test-001",
            "format": "A",
            "uses_cgm": True,
            "max_rounds": 12,
        },
    }
