"""Smoke-test: create a share record with synthetic data and print the ID.

Run via ``uv run python scripts/_smoke_share.py``.  Prints the share id on
stdout so the caller can hit ``/share/<id>`` and ``/share/<id>/image.png``.
"""
from __future__ import annotations

from datetime import datetime

from sugar_sugar import share_store


def _fake_round(round_number: int, window_size: int = 36) -> dict:
    actual: dict[str, str] = {"metric": "Actual Glucose"}
    predicted: dict[str, str] = {"metric": "Predicted"}
    for i in range(window_size):
        # Smooth sine-ish ground truth
        g: float = 140.0 + 25.0 * ((-1) ** (i % 4)) + 2.0 * i
        actual[f"t{i}"] = f"{g:.1f}"
        # Prediction only in last 12 points, with a small offset
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


def main() -> None:
    # Round 1 and 2: format A (generic), rounds 3 and 4: format B (user data),
    # round 5: format C (mixed) -- so we can confirm the share page shows
    # per-format rankings and merges all rounds into the synthesis chart.
    rounds: list[dict] = []
    r1 = _fake_round(1); r1["format"] = "A"; rounds.append(r1)
    r2 = _fake_round(2); r2["format"] = "A"; rounds.append(r2)
    r3 = _fake_round(3); r3["format"] = "B"; rounds.append(r3)
    r4 = _fake_round(4); r4["format"] = "B"; rounds.append(r4)
    r5 = _fake_round(5); r5["format"] = "C"; rounds.append(r5)

    record: dict = {
        "schema_version": 2,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "locale": "en",
        "rounds": rounds,
        "played_formats": ["C", "B", "A"],
        "rankings": {
            "per_format": [
                {"format": "C", "rank": 4, "total": 20},
                {"format": "B", "rank": 11, "total": 42},
                {"format": "A", "rank": 7, "total": 63},
            ],
            "overall": {"rank": 9, "total": 70},
        },
        "user_info": {
            "name": "Smoke Tester",
            "study_id": "smoke-001",
            "format": "C",
            "uses_cgm": True,
            "max_rounds": 12,
        },
    }
    share_id: str = share_store.save_share(record)
    print(share_id)


if __name__ == "__main__":
    main()
