"""Complete Analysis puts Exit, remaining formats, and Share on one row under the title."""
from __future__ import annotations

from typing import Any

from sugar_sugar.app import create_final_layout


def _by_id(node: Any, target: str) -> Any:
    if getattr(node, "id", None) == target:
        return node
    kids = getattr(node, "children", None)
    if isinstance(kids, (list, tuple)):
        for kid in kids:
            found = _by_id(kid, target)
            if found is not None:
                return found
    elif kids is not None and not isinstance(kids, str):
        return _by_id(kids, target)
    return None


def _user(*, uses_cgm: bool, played: list[str]) -> dict[str, Any]:
    current = played[-1] if played else "A"
    rounds = [
        {
            "format": fmt,
            "is_example_data": fmt == "A",
            "data_source_name": "example.csv" if fmt == "A" else "own.csv",
            "prediction_table_data": [
                {"metric": "Actual Glucose", **{f"t{i}": "100" for i in range(6)}},
                {"metric": "Predicted", **{f"t{i}": "110" for i in range(6)}},
            ],
        }
        for fmt in played
    ]
    runs_by_format: dict[str, list[dict[str, Any]]] = {}
    for fmt in played[:-1]:
        runs_by_format[fmt] = [{"rounds": [{"format": fmt}], "ended_at": "2026-08-01"}]
    return {
        "study_id": "s1",
        "email": "ann@x.com",
        "format": current,
        "uses_cgm": uses_cgm,
        "max_rounds": 12,
        "rounds": rounds,
        "runs_by_format": runs_by_format,
        "consent_completed": True,
    }


def test_action_row_sits_under_title_with_x_formats_then_share() -> None:
    layout = create_final_layout(_user(uses_cgm=True, played=["A"]), "mg/dL", locale="en")
    row = _by_id(layout, "final-action-row")
    assert row is not None
    child_ids = [getattr(child, "id", None) for child in row.children]
    assert child_ids == [
        "restart-button",
        "switch-format-a",
        "switch-format-b",
        "switch-format-c",
        "share-results-button",
    ]
    assert row.style["flexWrap"] == "nowrap"

    assert _by_id(row, "switch-format-a").style["display"] == "none"
    assert _by_id(row, "switch-format-b").style["display"] == "inline-flex"
    assert _by_id(row, "switch-format-c").style["display"] == "inline-flex"
    assert _by_id(row, "switch-format-b").children == "Try My Data"
    assert _by_id(row, "switch-format-c").children == "Try Generic + My Data"
    assert _by_id(row, "share-results-button").style["backgroundColor"] == "#4CBB17"
    assert _by_id(row, "restart-button").style["backgroundColor"] == "#007bff"


def test_played_formats_line_is_hidden() -> None:
    layout = create_final_layout(_user(uses_cgm=True, played=["A"]), "mg/dL", locale="en")
    played = _by_id(layout, "final-played-formats")
    assert played is not None
    assert played.style["display"] == "none"
    assert played.children == ""


def test_no_cgm_shows_only_x_and_share() -> None:
    layout = create_final_layout(_user(uses_cgm=False, played=["A"]), "mg/dL", locale="en")
    row = _by_id(layout, "final-action-row")
    assert _by_id(row, "switch-format-a").style["display"] == "none"
    assert _by_id(row, "switch-format-b").style["display"] == "none"
    assert _by_id(row, "switch-format-c").style["display"] == "none"
    assert _by_id(row, "restart-button") is not None
    assert _by_id(row, "share-results-button") is not None


def test_all_formats_played_hides_switch_buttons() -> None:
    layout = create_final_layout(
        _user(uses_cgm=True, played=["A", "B", "C"]), "mg/dL", locale="en"
    )
    row = _by_id(layout, "final-action-row")
    assert _by_id(row, "switch-format-a").style["display"] == "none"
    assert _by_id(row, "switch-format-b").style["display"] == "none"
    assert _by_id(row, "switch-format-c").style["display"] == "none"
