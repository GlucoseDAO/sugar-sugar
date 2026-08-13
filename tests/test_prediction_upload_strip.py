"""Upload strip on /prediction: only while B/C still need a file."""
from __future__ import annotations

from typing import Any

from sugar_sugar.app import (
    _is_upload_gated,
    _prediction_upload_strip_visible,
    create_prediction_layout,
)


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


def _ids(node: Any) -> set[str]:
    found: set[str] = set()
    node_id = getattr(node, "id", None)
    if isinstance(node_id, str):
        found.add(node_id)
    kids = getattr(node, "children", None)
    if isinstance(kids, (list, tuple)):
        for kid in kids:
            found |= _ids(kid)
    elif kids is not None and not isinstance(kids, str):
        found |= _ids(kids)
    return found


def test_upload_gate_b_and_c_until_file_exists() -> None:
    assert _is_upload_gated({"format": "A"}) is False
    assert _is_upload_gated({"format": "B"}) is True
    assert _is_upload_gated({"format": "C", "current_round_number": 1}) is True
    assert _is_upload_gated({"format": "C", "current_round_number": 2}) is True
    assert _is_upload_gated({"format": "B", "uploaded_data_path": "/tmp/x.csv"}) is False
    assert _is_upload_gated({"format": "C", "uploaded_data_path": "/tmp/x.csv"}) is False
    assert _prediction_upload_strip_visible({"format": "C", "uploaded_data_path": "/tmp/x.csv"}) is False


def test_layout_hides_upload_strip_when_file_already_loaded() -> None:
    layout = create_prediction_layout(
        locale="en",
        format_value="C",
        user_info={
            "format": "C",
            "max_rounds": 12,
            "current_round_number": 2,
            "consent_completed": True,
            "uploaded_data_path": "/tmp/clarity.csv",
            "uploaded_data_filename": "Clarity_Export.csv",
            "data_source_name": "Clarity_Export.csv",
            "rounds": [{"round_number": 1}],
        },
    )
    slot = _by_id(layout, "prediction-upload-slot")
    actions = _by_id(layout, "prediction-mobile-actions")
    assert slot is not None
    assert slot.className == "prediction-upload-hidden"
    assert actions.className == ""
    # Keep the control in the DOM so upload callbacks stay wired.
    assert "upload-data" in _ids(slot)
    gate = _by_id(layout, "prediction-upload-gate")
    assert gate.style["display"] == "none"


def test_layout_shows_upload_strip_only_while_gated() -> None:
    layout = create_prediction_layout(
        locale="en",
        format_value="B",
        user_info={
            "format": "B",
            "max_rounds": 12,
            "current_round_number": 1,
            "consent_completed": True,
        },
    )
    slot = _by_id(layout, "prediction-upload-slot")
    actions = _by_id(layout, "prediction-mobile-actions")
    assert slot.className == "prediction-upload-visible"
    assert actions.className == "has-upload"
    assert "upload-data" in _ids(slot)
    chart = _by_id(layout, "prediction-glucose-chart-container")
    gate = _by_id(layout, "prediction-upload-gate")
    assert chart.style["display"] == "none"
    assert gate.style["display"] == "block"


def test_mixed_without_file_gates_before_graph() -> None:
    layout = create_prediction_layout(
        locale="en",
        format_value="C",
        user_info={
            "format": "C",
            "max_rounds": 12,
            "current_round_number": 1,
            "consent_completed": True,
        },
    )
    chart = _by_id(layout, "prediction-glucose-chart-container")
    gate = _by_id(layout, "prediction-upload-gate")
    actions = _by_id(layout, "prediction-mobile-actions")
    assert chart.style["display"] == "none"
    assert gate.style["display"] == "block"
    assert actions.className == "has-upload"


def test_format_a_never_shows_strip_upload() -> None:
    layout = create_prediction_layout(
        locale="en",
        format_value="A",
        user_info={"format": "A", "max_rounds": 12, "consent_completed": True},
    )
    slot = _by_id(layout, "prediction-upload-slot")
    actions = _by_id(layout, "prediction-mobile-actions")
    assert slot.className == "prediction-upload-hidden"
    assert actions.className == ""
    assert "upload-data" not in _ids(slot)
