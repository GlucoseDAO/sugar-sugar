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


# --------------------------------------------------------------------------
# Nightscout in the upload gate.
#
# A B/C player who has not provided data yet is shown a gate telling them to
# press Upload and pick a CGM file -- and until this, a CSV button was the only
# thing on the page. Someone whose data lives on a Nightscout site and nowhere
# else had no way forward, which is what "I chose to play with my own data and
# there was no Nightscout button" describes.
# --------------------------------------------------------------------------

NIGHTSCOUT_IDS: frozenset[str] = frozenset(
    {'nightscout-url-input', 'nightscout-token-input', 'nightscout-load-button', 'nightscout-status'}
)


def _upload_section_display(layout: Any) -> Any:
    """The upload section's own display, read off the parent of its tabs."""

    def walk(node: Any, parent: Any = None) -> Any:
        if getattr(node, "id", None) == "data-input-tabs":
            return (getattr(parent, "style", None) or {}).get("display")
        kids = getattr(node, "children", None)
        if isinstance(kids, (list, tuple)):
            for kid in kids:
                hit = walk(kid, node)
                if hit is not None:
                    return hit
        elif kids is not None and not isinstance(kids, str):
            return walk(kids, node)
        return None

    return walk(layout)


def _count_ids(node: Any, target: str) -> int:
    total = 1 if getattr(node, "id", None) == target else 0
    kids = getattr(node, "children", None)
    if isinstance(kids, (list, tuple)):
        for kid in kids:
            total += _count_ids(kid, target)
    elif kids is not None and not isinstance(kids, str):
        total += _count_ids(kids, target)
    return total


def _layout(format_value: str, **info: Any) -> Any:
    user_info = {"format": format_value, "consent_completed": True, **info}
    return create_prediction_layout(locale="en", format_value=format_value, user_info=user_info)


def test_nightscout_controls_are_always_in_the_dom() -> None:
    """The language-change callback outputs to them on every format.

    A Dash callback whose component is missing raises, so these ids may be
    hidden but never dropped.
    """
    for format_value in ("A", "B", "C"):
        assert NIGHTSCOUT_IDS <= _ids(_layout(format_value)), format_value


def test_gated_own_data_player_can_reach_nightscout() -> None:
    for format_value in ("B", "C"):
        assert _upload_section_display(_layout(format_value)) == "block", format_value


def test_nightscout_is_hidden_once_data_is_loaded() -> None:
    """It must not steal chart space once the player is actually playing."""
    layout = _layout("B", uploaded_data_path="/tmp/example.csv")
    assert _upload_section_display(layout) == "none"


def test_public_data_format_never_shows_it() -> None:
    assert _upload_section_display(_layout("A")) == "none"


def test_gate_does_not_duplicate_the_csv_upload() -> None:
    """`upload-data` lives in the action strip for B/C.

    Rendering the CSV tab here too would put the id on the page twice, which
    breaks the callback that reads it.
    """
    for format_value in ("A", "B", "C"):
        assert _count_ids(_layout(format_value), "upload-data") == 1, format_value
