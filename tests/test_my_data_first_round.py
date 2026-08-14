"""Format B (My Data) must open on the uploaded file, not a leftover generic window.

Also: Exit from /final must not bounce a cleared session back to /startup
or /prediction (that reused the old form and hid the upload box).
"""
from __future__ import annotations

from typing import Any

import pytest
from dash import no_update
from dash.exceptions import PreventUpdate

from sugar_sugar.app import (
    EXAMPLE_DATASET_PATH,
    _load_round_one_stores,
    initialize_data_on_url_change,
    redirect_landing_to_game,
)


def _own_info() -> dict[str, Any]:
    return {
        "format": "B",
        "uploaded_data_path": str(EXAMPLE_DATASET_PATH),
        "uploaded_data_filename": "my_clarity.csv",
        "is_example_data": False,
        "data_source_name": "my_clarity.csv",
        "consent_completed": True,
    }


def test_round_one_stores_use_the_uploaded_file_for_my_data() -> None:
    window, _events, is_example, source, slider, _rand = _load_round_one_stores(_own_info())
    assert is_example is False
    assert source == "my_clarity.csv"
    assert window is not None
    assert window.get("time")
    assert isinstance(slider, int)


def test_initialize_data_replaces_leftover_generic_window_for_my_data() -> None:
    leftover = {
        "time": ["2020-01-01T00:00:00"],
        "gl": [100.0],
        "prediction": [0.0],
        "age": [1],
        "user_id": [1],
    }
    window, _events, is_example, source, _rand, _slider = initialize_data_on_url_change(
        "/prediction", leftover, _own_info(), "loop_758_chronological.csv"
    )
    assert is_example is False
    assert source == "my_clarity.csv"
    assert window != leftover
    assert window is not no_update


def test_initialize_data_keeps_matching_own_data_window() -> None:
    leftover = {
        "time": ["2020-01-01T00:00:00"],
        "gl": [100.0],
        "prediction": [0.0],
        "age": [1],
        "user_id": [1],
    }
    result = initialize_data_on_url_change(
        "/prediction", leftover, _own_info(), "my_clarity.csv"
    )
    assert result[0] is no_update


@pytest.mark.parametrize("last_page", ["/final", "/startup", "/prediction", "/ending", "/profile"])
def test_redirect_does_not_bounce_a_cleared_session(last_page: str) -> None:
    with pytest.raises(PreventUpdate):
        redirect_landing_to_game("/", last_page, None, None)


def test_redirect_still_returns_in_session_players_to_their_game() -> None:
    assert redirect_landing_to_game("/", "/startup", {"consent_completed": True}, None) == "/startup"
    assert redirect_landing_to_game("/", "/prediction", {"consent_completed": True}, None) == "/prediction"
