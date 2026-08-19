"""The gamified pacing: one click per round, and Finish -> name -> share.

Two behaviours are locked down here.

* ``/ending`` starts the next round on its own after
  ``ENDING_AUTO_ADVANCE_SECONDS``, so an ordinary round costs a single click
  (Submit).  The button and the "Stay on results" escape hatch both remain, and
  the timer is armed **only** by Submit -- never when the player arrived at
  ``/ending`` by pressing Finish/Exit.
* The last round ends at Finish, which lands on ``/final``: the highscore-like
  page whose first card takes a leaderboard name and offers Save & Share.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import polars as pl
import pytest

from sugar_sugar import app as app_module
from sugar_sugar.app import (
    EXAMPLE_DATASET_PATH,
    create_ending_layout,
    dataframe_to_store_dict,
    handle_finish_study_from_prediction,
    handle_next_round_button,
    handle_save_and_share,
    handle_submit_button,
    load_dataset,
    pause_auto_advance,
    tick_auto_advance_countdown,
)
from sugar_sugar.components.submit import is_last_round
from sugar_sugar.config import DEFAULT_POINTS, PREDICTION_HOUR_OFFSET
from sugar_sugar.i18n import t

AUTO_SECONDS: int = app_module.ENDING_AUTO_ADVANCE_SECONDS


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


def _drawn_window() -> pl.DataFrame:
    """An example window with the hidden hour drawn to its last point."""
    full_df, _ = load_dataset(EXAMPLE_DATASET_PATH)
    window = full_df.head(DEFAULT_POINTS)
    size = len(window)
    return window.with_columns(
        pl.when(pl.int_range(pl.len()) >= size - PREDICTION_HOUR_OFFSET)
        .then(pl.col("gl") + 2.0)
        .otherwise(pl.col("prediction"))
        .alias("prediction")
    )


def _table_data(window: pl.DataFrame) -> list[dict[str, str]]:
    actual: dict[str, str] = {"metric": "Actual Glucose"}
    pred: dict[str, str] = {"metric": "Predicted"}
    gl = window.get_column("gl").to_list()
    pr = window.get_column("prediction").to_list()
    for i in range(len(window)):
        actual[f"t{i}"] = f"{float(gl[i]):.1f}"
        pred[f"t{i}"] = "-" if pr[i] == 0.0 else f"{float(pr[i]):.1f}"
    return [actual, pred]


def _session(*, round_n: int, max_rounds: int = 12, **extra: Any) -> dict[str, Any]:
    """A session sitting on round ``round_n`` with the previous rounds stored.

    No ``consent_completed``: `should_persist_study_data` then declines, so these
    tests never touch the study CSVs.
    """
    info: dict[str, Any] = {
        "format": "A",
        "max_rounds": max_rounds,
        "current_round_number": round_n,
        "prediction_window_size": DEFAULT_POINTS,
        "is_example_data": True,
        "data_source_name": "example.csv",
        "rounds": [{"round_number": i} for i in range(1, round_n)],
        "age": 30,
    }
    info.update(extra)
    return info


# --- the last round ends at Finish -----------------------------------------


def test_is_last_round_tracks_the_run_length() -> None:
    assert is_last_round(_session(round_n=12)) is True
    assert is_last_round(_session(round_n=11)) is False
    # A 2-round run is over at round 2, not at round 12.
    assert is_last_round(_session(round_n=2, max_rounds=2)) is True
    # No session at all is not "the last round" -- Submit must not read Finish.
    assert is_last_round(None) is False


def test_the_submit_button_is_told_which_round_it_is_on() -> None:
    """The Finish relabel is computed server-side from ``user-info-store``.

    A Dash callback cannot read a store it does not declare, so losing this State
    would silently leave the last round's button reading "Submit".
    """
    app_module._register_all_callbacks()
    key = next(k for k in app_module.app.callback_map if "submit-button.children" in k)
    states = {
        (dep["id"], dep["property"])
        for dep in app_module.app.callback_map[key]["state"]
    }
    assert ("user-info-store", "data") in states


def test_the_finish_labels_exist_in_every_locale() -> None:
    for locale in ("en", "de", "uk", "ro", "ru", "zh", "fr", "es"):
        finish = t("ui.submit.finish", locale=locale)
        ready = t("ui.submit.progress_finish_ready", locale=locale)
        assert finish and "missing" not in finish.lower()
        assert ready and "missing" not in ready.lower()
        # A relabel that reads the same as Submit would not signal the last round.
        assert finish != t("ui.submit.submit", locale=locale)


def test_submit_on_the_last_round_goes_to_final() -> None:
    window = _drawn_window()
    pathname, info, chart_mode, _window_store = handle_submit_button(
        1, _session(round_n=12), dataframe_to_store_dict(window), 0
    )
    assert pathname == "/final"
    # The round is still recorded -- Finish submits, it does not discard.
    assert len(info["rounds"]) == 12
    assert chart_mode == {"hide_last_hour": False}
    # Nothing to advance to, so no countdown is armed.
    assert info["auto_advance_pending"] is False


def test_submit_mid_run_goes_to_ending_and_arms_the_countdown() -> None:
    window = _drawn_window()
    pathname, info, _mode, _store = handle_submit_button(
        1, _session(round_n=3), dataframe_to_store_dict(window), 0
    )
    assert pathname == "/ending"
    assert len(info["rounds"]) == 3
    assert info["auto_advance_pending"] is True


def test_exit_never_arms_the_countdown() -> None:
    """Finish/Exit from the chart is a deliberate stop, not a pause between rounds."""
    window = _drawn_window()
    pathname, info, _mode, _last_page = handle_finish_study_from_prediction(
        1, _session(round_n=3), dataframe_to_store_dict(window), 0
    )
    assert pathname == "/ending"
    assert info["auto_advance_pending"] is False


def _submitted(round_n: int, *, max_rounds: int = 12) -> dict[str, Any]:
    """A session as it stands on ``/ending``: round ``round_n`` already stored."""
    info = _session(round_n=round_n, max_rounds=max_rounds)
    info["rounds"] = [{"round_number": i} for i in range(1, round_n + 1)]
    return info


# --- the /ending countdown --------------------------------------------------


def _ending(*, round_n: int, armed: bool, max_rounds: int = 12) -> Any:
    window = _drawn_window()
    info = _session(
        round_n=round_n,
        max_rounds=max_rounds,
        prediction_table_data=_table_data(window),
        prediction_window_start=0,
        uses_cgm=False,
        auto_advance_pending=armed,
    )
    info["rounds"] = [{"round_number": i} for i in range(1, round_n + 1)]
    return create_ending_layout(
        dataframe_to_store_dict(window), None, info, "mg/dL", locale="en"
    )


def test_countdown_is_visible_and_both_intervals_run_when_armed() -> None:
    layout = _ending(round_n=3, armed=True)
    assert _by_id(layout, "ending-auto-next-row").style["display"] == "flex"
    assert _by_id(layout, "ending-auto-next-tick").disabled is False
    assert _by_id(layout, "ending-auto-next-timer").disabled is False
    # The single-shot advance waits out the whole countdown.
    assert _by_id(layout, "ending-auto-next-timer").interval == AUTO_SECONDS * 1000
    assert _by_id(layout, "ending-auto-next-timer").max_intervals == 1
    assert _by_id(layout, "ending-auto-next-note").children.endswith(f"{AUTO_SECONDS} s")
    # The button it replaces a click for is still there and still enabled.
    next_btn = _by_id(layout, "next-round-button")
    assert next_btn.disabled is False
    assert next_btn.style["display"] == "inline-flex"


def test_countdown_is_dormant_when_not_armed() -> None:
    """Reaching /ending by Exit shows the results and waits."""
    layout = _ending(round_n=3, armed=False)
    assert _by_id(layout, "ending-auto-next-row").style["display"] == "none"
    assert _by_id(layout, "ending-auto-next-tick").disabled is True
    assert _by_id(layout, "ending-auto-next-timer").disabled is True
    assert _by_id(layout, "ending-auto-next-timer").max_intervals == 0


def test_countdown_never_runs_on_the_last_round() -> None:
    layout = _ending(round_n=12, armed=True)
    assert _by_id(layout, "ending-auto-next-row").style["display"] == "none"
    assert _by_id(layout, "ending-auto-next-timer").disabled is True


def test_both_intervals_stay_mounted_when_dormant() -> None:
    """A callback fires only while every component it references is mounted, and
    `handle_next_round_button` reads the advance interval -- dropping it would
    silence the Next round button as well."""
    layout = _ending(round_n=12, armed=False)
    assert _by_id(layout, "ending-auto-next-tick") is not None
    assert _by_id(layout, "ending-auto-next-timer") is not None


def test_zero_seconds_disables_the_feature(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "ENDING_AUTO_ADVANCE_SECONDS", 0)
    layout = _ending(round_n=3, armed=True)
    assert _by_id(layout, "ending-auto-next-row").style["display"] == "none"
    assert _by_id(layout, "ending-auto-next-timer").disabled is True
    # ... and a stray tick cannot advance a round either.
    assert handle_next_round_button(None, 1, _submitted(3))[0] is app_module.no_update


def test_the_caption_counts_down_then_announces_the_start() -> None:
    assert str(AUTO_SECONDS - 1) in tick_auto_advance_countdown(1, "en")
    assert tick_auto_advance_countdown(AUTO_SECONDS, "en") == "Starting the next round…"
    # Overshoot must not produce a negative countdown.
    assert tick_auto_advance_countdown(AUTO_SECONDS + 3, "en") == "Starting the next round…"


def test_pausing_stops_both_intervals_and_hides_the_row() -> None:
    tick_off, advance_off, style = pause_auto_advance(1, {"display": "flex", "gap": "12px"})
    assert (tick_off, advance_off) == (True, True)
    assert style["display"] == "none"
    assert style["gap"] == "12px"  # only the visibility changes


def test_the_timer_advances_without_a_click() -> None:
    result = handle_next_round_button(None, 1, _submitted(3))
    assert result[0] == "/prediction"
    assert result[1]["current_round_number"] == 4
    # The timer is spent, so a later re-render of /ending cannot re-arm it.
    assert result[1]["auto_advance_pending"] is False


def test_an_untriggered_render_changes_nothing() -> None:
    assert handle_next_round_button(None, 0, _submitted(3))[0] is app_module.no_update


def test_the_button_still_advances() -> None:
    result = handle_next_round_button(1, 0, _submitted(3))
    assert result[0] == "/prediction"
    assert result[1]["current_round_number"] == 4


def test_the_timer_cannot_run_past_the_last_round() -> None:
    assert handle_next_round_button(None, 1, _submitted(12))[0] is app_module.no_update


# --- Save & Share on /final -------------------------------------------------


@pytest.fixture()
def share_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Throwaway share + ranking trees, and a nickname write that stays in memory."""
    (tmp_path / "data" / "input").mkdir(parents=True)
    monkeypatch.setenv("SUGAR_SHARE_DIR", str(tmp_path / "shares"))
    monkeypatch.setattr(app_module, "project_root", tmp_path)
    written: dict[str, str] = {}

    def _record(*, study_id: str, key: str, nickname: str) -> int:
        written["study_id"] = study_id
        written["nickname"] = nickname
        return 1

    monkeypatch.setattr(app_module.submit_component, "set_study_nickname", _record)
    monkeypatch.setattr(app_module, "_nickname_writes", written, raising=False)
    yield tmp_path


def _finished_session() -> dict[str, Any]:
    window = _drawn_window()
    info = _session(round_n=12, prediction_table_data=_table_data(window))
    info["study_id"] = "s1"
    info["rounds"] = [
        {
            "round_number": i,
            "format": "A",
            "is_example_data": True,
            "data_source_name": "example.csv",
            "prediction_table_data": _table_data(window),
        }
        for i in range(1, 13)
    ]
    return info


def test_save_and_share_stores_the_name_and_lands_on_the_share_page(
    share_root: Path,
) -> None:
    pathname, info, status = handle_save_and_share(1, "  Ninja  ", _finished_session(), "en")

    assert pathname.startswith("/share/")
    assert info["nickname"] == "Ninja"  # normalised
    assert app_module._nickname_writes == {"study_id": "s1", "nickname": "Ninja"}
    assert status

    # The share record on disk carries the name just typed, not the old one.
    share_id = pathname.rsplit("/", 1)[-1]
    record = json.loads((share_root / "shares" / f"{share_id}.json").read_text(encoding="utf-8"))
    assert record["user_info"]["nickname"] == "Ninja"
    assert len(record["rounds"]) == 12


def test_save_and_share_with_nothing_played_keeps_the_name_and_stays(
    share_root: Path,
) -> None:
    pathname, info, status = handle_save_and_share(
        1, "Ninja", {"study_id": "s1", "format": "A", "rounds": []}, "en"
    )
    assert pathname is app_module.no_update
    assert info["nickname"] == "Ninja"
    assert status


def test_an_empty_name_is_allowed_and_stays_anonymous(share_root: Path) -> None:
    """The field is optional: Save & Share with a blank box still shares."""
    pathname, info, _status = handle_save_and_share(1, "   ", _finished_session(), "en")
    assert pathname.startswith("/share/")
    assert info["nickname"] == ""
