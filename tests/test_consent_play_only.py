"""Play-only is gone: mandatory consent is study consent.

The August 2026 incident: a player filled the startup form, played, exited,
and nothing landed in prediction_statistics.csv -- their consent row had
play_only=True. That checkbox is removed. Leftover localStorage with
``consent_play_only=True`` must still persist on Exit.

There is also no 6- or 12-round minimum at the write boundary: two submitted
rounds on Exit are stored the same way as a full run. ``uv run chart`` skips
writes via ``_CHART_MODE`` so local debug does not pollute the study CSVs.
"""

from __future__ import annotations

from typing import Any

from sugar_sugar.consent import (
    reconcile_stored_consents,
    resolve_optional_consents,
    should_persist_study_data,
)


def test_no_optional_boxes_still_participates() -> None:
    play_only, participate, no_selection = resolve_optional_consents(
        receive_results=False, keep_updated=False
    )
    assert play_only is False
    assert participate is True
    assert no_selection is True


def test_receive_results_is_an_extra_not_a_gate() -> None:
    play_only, participate, no_selection = resolve_optional_consents(
        receive_results=True, keep_updated=False
    )
    assert play_only is False
    assert participate is True
    assert no_selection is False


def test_keep_updated_is_an_extra_not_a_gate() -> None:
    play_only, participate, no_selection = resolve_optional_consents(
        receive_results=False, keep_updated=True
    )
    assert play_only is False
    assert participate is True
    assert no_selection is False


def _session(**overrides: Any) -> dict[str, Any]:
    info: dict[str, Any] = {
        "consent_completed": True,
        "consent_play_only": False,
        "statistics_saved": False,
        "rounds": [{"round_number": 1}],
    }
    info.update(overrides)
    return info


def test_two_rounds_are_enough_to_persist() -> None:
    info = _session(rounds=[{"round_number": 1}, {"round_number": 2}])
    assert should_persist_study_data(info) is True


def test_thirty_six_rounds_persist() -> None:
    info = _session(rounds=[{"round_number": i} for i in range(1, 37)])
    assert should_persist_study_data(info) is True


def test_stale_play_only_session_still_persists() -> None:
    """In-progress localStorage from when the checkbox existed still saves."""
    info = _session(
        consent_play_only=True,
        rounds=[{"round_number": 1}, {"round_number": 2}],
    )
    assert should_persist_study_data(info) is True
    assert info["consent_play_only"] is False
    assert info["consent_participate_in_study"] is True


def test_stale_contradictory_session_still_persists() -> None:
    info = _session(
        consent_play_only=True,
        consent_receive_results_later=True,
        consent_keep_up_to_date=True,
        consent_upload_own_data=True,
        rounds=[{"round_number": 1}, {"round_number": 2}],
    )
    assert should_persist_study_data(info) is True
    assert info["consent_play_only"] is False
    assert info["consent_participate_in_study"] is True


def test_already_saved_is_updated_again() -> None:
    """Incremental upsert: Exit after a mid-game save must still refresh the row."""
    info = _session(statistics_saved=True)
    assert should_persist_study_data(info) is True


def test_starter_with_no_rounds_is_still_written() -> None:
    """People who hit Start and forget before submitting still leave a study row."""
    info = _session(rounds=[])
    assert should_persist_study_data(info) is True


def test_missing_consent_is_not_written() -> None:
    info = _session(consent_completed=False)
    assert should_persist_study_data(info) is False


def test_reconcile_clears_stale_play_only() -> None:
    info = reconcile_stored_consents(_session(consent_play_only=True, rounds=[]))
    assert info["consent_play_only"] is False
    assert info["consent_participate_in_study"] is True


def test_chart_mode_does_not_persist(monkeypatch: Any) -> None:
    monkeypatch.setenv("_CHART_MODE", "1")
    assert should_persist_study_data(_session()) is False


def test_play_only_checkbox_absent_from_shared_and_mobile_consent() -> None:
    """Mobile wizard step 0 imports the same children as desktop landing."""
    from sugar_sugar.components.landing import consent_controls_children
    from sugar_sugar.components.startup import StartupPageMobile
    from sugar_sugar.i18n import setup_i18n

    setup_i18n()
    shared = "".join(str(child.to_plotly_json()) for child in consent_controls_children("en"))
    assert "consent-play-only" not in shared
    assert "consent-acknowledge" in shared
    assert "consent-gdpr" in shared

    mobile = str(StartupPageMobile(locale="en").to_plotly_json())
    assert "consent-play-only" not in mobile
    assert "consent-acknowledge" in mobile
