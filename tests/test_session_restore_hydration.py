"""Cold-load hydration guard for the game routes.

Regression cover for the reported mobile failure: a player who saved a session on
Friday and re-opened the app (Samsung/Android, portrait, Romanian) landed back on
the consent form. Cause: a *full page load* whose URL is already a game route
renders before the localStorage stores hydrate, and `display_page` reads them as
``State``, so `/prediction` fell through to the landing page (→ consent wizard on
mobile) and nothing ever re-rendered.

The tests drive the real callbacks over HTTP with a mobile User-Agent, because the
UA header (not the async `user-agent` store) is what picks the mobile builders.
"""
from __future__ import annotations

import json
from typing import Any, Optional

import pytest

from sugar_sugar.app import (
    _GAME_ROUTES,
    _RESTORE_GIVE_UP_TICKS,
    _game_stores_ready,
    _renders_prediction_chart,
    app,
)

MOBILE_UA: str = (
    "Mozilla/5.0 (Linux; Android 16; SM-S931B) AppleWebKit/537.36 "
    "(KHTML, like Gecko) SamsungBrowser/28.0 Chrome/130.0.0.0 Mobile Safari/537.36"
)

_DISPLAY_OUTPUT: str = (
    "..page-content.children...mobile-warning.children...navbar-container.children.."
)

_WINDOW_POINTS: int = 24


def _saved_session() -> dict[str, Any]:
    """A mid-study session as it would sit in localStorage after a few rounds."""
    return {
        "study_id": "study-abc",
        "consent_completed": True,
        "consent_gdpr": True,
        "format": "A",
        "email": "player@example.com",
        "age": 34,
        "gender": "F",
        "location": "Bucharest, Romania",
        "resume_code": "ABCDEF",
        "rounds": [{"round_number": 1, "mae": 12.0, "rmse": 15.0}],
        "current_round_number": 2,
        "max_rounds": 12,
        "is_example_data": True,
        "data_source_name": "example.csv",
        "prediction_table_data": [
            {"metric": "actual", "10:00": "100"},
            {"metric": "predicted", "10:00": "110"},
        ],
    }


def _window_store() -> dict[str, list[Any]]:
    """A `current-window-df` payload in the shape reconstruct_dataframe_from_dict wants."""
    times = [
        f"2026-08-07T10:{5 * i:02d}:00" if 5 * i < 60 else f"2026-08-07T11:{5 * i - 60:02d}:00"
        for i in range(_WINDOW_POINTS)
    ]
    return {
        "time": times,
        "gl": [120.0 + i for i in range(_WINDOW_POINTS)],
        "prediction": [None] * _WINDOW_POINTS,
        "age": [34] * _WINDOW_POINTS,
        "user_id": [1] * _WINDOW_POINTS,
    }


@pytest.fixture(name="client")
def _client() -> Any:
    return app.server.test_client()


def _render(
    client: Any,
    pathname: str,
    *,
    user_info: Optional[dict[str, Any]] = None,
    current_df: Optional[dict[str, list[Any]]] = None,
    hydrated: bool = False,
    locale: str = "ro",
) -> str:
    """Invoke `display_page` over HTTP and return its page-content as JSON text."""
    body = {
        "output": _DISPLAY_OUTPUT,
        "outputs": [
            {"id": "page-content", "property": "children"},
            {"id": "mobile-warning", "property": "children"},
            {"id": "navbar-container", "property": "children"},
        ],
        "inputs": [
            {"id": "url", "property": "pathname", "value": pathname},
            {"id": "game-stores-hydrated", "property": "data", "value": hydrated},
        ],
        "state": [
            {"id": "interface-language", "property": "data", "value": locale},
            {"id": "user-info-store", "property": "data", "value": user_info},
            {"id": "current-window-df", "property": "data", "value": current_df},
            {"id": "events-df", "property": "data", "value": None},
            {"id": "glucose-unit", "property": "data", "value": "mg/dL"},
            {"id": "user-agent", "property": "data", "value": MOBILE_UA},
        ],
        "changedPropIds": ["url.pathname"],
    }
    response = client.post(
        "/_dash-update-component",
        data=json.dumps(body),
        content_type="application/json",
        headers={"User-Agent": MOBILE_UA},
    )
    assert response.status_code == 200, response.get_data(as_text=True)[:400]
    return json.dumps(response.get_json()["response"]["page-content"]["children"], ensure_ascii=False)


def _poll(
    client: Any,
    pathname: str,
    *,
    user_info: Optional[dict[str, Any]],
    current_df: Optional[dict[str, list[Any]]],
    n_intervals: int,
) -> Optional[dict[str, Any]]:
    """Invoke `resolve_session_restore`; None means PreventUpdate (HTTP 204)."""
    key = next(
        k
        for k in app.callback_map
        if "game-stores-hydrated.data" in k and "url.pathname" in k
    )
    body = {
        "output": key,
        "outputs": [
            {"id": "game-stores-hydrated", "property": "data"},
            {"id": "url", "property": "pathname"},
        ],
        "inputs": [{"id": "session-restore-poll", "property": "n_intervals", "value": n_intervals}],
        "state": [
            {"id": "user-info-store", "property": "data", "value": user_info},
            {"id": "current-window-df", "property": "data", "value": current_df},
            {"id": "url", "property": "pathname", "value": pathname},
        ],
        "changedPropIds": ["session-restore-poll.n_intervals"],
    }
    response = client.post(
        "/_dash-update-component",
        data=json.dumps(body),
        content_type="application/json",
        headers={"User-Agent": MOBILE_UA},
    )
    if response.status_code == 204:
        return None
    assert response.status_code == 200, response.get_data(as_text=True)[:400]
    return response.get_json()["response"]


def _route_flag(
    client: Any,
    pathname: str,
    *,
    user_info: Optional[dict[str, Any]],
    hydrated: bool = True,
) -> bool:
    """Invoke `mark_prediction_chart_rendered` (drives the route-prediction class)."""
    body = {
        "output": "prediction-chart-rendered.data",
        "outputs": {"id": "prediction-chart-rendered", "property": "data"},
        "inputs": [
            {"id": "url", "property": "pathname", "value": pathname},
            {"id": "game-stores-hydrated", "property": "data", "value": hydrated},
        ],
        "state": [{"id": "user-info-store", "property": "data", "value": user_info}],
        "changedPropIds": ["url.pathname"],
    }
    response = client.post(
        "/_dash-update-component",
        data=json.dumps(body),
        content_type="application/json",
        headers={"User-Agent": MOBILE_UA},
    )
    assert response.status_code == 200, response.get_data(as_text=True)[:400]
    return bool(response.get_json()["response"]["prediction-chart-rendered"]["data"])


# --- the predicates ---------------------------------------------------------


@pytest.mark.parametrize(
    "pathname, user_info, current_df, expected",
    [
        # Un-hydrated game routes are not ready...
        ("/prediction", None, None, False),
        ("/ending", None, None, False),
        ("/final", None, None, False),
        # ...but non-game routes never depend on the stores.
        ("/", None, None, True),
        ("/startup", None, None, True),
        ("/faq", None, None, True),
        ("/highscore", None, None, True),
        # /ending additionally needs the played window back.
        ("/ending", {"study_id": "a"}, None, False),
        ("/ending", {"study_id": "a"}, {"time": []}, True),
        ("/prediction", {"study_id": "a"}, None, True),
        ("/final", {"study_id": "a"}, None, True),
    ],
)
def test_game_stores_ready(
    pathname: str,
    user_info: Optional[dict[str, Any]],
    current_df: Optional[dict[str, Any]],
    expected: bool,
) -> None:
    assert _game_stores_ready(pathname, user_info, current_df) is expected


def test_game_routes_are_the_store_backed_ones() -> None:
    assert _GAME_ROUTES == frozenset({"/prediction", "/ending", "/final"})


@pytest.mark.parametrize(
    "pathname, user_info, expected",
    [
        ("/prediction", {"consent_completed": True}, True),
        # No consent on record -> display_page bounces to landing, so the
        # route-prediction CSS class must stay off.
        ("/prediction", {"consent_completed": False}, False),
        ("/prediction", {}, False),
        # Un-hydrated cold load: the placeholder is on screen, not the chart.
        ("/prediction", None, False),
        ("/ending", {"consent_completed": True}, False),
        ("/", {"consent_completed": True}, False),
    ],
)
def test_renders_prediction_chart(
    pathname: str, user_info: Optional[dict[str, Any]], expected: bool
) -> None:
    assert _renders_prediction_chart(pathname, user_info) is expected


# --- display_page -----------------------------------------------------------


@pytest.mark.parametrize("pathname", sorted(_GAME_ROUTES))
def test_cold_load_on_game_route_holds_a_placeholder(client: Any, pathname: str) -> None:
    """The reported bug: never render landing/consent (or "session expired") first."""
    content = _render(client, pathname, user_info=None)
    assert "session-restoring" in content
    assert "session-restore-poll" in content
    # The exact regression: no landing page, and so no consent checkboxes.
    assert "landing-page" not in content
    assert "consent-acknowledge" not in content
    assert "Session Expired" not in content


def test_placeholder_is_localized(client: Any) -> None:
    content = _render(client, "/prediction", user_info=None, locale="ro")
    assert "Îți restaurăm jocul" in content


def test_hydrated_cold_load_renders_the_game(client: Any) -> None:
    """After the poll flips the flag, the same URL renders the real page."""
    content = _render(
        client, "/prediction", user_info=_saved_session(), current_df=_window_store(), hydrated=True
    )
    assert "prediction-glucose-chart-container" in content
    assert "session-restoring" not in content
    assert "landing-page" not in content


def test_hydrated_cold_load_renders_ending(client: Any) -> None:
    content = _render(
        client, "/ending", user_info=_saved_session(), current_df=_window_store(), hydrated=True
    )
    assert "ending-title" in content
    assert "session-restoring" not in content
    assert "Session Expired" not in content


def test_missing_consent_still_bounces_to_landing(client: Any) -> None:
    """The consent guard is untouched: a session without consent goes to landing."""
    info = _saved_session()
    del info["consent_completed"]
    content = _render(client, "/prediction", user_info=info, current_df=_window_store(), hydrated=True)
    assert "landing-page" in content
    assert "session-restoring" not in content


def test_fresh_visitor_flow_unaffected(client: Any) -> None:
    """`/` and `/startup` must not be gated by the hydration guard."""
    assert "landing-page" in _render(client, "/", user_info=None)
    startup = _render(client, "/startup", user_info=None)
    assert "startup-page" in startup
    # Mobile /startup is the consent entry, so its checkboxes stay put.
    assert "consent-acknowledge" in startup


# --- restore poll -----------------------------------------------------------


def test_poll_flips_flag_once_stores_arrive(client: Any) -> None:
    response = _poll(
        client, "/prediction", user_info=_saved_session(), current_df=_window_store(), n_intervals=1
    )
    assert response == {"game-stores-hydrated": {"data": True}}


def test_poll_keeps_waiting_while_stores_are_empty(client: Any) -> None:
    assert _poll(client, "/prediction", user_info=None, current_df=None, n_intervals=1) is None
    # /ending waits for the window store too, not just user_info.
    assert _poll(client, "/ending", user_info=_saved_session(), current_df=None, n_intervals=2) is None


def test_poll_gives_up_to_landing_when_there_is_no_session(client: Any) -> None:
    """A deep link with genuinely empty localStorage must not spin forever."""
    response = _poll(
        client, "/prediction", user_info=None, current_df=None, n_intervals=_RESTORE_GIVE_UP_TICKS
    )
    assert response == {"url": {"pathname": "/"}}


def test_poll_is_scoped_to_the_placeholder(client: Any) -> None:
    """Only the placeholder's interval can trigger a re-render.

    `session-restore-poll` must live *inside* `_restoring_layout` and nowhere in
    the base layout: a Dash callback only fires while its components are mounted,
    which is what makes it impossible for this re-render to fire mid-round and
    reset the chart the player is drawing on.
    """
    layout = client.get("/_dash-layout").get_data(as_text=True)
    assert "session-restore-poll" not in layout
    assert "game-stores-hydrated" in layout

    writers = [
        [dep["id"] + "." + dep["property"] for dep in cb["inputs"]]
        for key, cb in app.callback_map.items()
        if "game-stores-hydrated.data" in key
    ]
    assert writers == [["session-restore-poll.n_intervals"]]


# --- route-prediction class -------------------------------------------------


@pytest.mark.parametrize(
    "pathname, user_info, expected",
    [
        ("/prediction", _saved_session(), True),
        # URL says /prediction but the content is the placeholder or the consent
        # bounce: the class must stay off, or mobile.css releases its overflow cap
        # and `touch-action: manipulation` on that foreign content (taps get
        # swallowed -- "the button activated but she couldn't progress").
        ("/prediction", None, False),
        ("/prediction", {k: v for k, v in _saved_session().items() if k != "consent_completed"}, False),
        ("/ending", _saved_session(), False),
        ("/startup", _saved_session(), False),
    ],
)
def test_route_prediction_flag_follows_the_render(
    client: Any, pathname: str, user_info: Optional[dict[str, Any]], expected: bool
) -> None:
    assert _route_flag(client, pathname, user_info=user_info) is expected


# --- restore_page_on_load hydration ORDER -----------------------------------
#
# The second half of the same report, on the plain `/` entry path: no unusual URL
# needed. `last-visited-page` mounts *after* user-info-store / current-window-df,
# so the callback's first firing carries a populated session with last_page still
# None. It used to burn its one-shot guard there, so the dialog/redirect never
# happened and the player was left on the landing page -> "Take me in" -> the
# consent wizard.


def _restore(
    client: Any,
    *,
    changed: list[str],
    last_page: Optional[str],
    user_info: Optional[dict[str, Any]],
    current_df: Optional[dict[str, list[Any]]],
    done: bool = False,
    pathname: str = "/",
    session_active: bool = False,
) -> Optional[dict[str, Any]]:
    """Invoke `restore_page_on_load`; None means PreventUpdate (HTTP 204)."""
    key = next(
        k
        for k in app.callback_map
        if "resume-dialog-target.data" in k and "page-restore-done.data" in k
    )
    body = {
        "output": key,
        "outputs": [
            {"id": "resume-dialog-target", "property": "data"},
            {"id": "page-restore-done", "property": "data"},
            {"id": "url", "property": "pathname"},
            {"id": "session-active", "property": "data"},
        ],
        "inputs": [
            {"id": "last-visited-page", "property": "data", "value": last_page},
            {"id": "user-info-store", "property": "data", "value": user_info},
            {"id": "current-window-df", "property": "data", "value": current_df},
        ],
        "state": [
            {"id": "page-restore-done", "property": "data", "value": done},
            {"id": "url", "property": "pathname", "value": pathname},
            {"id": "session-active", "property": "data", "value": session_active},
        ],
        "changedPropIds": changed,
    }
    response = client.post(
        "/_dash-update-component",
        data=json.dumps(body),
        content_type="application/json",
        headers={"User-Agent": MOBILE_UA},
    )
    if response.status_code == 204:
        return None
    assert response.status_code == 200, response.get_data(as_text=True)[:400]
    return response.get_json()["response"]


def test_last_visited_page_mounts_after_the_session_stores() -> None:
    """The layout order that makes the race real — keep the guard if it changes."""
    layout = app.server.test_client().get("/_dash-layout").get_data(as_text=True)
    assert layout.index("user-info-store") < layout.index("last-visited-page")
    assert layout.index("current-window-df") < layout.index("last-visited-page")


@pytest.mark.parametrize(
    "changed, user_info, current_df",
    [
        (["user-info-store.data"], "session", None),
        (["current-window-df.data"], None, "window"),
        (["user-info-store.data"], "session", "window"),
    ],
)
def test_restore_waits_when_last_visited_page_has_not_hydrated(
    client: Any, changed: list[str], user_info: Optional[str], current_df: Optional[str]
) -> None:
    """A populated session + a still-None last_page must NOT spend the one-shot."""
    result = _restore(
        client,
        changed=changed,
        last_page=None,
        user_info=_saved_session() if user_info else None,
        current_df=_window_store() if current_df else None,
    )
    assert result is None


def test_restore_shows_the_dialog_when_last_visited_page_finally_arrives(client: Any) -> None:
    result = _restore(
        client,
        changed=["last-visited-page.data"],
        last_page="/prediction",
        user_info=_saved_session(),
        current_df=_window_store(),
    )
    assert result is not None
    assert result["resume-dialog-target"]["data"]["target"] == "/prediction"
    assert result["page-restore-done"]["data"] is True


def test_restore_redirects_silently_within_an_active_tab(client: Any) -> None:
    """Scenario 3 of the navigation contract: no dialog on an in-tab reload."""
    result = _restore(
        client,
        changed=["last-visited-page.data"],
        last_page="/prediction",
        user_info=_saved_session(),
        current_df=_window_store(),
        session_active=True,
    )
    assert result is not None
    assert result["url"]["pathname"] == "/prediction"
    assert "resume-dialog-target" not in result


def test_restore_marks_done_for_a_genuinely_fresh_visitor(client: Any) -> None:
    """Empty localStorage everywhere: nothing to restore, so stop looking."""
    result = _restore(
        client, changed=["user-info-store.data"], last_page=None, user_info=None, current_df=None
    )
    assert result is not None
    assert result["page-restore-done"]["data"] is True
    assert "resume-dialog-target" not in result


def test_restore_one_shot_guard_still_holds(client: Any) -> None:
    result = _restore(
        client,
        changed=["user-info-store.data"],
        last_page="/prediction",
        user_info=_saved_session(),
        current_df=_window_store(),
        done=True,
    )
    assert result is None


def test_viewport_callback_consumes_the_render_flag() -> None:
    """The clientside route-class callback must read the flag, not just the URL."""
    key = next(k for k in app.callback_map if "viewport-sink.children" in k)
    inputs = [dep["id"] + "." + dep["property"] for dep in app.callback_map[key]["inputs"]]
    assert inputs == ["url.pathname", "prediction-chart-rendered.data"]
