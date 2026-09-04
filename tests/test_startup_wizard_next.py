"""Mobile wizard Next must stay a real button on Samsung Internet.

Fomantic's ``.ui.button:disabled { pointer-events: none !important }`` plus
Samsung Internet's failure to restore hit-testing after React removes ``disabled``
swallows taps: the button turns blue and Dash never sees ``n_clicks``. The gate
is therefore class + store only; ``navigate_startup_wizard`` no-ops when the
store says the step is incomplete.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import dash
import pytest
from dash.exceptions import PreventUpdate

from sugar_sugar.app import _register_all_callbacks, app

SAMSUNG_UA: str = (
    "Mozilla/5.0 (Linux; Android 16; SM-S931B) AppleWebKit/537.36 "
    "(KHTML, like Gecko) SamsungBrowser/28.0 Chrome/130.0.0.0 Mobile Safari/537.36"
)

_DISPLAY_OUTPUT: str = (
    "..page-content.children...mobile-warning.children..."
    "navbar-container.children...final-fill-step.data.."
)
from sugar_sugar.components.startup import (
    WIZARD_STEPS,
    StartupPage,
    StartupPageMobile,
    wizard_next_is_allowed,
)
from sugar_sugar.i18n import setup_i18n


@pytest.fixture(scope="module", autouse=True)
def _load_translations() -> None:
    setup_i18n()


def _ids(node: Any) -> set[str]:
    found: set[str] = set()
    node_id = getattr(node, "id", None)
    if isinstance(node_id, str):
        found.add(node_id)
    children = getattr(node, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            found |= _ids(child)
    elif children is not None and not isinstance(children, str):
        found |= _ids(children)
    return found


def _find(node: Any, target: str) -> Any:
    if getattr(node, "id", None) == target:
        return node
    children = getattr(node, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            hit = _find(child, target)
            if hit is not None:
                return hit
    elif children is not None and not isinstance(children, str):
        return _find(children, target)
    return None


def test_mobile_next_is_not_html_disabled_on_first_paint() -> None:
    """A native disabled button is what Samsung Internet cannot re-enable."""
    page = StartupPageMobile(locale="en")
    button = _find(page, "startup-next")
    assert button is not None
    assert getattr(button, "disabled", None) in (None, False)
    assert "startup-next-disabled" in (button.className or "")


def test_mobile_next_allowed_store_is_on_the_page() -> None:
    page = StartupPageMobile(locale="en")
    store = _find(page, "startup-next-allowed")
    assert store is not None
    assert store.data is False
    assert "startup-next-allowed" in _ids(page)


def test_consent_step_requires_both_mandatory_boxes() -> None:
    kwargs: dict[str, Any] = dict(
        step=0,
        acknowledge_value=None,
        gdpr_value=None,
        email=None,
        age=None,
        gender=None,
        uses_cgm=None,
        cgm_duration=None,
        cgm_duration_unit=None,
        is_diabetic=None,
        diabetic_type=None,
        diabetes_duration=None,
        location=None,
        format_value=None,
        data_usage_consent=None,
        user_info=None,
        locale="en",
    )
    assert wizard_next_is_allowed(**kwargs) is False
    assert wizard_next_is_allowed(**{**kwargs, "acknowledge_value": ["ack"]}) is False
    assert wizard_next_is_allowed(**{**kwargs, "gdpr_value": ["gdpr"]}) is False
    assert wizard_next_is_allowed(
        **{**kwargs, "acknowledge_value": ["ack"], "gdpr_value": ["gdpr"]}
    ) is True


def test_gate_callback_does_not_write_disabled() -> None:
    _register_all_callbacks()
    outputs = [
        part
        for key in app.callback_map
        for part in key.strip(".").split("...")
        if "startup-next" in part
    ]
    assert outputs, "startup-next callbacks were not registered"
    assert "startup-next.disabled" not in outputs
    assert any("startup-next-allowed.data" in part for part in outputs)
    assert any("startup-next.className" in part for part in outputs)


def _register_wizard_callbacks() -> dict[str, Any]:
    captured: dict[str, Any] = {}

    class _App:
        def callback(self, *args: Any, **kwargs: Any) -> Any:
            def wrap(func: Any) -> Any:
                captured[func.__name__] = func
                return func

            return wrap

        def clientside_callback(self, *args: Any, **kwargs: Any) -> None:
            return None

    StartupPage(locale="en").register_callbacks(_App())  # type: ignore[arg-type]
    return captured


def _with_trigger(trigger: Optional[str], func: Any, *args: Any) -> Any:
    class _Ctx:
        triggered_id = trigger

    original = dash.callback_context
    dash.callback_context = _Ctx()  # type: ignore[assignment]
    try:
        return func(*args)
    finally:
        dash.callback_context = original  # type: ignore[assignment]


def test_navigate_refuses_next_when_store_says_blocked() -> None:
    navigate = _register_wizard_callbacks()["navigate_startup_wizard"]
    with pytest.raises(PreventUpdate):
        _with_trigger("startup-next", navigate, 1, 1, 0, "en", False)


def test_navigate_advances_when_store_allows() -> None:
    navigate = _register_wizard_callbacks()["navigate_startup_wizard"]
    result = _with_trigger("startup-next", navigate, 1, 1, 0, "en", True)
    assert result[0] == 1
    assert result[1]["display"] == "none"
    assert result[2]["display"] == "block"
    assert len(result) == 1 + WIZARD_STEPS + 3


def test_gate_enables_next_after_both_consent_ticks() -> None:
    gate = _register_wizard_callbacks()["gate_mobile_consent_step"]
    class_name, hint, allowed = gate(
        ["ack"],
        ["gdpr"],
        0,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        "en",
    )
    assert allowed is True
    assert class_name == "ui blue button"
    assert hint.get("visibility") == "hidden"


def _walk(node: Any) -> Any:
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


def test_samsung_ua_startup_next_is_a_real_button_over_http() -> None:
    """Samsung Internet UA must get the mobile wizard with a tappable Next."""
    _register_all_callbacks()
    client = app.server.test_client()
    body = {
        "output": _DISPLAY_OUTPUT,
        "outputs": [
            {"id": "page-content", "property": "children"},
            {"id": "mobile-warning", "property": "children"},
            {"id": "navbar-container", "property": "children"},
            {"id": "final-fill-step", "property": "data"},
        ],
        "inputs": [
            {"id": "url", "property": "pathname", "value": "/startup"},
            {"id": "game-stores-hydrated", "property": "data", "value": False},
        ],
        "state": [
            {"id": "interface-language", "property": "data", "value": "en"},
            {"id": "user-info-store", "property": "data", "value": None},
            {"id": "current-window-df", "property": "data", "value": None},
            {"id": "events-df", "property": "data", "value": None},
            {"id": "glucose-unit", "property": "data", "value": "mg/dL"},
            {"id": "user-agent", "property": "data", "value": SAMSUNG_UA},
        ],
        "changedPropIds": ["url.pathname"],
    }
    response = client.post(
        "/_dash-update-component",
        data=json.dumps(body),
        content_type="application/json",
        headers={"User-Agent": SAMSUNG_UA},
    )
    assert response.status_code == 200, response.get_data(as_text=True)[:400]
    page = response.get_json()["response"]["page-content"]["children"]
    next_btn = next(
        (
            node
            for node in _walk(page)
            if isinstance(node, dict) and node.get("props", {}).get("id") == "startup-next"
        ),
        None,
    )
    assert next_btn is not None
    props = next_btn["props"]
    assert props.get("disabled") not in (True, "true")
    assert "startup-next-disabled" in (props.get("className") or "")

    allowed_store = next(
        (
            node
            for node in _walk(page)
            if isinstance(node, dict)
            and node.get("props", {}).get("id") == "startup-next-allowed"
        ),
        None,
    )
    assert allowed_store is not None
    assert allowed_store["props"].get("data") is False


def test_gate_keeps_visual_block_until_consent() -> None:
    gate = _register_wizard_callbacks()["gate_mobile_consent_step"]
    class_name, hint, allowed = gate(
        ["ack"],
        [],
        0,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        "en",
    )
    assert allowed is False
    assert "startup-next-disabled" in class_name
    assert hint.get("visibility") == "visible"
