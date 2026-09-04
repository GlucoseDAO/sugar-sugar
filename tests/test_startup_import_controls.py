"""The CGM import block on `/startup`: CSV upload + Nightscout URL.

This block had no test coverage at all, which is how a player could report that
"the Nightscout button disappeared" with nothing failing. It is the only
reachable Nightscout entry point in the app -- the prediction page renders the
same controls at `display:none` for every format -- so the ids, its placement in
the mobile wizard, and the gate that reveals it are all pinned here.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest

from sugar_sugar.components.startup import (
    WIZARD_STEPS,
    StartupPage,
    StartupPageMobile,
    import_controls_children,
)
from sugar_sugar.i18n import setup_i18n

# Everything `handle_startup_nightscout_import` / `handle_startup_csv_upload`
# reference. A Dash callback only fires when every one of its components is in
# the layout, so a builder missing any of these has a dead import section.
IMPORT_CONTROL_IDS: frozenset[str] = frozenset(
    {
        'startup-upload-data',
        'startup-ns-url',
        'startup-ns-token',
        'startup-ns-import',
        'startup-import-status',
    }
)


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


@pytest.fixture(params=["desktop", "mobile"])
def startup_page(request: pytest.FixtureRequest) -> Any:
    return StartupPage(locale="en") if request.param == "desktop" else StartupPageMobile(locale="en")


def test_import_controls_exist_on_both_builders(startup_page: Any) -> None:
    missing = IMPORT_CONTROL_IDS - _ids(startup_page)
    assert not missing, f"import section is dead; missing: {sorted(missing)}"


def _ordered_ids(node: Any) -> list[str]:
    if isinstance(node, (list, tuple)):
        found: list[str] = []
        for child in node:
            found.extend(_ordered_ids(child))
        return found
    found = []
    node_id = getattr(node, "id", None)
    if isinstance(node_id, str):
        found.append(node_id)
    children = getattr(node, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            found.extend(_ordered_ids(child))
    elif children is not None and not isinstance(children, str):
        found.extend(_ordered_ids(children))
    return found


def test_nightscout_url_comes_before_csv_upload() -> None:
    ids = _ordered_ids(import_controls_children("en"))
    assert ids.index("startup-ns-url") < ids.index("startup-upload-data")
    assert ids.index("startup-ns-import") < ids.index("startup-upload-data")


def test_import_subtitle_prefers_nightscout_url() -> None:
    children = import_controls_children("en")
    subtitle = next(
        node
        for node in children
        if getattr(node, "id", None) == "startup-import-subtitle"
    )
    text = str(subtitle.children)
    assert "Nightscout site URL" in text
    assert "leaves some of that out" in text
    assert "entries.json" not in text


def test_upload_accepts_several_files(startup_page: Any) -> None:
    """A Nightscout export is entries + treatments (+ a profile file to discard)."""
    upload = _find(startup_page, 'startup-upload-data')

    assert upload.multiple is True
    assert '.json' in upload.accept


def test_import_block_lives_in_a_reachable_wizard_step() -> None:
    """The mobile wizard can only show steps 0..WIZARD_STEPS-1.

    If the import block ever lands past the last reachable index it is
    unreachable on mobile while still present in the layout, so every id check
    above would still pass.
    """
    page = StartupPageMobile(locale="en")

    holding_steps = [
        index
        for index in range(WIZARD_STEPS)
        if IMPORT_CONTROL_IDS & _ids(_find(page, f'mobile-step-{index}'))
    ]

    assert holding_steps, "no reachable mobile wizard step contains the import controls"
    assert all(index < WIZARD_STEPS for index in holding_steps)


def test_import_block_sits_inside_the_data_usage_gate(startup_page: Any) -> None:
    """Its visibility is the consent container's, which is why the gate matters."""
    container = _find(startup_page, 'data-usage-consent-container')

    assert container is not None
    assert IMPORT_CONTROL_IDS <= _ids(container)
    # Baked hidden: only `toggle_data_usage_consent` reveals it.
    assert container.style.get('display') == 'none'


def _toggle(format_value: Optional[str], *, initial: bool) -> tuple[Any, Any]:
    """Drive `toggle_data_usage_consent` the way Dash would."""
    import dash

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
    toggle = captured['toggle_data_usage_consent']

    class _Ctx:
        triggered_id = None if initial else 'format-dropdown'

    original = dash.callback_context
    dash.callback_context = _Ctx()  # type: ignore[assignment]
    try:
        return toggle(format_value, None, [])
    finally:
        dash.callback_context = original  # type: ignore[assignment]


@pytest.mark.parametrize("format_value", ["B", "C"])
def test_own_data_formats_reveal_the_import_block(format_value: str) -> None:
    style, _consent = _toggle(format_value, initial=False)
    assert style['display'] == 'block'


def test_public_data_format_hides_the_import_block() -> None:
    style, _consent = _toggle("A", initial=False)
    assert style['display'] == 'none'


def test_a_cleared_format_hides_the_block_instead_of_stranding_it() -> None:
    """Answering "no CGM" later clears a My Data / Mixed choice to None.

    This used to return `no_update`, which left the consent box and the
    Nightscout import on screen for a format the user no longer had -- and once
    the cleared value persisted, the block never came back on a later render.
    """
    style, consent = _toggle(None, initial=False)

    assert style['display'] == 'none'
    assert consent == []


def test_hydration_does_not_clear_a_persisted_consent_tick() -> None:
    """`data-usage-consent` is persisted, so the initial call must not write []."""
    from dash import no_update

    style, consent = _toggle(None, initial=True)

    assert style is no_update
    assert consent is no_update
