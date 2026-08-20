"""The optional nickname field on `/startup`, desktop and mobile.

Guards the trap documented in CLAUDE.md that bit twice: a Dash callback only fires
when *every* one of its Input/State components is in the current layout, so any id
`handle_start_button` reads must exist on BOTH startup builders. A miss there leaves
the Start button activating but navigating nowhere.
"""

from __future__ import annotations

from typing import Any

import pytest

from sugar_sugar.components.startup import (
    WIZARD_STEPS,
    StartupPage,
    StartupPageMobile,
)
from sugar_sugar.i18n import setup_i18n
from sugar_sugar.nickname import MAX_NICKNAME_LENGTH

# Every component id `handle_start_button` takes as State (sugar_sugar/app.py).
START_BUTTON_STATE_IDS: frozenset[str] = frozenset(
    {
        'nickname-input',
        'email-input',
        'age-input',
        'gender-dropdown',
        'cgm-dropdown',
        'cgm-duration-input',
        'cgm-duration-unit',
        'format-dropdown',
        'data-usage-consent',
        'diabetic-dropdown',
        'diabetic-type-dropdown',
        'diabetes-duration-input',
        'challenge-unknown-check',
        'paper-mention-check',
        'paper-full-name-input',
        'location-input',
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


def test_nickname_input_exists_on_both_builders(startup_page: Any) -> None:
    assert 'nickname-input' in _ids(startup_page)


def test_every_start_button_state_id_is_present(startup_page: Any) -> None:
    missing = START_BUTTON_STATE_IDS - _ids(startup_page)
    assert not missing, f"handle_start_button would never fire; missing: {sorted(missing)}"


def test_challenge_unknown_is_a_checkbox_on_both_builders(startup_page: Any) -> None:
    field = _find(startup_page, "challenge-unknown-check")
    assert field is not None
    assert field.id == "challenge-unknown-check"
    assert any(option.get("value") == "on" for option in field.options)
    assert _find(startup_page, "challenge-unknown-slider") is None


def test_paper_mention_fields_exist_on_both_builders(startup_page: Any) -> None:
    check = _find(startup_page, "paper-mention-check")
    name = _find(startup_page, "paper-full-name-input")
    wrap = _find(startup_page, "paper-full-name-wrap")
    assert check is not None
    assert any(option.get("value") == "on" for option in check.options)
    assert name is not None
    assert wrap is not None
    assert (wrap.style or {}).get("display") == "none"


def test_nickname_is_length_capped_and_persistent(startup_page: Any) -> None:
    field = _find(startup_page, 'nickname-input')
    assert field.maxLength == MAX_NICKNAME_LENGTH
    # Survives the layout rebuild on language change.
    assert field.persistence is True


def test_nickname_has_no_required_asterisk(startup_page: Any) -> None:
    """It is optional, so it must not grow a managed `*-required` span."""
    assert 'nickname-required' not in _ids(startup_page)


def test_mobile_nickname_is_in_the_identity_step_not_the_consent_step() -> None:
    """mobile-step-0 is consent; the nickname is not study data and must not sit there."""
    page = StartupPageMobile(locale="en")
    consent_step = _find(page, 'mobile-step-0')
    identity_step = _find(page, 'mobile-step-1')
    assert consent_step is not None and identity_step is not None
    assert 'nickname-input' not in _ids(consent_step)
    assert 'nickname-input' in _ids(identity_step)
    assert 'paper-mention-check' not in _ids(consent_step)
    assert 'paper-mention-check' in _ids(identity_step)
    assert 'paper-full-name-input' in _ids(identity_step)


def test_wizard_step_count_is_unchanged() -> None:
    """The nickname joined an existing step rather than adding one."""
    page = StartupPageMobile(locale="en")
    ids = _ids(page)
    assert WIZARD_STEPS == 6
    assert all(f'mobile-step-{i}' in ids for i in range(WIZARD_STEPS))
    assert f'mobile-step-{WIZARD_STEPS}' not in ids


def test_challenge_unknown_wrap_starts_hidden(startup_page: Any) -> None:
    """Eligibility is decided by a callback, so the default paint must hide it.

    A `display: block` default flashed "Challenge the unknown" at type 2 /
    prediabetes / LADA players before `update_challenge_unknown_visibility`
    resolved -- the exact groups the control is not offered to.
    """
    wrap = _find(startup_page, "challenge-unknown-wrap")
    assert wrap is not None
    assert (wrap.style or {}).get("display") == "none"
