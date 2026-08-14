"""Play-form ids on `/startup` that `handle_start_button` reads as State.

Identity fields (nickname/email/age/gender/location) now live on `/profile`.
A Dash callback only fires when every State id is in the current layout, so
those identity ids must NOT appear on either startup builder.
"""

from __future__ import annotations

from typing import Any

import pytest

from sugar_sugar.components.profile import PROFILE_FIELD_IDS, create_profile_layout
from sugar_sugar.components.startup import (
    WIZARD_STEPS,
    StartupPage,
    StartupPageMobile,
)
from sugar_sugar.i18n import setup_i18n
from sugar_sugar.nickname import MAX_NICKNAME_LENGTH

START_BUTTON_STATE_IDS: frozenset[str] = frozenset(
    {
        'cgm-dropdown',
        'cgm-duration-input',
        'format-dropdown',
        'data-usage-consent',
        'diabetic-dropdown',
        'diabetic-type-dropdown',
        'diabetes-duration-input',
    }
)

IDENTITY_FIELD_IDS: frozenset[str] = frozenset(
    {
        'nickname-input',
        'email-input',
        'age-input',
        'gender-dropdown',
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


def test_every_start_button_state_id_is_present(startup_page: Any) -> None:
    missing = START_BUTTON_STATE_IDS - _ids(startup_page)
    assert not missing, f"handle_start_button would never fire; missing: {sorted(missing)}"


def test_identity_fields_are_not_on_startup(startup_page: Any) -> None:
    overlap = IDENTITY_FIELD_IDS & _ids(startup_page)
    assert not overlap, f"identity fields leaked onto startup: {sorted(overlap)}"


def test_mobile_wizard_has_five_steps() -> None:
    page = StartupPageMobile(locale="en")
    ids = _ids(page)
    assert WIZARD_STEPS == 5
    assert all(f'mobile-step-{i}' in ids for i in range(WIZARD_STEPS))
    assert f'mobile-step-{WIZARD_STEPS}' not in ids


def test_mobile_step_1_is_diabetes_not_identity() -> None:
    page = StartupPageMobile(locale="en")
    consent_step = _find(page, 'mobile-step-0')
    diabetes_step = _find(page, 'mobile-step-1')
    assert consent_step is not None and diabetes_step is not None
    assert 'nickname-input' not in _ids(consent_step)
    assert 'diabetic-dropdown' in _ids(diabetes_step)


def test_profile_has_identity_fields() -> None:
    page = create_profile_layout(
        {"uses_cgm": True, "format": "A", "rounds": [{"round_number": 1}]},
        locale="en",
    )
    ids = _ids(page)
    assert IDENTITY_FIELD_IDS <= ids
    assert PROFILE_FIELD_IDS <= ids
    nickname = _find(page, 'nickname-input')
    assert nickname.maxLength == MAX_NICKNAME_LENGTH
    assert nickname.persistence is True
