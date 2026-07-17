from __future__ import annotations

import pytest

from sugar_sugar.components.startup import MAX_AGE, validate_startup_form
from sugar_sugar.i18n import setup_i18n


@pytest.fixture(scope="module", autouse=True)
def _load_translations() -> None:
    setup_i18n()


def _base_kwargs(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "email": None,
        "age": 30,
        "gender": "F",
        "format_value": "A",
        "data_usage_consent": None,
        "is_diabetic": True,
        "diabetic_type": "Type 1",
        "diabetes_duration": 10,
        "location": "Berlin",
        "uses_cgm": True,
        "cgm_duration": 5,
        "wants_contact": False,
        "locale": "en",
    }
    values.update(overrides)
    return values


def test_age_allows_up_to_max() -> None:
    result = validate_startup_form(**_base_kwargs(age=MAX_AGE))
    assert result.form_complete is True
    assert result.age_error == ""


def test_age_over_max_is_rejected() -> None:
    result = validate_startup_form(**_base_kwargs(age=MAX_AGE + 1))
    assert result.form_complete is False
    assert result.has_range_errors is True
    assert "130" in result.age_error


def test_diabetes_duration_cannot_exceed_age() -> None:
    result = validate_startup_form(**_base_kwargs(age=20, diabetes_duration=25))
    assert result.form_complete is False
    assert result.diabetes_duration_error
    assert result.has_range_errors is True


def test_cgm_duration_cannot_exceed_age() -> None:
    result = validate_startup_form(**_base_kwargs(age=15, cgm_duration=16, uses_cgm=True))
    assert result.form_complete is False
    assert result.cgm_duration_error
    assert result.has_range_errors is True


def test_cgm_duration_not_checked_when_no_cgm() -> None:
    result = validate_startup_form(**_base_kwargs(uses_cgm=False, cgm_duration=999))
    assert result.cgm_duration_error == ""


def test_missing_fields_message_lists_labels() -> None:
    result = validate_startup_form(
        **_base_kwargs(age=None, gender=None, location=None),
        wizard_step=1,
    )
    assert result.step_complete is False
    assert "ui.startup.age_label" in result.missing_label_keys
    assert "Age" in result.missing_fields_message


@pytest.mark.parametrize(
    "wizard_step, missing_key",
    [
        (2, "ui.startup.cgm_label"),
        (3, "ui.startup.diabetic_label"),
        (4, "ui.startup.format_label"),
    ],
)
def test_wizard_step_missing_fields(wizard_step: int, missing_key: str) -> None:
    kwargs = _base_kwargs()
    if wizard_step == 2:
        kwargs["uses_cgm"] = None
    elif wizard_step == 3:
        kwargs["is_diabetic"] = None
    elif wizard_step == 4:
        kwargs["format_value"] = None
    result = validate_startup_form(**kwargs, wizard_step=wizard_step)
    assert missing_key in result.missing_label_keys
