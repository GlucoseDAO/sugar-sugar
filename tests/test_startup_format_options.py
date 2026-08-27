from __future__ import annotations

from typing import Any, Optional

import pytest

from sugar_sugar.components.startup import (
    StartupPage,
    StartupPageMobile,
    _compute_format_options,
)
from sugar_sugar.i18n import SUPPORTED_LOCALES, setup_i18n, t


@pytest.fixture(scope="module", autouse=True)
def _load_translations() -> None:
    setup_i18n()


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


def labels(options: list[dict]) -> list[str]:
    return [opt["label"] for opt in options]


def values(options: list[dict]) -> list[str]:
    return [opt["value"] for opt in options]


def disabled_flags(options: list[dict]) -> list[bool]:
    return [opt.get("disabled", False) for opt in options]


@pytest.mark.parametrize(
    "uses_cgm, current_format, expected_values, expected_start_value, expected_disabled",
    [
        # no format yet: leave the placeholder, do not infer A/C from CGM
        (False, None, ["A", "B", "C"], None, [False, True, True]),
        (True, None, ["A", "B", "C"], None, [False, False, False]),
        # current format preserved if valid
        (True, "B", ["A", "B", "C"], "B", [False, False, False]),
        (True, "A", ["A", "B", "C"], "A", [False, False, False]),
        # ineligible B/C is cleared back to the placeholder, not replaced with A
        (False, "B", ["A", "B", "C"], None, [False, True, True]),
        (False, "C", ["A", "B", "C"], None, [False, True, True]),
    ],
)
def test_compute_format_options(
    uses_cgm: bool,
    current_format: Optional[str],
    expected_values: list[str],
    expected_start_value: Optional[str],
    expected_disabled: list[bool],
) -> None:
    options, start = _compute_format_options(uses_cgm, "en", current_format)
    assert values(options) == expected_values
    assert disabled_flags(options) == expected_disabled
    assert start == expected_start_value


@pytest.mark.parametrize("builder", [StartupPage, StartupPageMobile])
@pytest.mark.parametrize("locale", list(SUPPORTED_LOCALES))
def test_format_dropdown_starts_on_placeholder(builder: type, locale: str) -> None:
    page = builder(locale=locale)
    field = _find(page, "format-dropdown")
    assert field is not None
    props = field.to_plotly_json().get("props", {})
    assert props.get("value") is None
    assert props.get("placeholder") == t("ui.startup.format_placeholder", locale=locale)
    assert props.get("placeholder")


def test_english_format_placeholder_reads_select_an_option() -> None:
    assert t("ui.startup.format_placeholder", locale="en") == "Select an option"
