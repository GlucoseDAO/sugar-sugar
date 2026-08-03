from __future__ import annotations

from typing import Optional

import pytest

from sugar_sugar.components.startup import _compute_format_options


def labels(options: list[dict]) -> list[str]:
    return [opt["label"] for opt in options]


def values(options: list[dict]) -> list[str]:
    return [opt["value"] for opt in options]


def disabled_flags(options: list[dict]) -> list[bool]:
    return [opt.get("disabled", False) for opt in options]


@pytest.mark.parametrize(
    "uses_cgm, current_format, expected_values, expected_start_value, expected_disabled",
    [
        # when no CGM, only A is enabled and selected by default
        (False, None, ["A", "B", "C"], "A", [False, True, True]),
        # when CGM allowed default to C
        (True, None, ["A", "B", "C"], "C", [False, False, False]),
        # current format preserved if valid
        (True, "B", ["A", "B", "C"], "B", [False, False, False]),
        (False, "B", ["A", "B", "C"], "A", [False, True, True]),
        # a CGM owner who picked A keeps it: playing the generic data without
        # uploading must stay possible (C is only ever suggested, never forced)
        (True, "A", ["A", "B", "C"], "A", [False, False, False]),
        # drop back to A if eligibility lost from B/C
        (False, "C", ["A", "B", "C"], "A", [False, True, True]),
    ],
)
def test_compute_format_options(
    uses_cgm: bool,
    current_format: Optional[str],
    expected_values: list[str],
    expected_start_value: str,
    expected_disabled: list[bool],
) -> None:
    options, start = _compute_format_options(uses_cgm, "en", current_format)
    assert values(options) == expected_values
    assert disabled_flags(options) == expected_disabled
    assert start == expected_start_value
