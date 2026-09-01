"""Every locale carries every key, and a missing one degrades to English.

Two real bugs motivate this file, and neither failed a test at the time:

* The whole ``ui.startup.import_*`` block existed only in ``en`` and ``ro``, so
  six locales had no translation for the CGM import screen at all.
* `i18n.set` drops the fallback when it equals the current locale, and
  `setup_i18n` set both to "en". With the fallback off, a missing key renders as
  **the key itself** -- users saw `ui.startup.import_ns_button` as a button
  label. That reads as a broken page rather than as untranslated text.

Parity is the primary guard; the fallback is the safety net under it.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest
import yaml

from sugar_sugar.i18n import DEFAULT_LOCALE, SUPPORTED_LOCALES, setup_i18n, t

TRANSLATIONS_DIR = Path(__file__).parent.parent / "sugar_sugar" / "translations"
OTHER_LOCALES = sorted(SUPPORTED_LOCALES - {DEFAULT_LOCALE})
PLACEHOLDER = re.compile(r"%\{(\w+)\}")


@pytest.fixture(scope="session", autouse=True)
def _load_translations() -> None:
    setup_i18n()


# `true:` / `false:` in YAML parse as Python booleans, not strings -- the CGM
# yes/no labels are stored that way -- so those paths cannot be rebuilt as
# dotted i18n keys. They still take part in the parity check; they are only
# excluded from the render check.
_UNADDRESSABLE = ".True", ".False"


def _flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            flat.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            flat.update(_flatten(value, f"{prefix}[{index}]"))
    else:
        flat[prefix] = node
    return flat


def _is_addressable(key: str) -> bool:
    """Can this flattened path be handed to `t()` as a dotted key?"""
    return "[" not in key and not key.endswith(_UNADDRESSABLE)


@lru_cache(maxsize=None)
def _locale_keys(locale: str) -> dict[str, Any]:
    """Flattened key -> value for one locale file. Cached: parsed once per run."""
    raw = yaml.safe_load((TRANSLATIONS_DIR / f"ui.{locale}.yml").read_text(encoding="utf-8"))
    return _flatten(raw.get(locale, {}))


@pytest.fixture(scope="session")
def english() -> dict[str, Any]:
    return _locale_keys(DEFAULT_LOCALE)


@pytest.mark.parametrize("locale", OTHER_LOCALES)
def test_locale_has_every_english_key(locale: str, english: dict[str, Any]) -> None:
    missing = sorted(key for key in english if key not in _locale_keys(locale))
    assert not missing, f"ui.{locale}.yml is missing {len(missing)} keys, e.g. {missing[:5]}"


@pytest.mark.parametrize("locale", OTHER_LOCALES)
def test_placeholders_survive_translation(locale: str, english: dict[str, Any]) -> None:
    """A dropped %{count} renders as literal text; a renamed one raises."""
    translated = _locale_keys(locale)
    mismatched = {
        key: translated[key]
        for key, value in english.items()
        if isinstance(value, str)
        and isinstance(translated.get(key), str)
        and sorted(PLACEHOLDER.findall(value)) != sorted(PLACEHOLDER.findall(translated[key]))
    }
    assert not mismatched, f"ui.{locale}.yml placeholder mismatch: {list(mismatched)[:5]}"


# i18nice treats `%` as interpolation. A lone `10%` raises; a leftover `%%` after
# render shows two percent signs on the share card / ranking. Literal percent in
# YAML must be `%%`, and `t()` must emit a single `%`.
_LONE_PERCENT = re.compile(r"(?<!%)%(?!%|\{)")


@pytest.mark.parametrize("locale", sorted(SUPPORTED_LOCALES))
def test_literal_percent_is_escaped_and_renders_once(locale: str, english: dict[str, Any]) -> None:
    translated = _locale_keys(locale)
    lone: list[str] = []
    leftover: list[str] = []
    for key, en_value in english.items():
        if not isinstance(en_value, str) or "%%" not in en_value:
            continue
        loc_value = translated.get(key)
        assert isinstance(loc_value, str), f"{locale}: {key} missing while English has %%"
        assert loc_value.count("%%") == en_value.count("%%"), f"{locale}: {key} %% count drifted"
        if _LONE_PERCENT.search(loc_value):
            lone.append(key)
        kwargs = {name: 0 for name in PLACEHOLDER.findall(en_value)}
        rendered = t(f"ui.{key}", locale=locale, **kwargs)
        if "%%" in rendered:
            leftover.append(key)
        assert "%" in rendered, f"{locale}: {key} rendered without a percent sign"
    assert not lone, f"{locale}: lone % (not %% / %{{}}) in {lone[:5]}"
    assert not leftover, f"{locale}: t() left %% in {leftover[:5]}"


def test_fallback_is_actually_enabled() -> None:
    """Guards the `i18n.set` trap: it nulls the fallback when it equals locale."""
    import i18n

    assert i18n.get("fallback") == DEFAULT_LOCALE


@pytest.mark.parametrize("locale", sorted(SUPPORTED_LOCALES))
def test_no_key_renders_as_its_own_name(locale: str, english: dict[str, Any]) -> None:
    """The end-to-end symptom: `t()` returning 'ui.startup.import_ns_button'.

    Covers whichever of the two causes bites -- a key missing everywhere, or the
    fallback being off -- because both produce the same thing on screen.
    """
    leaked: list[str] = []
    for key, value in english.items():
        if not isinstance(value, str) or not _is_addressable(key):
            continue
        kwargs = {name: 0 for name in PLACEHOLDER.findall(value)}
        if t(f"ui.{key}", locale=locale, **kwargs).startswith("ui."):
            leaked.append(key)
    assert not leaked, f"{locale}: {len(leaked)} keys render as key strings, e.g. {leaked[:5]}"
