"""Translate BIG IDEAs food-note lines at display time.

Source logs stay English. Known phrases, food words, and units are replaced
for the active locale; brand names and anything missing stay as written.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

from sugar_sugar.food_glossary import (
    PHRASES,
    UNIT_ALIASES,
    UNITS,
    WORD_ALIASES,
    WORDS,
)
from sugar_sugar.i18n import normalize_locale

_LINE_RE = re.compile(r"^(?P<name>.+?)(?:\s+\((?P<extra>[^)]+)\))?$")
_AMOUNT_UNIT_RE = re.compile(r"^(?P<amount>\d+(?:\.\d+)?)\s+(?P<unit>.+)$")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*")
_PHRASE_KEYS: tuple[str, ...] = tuple(sorted(PHRASES, key=len, reverse=True))


def _match_case(original: str, translated: str) -> str:
    if not translated:
        return translated
    if original.isupper():
        return translated.upper()
    if original[:1].isupper():
        return translated[:1].upper() + translated[1:]
    return translated


@lru_cache(maxsize=8)
def _phrase_patterns(locale: str) -> tuple[tuple[re.Pattern[str], str], ...]:
    compiled: list[tuple[re.Pattern[str], str]] = []
    for phrase in _PHRASE_KEYS:
        label = PHRASES[phrase].get(locale)
        if not label:
            continue
        compiled.append((re.compile(rf"(?i)\b{re.escape(phrase)}\b"), label))
    return tuple(compiled)


def translate_food_name(name: str, locale: str) -> str:
    text = name.strip()
    if not text:
        return name
    exact = PHRASES.get(text.lower()) or WORDS.get(text.lower())
    if exact and locale in exact:
        return _match_case(text, exact[locale])
    result = text
    for pattern, label in _phrase_patterns(locale):
        result = pattern.sub(lambda match, mapped=label: _match_case(match.group(0), mapped), result)

    def _word(match: re.Match[str]) -> str:
        raw = match.group(0)
        key = WORD_ALIASES.get(raw.lower(), raw.lower())
        mapped = WORDS.get(key, {}).get(locale)
        return _match_case(raw, mapped) if mapped else raw

    return _WORD_RE.sub(_word, result)


def translate_unit(unit: str, locale: str) -> str:
    raw = unit.strip().lower()
    canonical = UNIT_ALIASES.get(raw, raw)
    mapped = UNITS.get(canonical) or UNITS.get(raw)
    if mapped and locale in mapped:
        return mapped[locale]
    return unit


def _translate_extra(extra: str, locale: str) -> str:
    text = extra.strip()
    match = _AMOUNT_UNIT_RE.match(text)
    if match:
        return f"{match.group('amount')} {translate_unit(match.group('unit'), locale)}"
    return translate_unit(text, locale)


def translate_food_note(note: str, locale: Optional[str] = None) -> str:
    """Translate one notepad blob (one sitting, newline-separated items)."""
    loc = normalize_locale(locale)
    if loc == "en" or not note.strip():
        return note
    lines: list[str] = []
    for raw_line in note.splitlines():
        line = raw_line.strip()
        if not line:
            lines.append(raw_line)
            continue
        parsed = _LINE_RE.match(line)
        if parsed is None:
            lines.append(line)
            continue
        name = translate_food_name(parsed.group("name"), loc)
        extra = parsed.group("extra")
        if extra:
            lines.append(f"{name} ({_translate_extra(extra, loc)})")
        else:
            lines.append(name)
    return "\n".join(lines)
