"""Localized location autocomplete data for the startup form."""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from sugar_sugar.i18n import normalize_locale
from sugar_sugar.location_catalog import (
    CITY_SPECS,
    COUNTRY_NAMES,
    LOCALES,
    CitySpec,
    country_labels,
)

MIN_QUERY_LEN: int = 2
MAX_SUGGESTIONS: int = 8


@dataclass(frozen=True)
class PlaceEntry:
    """One autocomplete row with localized labels and search aliases."""

    canonical: str
    labels: dict[str, str]
    search: tuple[str, ...]
    rank: int = 1000


def _ascii_fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("ascii")


def _search_tokens(*texts: str) -> tuple[str, ...]:
    tokens: set[str] = set()
    for text in texts:
        cleaned = text.strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        tokens.add(lowered)
        folded = _ascii_fold(lowered)
        if folded and folded != lowered:
            tokens.add(folded)
    return tuple(sorted(tokens))


def _country_entry(en_name: str) -> PlaceEntry:
    labels = country_labels(en_name)
    return PlaceEntry(
        canonical=en_name,
        labels=labels,
        search=_search_tokens(*labels.values()),
    )


def _city_entry(spec: CitySpec) -> PlaceEntry:
    country = country_labels(spec.country)
    labels: dict[str, str] = {}
    for locale in LOCALES:
        city_name = spec.city.get(locale, spec.city["en"])
        labels[locale] = f"{city_name}, {country[locale]}"
    canonical = labels["en"]
    search_texts = list(labels.values()) + list(spec.extra_search)
    search_texts.extend(spec.city.values())
    return PlaceEntry(
        canonical=canonical,
        labels=labels,
        search=_search_tokens(*search_texts),
        rank=spec.rank,
    )


@lru_cache(maxsize=1)
def place_entries() -> tuple[PlaceEntry, ...]:
    entries: list[PlaceEntry] = [_country_entry(name) for name in COUNTRY_NAMES]
    entries.extend(_city_entry(spec) for spec in CITY_SPECS)
    return tuple(entries)


def place_label(entry: PlaceEntry, locale: str | None) -> str:
    loc = normalize_locale(locale)
    return entry.labels.get(loc) or entry.labels["en"]


def filter_location_suggestions(
    query: str,
    *,
    locale: str | None = None,
    limit: int = MAX_SUGGESTIONS,
) -> list[str]:
    """Return localized labels matching ``query`` (prefix matches first)."""
    q = query.strip().lower()
    if len(q) < MIN_QUERY_LEN:
        return []

    q_fold = _ascii_fold(q)
    loc = normalize_locale(locale)

    def _matches(token: str) -> bool:
        if token.startswith(q):
            return True
        if q_fold and token.startswith(q_fold):
            return True
        return q in token or (bool(q_fold) and q_fold in token)

    prefix_hits: list[PlaceEntry] = []
    contains_hits: list[PlaceEntry] = []
    for entry in place_entries():
        matched_prefix = False
        matched_contains = False
        for token in entry.search:
            if token.startswith(q) or (q_fold and token.startswith(q_fold)):
                matched_prefix = True
                break
            if q in token or (q_fold and q_fold in token):
                matched_contains = True
        if matched_prefix:
            prefix_hits.append(entry)
        elif matched_contains:
            contains_hits.append(entry)

    sort_key = lambda entry: (entry.rank, entry.canonical)
    prefix_hits.sort(key=sort_key)
    contains_hits.sort(key=sort_key)

    seen: set[str] = set()
    results: list[str] = []
    for entry in prefix_hits + contains_hits:
        label = place_label(entry, loc)
        if label in seen:
            continue
        seen.add(label)
        results.append(label)
        if len(results) >= limit:
            break
    return results


def export_suggestions_asset() -> list[dict[str, Any]]:
    return [
        {
            "canonical": entry.canonical,
            "labels": entry.labels,
            "search": list(entry.search),
            "rank": entry.rank,
        }
        for entry in place_entries()
    ]


def write_suggestions_asset(path: Path | None = None) -> Path:
    """Write the bundled autocomplete list to ``assets/location-suggestions.json``."""
    target = path or Path(__file__).resolve().parents[1] / "assets" / "location-suggestions.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(export_suggestions_asset(), ensure_ascii=False, indent=0),
        encoding="utf-8",
    )
    return target


# Backwards-compatible alias used in tests and older imports.
PLACES: tuple[str, ...] = tuple(entry.canonical for entry in place_entries())


if __name__ == "__main__":
    written = write_suggestions_asset()
    print(f"Wrote {len(place_entries())} places to {written}")
