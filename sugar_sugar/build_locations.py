"""Split the location-suggestion catalog into small per-locale asset files.

The autocomplete used to ship one 824 KB `assets/location-suggestions.json`
carrying every label in all supported UI languages plus every folded search token.
The browser fetched and parsed it eagerly on **every** page, long before the
location field on wizard step 3 existed -- a pure tax on low-spec phones.

This build step emits one compact file per locale instead.  Each row is
``[label, rank]`` or ``[label, rank, extra_tokens]``; the runtime derives the
lowercase and ASCII-folded search tokens from *label* itself, so only tokens it
cannot derive -- other Latin-script spellings, aliases like "nyc" -- are stored.
Files land in ``assets/`` because the browser fetches them by URL.

`sugar_sugar/location_catalog.py` remains the single source of truth; no
generated corpus is kept on disk.  Run ``uv run build-locations`` after editing
the catalog, and `tests/test_location_suggestions.py` fails if the shipped
assets drift from it.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any, Optional

import typer
from eliot import start_action
from pycomfort.logging import to_nice_stdout

from sugar_sugar.location_catalog import LOCALES
from sugar_sugar.location_suggestions import PlaceEntry, place_entries, place_label

DEFAULT_OUT_DIR: Path = Path(__file__).resolve().parents[1] / "assets"

app = typer.Typer(add_completion=False)


def ascii_fold(text: str) -> str:
    """Lowercase and strip combining marks -- mirrors `asciiFold` in the JS."""
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn").lower()


def derivable_tokens(label: str) -> set[str]:
    """The tokens the runtime builds from *label* without being told."""
    return {label.lower(), ascii_fold(label)}


def extra_tokens(entry: PlaceEntry, locale: str) -> list[str]:
    """Search tokens for *locale* that the label alone does not yield.

    What earns its bytes is the *bare* alternate spelling -- "munchen" next to
    "Munich", "nyc", "peking" -- because that is what a user actually types.
    What does not is another locale's full "City, Country" label: the catalog
    carries eight of them per row, they are the bulk of the corpus, and the
    label's own prefix already matches the city half.  So rival labels are
    dropped and everything else in `search` is kept, whatever its script.
    """
    label = place_label(entry, locale)
    known = derivable_tokens(label)
    rival_labels: set[str] = set()
    for other in LOCALES:
        if other == locale:
            continue
        rival_labels |= derivable_tokens(entry.labels[other])

    return [
        token
        for token in entry.search
        if token and token not in known and token not in rival_labels
    ]


def build_rows(locale: str) -> list[list[Any]]:
    """The shipped payload for one locale: `[label, rank]` or `[label, rank, extras]`."""
    rows: list[list[Any]] = []
    for entry in place_entries():
        label = place_label(entry, locale)
        if not label:
            continue
        extras = extra_tokens(entry, locale)
        rows.append([label, entry.rank, extras] if extras else [label, entry.rank])
    return rows


def locale_asset_path(locale: str, out_dir: Path = DEFAULT_OUT_DIR) -> Path:
    return out_dir / f"location-suggestions.{locale}.json"


def serialize_rows(rows: list[list[Any]]) -> str:
    return json.dumps(rows, ensure_ascii=False, separators=(",", ":"))


@app.command()
def main(
    out_dir: Path = typer.Option(DEFAULT_OUT_DIR, "--out-dir", help="Where the per-locale files go."),
    locales: Optional[str] = typer.Option(None, "--locales", help="Comma-separated subset to build."),
) -> None:
    """Emit `location-suggestions.<locale>.json` for every supported locale."""
    to_nice_stdout()
    wanted = tuple(locales.split(",")) if locales else tuple(LOCALES)
    with start_action(action_type="build_location_suggestions") as action:
        action.log(message_type="catalog_loaded", places=len(place_entries()))
        out_dir.mkdir(parents=True, exist_ok=True)
        for locale in wanted:
            rows = build_rows(locale)
            target = locale_asset_path(locale, out_dir)
            target.write_text(serialize_rows(rows), encoding="utf-8")
            action.log(
                message_type="locale_written",
                locale=locale,
                rows=len(rows),
                bytes=target.stat().st_size,
            )


def cli_main() -> None:
    app()


if __name__ == "__main__":
    cli_main()
