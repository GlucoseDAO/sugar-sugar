"""Download Format A public corpora (BIG IDEAs + D1NAMO).

``uv run start`` works without these: Format A falls back to ``data/example.csv``.
``uv run download`` fetches the real study mix. Per-dataset commands remain.
CGMacros is unused in Format A — pass ``--all`` to include it.
"""

from __future__ import annotations

import typer
from eliot import start_action

from sugar_sugar.download_bigideas import default_dest as bigideas_dest
from sugar_sugar.download_bigideas import fetch_bigideas
from sugar_sugar.download_cgmacros import default_dest as cgmacros_dest
from sugar_sugar.download_cgmacros import fetch_cgmacros
from sugar_sugar.download_d1namo import default_dest as d1namo_dest
from sugar_sugar.download_d1namo import fetch_d1namo


def fetch_public_datasets(
    *,
    force: bool = False,
    include_photos: bool = True,
    keep_zip: bool = False,
    include_cgmacros: bool = False,
) -> None:
    """Fetch the Format A corpora, optionally plus CGMacros."""
    with start_action(
        action_type=u"download_public_datasets",
        force=force,
        include_photos=include_photos,
        include_cgmacros=include_cgmacros,
    ) as action:
        fetch_bigideas(bigideas_dest(), force=force)
        fetch_d1namo(
            d1namo_dest(),
            include_photos=include_photos,
            force=force,
            keep_zip=keep_zip,
        )
        if include_cgmacros:
            fetch_cgmacros(
                cgmacros_dest(),
                include_photos=False,
                force=force,
                keep_zip=keep_zip,
            )
        action.add_success_fields(ok=True)


def download(
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-download even if a dataset is already present.",
    ),
    no_photos: bool = typer.Option(
        False,
        "--no-photos",
        help="Skip D1NAMO meal JPEGs and keep only glucose/insulin/food CSVs.",
    ),
    keep_zip: bool = typer.Option(
        False,
        "--keep-zip",
        help="Keep downloaded zip archives after extracting.",
    ),
    include_all: bool = typer.Option(
        False,
        "--all",
        help="Also download CGMacros (unused in Format A, ~627 MB).",
    ),
) -> None:
    """Download Format A public datasets (BIG IDEAs + D1NAMO)."""
    fetch_public_datasets(
        force=force,
        include_photos=not no_photos,
        keep_zip=keep_zip,
        include_cgmacros=include_all,
    )


def main() -> None:
    typer.run(download)
