"""Rewrite legacy ``cgm_duration_years`` integers to ``value,unit`` tuples.

Bare numbers are treated as years (the old form asked for years only).
Already-migrated ``3,years`` cells are left as-is.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import typer
from eliot import start_action

from sugar_sugar.cgm_duration import migrate_cgm_duration_cell

COLUMN_NAME: str = "cgm_duration_years"
DEFAULT_STATS_PATH: Path = Path("data/input/prediction_statistics.csv")

app = typer.Typer(add_completion=False, help="Migrate CGM duration CSV cells to value,unit tuples.")


def migrate_cgm_duration_csv(path: Path) -> int:
    """Rewrite ``cgm_duration_years`` in place. Returns the number of cells changed."""
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if COLUMN_NAME not in fieldnames:
        return 0

    changed = 0
    for row in rows:
        original = row.get(COLUMN_NAME, "")
        migrated = migrate_cgm_duration_cell(original)
        if migrated != str(original or ""):
            row[COLUMN_NAME] = migrated
            changed += 1

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)
    return changed


@app.callback(invoke_without_command=True)
def main(
    path: Optional[Path] = typer.Option(
        None,
        "--path",
        help="Statistics CSV to rewrite. Defaults to data/input/prediction_statistics.csv.",
    ),
) -> None:
    target = path or DEFAULT_STATS_PATH
    with start_action(action_type="migrate_cgm_duration", path=str(target)) as action:
        changed = migrate_cgm_duration_csv(target)
        action.log(message_type="migrate_cgm_duration_done", changed=changed)
        typer.echo(f"Updated {changed} CGM duration cell(s) in {target}")


def cli_main() -> None:
    app()
