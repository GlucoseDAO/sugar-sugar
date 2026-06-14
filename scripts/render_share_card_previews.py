"""Render share-card previews for every supported locale.

Usage:
    uv run python scripts/render_share_card_previews.py

The script writes deterministic PNG fixtures under
``data/output/share-card-previews`` so social-card layout changes can be
checked visually across translations before deployment.
"""
from __future__ import annotations

import copy
import base64
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import typer
import kaleido
import plotly.graph_objects as go

from sugar_sugar.i18n import SUPPORTED_LOCALES, normalize_locale, setup_i18n
from sugar_sugar.share_png import render_share_card_png_bytes

app = typer.Typer(add_completion=False)


def _image_data_uri(path: Path) -> str:
    """Return a PNG file as a data URI for contact-sheet rendering."""
    encoded: str = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _round_record(round_number: int, fmt: str, *, noise_scale: float) -> dict[str, Any]:
    """Build one deterministic prediction round shaped like stored share data."""
    n_points: int = 36
    visible_points: int = 24
    actual_row: dict[str, str] = {"metric": "Actual Glucose"}
    pred_row: dict[str, str] = {"metric": "Predicted"}
    abs_err_row: dict[str, str] = {"metric": "Absolute Error"}
    rel_err_row: dict[str, str] = {"metric": "Relative Error (%)"}
    for idx in range(n_points):
        actual: float = 105.0 + 18.0 * math.sin(idx / 4.2) + 7.0 * math.cos(idx / 2.7)
        actual += float((round_number % 4) * 2)
        actual_row[f"t{idx}"] = f"{actual:.1f}"
        if idx < visible_points:
            pred_row[f"t{idx}"] = "-"
            abs_err_row[f"t{idx}"] = "-"
            rel_err_row[f"t{idx}"] = "-"
            continue
        step: float = (idx - visible_points) / float(max(n_points - visible_points - 1, 1))
        bias: float = math.sin(round_number * 1.7 + idx * 0.9) * noise_scale * step
        pred: float = actual * (1.0 + bias)
        err: float = abs(actual - pred)
        pred_row[f"t{idx}"] = f"{pred:.1f}"
        abs_err_row[f"t{idx}"] = f"{err:.1f}"
        rel_err_row[f"t{idx}"] = f"{(err / actual * 100.0):.1f}%"
    return {
        "round_number": round_number,
        "prediction_window_start": round_number * 12,
        "prediction_window_size": n_points,
        "prediction_table_data": [actual_row, pred_row, abs_err_row, rel_err_row],
        "format": fmt,
        "is_example_data": fmt == "A",
        "data_source_name": "preview.csv",
    }


def _base_share_record(*, case: str) -> dict[str, Any]:
    """Build a deterministic single-format or A/B/C stress-test share record."""
    if case == "multi":
        formats: list[str] = ["A", "B", "C"]
        rankings: dict[str, Any] = {
            "overall": {"rank": 3, "total": 57},
            "per_format": [
                {"format": "A", "rank": 2, "total": 20},
                {"format": "B", "rank": 5, "total": 18},
                {"format": "C", "rank": 3, "total": 19},
            ],
        }
    else:
        formats = ["A"]
        rankings = {
            "overall": {"rank": 1, "total": 27},
            "per_format": [{"format": "A", "rank": 1, "total": 27}],
        }
    rounds: list[dict[str, Any]] = [
        _round_record(
            idx + 1,
            formats[idx % len(formats)],
            noise_scale=0.06 + (idx % 5) * 0.035,
        )
        for idx in range(12)
    ]
    return {
        "schema_version": 2,
        "created_at": datetime(2026, 6, 14, 12, 0, tzinfo=timezone.utc).isoformat(),
        "locale": "en",
        "rounds": rounds,
        "played_formats": list(dict.fromkeys(formats)),
        "rankings": rankings,
        "user_info": {
            "name": "Dev Tester",
            "study_id": "preview",
            "format": formats[0],
            "uses_cgm": True,
            "max_rounds": 12,
        },
    }


def _write_contact_sheet(output_dir: Path, *, case: str, locales: list[str]) -> Path:
    """Write a 2-column overview sheet for quick manual inspection."""
    cols: int = 2
    rows: int = max(1, math.ceil(len(locales) / cols))
    fig = go.Figure()
    fig.update_xaxes(visible=False, range=[0, cols])
    fig.update_yaxes(visible=False, range=[0, rows])
    fig.update_layout(
        width=2400,
        height=630 * rows,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    for idx, loc in enumerate(locales):
        col: int = idx % cols
        row: int = rows - 1 - idx // cols
        preview_path: Path = output_dir / f"share-card-{loc}-{case}.png"
        fig.add_layout_image(
            dict(
                source=_image_data_uri(preview_path),
                xref="x",
                yref="y",
                x=col,
                y=row + 1,
                sizex=1,
                sizey=1,
                xanchor="left",
                yanchor="top",
                sizing="contain",
                layer="above",
            )
        )
        fig.add_annotation(
            x=col + 0.02,
            y=row + 0.98,
            xref="x",
            yref="y",
            text=f"<b>{loc}-{case}</b>",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font=dict(size=26, color="#0f172a"),
            bgcolor="rgba(255,255,255,0.75)",
        )
    path: Path = output_dir / f"contact-sheet-{case}.png"
    fig.write_image(str(path), width=2400, height=630 * rows, scale=1)
    return path


@app.command()
def render(
    output_dir: Path = typer.Option(
        Path("data/output/share-card-previews"),
        "--output-dir",
        "-o",
        help="Directory where preview PNGs are written.",
    ),
    share_url: str = typer.Option(
        "https://sugar-sugar.study/share/preview",
        "--share-url",
        help="Public URL encoded into the QR code.",
    ),
    locale: Optional[str] = typer.Option(
        None,
        "--locale",
        "-l",
        help="Render one locale only; defaults to all supported locales.",
    ),
    contact_sheets: bool = typer.Option(
        True,
        "--contact-sheets/--no-contact-sheets",
        help="Also render contact-sheet-single.png and contact-sheet-multi.png.",
    ),
) -> None:
    """Render single-format and multi-format cards for each locale."""
    setup_i18n()
    output_dir.mkdir(parents=True, exist_ok=True)
    locales: list[str] = (
        [normalize_locale(locale)]
        if locale
        else sorted(SUPPORTED_LOCALES)
    )
    cases: dict[str, dict[str, Any]] = {
        "single": _base_share_record(case="single"),
        "multi": _base_share_record(case="multi"),
    }
    written: list[Path] = []
    kaleido.start_sync_server(silence_warnings=True)
    try:
        for loc in locales:
            for case_name, record in cases.items():
                record_for_locale: dict[str, Any] = copy.deepcopy(record)
                record_for_locale["locale"] = loc
                png: bytes = render_share_card_png_bytes(
                    record_for_locale,
                    share_url=share_url,
                    locale=loc,
                    seed=f"{loc}-{case_name}",
                    allow_fallback=False,
                )
                path: Path = output_dir / f"share-card-{loc}-{case_name}.png"
                path.write_bytes(png)
                written.append(path)
    finally:
        kaleido.stop_sync_server(silence_warnings=True)
    if contact_sheets:
        for case_name in cases:
            written.append(_write_contact_sheet(output_dir, case=case_name, locales=locales))
    for path in written:
        typer.echo(str(path))


if __name__ == "__main__":
    app()
