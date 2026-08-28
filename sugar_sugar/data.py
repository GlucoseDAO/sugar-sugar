import base64
import gzip
import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import polars as pl
from cgm_format import FormatParser
from eliot import start_action

from sugar_sugar.bigideas import is_bigideas_path, load_bigideas_data
from sugar_sugar.cgmacros import is_cgmacros_csv, load_cgmacros_data
from sugar_sugar.corpus import adapt_events_df, adapt_glucose_df, unified_processor
from sugar_sugar.d1namo import is_d1namo_path, load_d1namo_data


def decode_upload_bytes(payload: Optional[str]) -> Optional[bytes]:
    """Decode an upload payload to raw file bytes.

    Uploads are gzip-compressed client-side (``"gzip:<base64>"``) so the ~3.3 MB
    base64 of a multi-MB CGM export never has to cross the wire from the phone --
    mobile browsers reliably fail to POST a payload that large, which is why a big
    Dexcom export uploaded fine on desktop but silently failed on mobile (server
    parsing was never the problem). Falls back to a plain data URL
    (``"<mime>,<base64>"``) for desktop/older browsers without CompressionStream.
    Returns None if the payload is unrecognisable.
    """
    if not payload:
        return None
    try:
        if payload.startswith("gzip:"):
            return gzip.decompress(base64.b64decode(payload[5:]))
        if "," in payload:
            return base64.b64decode(payload.split(",", 1)[1])
    except Exception:
        return None
    return None

def load_glucose_data_from_nightscout(
    base_url: str,
    *,
    token: Optional[str] = None,
    api_secret: Optional[str] = None,
    days: Optional[int] = None,
    save_dir: Path = Path("data/input/users"),
) -> tuple[pl.DataFrame, pl.DataFrame, Path]:
    """Fetch CGM data from a Nightscout server and adapt to the app store schema.

    Downloads entries and treatments via the Nightscout REST API, serialises the
    unified DataFrame to a timestamped CSV under *save_dir* (so subsequent rounds
    can reload it via ``load_glucose_data``), and returns the adapted frames.

    Returns:
        (glucose_df, events_df, save_path)
    """
    with start_action(action_type=u"load_glucose_data_from_nightscout", base_url=base_url):
        unified_df = FormatParser.from_nightscout_url(
            base_url, token=token, api_secret=api_secret, days=days
        )
        glucose_df, events_df = unified_processor(unified_df).split_glucose_events(unified_df)
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = save_dir / f"{timestamp}_nightscout.csv"
        FormatParser.to_csv_file(unified_df, str(save_path))
        return adapt_glucose_df(glucose_df), adapt_events_df(events_df), save_path


# How much of a file to look at before deciding it is JSON at all. No CSV export
# we support opens with a bare array, so the first non-blank byte settles it.
_JSON_SNIFF_BYTES: int = 8192


def is_nightscout_entries_json(file_path: Path) -> bool:
    """True if *file_path* is a Nightscout ``entries.json`` export.

    Nightscout's built-in CSV export is not a usable alternative and the upload
    hint deliberately does not offer it: ``/api/v1/entries.csv`` is headerless
    with five hardcoded columns, so cgm-format cannot even detect it, and there
    is no treatments CSV at all. The JSON served at ``/api/v1/entries.json`` is
    the one export that survives intact.

    This has to run ahead of the library for the same reason the LOOP and corpus
    detectors do: ``FormatParser.detect_format`` pattern-matches CSV headers and
    deliberately does not handle JSON, so a Nightscout export reaching it raises
    ``UnknownFormatError``.

    Detection is on content, never on the file name -- both upload handlers
    rewrite the extension of whatever they are given.
    """
    if not file_path.exists():
        return False
    if not file_path.read_bytes()[:_JSON_SNIFF_BYTES].lstrip().startswith(b"["):
        return False
    try:
        records = json.loads(file_path.read_text(encoding="utf-8", errors="replace"))
    except ValueError:
        return False
    # Scan the whole array rather than a prefix: an export may open with
    # calibration or meter records before the first sensor reading.
    return isinstance(records, list) and any(
        isinstance(record, dict) and ("sgv" in record or record.get("type") == "sgv")
        for record in records
    )


def load_nightscout_json_data(
    file_path: Path,
    treatments_path: Optional[Path] = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load a Nightscout entries JSON export into the app store schema.

    *treatments_path* is optional and is never inferred from a neighbouring
    file. Every upload lands in one shared ``data/input/users/`` directory under
    a timestamped name, so guessing at a sibling would eventually pair one
    player's entries with another player's treatments. The upload path passes
    entries alone and gets the glucose trace with no event markers; a caller
    holding both files may pass both.

    Note that passing treatments currently fails on cgm-format 0.12.0 whenever
    a treatment field is null for the first 100 records -- see ``FEEDBACK.md``
    issue 1. That is why the upload hint asks for the entries file only.
    """
    with start_action(action_type=u"load_nightscout_json_data", file_path=str(file_path)):
        entries_data = file_path.read_bytes()
        treatments_data = treatments_path.read_bytes() if treatments_path is not None else None
        unified_df = FormatParser.parse_nightscout(entries_data, treatments_data)
        split_glucose, split_events = unified_processor(unified_df).split_glucose_events(unified_df)
        return adapt_glucose_df(split_glucose), adapt_events_df(split_events)


def load_glucose_data(file_path: Path = Path("data/example.csv")) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load CGM data through cgm-format and adapt it to the app store schema.

    Five hand-rolled detectors run before the library: the three research corpora
    are directory-shaped rather than file-shaped (and BIG IDEAs' glucose half is
    an ordinary Clarity export, so only its directory tells it apart from a plain
    upload), LOOP has no library counterpart at all, and a Nightscout export is
    JSON, which ``detect_format`` deliberately does not handle. Everything else
    -- every vendor export -- goes straight through ``FormatParser``.

    Every branch returns the same pair of frames:
    ``(time, gl, prediction, age, user_id)`` and
    ``(time, event_type, event_subtype, insulin_value, ...)``.
    """
    with start_action(action_type=u"load_glucose_data", file_path=str(file_path)):
        if _is_loop_chronological_csv(file_path):
            glucose_df, events_df = load_loop_chronological_data(file_path)
        elif is_cgmacros_csv(file_path):
            glucose_df, events_df = load_cgmacros_data(file_path)
        elif is_d1namo_path(file_path):
            glucose_df, events_df = load_d1namo_data(file_path)
        elif is_bigideas_path(file_path):
            glucose_df, events_df = load_bigideas_data(file_path)
        elif is_nightscout_entries_json(file_path):
            glucose_df, events_df = load_nightscout_json_data(file_path)
        else:
            unified_df = FormatParser.parse_file(file_path)
            split_glucose, split_events = unified_processor(unified_df).split_glucose_events(
                unified_df
            )
            glucose_df = adapt_glucose_df(split_glucose)
            events_df = adapt_events_df(split_events)
        return glucose_df, events_df


def _is_loop_chronological_csv(file_path: Path) -> bool:
    name = file_path.name.lower()
    if name.endswith("_chronological.csv"):
        return True
    if not file_path.exists():
        return False
    header = file_path.read_text(encoding="utf-8", errors="replace").splitlines()[:1]
    if not header:
        return False
    return "Glucose (mg/dL)" in header[0] and "Recommended Split" in header[0]


def _non_empty_str(column: str) -> pl.Expr:
    as_text = pl.col(column).cast(pl.Utf8, strict=False).str.strip_chars()
    return pl.col(column).is_not_null() & (as_text != "")


def _parse_loop_numeric(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.Utf8, strict=False).str.strip_chars().cast(pl.Float64, strict=False)


def load_loop_chronological_data(file_path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load LOOP study chronological CSV exports into the app store schema."""
    with start_action(action_type=u"load_loop_chronological_data", file_path=str(file_path)):
        raw_df = pl.read_csv(file_path, infer_schema_length=10000)
        time_expr = pl.col("Timestamp").str.to_datetime(strict=False)

        glucose_df = (
            raw_df.filter(pl.col("Event Type").is_in(["EGV", "Interpolated"]))
            .filter(_non_empty_str("Glucose (mg/dL)"))
            .select(
                [
                    time_expr.alias("time"),
                    _parse_loop_numeric("Glucose (mg/dL)").alias("gl"),
                    pl.lit(0.0).alias("prediction"),
                    pl.lit(0).alias("age"),
                    pl.lit(1).alias("user_id"),
                ]
            )
            .filter(pl.col("time").is_not_null() & pl.col("gl").is_not_null())
            .sort("time")
        )

        carb_events = (
            raw_df.filter(_non_empty_str("Carbohydrates (g)"))
            .select(
                [
                    time_expr.alias("time"),
                    pl.lit("Carbohydrates").alias("event_type"),
                    pl.lit("Carbs").alias("event_subtype"),
                    pl.lit(None, dtype=pl.Float64).alias("insulin_value"),
                ]
            )
            .filter(pl.col("time").is_not_null())
        )
        bolus_events = (
            raw_df.filter(_non_empty_str("Bolus Insulin (U)"))
            .select(
                [
                    time_expr.alias("time"),
                    pl.lit("Insulin").alias("event_type"),
                    pl.lit("Fast Acting").alias("event_subtype"),
                    _parse_loop_numeric("Bolus Insulin (U)").alias("insulin_value"),
                ]
            )
            .filter(
                pl.col("time").is_not_null()
                & pl.col("insulin_value").is_not_null()
                & (pl.col("insulin_value") != 0)
            )
        )
        basal_events = (
            raw_df.filter(_non_empty_str("Basal Rate (U/h)"))
            .select(
                [
                    time_expr.alias("time"),
                    pl.lit("Insulin").alias("event_type"),
                    pl.lit("Long Acting").alias("event_subtype"),
                    _parse_loop_numeric("Basal Rate (U/h)").alias("insulin_value"),
                ]
            )
            .filter(
                pl.col("time").is_not_null()
                & pl.col("insulin_value").is_not_null()
                & (pl.col("insulin_value") != 0)
            )
        )

        events_df = pl.concat([carb_events, bolus_events, basal_events], how="vertical").sort("time")
        return glucose_df, events_df
