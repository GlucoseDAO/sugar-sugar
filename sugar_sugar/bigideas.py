"""Import BIG IDEAs Dexcom + food-log bundles into the app store schema.

Parsing is ``cgm-format``'s job since 0.11: :func:`cgm_format.FormatParser.parse_subject_directory`
reads the subject bundle (``Dexcom_NNN.csv`` + ``Food_Log_NNN.csv``), which means it
absorbs the three on-disk food-log layouts the published sixteen subjects mix
(canonical 14-column header, the ``time_of_day`` alias four of them use, and the
headerless 11-column log subject ``003`` ships), the ``date`` + ``time`` fallback for
the one row with a blank ``time_begin``, and the Clarity metadata-row drift every
subject carries. BIG IDEAs is identified by *directory shape*, not by sniffing the
Dexcom header -- the glucose file genuinely is a Clarity export and would otherwise
parse as a plain vendor upload with no meals.

What stays here is what the library does not do:

* ``Demographics.csv`` -- per-subject attributes, which have no home in a frame keyed
  by timestamp, so the library leaves them on disk.
* Folding the per-item food rows into one marker per sitting. The library emits one
  event per logged item ("clustering items into a sitting is a consumer concern"),
  and a player looking at the chart wants one apple icon for a meal, not four.
* The ``food_note`` text itself. BIG IDEAs has no meal photographs, so the apple icon
  opens a notepad; the note is built from the food names, amounts and units the
  library carries in the JSON ``annotations`` column.

Download a local copy with ``uv run download-bigideas``.
Paper: Bent et al., npj Digital Medicine, 2021.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterable, Optional

import polars as pl
from cgm_format import FormatParser, SupportedCGMFormat, UnifiedEventType
from cgm_format.interface.cgm_interface import UnknownFormatError
from eliot import start_action

from sugar_sugar.corpus import (
    FOOD_NOTE_EVENTS_SCHEMA,
    adapt_events_df,
    adapt_glucose_df,
    annotation_field,
    legacy_event_subtype_expr,
    legacy_event_type_expr,
    unified_processor,
)
from sugar_sugar.download_bigideas import dataset_is_present, default_dest

_SUBJECT_DIR_RE = re.compile(r"^(\d{3})$")
_SOURCE_NAME_RE = re.compile(r"^BIGIDEAS-(\d{3})\.csv$", re.IGNORECASE)
_DEXCOM_NAME_RE = re.compile(r"^dexcom_(\d{3})\.csv$", re.IGNORECASE)

#: Two food rows this far apart or closer are one sitting. The food log records
#: every item separately -- a smoothie and a chicken leg logged at 18:00 are one
#: dinner, not two carbohydrate events an hour apart.
_FOOD_CLUSTER_GAP = timedelta(minutes=30)

#: Amounts and units the food log spells out as words meaning "nothing".
_EMPTY_TOKENS: tuple[str, ...] = ("none", "nan")

_CARBS_EVENT: str = UnifiedEventType.CARBOHYDRATES.value


@dataclass(frozen=True, slots=True)
class BigIdeasSource:
    subject_id: str
    source_name: str
    csv_path: Path
    subject_dir: Path
    age_years: int | None
    gender: str
    weight: str
    sensor: str
    hba1c: str
    intervention: str = "bigideas"


def dataset_root(dest: Optional[Path] = None) -> Path:
    return dest if dest is not None else default_dest()


def is_bigideas_source_name(source_name: str) -> bool:
    return bool(_SOURCE_NAME_RE.match(Path(str(source_name or "")).name))


def subject_format(subject_dir: Path) -> SupportedCGMFormat | None:
    """``BIGIDEAS`` when the library recognises *subject_dir*, else ``None``.

    The probe is conjunctive on the library's side -- a subject directory holds
    a Dexcom export *and* a food log -- so a folder of renamed Clarity exports
    does not answer here, which is exactly the discrimination the app needs.
    """
    try:
        detected = FormatParser.detect_subject_format(subject_dir)
    except (UnknownFormatError, OSError, ValueError):
        return None
    return detected if detected == SupportedCGMFormat.BIGIDEAS else None


def is_bigideas_path(file_path: Path) -> bool:
    """True when *file_path* is a BIG IDEAs Dexcom table or virtual source name.

    The virtual ``BIGIDEAS-NNN.csv`` name never exists on disk and is answered by
    name alone. A real ``Dexcom_NNN.csv`` is confirmed through the library's
    subject-shape probe rather than its own header: the file *is* a Clarity
    export, so the header cannot tell the two apart -- only the food log sitting
    beside it can.
    """
    path = Path(file_path)
    if is_bigideas_source_name(path.name):
        return True
    if _DEXCOM_NAME_RE.match(path.name) and subject_format(path.parent) is not None:
        return True
    # Not a shape the library claims, but unmistakably from the extract tree --
    # let it through so the caller fails with a real parse error, not a silent
    # fallthrough to the generic vendor parser.
    return "bigideas" in str(path).replace("\\", "/").lower() and path.suffix.lower() == ".csv"


def subject_id_from_path(path: Path) -> str | None:
    for token in (path.name, path.stem, path.parent.name):
        source_match = _SOURCE_NAME_RE.match(token) or _DEXCOM_NAME_RE.match(token)
        if source_match:
            return source_match.group(1)
        dir_match = _SUBJECT_DIR_RE.match(token)
        if dir_match:
            return dir_match.group(1)
    return None


def _norm_header(name: str) -> str:
    return " ".join(name.replace("\ufeff", "").strip().lower().split())


def _column(df: pl.DataFrame, aliases: Iterable[str]) -> str | None:
    by_norm = {_norm_header(name): name for name in df.columns}
    for alias in aliases:
        found = by_norm.get(_norm_header(alias))
        if found is not None:
            return found
    return None


def _normalize_gender(raw: str) -> str:
    lowered = str(raw or "").strip().lower()
    if lowered == "female":
        return "female"
    if lowered == "male":
        return "male"
    return ""


def load_bigideas_demographics(dest: Optional[Path] = None) -> dict[str, tuple[str, str]]:
    """Map subject id ``001`` -> (gender, hba1c).

    ``Demographics.csv`` sits at the corpus root and is per-subject attributes,
    not a time series, so ``cgm-format`` has nowhere to put it and the app reads
    it directly.
    """
    path = dataset_root(dest) / "Demographics.csv"
    if not path.is_file():
        return {}
    raw = pl.read_csv(path, infer_schema_length=200)
    id_col = _column(raw, ("id", "subject", "participant"))
    gender_col = _column(raw, ("gender", "sex"))
    hba1c_col = _column(raw, ("hba1c", "hba1c "))
    if id_col is None:
        return {}
    out: dict[str, tuple[str, str]] = {}
    for row in raw.iter_rows(named=True):
        raw_id = str(row.get(id_col) or "").strip()
        if not raw_id:
            continue
        digits = re.search(r"(\d+)", raw_id)
        if digits is None:
            continue
        subject_id = f"{int(digits.group(1)):03d}"
        gender = _normalize_gender(str(row.get(gender_col) or "") if gender_col else "")
        hba1c = str(row.get(hba1c_col) or "").strip() if hba1c_col else ""
        out[subject_id] = (gender, hba1c)
    return out


def _annotation_text(*paths: str) -> pl.Expr:
    """An annotation field with the food log's word-shaped blanks removed."""
    text = annotation_field(*paths).str.strip_chars()
    return pl.when(text.str.to_lowercase().is_in(list(_EMPTY_TOKENS))).then(pl.lit("")).otherwise(text)


def _food_line_expr() -> pl.Expr:
    """One logged item as a line of the note: ``Berry Smoothie (20.0 fluid ounce)``.

    The participant's own wording wins over the database match the log found for
    it; the amount and unit are appended only when the source recorded them, so a
    bare ``Chicken Leg (1.0)`` and a bare ``Asparagus`` both read naturally.
    """
    logged = _annotation_text("logged_food")
    name = pl.when(logged != "").then(logged).otherwise(_annotation_text("searched_food"))
    amount = _annotation_text("amount")
    unit = _annotation_text("unit")
    extras = (
        pl.when((amount != "") & (unit != ""))
        .then(amount + pl.lit(" ") + unit)
        .when(amount != "")
        .then(amount)
        .otherwise(unit)
    )
    return (
        pl.when(name == "")
        .then(pl.lit(""))
        .when(extras != "")
        .then(name + pl.lit(" (") + extras + pl.lit(")"))
        .otherwise(name)
    )


def cluster_food_events(events_df: pl.DataFrame) -> pl.DataFrame:
    """Unified carbohydrate rows -> one app event per sitting.

    Items logged within :data:`_FOOD_CLUSTER_GAP` of the previous one join it,
    chained -- a meal eaten over an hour in five-minute steps stays one marker.
    Carbohydrates are summed across the sitting and stay null when the source
    reported none at all, which is a different statement from zero grams.

    Rows the source placed in time but never named are kept for their
    carbohydrates and contribute no line; a sitting with no named item at all is
    dropped, because an apple icon opening an empty notepad tells a player
    nothing.
    """
    carbs = events_df.filter(
        pl.col("datetime").is_not_null() & (pl.col("event_type") == _CARBS_EVENT)
    ).sort("datetime")
    if carbs.height == 0:
        return pl.DataFrame(schema=FOOD_NOTE_EVENTS_SCHEMA)

    named = carbs.with_columns(
        _food_line_expr().alias("line"),
        legacy_event_type_expr().alias("event_type_legacy"),
        legacy_event_subtype_expr().alias("event_subtype_legacy"),
    ).with_columns(
        (pl.col("datetime").diff().fill_null(timedelta(0)) > _FOOD_CLUSTER_GAP)
        .cum_sum()
        .alias("sitting")
    )

    lines = pl.col("line").filter(pl.col("line") != "")
    grouped = (
        named.group_by("sitting", maintain_order=True)
        .agg(
            pl.col("datetime").first().alias("time"),
            pl.col("event_type_legacy").first().alias("event_type"),
            pl.col("event_subtype_legacy").first().alias("event_subtype"),
            pl.lit(None, dtype=pl.Float64).alias("insulin_value"),
            pl.lit("").alias("photo_path"),
            lines.first().alias("meal_type"),
            pl.when(pl.col("carbs").is_not_null().any())
            .then(pl.col("carbs").sum())
            .otherwise(None)
            .alias("carbs_g"),
            lines.str.join("\n").alias("food_note"),
        )
        .filter(pl.col("food_note") != "")
        .drop("sitting")
        .sort("time")
    )
    return grouped.cast(FOOD_NOTE_EVENTS_SCHEMA)  # type: ignore[arg-type]


def adapt_bigideas_unified(
    unified_df: pl.DataFrame,
    *,
    subject_id: int = 1,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split a BIG IDEAs unified frame and adapt both halves to the app's stores.

    Meals take the clustering path above; anything else the chart can draw --
    insulin or exercise a participant typed into the Dexcom app -- goes through
    the shared adapter and simply carries no note.
    """
    processor = unified_processor(unified_df)
    split_glucose, split_events = processor.split_glucose_events(unified_df)
    meals = cluster_food_events(split_events)
    other = adapt_events_df(split_events.filter(pl.col("event_type") != _CARBS_EVENT))
    if other.height > 0:
        other = other.with_columns(
            pl.lit("").alias("photo_path"),
            pl.lit("").alias("meal_type"),
            pl.lit(None, dtype=pl.Float64).alias("carbs_g"),
            pl.lit("").alias("food_note"),
        ).cast(FOOD_NOTE_EVENTS_SCHEMA)  # type: ignore[arg-type]
        meals = pl.concat([meals, other.select(meals.columns)], how="vertical").sort("time")
    return adapt_glucose_df(split_glucose, subject_id=subject_id), meals


def discover_bigideas_sources(dest: Optional[Path] = None) -> list[BigIdeasSource]:
    """Discover BIG IDEAs Dexcom tables under ``data/bigideas/`` (or *dest*)."""
    root = dataset_root(dest)
    if not dataset_is_present(root) and not root.is_dir():
        return []

    demographics = load_bigideas_demographics(root)
    sources: list[BigIdeasSource] = []
    seen: set[str] = set()
    for dexcom_path in sorted(root.rglob("Dexcom_*.csv")):
        subject_id = subject_id_from_path(dexcom_path)
        if subject_id is None or subject_id in seen:
            continue
        # Only offer subjects the library can actually parse: a round that picks
        # an unparseable source fails in front of a player.
        if subject_format(dexcom_path.parent) is None:
            continue
        seen.add(subject_id)
        gender, hba1c = demographics.get(subject_id, ("", ""))
        sources.append(
            BigIdeasSource(
                subject_id=subject_id,
                source_name=f"BIGIDEAS-{subject_id}.csv",
                csv_path=dexcom_path,
                subject_dir=dexcom_path.parent,
                age_years=None,
                gender=gender,
                weight="",
                sensor="Dexcom G6",
                hba1c=hba1c,
            )
        )
    return sources


def load_bigideas_data(file_path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load one BIG IDEAs participant into the app store schema."""
    path = Path(file_path)
    with start_action(action_type=u"load_bigideas_data", file_path=str(path)) as action:
        dexcom_path = path
        if not dexcom_path.is_file() and is_bigideas_source_name(path.name):
            subject_id = subject_id_from_path(path)
            for source in discover_bigideas_sources():
                if source.subject_id == subject_id:
                    dexcom_path = source.csv_path
                    break
        if not dexcom_path.is_file():
            raise FileNotFoundError(f"BIG IDEAs Dexcom CSV not found for {path}")

        subject_dir = dexcom_path.parent
        subject_token = subject_id_from_path(dexcom_path) or "001"
        # The subject is a *bundle*: the Clarity export and the food log are two
        # files that parse into one frame, so the library takes the directory.
        unified_df = FormatParser.parse_subject_directory(subject_dir)
        glucose_df, events_df = adapt_bigideas_unified(
            unified_df,
            subject_id=int(subject_token),
        )
        action.add_success_fields(
            subject_id=subject_token,
            glucose_rows=glucose_df.height,
            event_rows=events_df.height,
            food_notes=events_df.filter(pl.col("food_note") != "").height,
        )
        return glucose_df, events_df
