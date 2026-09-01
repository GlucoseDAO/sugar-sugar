import base64
import gzip
import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Sequence

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

def safe_upload_filename(name: Optional[str]) -> str:
    """Make an uploaded file name safe to write, keeping what it says it is.

    A Nightscout export is JSON, so ``.json`` is preserved; everything else is
    given ``.csv``. Both upload handlers used to force ``.csv`` onto every
    upload, which stored ``entries.json`` as ``entries.json.csv``.
    """
    safe = (name or "uploaded").replace(" ", "_").replace("/", "_")
    if not safe.lower().endswith((".csv", ".json")):
        safe += ".csv"
    return safe


def decode_upload_files(payload: object, filename: object) -> list[tuple[str, bytes]]:
    """Decode a possibly-multi-file upload into ``(filename, bytes)`` pairs.

    Dash hands ``contents`` and ``filename`` as a bare value when the Upload is
    single-file and as a list when it is ``multiple=True``, and the clientside
    compressor preserves whichever shape it was given. Both are normalised here
    so the two upload callbacks do not each grow the same branch.

    A member that will not decode is dropped rather than failing the upload:
    one unreadable file out of three should not cost the user the other two.
    The caller sees an empty list only when nothing at all decoded.
    """
    payloads = payload if isinstance(payload, list) else [payload]
    names = filename if isinstance(filename, list) else [filename]

    decoded: list[tuple[str, bytes]] = []
    for index, item in enumerate(payloads):
        if not isinstance(item, str):
            continue
        data = decode_upload_bytes(item)
        if data is None:
            continue
        name = names[index] if index < len(names) else None
        decoded.append((str(name or f"uploaded_{index}"), data))
    return decoded


_NIGHTSCOUT_ERROR_MAX: int = 180
_HTML_MARKERS: tuple[str, ...] = ("<html", "<!doctype", "<body", "<head")


def _looks_like_html(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in _HTML_MARKERS)


def _ascii_one_line(text: str) -> str:
    """Drop newlines and non-ASCII so a leftover 🟢 cannot reach a locale log."""
    cleaned = text.replace("\n", " ").replace("\r", " ").strip()
    cleaned = cleaned.encode("ascii", errors="replace").decode("ascii")
    if len(cleaned) > _NIGHTSCOUT_ERROR_MAX:
        cleaned = cleaned[:_NIGHTSCOUT_ERROR_MAX] + "…"
    return cleaned


def short_nightscout_error(exc: BaseException) -> str:
    """One-line Nightscout failure, never an HTML body or an emoji dump.

    ``httpx`` embeds the response text in ``str(HTTPStatusError)``. Eliot then
    writes that into the log; on Windows cp1252 a 🟢 in the page kills the
    import with ``UnicodeEncodeError``. Keep the logged/UI reason short.
    """
    if isinstance(exc, RuntimeError) and exc.__cause__ is not None:
        return short_nightscout_error(exc.__cause__)
    try:
        import httpx
    except ImportError:
        httpx = None  # type: ignore[assignment]
    if httpx is not None and isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code if exc.response is not None else "?"
        return f"HTTP {code}"
    if httpx is not None and isinstance(exc, httpx.TimeoutException):
        return "timeout"
    if httpx is not None and isinstance(
        exc, (httpx.ConnectError, httpx.NetworkError)
    ):
        return "unreachable"
    if isinstance(exc, UnicodeEncodeError):
        return "encoding"
    name = type(exc).__name__
    text = str(exc)
    if _looks_like_html(text):
        return f"{name}: html error page"
    text = _ascii_one_line(text)
    return f"{name}: {text}" if text else name


@contextmanager
def utf8_path_writes() -> Iterator[None]:
    """Force ``Path.write_text`` to UTF-8 when the caller omitted encoding.

    cgm-format's Nightscout downloader does ``json.dumps(..., ensure_ascii=False)``
    then ``Path.write_text(payload)``. On Windows that open uses cp1252, and a
    🟢 in a treatment note raises ``UnicodeEncodeError`` after a successful
    download. Parsing itself uses ``read_bytes``, so only the write needs this.
    """
    original = Path.write_text

    def write_text(
        self: Path,
        data: str,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
    ) -> int:
        return original(self, data, encoding=encoding or "utf-8", errors=errors, newline=newline)

    Path.write_text = write_text  # type: ignore[method-assign]
    try:
        yield
    finally:
        Path.write_text = original


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
    with start_action(action_type=u"load_glucose_data_from_nightscout", base_url=base_url) as action:
        try:
            with utf8_path_writes():
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
        except Exception as exc:
            short = short_nightscout_error(exc)
            action.log(message_type="nightscout_import_failed", error=short)
            raise RuntimeError(short) from exc


# How much of a file to look at before deciding it is JSON at all. No CSV export
# we support opens with a bare array, so the first non-blank byte settles it.
_JSON_SNIFF_BYTES: int = 8192


ENTRIES: str = "entries"
TREATMENTS: str = "treatments"
PROFILE: str = "profile"


def nightscout_json_kind(data: bytes) -> Optional[str]:
    """Name which of Nightscout's three JSON exports *data* is, if any.

    A Nightscout export is three sibling files and users hand over whichever
    subset they happened to download, so the three have to be told apart by
    content: they all deserialise to a bare array of objects and the file names
    are whatever the browser saved them as.

    * ``entries``    -- glucose. The only one that can carry a trace, so the only
                       one that is required.
    * ``treatments`` -- boluses, temp basals, carbs. Optional detail; it becomes
                       the event markers on the chart.
    * ``profile``    -- basal schedules, targets, display unit. Contributes no
                       rows to a unified frame at all; cgm-format downloads it
                       and discards it, and so do we.

    Returns ``None`` for anything else, which is what keeps this from claiming a
    file the library should handle.
    """
    if not data[:_JSON_SNIFF_BYTES].lstrip().startswith(b"["):
        return None
    try:
        records = json.loads(data.decode("utf-8", errors="replace"))
    except ValueError:
        return None
    if not isinstance(records, list):
        return None

    objects = [record for record in records if isinstance(record, dict)]
    if not objects:
        return None

    # Scan every record rather than a prefix: an entries export may open with
    # calibration or meter rows before the first sensor reading, and a profile
    # export is short enough that a prefix proves nothing either way.
    if any("sgv" in record or record.get("type") == "sgv" for record in objects):
        return ENTRIES
    if any("eventType" in record for record in objects):
        return TREATMENTS
    if any("defaultProfile" in record or "store" in record for record in objects):
        return PROFILE
    return None


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
    return nightscout_json_kind(file_path.read_bytes()) == ENTRIES


@dataclass(frozen=True)
class NightscoutUpload:
    """One uploaded Nightscout export, sorted into its role."""

    filename: str
    data: bytes


@dataclass(frozen=True)
class NightscoutBundle:
    """What a multi-file Nightscout upload turned out to contain.

    ``entries`` is the only member that matters for playability; without it
    there is no glucose and the upload cannot be used. ``treatments`` is
    optional detail. ``discarded`` holds the profile export, which users
    reasonably include because it is part of the same download, plus anything
    else that came along -- naming them lets the UI say what it ignored instead
    of silently dropping files the user chose.
    """

    entries: Optional[NightscoutUpload]
    treatments: Optional[NightscoutUpload]
    discarded: tuple[str, ...]

    @property
    def is_usable(self) -> bool:
        return self.entries is not None


def classify_nightscout_uploads(files: Sequence[tuple[str, bytes]]) -> Optional[NightscoutBundle]:
    """Sort uploaded files into the Nightscout bundle roles.

    Returns ``None`` when nothing in *files* is a Nightscout export, which is
    how the caller tells a Nightscout upload apart from an ordinary CGM CSV and
    falls back to the single-file path.

    Duplicates keep the first of each kind. A user who selects an export twice
    (or picks up a re-download) should not have the second copy silently replace
    the first, and there is no principled way to choose between them.
    """
    entries: Optional[NightscoutUpload] = None
    treatments: Optional[NightscoutUpload] = None
    discarded: list[str] = []
    saw_nightscout = False

    for filename, data in files:
        kind = nightscout_json_kind(data)
        if kind is not None:
            saw_nightscout = True
        if kind == ENTRIES and entries is None:
            entries = NightscoutUpload(filename, data)
        elif kind == TREATMENTS and treatments is None:
            treatments = NightscoutUpload(filename, data)
        else:
            discarded.append(filename)

    if not saw_nightscout:
        return None
    return NightscoutBundle(entries=entries, treatments=treatments, discarded=tuple(discarded))


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

    Passing treatments used to fail on cgm-format 0.12.0 whenever a treatment
    field was null for the first 100 records (``FEEDBACK.md`` issue 1); 0.12.2
    fixed it and is the floor. The upload hint still asks for the entries file
    only, because the upload is single-file -- not because treatments are broken.
    """
    with start_action(action_type=u"load_nightscout_json_data", file_path=str(file_path)):
        entries_data = file_path.read_bytes()
        treatments_data = treatments_path.read_bytes() if treatments_path is not None else None
        unified_df = FormatParser.parse_nightscout(entries_data, treatments_data)
        split_glucose, split_events = unified_processor(unified_df).split_glucose_events(unified_df)
        return adapt_glucose_df(split_glucose), adapt_events_df(split_events)


def load_nightscout_uploads(
    bundle: NightscoutBundle,
    save_dir: Path = Path("data/input/users"),
) -> tuple[pl.DataFrame, pl.DataFrame, Path]:
    """Parse an uploaded Nightscout bundle and persist it as a unified CSV.

    What gets saved is the **unified frame**, not the raw ``entries.json``, and
    that is the whole reason two-file upload works across rounds: later rounds
    reload from ``user_info['uploaded_data_path']`` through ``load_glucose_data``,
    so pointing it at the entries file would quietly drop the treatments after
    round one and the event markers would vanish mid-game. Saving the merge is
    what ``load_glucose_data_from_nightscout`` already does for the URL import;
    this is the same shape reached from an upload instead of a download.

    Raises ``ValueError`` if the bundle has no entries file -- callers check
    ``bundle.is_usable`` first and tell the user which file is missing.
    """
    if bundle.entries is None:
        raise ValueError("A Nightscout upload needs the entries export; it holds the glucose.")

    with start_action(
        action_type=u"load_nightscout_uploads",
        entries=bundle.entries.filename,
        treatments=bundle.treatments.filename if bundle.treatments else None,
        discarded=list(bundle.discarded),
    ):
        unified_df = FormatParser.parse_nightscout(
            bundle.entries.data,
            bundle.treatments.data if bundle.treatments is not None else None,
        )
        split_glucose, split_events = unified_processor(unified_df).split_glucose_events(unified_df)
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = save_dir / f"{timestamp}_nightscout.csv"
        FormatParser.to_csv_file(unified_df, str(save_path))
        return adapt_glucose_df(split_glucose), adapt_events_df(split_events), save_path


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
