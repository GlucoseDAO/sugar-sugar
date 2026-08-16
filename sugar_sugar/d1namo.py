"""Import D1NAMO (Dubosson) T1D subjects into the app store schema.

Parsing is ``cgm-format``'s job: :func:`cgm_format.FormatParser.parse_subject_directory`
reads the subject bundle (``glucose.csv`` + ``insulin.csv`` + ``food.csv``), converts
mmol/L to mg/dL through the *declared* unit rather than a guess, handles the four
different timestamp conventions D1NAMO mixes inside one subject directory (including
the EXIF-style ``2014:10:01 12:15:00`` in ``food.csv``), and maps fingersticks to
``CALIBRAT`` so they never reach the glucose trace.

What stays here is what the library does not do: discovering subjects on disk,
resolving meal photographs to a servable path, and the Flask-facing URL helpers.
D1NAMO is the type-1 / insulin-using Format A arm; non-insulin arms use BIG IDEAs,
which ``bigideas.py`` loads through the same library entry point.

Download a local copy with ``uv run download-d1namo``.
Paper: Dubosson et al., Informatics in Medicine Unlocked, 2018.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import polars as pl
from cgm_format import FormatParser, SupportedCGMFormat
from cgm_format.interface.cgm_interface import UnknownFormatError
from eliot import start_action

from sugar_sugar.corpus import adapt_unified
from sugar_sugar.download_d1namo import dataset_is_present, default_dest

_PHOTO_SUFFIXES: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".heic")
_SUBJECT_DIR_RE = re.compile(r"^(\d{3})$")
_SOURCE_NAME_RE = re.compile(r"^D1NAMO-(\d{3})\.csv$", re.IGNORECASE)

#: The subject shapes the library recognises for this corpus. A directory that
#: matches neither is not a D1NAMO subject we can parse, and is skipped during
#: discovery rather than blowing up mid-round.
_D1NAMO_FORMATS: tuple[SupportedCGMFormat, ...] = (
    SupportedCGMFormat.D1NAMO_DIABETES,
    SupportedCGMFormat.D1NAMO_HEALTHY,
)


@dataclass(frozen=True, slots=True)
class D1NamoSource:
    subject_id: str
    source_name: str
    csv_path: Path
    subject_dir: Path
    age_years: int | None
    gender: str
    weight: str
    sensor: str
    intervention: str = "d1namo"


def dataset_root(dest: Optional[Path] = None) -> Path:
    return dest if dest is not None else default_dest()


def is_d1namo_source_name(source_name: str) -> bool:
    return bool(_SOURCE_NAME_RE.match(Path(str(source_name or "")).name))


def is_d1namo_path(file_path: Path) -> bool:
    """True when *file_path* is a D1NAMO glucose table or virtual source name.

    Kept as a cheap router predicate for ``load_glucose_data``: the virtual
    ``D1NAMO-NNN.csv`` name never exists on disk, so it is answered by name
    alone, and a real ``glucose.csv`` is confirmed through the library's own
    subject-shape probes rather than a path substring.
    """
    path = Path(file_path)
    if is_d1namo_source_name(path.name):
        return True
    if path.name.lower() != "glucose.csv":
        return False
    if subject_format(path.parent) is not None:
        return True
    # Not a shape the library claims, but unmistakably from the extract tree --
    # let it through so the caller fails with a real parse error, not a
    # silent fallthrough to the generic vendor parser.
    return "d1namo" in str(path).replace("\\", "/").lower()


def subject_id_from_path(path: Path) -> str | None:
    for token in (path.name, path.stem, path.parent.name):
        source_match = _SOURCE_NAME_RE.match(token)
        if source_match:
            return source_match.group(1)
        dir_match = _SUBJECT_DIR_RE.match(token)
        if dir_match:
            return dir_match.group(1)
    return None


def subject_format(subject_dir: Path) -> SupportedCGMFormat | None:
    """The D1NAMO subset *subject_dir* belongs to, or ``None`` if it is not one.

    The library discriminates the two subsets by which modality file is present
    (``insulin.csv`` for diabetes, ``annotations.csv`` for healthy), so a folder
    holding a bare ``glucose.csv`` is not a subject it can parse. Answering
    ``None`` here lets discovery skip such a folder instead of handing the round
    picker a source that raises when loaded.
    """
    try:
        detected = FormatParser.detect_subject_format(subject_dir)
    except (UnknownFormatError, OSError, ValueError):
        return None
    return detected if detected in _D1NAMO_FORMATS else None


def resolve_photo_path(image_path: str, subject_dir: Path) -> str:
    """Return a subject-relative posix path for a meal photograph.

    ``cgm-format`` reports the reference the source recorded (a bare filename)
    and warns when it cannot resolve it, but never rewrites it -- serving the
    file is the app's concern, and published subjects disagree about which
    folder holds the JPEGs.
    """
    raw = str(image_path or "").strip().replace("\\", "/")
    if not raw:
        return ""
    parts = [part for part in raw.split("/") if part and part != "."]
    if any(part == ".." for part in parts):
        return ""
    candidate = subject_dir.joinpath(*parts)
    if candidate.is_file():
        return "/".join(parts)
    filename = Path(raw).name
    for folder in ("pictures", "photos", "food", "food_pictures"):
        fallback = subject_dir / folder / filename
        if fallback.is_file():
            return f"{folder}/{filename}"
    return raw


def discover_d1namo_sources(dest: Optional[Path] = None) -> list[D1NamoSource]:
    """Discover D1NAMO subject folders under ``data/d1namo/`` (or *dest*)."""
    root = dataset_root(dest)
    if not dataset_is_present(root) and not root.is_dir():
        return []

    sources: list[D1NamoSource] = []
    seen: set[str] = set()
    for glucose_path in sorted(root.rglob("glucose.csv")):
        subject_id = subject_id_from_path(glucose_path)
        if subject_id is None or subject_id in seen:
            continue
        # Only offer subjects the library can actually parse: a round that
        # picks an unparseable source fails in front of a player.
        if subject_format(glucose_path.parent) is None:
            continue
        seen.add(subject_id)
        sources.append(
            D1NamoSource(
                subject_id=subject_id,
                source_name=f"D1NAMO-{subject_id}.csv",
                csv_path=glucose_path,
                subject_dir=glucose_path.parent,
                age_years=None,
                gender="",
                weight="",
                sensor="Medtronic iPro2",
            )
        )
    return sources


def d1namo_photo_url(source_name: str, photo_path: str) -> str:
    """Public URL for a meal photo belonging to ``D1NAMO-NNN.csv``."""
    match = _SOURCE_NAME_RE.match(Path(str(source_name or "")).name)
    subject = match.group(1) if match else Path(str(source_name or "")).stem
    rel = str(photo_path or "").replace("\\", "/").lstrip("/")
    return f"/d1namo/{subject}/photo/{rel}"


def resolve_served_photo(
    subject: str,
    rel_path: str,
    dest: Optional[Path] = None,
) -> Path | None:
    """Resolve a meal photo under the D1NAMO extract. Rejects path escape."""
    match = _SUBJECT_DIR_RE.match(str(subject or "").strip()) or _SOURCE_NAME_RE.match(
        str(subject or "").strip()
    )
    if match is None:
        digits = re.search(r"(\d{3})", str(subject or ""))
        if digits is None:
            return None
        subject_id = digits.group(1)
    else:
        subject_id = match.group(1)

    raw = str(rel_path or "").replace("\\", "/").strip()
    parts = [part for part in raw.split("/") if part and part != "."]
    if not parts or any(part == ".." for part in parts):
        return None
    if Path(parts[-1]).suffix.lower() not in _PHOTO_SUFFIXES:
        return None

    roots: list[Path] = []
    for source in discover_d1namo_sources(dest):
        if source.subject_id == subject_id:
            roots.append(source.subject_dir)
    if not roots:
        roots.append(dataset_root(dest) / subject_id)

    for root in roots:
        candidate = root.joinpath(*parts).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            continue
        if candidate.is_file():
            return candidate
    return None


def load_d1namo_data(file_path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load one D1NAMO participant into the app store schema."""
    path = Path(file_path)
    with start_action(action_type=u"load_d1namo_data", file_path=str(path)) as action:
        glucose_path = path if path.name.lower() == "glucose.csv" else path.parent / "glucose.csv"
        if not glucose_path.is_file() and is_d1namo_source_name(path.name):
            subject_id = subject_id_from_path(path)
            for source in discover_d1namo_sources():
                if source.subject_id == subject_id:
                    glucose_path = source.csv_path
                    break
        if not glucose_path.is_file():
            raise FileNotFoundError(f"D1NAMO glucose.csv not found for {path}")

        subject_dir = glucose_path.parent
        subject_token = subject_id_from_path(glucose_path) or "001"
        # The subject is a *bundle*: glucose, insulin and food are three files
        # that parse into one frame, so the library takes the directory.
        unified_df = FormatParser.parse_subject_directory(subject_dir)
        glucose_df, events_df = adapt_unified(
            unified_df,
            subject_id=int(subject_token),
            photo_resolver=lambda raw: resolve_photo_path(raw, subject_dir),
        )
        action.add_success_fields(
            subject_id=subject_token,
            glucose_rows=glucose_df.height,
            event_rows=events_df.height,
            photo_events=events_df.filter(pl.col("photo_path") != "").height,
            insulin_events=events_df.filter(pl.col("event_type") == "Insulin").height,
        )
        return glucose_df, events_df
