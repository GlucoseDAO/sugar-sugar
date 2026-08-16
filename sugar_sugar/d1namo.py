"""D1NAMO (Dubosson) discovery, photo serving, and library-backed load.

Parsing is ``cgm-format`` 0.10+ (``parse_subject_directory``). This module
keeps the app-specific pieces: Format A source listing, meal-photo URLs, and
path safety. Download a local copy with ``uv run download-d1namo``.
Paper: Dubosson et al., Informatics in Medicine Unlocked, 2018.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import polars as pl

from sugar_sugar.download_d1namo import dataset_is_present, default_dest

_PHOTO_SUFFIXES: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".heic")
_SUBJECT_DIR_RE = re.compile(r"^(\d{3})$")
_SOURCE_NAME_RE = re.compile(r"^D1NAMO-(\d{3})\.csv$", re.IGNORECASE)


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
    """True when *file_path* is a D1NAMO glucose table or virtual source name."""
    path = Path(file_path)
    if is_d1namo_source_name(path.name):
        return True
    if path.name.lower() != "glucose.csv":
        return False
    if _SUBJECT_DIR_RE.match(path.parent.name):
        return True
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


def resolve_photo_path(image_path: str, subject_dir: Path) -> str:
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
    """Load one D1NAMO participant through ``cgm-format`` into the app schema."""
    from sugar_sugar.data import load_glucose_data

    return load_glucose_data(Path(file_path))
