from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from typing import Dict

import pytest

from sugar_sugar.download_cgmacros import (
    dataset_is_present,
    extract_zip,
    is_photo_member,
    sha256_file,
    verify_sha256,
)


def _write_zip(path: Path, members: Dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


def test_is_photo_member_detects_meal_photos() -> None:
    assert is_photo_member("CGMacros/CGMacros-001/photos/0001.jpg")
    assert is_photo_member("CGMacros/CGMacros-001/photos\\meal.JPEG")
    assert not is_photo_member("CGMacros/CGMacros-001/CGMacros-001.csv")
    assert not is_photo_member("CGMacros/bio.csv")


def test_extract_zip_skips_photos_by_default(tmp_path: Path) -> None:
    zip_path = tmp_path / "CGMacros_dateshifted365.zip"
    _write_zip(
        zip_path,
        {
            "CGMacros/CGMacros-001/CGMacros-001.csv": b"Timestamp,Libre GL\n",
            "CGMacros/CGMacros-001/photos/meal.jpg": b"not-a-real-jpeg",
            "CGMacros/bio.csv": b"id,age\n",
        },
    )
    dest = tmp_path / "out"
    written = extract_zip(zip_path, dest, include_photos=False)
    assert written == 2
    assert (dest / "CGMacros/CGMacros-001/CGMacros-001.csv").read_bytes() == (
        b"Timestamp,Libre GL\n"
    )
    assert (dest / "CGMacros/bio.csv").is_file()
    assert not (dest / "CGMacros/CGMacros-001/photos/meal.jpg").exists()
    assert dataset_is_present(dest)


def test_extract_zip_keeps_photos_when_requested(tmp_path: Path) -> None:
    zip_path = tmp_path / "data.zip"
    _write_zip(
        zip_path,
        {
            "CGMacros/CGMacros-002/CGMacros-002.csv": b"x\n",
            "CGMacros/CGMacros-002/photos/meal.jpg": b"jpeg",
        },
    )
    dest = tmp_path / "out"
    written = extract_zip(zip_path, dest, include_photos=True)
    assert written == 2
    assert (dest / "CGMacros/CGMacros-002/photos/meal.jpg").read_bytes() == b"jpeg"


def test_extract_zip_rejects_path_escape(tmp_path: Path) -> None:
    zip_path = tmp_path / "evil.zip"
    _write_zip(zip_path, {"../outside.csv": b"nope"})
    with pytest.raises(ValueError, match="unsafe zip path"):
        extract_zip(zip_path, tmp_path / "out")


def test_verify_sha256_accepts_match_and_rejects_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "LICENSE.txt"
    path.write_bytes(b"")
    empty_sha = hashlib.sha256(b"").hexdigest()
    assert sha256_file(path) == empty_sha
    verify_sha256(path, empty_sha)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_sha256(path, "0" * 64)


def test_dataset_is_present_requires_participant_csv(tmp_path: Path) -> None:
    dest = tmp_path / "cgmmacros"
    dest.mkdir()
    assert not dataset_is_present(dest)
    (dest / "CGMacros").mkdir()
    (dest / "CGMacros" / "bio.csv").write_text("id\n")
    assert not dataset_is_present(dest)
    participant = dest / "CGMacros" / "CGMacros-001"
    participant.mkdir()
    (participant / "CGMacros-001.csv").write_text("Timestamp\n")
    assert dataset_is_present(dest)
