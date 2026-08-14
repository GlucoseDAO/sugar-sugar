from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from typing import Dict

import pytest

from sugar_sugar.download_cgmacros import extract_zip
from sugar_sugar.download_d1namo import dataset_is_present, md5_file, verify_md5


def _write_zip(path: Path, members: Dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


def test_dataset_is_present_requires_glucose_csv(tmp_path: Path) -> None:
    dest = tmp_path / "d1namo"
    dest.mkdir()
    assert not dataset_is_present(dest)
    (dest / "001").mkdir()
    (dest / "001" / "insulin.csv").write_text("date,time\n")
    assert not dataset_is_present(dest)
    (dest / "001" / "glucose.csv").write_text("date,time,glucose,type\n")
    assert dataset_is_present(dest)


def test_extract_zip_skips_d1namo_pictures_without_flag(tmp_path: Path) -> None:
    zip_path = tmp_path / "diabetes_subset_pictures-glucose-food-insulin.zip"
    _write_zip(
        zip_path,
        {
            "diabetes_subset_pictures-glucose-food-insulin/001/glucose.csv": b"date,time\n",
            "diabetes_subset_pictures-glucose-food-insulin/001/insulin.csv": b"date,time\n",
            "diabetes_subset_pictures-glucose-food-insulin/001/pictures/meal.jpg": b"jpeg",
        },
    )
    dest = tmp_path / "out"
    written = extract_zip(zip_path, dest, include_photos=False)
    assert written == 2
    assert dataset_is_present(dest)
    assert not (
        dest / "diabetes_subset_pictures-glucose-food-insulin/001/pictures/meal.jpg"
    ).exists()


def test_extract_zip_keeps_d1namo_pictures_when_requested(tmp_path: Path) -> None:
    zip_path = tmp_path / "data.zip"
    _write_zip(
        zip_path,
        {
            "001/glucose.csv": b"x\n",
            "001/pictures/meal.jpg": b"jpeg",
        },
    )
    dest = tmp_path / "out"
    written = extract_zip(zip_path, dest, include_photos=True)
    assert written == 2
    assert (dest / "001/pictures/meal.jpg").read_bytes() == b"jpeg"


def test_verify_md5_accepts_match_and_rejects_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")
    empty_md5 = hashlib.md5(b"").hexdigest()
    assert md5_file(path) == empty_md5
    verify_md5(path, empty_md5)
    with pytest.raises(ValueError, match="MD5 mismatch"):
        verify_md5(path, "0" * 32)
