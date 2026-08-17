from __future__ import annotations

from pathlib import Path

import pytest

from sugar_sugar import download_bigideas
from sugar_sugar.download_bigideas import (
    FILES_BASE,
    S3_BASE,
    SUBJECT_IDS,
    dataset_is_present,
    fetch_bigideas,
)


def test_dataset_is_present_requires_dexcom_csv(tmp_path: Path) -> None:
    dest = tmp_path / "bigideas"
    dest.mkdir()
    assert not dataset_is_present(dest)
    (dest / "001").mkdir()
    (dest / "001" / "Food_Log_001.csv").write_text("date,time\n")
    assert not dataset_is_present(dest)
    (dest / "001" / "Dexcom_001.csv").write_text("Timestamp,Glucose Value (mg/dL)\n")
    assert dataset_is_present(dest)


def _fake_download(served: set[str]) -> object:
    """Stub for ``download_url``: serves only URLs whose tail is in *served*."""

    def download(url: str, dest: Path, **_: object) -> None:
        rel = url.split("/1.1.3/", 1)[1]
        if rel not in served:
            raise OSError(f"HTTP Error 404: Not Found ({url})")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("Timestamp,Glucose Value (mg/dL)\n")

    return download


def _all_rel_paths() -> set[str]:
    paths = {"Demographics.csv"}
    for subject_id in SUBJECT_IDS:
        folder = f"{subject_id:03d}"
        paths.add(f"{folder}/Dexcom_{folder}.csv")
        paths.add(f"{folder}/Food_Log_{folder}.csv")
    return paths


def test_files_mirror_is_tried_before_s3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The S3 bucket has no 1.1.3, so a healthy run must never fall back to it."""
    tried: list[str] = []

    def download(url: str, dest: Path, **_: object) -> None:
        tried.append(url)
        _fake_download(_all_rel_paths())(url, dest)

    monkeypatch.setattr(download_bigideas, "download_url", download)
    fetch_bigideas(tmp_path / "bigideas")

    assert tried, "no download was attempted"
    assert all(url.startswith(FILES_BASE) for url in tried)
    assert not any(url.startswith(S3_BASE) for url in tried)
    assert len(tried) == 1 + 2 * len(SUBJECT_IDS)


def test_subject_missing_food_log_fails_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A half-downloaded subject is unplayable, so the fetch must not report success."""
    served = _all_rel_paths() - {"007/Food_Log_007.csv"}
    monkeypatch.setattr(download_bigideas, "download_url", _fake_download(served))

    with pytest.raises(FileNotFoundError, match="007"):
        fetch_bigideas(tmp_path / "bigideas")
