from __future__ import annotations

from pathlib import Path

from sugar_sugar.download_bigideas import dataset_is_present


def test_dataset_is_present_requires_dexcom_csv(tmp_path: Path) -> None:
    dest = tmp_path / "bigideas"
    dest.mkdir()
    assert not dataset_is_present(dest)
    (dest / "001").mkdir()
    (dest / "001" / "Food_Log_001.csv").write_text("date,time\n")
    assert not dataset_is_present(dest)
    (dest / "001" / "Dexcom_001.csv").write_text("Timestamp,Glucose Value (mg/dL)\n")
    assert dataset_is_present(dest)
