from __future__ import annotations

from pathlib import Path

import polars as pl

from sugar_sugar.data import CGMType, detect_cgm_type, load_glucose_data


def test_detect_and_load_medtronic_csv(tmp_path: Path) -> None:
    csv_text = "\n".join(
        [
            "Index;Date;Time;Sensor Glucose (mg/dL);BG Reading (mg/dL);Bolus Volume Delivered (U);BWZ Carb Input (grams);Event Marker",
            "1;2025/01/01;09:00:00;120,0;;;;",
            "2;2025/01/01;09:05:00;;------- ;;;",
            "3;2025/01/01;09:10:00;140,0;;1,50;;",
            "4;2025/01/01;09:15:00;150,0;;;;Meal: 60,00grams",
        ]
    )
    path = tmp_path / "medtronic.csv"
    path.write_text(csv_text, encoding="utf-8")

    assert detect_cgm_type(path) == CGMType.MEDTRONIC

    glucose_df, events_df = load_glucose_data(path)

    assert glucose_df.columns == ["time", "gl", "prediction", "age", "user_id"]
    assert glucose_df.get_column("time").dtype == pl.Datetime
    assert glucose_df.get_column("gl").to_list() == [120.0, 140.0, 150.0]

    assert events_df.columns == ["time", "event_type", "event_subtype", "insulin_value"]
    assert set(events_df.get_column("event_type").to_list()) == {"Insulin", "Carbohydrates"}
    assert events_df.filter(pl.col("event_type") == "Insulin").get_column("insulin_value").to_list() == [1.5]
