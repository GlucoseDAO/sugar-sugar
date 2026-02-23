from enum import Enum
from typing import Iterable, Tuple
import polars as pl
from pathlib import Path
from eliot import start_action

class CGMType(Enum):
    LIBRE = "libre"
    DEXCOM = "dexcom"
    MEDTRONIC = "medtronic"

'''
Load the data from the csv file
'''

# Modify load_glucose_data to load all data without limit
def load_glucose_data(file_path: Path = Path("data/example.csv")) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Load CGM data based on detected type."""
    with start_action(action_type=u"load_glucose_data", file_path=str(file_path)):
        cgm_type = detect_cgm_type(file_path)
        if cgm_type == CGMType.LIBRE:
            glucose_data, events_data = load_libre_data(file_path)
        elif cgm_type == CGMType.MEDTRONIC:
            glucose_data, events_data = load_medtronic_data(file_path)
        else:
            glucose_data, events_data = load_dexcom_data(file_path)

        # Add age and user_id columns
        glucose_data = glucose_data.with_columns([
            pl.lit(0).alias("age"),  # Default age of 0
            pl.lit(1).alias("user_id")  # Default user_id of 1
        ])

        return glucose_data, events_data


def detect_cgm_type(file_path: Path) -> CGMType:
    """Detect if the CSV file is from Libre, Dexcom, or Medtronic CGM."""
    with start_action(action_type=u"detect_cgm_type", file_path=str(file_path)):
        first_lines = _read_first_lines(file_path, max_lines=20)

        # Check for Libre indicators
        if any("Glucose Data,Generated" in line for line in first_lines):
            return CGMType.LIBRE
        # Check for Dexcom indicators
        if any("Dexcom" in line for line in first_lines):
            return CGMType.DEXCOM
        # Check for Medtronic indicators (Guardian / CareLink export)
        if _find_medtronic_header_line(first_lines) is not None:
            return CGMType.MEDTRONIC
        raise ValueError("Unknown CGM data format")

def load_cgm_data(file_path: Path) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Load CGM data based on detected type."""
    cgm_type = detect_cgm_type(file_path)
    
    if cgm_type == CGMType.LIBRE:
        return load_libre_data(file_path)
    elif cgm_type == CGMType.MEDTRONIC:
        return load_medtronic_data(file_path)
    else:
        return load_dexcom_data(file_path)  # existing function


def _read_first_lines(file_path: Path, *, max_lines: int) -> list[str]:
    with file_path.open("r", encoding="utf-8-sig", errors="replace") as f:
        lines: list[str] = []
        for _ in range(max_lines):
            line = f.readline()
            if not line:
                break
            lines.append(line)
        return lines


_MEDTRONIC_REQUIRED_HEADERS: tuple[str, ...] = (
    "Index",
    "Date",
    "Time",
    "Sensor Glucose (mg/dL)",
)


def _split_header_candidates(line: str) -> list[list[str]]:
    stripped = line.strip().lstrip("\ufeff")
    if not stripped:
        return []
    candidates: list[list[str]] = []
    if ";" in stripped:
        candidates.append([c.strip().strip('"') for c in stripped.split(";")])
    if "," in stripped:
        candidates.append([c.strip().strip('"') for c in stripped.split(",")])
    # If the line is already split-like (no delimiter), treat as one header blob (unlikely).
    candidates.append([stripped.strip().strip('"')])
    return candidates


def _find_medtronic_header_line(first_lines: Iterable[str]) -> int | None:
    for idx, line in enumerate(first_lines):
        for headers in _split_header_candidates(line):
            # Header sometimes comes as one cell like 'Index;Date;Time;...'
            if len(headers) == 1 and ";" in headers[0]:
                headers = [c.strip().strip('"') for c in headers[0].split(";")]
            if all(req in headers for req in _MEDTRONIC_REQUIRED_HEADERS):
                return idx
    return None


def _euro_number_to_float(expr: pl.Expr) -> pl.Expr:
    return (
        expr.cast(pl.Utf8, strict=False)
        .str.replace_all(",", ".")
        .cast(pl.Float64, strict=False)
    )


def load_medtronic_data(file_path: Path) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Load and process Medtronic Guardian Connect / CareLink CSV export."""
    with start_action(action_type=u"load_medtronic_data", file_path=str(file_path)):
        first_lines = _read_first_lines(file_path, max_lines=30)
        header_line_idx = _find_medtronic_header_line(first_lines)
        skip_lines = int(header_line_idx or 0)

        # Medtronic exports often contain placeholders like "-------" in numeric columns.
        # If schema inference doesn't see them early, Polars may infer an int/float dtype
        # and then fail parsing later. We force these columns to be strings and later
        # convert with `strict=False` to safely produce nulls for placeholders.
        schema_overrides: dict[str, pl.DataType] = {
            "Sensor Glucose (mg/dL)": pl.Utf8,
            "BG Reading (mg/dL)": pl.Utf8,
            "Basal Rate (U/h)": pl.Utf8,
            "Bolus Volume Delivered (U)": pl.Utf8,
            "BWZ Carb Input (grams)": pl.Utf8,
            "Sensor Calibration BG (mg/dL)": pl.Utf8,
            "Event Marker": pl.Utf8,
            "Alarm": pl.Utf8,
        }

        df = pl.read_csv(
            file_path,
            separator=";",
            skip_lines=skip_lines,
            truncate_ragged_lines=True,
            infer_schema_length=200,
            schema_overrides=schema_overrides,
        )

        ts_raw = pl.concat_str([pl.col("Date"), pl.col("Time")], separator=" ").alias("_ts_raw")
        time_expr = pl.coalesce(
            [
                ts_raw.str.strptime(pl.Datetime, "%Y/%m/%d %H:%M:%S", strict=False),
                ts_raw.str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False),
                ts_raw.str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S", strict=False),
            ]
        ).alias("time")

        sensor_gl = _euro_number_to_float(pl.col("Sensor Glucose (mg/dL)")).alias("_sensor_gl")
        bg_gl = (
            _euro_number_to_float(pl.col("BG Reading (mg/dL)"))
            if "BG Reading (mg/dL)" in df.columns
            else pl.lit(None, dtype=pl.Float64)
        ).alias("_bg_gl")

        glucose_data = (
            df.with_columns([time_expr, sensor_gl, bg_gl])
            .with_columns(pl.coalesce([pl.col("_sensor_gl"), pl.col("_bg_gl")]).alias("gl"))
            .filter(pl.col("time").is_not_null() & pl.col("gl").is_not_null())
            .select(
                [
                    pl.col("time"),
                    pl.col("gl"),
                    pl.lit(0.0).alias("prediction"),
                ]
            )
            .sort("time")
        )

        # Events: insulin boluses + carbohydrate entries (if present)
        bolus_u = (
            _euro_number_to_float(pl.col("Bolus Volume Delivered (U)"))
            if "Bolus Volume Delivered (U)" in df.columns
            else pl.lit(None, dtype=pl.Float64)
        ).alias("_bolus_u")
        carbs_g = (
            _euro_number_to_float(pl.col("BWZ Carb Input (grams)"))
            if "BWZ Carb Input (grams)" in df.columns
            else pl.lit(None, dtype=pl.Float64)
        ).alias("_carbs_g")

        marker_expr = (
            pl.col("Event Marker").cast(pl.Utf8, strict=False).fill_null("")
            if "Event Marker" in df.columns
            else pl.lit("", dtype=pl.Utf8)
        )
        marker = marker_expr.alias("_marker")
        marker_insulin = _euro_number_to_float(
            marker_expr.str.extract(r"Insulin:\s*([\d,\.]+)", 1)
        ).alias("_marker_insulin_u")
        marker_carbs = _euro_number_to_float(
            marker_expr.str.extract(r"Meal:\s*([\d,\.]+)\s*grams?", 1)
        ).alias("_marker_carbs_g")

        events_base = df.with_columns([time_expr, bolus_u, carbs_g, marker, marker_insulin, marker_carbs])

        insulin_value = pl.coalesce([pl.col("_bolus_u"), pl.col("_marker_insulin_u")]).alias("_insulin_value")
        insulin_subtype = (
            pl.when(pl.col("_bolus_u").is_not_null())
            .then(pl.lit("Bolus"))
            .when(pl.col("_marker_insulin_u").is_not_null())
            .then(pl.lit("Event Marker"))
            .otherwise(pl.lit(""))
            .alias("_insulin_subtype")
        )
        insulin_events = (
            events_base.with_columns([insulin_value, insulin_subtype])
            .filter(pl.col("time").is_not_null() & pl.col("_insulin_value").is_not_null())
            .select(
                [
                    pl.col("time"),
                    pl.lit("Insulin").alias("event_type"),
                    pl.col("_insulin_subtype").alias("event_subtype"),
                    pl.col("_insulin_value").alias("insulin_value"),
                ]
            )
        )

        carb_value = pl.coalesce([pl.col("_carbs_g"), pl.col("_marker_carbs_g")]).alias("_carb_value")
        carb_subtype = (
            pl.when(pl.col("_carbs_g").is_not_null())
            .then(pl.lit("Carbs"))
            .when(pl.col("_marker_carbs_g").is_not_null())
            .then(pl.lit("Event Marker"))
            .otherwise(pl.lit(""))
            .alias("_carb_subtype")
        )
        carb_events = (
            events_base.with_columns([carb_value, carb_subtype])
            .filter(pl.col("time").is_not_null() & pl.col("_carb_value").is_not_null())
            .select(
                [
                    pl.col("time"),
                    pl.lit("Carbohydrates").alias("event_type"),
                    pl.col("_carb_subtype").alias("event_subtype"),
                    pl.lit(None).cast(pl.Float64).alias("insulin_value"),
                ]
            )
        )

        events_data = pl.concat([insulin_events, carb_events], how="vertical_relaxed").sort("time")
        return glucose_data, events_data

def load_libre_data(file_path: Path) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Load and process Libre CGM data to match Dexcom format."""
    with start_action(action_type=u"load_libre_data", file_path=str(file_path)):
        # Read CSV skipping first 2 header rows
        df = pl.read_csv(
            file_path,
            skip_lines=1,
            truncate_ragged_lines=True
        )

        # Filter glucose data (Record Type = 0 for historic readings)
        glucose_data = (df
            .filter(pl.col("Record Type").cast(pl.Int64) == 0)
            .select([
                pl.col("Device Timestamp").alias("time"),
                pl.col("Historic Glucose mg/dL").cast(pl.Float64).alias("gl")
            ])
            .with_columns([
                pl.col("time").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M"),
                pl.lit(0.0).alias("prediction")
            ])
            .sort("time")
        )

        # Filter scan data (Record Type = 1 for manual scans)
        events_data = (df
            .filter(pl.col("Record Type").cast(pl.Int64) == 1)
            .select([
                pl.col("Device Timestamp").alias("time"),
                pl.lit("Scan").alias("event_type"),
                pl.lit("Manual Scan").alias("event_subtype"),
                pl.lit(None).cast(pl.Float64).alias("insulin_value")
            ])
            .with_columns([
                pl.col("time").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M")
            ])
            .sort("time")
        )

        return glucose_data, events_data

def load_dexcom_data(file_path: Path) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Load and process Dexcom CGM data."""
    with start_action(action_type=u"load_dexcom_data", file_path=str(file_path)):
        df = pl.read_csv(
            file_path,
            null_values=["Low", "High"],
            truncate_ragged_lines=True
        )

        # Filter glucose data (EGV rows)
        glucose_data = (df
            .filter(pl.col("Event Type") == "EGV")
            .select([
                pl.col("Timestamp (YYYY-MM-DDThh:mm:ss)").alias("time"),
                pl.col("Glucose Value (mg/dL)").cast(pl.Float64).alias("gl")
            ])
            .with_columns([
                pl.col("time").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S"),
                pl.lit(0.0).alias("prediction")
            ])
            .sort("time")
        )

        # Filter event data (non-EGV rows we want to show)
        events_data = (df
            .filter(
                (pl.col("Event Type") == "Insulin") |
                (pl.col("Event Type") == "Exercise") |
                (pl.col("Event Type") == "Carbohydrates")
            )
            .select([
                pl.col("Timestamp (YYYY-MM-DDThh:mm:ss)").alias("time"),
                pl.col("Event Type").alias("event_type"),
                pl.col("Event Subtype").alias("event_subtype"),
                pl.col("Insulin Value (u)").alias("insulin_value")
            ])
            .with_columns([
                pl.col("time").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S")
            ])
            .sort("time")
        )

        return glucose_data, events_data

