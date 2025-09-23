from enum import Enum, auto
from typing import List, Dict, Tuple, Optional, Any, Union
import polars as pl
from datetime import datetime
from pathlib import Path

class CGMType(Enum):
    LIBRE = "libre"
    DEXCOM = "dexcom"

'''
Load the data from the csv file
'''

# Modify load_glucose_data to load all data without limit
def load_glucose_data(file_path: Path = Path("data/example.csv")) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Load CGM data based on detected type."""
    print(f"DEBUG[data.load_glucose_data]: called with file_path={file_path}")
    cgm_type = detect_cgm_type(file_path)
    print(f"DEBUG[data.load_glucose_data]: detected cgm_type={cgm_type}")
    
    if cgm_type == CGMType.LIBRE:
        glucose_data, events_data = load_libre_data(file_path)
    else:
        glucose_data, events_data = load_dexcom_data(file_path)
    
    # Add age and user_id columns
    glucose_data = glucose_data.with_columns([
        pl.lit(0).alias("age"),  # Default age of 0
        pl.lit(1).alias("user_id")  # Default user_id of 1
    ])
    print(
        f"DEBUG[data.load_glucose_data]: loaded glucose rows={glucose_data.height}, events rows={events_data.height}"
    )
    
    return glucose_data, events_data


def detect_cgm_type(file_path: Path) -> CGMType:
    """Detect if the CSV file is from Libre or Dexcom CGM."""
    with open(file_path, 'r') as file:
        # Read first few lines to detect the format
        first_lines = [next(file) for _ in range(12)]
        try:
            preview = "".join(first_lines[:3]).strip().replace("\n", " | ")
        except Exception:
            preview = "<unavailable>"
        print(f"DEBUG[data.detect_cgm_type]: preview first lines: {preview}")
        
        # Check for Libre indicators
        if any("Glucose Data,Generated" in line for line in first_lines):
            print("DEBUG[data.detect_cgm_type]: identified LIBRE by header match")
            return CGMType.LIBRE
        # Check for Dexcom indicators
        elif any("Dexcom" in line for line in first_lines):
            print("DEBUG[data.detect_cgm_type]: identified DEXCOM by header match")
            return CGMType.DEXCOM
        else:
            print("DEBUG[data.detect_cgm_type]: unknown format; raising ValueError")
            raise ValueError("Unknown CGM data format")

def load_cgm_data(file_path: Path) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Load CGM data based on detected type."""
    cgm_type = detect_cgm_type(file_path)
    
    if cgm_type == CGMType.LIBRE:
        return load_libre_data(file_path)
    else:
        return load_dexcom_data(file_path)  # existing function

def load_libre_data(file_path: Path) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Load and process Libre CGM data to match Dexcom format."""
    print(f"DEBUG[data.load_libre_data]: reading {file_path}")
    # Read CSV skipping first 2 header rows
    df = pl.read_csv(
        file_path,
        skip_lines=1,
        truncate_ragged_lines=True
    )
    print(f"DEBUG[data.load_libre_data]: raw rows={df.height}, cols={len(df.columns)}")
    
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
    try:
        tmin = glucose_data.get_column("time").min()
        tmax = glucose_data.get_column("time").max()
        print(f"DEBUG[data.load_libre_data]: glucose rows={glucose_data.height}, time_range=[{tmin} .. {tmax}]")
    except Exception:
        print("DEBUG[data.load_libre_data]: glucose time range unavailable")
    
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
    print(f"DEBUG[data.load_libre_data]: events rows={events_data.height}")
    
    return glucose_data, events_data

def load_dexcom_data(file_path: Path) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Load and process Dexcom CGM data."""
    print(f"DEBUG[data.load_dexcom_data]: reading {file_path}")
    df = pl.read_csv(
        file_path,
        null_values=["Low", "High"],
        truncate_ragged_lines=True
    )
    print(f"DEBUG[data.load_dexcom_data]: raw rows={df.height}, cols={len(df.columns)}")
    
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
    try:
        tmin = glucose_data.get_column("time").min()
        tmax = glucose_data.get_column("time").max()
        print(f"DEBUG[data.load_dexcom_data]: glucose rows={glucose_data.height}, time_range=[{tmin} .. {tmax}]")
    except Exception:
        print("DEBUG[data.load_dexcom_data]: glucose time range unavailable")
    
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
    print(f"DEBUG[data.load_dexcom_data]: events rows={events_data.height}")
    
    return glucose_data, events_data

