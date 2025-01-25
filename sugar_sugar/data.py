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
    cgm_type = detect_cgm_type(file_path)
    
    if cgm_type == CGMType.LIBRE:
        return load_libre_data(file_path)
    else:
        return load_dexcom_data(file_path)  # existing function


def detect_cgm_type(file_path: Path) -> CGMType:
    """Detect if the CSV file is from Libre or Dexcom CGM."""
    with open(file_path, 'r') as file:
        # Read first few lines to detect the format
        first_lines = [next(file) for _ in range(5)]
        
        # Check for Libre indicators
        if any("Glucose Data,Generated" in line for line in first_lines):
            return CGMType.LIBRE
        # Check for Dexcom indicators
        elif any("Index,Timestamp" in line for line in first_lines):
            return CGMType.DEXCOM
        else:
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

