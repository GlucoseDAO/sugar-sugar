from typing import List, Dict, Tuple, Optional, Any, Union
import polars as pl
from datetime import datetime
from pathlib import Path

'''
Load the data from the csv file
'''

# Modify load_glucose_data to load all data without limit
def load_glucose_data(file_path: Path = Path("data/example.csv")) -> Tuple[pl.DataFrame, pl.DataFrame]:
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

