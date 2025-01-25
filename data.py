from typing import Tuple
from pathlib import Path
import polars as pl

def load_glucose_data(file_path: Path = Path("data/example.csv")) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Load glucose data from a CSV file that could be Dexcom or Libre.  
    We first try reading the file normally, detect the type, and if that fails,
    skip the first line and try again. 
    """
    def _read_csv_skipping(n: int) -> pl.DataFrame:
        return pl.read_csv(
            file_path,
            skip_rows=n,
            null_values=["Low", "High"],
            truncate_ragged_lines=True
        )

    # --- 1) Read once without skipping ---
    try:
        temp_df = _read_csv_skipping(0)
        # Try finding out if it’s Dexcom vs Libre
        cgm_type = detect_cgm_type(temp_df)
        df = temp_df
    except ValueError:
        # Possibly the first line is just metadata, so skip one row and re‐read…
        temp_df2 = _read_csv_skipping(1)
        cgm_type = detect_cgm_type(temp_df2)  # This should succeed now.
        df = temp_df2

    # Now branch on cgm_type rather than “libre” in filename:
    if cgm_type == "libre":
        # Same “Libre” processing as before
        glucose_data = (
            df
            .filter(pl.col("Record Type") == "0")
            .select([
                pl.col("Device Timestamp").alias("time"),
                pl.col("Historic Glucose mg/dL").cast(pl.Float64).alias("gl"),
            ])
            .filter(pl.col("gl").is_not_null())
            .with_columns([
                pl.col("time").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M"),
                pl.lit(0.0).alias("prediction"),
            ])
            .sort("time")
        )

        events_data = (
            df
            .filter(
                pl.col("Rapid-Acting Insulin (units)").is_not_null()
                | pl.col("Long-Acting Insulin Value (units)").is_not_null()
                | pl.col("Carbohydrates (grams)").is_not_null()
            )
            .select([
                pl.col("Device Timestamp").alias("time"),
                pl.when(pl.col("Rapid-Acting Insulin (units)").is_not_null())
                  .then(pl.lit("Insulin"))
                  .when(pl.col("Long-Acting Insulin Value (units)").is_not_null())
                  .then(pl.lit("Insulin"))
                  .when(pl.col("Carbohydrates (grams)").is_not_null())
                  .then(pl.lit("Carbohydrates"))
                  .alias("event_type"),
                pl.when(pl.col("Rapid-Acting Insulin (units)").is_not_null())
                  .then(pl.lit("Fast-Acting"))
                  .when(pl.col("Long-Acting Insulin Value (units)").is_not_null())
                  .then(pl.lit("Long-Acting"))
                  .otherwise(None)
                  .alias("event_subtype"),
                pl.coalesce([
                    pl.col("Rapid-Acting Insulin (units)"),
                    pl.col("Long-Acting Insulin Value (units)")
                ]).alias("insulin_value"),
            ])
            .with_columns([
                pl.col("time").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M")
            ])
            .sort("time")
        )

    else:
        # Dexcom branch
        glucose_data = (
            df
            .filter(pl.col("Event Type") == "EGV")
            .select([
                pl.col("Timestamp (YYYY-MM-DDThh:mm:ss)").alias("time"),
                pl.col("Glucose Value (mg/dL)").cast(pl.Float64).alias("gl"),
            ])
            .with_columns([
                pl.col("time").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S"),
                pl.lit(0.0).alias("prediction"),
            ])
            .sort("time")
        )

        events_data = (
            df
            .filter(
                (pl.col("Event Type") == "Insulin")
                | (pl.col("Event Type") == "Carbohydrates")
            )
            .select([
                pl.col("Timestamp (YYYY-MM-DDThh:mm:ss)").alias("time"),
                pl.col("Event Type").alias("event_type"),
                pl.col("Event Subtype").alias("event_subtype"),
                pl.col("Insulin Value (u)").alias("insulin_value"),
            ])
            .with_columns([
                pl.col("time").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S")
            ])
            .sort("time")
        )

    return glucose_data, events_data