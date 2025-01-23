import polars as pl
from datetime import datetime
from typing import Tuple, Union
from pathlib import Path
import typer

def extract_glucose_dexcom(
    csv_path: Union[str, Path], 
    high_value: float = 400, 
    low_value: float = 40,
    write_to: Union[str, Path, None] = None
) -> pl.DataFrame:
    """
    Extract glucose values and timestamps from Dexcom CGM data using polars.
    
    Args:
        csv_path: Path to the CSV file or URL containing Dexcom CGM data (str or Path object)
        high_value: Value to use when glucose reading is "High" (default: 400)
        low_value: Value to use when glucose reading is "Low" (default: 40)
        write_to: Optional path to save the transformed DataFrame as CSV (default: None)
        
    Returns:
        DataFrame with columns:
        - time: Timestamp column
        - gl: Glucose values column
    """
    # If input is a Path object, convert to string
    csv_path = str(csv_path)
    
    # Read CSV with polars - works with both local files and URLs
    df = pl.read_csv(csv_path)
    
    # Filter for EGV (Estimated Glucose Value) rows and non-null glucose values
    glucose_df = df.filter(
        (pl.col("Event Type") == "EGV") & 
        (pl.col("Glucose Value (mg/dL)").is_not_null())
    )
    
    result_df = glucose_df.select([
        pl.col("Timestamp (YYYY-MM-DDThh:mm:ss)").alias("time"),
        pl.col("Glucose Value (mg/dL)")
        .map_elements(lambda x: high_value if x == "High" else (low_value if x == "Low" else x))
        .cast(pl.Float64)
        .alias("gl")
    ])
    
    if write_to is not None:
        result_df.write_csv(str(write_to))
    
    return result_df

app = typer.Typer()

@app.command()
def process_dexcom(
    csv_path: str = typer.Argument(..., help="Path to the Dexcom CSV file"),
    output: str = typer.Option(None, "--output", "-o", help="Output CSV file path"),
    high_value: float = typer.Option(400, "--high", "-h", help="Value to use for 'High' readings"),
    low_value: float = typer.Option(40, "--low", "-l", help="Value to use for 'Low' readings"),
):
    """Process Dexcom CGM data and extract glucose values."""
    result = extract_glucose_dexcom(csv_path, high_value, low_value, output)
    if output:
        print(f"Processed data saved to: {output}")
    else:
        print(result)

if __name__ == "__main__":
    app()