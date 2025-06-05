from dash import html, dcc
from dash.html import Div
import dash_bootstrap_components as dbc
import polars as pl
from datetime import datetime
import uuid
import csv

class SubmitComponent(Div):
    def __init__(self):
        super().__init__([
            dbc.Button(
                "Submit",
                id="submit-button",
                color="primary",
                className="mt-4",
                style={'width': '300px', 'fontSize': '18px', 'padding': '15px 0', 'textAlign': 'center', 'verticalAlign': 'middle', 'lineHeight': '1.5', 'height': '60px'}
            ),
            dcc.Store(id='prediction-stats-store', data=None)
        ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'})

    def save_statistics(self, df: pl.DataFrame, user_info: dict) -> None:
        """Save prediction statistics to CSV file"""
        # Define the CSV file path
        csv_file_path = 'prediction_statistics.csv'

        # Extract prediction data from the prediction table
        prediction_table_data = user_info.get('prediction_table_data', [])
        parameters = [row.get('parameter', '') for row in prediction_table_data]
        actual_values = [row.get('actual_value', '') for row in prediction_table_data]
        prediction_time = [row.get('prediction_time', '') for row in prediction_table_data]

        # Prepare the data to write
        data = {
            'id': user_info.get('id', ''),
            'number': user_info.get('number', ''),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'email': user_info.get('email', ''),
            'gender': user_info.get('gender', ''),
            'diabetic': user_info.get('diabetic', ''),
            'diabetic_type': user_info.get('diabetic_type', ''),
            'diabetes_duration': user_info.get('diabetes_duration', ''),
            'other_medical_conditions': user_info.get('other_medical_conditions', ''),
            'location': user_info.get('location', ''),
            'parameters': parameters,
            'actual_values': actual_values,
            'prediction_time': prediction_time
        }

        # Write to CSV with headers
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=data.keys())
            if file.tell() == 0:  # Check if the file is empty
                writer.writeheader()  # Write headers if the file is empty
            writer.writerow(data)

        # Reset the prediction
        df = df.with_columns(pl.lit(0.0).alias("prediction")) 