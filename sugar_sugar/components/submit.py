from dash import html, dcc
from dash.html import Div
import dash_bootstrap_components as dbc
import polars as pl
from datetime import datetime
import uuid
import csv
import os

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

    def _get_next_number(self):
        """Get the next number for the prediction statistics."""
        csv_file = 'prediction_statistics.csv'
        if not os.path.exists(csv_file):
            return 0
        
        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                numbers = [int(row['number']) for row in reader if row['number'].isdigit()]
                return max(numbers) + 1 if numbers else 0
        except Exception:
            return 0

    def save_statistics(self, df: pl.DataFrame, user_info: dict) -> None:
        """Save prediction statistics to CSV file"""
        csv_file = 'prediction_statistics.csv'
        
        # Extract parameters, actual values, and prediction time from the prediction table
        table_data = user_info.get('prediction_table_data', [])
        parameters = []
        actual_values = []
        prediction_times = []
        
        if len(table_data) >= 2:  # Check if we have prediction data
            # Get actual values (first row)
            actual_row = table_data[0]
            # Get predictions (second row)
            prediction_row = table_data[1]
            
            # Get times from the DataFrame
            time_col = df.get_column('time')
            if time_col.dtype == pl.String:
                times = time_col.to_list()
            else:
                times = time_col.dt.strftime('%Y-%m-%d %H:%M:%S').to_list()
            
            # Group values by time point
            for i in range(len(df)):
                time_key = f't{i}'
                if time_key in prediction_row and prediction_row[time_key] != '-':
                    # Add prediction
                    parameters.append(prediction_row[time_key])
                    # Add corresponding actual value
                    actual_values.append(actual_row[time_key])
                    # Add corresponding time
                    prediction_times.append(times[i])
        
        # Get age and user_id from DataFrame
        age = df.get_column('age')[0] if 'age' in df.columns else 0
        user_id = df.get_column('user_id')[0] if 'user_id' in df.columns else 1
        
        # Prepare the data dictionary
        data = {
            'id': str(uuid.uuid4()),
            'number': self._get_next_number(),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'email': user_info.get('email', ''),
            'age': age,
            'user_id': user_id,
            'gender': user_info.get('gender', ''),
            'diabetic': user_info.get('diabetic', ''),
            'diabetic_type': user_info.get('diabetic_type', ''),
            'diabetes_duration': user_info.get('diabetes_duration', ''),
            'other_medical_conditions': user_info.get('other_medical_conditions', ''),
            'medical_conditions_description': user_info.get('medical_conditions_input', ''),
            'location': user_info.get('location', ''),
            'parameters': str(parameters),  # Convert list to string
            'actual_values': str(actual_values),  # Convert list to string
            'prediction_time': str(prediction_times)  # Convert list to string
        }
        
        # Write to CSV
        file_exists = os.path.exists(csv_file)
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)

        # Reset the prediction
        df = df.with_columns(pl.lit(0.0).alias("prediction")) 