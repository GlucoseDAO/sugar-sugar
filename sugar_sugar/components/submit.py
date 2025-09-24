from typing import Any, Optional
from dash import html, dcc, Dash, Output, Input, State
import dash_bootstrap_components as dbc
import polars as pl
from datetime import datetime
import uuid
import csv
import os
from sugar_sugar.config import PREDICTION_HOUR_OFFSET

class SubmitComponent(html.Div):
    def __init__(self) -> None:
        super().__init__([
            html.Div(
                id="prediction-progress-label",
                children="Make predictions to the end of the hidden area to submit",
                style={
                    'textAlign': 'center',
                    'marginBottom': '10px',
                    'fontSize': '16px',
                    'color': '#6c757d',
                    'fontStyle': 'italic'
                }
            ),
            dbc.Button(
                "Submit",
                id="submit-button",
                color="primary",
                className="mt-4",
                disabled=True,  # Start disabled
                style={'width': '300px', 'fontSize': '25px', 'padding': '15px 0', 'textAlign': 'center', 'verticalAlign': 'middle', 'lineHeight': '1.5', 'height': '60px'}
            ),
            dcc.Store(id='prediction-stats-store', data=None)
        ], style={'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center', 'alignItems': 'center'})

    def _get_next_number(self) -> int:
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

    def save_statistics(self, df: pl.DataFrame, user_info: dict[str, Any]) -> None:
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

    def register_callbacks(self, app: Dash) -> None:
        """Register callbacks for the submit component"""
        
        @app.callback(
            [Output('submit-button', 'disabled'),
             Output('submit-button', 'style'),
             Output('prediction-progress-label', 'children'),
             Output('prediction-progress-label', 'style')],
            [Input('current-window-df', 'data')],
            prevent_initial_call=False
        )
        def update_submit_button_state(df_data: Optional[dict[str, Any]]) -> tuple[bool, dict[str, str], str, dict[str, str]]:
            """Enable submit button only when there are predictions to the end of the hidden area"""
            base_style = {
                'width': '300px', 
                'fontSize': '25px', 
                'padding': '15px 0', 
                'textAlign': 'center', 
                'verticalAlign': 'middle', 
                'lineHeight': '1.5', 
                'height': '60px'
            }
            
            base_label_style = {
                'textAlign': 'center',
                'marginBottom': '10px',
                'fontSize': '16px',
                'fontStyle': 'italic'
            }
            
            if not df_data:
                # No data, keep disabled with gray style
                disabled_style = {**base_style, 'backgroundColor': '#cccccc', 'color': '#666666', 'cursor': 'not-allowed'}
                label_style = {**base_label_style, 'color': '#6c757d'}
                return True, disabled_style, "Make predictions to the end of the hidden area to submit", label_style
            
            # Reconstruct DataFrame to check for predictions
            df = self._reconstruct_dataframe_from_dict(df_data)
            
            # Check predictions in the hidden area (last PREDICTION_HOUR_OFFSET points)
            visible_points = len(df) - PREDICTION_HOUR_OFFSET
            hidden_area_df = df.slice(visible_points, PREDICTION_HOUR_OFFSET)
            
            # Find the last time point with a prediction
            predictions_mask = hidden_area_df.get_column("prediction") != 0.0
            if predictions_mask.any():
                # Get indices of predictions in hidden area
                prediction_indices = [i for i, has_pred in enumerate(predictions_mask) if has_pred]
                last_prediction_idx = max(prediction_indices)
                total_hidden_points = len(hidden_area_df)
                
                # Check if predictions go to the end (must reach the actual end)
                predictions_to_end = last_prediction_idx >= total_hidden_points - 1
                
                # Check if first point is auto-snapped to ground truth
                first_point_is_snapped = False
                if len(prediction_indices) > 0 and prediction_indices[0] == 0:
                    # First prediction point exists - check if it matches ground truth (auto-snapped)
                    first_prediction_value = hidden_area_df.get_column("prediction")[0]
                    first_ground_truth_value = hidden_area_df.get_column("gl")[0]
                    # Allow small floating point tolerance
                    if abs(first_prediction_value - first_ground_truth_value) < 0.01:
                        first_point_is_snapped = True
                
                # Calculate user-made predictions (excluding auto-snapped first point)
                user_predictions_count = len(prediction_indices)
                if first_point_is_snapped:
                    user_predictions_count -= 1
                
                # Required predictions is always the full hidden area
                required_user_predictions = total_hidden_points
                
                # Debug output
                print(f"DEBUG: Prediction count - total_hidden_points: {total_hidden_points}, prediction_indices: {prediction_indices}")
                print(f"DEBUG: first_point_is_snapped: {first_point_is_snapped}, user_predictions_count: {user_predictions_count}, required: {required_user_predictions}")
                
                if predictions_to_end:
                    # Enable button - predictions reach the end
                    enabled_style = {**base_style, 'backgroundColor': '#28a745', 'color': 'white', 'cursor': 'pointer'}
                    label_style = {**base_label_style, 'color': '#28a745', 'fontWeight': 'bold'}
                    return False, enabled_style, "✓ Ready to submit!", label_style
                else:
                    # Some predictions but not to the end
                    disabled_style = {**base_style, 'backgroundColor': '#ffc107', 'color': '#212529', 'cursor': 'not-allowed'}
                    label_style = {**base_label_style, 'color': '#856404'}
                    status_text = f"Continue predictions to the end ({user_predictions_count}/{required_user_predictions} points)"
                    return True, disabled_style, status_text, label_style
            else:
                # No predictions in hidden area
                disabled_style = {**base_style, 'backgroundColor': '#cccccc', 'color': '#666666', 'cursor': 'not-allowed'}
                label_style = {**base_label_style, 'color': '#6c757d'}
                return True, disabled_style, "Make predictions in the hidden area to submit", label_style

    def _reconstruct_dataframe_from_dict(self, df_data: dict[str, list[Any]]) -> pl.DataFrame:
        """Reconstruct a Polars DataFrame from stored dictionary data"""
        return pl.DataFrame({
            'time': pl.Series(df_data['time']).str.strptime(pl.Datetime, format='%Y-%m-%dT%H:%M:%S'),
            'gl': pl.Series(df_data['gl'], dtype=pl.Float64),
            'prediction': pl.Series(df_data['prediction'], dtype=pl.Float64),
            'age': pl.Series([int(float(x)) for x in df_data['age']], dtype=pl.Int64),
            'user_id': pl.Series([int(float(x)) for x in df_data['user_id']], dtype=pl.Int64)
        }) 