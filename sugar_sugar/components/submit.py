from typing import Any, Optional
from dash import html, dcc, Dash, Output, Input, State
import dash_bootstrap_components as dbc
import polars as pl
from datetime import datetime
import uuid
import csv
from pathlib import Path
from sugar_sugar.config import PREDICTION_HOUR_OFFSET

class SubmitComponent(html.Div):
    def __init__(self) -> None:
        self._stats_csv_path = (
            Path(__file__).resolve().parents[2]
            / 'data'
            / 'input'
            / 'prediction_statistics.csv'
        )
        self._stats_csv_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_csv_path = Path(__file__).resolve().parents[2] / 'prediction_statistics.csv'
        if legacy_csv_path.exists() and not self._stats_csv_path.exists():
            legacy_csv_path.replace(self._stats_csv_path)
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
            dbc.Button(
                "Finish / Exit",
                id="finish-study-button",
                color="secondary",
                className="mt-3",
                style={
                    'width': '300px',
                    'fontSize': '18px',
                    'padding': '12px 0',
                    'textAlign': 'center',
                    'verticalAlign': 'middle',
                    'lineHeight': '1.5',
                    'height': '50px'
                }
            ),
            dcc.Store(id='prediction-stats-store', data=None)
        ], style={'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center', 'alignItems': 'center'})

    def _get_next_number(self) -> int:
        """Get the next number for the prediction statistics."""
        csv_file_path = self._stats_csv_path
        if not csv_file_path.exists():
            return 0
        
        try:
            with csv_file_path.open('r', newline='') as file_handle:
                reader = csv.DictReader(file_handle)
                numbers = [int(row['number']) for row in reader if row['number'].isdigit()]
                return max(numbers) + 1 if numbers else 0
        except Exception:
            return 0

    def save_statistics(self, df: pl.DataFrame, user_info: dict[str, Any]) -> None:
        """Save prediction statistics to CSV file.

        This writes a single row for the whole "study entry".
        If `user_info["rounds"]` is present, statistics are aggregated across rounds.
        """
        csv_file_path = self._stats_csv_path
        
        rounds: list[dict[str, Any]] = user_info.get('rounds') or []
        parameters: list[dict[str, Any]] = []
        actual_values: list[dict[str, Any]] = []
        prediction_times: list[dict[str, Any]] = []

        def _time_list(window_df: pl.DataFrame) -> list[str]:
            time_col = window_df.get_column('time')
            if time_col.dtype == pl.String:
                return [str(t) for t in time_col.to_list()]
            return time_col.dt.strftime('%Y-%m-%d %H:%M:%S').to_list()

        if rounds:
            # Aggregate across played rounds
            for round_idx, round_info in enumerate(rounds, start=1):
                table_data = round_info.get('prediction_table_data') or []
                if len(table_data) < 2:
                    continue

                window_start = int(round_info.get('prediction_window_start') or 0)
                window_size = int(round_info.get('prediction_window_size') or 0)
                if window_size <= 0:
                    continue

                max_start = max(0, len(df) - window_size)
                safe_start = max(0, min(window_start, max_start))
                window_df = df.slice(safe_start, window_size)

                actual_row = table_data[0]
                prediction_row = table_data[1]
                times = _time_list(window_df)

                for i in range(window_size):
                    time_key = f"t{i}"
                    pred_str = prediction_row.get(time_key, "-")
                    act_str = actual_row.get(time_key, "-")
                    if pred_str != "-" and act_str != "-" and i < len(times):
                        parameters.append({"round": round_idx, "value": pred_str})
                        actual_values.append({"round": round_idx, "value": act_str})
                        prediction_times.append({"round": round_idx, "time": times[i]})
        else:
            # Backwards-compatible single-round behavior (still a single row)
            table_data = user_info.get('prediction_table_data', []) or []
            if len(table_data) >= 2:
                actual_row = table_data[0]
                prediction_row = table_data[1]
                times = _time_list(df)

                for i in range(len(df)):
                    time_key = f"t{i}"
                    pred_str = prediction_row.get(time_key, "-")
                    act_str = actual_row.get(time_key, "-")
                    if pred_str != "-" and act_str != "-" and i < len(times):
                        parameters.append({"round": 1, "value": pred_str})
                        actual_values.append({"round": 1, "value": act_str})
                        prediction_times.append({"round": 1, "time": times[i]})
        
        # Get age and user_id from DataFrame
        age = df.get_column('age')[0] if 'age' in df.columns else 0
        user_id = df.get_column('user_id')[0] if 'user_id' in df.columns else 1
        
        # Prepare the data dictionary
        data = {
            'id': str(uuid.uuid4()),
            'number': self._get_next_number(),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'email': user_info.get('email', ''),
            'is_example_data': bool(user_info.get('is_example_data', True)),
            'data_source_name': str(user_info.get('data_source_name', 'example.csv')),
            'age': age,
            'user_id': user_id,
            'gender': user_info.get('gender', ''),
            'diabetic': user_info.get('diabetic', ''),
            'diabetic_type': user_info.get('diabetic_type', ''),
            'diabetes_duration': user_info.get('diabetes_duration', ''),
            'other_medical_conditions': user_info.get('other_medical_conditions', ''),
            'medical_conditions_description': user_info.get('medical_conditions_input', ''),
            'location': user_info.get('location', ''),
            # Clear naming: "real" == ground truth, "predicted" == user prediction
            'predicted_values': str(parameters),
            'real_values': str(actual_values),
            'prediction_times': str(prediction_times)
        }
        
        # Write to CSV
        file_exists = csv_file_path.exists()
        desired_fieldnames = list(data.keys())

        if file_exists:
            # If the file exists but header is missing new columns, upgrade it in place.
            with csv_file_path.open('r', newline='') as file_handle:
                reader = csv.DictReader(file_handle)
                existing_fieldnames = reader.fieldnames or []
                existing_rows = list(reader)

            legacy_to_new = {
                'parameters': 'predicted_values',
                'actual_values': 'real_values',
                'prediction_time': 'prediction_times',
            }

            # If we have legacy columns, rewrite with the new names (and keep other columns).
            needs_upgrade = (
                any(field in existing_fieldnames for field in legacy_to_new.keys())
                or any(field not in existing_fieldnames for field in desired_fieldnames)
            )

            if needs_upgrade:
                # Keep all existing non-legacy fields, then enforce desired new schema ordering.
                preserved_existing = [f for f in existing_fieldnames if f and f not in legacy_to_new.keys()]
                upgraded_fieldnames = []
                for f in preserved_existing + desired_fieldnames:
                    if f not in upgraded_fieldnames:
                        upgraded_fieldnames.append(f)

                tmp_path = csv_file_path.with_suffix('.tmp')
                with tmp_path.open('w', newline='') as out_handle:
                    writer = csv.DictWriter(out_handle, fieldnames=upgraded_fieldnames)
                    writer.writeheader()
                    for row in existing_rows:
                        upgraded_row: dict[str, Any] = {key: row.get(key, "") for key in upgraded_fieldnames}
                        for old_key, new_key in legacy_to_new.items():
                            if upgraded_row.get(new_key, "") in ("", None) and old_key in row:
                                upgraded_row[new_key] = row.get(old_key, "")
                        writer.writerow(upgraded_row)

                tmp_path.replace(csv_file_path)

        # Append row (header is guaranteed to be present now)
        with csv_file_path.open('a', newline='') as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=desired_fieldnames)
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