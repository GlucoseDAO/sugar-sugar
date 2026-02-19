from typing import Any, Optional
from dash import html, dcc, Dash, Output, Input, State
import dash_bootstrap_components as dbc
import polars as pl
from datetime import datetime
import uuid
import csv
from pathlib import Path
from sugar_sugar.config import PREDICTION_HOUR_OFFSET
from sugar_sugar.components.metrics import MetricsComponent
from sugar_sugar.i18n import t, normalize_locale

class SubmitComponent(html.Div):
    def __init__(self, *, locale: str = "en") -> None:
        self._locale: str = normalize_locale(locale)
        self._stats_csv_path = (
            Path(__file__).resolve().parents[2]
            / 'data'
            / 'input'
            / 'prediction_statistics.csv'
        )
        # Overall ranking: written only after participant completed all eligible formats.
        self._ranking_csv_path = (
            Path(__file__).resolve().parents[2]
            / 'data'
            / 'input'
            / 'prediction_ranking.csv'
        )
        ranking_dir = self._stats_csv_path.parent
        self._ranking_by_format_paths: dict[str, Path] = {
            "A": ranking_dir / "prediction_ranking_A.csv",
            "B": ranking_dir / "prediction_ranking_B.csv",
            "C": ranking_dir / "prediction_ranking_C.csv",
        }
        self._stats_csv_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_csv_path = Path(__file__).resolve().parents[2] / 'prediction_statistics.csv'
        if legacy_csv_path.exists() and not self._stats_csv_path.exists():
            legacy_csv_path.replace(self._stats_csv_path)
        self._repair_misaligned_csv_rows()
        super().__init__([
            html.Div(
                id="prediction-progress-label",
                children=t("ui.submit.progress_no_data", locale=self._locale),
                style={
                    'textAlign': 'center',
                    'marginBottom': '10px',
                    'fontSize': '16px',
                    'color': '#6c757d',
                    'fontStyle': 'italic'
                }
            ),
            dbc.Button(
                t("ui.submit.submit", locale=self._locale),
                id="submit-button",
                color="primary",
                className="mt-4",
                disabled=True,  # Start disabled
                style={'width': '300px', 'fontSize': '25px', 'padding': '15px 0', 'textAlign': 'center', 'verticalAlign': 'middle', 'lineHeight': '1.5', 'height': '60px'}
            ),
            dbc.Button(
                t("ui.common.finish_exit", locale=self._locale),
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

    def _repair_misaligned_csv_rows(self) -> None:
        """Repair CSV rows whose columns were written in desired_fieldnames order instead of
        the file's actual header order.  This happened when run_id and format were inserted
        near the start of the dict but appended to files that already had run_id/format at
        the end of the header.

        General algorithm: if the file's header is H and the values were written in order D
        (desired_names), then corrupted_row[H[i]] = actual value of D[i].
        To recover: correct_value_for_header_col H[k] = corrupted_row[ H[ D.index(H[k]) ] ].
        """
        # The desired dict key order that was used when the bug was present.
        ranking_desired: list[str] = [
            'study_id', 'run_id', 'number', 'timestamp', 'format', 'rounds_played',
            'is_example_data', 'data_source_name', 'overall_mae_mgdl', 'overall_mse_mgdl',
            'overall_rmse_mgdl', 'overall_mape_pct',
        ]
        stats_desired: list[str] = [
            'study_id', 'run_id', 'number', 'timestamp', 'email', 'format',
            'is_example_data', 'data_source_name', 'age', 'user_id', 'gender',
            'uses_cgm', 'cgm_duration_years', 'diabetic', 'diabetic_type',
            'diabetes_duration', 'location', 'rounds_played', 'predicted_values',
            'real_values', 'prediction_times', 'overall_mae_mgdl', 'overall_mse_mgdl',
            'overall_rmse_mgdl', 'overall_mape_pct', 'per_round_metrics',
        ]
        self._repair_csv(
            self._ranking_csv_path,
            desired_order=ranking_desired,
            corrupt_check_col='overall_mae_mgdl',
            corrupt_check=lambda v: not self._is_numeric(v),
        )
        for path in self._ranking_by_format_paths.values():
            self._repair_csv(
                path,
                desired_order=ranking_desired,
                corrupt_check_col='overall_mae_mgdl',
                corrupt_check=lambda v: not self._is_numeric(v),
            )
        self._repair_csv(
            self._stats_csv_path,
            desired_order=stats_desired,
            corrupt_check_col='timestamp',
            corrupt_check=lambda v: self._is_integer_string(v),
        )

    @staticmethod
    def _is_numeric(value: str) -> bool:
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _is_integer_string(value: str) -> bool:
        try:
            int(str(value).strip())
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _repair_csv(
        path: Path,
        desired_order: list[str],
        corrupt_check_col: str,
        corrupt_check: Any,
    ) -> None:
        """Rewrite corrupted rows in-place using the known desired write order."""
        if not path.exists():
            return
        with path.open('r', newline='') as fh:
            reader = csv.DictReader(fh)
            header = list(reader.fieldnames or [])
            rows = list(reader)
        if not header:
            return
        # Only attempt repair when the file already has the upgraded schema
        # (run_id and format present somewhere after the first column).
        if 'run_id' not in header or 'format' not in header:
            return

        # Build a lookup: desired column name → index in desired_order
        desired_index: dict[str, int] = {name: i for i, name in enumerate(desired_order)}

        repaired: list[dict[str, Any]] = []
        changed = False
        for row in rows:
            if not corrupt_check(row.get(corrupt_check_col, '')):
                repaired.append(row)
                continue

            # Recover correct values: correct_row[H[k]] = row[ H[ D.index(H[k]) ] ]
            fixed: dict[str, str] = {}
            for k, h_col in enumerate(header):
                d_idx = desired_index.get(h_col)
                if d_idx is not None and d_idx < len(header):
                    fixed[h_col] = row.get(header[d_idx], '')
                else:
                    fixed[h_col] = row.get(h_col, '')
            repaired.append(fixed)
            changed = True

        if not changed:
            return
        tmp_path = path.with_suffix('.tmp')
        with tmp_path.open('w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=header)
            writer.writeheader()
            for row in repaired:
                writer.writerow({k: row.get(k, '') for k in header})
        tmp_path.replace(path)

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
        version = str(user_info.get('run_format') or user_info.get('format') or '')

        # Stable ID across derived outputs (stats + ranking)
        study_id = user_info.get('study_id')
        if not study_id:
            study_id = str(uuid.uuid4())
            user_info['study_id'] = study_id

        metrics_component = MetricsComponent()

        def _metrics_from_table(table_data: list[dict[str, str]]) -> dict[str, Optional[float]]:
            metrics = metrics_component._calculate_metrics_from_table_data(table_data) if len(table_data) >= 2 else {}
            def _val(name: str) -> Optional[float]:
                item = metrics.get(name)
                if not item:
                    return None
                v = item.get('value')
                return float(v) if v is not None else None
            return {
                'mae': _val('MAE'),
                'mse': _val('MSE'),
                'rmse': _val('RMSE'),
                'mape': _val('MAPE'),
            }

        def _build_aggregate_table_data(rounds_in: list[dict[str, Any]]) -> list[dict[str, str]]:
            actual_row: dict[str, str] = {'metric': 'Actual Glucose'}
            pred_row: dict[str, str] = {'metric': 'Predicted'}
            out_idx = 0
            for round_info in rounds_in:
                table_data = round_info.get('prediction_table_data') or []
                if len(table_data) < 2:
                    continue
                round_actual = table_data[0]
                round_pred = table_data[1]
                i = 0
                while True:
                    key = f"t{i}"
                    if key not in round_actual or key not in round_pred:
                        break
                    actual_row[f"t{out_idx}"] = round_actual.get(key, "-")
                    pred_row[f"t{out_idx}"] = round_pred.get(key, "-")
                    out_idx += 1
                    i += 1
            return [actual_row, pred_row]

        def _time_list(window_df: pl.DataFrame) -> list[str]:
            time_col = window_df.get_column('time')
            if time_col.dtype == pl.String:
                return [str(t) for t in time_col.to_list()]
            return time_col.dt.strftime('%Y-%m-%d %H:%M:%S').to_list()

        # Per-round + overall metrics (computed in mg/dL, regardless of UI unit)
        per_round_metrics: list[dict[str, Any]] = []
        if rounds:
            for round_info in rounds:
                table_data = round_info.get('prediction_table_data') or []
                round_number = int(round_info.get('round_number') or (len(per_round_metrics) + 1))
                m = _metrics_from_table(table_data)
                per_round_metrics.append({
                    'round_number': round_number,
                    'mae': m['mae'],
                    'mse': m['mse'],
                    'rmse': m['rmse'],
                    'mape': m['mape'],
                })
        else:
            table_data = user_info.get('prediction_table_data', []) or []
            m = _metrics_from_table(table_data)
            per_round_metrics.append({
                'round_number': 1,
                'mae': m['mae'],
                'mse': m['mse'],
                'rmse': m['rmse'],
                'mape': m['mape'],
            })

        overall_table_data = _build_aggregate_table_data(rounds) if rounds else (user_info.get('prediction_table_data', []) or [])
        overall = _metrics_from_table(overall_table_data)

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
                        parameters.append({"version": version, "round": round_idx, "value": pred_str})
                        actual_values.append({"version": version, "round": round_idx, "value": act_str})
                        prediction_times.append({"version": version, "round": round_idx, "value": times[i]})
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
                        parameters.append({"version": version, "round": 1, "value": pred_str})
                        actual_values.append({"version": version, "round": 1, "value": act_str})
                        prediction_times.append({"version": version, "round": 1, "value": times[i]})
        
        # Get age and user_id from DataFrame
        age = df.get_column('age')[0] if 'age' in df.columns else 0
        user_id = df.get_column('user_id')[0] if 'user_id' in df.columns else 1
        
        # Prepare the data dictionary
        rounds_played = len(rounds) if rounds else 1
        number = user_info.get("number")
        if number is None or (isinstance(number, str) and number.strip() == ""):
            number = self._get_next_number()
            user_info["number"] = number
        data = {
            'study_id': study_id,
            'run_id': str(user_info.get('run_id') or ''),
            'number': number,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'email': user_info.get('email', ''),
            'format': version,
            'is_example_data': bool(user_info.get('is_example_data', True)),
            'data_source_name': str(user_info.get('data_source_name', 'example.csv')),
            'age': age,
            'user_id': user_id,
            'gender': user_info.get('gender', ''),
            'uses_cgm': bool(user_info.get('uses_cgm', False)),
            'cgm_duration_years': user_info.get('cgm_duration_years', ''),
            'diabetic': user_info.get('diabetic', ''),
            'diabetic_type': user_info.get('diabetic_type', ''),
            'diabetes_duration': user_info.get('diabetes_duration', ''),
            'location': user_info.get('location', ''),
            'rounds_played': rounds_played,
            # Clear naming: "real" == ground truth, "predicted" == user prediction
            'predicted_values': str(parameters),
            'real_values': str(actual_values),
            'prediction_times': str(prediction_times),
            # Metrics
            'overall_mae_mgdl': overall['mae'],
            'overall_mse_mgdl': overall['mse'],
            'overall_rmse_mgdl': overall['rmse'],
            'overall_mape_pct': overall['mape'],
            'per_round_metrics': str(per_round_metrics),
        }
        
        def _upgrade_and_append_csv(
            path: Path,
            row: dict[str, Any],
            legacy_to_new: dict[str, str]
        ) -> None:
            file_exists = path.exists()
            desired_fieldnames = list(row.keys())
            # Track the actual file header order so append uses the same column order.
            final_fieldnames = desired_fieldnames

            if file_exists:
                with path.open('r', newline='') as file_handle:
                    reader = csv.DictReader(file_handle)
                    existing_fieldnames = reader.fieldnames or []
                    existing_rows = list(reader)

                needs_upgrade = (
                    any(field in existing_fieldnames for field in legacy_to_new.keys())
                    or any(field not in existing_fieldnames for field in desired_fieldnames)
                )

                if needs_upgrade:
                    preserved_existing = [f for f in existing_fieldnames if f and f not in legacy_to_new.keys()]
                    upgraded_fieldnames: list[str] = []
                    for f in preserved_existing + desired_fieldnames:
                        if f not in upgraded_fieldnames:
                            upgraded_fieldnames.append(f)

                    tmp_path = path.with_suffix('.tmp')
                    with tmp_path.open('w', newline='') as out_handle:
                        writer = csv.DictWriter(out_handle, fieldnames=upgraded_fieldnames)
                        writer.writeheader()
                        for old_row in existing_rows:
                            upgraded_row: dict[str, Any] = {key: old_row.get(key, "") for key in upgraded_fieldnames}
                            for old_key, new_key in legacy_to_new.items():
                                if upgraded_row.get(new_key, "") in ("", None) and old_key in old_row:
                                    upgraded_row[new_key] = old_row.get(old_key, "")
                            writer.writerow(upgraded_row)
                    tmp_path.replace(path)
                    # Use the upgraded schema for the append so column order matches header.
                    final_fieldnames = upgraded_fieldnames
                else:
                    # No upgrade needed but must still append in the file's existing column order.
                    final_fieldnames = list(existing_fieldnames)

            with path.open('a', newline='') as file_handle:
                writer = csv.DictWriter(file_handle, fieldnames=final_fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)

        # Write full statistics row
        _upgrade_and_append_csv(
            csv_file_path,
            data,
            legacy_to_new={
                'id': 'study_id',
                'parameters': 'predicted_values',
                'actual_values': 'real_values',
                'prediction_time': 'prediction_times',
            }
        )

        # Write ranking row for fast leaderboard lookups
        ranking_row = {
            'study_id': study_id,
            'run_id': str(user_info.get('run_id') or ''),
            'number': data['number'],
            'timestamp': data['timestamp'],
            'format': data['format'],
            'rounds_played': rounds_played,
            'is_example_data': data['is_example_data'],
            'data_source_name': data['data_source_name'],
            'overall_mae_mgdl': overall['mae'],
            'overall_mse_mgdl': overall['mse'],
            'overall_rmse_mgdl': overall['rmse'],
            'overall_mape_pct': overall['mape'],
        }
        # Ranking within the current format/category (A, B, or C)
        if version in self._ranking_by_format_paths:
            _upgrade_and_append_csv(
                self._ranking_by_format_paths[version],
                ranking_row,
                legacy_to_new={},
            )

        # Overall (cumulative) ranking across formats played so far.
        uses_cgm = bool(user_info.get("uses_cgm", False))
        eligible_formats: set[str] = {"A"} | ({"B", "C"} if uses_cgm else set())
        runs_by_format: dict[str, list[dict[str, Any]]] = dict(user_info.get("runs_by_format") or {})
        played_formats: set[str] = {str(fmt) for fmt, runs in runs_by_format.items() if runs}
        if rounds_played > 0 and version:
            played_formats.add(version)
        all_rounds: list[dict[str, Any]] = []
        for fmt in sorted(played_formats):
            for run in (runs_by_format.get(fmt) or []):
                all_rounds.extend(list(run.get("rounds") or []))
        # Include the current run (not yet archived)
        all_rounds.extend(rounds if rounds else [])

        overall_all_table = _build_aggregate_table_data(all_rounds) if all_rounds else overall_table_data
        overall_all = _metrics_from_table(overall_all_table)

        source_names: set[str] = set()
        any_uploaded = False
        for fmt in played_formats:
            for run in (runs_by_format.get(fmt) or []):
                any_uploaded = any_uploaded or (not bool(run.get("is_example_data", True)))
                name = str(run.get("data_source_name") or "")
                if name:
                    source_names.add(name)
        any_uploaded = any_uploaded or (not bool(user_info.get("is_example_data", True)))
        current_source = str(user_info.get("data_source_name") or "")
        if current_source:
            source_names.add(current_source)

        data_source_name = next(iter(source_names)) if len(source_names) == 1 else "multiple"
        total_rounds_played = int(
            sum(len(list(run.get("rounds") or [])) for fmt in played_formats for run in (runs_by_format.get(fmt) or []))
            + (len(rounds) if rounds else 1)
        )

        overall_ranking_row = {
            'study_id': study_id,
            'run_id': str(uuid.uuid4()),
            'number': data['number'],
            'timestamp': data['timestamp'],
            'format': "ALL",
            'rounds_played': total_rounds_played,
            'is_example_data': (not any_uploaded),
            'data_source_name': data_source_name,
            'overall_mae_mgdl': overall_all['mae'],
            'overall_mse_mgdl': overall_all['mse'],
            'overall_rmse_mgdl': overall_all['rmse'],
            'overall_mape_pct': overall_all['mape'],
        }
        _upgrade_and_append_csv(
            self._ranking_csv_path,
            overall_ranking_row,
            legacy_to_new={},
        )

        # Reset the prediction
        df = df.with_columns(pl.lit(0.0).alias("prediction"))

    def register_callbacks(self, app: Dash) -> None:
        """Register callbacks for the submit component"""
        
        @app.callback(
            [Output('submit-button', 'disabled'),
             Output('submit-button', 'style'),
             Output('prediction-progress-label', 'children'),
             Output('prediction-progress-label', 'style')],
            [Input('current-window-df', 'data'),
             Input('interface-language', 'data')],
            prevent_initial_call=False
        )
        def update_submit_button_state(
            df_data: Optional[dict[str, Any]],
            interface_language: Optional[str],
        ) -> tuple[bool, dict[str, str], str, dict[str, str]]:
            """Enable submit button only when there are predictions to the end of the hidden area"""
            locale = normalize_locale(interface_language)
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
                return True, disabled_style, t("ui.submit.progress_no_data", locale=locale), label_style
            
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
                    return False, enabled_style, t("ui.submit.progress_ready", locale=locale), label_style
                else:
                    # Some predictions but not to the end
                    disabled_style = {**base_style, 'backgroundColor': '#ffc107', 'color': '#212529', 'cursor': 'not-allowed'}
                    label_style = {**base_label_style, 'color': '#856404'}
                    status_text = t(
                        "ui.submit.progress_some",
                        locale=locale,
                        done=user_predictions_count,
                        total=required_user_predictions,
                    )
                    return True, disabled_style, status_text, label_style
            else:
                # No predictions in hidden area
                disabled_style = {**base_style, 'backgroundColor': '#cccccc', 'color': '#666666', 'cursor': 'not-allowed'}
                label_style = {**base_label_style, 'color': '#6c757d'}
                return True, disabled_style, t("ui.submit.progress_hidden_area", locale=locale), label_style

    def _reconstruct_dataframe_from_dict(self, df_data: dict[str, list[Any]]) -> pl.DataFrame:
        """Reconstruct a Polars DataFrame from stored dictionary data"""
        return pl.DataFrame({
            'time': pl.Series(df_data['time']).str.strptime(pl.Datetime, format='%Y-%m-%dT%H:%M:%S'),
            'gl': pl.Series(df_data['gl'], dtype=pl.Float64),
            'prediction': pl.Series(df_data['prediction'], dtype=pl.Float64),
            'age': pl.Series([int(float(x)) for x in df_data['age']], dtype=pl.Int64),
            'user_id': pl.Series([int(float(x)) for x in df_data['user_id']], dtype=pl.Int64)
        }) 