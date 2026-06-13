from typing import Optional, Any
from dash import html, dcc, Output, Input
import polars as pl
import dash
from sugar_sugar.components.ag_grid import build_readonly_ag_grid, build_readonly_column_defs
from sugar_sugar.config import STORAGE_TYPE


TableData = list[dict[str, str]]
GLUCOSE_MGDL_PER_MMOLL: float = 18.0

class PredictionTableComponent(html.Div):
    def __init__(self) -> None:
        # Create the layout with session storage and initial empty table
        super().__init__(
            children=[
                dcc.Store(id='current-df-store', data=None, storage_type=STORAGE_TYPE),
                html.H4("Predictions Table", style={'fontSize': '20px', 'marginBottom': '10px'}),
                build_readonly_ag_grid(
                    table_id='prediction-table-data',
                    row_data=[],
                    column_defs=[],
                    style={
                        'width': '100%',
                        'height': 'auto',
                        'maxHeight': 'clamp(300px, 40vh, 500px)',  # Responsive max height
                        'overflowY': 'auto',  # Allow vertical scroll if needed
                        'overflowX': 'auto',  # Allow horizontal scroll for small screens
                        'tableLayout': 'fixed'  # Fixed layout for equal column distribution
                    },
                    highlight_first_two_rows=True,
                )
            ],
            style={
                'padding': 'clamp(10px, 2vw, 20px)',  # Responsive padding
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'marginBottom': '20px',
                'width': '100%',
                'maxWidth': '100%',
                'display': 'flex',
                'flexDirection': 'column',
                'flex': '1',  # Allow the table container to grow within the parent flexbox
                'minHeight': '0',  # Allow shrinking if needed
                'boxSizing': 'border-box',
                'overflowX': 'auto'  # Allow horizontal scroll on small screens
            }
        )

    def register_callbacks(self, app: dash.Dash) -> None:
        """Register all prediction table related callbacks"""
        
        @app.callback(
            Output('current-df-store', 'data'),
            [Input('current-window-df', 'data')]  # Listen to session storage instead
        )
        def store_current_df(df_data: Optional[dict]) -> Optional[dict]:
            """Store the current DataFrame state when it changes"""
            # Just pass through the session storage data
            return df_data

        @app.callback(
            [Output('prediction-table-data', 'rowData'),
             Output('prediction-table-data', 'columnDefs')],
            [Input('current-df-store', 'data'),
             Input('glucose-unit', 'data')]
        )
        def update_table(
            df_data: Optional[dict],
            glucose_unit: Optional[str],
        ) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
            """Updates the predictions table based on the stored DataFrame state."""
            if not df_data:
                return [], []
            
            # Reconstruct DataFrame from stored data
            df = self._reconstruct_dataframe_from_dict(df_data)
            
            # Generate table data
            unit = glucose_unit if glucose_unit in ('mg/dL', 'mmol/L') else 'mg/dL'
            table_data = self._generate_table_data(df, unit)
            
            # Generate columns configuration with dynamic widths
            columns = [{'name': 'Metric', 'id': 'metric'}]
            num_data_columns = len(df)
            for i in range(num_data_columns):
                # Use shorter column names for better fit - show time index
                columns.append({
                    'name': f'T{i}', 
                    'id': f't{i}',
                    'type': 'text'
                })
            
            return table_data, build_readonly_column_defs(columns)

    def _reconstruct_dataframe_from_dict(self, df_data: dict[str, list[Any]]) -> pl.DataFrame:
        """Reconstruct a Polars DataFrame from stored dictionary data"""
        return pl.DataFrame({
            'time': pl.Series(df_data['time']).str.strptime(pl.Datetime, format='%Y-%m-%dT%H:%M:%S'),
            'gl': pl.Series(df_data['gl'], dtype=pl.Float64),
            'prediction': pl.Series(df_data['prediction'], dtype=pl.Float64),
            'age': pl.Series([int(float(x)) for x in df_data['age']], dtype=pl.Int64),
            'user_id': pl.Series([int(float(x)) for x in df_data['user_id']], dtype=pl.Int64)
        })

    def _generate_table_data(self, df: pl.DataFrame, glucose_unit: str = 'mg/dL') -> TableData:
        """Generates the table data with actual values, predictions, and errors (display-only units)."""
        factor = 1.0 / GLUCOSE_MGDL_PER_MMOLL if glucose_unit == 'mmol/L' else 1.0
        n = len(df)

        actual_raw = df.get_column("gl").to_list()
        pred_raw = df.get_column("prediction").to_list()

        actual_mg: list[Optional[float]] = [None if x is None else float(x) for x in actual_raw]
        pred_col: list[Optional[float]] = [None if x is None else float(x) for x in pred_raw]

        # Build predicted values in mg/dL, with interpolation inside the drawn segment.
        pred_mg: list[Optional[float]] = [None] * n
        non_zero_indices = [i for i, p in enumerate(pred_col) if (p is not None and p != 0.0)]
        if len(non_zero_indices) >= 2:
            start_idx = non_zero_indices[0]
            end_idx = non_zero_indices[-1]
            for i in range(n):
                if i < start_idx or i > end_idx:
                    pred_mg[i] = None
                elif pred_col[i] is not None and pred_col[i] != 0.0:
                    pred_mg[i] = pred_col[i]
                else:
                    prev_idx = max(j for j in non_zero_indices if j < i)
                    next_idx = min(j for j in non_zero_indices if j > i)
                    total_steps = next_idx - prev_idx
                    current_step = i - prev_idx
                    prev_val = pred_col[prev_idx]
                    next_val = pred_col[next_idx]
                    if prev_val is None or next_val is None:
                        pred_mg[i] = None
                    else:
                        pred_mg[i] = prev_val + (next_val - prev_val) * (current_step / total_steps)
        else:
            for i in range(n):
                p = pred_col[i]
                pred_mg[i] = p if (p is not None and p != 0.0) else None

        # Rows
        glucose_row: dict[str, str] = {'metric': 'Actual Glucose'}
        prediction_row: dict[str, str] = {'metric': 'Predicted'}
        abs_error_row: dict[str, str] = {'metric': 'Absolute Error'}
        rel_error_row: dict[str, str] = {'metric': 'Relative Error (%)'}

        for i in range(n):
            a = actual_mg[i]
            p = pred_mg[i]

            glucose_row[f"t{i}"] = "-" if a is None else f"{a * factor:.1f}"
            if p is None or a is None:
                prediction_row[f"t{i}"] = "-"
                abs_error_row[f"t{i}"] = "-"
                rel_error_row[f"t{i}"] = "-"
            else:
                prediction_row[f"t{i}"] = f"{p * factor:.1f}"
                err = abs(a - p)
                abs_error_row[f"t{i}"] = f"{err * factor:.1f}"
                rel_error_row[f"t{i}"] = f"{(err / a * 100):.1f}%" if a != 0 else "-"

        return [glucose_row, prediction_row, abs_error_row, rel_error_row]

 