from typing import Optional, Tuple, Any
from dash import html, dash_table, dcc, Output, Input
import polars as pl
import dash


TableData = list[dict[str, str]]
GLUCOSE_MGDL_PER_MMOLL: float = 18.0

class PredictionTableComponent(html.Div):
    def __init__(self) -> None:
        # Create the layout with session storage and initial empty table
        super().__init__(
            children=[
                dcc.Store(id='current-df-store', data=None),  # Session storage for current DataFrame
                html.H4("Predictions Table", style={'fontSize': '20px', 'marginBottom': '10px'}),
                dash_table.DataTable(
                    id='prediction-table-data',
                    data=[],  # Start empty - will be populated by callbacks
                    columns=[],  # Start empty - will be populated by callbacks
                    style_table={
                        'width': '100%',
                        'height': 'auto',
                        'maxHeight': 'clamp(300px, 40vh, 500px)',  # Responsive max height
                        'overflowY': 'auto',  # Allow vertical scroll if needed
                        'overflowX': 'auto',  # Allow horizontal scroll for small screens
                        'tableLayout': 'fixed'  # Fixed layout for equal column distribution
                    },
                    style_cell={
                        'textAlign': 'center',
                        'padding': 'clamp(2px, 1vw, 4px) clamp(1px, 0.5vw, 2px)',  # Responsive padding
                        'fontSize': 'clamp(8px, 1.5vw, 12px)',  # Responsive font size
                        'whiteSpace': 'nowrap',  # Prevent text wrapping to maintain compact layout
                        'overflow': 'hidden',
                        'textOverflow': 'ellipsis',
                        'lineHeight': '1.2',
                        'minWidth': '40px'  # Minimum width for readability
                    },
                    style_cell_conditional=[],  # Will be set dynamically by callback
                    style_header={
                        'backgroundColor': '#f8fafc',
                        'fontWeight': 'bold',
                        'fontSize': 'clamp(8px, 1.5vw, 12px)',  # Responsive font size
                        'padding': 'clamp(4px, 1vw, 6px) clamp(2px, 0.5vw, 4px)',  # Responsive padding
                        'textAlign': 'center',
                        'whiteSpace': 'normal',
                        'height': 'auto',
                        'minWidth': '40px'  # Minimum width for readability
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 0},
                            'backgroundColor': 'rgba(200, 240, 200, 0.5)'
                        },
                        {
                            'if': {'row_index': 1},
                            'backgroundColor': 'rgba(255, 200, 200, 0.5)'
                        }
                    ]
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
            [Output('prediction-table-data', 'data'),
             Output('prediction-table-data', 'columns'),
             Output('prediction-table-data', 'style_cell_conditional')],
            [Input('current-df-store', 'data'),
             Input('glucose-unit', 'data')]
        )
        def update_table(df_data: Optional[dict], glucose_unit: Optional[str]) -> Tuple[TableData, list[dict], list[dict]]:
            """Updates the predictions table based on the stored DataFrame state."""
            if not df_data:
                return [], [], []
            
            # Reconstruct DataFrame from stored data
            df = self._reconstruct_dataframe_from_dict(df_data)
            
            # Generate table data
            unit = glucose_unit if glucose_unit in ('mg/dL', 'mmol/L') else 'mg/dL'
            table_data = self._generate_table_data(df, unit)
            
            # Generate columns configuration with dynamic widths
            columns = [{'name': 'Metric', 'id': 'metric'}]
            num_data_columns = len(df)
            # Calculate width for data columns (75% total width divided by number of columns)
            data_column_width = f"{75 / num_data_columns}%" if num_data_columns > 0 else "75%"
            
            # Create style conditional for data columns
            style_cell_conditional = [
                {
                    'if': {'column_id': 'metric'},
                    'textAlign': 'left',
                    'fontWeight': 'bold',
                    'width': '25%',  # Fixed percentage width for metric column
                    'backgroundColor': '#f8fafc',
                    'whiteSpace': 'nowrap',
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis'
                }
            ]
            
            for i in range(num_data_columns):
                # Use shorter column names for better fit - show time index
                columns.append({
                    'name': f'T{i}', 
                    'id': f't{i}',
                    'type': 'text'
                })
                
                # Add width styling for each data column
                style_cell_conditional.append({
                    'if': {'column_id': f't{i}'},
                    'width': data_column_width
                })
            
            return table_data, columns, style_cell_conditional

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

 