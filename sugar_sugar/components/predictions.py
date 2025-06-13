from typing import List, Dict, Optional
from dash import html, dash_table, dcc, Output, Input
import polars as pl
import dash


TableData = List[Dict[str, str]]

class PredictionTableComponent(html.Div):
    def __init__(self):
        # Create the layout with session storage and initial empty table
        super().__init__(
            children=[
                dcc.Store(id='current-df-store', data=None),  # Session storage for current DataFrame
                html.H4("Predictions Table", style={'fontSize': '20px', 'marginBottom': '10px'}),
                dash_table.DataTable(
                    id='prediction-table-data',
                    data=[],  # Start empty - will be populated by callbacks
                    columns=[],  # Start empty - will be populated by callbacks
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'textAlign': 'center',
                        'padding': '5px',
                        'minWidth': '70px'
                    },
                    style_header={
                        'backgroundColor': '#f8fafc',
                        'fontWeight': 'bold'
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
                'padding': '20px',
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'marginBottom': '20px'
            }
        )

    def register_callbacks(self, app: dash.Dash) -> None:
        """Register all prediction table related callbacks"""
        
        @app.callback(
            Output('current-df-store', 'data'),
            [Input('last-click-time', 'data')]
        )
        def store_current_df(last_click_time: int) -> Optional[Dict]:
            """Store the current DataFrame state when it changes"""
            from sugar_sugar.app import df  # Import the global df
            
            if df is None or df.height == 0:
                return None
                
            # Convert DataFrame to storable format
            return {
                'time': [t.isoformat() for t in df.get_column("time")],
                'gl': df.get_column("gl").to_list(),
                'prediction': df.get_column("prediction").to_list(),
                'age': df.get_column("age").to_list(),
                'user_id': df.get_column("user_id").to_list()
            }

        @app.callback(
            [Output('prediction-table-data', 'data'),
             Output('prediction-table-data', 'columns')],
            [Input('current-df-store', 'data')]
        )
        def update_table(df_data: Optional[Dict]) -> tuple[TableData, List[Dict]]:
            """Updates the predictions table based on the stored DataFrame state."""
            if not df_data:
                return [], []
            
            # Reconstruct DataFrame from stored data
            df = self._reconstruct_dataframe_from_dict(df_data)
            
            # Generate table data
            table_data = self._generate_table_data(df)
            
            # Generate columns configuration
            columns = [{'name': 'Metric', 'id': 'metric'}]
            for i in range(len(df)):
                columns.append({'name': f'T{i}', 'id': f't{i}'})
            
            return table_data, columns

    def _reconstruct_dataframe_from_dict(self, df_data: Dict) -> pl.DataFrame:
        """Reconstruct a Polars DataFrame from stored dictionary data"""
        return pl.DataFrame({
            'time': pl.Series(df_data['time']).str.strptime(pl.Datetime, format='%Y-%m-%dT%H:%M:%S'),
            'gl': pl.Series(df_data['gl'], dtype=pl.Float64),
            'prediction': pl.Series(df_data['prediction'], dtype=pl.Float64),
            'age': pl.Series([int(float(x)) for x in df_data['age']], dtype=pl.Int64),
            'user_id': pl.Series([int(float(x)) for x in df_data['user_id']], dtype=pl.Int64)
        })

    def _generate_table_data(self, df: pl.DataFrame) -> TableData:
        """Generates the table data with actual values, predictions, and errors."""
        table_data = []
        
        # Row 1: Actual glucose values
        glucose_row = {'metric': 'Actual Glucose'}
        for i, gl in enumerate(df.get_column("gl")):
            glucose_row[f't{i}'] = f"{gl:.1f}" if gl is not None else "-"
        table_data.append(glucose_row)
        
        # Row 2: Predicted values with interpolation
        prediction_row = {'metric': 'Predicted'}
        predictions = df.get_column("prediction")
        non_zero_indices = [i for i, p in enumerate(predictions) if p != 0]
        
        if len(non_zero_indices) >= 2:
            start_idx = non_zero_indices[0]
            end_idx = non_zero_indices[-1]
            
            for i in range(len(predictions)):
                if i < start_idx or i > end_idx:
                    prediction_row[f't{i}'] = "-"
                elif predictions[i] != 0:
                    prediction_row[f't{i}'] = f"{predictions[i]:.1f}"
                else:
                    prev_idx = max([j for j in non_zero_indices if j < i])
                    next_idx = min([j for j in non_zero_indices if j > i])
                    total_steps = next_idx - prev_idx
                    current_step = i - prev_idx
                    prev_val = predictions[prev_idx]
                    next_val = predictions[next_idx]
                    interpolated = prev_val + (next_val - prev_val) * (current_step / total_steps)
                    prediction_row[f't{i}'] = f"{interpolated:.1f}"
        else:
            for i, pred_val in enumerate(predictions):
                prediction_row[f't{i}'] = f"{pred_val:.1f}" if pred_val != 0 else "-"
        
        table_data.append(prediction_row)
        
        # Add error rows
        table_data.extend(self._calculate_error_rows(df, prediction_row))
        
        return table_data

    def _calculate_error_rows(self, df: pl.DataFrame, prediction_row: Dict[str, str]) -> List[Dict[str, str]]:
        """Calculates absolute and relative error rows for the table."""
        error_rows = []
        
        # Absolute Error
        error_row = {'metric': 'Absolute Error'}
        for i, gl in enumerate(df.get_column("gl")):
            pred_str = prediction_row[f't{i}']
            if pred_str != "-" and gl is not None:
                pred = float(pred_str)
                error = abs(gl - pred)
                error_row[f't{i}'] = f"{error:.1f}"
            else:
                error_row[f't{i}'] = "-"
        error_rows.append(error_row)
        
        # Relative Error
        rel_error_row = {'metric': 'Relative Error (%)'}
        for i, gl in enumerate(df.get_column("gl")):
            pred_str = prediction_row[f't{i}']
            if pred_str != "-" and gl is not None and gl != 0:
                pred = float(pred_str)
                rel_error = (abs(gl - pred) / gl * 100)
                rel_error_row[f't{i}'] = f"{rel_error:.1f}%"
            else:
                rel_error_row[f't{i}'] = "-"
        error_rows.append(rel_error_row)
        
        return error_rows

    # Keep these methods for backward compatibility with metrics component
    def update_dataframe(self, df: pl.DataFrame):
        """Backward compatibility method - now handled by session storage"""
        pass

    def generate_table_data(self) -> TableData:
        """Backward compatibility method that returns the last generated table data"""
        # This is used by the metrics component, so we need to provide access to the data
        from sugar_sugar.app import df
        if df is None:
            return []
        return self._generate_table_data(df) 