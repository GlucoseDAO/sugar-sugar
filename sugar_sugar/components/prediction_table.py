from typing import List, Dict
import dash
from dash import html, dash_table
import polars as pl
from dash import Output, Input


TableData = List[Dict[str, str]]

class PredictionTableComponent(html.Div):
    def __init__(self, df: pl.DataFrame):
        self.df = df
        # Create the layout directly instead of storing it in a variable
        super().__init__(
            children=[
                html.H4("Predictions Table", style={'fontSize': '20px', 'marginBottom': '10px'}),
                dash_table.DataTable(
                    id='prediction-table-data',
                    data=self.generate_table_data(),
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

    def update_dataframe(self, df: pl.DataFrame):
        """Updates the component's DataFrame reference"""
        self.df = df

    def generate_table_data(self) -> TableData:
        """Generates the table data with actual values, predictions, and errors."""
        table_data = []
        
        # Row 1: Actual glucose values
        glucose_row = {'metric': 'Actual Glucose'}
        for i, gl in enumerate(self.df.get_column("gl")):
            glucose_row[f't{i}'] = f"{gl:.1f}" if gl is not None else "-"
        table_data.append(glucose_row)
        
        # Row 2: Predicted values with interpolation
        prediction_row = {'metric': 'Predicted'}
        predictions = self.df.get_column("prediction")
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
        table_data.extend(self._calculate_error_rows(prediction_row))
        
        return table_data

    def _calculate_error_rows(self, prediction_row: Dict[str, str]) -> List[Dict[str, str]]:
        """Calculates absolute and relative error rows for the table."""
        error_rows = []
        
        # Absolute Error
        error_row = {'metric': 'Absolute Error'}
        for i, gl in enumerate(self.df.get_column("gl")):
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
        for i, gl in enumerate(self.df.get_column("gl")):
            pred_str = prediction_row[f't{i}']
            if pred_str != "-" and gl is not None and gl != 0:
                pred = float(pred_str)
                rel_error = (abs(gl - pred) / gl * 100)
                rel_error_row[f't{i}'] = f"{rel_error:.1f}%"
            else:
                rel_error_row[f't{i}'] = "-"
        error_rows.append(rel_error_row)
        
        return error_rows

    def register_callbacks(self, app):
        @app.callback(
            [Output('prediction-table-data', 'data'),
             Output('prediction-table-data', 'columns')],
            [Input('last-click-time', 'data')]
        )
        def update_table(last_click_time: int):
            """Updates the predictions table based on the DataFrame state."""
            table_data = self.generate_table_data()
            
            # Generate columns configuration
            columns = [{'name': 'Metric', 'id': 'metric'}]
            for i in range(len(self.df)):
                columns.append({'name': f'T{i}', 'id': f't{i}'})
            
            return table_data, columns 