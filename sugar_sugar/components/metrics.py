from typing import List, Dict, Tuple, Optional
from dash import html, dcc, Output, Input, State, dash_table
import polars as pl

TableData = List[Dict[str, str]]  # Format for the predictions table data

class MetricsComponent(html.Div):
    def __init__(self, df: pl.DataFrame):
        self.df = df  # Store DataFrame reference
        
        # Create the component layout
        layout = html.Div([
            html.H4("Metrics", style={'fontSize': '20px', 'marginBottom': '10px'}),
            html.Div([
                dash_table.DataTable(
                    id='metrics-table-data',
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'textAlign': 'center',
                        'minWidth': '60px',
                        'maxWidth': '100px',
                    },
                    data=[],
                    columns=[]
                )
            ], id='metrics-table-container'),
            html.Div(id='metrics-content-data')
        ])

        # Initialize the parent html.Div with our layout
        super().__init__(
            layout,
            id='metrics-container',
            style={
                'padding': '20px',
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'marginBottom': '20px'
            }
        )

    def update_dataframe(self, df: pl.DataFrame):
        """Update the DataFrame reference"""
        self.df = df

    def generate_table_data(self, df: pl.DataFrame) -> TableData:
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
        table_data.extend(self.calculate_error_rows(df, prediction_row))
        
        return table_data

    def calculate_error_rows(self, df: pl.DataFrame, prediction_row: Dict[str, str]) -> List[Dict[str, str]]:
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

    def calculate_error_metrics(self, df: pl.DataFrame, prediction_row: Dict[str, str]) -> Optional[html.Div]:
        """Calculates error metrics when there are 5 or more predictions."""
        # Count valid predictions (non-"-" values)
        valid_predictions = sum(1 for key, value in prediction_row.items() if value != "-" and key != "metric")
        
        if valid_predictions < 5:
            return html.Div(
                "Add at least 5 prediction points to see accuracy metrics", 
                style={
                    'color': 'gray',
                    'fontStyle': 'italic',
                    'fontSize': '16px',
                    'padding': '10px',
                    'border': '2px dashed #ccc',
                    'borderRadius': '10px',
                    'margin': '10px'
                }
            )
            
        actual_values = []
        predicted_values = []
        
        for i, gl in enumerate(df.get_column("gl")):
            pred_str = prediction_row[f't{i}']
            if pred_str != "-":
                actual_values.append(gl)
                predicted_values.append(float(pred_str))
        
        # Calculate metrics
        n = len(actual_values)
        mae = sum(abs(a - p) for a, p in zip(actual_values, predicted_values)) / n
        mse = sum((a - p) ** 2 for a, p in zip(actual_values, predicted_values)) / n
        rmse = mse ** 0.5
        mape = sum(abs((a - p) / a) * 100 for a, p in zip(actual_values, predicted_values)) / n
        
        metrics = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE": mape
        }
        
        metric_descriptions = {
            "MAE": "Average difference between predicted and actual values",
            "MSE": "Emphasizes larger prediction errors",
            "RMSE": "Similar to MAE but penalizes large errors more",
            "MAPE": "Average percentage difference from actual values"
        }
        
        return html.Div([
            html.H4("Prediction Accuracy", style={'fontSize': '20px', 'marginBottom': '10px'}),
            html.Div([
                html.Div([
                    html.Div([
                        html.Strong(f"{metric}", style={'fontSize': '16px'}),
                        html.Div(f"{value:.2f}" + ("%" if metric == "MAPE" else ""), 
                               style={'fontSize': '20px', 'color': '#2c5282', 'margin': '5px 0'}),
                        html.Div(metric_descriptions[metric],
                               style={'fontSize': '14px', 'color': '#4a5568'})
                    ], style={
                        'padding': '10px',
                        'margin': '5px',
                        'border': '2px solid #e2e8f0',
                        'borderRadius': '8px',
                        'backgroundColor': '#f8fafc',
                        'minWidth': '150px',
                        'flex': '1'
                    })
                    for metric, value in metrics.items()
                ], style={
                    'display': 'flex',
                    'flexWrap': 'wrap',
                    'gap': '10px',
                    'justifyContent': 'center'
                })
            ], style={
                'border': '2px solid #cbd5e0',
                'borderRadius': '12px',
                'padding': '10px',
                'margin': '10px',
                'backgroundColor': 'white',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
            })
        ])

    def register_callbacks(self, app):
        @app.callback(
            [
                Output('metrics-table-data', 'data'),  # Updated ID
                Output('metrics-table-data', 'columns'),  # Updated ID
                Output('metrics-content-data', 'children')  # Updated ID
            ],
            [Input('last-click-time', 'data')]
        )
        def update_metrics(last_click_time: int):
            """Updates the predictions table and error metrics based on the DataFrame state."""
            # Generate table data using the current DataFrame state
            table_data = self.generate_table_data(self.df)
            
            # Generate columns configuration
            columns = [{'name': 'Metric', 'id': 'metric'}]
            for i in range(len(self.df)):
                columns.append({'name': f'T{i}', 'id': f't{i}'})
            
            # Calculate metrics
            metrics = self.calculate_error_metrics(self.df, table_data[1])
            
            return table_data, columns, metrics

    def create_layout(self):
        """Creates the metrics layout"""
        return html.Div(id='metrics-container')

    def calculate_metrics(self, prediction_row: Dict[str, str]) -> Optional[Dict[str, float]]:
        """Calculates error metrics when there are 5 or more predictions."""
        # ... (same as original calculate_error_metrics function but returns just the metrics dict)
        pass

    def generate_metrics_div(self, metrics: Optional[Dict[str, float]]) -> html.Div:
        """Generates the metrics visualization div"""
        # ... (same as the visualization part of original calculate_error_metrics)
        pass 