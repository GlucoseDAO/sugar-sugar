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
            Output('metrics-content-data', 'children'),
            [Input('prediction-table-data', 'data')]
        )
        def update_metrics(table_data: List[Dict[str, str]]):
            """Updates the error metrics based on the prediction table data."""
            if not table_data or len(table_data) < 2:
                return None
            
            prediction_row = table_data[1]  # Get the predictions row
            metrics = self.calculate_error_metrics(self.df, prediction_row)
            
            return metrics 