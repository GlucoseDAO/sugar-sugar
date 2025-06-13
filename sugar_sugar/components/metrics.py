from typing import List, Dict, Tuple, Optional, Any, Union
from dash import html, dcc, Output, Input, State, dash_table
import polars as pl
import dash


class MetricsComponent(html.Div):
    def __init__(self):
        # Create the layout with session storage
        super().__init__(
            children=[
                dcc.Store(id='metrics-store', data=None),  # Session storage for metrics
                html.Div(
                    id='metrics-container',
                    children=[
                        html.H4("Metrics", style={'fontSize': '20px', 'marginBottom': '10px'}),
                        html.Div(
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
                    ],
                    style={
                        'padding': '20px',
                        'backgroundColor': 'white',
                        'borderRadius': '10px',
                        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                        'marginBottom': '20px'
                    }
                )
            ]
        )

    def register_callbacks(self, app: dash.Dash, prediction_table_instance) -> None:
        """Register all metrics-related callbacks"""
        
        @app.callback(
            Output('metrics-store', 'data'),
            [Input('last-click-time', 'data')]
        )
        def calculate_and_store_metrics(last_click_time: int) -> Union[Dict[str, Any], None]:
            """Calculate metrics when predictions change and store them"""
            from sugar_sugar.app import df  # Import the global df
            
            # Generate table data to get predictions
            prediction_table_instance.update_dataframe(df)
            table_data = prediction_table_instance.generate_table_data()
            
            if len(table_data) < 2:
                return None
                
            prediction_row = table_data[1]  # Index 1 contains predictions
            
            # Count valid predictions (non-"-" values)
            valid_predictions = sum(1 for key, value in prediction_row.items() if value != "-" and key != "metric")
            print(f"DEBUG: Found {valid_predictions} valid predictions for metrics storage")
            
            if valid_predictions < 5:
                print("DEBUG: Not enough predictions for metrics, storing None")
                return None
            
            # Calculate metrics
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
                "MAE": {
                    "value": mae,
                    "description": "Average difference between predicted and actual values"
                },
                "MSE": {
                    "value": mse,
                    "description": "Emphasizes larger prediction errors"
                },
                "RMSE": {
                    "value": rmse,
                    "description": "Similar to MAE but penalizes large errors more"
                },
                "MAPE": {
                    "value": mape,
                    "description": "Average percentage difference from actual values"
                }
            }
            
            print(f"DEBUG: Calculated and stored metrics: {metrics}")
            return metrics

        @app.callback(
            Output('metrics-container', 'children'),
            [Input('metrics-store', 'data')]
        )
        def update_metrics_display(stored_metrics: Optional[Dict[str, Any]]) -> List[html.Div]:
            """Updates the metrics display based on stored metrics."""
            print(f"DEBUG: update_metrics_display called with stored metrics: {bool(stored_metrics)}")
            
            if not stored_metrics:
                # Return placeholder when no metrics available
                return [
                    html.H4("Metrics", style={'fontSize': '20px', 'marginBottom': '10px'}),
                    html.Div(
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
                ]
            
            # Create metrics display from stored data
            return [
                html.H4("Prediction Accuracy", style={'fontSize': '20px', 'marginBottom': '10px'}),
                html.Div([
                    html.Div([
                        html.Div([
                            html.Strong(f"{metric}", style={'fontSize': '16px'}),
                            html.Div(f"{data['value']:.2f}" + ("%" if metric == "MAPE" else ""), 
                                   style={'fontSize': '20px', 'color': '#2c5282', 'margin': '5px 0'}),
                            html.Div(data['description'],
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
                        for metric, data in stored_metrics.items()
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
            ]

    @staticmethod
    def create_ending_metrics_display(stored_metrics: Optional[Dict[str, Any]]) -> List[html.Div]:
        """Create metrics display for the ending page"""
        print(f"DEBUG: create_ending_metrics_display called with stored metrics: {bool(stored_metrics)}")
        
        if not stored_metrics:
            return [
                html.H3("Accuracy Metrics", style={'textAlign': 'center'}),
                html.Div(
                    "No metrics available", 
                    style={
                        'color': 'gray',
                        'fontStyle': 'italic',
                        'fontSize': '16px',
                        'padding': '10px',
                        'textAlign': 'center'
                    }
                )
            ]
        
        # Create metrics display from stored data
        return [
            html.H3("Accuracy Metrics", style={'textAlign': 'center'}),
            html.Div([
                html.Div([
                    html.Div([
                        html.Strong(f"{metric}", style={'fontSize': '18px'}),
                        html.Div(f"{data['value']:.2f}" + ("%" if metric == "MAPE" else ""), 
                               style={'fontSize': '24px', 'color': '#2c5282', 'margin': '5px 0'}),
                        html.Div(data['description'],
                               style={'fontSize': '16px', 'color': '#4a5568'})
                    ], style={
                        'padding': '15px',
                        'margin': '10px',
                        'border': '2px solid #e2e8f0',
                        'borderRadius': '8px',
                        'backgroundColor': '#f8fafc',
                        'minWidth': '200px',
                        'flex': '1'
                    })
                    for metric, data in stored_metrics.items()
                ], style={
                    'display': 'flex',
                    'flexWrap': 'wrap',
                    'gap': '10px',
                    'justifyContent': 'center'
                })
            ], style={
                'border': '2px solid #cbd5e0',
                'borderRadius': '12px',
                'padding': '15px',
                'margin': '15px',
                'backgroundColor': 'white',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
            })
        ] 