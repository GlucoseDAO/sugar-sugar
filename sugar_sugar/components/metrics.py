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
            [Input('current-window-df', 'data')]  # Listen to session storage instead
        )
        def store_metrics_data(df_data: Optional[Dict]) -> Optional[Dict]:
            """Calculate and store metrics when DataFrame changes"""
            if not df_data:
                return None
                
            # Reconstruct DataFrame from session storage
            df = self._reconstruct_dataframe_from_dict(df_data)
            
            # Calculate metrics using the table data
            table_data = self._generate_table_data(df)
            
            if len(table_data) < 4:  # Need actual, predicted, absolute error, relative error rows
                return None
            
            # Extract prediction data for metrics calculation
            prediction_row = table_data[1]  # Predicted values row
            actual_row = table_data[0]      # Actual glucose values row
            
            # Count valid predictions (non-dash values)
            valid_predictions = sum(1 for key, value in prediction_row.items() 
                                  if key != 'metric' and value != "-")
            
            if valid_predictions < 5:
                return None
            
            # Calculate metrics from the table data
            return self._calculate_metrics_from_table_data(table_data)

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
            html.H3("Accuracy Metrics", style={
                'textAlign': 'center',
                'fontSize': 'clamp(20px, 3vw, 28px)',  # Responsive font size
                'marginBottom': 'clamp(10px, 2vh, 20px)'
            }),
            html.Div([
                html.Div([
                    html.Div([
                        html.Strong(f"{metric}", style={
                            'fontSize': 'clamp(14px, 2.5vw, 18px)',  # Responsive font size
                            'display': 'block',
                            'marginBottom': '8px'
                        }),
                        html.Div(f"{data['value']:.2f}" + ("%" if metric == "MAPE" else ""), 
                               style={
                                   'fontSize': 'clamp(18px, 3vw, 24px)',  # Responsive font size
                                   'color': '#2c5282', 
                                   'margin': '5px 0',
                                   'fontWeight': 'bold'
                               }),
                        html.Div(data['description'],
                               style={
                                   'fontSize': 'clamp(12px, 2vw, 16px)',  # Responsive font size 
                                   'color': '#4a5568',
                                   'lineHeight': '1.4'
                               })
                    ], style={
                        'padding': 'clamp(10px, 2vw, 15px)',  # Responsive padding
                        'margin': 'clamp(5px, 1vw, 10px)',  # Responsive margin
                        'border': '2px solid #e2e8f0',
                        'borderRadius': '8px',
                        'backgroundColor': '#f8fafc',
                        'minWidth': 'clamp(150px, 20vw, 250px)',  # Responsive min width
                        'flex': '1',
                        'textAlign': 'center',
                        'boxSizing': 'border-box'
                    })
                    for metric, data in stored_metrics.items()
                ], style={
                    'display': 'flex',
                    'flexWrap': 'wrap',
                    'gap': 'clamp(8px, 1.5vw, 15px)',  # Responsive gap
                    'justifyContent': 'center',
                    'alignItems': 'stretch'  # Make all metric cards same height
                })
            ], style={
                'border': '2px solid #cbd5e0',
                'borderRadius': '12px',
                'padding': 'clamp(10px, 2vw, 20px)',  # Responsive padding
                'margin': 'clamp(10px, 2vw, 15px)',  # Responsive margin
                'backgroundColor': 'white',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'width': '100%',
                'boxSizing': 'border-box'
            })
        ]

    def _reconstruct_dataframe_from_dict(self, df_data: Dict) -> pl.DataFrame:
        """Reconstruct a Polars DataFrame from stored dictionary data"""
        return pl.DataFrame({
            'time': pl.Series(df_data['time']).str.strptime(pl.Datetime, format='%Y-%m-%dT%H:%M:%S'),
            'gl': pl.Series(df_data['gl'], dtype=pl.Float64),
            'prediction': pl.Series(df_data['prediction'], dtype=pl.Float64),
            'age': pl.Series([int(float(x)) for x in df_data['age']], dtype=pl.Int64),
            'user_id': pl.Series([int(float(x)) for x in df_data['user_id']], dtype=pl.Int64)
        })

    def _generate_table_data(self, df: pl.DataFrame) -> List[Dict[str, str]]:
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

    def _calculate_metrics_from_table_data(self, table_data: List[Dict[str, str]]) -> Dict[str, Any]:
        """Calculate metrics from table data"""
        if len(table_data) < 2:
            return {}
        
        actual_row = table_data[0]      # Actual glucose values row
        prediction_row = table_data[1]  # Predicted values row
        
        # Extract valid prediction pairs
        actual_values = []
        predicted_values = []
        
        for key in actual_row.keys():
            if key == 'metric':
                continue
            
            actual_str = actual_row[key]
            pred_str = prediction_row[key]
            
            if actual_str != "-" and pred_str != "-":
                try:
                    actual_val = float(actual_str)
                    pred_val = float(pred_str)
                    actual_values.append(actual_val)
                    predicted_values.append(pred_val)
                except ValueError:
                    continue
        
        if len(actual_values) < 5:
            return {}
        
        # Calculate metrics
        n = len(actual_values)
        mae = sum(abs(a - p) for a, p in zip(actual_values, predicted_values)) / n
        mse = sum((a - p) ** 2 for a, p in zip(actual_values, predicted_values)) / n
        rmse = mse ** 0.5
        mape = sum(abs((a - p) / a) * 100 for a, p in zip(actual_values, predicted_values) if a != 0) / n
        
        return {
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