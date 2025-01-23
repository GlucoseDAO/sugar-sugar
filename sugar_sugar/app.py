from typing import List, Dict, Tuple, Optional, Any, Union
import dash
from dash import dcc, html, Output, Input, State, dash_table
import plotly.graph_objs as go
import pandas as pd
import polars as pl
from datetime import datetime
import time
from pathlib import Path

# Type aliases for clarity
TimePoint = Dict[str, Union[datetime, float]]  # Represents a single point with time and glucose value
DrawnLines = List[TimePoint]  # List of points that make up the drawn lines
TableData = List[Dict[str, str]]  # Format for the predictions table data
Figure = go.Figure  # Plotly figure type

# Load glucose data from CSV and get last 24 values
def load_glucose_data(limit: int = 24) -> pl.DataFrame:
    csv_path = Path("data/example.csv")
    df = pl.read_csv(
        csv_path,
        null_values=["Low", "High"]  # Treat Low/High as null values
    )
    # Filter only EGV rows and required columns
    glucose_data = (df
        .filter(pl.col("Event Type") == "EGV")
        .select([
            pl.col("Timestamp (YYYY-MM-DDThh:mm:ss)").alias("time"),
            pl.col("Glucose Value (mg/dL)").cast(pl.Float64).fill_null(40.0).alias("gl")  # Convert Low values to 40 mg/dL
        ])
        # Convert timestamp to datetime
        .with_columns([
            pl.col("time").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S"),
            pl.lit(0.0).alias("prediction")
        ])
        # Sort by time and get last 24 values
        .sort("time")
        .tail(limit)
    )
    return glucose_data

# Load the data
df = load_glucose_data()

external_stylesheets: List[str] = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    # Interactive glucose graph component
    dcc.Graph(
        id='glucose-graph',
        config={
            'displayModeBar': True,     # Shows the plotly toolbar
            'scrollZoom': False,        # Disables scroll zoom for better drawing
            'doubleClick': 'reset',     # Enables double-click to reset view
            'showAxisDragHandles': False,  # Simplifies interface
            'showAxisRangeEntryBoxes': False,
            'displaylogo': False
        }
    ),
    # Remove drawn-lines store, keep only last-click-time
    dcc.Store(id='last-click-time', data=0),
    
    # Updated table section
    html.Div([
        html.H4('Glucose Values and Predictions', style={'textAlign': 'center'}),
        dash_table.DataTable(
            id='predictions-table',
            columns=[
                {'name': 'Metric', 'id': 'metric'},
                *[{'name': f'T{i}', 'id': f't{i}'} for i in range(24)]  # Dynamic columns for each timepoint
            ],
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'center',
                'padding': '5px',
                'minWidth': '60px'
            },
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 0},  # Actual glucose values
                    'backgroundColor': 'rgba(200, 240, 200, 0.5)'
                },
                {
                    'if': {'row_index': 1},  # Predictions
                    'backgroundColor': 'rgba(255, 200, 200, 0.5)'
                }
            ]
        )
    ], style={'margin': '20px'}),
    
    # Add metrics section
    html.Div([
        html.Div(id='error-metrics', style={'textAlign': 'center', 'margin': '20px'})
    ])
])

DOUBLE_CLICK_THRESHOLD: int = 500  # milliseconds

def find_nearest_time(x: Union[str, float, datetime]) -> datetime:
    """
    Finds the nearest allowed time from the DataFrame 'df' for a given x-coordinate.
    x can be either an index (float) or a timestamp string.
    """
    if isinstance(x, (int, float)):
        # If x is a numerical index, round to nearest integer and get corresponding time
        idx = round(float(x))
        idx = max(0, min(idx, len(df) - 1))  # Ensure index is within bounds
        return df.get_column("time")[idx]
    
    # If x is a timestamp string, convert to datetime
    x_ts = pd.to_datetime(x)
    time_diffs = df.select([
        (pl.col("time").cast(pl.Int64) - pl.lit(int(x_ts.timestamp() * 1000)))
        .abs()
        .alias("diff")
    ])
    nearest_idx = time_diffs.select(pl.col("diff").arg_min()).item()
    return df.get_column("time")[nearest_idx]

@app.callback(
    Output('last-click-time', 'data'),
    [
        Input('glucose-graph', 'clickData'),
        Input('glucose-graph', 'relayoutData'),
    ],
    [
        State('last-click-time', 'data'),
    ]
)
def handle_click(
    click_data: Optional[Dict[str, Any]],
    relayout_data: Optional[Dict[str, Any]], 
    last_click_time: int,
) -> int:
    global df
    current_time: int = int(time.time() * 1000)
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return last_click_time
        
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Handle double-click reset
    if trigger_id == 'glucose-graph' and click_data:
        if current_time - last_click_time <= DOUBLE_CLICK_THRESHOLD:
            print("Double-click detected: Resetting drawn lines.")
            df = df.with_columns(pl.lit(0.0).alias("prediction"))
            return current_time
        
        # Handle single click
        point_data = click_data['points'][0]
        click_x = point_data['x']
        click_y = point_data['y']
        
        nearest_time = find_nearest_time(click_x)
        # Update using Polars syntax
        df = df.with_columns(
            pl.when(pl.col("time") == nearest_time)
            .then(click_y)
            .otherwise(pl.col("prediction"))
            .alias("prediction")
        )
        print(f"Updated line value at {nearest_time} to {click_y}")
        
        return current_time
    
    # Handle drawing mode
    if trigger_id == 'glucose-graph' and relayout_data:
        if 'shapes' in relayout_data:
            shapes = relayout_data['shapes']
            if shapes and len(shapes) > 0:
                latest_shape = shapes[-1]
                
                start_x = latest_shape.get('x0')
                end_x = latest_shape.get('x1')
                start_y = latest_shape.get('y0')
                end_y = latest_shape.get('y1')
                
                if all(v is not None for v in [start_x, end_x, start_y, end_y]):
                    start_time = find_nearest_time(start_x)
                    end_time = find_nearest_time(end_x)
                    
                    print(f"Drawing line from ({start_x}, {start_y}) to ({end_x}, {end_y})")
                    print(f"Snapped times: {start_time} to {end_time}")
                    
                    # Update both points using Polars syntax
                    df = df.with_columns(
                        pl.when(pl.col("time").is_in([start_time, end_time]))
                        .then(
                            pl.when(pl.col("time") == start_time)
                            .then(float(start_y))
                            .otherwise(float(end_y))
                        )
                        .otherwise(pl.col("prediction"))
                        .alias("prediction")
                    )
                    
                    return current_time
    
    return last_click_time

def get_time_position(time_point: datetime) -> float:
    """
    Converts a datetime to its corresponding x-axis position.
    Returns the index (0 to len(df)-1) which represents the position on x-axis.
    """
    time_series = df.get_column("time")
    for idx, t in enumerate(time_series):
        if t == time_point:
            return idx
    return 0  # fallback, shouldn't happen with find_nearest_time

def create_base_figure() -> Figure:
    """Creates the base figure with the normal glucose range rectangle."""
    fig = go.Figure()
    fig.add_hrect(
        y0=70, y1=180,
        fillcolor="rgba(200, 240, 200, 0.5)",
        line=dict(color="rgba(0, 100, 0, 0.5)", width=1),
        layer="below",
        name="Normal Range"
    )
    return fig

def add_glucose_trace(fig: Figure, df: pl.DataFrame) -> None:
    """Adds the main glucose data line to the figure."""
    fig.add_trace(go.Scatter(
        x=list(range(len(df))),
        y=df['gl'],
        mode='lines+markers',
        name='Glucose Level',
        line=dict(color='blue'),
    ))

def add_prediction_traces(fig: Figure, df: pl.DataFrame) -> None:
    """Adds prediction points and connecting lines to the figure."""
    line_points = df.filter(pl.col("prediction") != 0.0)
    if line_points.height > 0:
        x_positions = [get_time_position(t) for t in line_points.get_column("time")]
        
        fig.add_trace(go.Scatter(
            x=x_positions,
            y=line_points.get_column("prediction"),
            mode='markers',
            name='Prediction Points',
            marker=dict(color='red', size=8)
        ))

        if line_points.height >= 2:
            line_points_sorted = line_points.sort("time")
            times = line_points_sorted.get_column("time")
            predictions = line_points_sorted.get_column("prediction")
            
            for i in range(line_points.height - 1):
                start_pos = get_time_position(times[i])
                end_pos = get_time_position(times[i + 1])
                
                fig.add_trace(go.Scatter(
                    x=[start_pos, end_pos],
                    y=[predictions[i], predictions[i + 1]],
                    mode='lines',
                    line=dict(color='red', width=2),
                    showlegend=False
                ))

def calculate_y_axis_range(df: pl.DataFrame) -> Tuple[float, float]:
    """Calculates the y-axis range based on glucose and prediction values."""
    line_points = df.filter(pl.col("prediction") != 0.0)
    lower_bound = max(0, min(df.get_column("gl").min() * 0.9, 70))
    upper_bound = max(
        df.get_column("gl").max() * 1.1,
        (line_points.get_column("prediction").max() * 1.1 if line_points.height > 0 else 0)
    )
    return lower_bound, upper_bound

def update_figure_layout(fig: Figure, df: pl.DataFrame) -> None:
    """Updates the figure layout with axes, margins, and interaction settings."""
    y_range = calculate_y_axis_range(df)
    fig.update_layout(
        title='Glucose Levels Over Time',
        height=600,
        xaxis=dict(
            title='Time',
            tickmode='array',
            tickvals=list(range(len(df))),
            ticktext=[t.strftime('%Y-%m-%d %H:%M') for t in df.get_column("time")],
            fixedrange=True,
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            gridcolor='rgba(128, 128, 128, 0.2)',
            showgrid=True,
            range=[-0.5, len(df) - 0.5]  # This ensures the plot starts and ends at the data points
        ),
        yaxis=dict(
            title='Glucose Level (mg/dL)',
            fixedrange=True,
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            gridcolor='rgba(128, 128, 128, 0.2)',
            showgrid=True,
            range=y_range
        ),
        margin=dict(l=50, r=50, t=50, b=50),
        showlegend=True,
        dragmode='drawline',
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

def generate_table_data(df: pl.DataFrame) -> TableData:
    """Generates the table data with actual values, predictions, and errors."""
    table_data = []
    
    # Row 1: Actual glucose values
    glucose_row = {'metric': 'Actual Glucose'}
    for i, gl in enumerate(df.get_column("gl")):
        glucose_row[f't{i}'] = f"{gl:.1f}"
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
    table_data.extend(calculate_error_rows(df, prediction_row))
    
    return table_data

def calculate_error_rows(df: pl.DataFrame, prediction_row: Dict[str, str]) -> List[Dict[str, str]]:
    """Calculates absolute and relative error rows for the table."""
    error_rows = []
    
    # Absolute Error
    error_row = {'metric': 'Absolute Error'}
    for i, gl in enumerate(df.get_column("gl")):
        pred_str = prediction_row[f't{i}']
        if pred_str != "-":
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
        if pred_str != "-" and gl != 0:
            pred = float(pred_str)
            rel_error = (abs(gl - pred) / gl * 100)
            rel_error_row[f't{i}'] = f"{rel_error:.1f}%"
        else:
            rel_error_row[f't{i}'] = "-"
    error_rows.append(rel_error_row)
    
    return error_rows

def calculate_error_metrics(df: pl.DataFrame, prediction_row: Dict[str, str]) -> Optional[Dict[str, float]]:
    """Calculates error metrics when there are 5 or more predictions."""
    # Count valid predictions (non-"-" values)
    valid_predictions = sum(1 for key, value in prediction_row.items() if value != "-" and key != "metric")
    
    if valid_predictions < 5:
        return None
        
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
    
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape
    }

@app.callback(
    [
        Output('glucose-graph', 'figure'),
        Output('predictions-table', 'data'),
        Output('error-metrics', 'children')
    ],
    [Input('last-click-time', 'data')]
)
def update_graph(last_click_time: int) -> Tuple[Figure, TableData, Union[List[html.Div], html.Div]]:
    """Updates the graph, predictions table, and error metrics based on the DataFrame state."""
    # Create and populate the figure
    fig = create_base_figure()
    add_glucose_trace(fig, df)
    add_prediction_traces(fig, df)
    update_figure_layout(fig, df)
    
    # Generate table data
    table_data = generate_table_data(df)
    
    # Calculate and format error metrics
    metrics = calculate_error_metrics(df, table_data[1])
    
    if metrics is None:
        metrics_div = html.Div(
            "Add at least 5 prediction points to see accuracy metrics", 
            style={
                'color': 'gray',
                'fontStyle': 'italic',
                'fontSize': '18px',
                'padding': '20px',
                'border': '2px dashed #ccc',
                'borderRadius': '10px',
                'margin': '20px'
            }
        )
    else:
        metric_descriptions = {
            "MAE": "Mean Absolute Error - The average difference between predicted and actual values",
            "MSE": "Mean Squared Error - Emphasizes larger prediction errors by squaring them",
            "RMSE": "Root Mean Square Error - Similar to MAE but penalizes large errors more",
            "MAPE": "Mean Absolute Percentage Error - Shows the average percentage difference from actual values"
        }
        
        metrics_div = html.Div([
            html.H4("Prediction Accuracy Metrics", style={'fontSize': '24px', 'marginBottom': '15px'}),
            html.Div([
                html.Div([
                    html.Div([
                        html.Strong(f"{metric}", style={'fontSize': '20px'}),
                        html.Div(f"{value:.2f}" + ("%" if metric == "MAPE" else ""), 
                               style={'fontSize': '24px', 'color': '#2c5282', 'margin': '10px 0'}),
                        html.Div(metric_descriptions[metric],  # Use the description from the dictionary
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
                    for metric, value in metrics.items()
                ], style={
                    'display': 'flex',
                    'flexWrap': 'wrap',
                    'gap': '20px',
                    'justifyContent': 'center'
                })
            ], style={
                'border': '2px solid #cbd5e0',
                'borderRadius': '12px',
                'padding': '20px',
                'margin': '20px',
                'backgroundColor': 'white',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
            })
        ])
    
    return fig, table_data, metrics_div

def main() -> None:
    """Starts the Dash server."""
    app.run_server(debug=True)

if __name__ == '__main__':
    main()
