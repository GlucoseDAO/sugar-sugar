from typing import List, Dict, Tuple, Optional, Any, Union
import dash
from dash import dcc, html, Output, Input, State, dash_table
import plotly.graph_objs as go
import pandas as pd
import polars as pl
from datetime import datetime
import time
from pathlib import Path
import base64
import tempfile

# Type aliases for clarity
TimePoint = Dict[str, Union[datetime, float]]  # Represents a single point with time and glucose value
DrawnLines = List[TimePoint]  # List of points that make up the drawn lines
TableData = List[Dict[str, str]]  # Format for the predictions table data
Figure = go.Figure  # Plotly figure type

# Add this near the top with other type aliases
DEFAULT_POINTS = 24
MIN_POINTS = 10
MAX_POINTS = 30

# Add new global variables
window_start = 0  # Index of first visible point
full_df = None  # Store complete dataset
df = None  # Initial window view

# Modify load_glucose_data to load all data without limit
def load_glucose_data(file_path: Path = Path("data/example.csv")) -> Tuple[pl.DataFrame, pl.DataFrame]:
    df = pl.read_csv(
        file_path,
        null_values=["Low", "High"]
    )
    
    # Filter glucose data (EGV rows)
    glucose_data = (df
        .filter(pl.col("Event Type") == "EGV")
        .select([
            pl.col("Timestamp (YYYY-MM-DDThh:mm:ss)").alias("time"),
            pl.col("Glucose Value (mg/dL)").cast(pl.Float64).alias("gl")
        ])
        .with_columns([
            pl.col("time").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S"),
            pl.lit(0.0).alias("prediction")
        ])
        .sort("time")
    )
    
    # Filter event data (non-EGV rows we want to show)
    events_data = (df
        .filter(
            (pl.col("Event Type") == "Insulin") |
            (pl.col("Event Type") == "Exercise") |
            (pl.col("Event Type") == "Carbohydrates")
        )
        .select([
            pl.col("Timestamp (YYYY-MM-DDThh:mm:ss)").alias("time"),
            pl.col("Event Type").alias("event_type"),
            pl.col("Event Subtype").alias("event_subtype"),
            pl.col("Insulin Value (u)").alias("insulin_value")
        ])
        .with_columns([
            pl.col("time").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S")
        ])
        .sort("time")
    )
    
    return glucose_data, events_data

# Update initial loading
full_df, events_df = load_glucose_data()  # Unpack both dataframes
df = full_df.slice(0, DEFAULT_POINTS)  # Now this will work
events_window = events_df  # Store events
is_example_data = True  # Track if we're using example data

external_stylesheets: List[str] = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    # Header section
    html.Div([
        html.H1('Sugar Sugar', style={
            'textAlign': 'center',
            'color': '#2c5282',
            'marginBottom': '10px',
            'fontSize': '48px',
            'fontWeight': 'bold',
        }),
        # New row with two columns
        html.Div([
            # Left column - Game description
            html.Div([
                html.P([
                    'Test your glucose prediction skills! ',
                    html.Br(),
                    'Click on the graph or draw lines to predict future glucose values. ',
                    'The game will show you how accurate your predictions are compared to actual measurements.'
                ], style={
                    'fontSize': '18px',
                    'color': '#4a5568',
                    'lineHeight': '1.5'
                })
            ], style={'flex': '1', 'paddingRight': '20px'}),
            
            # Right column - Upload and controls
            html.Div([
                # Points control and slider in same row
                html.Div([
                    # Points control
                    html.Div([
                        html.Label('Number of points to show:', style={'marginRight': '10px'}),
                        dcc.Input(
                            id='points-control',
                            type='number',
                            value=DEFAULT_POINTS,
                            min=MIN_POINTS,
                            max=MAX_POINTS,
                            style={'width': '80px'}
                        ),
                    ], style={'flex': '0 0 auto', 'display': 'flex', 'alignItems': 'center'}),
                    
                    # Time slider
                    html.Div([
                        html.Label('Time Window Position:', style={'marginRight': '10px'}),
                        dcc.Slider(
                            id='time-slider',
                            min=0,
                            max=len(full_df) - DEFAULT_POINTS,
                            value=0,
                            marks=None,
                            tooltip={"placement": "bottom", "always_visible": True},
                            updatemode='mouseup',
                            included=True,
                            step=1
                        ),
                    ], style={'flex': '1', 'marginLeft': '20px'}),
                ], style={
                    'display': 'flex',
                    'flexDirection': 'row',
                    'alignItems': 'center',
                    'gap': '10px',
                    'marginBottom': '10px'
                }),
                
                # Upload control
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select a Dexcom/Libre CSV File')
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                    }
                ),
                html.Div(id='example-data-warning', style={'marginTop': '10px'})
            ], style={'flex': '1'})
        ], style={
            'display': 'flex',
            'flexDirection': 'row',
            'gap': '20px',
            'alignItems': 'start'
        })
    ], style={
        'padding': '15px',
        'marginBottom': '15px',
        'backgroundColor': 'white',
        'borderRadius': '10px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    }),

    # Main content container
    html.Div([
        # Interactive glucose graph component
        html.Div([
            dcc.Graph(
                id='glucose-graph',
                config={
                    'displayModeBar': True,
                    'scrollZoom': False,
                    'doubleClick': 'reset',
                    'showAxisDragHandles': False,
                    'showAxisRangeEntryBoxes': False,
                    'displaylogo': False
                },
                style={
                    'height': '100%',  # Let it fill its container naturally
                }
            )
        ], style={
            'padding': '20px',
            'backgroundColor': 'white',
            'borderRadius': '10px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'marginBottom': '20px',
            'display': 'flex',
            'flexDirection': 'column'
        }),

        # Game description (new section)
        html.Div([
            html.P([
                'How to play: ',
                html.Br(),
                '1. Click on points in the graph to add predictions ',
                html.Br(),
                '2. Draw lines between points to create prediction curves ',
                html.Br(),
                '3. Double-click to reset your predictions ',
                html.Br(),
                '4. Try to predict at least 5 points to see your accuracy metrics'
            ], style={
                'fontSize': '16px',
                'color': '#4a5568',
                'lineHeight': '1.5',
                'margin': '10px 0'
            })
        ], style={
            'padding': '15px',
            'backgroundColor': 'white',
            'borderRadius': '10px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'marginBottom': '20px'
        }),

        # Store component
        dcc.Store(id='last-click-time', data=0),
        
        # Predictions table
        html.Div([
            html.H4('Glucose Values and Predictions', style={'textAlign': 'center'}),
            dash_table.DataTable(
                id='predictions-table',
                columns=[
                    {'name': 'Metric', 'id': 'metric'},
                    *[{'name': f'T{i}', 'id': f't{i}'} for i in range(24)]
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
                        'if': {'row_index': 0},
                        'backgroundColor': 'rgba(200, 240, 200, 0.5)'
                    },
                    {
                        'if': {'row_index': 1},
                        'backgroundColor': 'rgba(255, 200, 200, 0.5)'
                    }
                ]
            )
        ], style={
            'padding': '20px',
            'backgroundColor': 'white',
            'borderRadius': '10px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'marginBottom': '20px'
        }),
        
        # Metrics section (removed duplicate upload section)
        html.Div([
            html.Div(id='error-metrics', style={
                'marginBottom': '15px'
            })
        ], style={
            'padding': '15px',
            'backgroundColor': 'white',
            'borderRadius': '10px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
        })
    ], style={
        'margin': '0 auto',
        'padding': '0 20px',
        'display': 'flex',
        'flexDirection': 'column',
        'gap': '20px'
    })
], style={
    'backgroundColor': '#f7fafc',
    'minHeight': '100vh',
    'padding': '20px'
})

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
    """Creates the base figure with glucose range rectangles."""
    fig = go.Figure()
    
    # Dangerous low range (below 70)
    fig.add_hrect(
        y0=0, y1=70,
        fillcolor="rgba(255, 200, 200, 0.5)",  # Light red
        line=dict(color="rgba(200, 0, 0, 0.5)", width=1),
        layer="below",
        name="Low Range"
    )
    
    # Normal range (70-180)
    fig.add_hrect(
        y0=70, y1=180,
        fillcolor="rgba(200, 240, 200, 0.5)",  # Light green
        line=dict(color="rgba(0, 100, 0, 0.5)", width=1),
        layer="below",
        name="Normal Range"
    )
    
    # High range (180-250)
    fig.add_hrect(
        y0=180, y1=250,
        fillcolor="rgba(255, 255, 200, 0.5)",  # Light yellow
        line=dict(color="rgba(200, 200, 0, 0.5)", width=1),
        layer="below",
        name="High Range"
    )
    
    # Dangerous high range (above 250)
    fig.add_hrect(
        y0=250, y1=400,
        fillcolor="rgba(255, 200, 200, 0.5)",  # Light red
        line=dict(color="rgba(200, 0, 0, 0.5)", width=1),
        layer="below",
        name="Very High Range"
    )
    
    return fig

def add_glucose_trace(fig: Figure, df: pl.DataFrame) -> None:
    """Adds the main glucose data line to the figure."""
    # Use sequential indices for x-axis (0 to len(df)-1)
    x_indices = list(range(len(df)))
    
    fig.add_trace(go.Scatter(
        x=x_indices,  # Use indices instead of full range
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
    """Calculates the y-axis range based on glucose and prediction values, using standard CGM ranges."""
    # Standard CGM chart ranges
    STANDARD_MIN = 40  # Standard lower bound for CGM charts
    STANDARD_MAX = 300  # Upper bound for CGM chart
    
    line_points = df.filter(pl.col("prediction") != 0.0)
    
    # Get actual data ranges
    data_min = df.get_column("gl").min()
    data_max = df.get_column("gl").max()
    
    # Include prediction values in range calculation if they exist
    if line_points.height > 0:
        pred_max = line_points.get_column("prediction").max()
        data_max = max(data_max, pred_max)
    
    # Set lower bound: Use standard min unless data goes lower
    lower_bound = min(STANDARD_MIN, max(0, data_min * 0.9))
    
    # Set upper bound: Use standard max unless data goes higher
    upper_bound = max(STANDARD_MAX, data_max * 1.1)
    
    return lower_bound, upper_bound

def update_figure_layout(fig: Figure, df: pl.DataFrame) -> None:
    """Updates the figure layout with axes, margins, and interaction settings."""
    y_range = calculate_y_axis_range(df)
    
    # Calculate window info for title
    start_time = df.get_column("time")[0].strftime('%Y-%m-%d %H:%M')
    end_time = df.get_column("time")[-1].strftime('%Y-%m-%d %H:%M')
    
    fig.update_layout(
        title=f'Glucose Levels ({start_time} to {end_time})',
        autosize=True,
        xaxis=dict(
            title='Time',
            tickmode='array',
            tickvals=list(range(len(df))),  # Use indices for x-axis
            ticktext=[t.strftime('%Y-%m-%d %H:%M') for t in df.get_column("time")],
            fixedrange=True,
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            gridcolor='rgba(128, 128, 128, 0.2)',
            showgrid=True,
            range=[-0.5, len(df) - 0.5]  # Keep the view fixed to current window
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
    table_data.extend(calculate_error_rows(df, prediction_row))
    
    return table_data

def calculate_error_rows(df: pl.DataFrame, prediction_row: Dict[str, str]) -> List[Dict[str, str]]:
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

def calculate_error_metrics(df: pl.DataFrame, prediction_row: Dict[str, str]) -> Optional[Dict[str, str]]:
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
    
    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape
    }
    
    if metrics is None:
        metrics_div = html.Div(
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
    else:
        metric_descriptions = {
            "MAE": "Average difference between predicted and actual values",
            "MSE": "Emphasizes larger prediction errors",
            "RMSE": "Similar to MAE but penalizes large errors more",
            "MAPE": "Average percentage difference from actual values"
        }
        
        metrics_div = html.Div([
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
    
    return metrics_div

def add_event_markers(fig: Figure, events_df: pl.DataFrame, df: pl.DataFrame) -> None:
    """Adds event markers (insulin, exercise, carbs) to the figure."""
    # Create a mapping of event types to symbols and colors
    event_styles = {
        'Insulin': {'symbol': 'triangle-down', 'color': 'purple', 'size': 20},  # Increased from 12
        'Exercise': {'symbol': 'star', 'color': 'orange', 'size': 20},         # Increased from 12
        'Carbohydrates': {'symbol': 'square', 'color': 'green', 'size': 20}    # Increased from 12
    }
    
    # Filter events to only those within the current time window
    start_time = df.get_column("time")[0]
    end_time = df.get_column("time")[-1]
    
    window_events = events_df.filter(
        (pl.col("time") >= start_time) & 
        (pl.col("time") <= end_time)
    )
    
    # Add traces for each event type
    for event_type, style in event_styles.items():
        events = window_events.filter(pl.col("event_type") == event_type)
        if events.height > 0:
            # Get corresponding glucose values for y-position
            event_times = events.get_column("time")
            y_positions = []
            hover_texts = []
            x_positions = []
            
            for event_time in event_times:
                # Find nearest glucose reading time
                nearest_idx = df.with_columns(
                    (pl.col("time").cast(pl.Int64) - pl.lit(int(event_time.timestamp() * 1000)))
                    .abs()
                    .alias("diff")
                ).select(pl.col("diff").arg_min()).item()
                
                glucose_value = df.get_column("gl")[nearest_idx]
                x_pos = get_time_position(df.get_column("time")[nearest_idx])
                
                y_positions.append(glucose_value)
                x_positions.append(x_pos)
                
                # Create hover text
                event_row = events.filter(pl.col("time") == event_time)
                if event_type == 'Insulin':
                    hover_text = f"Insulin: {event_row.get_column('insulin_value')[0]}u<br>{event_time.strftime('%H:%M')}"
                else:
                    hover_text = f"{event_type}<br>{event_time.strftime('%H:%M')}"
                hover_texts.append(hover_text)
            
            fig.add_trace(go.Scatter(
                x=x_positions,
                y=y_positions,
                mode='markers',
                name=event_type,
                marker=dict(
                    symbol=style['symbol'],
                    size=style['size'],
                    color=style['color'],
                    line=dict(width=2, color='white'),  # Added white outline
                    opacity=0.8  # Added slight transparency
                ),
                text=hover_texts,
                hoverinfo='text'
            ))

@app.callback(
    Output('glucose-graph', 'figure'),
    [Input('last-click-time', 'data')]
)
def update_graph(last_click_time: int) -> Figure:
    """Updates the graph based on the DataFrame state."""
    global df, events_df
    
    # Create new figure (completely fresh)
    fig = go.Figure()
    
    # Add base rectangles
    fig = create_base_figure()
    
    # Verify we're only plotting the window
    if len(df) > MAX_POINTS:
        plot_df = df.head(MAX_POINTS)
    else:
        plot_df = df
    
    # Add only the current window of data to the graph
    x_indices = list(range(len(plot_df)))
    fig.add_trace(go.Scatter(
        x=x_indices,
        y=plot_df['gl'],
        mode='lines+markers',
        name='Glucose Level',
        line=dict(color='blue'),
    ))
    
    # Add predictions if any
    add_prediction_traces(fig, plot_df)
    
    # Add event markers
    add_event_markers(fig, events_df, plot_df)
    
    # Update layout
    update_figure_layout(fig, plot_df)
    
    return fig

@app.callback(
    [
        Output('predictions-table', 'data'),
        Output('error-metrics', 'children')
    ],
    [Input('last-click-time', 'data')]
)
def update_graph(last_click_time: int) -> Tuple[TableData, Union[List[html.Div], html.Div]]:
    """Updates the predictions table and error metrics based on the DataFrame state."""
    # Generate table data
    table_data = generate_table_data(df)
    
    # Calculate and format error metrics
    metrics = calculate_error_metrics(df, table_data[1])
    
    return table_data, metrics

# Add new callback for file upload
@app.callback(
    [Output('example-data-warning', 'children'),
     Output('last-click-time', 'data', allow_duplicate=True)],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')],
    prevent_initial_call=True
)
def update_data_source(contents: Optional[str], filename: Optional[str]) -> Tuple[Optional[html.Div], int]:
    global df, is_example_data, full_df
    
    if contents is None:
        warning = html.Div([
            html.I(className="fas fa-exclamation-triangle", style={'marginRight': '8px'}),
            "Currently using example data. Upload your own Dexcom/Libre CSV file for personalized analysis."
        ], style={
            'color': '#b7791f',  # Warm yellow color
            'backgroundColor': '#fefcbf',
            'padding': '10px',
            'borderRadius': '5px',
            'textAlign': 'center'
        }) if is_example_data else None
        return warning, int(time.time() * 1000)
    
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(decoded)
            tmp_path = Path(tmp_file.name)
        
        # Load the new data
        full_df, events_df = load_glucose_data(tmp_path)
        df = full_df.slice(0, DEFAULT_POINTS)
        events_window = events_df  # Store events
        is_example_data = False
        
        # Clean up temporary file
        tmp_path.unlink()
        
        success_msg = html.Div([
            html.I(className="fas fa-check-circle", style={'marginRight': '8px'}),
            f"Successfully loaded data from {filename}"
        ], style={
            'color': '#2f855a',  # Green color
            'backgroundColor': '#c6f6d5',
            'padding': '10px',
            'borderRadius': '5px',
            'textAlign': 'center'
        })
        return success_msg, int(time.time() * 1000)
        
    except Exception as e:
        error_msg = html.Div([
            html.I(className="fas fa-times-circle", style={'marginRight': '8px'}),
            f"Error loading file: {str(e)}"
        ], style={
            'color': '#c53030',  # Red color
            'backgroundColor': '#fed7d7',
            'padding': '10px',
            'borderRadius': '5px',
            'textAlign': 'center'
        })
        return error_msg, int(time.time() * 1000)

# Modify the points control callback to update range slider properties
@app.callback(
    [
        Output('last-click-time', 'data', allow_duplicate=True),
        Output('time-slider', 'max'),
        Output('time-slider', 'value'),
    ],
    [Input('points-control', 'value')],
    [State('time-slider', 'value')],
    prevent_initial_call=True
)
def update_points_shown(points: int, current_position: int) -> Tuple[int, int, int]:
    """Updates the window size and adjusts slider properties."""
    global df
    
    # Ensure points is within valid range
    points = max(MIN_POINTS, min(MAX_POINTS, points))
    
    # Calculate new maximum slider value (now it's the last possible starting position)
    new_max = len(full_df) - points
    
    # Adjust current window start to respect bounds
    new_start = min(current_position, new_max)
    new_start = max(0, new_start)
    
    # Update visible window
    df = full_df.slice(new_start, new_start + points)
    
    return (
        int(time.time() * 1000),  # last-click-time
        new_max,                  # slider max
        new_start                 # new slider value
    )

# Update the slider callback for single value
@app.callback(
    Output('last-click-time', 'data', allow_duplicate=True),
    [Input('time-slider', 'value')],
    [State('points-control', 'value')],
    prevent_initial_call=True
)
def update_time_window(start_idx: int, num_points: int) -> int:
    """Updates the visible window of data based on slider position."""
    global df
    
    # Update the visible window to show num_points values starting at start_idx
    df = full_df.slice(start_idx, start_idx + num_points)
    
    # Verify we only have the requested number of points
    if len(df) != num_points:
        df = df.head(num_points)
    
    return int(time.time() * 1000)

def main() -> None:
    """Starts the Dash server."""
    app.run_server(debug=True)

if __name__ == '__main__':
    main()
