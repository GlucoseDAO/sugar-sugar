from typing import List, Dict, Tuple, Optional, Any, Union
import dash
from dash import dcc, html, Output, Input, State
import plotly.graph_objs as go
import pandas as pd
import polars as pl
from datetime import datetime
import time
from pathlib import Path
import base64
import tempfile

from .data import load_glucose_data
from .config import DEFAULT_POINTS, MIN_POINTS, MAX_POINTS, DOUBLE_CLICK_THRESHOLD
from .components.glucose import GlucoseChart
from .components.metrics import MetricsComponent
from .components.predictions import PredictionTableComponent

# Type aliases for clarity
TableData = List[Dict[str, str]]  # Format for the predictions table data
Figure = go.Figure  # Plotly figure type



# Add new global variables
#note that index just slides across values- that means an increase +1 moves fwd with 5 minutes interval
window_start = 0  # Index of first visible point
full_df = None  # Store complete dataset
df = None  # Initial window view

# Update initial loading
full_df, events_df = load_glucose_data()  # Unpack both dataframes
df = full_df.slice(0, DEFAULT_POINTS)  # Now this will work
events_window = events_df  # Store events
is_example_data = True  # Track if we're using example data

external_stylesheets: List[str] = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

from dash import html, dcc
from .config import DEFAULT_POINTS, MIN_POINTS, MAX_POINTS
from .components.header import HeaderComponent

# Create global instances
glucose_chart = GlucoseChart(id='glucose-graph')
prediction_table = PredictionTableComponent(df)
metrics_component = MetricsComponent(df)  # Create global instance

'''
Create the layout of the app, the base on which user will interact with
'''
def create_layout() -> html.Div:
    """Create the main layout of the application"""
    return html.Div([
        # Header section using HeaderComponent
        HeaderComponent(),

        # Main content container
        html.Div([
            # Store components for state management
            dcc.Store(id='current-window-df', data=None),
            dcc.Store(id='full-df', data=None),
            dcc.Store(id='events-df', data=None),
            dcc.Store(id='last-click-time', data=0),
            dcc.Store(id='is-example-data', data=True),

            # Interactive glucose graph component
            glucose_chart,
            
            # Predictions table using new component
            prediction_table,
            
            # Use metrics_component directly instead of empty div
            metrics_component
            
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


app.layout = create_layout()


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
    global df, full_df
    current_time: int = int(time.time() * 1000)
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return last_click_time
        
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Handle double-click reset
    if trigger_id == 'glucose-graph' and click_data:
        if current_time - last_click_time <= DOUBLE_CLICK_THRESHOLD:
            print("Double-click detected: Resetting drawn lines.")
            full_df = full_df.with_columns(pl.lit(0.0).alias("prediction"))
            df = df.with_columns(pl.lit(0.0).alias("prediction"))
            return current_time
        
        # Handle single click
        point_data = click_data['points'][0]
        click_x = point_data['x']
        click_y = point_data['y']
        
        nearest_time = find_nearest_time(click_x)
        # Update both full_df and current window
        full_df = full_df.with_columns(
            pl.when(pl.col("time") == nearest_time)
            .then(click_y)
            .otherwise(pl.col("prediction"))
            .alias("prediction")
        )
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
                    
                    # Update both full_df and current window
                    full_df = full_df.with_columns(
                        pl.when(pl.col("time").is_in([start_time, end_time]))
                        .then(
                            pl.when(pl.col("time") == start_time)
                            .then(float(start_y))
                            .otherwise(float(end_y))
                        )
                        .otherwise(pl.col("prediction"))
                        .alias("prediction")
                    )
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


@app.callback(
    Output('glucose-graph', 'figure'),
    [Input('last-click-time', 'data')]
)
def update_graph(last_click_time: int) -> Figure:
    """Updates the graph based on the DataFrame state."""
    global df, events_df
    return glucose_chart.update(df, events_df)


@app.callback(
    Output('metrics-container', 'children'),  # Update the metrics-container instead
    [Input('last-click-time', 'data')]
)
def update_metrics(last_click_time: int) -> Union[List[html.Div], html.Div]:
    """Updates the error metrics based on the DataFrame state."""
    metrics_component.update_dataframe(df)  # Update the component's DataFrame
    prediction_table.update_dataframe(df)  # Update the prediction table's DataFrame
    table_data = prediction_table.generate_table_data()
    prediction_row = table_data[1]  # Index 1 contains predictions
    return metrics_component.calculate_error_metrics(df, prediction_row)


# Add new callback for file upload
@app.callback(
    [Output('example-data-warning', 'children'),
     Output('last-click-time', 'data', allow_duplicate=True)],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')],
    prevent_initial_call=True
)
def update_data_source(contents: Optional[str], filename: Optional[str]) -> Tuple[Optional[html.Div], int]:
    global df, is_example_data, full_df, events_df
    
    if contents is None:
        warning = html.Div([
            html.I(className="fas fa-exclamation-triangle", style={'marginRight': '8px'}),
            "Currently using example data. Upload your own Dexcom/Libre CSV file for personalized analysis."
        ], style={
            'color': '#b7791f',
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
        new_full_df, new_events_df = load_glucose_data(tmp_path)
        
        # Update all global dataframes
        full_df = new_full_df
        events_df = new_events_df
        df = full_df.slice(0, DEFAULT_POINTS)
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
    global df, full_df
    
    # Simply slice the full_df which already contains all predictions
    df = full_df.slice(start_idx, start_idx + num_points)
    
    # Verify we only have the requested number of points
    if len(df) != num_points:
        df = df.head(num_points)
    
    return int(time.time() * 1000)


@app.callback(
    [Output('full-df', 'data'),
     Output('current-window-df', 'data'),
     Output('events-df', 'data')],
    [Input('glucose-graph', 'id')],  # Using an existing component's id instead of dummy div
    prevent_initial_call=False
)
def initialize_data(_):
    """Initialize the data stores on page load"""
    full_df, events_df = load_glucose_data()
    current_df = full_df.slice(0, DEFAULT_POINTS)
    
    # Convert DataFrames to JSON-serializable dictionaries
    def convert_df_to_dict(df):
        return {
            'time': df.get_column('time').dt.strftime('%Y-%m-%dT%H:%M:%S').to_list(),
            'gl': df.get_column('gl').to_list(),
            'prediction': df.get_column('prediction').to_list()
        }
    
    def convert_events_df_to_dict(df):
        return {
            'time': df.get_column('time').dt.strftime('%Y-%m-%dT%H:%M:%S').to_list(),
            'event_type': df.get_column('event_type').to_list(),
            'event_subtype': df.get_column('event_subtype').to_list(),
            'insulin_value': df.get_column('insulin_value').to_list()
        }
    
    return (
        convert_df_to_dict(full_df),
        convert_df_to_dict(current_df),
        convert_events_df_to_dict(events_df)
    )

def main() -> None:
    """Starts the Dash server."""
    prediction_table.register_callbacks(app)  # Register the prediction table callbacks
    app.run_server(debug=True)

if __name__ == '__main__':
    main()
