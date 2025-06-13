from typing import List, Dict, Tuple, Optional, Any, Union
import dash
from dash import dcc, html, Output, Input, State, no_update
import plotly.graph_objs as go
import pandas as pd
import polars as pl
from datetime import datetime
import time
from pathlib import Path
import base64
import tempfile
import dash_bootstrap_components as dbc

from sugar_sugar.data import load_glucose_data
from sugar_sugar.config import DEFAULT_POINTS, MIN_POINTS, MAX_POINTS, DOUBLE_CLICK_THRESHOLD
from sugar_sugar.components.glucose import GlucoseChart
from sugar_sugar.components.metrics import MetricsComponent
from sugar_sugar.components.predictions import PredictionTableComponent
from sugar_sugar.components.startup import StartupPage
from sugar_sugar.components.submit import SubmitComponent
from sugar_sugar.components.header import HeaderComponent
from sugar_sugar.components.ending import EndingPage

# Type aliases for clarity
TableData = List[Dict[str, str]]  # Format for the predictions table data
Figure = go.Figure  # Plotly figure type

# Add new global variables
window_start = 0  # Index of first visible point
full_df = None  # Store complete dataset
df = None  # Initial window view
events_df = None  # Store events
is_example_data = True  # Track if we're using example data

# Update initial loading here is uploaded as a dataframe from the stored one
full_df, events_df = load_glucose_data()  # Unpack both dataframes
df = full_df.slice(0, DEFAULT_POINTS)  # Now this will work

external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    dbc.themes.BOOTSTRAP
]

app = dash.Dash(__name__, 
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True
)

# Create global instances
glucose_chart = GlucoseChart(id='glucose-graph')
prediction_table = PredictionTableComponent(df)
metrics_component = MetricsComponent(df)
submit_component = SubmitComponent()
startup_page = StartupPage()
ending_page = EndingPage()

# Set initial layout to startup page
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='user-info-store', data=None),
    dcc.Store(id='last-click-time', data=0),
    dcc.Store(id='current-window-df', data=None),
    dcc.Store(id='full-df', data=None),
    dcc.Store(id='events-df', data=None),
    dcc.Store(id='is-example-data', data=True),
    dcc.Store(id='submission-trigger', data=False),  # New store for submission trigger
    html.Div(id='page-content', children=[startup_page])  # Initialize with startup page
])

@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')],
    [State('user-info-store', 'data')],
    prevent_initial_call=False
)
def display_page(pathname: Optional[str], user_info: Optional[Dict[str, Any]]) -> html.Div:
    print(f"DEBUG: display_page called with pathname: {pathname}")
    if pathname == '/prediction' and user_info:
        return create_prediction_layout()
    elif pathname == '/ending':
        # Let the ending page callback handle this - return no_update to avoid conflicts
        return no_update
    return startup_page  # Return the startup page component

def create_prediction_layout() -> html.Div:
    """Create the prediction page layout"""
    return html.Div([
        HeaderComponent(),
        html.Div([
            glucose_chart,
            prediction_table,
            metrics_component,
            submit_component
        ], style={'flex': '1'})
    ], style={
        'margin': '0 auto',
        'padding': '0 20px',
        'display': 'flex',
        'flexDirection': 'column',
        'gap': '20px'
    })

@app.callback(
    Output('submission-trigger', 'data'),
    [Input('submit-button', 'n_clicks')],
    prevent_initial_call=True
)
def trigger_submission(n_clicks: Optional[int]) -> Union[bool, Any]:
    if n_clicks:
        return True
    return no_update

@app.callback(
    [Output('url', 'pathname', allow_duplicate=True),
     Output('user-info-store', 'data', allow_duplicate=True),
     Output('full-df', 'data', allow_duplicate=True),
     Output('current-window-df', 'data', allow_duplicate=True)],
    [Input('submission-trigger', 'data')],
    [State('user-info-store', 'data')],
    prevent_initial_call=True
)
def handle_submission(trigger: bool, user_info: Optional[Dict[str, Any]]) -> Tuple[Union[str, Any], Union[Dict[str, Any], Any], Union[Dict[str, Any], Any], Union[Dict[str, Any], Any]]:
    if trigger:
        print("DEBUG: handle_submission triggered")
        global full_df, df
        
        # Use the current global DataFrames which have the latest predictions
        # instead of the potentially stale stored data
        current_full_df = full_df.clone()
        current_df = df.clone()
        
        # Update age and user_id from user_info
        if user_info and 'age' in user_info:
            current_full_df = current_full_df.with_columns(pl.lit(int(user_info['age'])).alias("age"))
            current_df = current_df.with_columns(pl.lit(int(user_info['age'])).alias("age"))
        
        print("DEBUG: Submitting with prediction data:")
        print(f"Full DF predictions: {current_full_df.filter(pl.col('prediction') != 0.0).height} non-zero")
        print(f"Current DF predictions: {current_df.filter(pl.col('prediction') != 0.0).height} non-zero")
        
        # Save statistics before redirecting
        submit_component.save_statistics(current_full_df, user_info)
        
        # Convert to dict format for storage
        def convert_df_to_dict(df: pl.DataFrame) -> Dict[str, List[Any]]:
            return {
                'time': df.get_column('time').dt.strftime('%Y-%m-%dT%H:%M:%S').to_list(),
                'gl': df.get_column('gl').to_list(),
                'prediction': df.get_column('prediction').to_list(),
                'age': df.get_column('age').to_list(),
                'user_id': df.get_column('user_id').to_list()
            }
        
        return '/ending', user_info, convert_df_to_dict(current_full_df), convert_df_to_dict(current_df)
    return no_update, no_update, no_update, no_update

@app.callback(
    [Output('url', 'pathname', allow_duplicate=True),
     Output('user-info-store', 'data', allow_duplicate=True),
     Output('full-df', 'data', allow_duplicate=True),
     Output('current-window-df', 'data', allow_duplicate=True)],
    [Input('exit-button', 'n_clicks')],
    [State('full-df', 'data'),
     State('current-window-df', 'data')],
    prevent_initial_call=True
)
def exit_prediction(n_clicks: Optional[int], full_df_data: Optional[Dict[str, Any]], current_df_data: Optional[Dict[str, Any]]) -> Tuple[Union[str, Any], Union[None, Any], Union[Dict[str, Any], Any], Union[Dict[str, Any], Any]]:
    print(f"DEBUG: exit_prediction called with n_clicks: {n_clicks}")
    if n_clicks and n_clicks > 0:  # Only proceed if actually clicked
        print("DEBUG: Exit button clicked, returning to startup page")
        
        # If data stores are None, just load fresh data and reset
        if not full_df_data or not current_df_data:
            print("DEBUG: No stored data, loading fresh data for reset")
            # Load fresh data
            global full_df, events_df
            full_df, events_df = load_glucose_data()
            df = full_df.slice(0, DEFAULT_POINTS)
            
            # Reset predictions
            full_df = full_df.with_columns(pl.lit(0.0).alias("prediction"))
            df = df.with_columns(pl.lit(0.0).alias("prediction"))
            
            return '/', None, full_df.to_dict(as_series=False), df.to_dict(as_series=False)
        
        # Create DataFrames from the stored data
        full_df = reconstruct_dataframe_from_dict(full_df_data)
        current_df = reconstruct_dataframe_from_dict(current_df_data)
        
        # Reset predictions in both dataframes
        full_df = full_df.with_columns(pl.lit(0.0).alias("prediction"))
        current_df = current_df.with_columns(pl.lit(0.0).alias("prediction"))
        
        # Return to startup page, clear user info, and return reset dataframes
        return '/', None, full_df.to_dict(as_series=False), current_df.to_dict(as_series=False)
    
    print("DEBUG: Exit callback triggered but not clicked, ignoring")
    return no_update, no_update, no_update, no_update

@app.callback(
    Output('last-click-time', 'data'),
    [
        Input('glucose-graph-graph', 'clickData'),
        Input('glucose-graph-graph', 'relayoutData'),
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
    if trigger_id == 'glucose-graph-graph' and click_data:
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
    if trigger_id == 'glucose-graph-graph' and relayout_data:
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
    Output('glucose-graph-graph', 'figure'),
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
    metrics = metrics_component.calculate_error_metrics(df, prediction_row)
    
    return metrics



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
    [Input('url', 'pathname')],
    prevent_initial_call=False
)
def initialize_data(pathname: Optional[str]) -> Tuple[Union[Dict[str, List[Any]], Any], Union[Dict[str, List[Any]], Any], Union[Dict[str, List[Any]], Any]]:
    """Initialize the data stores on page load"""
    global full_df, df, events_df
    
    # Only reset data when going to startup or prediction pages, not ending page
    if pathname != '/ending':
        print(f"DEBUG: initialize_data called for pathname: {pathname}")
        # Load fresh data
        full_df, events_df = load_glucose_data()
        df = full_df.slice(0, DEFAULT_POINTS)
        
        # Reset predictions
        full_df = full_df.with_columns(pl.lit(0.0).alias("prediction"))
        df = df.with_columns(pl.lit(0.0).alias("prediction"))
    else:
        print("DEBUG: initialize_data skipped for ending page")
        # Don't reset data when on ending page - use existing data
        return no_update, no_update, no_update
    
    # Convert DataFrames to JSON-serializable dictionaries
    def convert_df_to_dict(df: pl.DataFrame) -> Dict[str, List[Any]]:
        return {
            'time': df.get_column('time').dt.strftime('%Y-%m-%dT%H:%M:%S').to_list(),
            'gl': df.get_column('gl').to_list(),
            'prediction': df.get_column('prediction').to_list(),
            'age': df.get_column('age').to_list(),
            'user_id': df.get_column('user_id').to_list()
        }
    
    def convert_events_df_to_dict(df: pl.DataFrame) -> Dict[str, List[Any]]:
        return {
            'time': df.get_column('time').dt.strftime('%Y-%m-%dT%H:%M:%S').to_list(),
            'event_type': df.get_column('event_type').to_list(),
            'event_subtype': df.get_column('event_subtype').to_list(),
            'insulin_value': df.get_column('insulin_value').to_list()
        }
    
    return (
        convert_df_to_dict(full_df),
        convert_df_to_dict(df),
        convert_events_df_to_dict(events_df)
    )

@app.callback(
    [Output('url', 'pathname'),
     Output('user-info-store', 'data')],
    [Input('start-button', 'n_clicks')],
    [State('email-input', 'value'),
     State('age-input', 'value'),
     State('gender-dropdown', 'value'),
     State('diabetic-dropdown', 'value'),
     State('diabetic-type-dropdown', 'value'),
     State('diabetes-duration-input', 'value'),
     State('medical-conditions-dropdown', 'value'),
     State('medical-conditions-input', 'value'),
     State('location-input', 'value')],
    prevent_initial_call=True
)
def start_prediction(n_clicks: Optional[int], email: Optional[str], age: Optional[int], gender: Optional[str], 
                    diabetic: Optional[str], diabetic_type: Optional[str], diabetes_duration: Optional[int], 
                    medical_conditions: Optional[List[str]], medical_conditions_input: Optional[str], location: Optional[str]) -> Tuple[str, Optional[Dict[str, Any]]]:
    if n_clicks and email and age:
        return '/prediction', {
            'email': email,
            'age': age,
            'gender': gender,
            'diabetic': diabetic,
            'diabetic_type': diabetic_type,
            'diabetes_duration': diabetes_duration,
            'other_medical_conditions': medical_conditions,
            'medical_conditions_input': medical_conditions_input,
            'location': location
        }
    return '/', None

@app.callback(
    Output('user-info-store', 'data', allow_duplicate=True),
    [Input('prediction-table-data', 'data')],
    [State('user-info-store', 'data')],
    prevent_initial_call=True
)
def update_user_info_with_table_data(table_data: Optional[TableData], user_info: Optional[Dict[str, Any]]) -> Union[Dict[str, Any], Any]:
    if user_info is None:
        return no_update
    user_info['prediction_table_data'] = table_data
    return user_info

def reconstruct_dataframe_from_dict(df_data: Dict) -> pl.DataFrame:
    """Safely reconstruct a Polars DataFrame from a dictionary with proper type handling."""
    return pl.DataFrame({
        'time': pl.Series(df_data['time']).str.strptime(pl.Datetime, format='%Y-%m-%dT%H:%M:%S'),
        'gl': pl.Series(df_data['gl'], dtype=pl.Float64),
        'prediction': pl.Series(df_data['prediction'], dtype=pl.Float64),
        'age': pl.Series([int(float(x)) for x in df_data['age']], dtype=pl.Int64),
        'user_id': pl.Series([int(float(x)) for x in df_data['user_id']], dtype=pl.Int64)
    })

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

def main() -> None:
    """Starts the Dash server."""
    prediction_table.register_callbacks(app)  # Register the prediction table callbacks
    startup_page.register_callbacks(app)  # Register the startup page callbacks
    ending_page.register_callbacks(app)  # Register the ending page callbacks
    app.run(debug=True)

if __name__ == '__main__':
    main()
