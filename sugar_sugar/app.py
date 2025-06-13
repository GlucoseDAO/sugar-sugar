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
metrics_component = MetricsComponent()
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

    html.Div(id='page-content', children=[startup_page])  # Initialize with startup page
])

@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')],
    [State('user-info-store', 'data'),
     State('full-df', 'data'),
     State('events-df', 'data')],
    prevent_initial_call=False
)
def display_page(pathname: Optional[str], user_info: Optional[Dict[str, Any]], 
                full_df_data: Optional[Dict], events_df_data: Optional[Dict]) -> html.Div:
    print(f"DEBUG: display_page called with pathname: {pathname}")
    
    if pathname == '/prediction' and user_info:
        return create_prediction_layout()
    elif pathname == '/ending':
        return create_ending_layout(full_df_data, events_df_data)
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

def create_ending_layout(full_df_data: Optional[Dict], events_df_data: Optional[Dict]) -> html.Div:
    """Create the ending page layout"""
    if not full_df_data or not events_df_data:
        print("DEBUG: No data available for ending page")
        return html.Div("No data available", style={'textAlign': 'center', 'padding': '50px'})
    
    print("DEBUG: Creating ending page with stored data")
    
    # Reconstruct DataFrames from stored data
    full_df = reconstruct_dataframe_from_dict(full_df_data)
    events_df = reconstruct_events_dataframe_from_dict(events_df_data)
    
    # Use only the first DEFAULT_POINTS for display
    df = full_df.slice(0, DEFAULT_POINTS)
    
    # Check if we have any predictions
    prediction_count = df.filter(pl.col("prediction") != 0.0).height
    print(f"DEBUG: Found {prediction_count} predictions in ending page data")
    
    if prediction_count == 0:
        print("DEBUG: No predictions found")
        return html.Div("No predictions to display", style={'textAlign': 'center', 'padding': '50px'})
    
    # Create new components with the updated data
    prediction_table = PredictionTableComponent(df)
    
    # Update glucose chart and get the figure
    figure = ending_page.glucose_chart.update(df, events_df)
    
    # Create the page content with metrics container that will be populated by the callback
    return html.Div([
        html.H1("Prediction Summary", style={'textAlign': 'center', 'marginBottom': '20px'}),
        
        # Graph section
        html.Div([
            dcc.Graph(
                id='ending-glucose-graph-graph',
                figure=figure,
                config={
                    'displayModeBar': True,
                    'scrollZoom': False,
                    'doubleClick': 'reset',
                    'displaylogo': False,
                },
                style={'height': '500px'}
            )
        ], style={'marginBottom': '20px', 'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
        
        # Prediction table section
        html.Div([
            html.H3("Prediction Results", style={'textAlign': 'center'}),
            *prediction_table.children  # Unpack the children list
        ], style={'marginBottom': '20px'}),
        
        # Metrics section - now using stored metrics
        html.Div(
            id='ending-metrics-container',  # Different ID for ending page metrics
            style={
                'padding': '20px',
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'marginBottom': '20px'
            }
        ),
        
        # Buttons section
        html.Div([
            dbc.Button("Exit", id='exit-button', color="secondary")
        ], style={'textAlign': 'center', 'marginTop': '20px'})
    ], style={
        'maxWidth': '1200px',
        'margin': '0 auto',
        'padding': '20px'
    })

def reconstruct_events_dataframe_from_dict(events_data: Dict) -> pl.DataFrame:
    """Reconstruct the events DataFrame from stored data.""" 
    return pl.DataFrame({
        'time': pl.Series(events_data['time']).str.strptime(pl.Datetime, format='%Y-%m-%dT%H:%M:%S'),
        'event_type': pl.Series(events_data['event_type'], dtype=pl.String),
        'event_subtype': pl.Series(events_data['event_subtype'], dtype=pl.String),
        'insulin_value': pl.Series(events_data['insulin_value'], dtype=pl.Float64)
    })

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
def handle_start_button(n_clicks, email, age, gender, diabetic, diabetic_type, 
                       diabetes_duration, medical_conditions, medical_conditions_input, location):
    """Handle start button on startup page"""
    if n_clicks and email and age:
        print("DEBUG: Start button clicked")
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
    return no_update, no_update

@app.callback(
    [Output('url', 'pathname', allow_duplicate=True),
     Output('user-info-store', 'data', allow_duplicate=True)],
    [Input('submit-button', 'n_clicks')],
    [State('user-info-store', 'data')],
    prevent_initial_call=True
)
def handle_submit_button(n_clicks, user_info):
    """Handle submit button on prediction page"""
    if n_clicks:
        print("DEBUG: Submit button clicked")
        global full_df, df
        
        # Use current global DataFrames with latest predictions
        current_full_df = full_df.clone()
        current_df = df.clone()
        
        # Update age and user_id from user_info
        if user_info and 'age' in user_info:
            current_full_df = current_full_df.with_columns(pl.lit(int(user_info['age'])).alias("age"))
            current_df = current_df.with_columns(pl.lit(int(user_info['age'])).alias("age"))
        
        # Save statistics before redirecting
        submit_component.save_statistics(current_full_df, user_info)
        
        return '/ending', user_info
    return no_update, no_update

@app.callback(
    [Output('url', 'pathname', allow_duplicate=True),
     Output('user-info-store', 'data', allow_duplicate=True)],
    [Input('exit-button', 'n_clicks')],
    prevent_initial_call=True
)
def handle_exit_button(n_clicks):
    """Handle exit button"""
    if n_clicks:
        print("DEBUG: Exit button clicked")
        global full_df, df, events_df
        
        # Load fresh data and reset
        full_df, events_df = load_glucose_data()
        df = full_df.slice(0, DEFAULT_POINTS)
        
        # Reset predictions
        full_df = full_df.with_columns(pl.lit(0.0).alias("prediction"))
        df = df.with_columns(pl.lit(0.0).alias("prediction"))
        
        return '/', None
    return no_update, no_update

@app.callback(
    [Output('full-df', 'data'),
     Output('current-window-df', 'data'),
     Output('events-df', 'data')],
    [Input('url', 'pathname')],
    prevent_initial_call=False
)
def handle_url_changes(pathname: str):
    """Handle URL changes"""
    global full_df, df, events_df
    
    if pathname == '/ending':
        # Don't reset data when on ending page
        return no_update, no_update, no_update
    
    # For startup and prediction pages, initialize fresh data if needed
    print(f"DEBUG: URL changed to {pathname}, initializing data")
    full_df, events_df = load_glucose_data()
    df = full_df.slice(0, DEFAULT_POINTS)
    
    # Reset predictions
    full_df = full_df.with_columns(pl.lit(0.0).alias("prediction"))
    df = df.with_columns(pl.lit(0.0).alias("prediction"))
    
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

# Consolidated callback for last-click-time updates
@app.callback(
    Output('last-click-time', 'data'),
    [Input('glucose-graph-graph', 'clickData'),
     Input('glucose-graph-graph', 'relayoutData'),
     Input('upload-data', 'contents'),
     Input('points-control', 'value'),
     Input('time-slider', 'value')],
    [State('last-click-time', 'data'),
     State('upload-data', 'filename'),
     State('time-slider', 'value'),
     State('points-control', 'value')],
)
def handle_all_interactions(click_data, relayout_data, upload_contents, points_value, slider_value,
                           last_click_time, filename, current_position, current_points):
    """Consolidated callback for all interactions that update last-click-time"""
    global df, full_df, events_df, is_example_data
    
    current_time = int(time.time() * 1000)
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return last_click_time
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Handle file upload
    if trigger_id == 'upload-data' and upload_contents:
        try:
            content_type, content_string = upload_contents.split(',')
            decoded = base64.b64decode(content_string)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(decoded)
                tmp_path = Path(tmp_file.name)
            
            new_full_df, new_events_df = load_glucose_data(tmp_path)
            full_df = new_full_df
            events_df = new_events_df
            df = full_df.slice(0, DEFAULT_POINTS)
            is_example_data = False
            tmp_path.unlink()
            
        except Exception as e:
            print(f"Error loading file: {e}")
        
        return current_time
    
    # Handle points control
    elif trigger_id == 'points-control':
        points = max(MIN_POINTS, min(MAX_POINTS, points_value))
        new_max = len(full_df) - points
        new_start = min(current_position, new_max)
        new_start = max(0, new_start)
        df = full_df.slice(new_start, new_start + points)
        return current_time
    
    # Handle time slider
    elif trigger_id == 'time-slider':
        df = full_df.slice(slider_value, slider_value + current_points)
        if len(df) != current_points:
            df = df.head(current_points)
        return current_time
    
    # Handle graph interactions (existing logic)
    elif trigger_id == 'glucose-graph-graph':
        if click_data:
            if current_time - last_click_time <= DOUBLE_CLICK_THRESHOLD:
                print("Double-click detected: Resetting drawn lines.")
                full_df = full_df.with_columns(pl.lit(0.0).alias("prediction"))
                df = df.with_columns(pl.lit(0.0).alias("prediction"))
                return current_time
            
            point_data = click_data['points'][0]
            click_x = point_data['x']
            click_y = point_data['y']
            
            nearest_time = find_nearest_time(click_x)
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
            return current_time
        
        elif relayout_data and 'shapes' in relayout_data:
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



# Add simplified callbacks for UI updates only
@app.callback(
    [Output('example-data-warning', 'children'),
     Output('time-slider', 'max'),
     Output('time-slider', 'value')],
    [Input('upload-data', 'contents'),
     Input('points-control', 'value')],
    [State('upload-data', 'filename'),
     State('time-slider', 'value')],
    prevent_initial_call=True
)
def update_ui_components(upload_contents, points_value, filename, current_position):
    """Update UI components based on file upload and points control"""
    global full_df, is_example_data
    
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'upload-data' and upload_contents:
        if is_example_data:
            success_msg = html.Div([
                html.I(className="fas fa-check-circle", style={'marginRight': '8px'}),
                f"Successfully loaded data from {filename}"
            ], style={
                'color': '#2f855a',
                'backgroundColor': '#c6f6d5',
                'padding': '10px',
                'borderRadius': '5px',
                'textAlign': 'center'
            })
        else:
            success_msg = None
        
        # Return current slider settings
        points = max(MIN_POINTS, min(MAX_POINTS, points_value))
        new_max = len(full_df) - points
        new_start = min(current_position, new_max)
        return success_msg, new_max, max(0, new_start)
    
    elif trigger_id == 'points-control':
        points = max(MIN_POINTS, min(MAX_POINTS, points_value))
        new_max = len(full_df) - points
        new_start = min(current_position, new_max)
        
        # Show warning if using example data
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
        
        return warning, new_max, max(0, new_start)
    
    return no_update, no_update, no_update

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

@app.callback(
    Output('ending-metrics-container', 'children'),
    [Input('metrics-store', 'data')],
    prevent_initial_call=True
)
def update_ending_metrics_display(stored_metrics: Optional[Dict[str, Any]]) -> List[html.Div]:
    """Updates the ending page metrics display based on stored metrics."""
    return MetricsComponent.create_ending_metrics_display(stored_metrics)

def main() -> None:
    """Starts the Dash server."""
    prediction_table.register_callbacks(app)  # Register the prediction table callbacks
    metrics_component.register_callbacks(app, prediction_table)  # Register the metrics component callbacks
    startup_page.register_callbacks(app)  # Register the startup page callbacks
    ending_page.register_callbacks(app)  # Register the ending page callbacks
    app.run(debug=True)

if __name__ == '__main__':
    main()
