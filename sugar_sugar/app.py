from typing import List, Dict, Tuple, Optional, Any, Union
import dash
from dash import dcc, html, Output, Input, State, no_update, dash_table
import plotly.graph_objs as go

import polars as pl
from datetime import datetime
import time
from pathlib import Path
import base64
import tempfile
import dash_bootstrap_components as dbc
import os
import typer

from sugar_sugar.data import load_glucose_data
from sugar_sugar.config import DEFAULT_POINTS, MIN_POINTS, MAX_POINTS, DOUBLE_CLICK_THRESHOLD, PREDICTION_HOUR_OFFSET, DEBUG_MODE
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

# Load example data once at startup for initial session storage
example_full_df, example_events_df = load_glucose_data()  # Unpack both dataframes

external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    dbc.themes.BOOTSTRAP,
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
]

app = dash.Dash(__name__, 
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True
)
app.title = "Sugar Sugar - Glucose Prediction Game"



# Create component instances
glucose_chart = GlucoseChart(id='glucose-graph', hide_last_hour=True)  # Hide last hour in prediction page
prediction_table = PredictionTableComponent()
metrics_component = MetricsComponent()
submit_component = SubmitComponent()
# startup_page will be created in main() after debug mode is set
startup_page = None  # Will be initialized in main()
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

    html.Div(id='page-content', children=[])  # Will be populated in main()
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
        return create_ending_layout(full_df_data, events_df_data, user_info)
    return startup_page if startup_page else html.Div("Loading...")  # Return the startup page component

def create_prediction_layout() -> html.Div:
    """Create the prediction page layout"""
    return html.Div([
        HeaderComponent(show_time_slider=False),  # Hide the time slider on prediction page
        html.Div([
            glucose_chart,
            submit_component
        ], style={'flex': '1'})
    ], style={
        'margin': '0 auto',
        'padding': '0 20px',
        'display': 'flex',
        'flexDirection': 'column',
        'gap': '20px'
    })

def create_ending_layout(full_df_data: Optional[Dict], events_df_data: Optional[Dict], user_info: Optional[Dict] = None) -> html.Div:
    """Create the ending page layout"""
    if not full_df_data or not events_df_data:
        print("DEBUG: No data available for ending page")
        return html.Div("No data available", style={'textAlign': 'center', 'padding': '50px'})
    
    print("DEBUG: Creating ending page with stored data")
    
    # Reconstruct DataFrames from stored data
    full_df = reconstruct_dataframe_from_dict(full_df_data)
    events_df = reconstruct_events_dataframe_from_dict(events_df_data)
    
    # Use the same window that was used for predictions if available
    if user_info and 'prediction_window_start' in user_info and 'prediction_window_size' in user_info:
        window_start = user_info['prediction_window_start']
        window_size = user_info['prediction_window_size']
        df = full_df.slice(window_start, window_size)
        print(f"DEBUG: Using prediction window starting at {window_start} with size {window_size}")
    else:
        # Fallback to first DEFAULT_POINTS for display
        df = full_df.slice(0, DEFAULT_POINTS)
        print("DEBUG: No prediction window info found, using default first 24 points")
    
    # Check if we have any predictions
    prediction_count = df.filter(pl.col("prediction") != 0.0).height
    print(f"DEBUG: Found {prediction_count} predictions in ending page data")
    
    if prediction_count == 0:
        print("DEBUG: No predictions found")
        return html.Div("No predictions to display", style={'textAlign': 'center', 'padding': '50px'})
    
    # Create new components with the updated data
    ending_prediction_table = PredictionTableComponent()
    
    # Create a new glucose chart for the ending page and get the figure (show complete data)
    ending_glucose_chart = GlucoseChart(id='ending-glucose-chart', hide_last_hour=False)
    figure = ending_glucose_chart._build_figure(df, events_df)
    
    # Calculate metrics directly from the data
    metrics_component_ending = MetricsComponent()
    table_data = metrics_component_ending._generate_table_data(df)
    stored_metrics = None
    
    if len(table_data) >= 2:  # Need at least actual and predicted rows
        stored_metrics = metrics_component_ending._calculate_metrics_from_table_data(table_data)
    
    # Create metrics display directly
    metrics_display = MetricsComponent.create_ending_metrics_display(stored_metrics) if stored_metrics else [
        html.H3("Accuracy Metrics", style={'textAlign': 'center'}),
        html.Div(
            "No metrics available - insufficient prediction data", 
            style={
                'color': 'gray',
                'fontStyle': 'italic',
                'fontSize': '16px',
                'padding': '10px',
                'textAlign': 'center'
            }
        )
    ]

    # Create the page content with metrics container that will be populated by the callback
    return html.Div([
        # Add a scroll-to-top trigger element
        html.Div(id='scroll-to-top-trigger', style={'display': 'none'}),
        html.H1("Prediction Summary", style={
            'textAlign': 'center', 
            'marginBottom': '20px',
            'fontSize': 'clamp(24px, 4vw, 48px)',  # Responsive font size
            'padding': '0 10px'
        }),
        
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
                    'responsive': True  # Make graph responsive
                },
                style={
                    'height': 'clamp(300px, 50vh, 600px)',  # Responsive height
                    'width': '100%'
                }
            )
        ], style={
            'marginBottom': '20px', 
            'padding': 'clamp(10px, 2vw, 20px)', 
            'backgroundColor': 'white', 
            'borderRadius': '10px', 
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'width': '100%',
            'boxSizing': 'border-box'
        }),
        
        # Prediction table section with responsive flexbox layout
        html.Div([
            html.H3("Prediction Results", style={
                'textAlign': 'center', 
                'marginBottom': '15px',
                'fontSize': 'clamp(18px, 3vw, 24px)'  # Responsive font size
            }),
            *ending_prediction_table.children  # Unpack the children list
        ], style={
            'marginBottom': '20px',
            'padding': 'clamp(10px, 2vw, 20px)',
            'backgroundColor': 'white',
            'borderRadius': '10px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'display': 'flex',
            'flexDirection': 'column',
            'width': '100%',
            'boxSizing': 'border-box',
            'overflowX': 'auto'  # Allow horizontal scroll for table if needed
        }),
        
        # Metrics section - now directly calculated and displayed
        html.Div(
            metrics_display,
            style={
                'padding': 'clamp(10px, 2vw, 20px)',
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'marginBottom': '20px',
                'width': '100%',
                'boxSizing': 'border-box'
            }
        ),
        
        # Buttons section
        html.Div([
            html.Button(
                'Exit',
                id='exit-button',
                autoFocus=False,
                style={
                    'backgroundColor': '#6c757d',  # Bootstrap secondary color
                    'color': 'white',
                    'padding': 'clamp(15px, 2vw, 20px) clamp(20px, 3vw, 30px)',
                    'border': 'none',
                    'borderRadius': '5px',
                    'fontSize': 'clamp(18px, 3vw, 24px)',  # Responsive font size
                    'cursor': 'pointer',
                    'minWidth': '200px',
                    'maxWidth': '400px',
                    'width': '100%',
                    'height': 'clamp(60px, 8vh, 80px)',  # Responsive height
                    'display': 'flex',
                    'alignItems': 'center',
                    'justifyContent': 'center',
                    'lineHeight': '1.2'
                }
            )
        ], style={
            'display': 'flex',
            'justifyContent': 'center',
            'alignItems': 'center',
            'marginTop': '20px',
            'padding': '0 10px'
        })
    ], style={
        'maxWidth': '100%',  # Allow full width usage
        'width': '100%',
        'margin': '0 auto',
        'padding': 'clamp(10px, 2vw, 20px)',  # Responsive padding
        'display': 'flex',
        'flexDirection': 'column',
        'minHeight': '100vh',
        'gap': 'clamp(10px, 2vh, 20px)',  # Responsive gap
        'boxSizing': 'border-box'
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
    [State('user-info-store', 'data'),
     State('full-df', 'data'),
     State('current-window-df', 'data'),
     State('time-slider', 'value')],
    prevent_initial_call=True
)
def handle_submit_button(n_clicks, user_info, full_df_data, current_df_data, slider_value):
    """Handle submit button on prediction page"""
    if n_clicks and full_df_data and current_df_data:
        print("DEBUG: Submit button clicked")
        
        # Reconstruct DataFrames from session storage
        current_full_df = reconstruct_dataframe_from_dict(full_df_data)
        current_df = reconstruct_dataframe_from_dict(current_df_data)
        
        # Update age and user_id from user_info
        if user_info and 'age' in user_info:
            current_full_df = current_full_df.with_columns(pl.lit(int(user_info['age'])).alias("age"))
            current_df = current_df.with_columns(pl.lit(int(user_info['age'])).alias("age"))
        
        # Generate prediction table data directly from DataFrame instead of relying on component
        if user_info is None:
            user_info = {}
        
        # Store the window position information for the ending page
        user_info['prediction_window_start'] = slider_value or 0
        user_info['prediction_window_size'] = len(current_df)
        
        # Create a temporary prediction table component to generate the table data
        temp_prediction_table = PredictionTableComponent()
        prediction_table_data = temp_prediction_table._generate_table_data(current_df)
        user_info['prediction_table_data'] = prediction_table_data
        
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
        # Just redirect to home - data will be reset by URL change callback
        return '/', None
    return no_update, no_update

# Add client-side callback to scroll to top when ending page loads
app.clientside_callback(
    """
    function(pathname) {
        if (pathname === '/ending') {
            window.scrollTo(0, 0);
            return '';
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('scroll-to-top-trigger', 'children'),
    Input('url', 'pathname')
)

@app.callback(
    [Output('full-df', 'data'),
     Output('current-window-df', 'data'),
     Output('events-df', 'data'),
     Output('is-example-data', 'data')],
    [Input('url', 'pathname')],
    prevent_initial_call=False
)
def handle_url_changes(pathname: str):
    """Handle URL changes and initialize session storage"""
    
    if pathname == '/ending':
        # Don't reset data when on ending page
        return no_update, no_update, no_update, no_update
    
    # For startup and prediction pages, initialize fresh data
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
        convert_events_df_to_dict(events_df),
        True  # is_example_data = True by default
    )

# Consolidated callback for last-click-time updates
@app.callback(
    [Output('last-click-time', 'data'),
     Output('full-df', 'data', allow_duplicate=True),
     Output('current-window-df', 'data', allow_duplicate=True),
     Output('events-df', 'data', allow_duplicate=True),
     Output('is-example-data', 'data', allow_duplicate=True)],
    [Input('glucose-graph-graph', 'clickData'),
     Input('glucose-graph-graph', 'relayoutData'),
     Input('upload-data', 'contents'),
     Input('points-control', 'value'),
     Input('time-slider', 'value')],
    [State('last-click-time', 'data'),
     State('upload-data', 'filename'),
     State('time-slider', 'value'),
     State('points-control', 'value'),
     State('full-df', 'data'),
     State('current-window-df', 'data'),
     State('events-df', 'data'),
     State('is-example-data', 'data')],
    prevent_initial_call=True
)
def handle_all_interactions(click_data, relayout_data, upload_contents, points_value, slider_value,
                           last_click_time, filename, current_position, current_points,
                           full_df_data, current_df_data, events_df_data, is_example_data):
    """Consolidated callback for all interactions that update last-click-time and data"""
    
    current_time = int(time.time() * 1000)
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return last_click_time, no_update, no_update, no_update, no_update
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Helper functions
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
    
    # Get current data from session storage
    if not full_df_data or not current_df_data or not events_df_data:
        return last_click_time, no_update, no_update, no_update, no_update
    
    full_df = reconstruct_dataframe_from_dict(full_df_data)
    df = reconstruct_dataframe_from_dict(current_df_data)
    events_df = reconstruct_events_dataframe_from_dict(events_df_data)
    
    # Handle file upload
    if trigger_id == 'upload-data' and upload_contents:
        try:
            content_type, content_string = upload_contents.split(',')
            decoded = base64.b64decode(content_string)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(decoded)
                tmp_path = Path(tmp_file.name)
            
            new_full_df, new_events_df = load_glucose_data(tmp_path)
            new_df = new_full_df.slice(0, DEFAULT_POINTS)
            tmp_path.unlink()
            
            return (current_time, 
                   convert_df_to_dict(new_full_df),
                   convert_df_to_dict(new_df),
                   convert_events_df_to_dict(new_events_df),
                   False)  # is_example_data = False for uploaded files
            
        except Exception as e:
            print(f"Error loading file: {e}")
            return current_time, no_update, no_update, no_update, no_update
    
    # Handle points control
    elif trigger_id == 'points-control':
        points = max(MIN_POINTS, min(MAX_POINTS, points_value))
        new_max = len(full_df) - points
        new_start = min(current_position, new_max)
        new_start = max(0, new_start)
        new_df = full_df.slice(new_start, new_start + points)
        
        return (current_time,
               no_update,
               convert_df_to_dict(new_df),
               no_update,
               no_update)
    
    # Handle time slider
    elif trigger_id == 'time-slider':
        new_df = full_df.slice(slider_value, slider_value + current_points)
        if len(new_df) != current_points:
            new_df = new_df.head(current_points)
        
        return (current_time,
               no_update,
               convert_df_to_dict(new_df),
               no_update,
               no_update)
    
    # Handle graph interactions
    elif trigger_id == 'glucose-graph-graph':
        if click_data:
            if current_time - last_click_time <= DOUBLE_CLICK_THRESHOLD:
                print("Double-click detected: Resetting drawn lines.")
                full_df = full_df.with_columns(pl.lit(0.0).alias("prediction"))
                df = df.with_columns(pl.lit(0.0).alias("prediction"))
                
                return (current_time,
                       convert_df_to_dict(full_df),
                       convert_df_to_dict(df),
                       no_update,
                       no_update)
            
            point_data = click_data['points'][0]
            click_x = point_data['x']
            click_y = point_data['y']
            
            # Restrict clicks to prediction area only (after PREDICTION_HOUR_OFFSET)
            visible_points = len(df) - PREDICTION_HOUR_OFFSET
            if click_x < visible_points:
                print(f"Click at x={click_x} is outside prediction area (starts at x={visible_points}). Ignoring click.")
                return (last_click_time, no_update, no_update, no_update, no_update)
            
            nearest_time = find_nearest_time(click_x, df)
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
            
            return (current_time,
                   convert_df_to_dict(full_df),
                   convert_df_to_dict(df),
                   no_update,
                   no_update)
        
        elif relayout_data and 'shapes' in relayout_data:
            shapes = relayout_data['shapes']
            if shapes and len(shapes) > 0:
                latest_shape = shapes[-1]
                
                start_x = latest_shape.get('x0')
                end_x = latest_shape.get('x1')
                start_y = latest_shape.get('y0')
                end_y = latest_shape.get('y1')
                
                if all(v is not None for v in [start_x, end_x, start_y, end_y]):
                    # Restrict drawing to prediction area only (after PREDICTION_HOUR_OFFSET)
                    visible_points = len(df) - PREDICTION_HOUR_OFFSET
                    if start_x < visible_points or end_x < visible_points:
                        print(f"Drawing area partially outside prediction area (starts at x={visible_points}). Ignoring shape.")
                        return (last_click_time, no_update, no_update, no_update, no_update)
                    
                    start_time = find_nearest_time(start_x, df)
                    
                    # Calculate the intersection with the first vertical guideline after start
                    actual_end_x, actual_end_y = calculate_first_guideline_intersection(
                        start_x, start_y, end_x, end_y, df
                    )
                    end_time = find_nearest_time(actual_end_x, df)
                    
                    # Get intermediate prediction points every 5 minutes
                    intermediate_points = create_intermediate_predictions(start_time, end_time, float(start_y), float(actual_end_y), df)
                    
                    # Collect all times that need prediction values
                    all_prediction_times = [start_time, end_time]
                    all_prediction_values = [float(start_y), float(actual_end_y)]
                    
                    # Add intermediate points
                    for time_point, glucose_value in intermediate_points:
                        all_prediction_times.append(time_point)
                        all_prediction_values.append(glucose_value)
                    
                    # Create a mapping for the predictions
                    time_to_value = dict(zip(all_prediction_times, all_prediction_values))
                    
                    # Update both DataFrames with all prediction points
                    full_df = full_df.with_columns(
                        pl.when(pl.col("time").is_in(all_prediction_times))
                        .then(
                            # Use a series of when conditions to map each time to its value
                            pl.when(pl.col("time") == start_time)
                            .then(float(start_y))
                            .when(pl.col("time") == end_time)
                            .then(float(actual_end_y))
                            .otherwise(
                                # For intermediate points, we need to match them individually
                                pl.col("time").map_elements(
                                    lambda x: time_to_value.get(x, 0.0),
                                    return_dtype=pl.Float64
                                )
                            )
                        )
                        .otherwise(pl.col("prediction"))
                        .alias("prediction")
                    )
                    df = df.with_columns(
                        pl.when(pl.col("time").is_in(all_prediction_times))
                        .then(
                            # Use a series of when conditions to map each time to its value
                            pl.when(pl.col("time") == start_time)
                            .then(float(start_y))
                            .when(pl.col("time") == end_time)
                            .then(float(actual_end_y))
                            .otherwise(
                                # For intermediate points, we need to match them individually
                                pl.col("time").map_elements(
                                    lambda x: time_to_value.get(x, 0.0),
                                    return_dtype=pl.Float64
                                )
                            )
                        )
                        .otherwise(pl.col("prediction"))
                        .alias("prediction")
                    )
                    
                    return (current_time,
                           convert_df_to_dict(full_df),
                           convert_df_to_dict(df),
                           no_update,
                           no_update)
    
    return last_click_time, no_update, no_update, no_update, no_update

# Add callback for random slider initialization when prediction page components are ready
@app.callback(
    [Output('time-slider', 'value', allow_duplicate=True),
     Output('current-window-df', 'data', allow_duplicate=True)],
    [Input('time-slider', 'max')],  # Triggers when slider is created and max is set
    [State('url', 'pathname'),
     State('full-df', 'data'),
     State('points-control', 'value')],
    prevent_initial_call=True
)
def randomize_slider_on_prediction_page(slider_max: int, pathname: str, full_df_data: Optional[Dict], points_value: int):
    """Set slider to random position when time-slider component is ready on prediction page"""
    if pathname == '/prediction' and full_df_data and slider_max is not None:
        import random
        full_df = reconstruct_dataframe_from_dict(full_df_data)
        points = max(MIN_POINTS, min(MAX_POINTS, points_value or DEFAULT_POINTS))
        max_start_index = len(full_df) - points
        
        if max_start_index > 0:
            # Generate random start position that is a multiple of the number of points
            # This ensures we get clean windows (0, 24, 48, 72, etc.)
            max_multiple = max_start_index // points
            if max_multiple > 0:
                random_multiple = random.randint(0, max_multiple)
                random_start = random_multiple * points
            else:
                random_start = 0
            
            print(f"DEBUG: Setting slider to random position {random_start} (multiple of {points})")
            
            # Update the data slice to match the random position
            new_df = full_df.slice(random_start, points)
            
            def convert_df_to_dict(df: pl.DataFrame) -> Dict[str, List[Any]]:
                return {
                    'time': df.get_column('time').dt.strftime('%Y-%m-%dT%H:%M:%S').to_list(),
                    'gl': df.get_column('gl').to_list(),
                    'prediction': df.get_column('prediction').to_list(),
                    'age': df.get_column('age').to_list(),
                    'user_id': df.get_column('user_id').to_list()
                }
            
            return random_start, convert_df_to_dict(new_df)
    
    return no_update, no_update

# Add simplified callbacks for UI updates only
@app.callback(
    [Output('example-data-warning', 'children'),
     Output('time-slider', 'max'),
     Output('time-slider', 'value')],
    [Input('upload-data', 'contents'),
     Input('points-control', 'value')],
    [State('upload-data', 'filename'),
     State('time-slider', 'value'),
     State('full-df', 'data'),
     State('is-example-data', 'data')],
    prevent_initial_call=True
)
def update_ui_components(upload_contents, points_value, filename, current_position, full_df_data, is_example_data):
    """Update UI components based on file upload and points control"""
    
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if not full_df_data:
        return no_update, no_update, no_update
    
    # Reconstruct full_df to get its length
    full_df = reconstruct_dataframe_from_dict(full_df_data)
    
    if trigger_id == 'upload-data' and upload_contents:
        if not is_example_data:  # File was successfully uploaded
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

def calculate_first_guideline_intersection(start_x: float, start_y: float, end_x: float, end_y: float, df: pl.DataFrame) -> Tuple[float, float]:
    """
    Calculate the intersection of the drawn line with the first vertical guideline after the start point.
    Returns the (x, y) coordinates of the intersection with the next time marker.
    """
    # Find the next integer x position (vertical guideline) after start_x
    next_x = int(start_x) + 1
    
    # If the line doesn't extend past the next guideline, use the original end point
    if next_x >= end_x:
        return end_x, end_y
    
    # Make sure the next_x is within the DataFrame bounds
    if next_x >= len(df):
        next_x = len(df) - 1
    
    # Calculate the y-value at the intersection using linear interpolation
    if end_x != start_x:  # Avoid division by zero
        slope = (end_y - start_y) / (end_x - start_x)
        intersect_y = start_y + slope * (next_x - start_x)
    else:
        intersect_y = start_y
    
    return float(next_x), float(intersect_y)


def create_intermediate_predictions(start_time: datetime, end_time: datetime, start_y: float, end_y: float, df: pl.DataFrame) -> List[Tuple[datetime, float]]:
    """
    Create intermediate prediction points every 5 minutes between start and end points.
    Returns a list of (time, glucose_value) tuples for intermediate points.
    """
    from datetime import timedelta
    
    intermediate_points = []
    time_diff = end_time - start_time
    
    # Only create intermediate points if the difference is more than 5 minutes
    if time_diff.total_seconds() <= 5 * 60:  # 5 minutes in seconds
        return intermediate_points
    
    # Get all available times in the DataFrame between start and end
    available_times = (df
        .filter((pl.col("time") > start_time) & (pl.col("time") < end_time))
        .get_column("time")
        .to_list()
    )
    
    if not available_times:
        return intermediate_points
    
    # Calculate the total time range in minutes for interpolation
    total_minutes = time_diff.total_seconds() / 60
    
    # Create prediction points for times that are approximately every 5 minutes
    target_interval = 5  # minutes
    for i, time_point in enumerate(available_times):
        # Calculate how far along we are in the time range (0 to 1)
        time_from_start = time_point - start_time
        progress = time_from_start.total_seconds() / time_diff.total_seconds()
        
        # Check if this time point is approximately at a 5-minute interval
        minutes_from_start = time_from_start.total_seconds() / 60
        
        # Add point if it's close to a 5-minute interval (within 2.5 minutes)
        nearest_interval = round(minutes_from_start / target_interval) * target_interval
        if abs(minutes_from_start - nearest_interval) <= 2.5 and nearest_interval > 0 and nearest_interval < total_minutes:
            # Interpolate the glucose value
            interpolated_value = start_y + (end_y - start_y) * progress
            intermediate_points.append((time_point, interpolated_value))
    
    return intermediate_points


def find_nearest_time(x: Union[str, float, datetime], df: pl.DataFrame) -> datetime:
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
    if isinstance(x, str):
        x_ts = datetime.fromisoformat(x.replace('Z', '+00:00'))
    else:
        x_ts = x
    time_diffs = df.select([
        (pl.col("time").cast(pl.Int64) - pl.lit(int(x_ts.timestamp() * 1000)))
        .abs()
        .alias("diff")
    ])
    nearest_idx = time_diffs.select(pl.col("diff").arg_min()).item()
    return df.get_column("time")[nearest_idx]



# Create typer app
cli = typer.Typer()

@cli.command()
def main(debug: bool = typer.Option(False, "--debug", help="Enable debug mode to show test button")) -> None:
    """Starts the Dash server."""
    # Import config here to update the global DEBUG_MODE variable
    import sugar_sugar.config as config
    
    # Set the global debug mode based on command line argument
    config.DEBUG_MODE = debug
    
    # Create components after setting debug mode
    global startup_page
    startup_page = StartupPage()
    
    prediction_table.register_callbacks(app)  # Register the prediction table callbacks
    metrics_component.register_callbacks(app, prediction_table)  # Register the metrics component callbacks
    glucose_chart.register_callbacks(app)  # Register the glucose chart callbacks
    startup_page.register_callbacks(app)  # Register the startup page callbacks
    ending_page.register_callbacks(app)  # Register the ending page callbacks
    
    # Update the app layout with the new startup page
    app.layout.children[-1].children = [startup_page]
    
    app.run(debug=True)

def cli_main() -> None:
    """CLI entry point"""
    cli()

if __name__ == '__main__':
    cli()
