from typing import Any, Dict, List, Optional, Tuple, Union
import dash
from dash import dcc, html, Output, Input, State, no_update, dash_table
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go

import polars as pl
from datetime import datetime
import time
from pathlib import Path
import base64
import dash_bootstrap_components as dbc
import os
import typer
from dotenv import load_dotenv
from eliot import start_action, start_task
from pycomfort.logging import to_nice_file, to_nice_stdout

# Load environment variables from .env file in project root
project_root = Path(__file__).parent.parent
env_path = project_root / '.env'
load_dotenv(env_path)

logs_dir = project_root / 'logs'
logs_dir.mkdir(exist_ok=True)
to_nice_stdout()
to_nice_file(logs_dir / 'sugar_sugar.json', logs_dir / 'sugar_sugar.log')

from sugar_sugar.data import load_glucose_data
from sugar_sugar.config import DEFAULT_POINTS, MIN_POINTS, MAX_POINTS, DOUBLE_CLICK_THRESHOLD, PREDICTION_HOUR_OFFSET
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

def dataframe_to_store_dict(df_in: pl.DataFrame) -> Dict[str, List[Any]]:
    """Convert a Polars DataFrame into a session-store friendly dictionary."""
    return {
        'time': df_in.get_column('time').dt.strftime('%Y-%m-%dT%H:%M:%S').to_list(),
        'gl': df_in.get_column('gl').to_list(),
        'prediction': df_in.get_column('prediction').to_list(),
        'age': df_in.get_column('age').to_list(),
        'user_id': df_in.get_column('user_id').to_list()
    }


def events_dataframe_to_store_dict(df_in: pl.DataFrame) -> Dict[str, List[Any]]:
    """Convert an events Polars DataFrame into a session-store dictionary."""
    return {
        'time': df_in.get_column('time').dt.strftime('%Y-%m-%dT%H:%M:%S').to_list(),
        'event_type': df_in.get_column('event_type').to_list(),
        'event_subtype': df_in.get_column('event_subtype').to_list(),
        'insulin_value': df_in.get_column('insulin_value').to_list()
    }


def get_random_data_window(full_df: pl.DataFrame, points: int) -> Tuple[pl.DataFrame, int]:
    """
    Get a random window of data from the full DataFrame.
    
    Args:
        full_df: The full glucose DataFrame
        points: Number of points to include in the window
        
    Returns:
        Tuple of (windowed_df, random_start_index)
    """
    import random
    max_start_index = len(full_df) - points
    if max_start_index > 0:
        # Generate random start position that is a multiple of the number of points
        max_multiple = max_start_index // points
        if max_multiple > 0:
            random_multiple = random.randint(0, max_multiple)
            if random_multiple == 0 and max_multiple >= 1:
                random_multiple = random.randint(1, max_multiple)
            random_start = random_multiple * points
        else:
            random_start = 0
    else:
        random_start = 0
    
    windowed_df = full_df.slice(random_start, points)
    return windowed_df, random_start

# Load example data once at startup for initial session storage with randomization
example_full_df, example_events_df = load_glucose_data()  # Unpack both dataframes
example_full_df = example_full_df.with_columns(pl.lit(0.0).alias('prediction'))
example_initial_df, example_initial_start = get_random_data_window(example_full_df, DEFAULT_POINTS)
example_initial_df = example_initial_df.with_columns(pl.lit(0.0).alias('prediction'))

example_full_df_store = dataframe_to_store_dict(example_full_df)
example_initial_df_store = dataframe_to_store_dict(example_initial_df)
example_events_df_store = events_dataframe_to_store_dict(example_events_df)
example_initial_slider_value = example_initial_start

external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    dbc.themes.BOOTSTRAP,
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css',
    'https://cdnjs.cloudflare.com/ajax/libs/github-fork-ribbon-css/0.2.3/gh-fork-ribbon.min.css'
]

app = dash.Dash(__name__, 
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True
)
app.title = "Sugar Sugar - Glucose Prediction Game"

app.clientside_callback(
    "function() { return window.navigator.userAgent || ''; }",
    Output('user-agent', 'data'),
    Input('url', 'href'),
    prevent_initial_call=False
)



# Create component instances
glucose_chart = GlucoseChart(id='glucose-graph', hide_last_hour=True)  # Hide last hour in prediction page
prediction_table = PredictionTableComponent()
metrics_component = MetricsComponent()
submit_component = SubmitComponent()
header_component = HeaderComponent(show_time_slider=False, initial_slider_value=example_initial_slider_value)
# startup_page will be created in main() after debug mode is set
startup_page = None  # Will be initialized in main()
ending_page = EndingPage()

# Set initial layout to startup page
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='user-info-store', data=None),
    dcc.Store(id='last-click-time', data=0),
    dcc.Store(id='current-window-df', data=example_initial_df_store),
    dcc.Store(id='full-df', data=example_full_df_store),
    dcc.Store(id='events-df', data=example_events_df_store),
    dcc.Store(id='is-example-data', data=True),
    dcc.Store(id='data-source-name', data="example.csv"),  # Store source filename
    dcc.Store(id='randomization-initialized', data=False),  # Track if randomization has been done
    dcc.Store(id='glucose-chart-mode', data={'hide_last_hour': True}),
    dcc.Store(id='user-agent', data=None, storage_type='session'),
    dcc.Store(id='initial-slider-value', data=example_initial_slider_value),  # Store initial random start

    html.Div(id='mobile-warning', style={'margin': '12px 0'}),

    html.A(
        "Fork me on GitHub",
        href="https://github.com/GlucoseDAO/sugar-sugar",
        target="_blank",
        rel="noopener noreferrer",
        className="github-fork-ribbon github-fork-ribbon-right-bottom fixed",
        **{"data-ribbon": "Fork me on GitHub"}
    ),

    html.Div(id='page-content', children=[])  # Will be populated in main()
])



@app.callback(
    [Output('page-content', 'children'),
     Output('mobile-warning', 'children')],
    [Input('url', 'pathname')],
    [State('user-info-store', 'data'),
     State('full-df', 'data'),
     State('events-df', 'data'),
     State('user-agent', 'data')],
    prevent_initial_call=False
)
def display_page(pathname: Optional[str], user_info: Optional[Dict[str, Any]], 
                full_df_data: Optional[Dict], events_df_data: Optional[Dict], user_agent: Optional[str]) -> tuple[html.Div, Optional[html.Div]]:
    with start_action(action_type=u"display_page", pathname=pathname):
        warning_content = render_mobile_warning(user_agent)
        if pathname == '/prediction' and user_info:
            return create_prediction_layout(), warning_content
        if pathname == '/ending':
            # Check if we have the required data for ending page
            if not full_df_data or not user_info or 'prediction_table_data' not in user_info:
                return html.Div([
                    html.H2("Session Expired", style={'textAlign': 'center', 'marginTop': '50px'}),
                    html.P("Please start over from the beginning.", style={'textAlign': 'center', 'marginBottom': '30px'}),
                    html.Div([
                        html.A(
                            "Go to Start Page", 
                            href="/",
                            style={
                                'backgroundColor': '#007bff',
                                'color': 'white',
                                'padding': '15px 30px',
                                'textDecoration': 'none',
                                'borderRadius': '5px',
                                'fontSize': '18px'
                            }
                        )
                    ], style={'textAlign': 'center'})
                ]), warning_content
            return create_ending_layout(full_df_data, events_df_data, user_info), warning_content
        return (startup_page if startup_page else html.Div("Loading..."), warning_content)  # Return the startup page component

def create_prediction_layout() -> html.Div:
    """Create the prediction page layout"""
    return html.Div([
        header_component,
        html.Div([
            html.Div(
                glucose_chart,
                id='prediction-glucose-chart-container'
            ),
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
    if not full_df_data:
        print("DEBUG: No data available for ending page")
        return html.Div("No data available", style={'textAlign': 'center', 'padding': '50px'})
    
    print("DEBUG: Creating ending page with stored data")
    
    # Reconstruct DataFrames from stored data
    full_df = reconstruct_dataframe_from_dict(full_df_data)
    events_df = reconstruct_events_dataframe_from_dict(events_df_data) if events_df_data else pl.DataFrame(
        {
            'time': [],
            'event_type': [],
            'event_subtype': [],
            'insulin_value': []
        }
    )
    
    # Check if we have stored prediction data from the submit button
    if user_info and 'prediction_table_data' in user_info:
        print("DEBUG: Using stored prediction table data from submit button")
        prediction_table_data = user_info['prediction_table_data']
        
        # Check if we have predictions in the stored data
        if len(prediction_table_data) >= 2:
            prediction_row = prediction_table_data[1]  # Second row contains predictions
            valid_predictions = sum(1 for key, value in prediction_row.items() 
                                  if key != 'metric' and value != "-")
            print(f"DEBUG: Found {valid_predictions} valid predictions in stored data")
            
            if valid_predictions == 0:
                print("DEBUG: No valid predictions in stored data")
                return html.Div("No predictions to display", style={'textAlign': 'center', 'padding': '50px'})
        else:
            print("DEBUG: No prediction table data available")
            return html.Div("No predictions to display", style={'textAlign': 'center', 'padding': '50px'})
        
        # Use the same window that was used for predictions if available
        if user_info and 'prediction_window_start' in user_info and 'prediction_window_size' in user_info:
            window_start = user_info['prediction_window_start']
            window_size = user_info['prediction_window_size']
            # Ensure we don't go beyond the available data
            max_start = len(full_df) - window_size
            safe_start = min(window_start, max_start)
            safe_start = max(0, safe_start)
            df = full_df.slice(safe_start, window_size)
            print(f"DEBUG: Using prediction window starting at {safe_start} with size {window_size}")
        else:
            # Fallback to first DEFAULT_POINTS for display
            df = full_df.slice(0, DEFAULT_POINTS)
            print("DEBUG: No prediction window info found, using default first 24 points")
    else:
        print("DEBUG: No stored prediction data found")
        return html.Div("No predictions to display", style={'textAlign': 'center', 'padding': '50px'})
    
    # Calculate metrics directly from the stored prediction table data
    metrics_component_ending = MetricsComponent()
    stored_metrics = None
    
    if len(prediction_table_data) >= 2:  # Need at least actual and predicted rows
        stored_metrics = metrics_component_ending._calculate_metrics_from_table_data(prediction_table_data)
    
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
        
        # Graph section - reuse the same glucose chart component
        html.Div([
            html.Div(
                glucose_chart,
                id='ending-glucose-chart-container'
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
            # Create prediction table directly from stored data
            dash_table.DataTable(
                data=prediction_table_data,
                columns=[{'name': 'Metric', 'id': 'metric'}] + [
                    {'name': f'T{i}', 'id': f't{i}', 'type': 'text'} 
                    for i in range(len(prediction_table_data[0]) - 1) if prediction_table_data
                ],
                style_table={
                    'width': '100%',
                    'height': 'auto',
                    'maxHeight': 'clamp(300px, 40vh, 500px)',
                    'overflowY': 'auto',
                    'overflowX': 'auto',
                    'tableLayout': 'fixed'
                },
                style_cell={
                    'textAlign': 'center',
                    'padding': 'clamp(2px, 1vw, 4px) clamp(1px, 0.5vw, 2px)',
                    'fontSize': 'clamp(8px, 1.5vw, 12px)',
                    'whiteSpace': 'nowrap',
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis',
                    'lineHeight': '1.2',
                    'minWidth': '40px'
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

def render_mobile_warning(user_agent: Optional[str]) -> Optional[html.Div]:
    if not user_agent:
        return None
    ua = user_agent.lower()
    mobile_keywords = ("iphone", "android", "ipad", "mobile", "opera mini", "mobi")
    if any(keyword in ua for keyword in mobile_keywords):
        return html.Div(
            "Note: Sugar Sugar is optimised for desktop at the moment. Mobile support is coming soon!",
            style={
                'backgroundColor': '#fff3cd',
                'border': '1px solid #ffeeba',
                'color': '#856404',
                'padding': '10px 14px',
                'borderRadius': '6px',
                'textAlign': 'center',
                'marginBottom': '12px',
                'fontWeight': '600'
            }
        )
    return None

def reconstruct_events_dataframe_from_dict(events_data: Dict[str, List[Any]]) -> pl.DataFrame:
    """Reconstruct the events DataFrame from stored data.""" 
    # Convert mixed types to strings first, then to float
    insulin_values = []
    for val in events_data['insulin_value']:
        if val is None or val == '':
            insulin_values.append(None)
        else:
            try:
                # Convert to float, handling both string and numeric inputs
                insulin_values.append(float(val))
            except (ValueError, TypeError):
                insulin_values.append(None)
    
    return pl.DataFrame({
        'time': pl.Series(events_data['time']).str.strptime(pl.Datetime, format='%Y-%m-%dT%H:%M:%S'),
        'event_type': pl.Series(events_data['event_type'], dtype=pl.String),
        'event_subtype': pl.Series(events_data['event_subtype'], dtype=pl.String),
        # Use pre-processed float values
        'insulin_value': pl.Series(insulin_values, dtype=pl.Float64)
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
def handle_start_button(n_clicks: Optional[int], email: Optional[str], age: Optional[int], 
                       gender: Optional[str], diabetic: Optional[bool], diabetic_type: Optional[str], 
                       diabetes_duration: Optional[int], medical_conditions: Optional[bool], 
                       medical_conditions_input: Optional[str], location: Optional[str]) -> Tuple[str, Dict[str, Any]]:
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
     Output('user-info-store', 'data', allow_duplicate=True),
     Output('glucose-chart-mode', 'data', allow_duplicate=True),
     Output('current-window-df', 'data', allow_duplicate=True)],
    [Input('submit-button', 'n_clicks')],
    [State('user-info-store', 'data'),
     State('full-df', 'data'),
     State('current-window-df', 'data'),
     State('time-slider', 'value')],
    prevent_initial_call=True
)
def handle_submit_button(n_clicks: Optional[int], user_info: Optional[Dict[str, Any]], 
                        full_df_data: Optional[Dict], current_df_data: Optional[Dict], 
                        slider_value: Optional[int]) -> Tuple[str, Optional[Dict[str, Any]], Dict[str, bool], Dict[str, List[Any]]]:
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
        
        # Debug: Check what predictions we have
        prediction_count = current_df.filter(pl.col("prediction") != 0.0).height
        print(f"DEBUG: Submit button - Found {prediction_count} predictions in current_df")
        print(f"DEBUG: Submit button - Sample predictions: {current_df.filter(pl.col('prediction') != 0.0).select(['time', 'prediction']).head(5).to_dicts()}")
        
        # Save statistics before redirecting
        submit_component.save_statistics(current_full_df, user_info)
        
        # Update chart mode to show ground truth and return the full window with ground truth
        chart_mode = {'hide_last_hour': False}
        
        # Convert the current DataFrame back to dict for the store
        def convert_df_to_dict(df_in: pl.DataFrame) -> Dict[str, List[Any]]:
            return {
                'time': df_in.get_column('time').dt.strftime('%Y-%m-%dT%H:%M:%S').to_list(),
                'gl': df_in.get_column('gl').to_list(),
                'prediction': df_in.get_column('prediction').to_list(),
                'age': df_in.get_column('age').to_list(),
                'user_id': df_in.get_column('user_id').to_list()
            }
        
        return '/ending', user_info, chart_mode, convert_df_to_dict(current_df)
    return no_update, no_update, no_update, no_update

@app.callback(
    [Output('url', 'pathname', allow_duplicate=True),
     Output('user-info-store', 'data', allow_duplicate=True),
     Output('glucose-chart-mode', 'data', allow_duplicate=True)],
    [Input('exit-button', 'n_clicks')],
    prevent_initial_call=True
)
def handle_exit_button(n_clicks: Optional[int]) -> Tuple[str, None, Dict[str, bool]]:
    """Handle exit button - navigate to start and clear user info. Data reset handled elsewhere."""
    if n_clicks:
        print("DEBUG: Exit button clicked")
        # Reset chart mode to hide last hour when going back to prediction
        chart_mode = {'hide_last_hour': True}
        return '/', None, chart_mode
    return no_update, no_update, no_update

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

## Removed URL-based data writer callback to enforce single-writer for data stores

# Data initialization callback (URL-based only)
@app.callback(
    [Output('full-df', 'data', allow_duplicate=True),
     Output('current-window-df', 'data', allow_duplicate=True),
     Output('events-df', 'data', allow_duplicate=True),
     Output('is-example-data', 'data', allow_duplicate=True),
     Output('data-source-name', 'data', allow_duplicate=True),
     Output('randomization-initialized', 'data', allow_duplicate=True)],
    [Input('url', 'pathname')],
    [State('full-df', 'data')],
    prevent_initial_call=True
)
def initialize_data_on_url_change(pathname: Optional[str], full_df_data: Optional[Dict]) -> Tuple[Dict[str, List[Any]], Dict[str, List[Any]], Dict[str, List[Any]], bool, str, bool]:
    """Initialize data when URL changes or on first load"""
    # Handle URL-driven initialization without requiring existing data
    if pathname == '/ending':
        return no_update, no_update, no_update, no_update, no_update, no_update
    # If prediction page and data already present, preserve
    if pathname == '/prediction' and full_df_data is not None:
        return no_update, no_update, no_update, no_update, no_update, no_update
    
    # Initialize fresh example data (startup or first load)
    full_df, events_df = load_glucose_data()
    df, random_start = get_random_data_window(full_df, DEFAULT_POINTS)
    full_df = full_df.with_columns(pl.lit(0.0).alias('prediction'))
    df = df.with_columns(pl.lit(0.0).alias('prediction'))
    return (
        dataframe_to_store_dict(full_df),
        dataframe_to_store_dict(df),
        events_dataframe_to_store_dict(events_df),
        True,
        'example.csv',
        False  # Keep randomization flag false so slider can be randomized
    )

# Separate callback for file upload handling
@app.callback(
    [Output('last-click-time', 'data'),
     Output('full-df', 'data', allow_duplicate=True),
     Output('current-window-df', 'data', allow_duplicate=True),
     Output('events-df', 'data', allow_duplicate=True),
     Output('is-example-data', 'data', allow_duplicate=True),
     Output('data-source-name', 'data', allow_duplicate=True),
     Output('randomization-initialized', 'data', allow_duplicate=True)],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename'),
     State('points-control', 'value')],
    prevent_initial_call=True
)
def handle_file_upload(upload_contents: Optional[str], filename: Optional[str], 
                      points_value: Optional[int]) -> Tuple[int, Dict[str, List[Any]], Dict[str, List[Any]], Dict[str, List[Any]], bool, str, bool]:
    """Handle file upload and data loading"""
    if not upload_contents:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update
    
    with start_action(action_type=u"handle_file_upload", filename=filename):
        current_time = int(time.time() * 1000)
        
        # Parse upload contents
        if ',' not in upload_contents:
            print(f"ERROR: Invalid upload format for file {filename}")
            return current_time, no_update, no_update, no_update, no_update, no_update, no_update
        
        content_type, content_string = upload_contents.split(',', 1)
        decoded = base64.b64decode(content_string)
        
        # Ensure user data directory exists under data/input/users
        users_data_dir = project_root / 'data' / 'input' / 'users'
        users_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = filename.replace(' ', '_').replace('/', '_') if filename else 'uploaded_data'
        if not safe_filename.endswith('.csv'):
            safe_filename += '.csv'
        unique_filename = f"{timestamp}_{safe_filename}"
        
        # Save file to the users data folder
        save_path = users_data_dir / unique_filename
        with open(save_path, 'wb') as f:
            f.write(decoded)
        
        print(f"DEBUG: saved uploaded file to {save_path}")
        
        # Load glucose data - let load_glucose_data handle its own error cases
        new_full_df, new_events_df = load_glucose_data(save_path)
        
        # Start at a random position for uploaded files too
        points = max(MIN_POINTS, min(MAX_POINTS, points_value or DEFAULT_POINTS))
        new_df, random_start = get_random_data_window(new_full_df, points)
        
        return (current_time, 
               convert_df_to_dict(new_full_df),
               convert_df_to_dict(new_df),
               convert_events_df_to_dict(new_events_df),
               False,  # is_example_data = False for uploaded files
               filename,  # store the original filename
               False)  # reset randomization flag for new data


# Separate callback for example data button
@app.callback(
    [Output('last-click-time', 'data', allow_duplicate=True),
     Output('full-df', 'data', allow_duplicate=True),
     Output('current-window-df', 'data', allow_duplicate=True),
     Output('events-df', 'data', allow_duplicate=True),
     Output('is-example-data', 'data', allow_duplicate=True),
     Output('data-source-name', 'data', allow_duplicate=True),
     Output('randomization-initialized', 'data', allow_duplicate=True),
     Output('time-slider', 'value', allow_duplicate=True)],  # Add slider value update
    [Input('use-example-data-button', 'n_clicks')],
    [State('points-control', 'value')],
    prevent_initial_call=True
)
def handle_example_data_button(example_button_clicks: Optional[int], points_value: Optional[int]) -> Tuple[int, Dict[str, List[Any]], Dict[str, List[Any]], Dict[str, List[Any]], bool, str, bool, int]:
    """Handle use example data button click"""
    if not example_button_clicks:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
    
    with start_action(action_type=u"handle_example_data_button"):
        current_time = int(time.time() * 1000)
        
        # Load fresh example data
        new_full_df, new_events_df = load_glucose_data()
        
        # Start at a random position for example data too
        points = max(MIN_POINTS, min(MAX_POINTS, points_value or DEFAULT_POINTS))
        new_df, random_start = get_random_data_window(new_full_df, points)
        
        # Reset predictions
        new_full_df = new_full_df.with_columns(pl.lit(0.0).alias("prediction"))
        new_df = new_df.with_columns(pl.lit(0.0).alias("prediction"))
        
        print(f"DEBUG: Generated new random start position for example data: {random_start}")
        
        return (current_time, 
               convert_df_to_dict(new_full_df),
               convert_df_to_dict(new_df),
               convert_events_df_to_dict(new_events_df),
               True,  # is_example_data = True for example data
               "example.csv",  # data_source_name for example data
               False,  # reset randomization flag for new data
               random_start)  # Set slider to the random start position


# Separate callback for points control
@app.callback(
    [Output('last-click-time', 'data', allow_duplicate=True),
     Output('current-window-df', 'data', allow_duplicate=True)],
    [Input('points-control', 'value')],
    [State('time-slider', 'value'),
     State('full-df', 'data')],
    prevent_initial_call=True
)
def handle_points_control(points_value: Optional[int], current_position: Optional[int], 
                         full_df_data: Optional[Dict]) -> Tuple[int, Dict[str, List[Any]]]:
    """Handle points control slider changes"""
    if not points_value or not full_df_data:
        return no_update, no_update
    
    with start_action(action_type=u"handle_points_control", points_value=points_value):
        current_time = int(time.time() * 1000)
        
        full_df = reconstruct_dataframe_from_dict(full_df_data)
        points = max(MIN_POINTS, min(MAX_POINTS, points_value))
        new_max = len(full_df) - points
        new_start = min(current_position or 0, new_max)
        new_start = max(0, new_start)
        new_df = full_df.slice(new_start, points)
        
        return current_time, convert_df_to_dict(new_df)


# Separate callback for time slider
@app.callback(
    [Output('last-click-time', 'data', allow_duplicate=True),
     Output('current-window-df', 'data', allow_duplicate=True)],
    [Input('time-slider', 'value')],
    [State('points-control', 'value'),
     State('full-df', 'data')],
    prevent_initial_call=True
)
def handle_time_slider(slider_value: Optional[int], current_points: Optional[int], 
                      full_df_data: Optional[Dict]) -> Tuple[int, Dict[str, List[Any]]]:
    """Handle time slider changes"""
    if slider_value is None or not full_df_data:
        return no_update, no_update
    
    with start_action(action_type=u"handle_time_slider", slider_value=slider_value):
        current_time = int(time.time() * 1000)
        
        full_df = reconstruct_dataframe_from_dict(full_df_data)
        
        # Ensure we don't go beyond the available data
        points = max(MIN_POINTS, min(MAX_POINTS, current_points or DEFAULT_POINTS))
        max_start = len(full_df) - points
        safe_slider_value = min(slider_value, max_start)
        safe_slider_value = max(0, safe_slider_value)
        
        new_df = full_df.slice(safe_slider_value, points)
        
        return current_time, convert_df_to_dict(new_df)

# Separate callback for glucose graph interactions (only active on prediction page)
@app.callback(
    [Output('last-click-time', 'data', allow_duplicate=True),
     Output('full-df', 'data', allow_duplicate=True),
     Output('current-window-df', 'data', allow_duplicate=True)],
    [Input('glucose-graph-graph', 'clickData'),
     Input('glucose-graph-graph', 'relayoutData')],
    [State('last-click-time', 'data'),
     State('full-df', 'data'),
     State('current-window-df', 'data')],
    prevent_initial_call=True
)
def handle_graph_interactions(click_data: Optional[Dict], relayout_data: Optional[Dict],
                            last_click_time: int, full_df_data: Optional[Dict], 
                            current_df_data: Optional[Dict]) -> Tuple[int, Dict[str, List[Any]], Dict[str, List[Any]]]:
    """Handle glucose graph click and draw interactions"""
    if not full_df_data or not current_df_data:
        return no_update, no_update, no_update
    
    current_time = int(time.time() * 1000)
    full_df = reconstruct_dataframe_from_dict(full_df_data)
    df = reconstruct_dataframe_from_dict(current_df_data)
    predictions_values = df.get_column("prediction").to_list()
    visible_points = len(df) - PREDICTION_HOUR_OFFSET
    
    
    def snap_index(x_value: Optional[float]) -> Optional[int]:
        """Snap a drawn x-coordinate to the nearest data index while respecting prediction bounds."""
        if x_value is None:
            return None
        snapped_idx = int(round(float(x_value)))
        snapped_idx = max(0, min(snapped_idx, len(df) - 1))
        if snapped_idx < visible_points and predictions_values[snapped_idx] == 0.0:
            return None
        return snapped_idx
    
    if click_data:
        if current_time - last_click_time <= DOUBLE_CLICK_THRESHOLD:
            full_df = full_df.with_columns(pl.lit(0.0).alias("prediction"))
            df = df.with_columns(pl.lit(0.0).alias("prediction"))
            
            return (current_time,
                   convert_df_to_dict(full_df),
                   convert_df_to_dict(df))
        
        point_data = click_data['points'][0]
        click_x = point_data['x']
        click_y = point_data['y']
        snapped_idx = snap_index(float(click_x))
        if snapped_idx is None:
            return no_update, no_update, no_update
        nearest_time = df.get_column("time")[snapped_idx]
        
        # Check if this is the first prediction point at the boundary - snap to ground truth
        prediction_y = click_y
        if snapped_idx == visible_points:  # First point in hidden area
            # Check if this is the start of a new prediction sequence
            existing_predictions = df.filter(pl.col("prediction") != 0.0).height
            if existing_predictions == 0:  # No existing predictions, snap to ground truth
                ground_truth_y = df.get_column("gl")[snapped_idx]
                prediction_y = ground_truth_y
        
        full_df = full_df.with_columns(
            pl.when(pl.col("time") == nearest_time)
            .then(prediction_y)
            .otherwise(pl.col("prediction"))
            .alias("prediction")
        )
        df = df.with_columns(
            pl.when(pl.col("time") == nearest_time)
            .then(prediction_y)
            .otherwise(pl.col("prediction"))
            .alias("prediction")
        )
        
        return (current_time,
               convert_df_to_dict(full_df),
               convert_df_to_dict(df))
    
    elif relayout_data and 'shapes' in relayout_data:
        shapes = relayout_data['shapes']
        if shapes and len(shapes) > 0:
            latest_shape = shapes[-1]
            
            start_x = latest_shape.get('x0')
            end_x = latest_shape.get('x1')
            start_y = latest_shape.get('y0')
            end_y = latest_shape.get('y1')
            
            if all(v is not None for v in [start_x, end_x, start_y, end_y]):
                start_idx = snap_index(float(start_x))
                end_idx = snap_index(float(end_x))
                if start_idx is None or end_idx is None:
                    return (
                        last_click_time,
                        convert_df_to_dict(full_df),
                        convert_df_to_dict(df)
                    )
                
                start_time = df.get_column("time")[start_idx]
                
                # Check if this is the first prediction starting at the boundary - snap to ground truth
                actual_start_y = start_y
                if start_idx == visible_points:  # Starting at first point in hidden area
                    # Check if this is the start of a new prediction sequence
                    existing_predictions = df.filter(pl.col("prediction") != 0.0).height
                    if existing_predictions == 0:  # No existing predictions, snap to ground truth
                        ground_truth_y = df.get_column("gl")[start_idx]
                        actual_start_y = ground_truth_y
                
                # Calculate the intersection with the first vertical guideline after start
                actual_end_x, actual_end_y = calculate_first_guideline_intersection(
                    float(start_idx), actual_start_y, float(end_idx), end_y, df
                )
                snapped_end_idx = snap_index(actual_end_x)
                if snapped_end_idx is None:
                    return (
                        last_click_time,
                        convert_df_to_dict(full_df),
                        convert_df_to_dict(df)
                    )
                end_time = df.get_column("time")[snapped_end_idx]
                
                # Get intermediate prediction points every 5 minutes
                intermediate_points = create_intermediate_predictions(start_time, end_time, float(actual_start_y), float(actual_end_y), df)
                
                # Collect all times that need prediction values
                all_prediction_times = [start_time, end_time]
                all_prediction_values = [float(actual_start_y), float(actual_end_y)]
                
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
                        .then(float(actual_start_y))
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
                        .then(float(actual_start_y))
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
                       convert_df_to_dict(df))
    
    return no_update, no_update, no_update

@app.callback(
    Output('data-source-display', 'children'),
    [Input('url', 'pathname'), Input('data-source-name', 'data')],
    prevent_initial_call=True
)
def update_data_source_display(pathname: str, source_name: Optional[str]) -> str:
    """Update the visible data source label only when on the prediction page."""
    if pathname != '/prediction':
        raise PreventUpdate
    return source_name if source_name else "example.csv"

# Add callback for random slider initialization when prediction page components are ready
@app.callback(
    [Output('time-slider', 'value', allow_duplicate=True),
     Output('randomization-initialized', 'data', allow_duplicate=True)],
    [Input('time-slider', 'max')],  # Triggers when slider is created and max is set
    [State('url', 'pathname'),
     State('full-df', 'data'),
     State('points-control', 'value'),
     State('randomization-initialized', 'data'),
     State('initial-slider-value', 'data')],
    prevent_initial_call=True
)
def randomize_slider_on_prediction_page(slider_max: int, pathname: str, full_df_data: Optional[Dict], 
                                       points_value: int, randomization_initialized: bool, 
                                       initial_slider_value: Optional[int]) -> Tuple[int, bool]:
    """Set slider to a random valid window start when slider mounts on prediction page. Returns slider value and updated randomization flag."""
    if pathname == '/prediction' and full_df_data and slider_max is not None and not randomization_initialized:
        # Use the stored initial slider value if available
        if initial_slider_value is not None:
            return initial_slider_value, True
        # Otherwise generate a new random start
        full_df = reconstruct_dataframe_from_dict(full_df_data)
        points = max(MIN_POINTS, min(MAX_POINTS, points_value or DEFAULT_POINTS))
        _, random_start = get_random_data_window(full_df, points)
        return random_start, True  # Set randomization flag to True after randomizing
    return no_update, no_update


# Separate UI callback for upload success message
@app.callback(
    Output('example-data-warning', 'children'),
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename'),
     State('is-example-data', 'data')],
    prevent_initial_call=True
)
def update_upload_success_message(upload_contents: Optional[str], filename: Optional[str], 
                                 is_example_data: Optional[bool]) -> Optional[html.Div]:
    """Show success message when file is uploaded"""
    if not upload_contents:
        return no_update
    
    if not is_example_data:  # File was successfully uploaded
        return html.Div([
            html.I(className="fas fa-check-circle", style={'marginRight': '8px'}),
            f"Successfully loaded data from {filename}. This data will be used for predictions."
        ], style={
            'color': '#2f855a',
            'backgroundColor': '#c6f6d5',
            'padding': '10px',
            'borderRadius': '5px',
            'textAlign': 'center'
        })
    return None


# Separate UI callback for example data button message and upload reset
@app.callback(
    [Output('example-data-warning', 'children', allow_duplicate=True),
     Output('time-slider', 'max', allow_duplicate=True),
     Output('upload-data', 'contents', allow_duplicate=True),  # Reset upload contents
     Output('upload-data', 'filename', allow_duplicate=True)],  # Reset filename
    [Input('use-example-data-button', 'n_clicks')],
    [State('points-control', 'value'),
     State('full-df', 'data')],
    prevent_initial_call=True
)
def reset_upload_on_example_data(example_button_clicks: Optional[int], points_value: Optional[int], 
                                full_df_data: Optional[Dict]) -> Tuple[Optional[html.Div], int, None, None]:
    """Reset upload component and show message when example data button is clicked"""
    if not example_button_clicks or not full_df_data:
        return no_update, no_update, no_update, no_update
    
    with start_action(action_type=u"reset_upload_on_example_data"):
        full_df = reconstruct_dataframe_from_dict(full_df_data)
        points = max(MIN_POINTS, min(MAX_POINTS, points_value))
        new_max = len(full_df) - points
        
        print("DEBUG: Resetting upload component to allow re-upload of same file")
        
        # Show message that we're now using example data
        example_msg = html.Div([
            html.I(className="fas fa-info-circle", style={'marginRight': '8px'}),
            "Now using example data. Upload a CSV file for personalized analysis."
        ], style={
            'color': '#0c5460',
            'backgroundColor': '#d1ecf1',
            'padding': '10px',
            'borderRadius': '5px',
            'textAlign': 'center'
        })
        
        # Reset upload component by clearing contents and filename
        # This allows the same file to be uploaded again after switching to example data
        return example_msg, new_max, None, None


# Separate UI callback for points control
@app.callback(
    [Output('example-data-warning', 'children', allow_duplicate=True),
     Output('time-slider', 'max', allow_duplicate=True),
     Output('time-slider', 'value', allow_duplicate=True)],
    [Input('points-control', 'value')],
    [State('time-slider', 'value'),
     State('full-df', 'data'),
     State('is-example-data', 'data')],
    prevent_initial_call=True
)
def update_points_control_ui(points_value: Optional[int], current_position: Optional[int], 
                            full_df_data: Optional[Dict], is_example_data: Optional[bool]) -> Tuple[Optional[html.Div], int, int]:
    """Update UI when points control changes"""
    if not points_value or not full_df_data:
        return no_update, no_update, no_update
    
    full_df = reconstruct_dataframe_from_dict(full_df_data)
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

def convert_df_to_dict(df: pl.DataFrame) -> Dict[str, List[Any]]:
    """Convert a Polars DataFrame to a session-store dictionary."""
    return {
        'time': df.get_column('time').dt.strftime('%Y-%m-%dT%H:%M:%S').to_list(),
        'gl': df.get_column('gl').to_list(),
        'prediction': df.get_column('prediction').to_list(),
        'age': df.get_column('age').to_list(),
        'user_id': df.get_column('user_id').to_list()
    }

def convert_events_df_to_dict(df: pl.DataFrame) -> Dict[str, List[Any]]:
    """Convert an events Polars DataFrame to a session-store dictionary."""
    return {
        'time': df.get_column('time').dt.strftime('%Y-%m-%dT%H:%M:%S').to_list(),
        'event_type': df.get_column('event_type').to_list(),
        'event_subtype': df.get_column('event_subtype').to_list(),
        'insulin_value': df.get_column('insulin_value').to_list()
    }

def reconstruct_dataframe_from_dict(df_data: Dict[str, List[Any]]) -> pl.DataFrame:
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
    for time_point in available_times:
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
def main(
    debug: Optional[bool] = typer.Option(None, "--debug", help="Enable debug mode to show test button"),
    host: Optional[str] = typer.Option(None, "--host", help="Host to run the server on"),
    port: Optional[int] = typer.Option(None, "--port", help="Port to run the server on")
) -> None:
    """Starts the Dash server."""
    # Import config here to update the global DEBUG_MODE variable
    import sugar_sugar.config as config
    
    # Get configuration from environment variables with fallbacks
    dash_host = host or os.getenv('DASH_HOST', '127.0.0.1')
    dash_port = port or int(os.getenv('DASH_PORT', '8050'))
    dash_debug = debug if debug is not None else os.getenv('DASH_DEBUG', 'True').lower() == 'true'
    
    # Set the global debug mode based on command line argument or environment
    config.DEBUG_MODE = debug if debug is not None else os.getenv('DEBUG_MODE', 'True').lower() == 'true'
    
    # Create components after setting debug mode
    global startup_page
    startup_page = StartupPage()
    
    prediction_table.register_callbacks(app)  # Register the prediction table callbacks
    metrics_component.register_callbacks(app, prediction_table)  # Register the metrics component callbacks
    glucose_chart.register_callbacks(app)  # Register the glucose chart callbacks
    submit_component.register_callbacks(app)  # Register the submit component callbacks
    startup_page.register_callbacks(app)  # Register the startup page callbacks
    ending_page.register_callbacks(app)  # Register the ending page callbacks
    
    # Update the app layout with the new startup page
    app.layout.children[-1].children = [startup_page]
    
    with start_action(
        action_type=u"start_dash_server",
        host=dash_host,
        port=dash_port,
        debug=dash_debug
    ):
        app.run(host=dash_host, port=dash_port, debug=dash_debug)

def cli_main() -> None:
    """CLI entry point"""
    cli()

if __name__ == '__main__':
    cli()
