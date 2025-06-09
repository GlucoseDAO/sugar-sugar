import dash
from dash import html, dcc, Output, Input, State, no_update
import dash_bootstrap_components as dbc
import polars as pl
from .glucose import GlucoseChart
from .predictions import PredictionTableComponent
from .metrics import MetricsComponent
from ..config import DEFAULT_POINTS

class EndingPage:
    def __init__(self):
        print("DEBUG: Initializing EndingPage")
        # Initialize components without data
        self.glucose_chart = GlucoseChart(id='ending-glucose-graph')
        self.cached_content = None  # Cache the ending page content
        
    def register_callbacks(self, app: dash.Dash) -> None:
        """Register callbacks for the ending page."""
        print("DEBUG: Registering ending page callbacks")
        
        @app.callback(
            Output('page-content', 'children', allow_duplicate=True),
            [Input('url', 'pathname'),
             Input('full-df', 'data'),
             Input('events-df', 'data')],
            prevent_initial_call=True
        )
        def update_ending_page(pathname, full_df_data, events_df_data):
            """Updates the entire ending page content when navigating to it."""
            print(f"DEBUG: update_ending_page called with pathname: {pathname}")
            print(f"DEBUG: Data available - full_df: {bool(full_df_data)}, events_df: {bool(events_df_data)}")
            
            if pathname != '/ending':
                print("DEBUG: Not ending page, returning no_update")
                return no_update
            
            if not full_df_data or not events_df_data:
                print("DEBUG: No data available for ending page")
                if self.cached_content is not None:
                    print("DEBUG: Returning cached ending page content")
                    return self.cached_content
                return no_update
                
            print("DEBUG: Updating ending page with stored data")
            
            # Reconstruct DataFrames from stored data
            full_df = self._reconstruct_dataframe(full_df_data)
            events_df = self._reconstruct_events_dataframe(events_df_data)
            
            # Use only the first DEFAULT_POINTS for display
            df = full_df.slice(0, DEFAULT_POINTS)
            
            # Check if we have any predictions
            prediction_count = df.filter(pl.col("prediction") != 0.0).height
            print(f"DEBUG: Found {prediction_count} predictions in ending page data")
            
            if prediction_count == 0:
                print("DEBUG: No predictions found, not showing ending page")
                return no_update
            
            print("+++++++++++++++++++++++++++++++" )
            print("Ending page DataFrame:")
            print(str(df))
            print("+++++++++++++++++++++++++++++++" )
            
            # Create new components with the updated data
            prediction_table = PredictionTableComponent(df)
            metrics_component = MetricsComponent(df)
            
            # Generate table data and metrics
            table_data = prediction_table.generate_table_data()
            prediction_row = table_data[1]  # Index 1 contains predictions
            metrics_content = metrics_component.calculate_error_metrics(df, prediction_row)
            
            # Update glucose chart and get the figure
            figure = self.glucose_chart.update(df, events_df)
            
            # Create the page content
            page_content = html.Div([
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
                
                # Metrics section
                html.Div([
                    html.H3("Accuracy Metrics", style={'textAlign': 'center'}),
                    html.Div(children=metrics_content)
                ], style={'marginBottom': '20px'}),
                
                # Buttons section
                html.Div([
                    dbc.Button("Exit", id='exit-button', color="secondary")
                ], style={'textAlign': 'center', 'marginTop': '20px'})
            ], style={
                'maxWidth': '1200px',
                'margin': '0 auto',
                'padding': '20px'
            })
            
            print("DEBUG: Ending page content created successfully")
            # Cache the content for stability
            self.cached_content = page_content
            return page_content
            
    def _reconstruct_dataframe(self, df_data):
        """Reconstruct the DataFrame from stored data."""
        return pl.DataFrame({
            'time': pl.Series(df_data['time']).str.strptime(pl.Datetime, format='%Y-%m-%dT%H:%M:%S'),
            'gl': pl.Series(df_data['gl'], dtype=pl.Float64),
            'prediction': pl.Series(df_data['prediction'], dtype=pl.Float64),
            'age': pl.Series([int(float(x)) for x in df_data['age']], dtype=pl.Int64),
            'user_id': pl.Series([int(float(x)) for x in df_data['user_id']], dtype=pl.Int64)
        })
    
    def _reconstruct_events_dataframe(self, events_data):
        """Reconstruct the events DataFrame from stored data.""" 
        return pl.DataFrame({
            'time': pl.Series(events_data['time']).str.strptime(pl.Datetime, format='%Y-%m-%dT%H:%M:%S'),
            'event_type': pl.Series(events_data['event_type'], dtype=pl.String),
            'event_subtype': pl.Series(events_data['event_subtype'], dtype=pl.String),
            'insulin_value': pl.Series(events_data['insulin_value'], dtype=pl.Float64)
        })
        
    def __call__(self) -> html.Div:
        """Render the ending page container - not used anymore."""
        print("DEBUG: EndingPage.__call__ should not be called anymore")
        return html.Div("This should not be visible") 