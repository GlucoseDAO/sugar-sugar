import dash
from dash import html, dcc, Output, Input, State, no_update
import dash_bootstrap_components as dbc
import polars as pl
from sugar_sugar.components.glucose import GlucoseChart
from sugar_sugar.components.predictions import PredictionTableComponent
from sugar_sugar.components.metrics import MetricsComponent
from sugar_sugar.config import DEFAULT_POINTS

class EndingPage:
    def __init__(self):
        print("DEBUG: Initializing EndingPage")
        # Initialize components without data
        self.glucose_chart = GlucoseChart(id='ending-glucose-graph')
        self.cached_content = None  # Cache the ending page content
        
    def register_callbacks(self, app: dash.Dash) -> None:
        """Register callbacks for the ending page."""
        print("DEBUG: Registering ending page callbacks")
        
        # Note: Page content updates are now handled in the main app.py
        # This component only handles internal ending page logic if needed
        pass
        
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