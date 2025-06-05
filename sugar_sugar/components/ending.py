import dash
from dash import html, dcc, Output, Input
import dash_bootstrap_components as dbc
import polars as pl
from .glucose import GlucoseChart
from .predictions import PredictionTableComponent
from .metrics import MetricsComponent

class EndingPage:
    def __init__(self, df: pl.DataFrame):
        print("DEBUG: Initializing EndingPage")
        self.df = df
        self.glucose_chart = GlucoseChart(id='ending-glucose-graph')
        self.prediction_table = PredictionTableComponent(df)
        self.metrics_component = MetricsComponent(df)
        
    def update_dataframe(self, df: pl.DataFrame) -> None:
        """Update the DataFrame used by the component."""
        print("DEBUG: Updating ending page dataframe...")
        print(f"DEBUG: DataFrame shape: {df.shape}")
        self.df = df
        self.prediction_table.update_dataframe(df)
        self.metrics_component.update_dataframe(df)
        print("DEBUG: DataFrame update complete")
        
    def register_callbacks(self, app: dash.Dash) -> None:
        """Register callbacks for the ending page."""
        print("DEBUG: Registering ending page callbacks")
        @app.callback(
            Output('ending-glucose-graph-graph', 'figure'),
            [Input('url', 'pathname')],
            prevent_initial_call=True
        )
        def update_ending_graph(pathname):
            print(f"DEBUG: update_ending_graph called with pathname={pathname}")
            if pathname == '/ending':
                print("DEBUG: Updating ending graph")
                return self.glucose_chart.update(self.df, None)
            return dash.no_update
            
    def __call__(self) -> html.Div:
        """Render the ending page."""
        print("DEBUG: Rendering ending page...")
        # Generate table data and metrics
        table_data = self.prediction_table.generate_table_data()
        print(f"DEBUG: Generated table data with {len(table_data)} rows")
        metrics = self.metrics_component.calculate_error_metrics(self.df, table_data[1])
        print("DEBUG: Calculated metrics")
        
        return html.Div([
            html.H1("Prediction Summary", style={'textAlign': 'center', 'marginBottom': '20px'}),
            
            # Graph section
            html.Div([
                self.glucose_chart
            ], style={'marginBottom': '20px'}),
            
            # Prediction table section
            html.Div([
                html.H3("Prediction Results", style={'textAlign': 'center'}),
                self.prediction_table()
            ], style={'marginBottom': '20px'}),
            
            # Metrics section
            html.Div([
                html.H3("Accuracy Metrics", style={'textAlign': 'center'}),
                html.Div(metrics, id='ending-metrics-container')
            ], style={'marginBottom': '20px'}),
            
            # Buttons section
            html.Div([
                dbc.Button("Try Again", id='try-again-button', color="primary", className="me-2"),
                dbc.Button("Exit", id='exit-button', color="secondary")
            ], style={'textAlign': 'center', 'marginTop': '20px'})
        ], style={
            'maxWidth': '1200px',
            'margin': '0 auto',
            'padding': '20px'
        }) 