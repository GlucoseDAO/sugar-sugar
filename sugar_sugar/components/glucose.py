import plotly.graph_objs as go
import polars as pl
from typing import Any, Optional
from datetime import datetime
from dash import dcc, Output, Input
from dash import Dash, html
from sugar_sugar.config import PREDICTION_HOUR_OFFSET


class GlucoseChart(html.Div):
    RANGE_COLORS = {
        "dangerous_low": {"fill": "rgba(255, 200, 200, 0.5)", "line": "rgba(200, 0, 0, 0.5)"},
        "normal": {"fill": "rgba(200, 240, 200, 0.5)", "line": "rgba(0, 100, 0, 0.5)"},
        "high": {"fill": "rgba(255, 255, 200, 0.5)", "line": "rgba(200, 200, 0, 0.5)"},
        "dangerous_high": {"fill": "rgba(255, 200, 200, 0.5)", "line": "rgba(200, 0, 0, 0.5)"}
    }
    
    EVENT_STYLES = {
        'Insulin': {'symbol': 'triangle-down', 'color': 'purple', 'size': 20},
        'Exercise': {'symbol': 'star', 'color': 'orange', 'size': 20},
        'Carbohydrates': {'symbol': 'square', 'color': 'green', 'size': 20}
    }

    def __init__(self, id: str = 'glucose-chart', hide_last_hour: bool = False) -> None:
        super().__init__([
            dcc.Store(id=f"{id}-df-store", data=None),
            dcc.Store(id=f"{id}-events-store", data=None),
            dcc.Store(id=f"{id}-source-store", data=None),
            dcc.Graph(
                id=f"{id}-graph",
                figure=self._create_empty_figure(),
                config={
                    'displayModeBar': True,
                    'scrollZoom': False,
                    'doubleClick': 'reset',
                    'showAxisDragHandles': False,
                    'showAxisRangeEntryBoxes': False,
                    'displaylogo': False,
                    'modeBarButtonsToAdd': ['drawopenpath', 'eraseshape'],
                    'editable': False,
                    'edits': {
                        'shapePosition': False,
                        'annotationPosition': False
                    }
                },
                style={'height': '100%'}
            )
        ])
        self.id = id
        self.hide_last_hour = hide_last_hour

    def _create_empty_figure(self) -> go.Figure:
        """Create an empty figure with basic layout"""
        fig = go.Figure()
        fig.update_layout(
            title='Glucose Levels',
            autosize=True,
            xaxis=dict(title='Time'),
            yaxis=dict(title='Glucose Level (mg/dL)'),
            margin=dict(l=50, r=50, t=50, b=50),
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        return fig

    def register_callbacks(self, app: Dash) -> None:
        """Register all glucose chart related callbacks"""
        
        @app.callback(
            [Output(f'{self.id}-df-store', 'data'),
             Output(f'{self.id}-events-store', 'data'),
             Output(f'{self.id}-source-store', 'data')],
            [Input('current-window-df', 'data'),
             Input('events-df', 'data'),
             Input('data-source-name', 'data')]
        )
        def store_chart_data(
            df_data: Optional[dict[str, Any]],
            events_data: Optional[dict[str, Any]],
            source_name: Optional[str]
        ) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]], Optional[str]]:
            """Store the current DataFrame and events data when they change"""
            print(f"DEBUG: GlucoseChart store_chart_data source={source_name}")
            # Just pass through the session storage data
            return df_data, events_data, source_name

        @app.callback(
            Output(f'{self.id}-graph', 'figure'),
            [Input(f'{self.id}-df-store', 'data'),
             Input(f'{self.id}-events-store', 'data'),
             Input(f'{self.id}-source-store', 'data'),
             Input('glucose-chart-mode', 'data')]
        )
        def update_chart_figure(
            df_data: Optional[dict[str, Any]],
            events_data: Optional[dict[str, Any]],
            source_name: Optional[str],
            mode_data: Optional[dict[str, Any]]
        ) -> go.Figure:
            """Update the chart figure when data changes"""
            if not df_data:
                return self._create_empty_figure()
            
            # Reconstruct DataFrames from stored data
            df = self._reconstruct_dataframe_from_dict(df_data)
            events_df = self._reconstruct_events_dataframe_from_dict(events_data) if events_data else pl.DataFrame()
            hide_mode = mode_data or {'hide_last_hour': self.hide_last_hour}
            hide_last_hour_flag = hide_mode.get('hide_last_hour', self.hide_last_hour)
            self.hide_last_hour = hide_last_hour_flag
            
            print(f"DEBUG: GlucoseChart updating figure - {len(df)} points, glucose range: {df.get_column('gl').min()}-{df.get_column('gl').max()} | source={source_name}")
            
            # Create the figure with source information
            return self._build_figure(df, events_df, source_name)

    def _reconstruct_dataframe_from_dict(self, df_data: dict[str, list[Any]]) -> pl.DataFrame:
        """Reconstruct a Polars DataFrame from stored dictionary data"""
        return pl.DataFrame({
            'time': pl.Series(df_data['time']).str.strptime(pl.Datetime, format='%Y-%m-%dT%H:%M:%S'),
            'gl': pl.Series(df_data['gl'], dtype=pl.Float64),
            'prediction': pl.Series(df_data['prediction'], dtype=pl.Float64),
            'age': pl.Series([int(float(x)) for x in df_data['age']], dtype=pl.Int64),
            'user_id': pl.Series([int(float(x)) for x in df_data['user_id']], dtype=pl.Int64)
        })

    def _reconstruct_events_dataframe_from_dict(self, events_data: dict[str, list[Any]]) -> pl.DataFrame:
        """Reconstruct the events DataFrame from stored data"""
        return pl.DataFrame({
            'time': pl.Series(events_data['time']).str.strptime(pl.Datetime, format='%Y-%m-%dT%H:%M:%S'),
            'event_type': pl.Series(events_data['event_type'], dtype=pl.String),
            'event_subtype': pl.Series(events_data['event_subtype'], dtype=pl.String),
            # Coerce numeric strings and mixed types to Float64
            'insulin_value': pl.Series(events_data['insulin_value']).cast(pl.Float64, strict=False)
        })

    def _build_figure(self, df: pl.DataFrame, events_df: pl.DataFrame, source_name: Optional[str] = None) -> go.Figure:
        """Build complete figure with all components"""
        figure = go.Figure()
        
        # Store data for internal methods
        self._current_df = df
        self._current_events = events_df
        self._current_source = source_name
        
        # Build all components
        self._add_range_rectangles(figure)
        self._add_glucose_trace(figure)
        self._add_prediction_traces(figure)
        self._add_event_markers(figure)
        self._update_layout(figure)
        
        return figure

    def _add_range_rectangles(self, figure: go.Figure) -> None:
        """Add colored range rectangles to indicate glucose ranges."""
        # Add rectangle for high range (>180 mg/dL)
        figure.add_hrect(
            y0=180, y1=400,
            fillcolor="rgba(255, 0, 0, 0.1)",
            line_width=0,
            xref='x',
            yref='y'
        )
        
        # Add rectangle for low range (<70 mg/dL)
        figure.add_hrect(
            y0=0, y1=70,
            fillcolor="rgba(255, 0, 0, 0.1)",
            line_width=0,
            xref='x',
            yref='y'
        )
        
        # Add rectangle for target range (70-180 mg/dL)
        figure.add_hrect(
            y0=70, y1=180,
            fillcolor="rgba(0, 255, 0, 0.1)",
            line_width=0,
            xref='x',
            yref='y'
        )

    def _calculate_y_axis_range(self) -> tuple[float, float]:
        """Calculates the y-axis range based on glucose and prediction values."""
        STANDARD_MIN = 40  # Standard lower bound for CGM charts
        STANDARD_MAX = 300  # Upper bound for CGM chart
        
        line_points = self._current_df.filter(pl.col("prediction") != 0.0)
        
        # Get actual data ranges
        data_min = self._current_df.get_column("gl").min()
        data_max = self._current_df.get_column("gl").max()
        
        # Include prediction values in range calculation if they exist
        if line_points.height > 0:
            pred_max = line_points.get_column("prediction").max()
            data_max = max(data_max, pred_max)
        
        # Set bounds with some padding
        lower_bound = min(STANDARD_MIN, max(0, data_min * 0.9))
        upper_bound = max(STANDARD_MAX, data_max * 1.1)
        
        return lower_bound, upper_bound

    def _add_glucose_trace(self, figure: go.Figure) -> None:
        """Adds the main glucose data line to the figure."""
        # Determine how many points to show based on hide_last_hour setting
        if self.hide_last_hour:
            # Show only data points minus PREDICTION_HOUR_OFFSET (hide last hour)
            visible_points = len(self._current_df) - PREDICTION_HOUR_OFFSET +1
            visible_df = self._current_df.slice(0, visible_points)
            x_indices = list(range(visible_points))
            glucose_values = visible_df['gl']
        else:
            # Show all data points
            x_indices = list(range(len(self._current_df)))
            glucose_values = self._current_df['gl']
        
        figure.add_trace(go.Scatter(
            x=x_indices,
            y=glucose_values,
            mode='lines+markers',
            name='Glucose Level',
            line=dict(color='blue'),
        ))


    def _get_time_position(self, time_point: datetime) -> float:
        """Converts a datetime to its corresponding x-axis position."""
        time_series = self._current_df.get_column("time")
        for idx, t in enumerate(time_series):
            if t == time_point:
                return idx
        return 0

    def _add_prediction_traces(self, figure: go.Figure) -> None:
        """Adds prediction points and connecting lines to the figure."""
        line_points = self._current_df.filter(pl.col("prediction") != 0.0)
        if line_points.height > 0:
            x_positions = [self._get_time_position(t) for t in line_points.get_column("time")]
            
            # Filter predictions to only show in allowed area when hiding last hour
            if self.hide_last_hour:
                visible_points = len(self._current_df) - PREDICTION_HOUR_OFFSET
                # Only show predictions in the hidden area (after PREDICTION_HOUR_OFFSET)
                filtered_data = []
                seen_positions = set()  # Track seen positions to avoid duplicates
                
                for i, (pos, pred, time_val) in enumerate(zip(x_positions, line_points.get_column("prediction"), line_points.get_column("time"))):
                    if pos >= visible_points and pos not in seen_positions:  # Only show unique predictions in the hidden area
                        filtered_data.append((pos, pred, time_val))
                        seen_positions.add(pos)
                
                # Sort by position to ensure proper order for line drawing
                filtered_data.sort(key=lambda x: x[0])
                
                if filtered_data:
                    x_positions, predictions, custom_data = zip(*filtered_data)
                    x_positions = list(x_positions)
                    predictions = list(predictions)
                    custom_data = list(custom_data)
                else:
                    x_positions, predictions, custom_data = [], [], []
            else:
                # Remove duplicates even when showing all predictions
                unique_data = []
                seen_positions = set()
                
                for i, (pos, pred, time_val) in enumerate(zip(x_positions, line_points.get_column("prediction"), line_points.get_column("time"))):
                    if pos not in seen_positions:
                        unique_data.append((pos, pred, time_val))
                        seen_positions.add(pos)
                
                # Sort by position to ensure proper order
                unique_data.sort(key=lambda x: x[0])
                
                if unique_data:
                    x_positions, predictions, custom_data = zip(*unique_data)
                    x_positions = list(x_positions)
                    predictions = list(predictions)
                    custom_data = list(custom_data)
                else:
                    predictions = line_points.get_column("prediction")
                    custom_data = line_points.get_column("time").to_list()
            
            if x_positions:  # Only add traces if we have data to show
                # Add prediction points
                figure.add_trace(go.Scatter(
                    x=x_positions,
                    y=predictions,
                    mode='markers',
                    name='Prediction Points',
                    marker=dict(
                        color='red',
                        size=8,
                        symbol='circle'
                    ),
                    hoverinfo='x+y',
                    hoverlabel=dict(bgcolor='white'),
                    customdata=custom_data
                ))

                # Add connecting lines between predictions
                if len(predictions) >= 2:
                    for i in range(len(predictions) - 1):
                        figure.add_trace(go.Scatter(
                            x=[x_positions[i], x_positions[i + 1]],
                            y=[predictions[i], predictions[i + 1]],
                            mode='lines',
                            line=dict(color='red', width=2),
                            showlegend=False,
                            hoverinfo='skip'
                        ))

    def _add_event_markers(self, figure: go.Figure) -> None:
        """Adds event markers (insulin, exercise, carbs) to the figure."""
        if self._current_events.height == 0:
            return
            
        # Filter events to only those within the current time window
        start_time = self._current_df.get_column("time")[0]
        end_time = self._current_df.get_column("time")[-1]
        
        window_events = self._current_events.filter(
            (pl.col("time") >= start_time) & 
            (pl.col("time") <= end_time)
        )
        
        # Add traces for each event type
        for event_type, style in self.EVENT_STYLES.items():
            events = window_events.filter(pl.col("event_type") == event_type)
            if events.height > 0:
                event_times = events.get_column("time")
                y_positions = []
                hover_texts = []
                x_positions = []
                
                for event_time in event_times:
                    # Find the glucose readings before and after the event
                    df_times = self._current_df.get_column("time")
                    
                    # Find indices of surrounding glucose readings
                    before_idx = None
                    after_idx = None
                    
                    for i, t in enumerate(df_times):
                        if t <= event_time:
                            before_idx = i
                        if t >= event_time and after_idx is None:
                            after_idx = i
                    
                    # Handle edge cases and interpolation
                    if before_idx is None:
                        before_idx = 0
                    if after_idx is None:
                        after_idx = len(df_times) - 1
                    
                    # Calculate position and glucose value
                    if df_times[before_idx] == event_time:
                        x_pos = before_idx
                        glucose_value = self._current_df.get_column("gl")[before_idx]
                    elif before_idx == after_idx:
                        x_pos = before_idx
                        glucose_value = self._current_df.get_column("gl")[before_idx]
                    else:
                        # Interpolate position and glucose value
                        before_time = df_times[before_idx].timestamp()
                        after_time = df_times[after_idx].timestamp()
                        event_timestamp = event_time.timestamp()
                        
                        factor = (event_timestamp - before_time) / (after_time - before_time)
                        x_pos = before_idx + factor
                        
                        before_glucose = self._current_df.get_column("gl")[before_idx]
                        after_glucose = self._current_df.get_column("gl")[after_idx]
                        glucose_value = before_glucose + (after_glucose - before_glucose) * factor
                    
                    y_positions.append(glucose_value)
                    x_positions.append(x_pos)
                    
                    # Create hover text
                    event_row = events.filter(pl.col("time") == event_time)
                    if event_type == 'Insulin':
                        hover_text = f"Insulin: {event_row.get_column('insulin_value')[0]}u<br>{event_time.strftime('%H:%M')}"
                    else:
                        hover_text = f"{event_type}<br>{event_time.strftime('%H:%M')}"
                    hover_texts.append(hover_text)
                
                figure.add_trace(go.Scatter(
                    x=x_positions,
                    y=y_positions,
                    mode='markers',
                    name=event_type,
                    marker=dict(
                        symbol=style['symbol'],
                        size=style['size'],
                        color=style['color'],
                        line=dict(width=2, color='white'),
                        opacity=0.8
                    ),
                    text=hover_texts,
                    hoverinfo='text'
                ))

    def _update_layout(self, figure: go.Figure) -> None:
        """Updates the figure layout with axes, margins, and interaction settings."""
        y_range = self._calculate_y_axis_range()
        
        # Calculate window info for title
        start_time = self._current_df.get_column("time")[0].strftime('%H:%M')
        end_time = self._current_df.get_column("time")[-1].strftime('%H:%M')
        
        # Create title with source information
        title_text = f'Glucose Levels ({start_time} to {end_time})'
        if self._current_source:
            title_text += f' - Source: {self._current_source}'
        
        figure.update_layout(
            title=title_text,
            autosize=True,
            xaxis=dict(
                title='Time',
                tickmode='array',
                tickvals=list(range(len(self._current_df))),
                ticktext=[t.strftime('%H:%M') for t in self._current_df.get_column("time")],
                fixedrange=True,
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                gridcolor='rgba(128, 128, 128, 0.2)',
                showgrid=True,
                range=[-0.5, len(self._current_df) - 0.5]
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

