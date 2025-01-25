import plotly.graph_objs as go
import polars as pl
from typing import Tuple
from datetime import datetime
from typing import Tuple
from dash import dcc, html
import plotly.graph_objs as go
import polars as pl
from datetime import datetime
from dash.html import Div


class GlucoseChart(Div):
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

    def __init__(self, id: str = 'glucose-chart'):
        # Initialize as a Div with the graph component
        super().__init__([
            dcc.Graph(
                id=id,
                config={
                    'displayModeBar': True,
                    'scrollZoom': False,
                    'doubleClick': 'reset',
                    'showAxisDragHandles': False,
                    'showAxisRangeEntryBoxes': False,
                    'displaylogo': False
                },
                style={'height': '100%'}
            )
        ], style={
            'padding': '20px',
            'backgroundColor': 'white',
            'borderRadius': '10px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'marginBottom': '20px',
            'display': 'flex',
            'flexDirection': 'column'
        })
        
        self.id = id
        self._data_store = {
            'glucose_data': None,
            'events_data': None
        }
        self._figure = go.Figure()

    def update(self, df: pl.DataFrame, events_df: pl.DataFrame) -> go.Figure:
        """Updates the chart with new data and returns the updated figure"""
        self._data_store['glucose_data'] = df
        self._data_store['events_data'] = events_df
        
        # Clear existing traces
        self._figure.data = []
        
        # Rebuild the figure
        self._build()
        return self._figure

    def _build(self):
        """Builds complete figure with all components."""
        self._add_range_rectangles()
        self._add_glucose_trace()
        self._add_prediction_traces()
        self._add_event_markers()
        self._update_layout()

    def _add_range_rectangles(self):
        """Adds the glucose range rectangles to the figure."""
        ranges = [
            (0, 70, "dangerous_low", "Low Range"),
            (70, 180, "normal", "Normal Range"),
            (180, 250, "high", "High Range"),
            (250, 400, "dangerous_high", "Very High Range")
        ]

        for y0, y1, style_key, name in ranges:
            colors = self.RANGE_COLORS[style_key]
            self._figure.add_hrect(
                y0=y0, y1=y1,
                fillcolor=colors["fill"],
                line=dict(color=colors["line"], width=1),
                layer="below",
                name=name
            )

    def calculate_y_axis_range(self) -> Tuple[float, float]:
        """Calculates the y-axis range based on glucose and prediction values."""
        STANDARD_MIN = 40  # Standard lower bound for CGM charts
        STANDARD_MAX = 300  # Upper bound for CGM chart
        
        line_points = self._data_store['glucose_data'].filter(pl.col("prediction") != 0.0)
        
        # Get actual data ranges
        data_min = self._data_store['glucose_data'].get_column("gl").min()
        data_max = self._data_store['glucose_data'].get_column("gl").max()
        
        # Include prediction values in range calculation if they exist
        if line_points.height > 0:
            pred_max = line_points.get_column("prediction").max()
            data_max = max(data_max, pred_max)
        
        # Set bounds
        lower_bound = min(STANDARD_MIN, max(0, data_min * 0.9))
        upper_bound = max(STANDARD_MAX, data_max * 1.1)
        
        return lower_bound, upper_bound

    def _add_glucose_trace(self):
        """Adds the main glucose data line to the figure."""
        x_indices = list(range(len(self._data_store['glucose_data'])))
        
        self._figure.add_trace(go.Scatter(
            x=x_indices,
            y=self._data_store['glucose_data']['gl'],
            mode='lines+markers',
            name='Glucose Level',
            line=dict(color='blue'),
        ))

    def get_time_position(self, time_point: datetime) -> float:
        """Converts a datetime to its corresponding x-axis position."""
        time_series = self._data_store['glucose_data'].get_column("time")
        for idx, t in enumerate(time_series):
            if t == time_point:
                return idx
        return 0

    def _add_prediction_traces(self):
        """Adds prediction points and connecting lines to the figure."""
        line_points = self._data_store['glucose_data'].filter(pl.col("prediction") != 0.0)
        if line_points.height > 0:
            x_positions = [self.get_time_position(t) for t in line_points.get_column("time")]
            
            # Add prediction points
            self._figure.add_trace(go.Scatter(
                x=x_positions,
                y=line_points.get_column("prediction"),
                mode='markers',
                name='Prediction Points',
                marker=dict(color='red', size=8)
            ))

            # Add connecting lines between predictions
            if line_points.height >= 2:
                line_points_sorted = line_points.sort("time")
                times = line_points_sorted.get_column("time")
                predictions = line_points_sorted.get_column("prediction")
                
                for i in range(line_points.height - 1):
                    start_pos = self.get_time_position(times[i])
                    end_pos = self.get_time_position(times[i + 1])
                    
                    self._figure.add_trace(go.Scatter(
                        x=[start_pos, end_pos],
                        y=[predictions[i], predictions[i + 1]],
                        mode='lines',
                        line=dict(color='red', width=2),
                        showlegend=False
                    ))

    def _add_event_markers(self):
        """Adds event markers (insulin, exercise, carbs) to the figure."""
        # Filter events to only those within the current time window
        start_time = self._data_store['glucose_data'].get_column("time")[0]
        end_time = self._data_store['glucose_data'].get_column("time")[-1]
        
        window_events = self._data_store['events_data'].filter(
            (pl.col("time") >= start_time) & 
            (pl.col("time") <= end_time)
        )
        
        # Add traces for each event type
        for event_type,style in self.EVENT_STYLES.items():
            events = window_events.filter(pl.col("event_type") == event_type)
            if events.height > 0:
                event_times = events.get_column("time")
                y_positions = []
                hover_texts = []
                x_positions = []
                
                for event_time in event_times:
                    # Find the glucose readings before and after the event
                    df_times = self._data_store['glucose_data'].get_column("time")
                    
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
                        glucose_value = self._data_store['glucose_data'].get_column("gl")[before_idx]
                    elif before_idx == after_idx:
                        x_pos = before_idx
                        glucose_value = self._data_store['glucose_data'].get_column("gl")[before_idx]
                    else:
                        # Interpolate position and glucose value
                        before_time = df_times[before_idx].timestamp()
                        after_time = df_times[after_idx].timestamp()
                        event_timestamp = event_time.timestamp()
                        
                        factor = (event_timestamp - before_time) / (after_time - before_time)
                        x_pos = before_idx + factor
                        
                        before_glucose = self._data_store['glucose_data'].get_column("gl")[before_idx]
                        after_glucose = self._data_store['glucose_data'].get_column("gl")[after_idx]
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
                
                self._figure.add_trace(go.Scatter(
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

    def _update_layout(self):
        """Updates the figure layout with axes, margins, and interaction settings."""
        y_range = self.calculate_y_axis_range()
        
        # Calculate window info for title
        start_time = self._data_store['glucose_data'].get_column("time")[0].strftime('%Y-%m-%d %H:%M')
        end_time = self._data_store['glucose_data'].get_column("time")[-1].strftime('%Y-%m-%d %H:%M')
        
        self._figure.update_layout(
            title=f'Glucose Levels ({start_time} to {end_time})',
            autosize=True,
            xaxis=dict(
                title='Time',
                tickmode='array',
                tickvals=list(range(len(self._data_store['glucose_data']))),
                ticktext=[t.strftime('%Y-%m-%d %H:%M') for t in self._data_store['glucose_data'].get_column("time")],
                fixedrange=True,
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                gridcolor='rgba(128, 128, 128, 0.2)',
                showgrid=True,
                range=[-0.5, len(self._data_store['glucose_data']) - 0.5]
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