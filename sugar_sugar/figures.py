import plotly.graph_objs as go
import polars as pl
from typing import Tuple, List, Dict
from datetime import datetime

"""
creating the figures that wil be uploaded interactively
"""

def create_base_figure() -> go.Figure:
        """Creates the base figure with glucose range rectangles."""
        fig = go.Figure()
        
        # Dangerous low range (below 70)
        fig.add_hrect(
            y0=0, y1=70,
            fillcolor="rgba(255, 200, 200, 0.5)",  # Light red
            line=dict(color="rgba(200, 0, 0, 0.5)", width=1),
            layer="below",
            name="Low Range"
        )
        
        # Normal range (70-180)
        fig.add_hrect(
            y0=70, y1=180,
            fillcolor="rgba(200, 240, 200, 0.5)",  # Light green
            line=dict(color="rgba(0, 100, 0, 0.5)", width=1),
            layer="below",
            name="Normal Range"
        )
        
        # High range (180-250)
        fig.add_hrect(
            y0=180, y1=250,
            fillcolor="rgba(255, 255, 200, 0.5)",  # Light yellow
            line=dict(color="rgba(200, 200, 0, 0.5)", width=1),
            layer="below",
            name="High Range"
        )
        
        # Dangerous high range (above 250)
        fig.add_hrect(
            y0=250, y1=400,
            fillcolor="rgba(255, 200, 200, 0.5)",  # Light red
            line=dict(color="rgba(200, 0, 0, 0.5)", width=1),
            layer="below",
            name="Very High Range"
        )
        
        return fig

def calculate_y_axis_range(df: pl.DataFrame) -> Tuple[float, float]:
        """Calculates the y-axis range based on glucose and prediction values."""
        STANDARD_MIN = 40  # Standard lower bound for CGM charts
        STANDARD_MAX = 300  # Upper bound for CGM chart
        
        line_points = df.filter(pl.col("prediction") != 0.0)
        
        # Get actual data ranges
        data_min = df.get_column("gl").min()
        data_max = df.get_column("gl").max()
        
        # Include prediction values in range calculation if they exist
        if line_points.height > 0:
            pred_max = line_points.get_column("prediction").max()
            data_max = max(data_max, pred_max)
        
        # Set bounds
        lower_bound = min(STANDARD_MIN, max(0, data_min * 0.9))
        upper_bound = max(STANDARD_MAX, data_max * 1.1)
        
        return lower_bound, upper_bound

def add_glucose_trace(fig: go.Figure, df: pl.DataFrame) -> None:
        """Adds the main glucose data line to the figure."""
        x_indices = list(range(len(df)))
        
        fig.add_trace(go.Scatter(
            x=x_indices,
            y=df['gl'],
            mode='lines+markers',
            name='Glucose Level',
            line=dict(color='blue'),
        ))

def get_time_position(df: pl.DataFrame, time_point: datetime) -> float:
        """Converts a datetime to its corresponding x-axis position."""
        time_series = df.get_column("time")
        for idx, t in enumerate(time_series):
            if t == time_point:
                return idx
        return 0

def add_prediction_traces(fig: go.Figure, df: pl.DataFrame) -> None:
        """Adds prediction points and connecting lines to the figure."""
        line_points = df.filter(pl.col("prediction") != 0.0)
        if line_points.height > 0:
            x_positions = [get_time_position(df, t) for t in line_points.get_column("time")]
            
            # Add prediction points
            fig.add_trace(go.Scatter(
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
                    start_pos = get_time_position(df, times[i])
                    end_pos = get_time_position(df, times[i + 1])
                    
                    fig.add_trace(go.Scatter(
                        x=[start_pos, end_pos],
                        y=[predictions[i], predictions[i + 1]],
                        mode='lines',
                        line=dict(color='red', width=2),
                        showlegend=False
                    ))

def add_event_markers(fig: go.Figure, events_df: pl.DataFrame, df: pl.DataFrame) -> None:
        """Adds event markers (insulin, exercise, carbs) to the figure."""
        event_styles = {
            'Insulin': {'symbol': 'triangle-down', 'color': 'purple', 'size': 20},
            'Exercise': {'symbol': 'star', 'color': 'orange', 'size': 20},
            'Carbohydrates': {'symbol': 'square', 'color': 'green', 'size': 20}
        }



        # Filter events to only those within the current time window
        start_time = df.get_column("time")[0]
        end_time = df.get_column("time")[-1]
        
        window_events = events_df.filter(
            (pl.col("time") >= start_time) & 
            (pl.col("time") <= end_time)
        )
        
        # Add traces for each event type
        for event_type,style in event_styles.items():
            events = window_events.filter(pl.col("event_type") == event_type)
            if events.height > 0:
                event_times = events.get_column("time")
                y_positions = []
                hover_texts = []
                x_positions = []
                
                for event_time in event_times:
                    # Find the glucose readings before and after the event
                    df_times = df.get_column("time")
                    
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
                        glucose_value = df.get_column("gl")[before_idx]
                    elif before_idx == after_idx:
                        x_pos = before_idx
                        glucose_value = df.get_column("gl")[before_idx]
                    else:
                        # Interpolate position and glucose value
                        before_time = df_times[before_idx].timestamp()
                        after_time = df_times[after_idx].timestamp()
                        event_timestamp = event_time.timestamp()
                        
                        factor = (event_timestamp - before_time) / (after_time - before_time)
                        x_pos = before_idx + factor
                        
                        before_glucose = df.get_column("gl")[before_idx]
                        after_glucose = df.get_column("gl")[after_idx]
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
                
                fig.add_trace(go.Scatter(
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

def update_figure_layout(fig: go.Figure, df: pl.DataFrame) -> None:
        """Updates the figure layout with axes, margins, and interaction settings."""
        y_range = calculate_y_axis_range(df)
        
        # Calculate window info for title
        start_time = df.get_column("time")[0].strftime('%Y-%m-%d %H:%M')
        end_time = df.get_column("time")[-1].strftime('%Y-%m-%d %H:%M')
        
        fig.update_layout(
            title=f'Glucose Levels ({start_time} to {end_time})',
            autosize=True,
            xaxis=dict(
                title='Time',
                tickmode='array',
                tickvals=list(range(len(df))),
                ticktext=[t.strftime('%Y-%m-%d %H:%M') for t in df.get_column("time")],
                fixedrange=True,
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                gridcolor='rgba(128, 128, 128, 0.2)',
                showgrid=True,
                range=[-0.5, len(df) - 0.5]
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

def build_figure(df: pl.DataFrame, events_df: pl.DataFrame) -> go.Figure:
        """Builds complete figure with all components."""
        fig = create_base_figure()
        add_glucose_trace(fig, df)
        add_prediction_traces(fig, df)
        add_event_markers(fig, events_df, df)
        update_figure_layout(fig, df)
        return fig