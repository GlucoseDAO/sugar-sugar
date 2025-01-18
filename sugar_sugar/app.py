from typing import List, Dict, Tuple, Optional, Any, Union
import dash
from dash import dcc, html, Output, Input, State, dash_table
import plotly.graph_objs as go
import pandas as pd
from datetime import datetime
import time
from pathlib import Path

# Type aliases for clarity
TimePoint = Dict[str, Union[pd.Timestamp, float]]
DrawnLines = List[TimePoint]
TableData = List[Dict[str, str]]
Figure = go.Figure

# Sample DataFrame
df = pd.DataFrame({
    'time': pd.date_range(start='2023-01-01', periods=24, freq='h'),
    'gl': [100, 110, 105, 115, 120, 125, 130, 128, 135, 140, 142, 138, 135, 132, 130, 128, 125, 123, 120, 118, 115, 113, 110, 108]
})

external_stylesheets: List[str] = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Graph(
        id='glucose-graph',
        config={
            'displayModeBar': True,
            'scrollZoom': False,
            'doubleClick': 'reset',
            'showAxisDragHandles': False,
            'showAxisRangeEntryBoxes': False,
            'displaylogo': False
        }
    ),
    dcc.Store(id='drawn-lines', data=[]),
    dcc.Store(id='last-click-time', data=0),
    html.Div([
        html.H4('Predictions', style={'textAlign': 'center'}),
        dash_table.DataTable(
            id='predictions-table',
            columns=[
                {'name': 'Start Time', 'id': 'start_time'},
                {'name': 'End Time', 'id': 'end_time'},
                {'name': 'Start Glucose', 'id': 'start_glucose'},
                {'name': 'End Glucose', 'id': 'end_glucose'},
                {'name': 'Slope (mg/dL/hour)', 'id': 'slope'}
            ],
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'center',
                'padding': '10px',
                'minWidth': '100px'
            },
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            }
        )
    ], style={'margin': '20px'})
])

DOUBLE_CLICK_THRESHOLD: int = 500  # milliseconds

def find_nearest_time(x: Union[str, pd.Timestamp]) -> pd.Timestamp:
    """
    Finds the nearest allowed time from the DataFrame 'df' for a given input time.

    Args:
        x: Input time value to snap to nearest allowed time

    Returns:
        pd.Timestamp: Nearest allowed time from the dataset
    """
    x_ts: pd.Timestamp = pd.to_datetime(x)
    time_diffs: pd.Series = (df['time'] - x_ts).abs()
    nearest_idx: int = time_diffs.idxmin()
    nearest_time: pd.Timestamp = df.loc[nearest_idx, 'time']
    
    print(f"Input Time: {x_ts}, Nearest Allowed Time: {nearest_time}")
    return nearest_time

@app.callback(
    [
        Output('drawn-lines', 'data'),
        Output('last-click-time', 'data')
    ],
    [
        Input('glucose-graph', 'clickData'),
        Input('glucose-graph', 'relayoutData'),  # Use relayoutData for drawing
    ],
    [
        State('last-click-time', 'data'),
        State('drawn-lines', 'data')
    ]
)
def handle_click(
    click_data: Optional[Dict[str, Any]],
    relayout_data: Optional[Dict[str, Any]], 
    last_click_time: int, 
    drawn_lines: Optional[DrawnLines]
) -> Tuple[DrawnLines, int]:
    """
    Handles click events and drawing on the graph.

    Args:
        click_data: Data about the click event
        relayout_data: Data about drawn shapes
        last_click_time: Timestamp of the last click
        drawn_lines: List of currently drawn lines

    Returns:
        Tuple containing updated drawn lines and last click timestamp
    """
    current_time: int = int(time.time() * 1000)
    drawn_lines = drawn_lines or []
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return drawn_lines, last_click_time
        
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Handle double-click reset
    if trigger_id == 'glucose-graph' and click_data:
        if current_time - last_click_time <= DOUBLE_CLICK_THRESHOLD:
            print("Double-click detected: Resetting drawn lines.")
            return [], current_time
    
    # Handle drawing mode
    if trigger_id == 'glucose-graph' and relayout_data:
        if 'shapes' in relayout_data:
            shapes = relayout_data['shapes']
            if shapes and len(shapes) > 0:
                latest_shape = shapes[-1]
                
                # Get start and end points of the drawn line
                start_x = latest_shape.get('x0')
                end_x = latest_shape.get('x1')
                start_y = latest_shape.get('y0')
                end_y = latest_shape.get('y1')
                
                if all(v is not None for v in [start_x, end_x, start_y, end_y]):
                    # Snap both points to nearest allowed times
                    start_time = find_nearest_time(start_x)
                    end_time = find_nearest_time(end_x)
                    
                    print(f"Drawing line from ({start_x}, {start_y}) to ({end_x}, {end_y})")
                    print(f"Snapped times: {start_time} to {end_time}")
                    
                    # Add both points
                    drawn_lines.extend([
                        {'x': start_time, 'y': float(start_y)},
                        {'x': end_time, 'y': float(end_y)}
                    ])
                    
                    return drawn_lines, current_time
    
    return drawn_lines, last_click_time

@app.callback(
    [
        Output('glucose-graph', 'figure'),
        Output('predictions-table', 'data')
    ],
    [
        Input('drawn-lines', 'data')
    ]
)
def update_graph(drawn_lines: Optional[DrawnLines]) -> Tuple[Figure, TableData]:
    """
    Updates the graph and predictions table based on drawn lines.

    Args:
        drawn_lines: List of points defining the drawn lines

    Returns:
        Tuple containing the updated figure and table data
    """
    fig: Figure = go.Figure()

    # Add range for normal glucose levels
    fig.add_hrect(
        y0=70, y1=180,
        fillcolor="rgba(200, 240, 200, 0.5)",
        line=dict(color="rgba(0, 100, 0, 0.5)", width=1),
        layer="below",
        name="Normal Range"
    )

    # Add glucose data line
    fig.add_trace(go.Scatter(
        x=df['time'],
        y=df['gl'],
        mode='lines+markers',
        name='Glucose Level',
        line=dict(color='blue'),
    ))

    # Add drawn lines
    drawn_lines = drawn_lines or []
    table_data: TableData = []
    
    for i in range(0, len(drawn_lines), 2):
        if i + 1 < len(drawn_lines):
            start: TimePoint = drawn_lines[i]
            end: TimePoint = drawn_lines[i + 1]
            
            fig.add_shape(
                type='line',
                x0=start['x'],
                y0=start['y'],
                x1=end['x'],
                y1=end['y'],
                line=dict(color='red', width=2),
            )
            print(f"Drawing line from {start} to {end}")

            # Calculate time difference in hours
            start_time: pd.Timestamp = pd.to_datetime(start['x'])
            end_time: pd.Timestamp = pd.to_datetime(end['x'])
            time_diff: float = (end_time - start_time).total_seconds() / 3600
            
            # Calculate slope
            slope: float = (end['y'] - start['y']) / time_diff if time_diff != 0 else 0

            table_data.append({
                'start_time': start_time.strftime('%Y-%m-%d %H:%M'),
                'end_time': end_time.strftime('%Y-%m-%d %H:%M'),
                'start_glucose': f"{start['y']:.1f}",
                'end_glucose': f"{end['y']:.1f}",
                'slope': f"{slope:.1f}"
            })

    # Update layout
    fig.update_layout(
        title='Glucose Levels Over Time',
        xaxis=dict(
            title='Time',
            tickmode='array',
            tickvals=df['time'],
            ticktext=df['time'].dt.strftime('%Y-%m-%d %H:%M'),
            fixedrange=True,
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            gridcolor='rgba(128, 128, 128, 0.2)',
            showgrid=True
        ),
        yaxis=dict(
            title='Glucose Level (mg/dL)',
            fixedrange=True,
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            gridcolor='rgba(128, 128, 128, 0.2)',
            showgrid=True
        ),
        showlegend=True,
        dragmode='drawline',  # Back to drawline mode
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    return fig, table_data

def main() -> None:
    """Starts the Dash server."""
    app.run_server(debug=True)

if __name__ == '__main__':
    main()
