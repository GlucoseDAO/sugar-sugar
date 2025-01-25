
from typing import List, Dict, Tuple, Optional, Any, Union
import dash
from dash import dcc, html, Output, Input, State, dash_table
import plotly.graph_objs as go
import pandas as pd
import polars as pl
from datetime import datetime
import time
from pathlib import Path
import base64
import tempfile
from dash.html import Div

class InstructionsComponent(Div):
    """Component for displaying game instructions"""
    
    def __init__(self) -> None:
        super().__init__(
            children=[
                html.P([
                    'How to play: ',
                    html.Br(),
                    '1. Click and drag in the graph to add predictions ',
                    html.Br(),
                    '2. Draw one line after another to create prediction curves ',
                    html.Br(),
                    '3. Double-click to reset your predictions ',
                    html.Br(),
                    '4. Try to predict at least 5 points to see your accuracy metrics'
                ], style={
                    'fontSize': '16px',
                    'color': '#4a5568',
                    'lineHeight': '1.5',
                    'margin': '10px 0'
                })
            ],
            style={
                'padding': '15px',
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'marginBottom': '20px'
            }
        )
