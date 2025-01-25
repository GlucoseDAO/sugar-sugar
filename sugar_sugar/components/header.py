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


class HeaderComponent(Div):
    def __init__(self, children: List = None, **kwargs):
        if children is None:
            children = self._create_header_content()
            
        super().__init__(
            children=children,
            style={
                'padding': '15px',
                'marginBottom': '15px',
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
            },
            **kwargs
        )

    def _create_header_content(self) -> List:
        """Create the header section content with title and description"""
        return [
            html.H1('Sugar Sugar', 
                    style={
                        'textAlign': 'center',
                        'color': '#2c5282',
                        'marginBottom': '10px',
                        'fontSize': '48px',
                        'fontWeight': 'bold',
                    }),
            html.Div([
                # Left column - Game description
                html.Div([
                    html.P([
                        'Test your glucose prediction skills! ',
                        html.Br(),
                        'Click on the graph or draw lines to predict future glucose values. ',
                        'The game will show you how accurate your predictions are compared to actual measurements.'
                    ], style={
                        'fontSize': '18px',
                        'color': '#4a5568',
                        'lineHeight': '1.5'
                    })
                ], style={'flex': '1', 'paddingRight': '20px'}),
                
                # Right column - Upload and controls
                html.Div([
                    self._create_controls(),
                    self._create_upload_section(),
                ], style={'flex': '1'})
            ], style={
                'display': 'flex',
                'flexDirection': 'row',
                'gap': '20px',
                'alignItems': 'start'
            })
        ]

    def _create_controls(self):
        """Placeholder for controls creation - implement in app.py"""
        return html.Div()  # Empty div by default

    def _create_upload_section(self):
        """Placeholder for upload section creation - implement in app.py"""
        return html.Div()  # Empty div by default
