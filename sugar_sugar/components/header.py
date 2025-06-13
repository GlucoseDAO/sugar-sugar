from typing import List, Dict, Tuple, Optional, Any, Union
import dash
from dash import dcc, html

from dash.html import Div
from sugar_sugar.config import DEFAULT_POINTS, MIN_POINTS, MAX_POINTS


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

    def create_controls(self) -> html.Div:
        """Create the points control and time slider section"""
        return html.Div([
            html.Div([
                # Points control
                html.Div([
                    html.Label('Number of points to show:', style={'marginRight': '10px'}),
                    dcc.Input(
                        id='points-control',
                        type='number',
                        value=DEFAULT_POINTS,
                        min=MIN_POINTS,
                        max=MAX_POINTS,
                        style={'width': '80px'}
                    ),
                ], style={'flex': '0 0 auto', 'display': 'flex', 'alignItems': 'center'}),
                
                # Time slider
                html.Div([
                    html.Label('Time Window Position:', style={'marginRight': '10px'}),
                    dcc.Slider(
                        id='time-slider',
                        min=0,
                        max=100,  # This will be updated by callback
                        value=0,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": True},
                        updatemode='mouseup',
                        included=True,
                        step=1
                    ),
                ], style={'flex': '1', 'marginLeft': '20px'}),
            ], style={
                'display': 'flex',
                'flexDirection': 'row',
                'alignItems': 'center',
                'gap': '10px',
                'marginBottom': '10px'
            })
        ])

    def create_upload_section(self) -> html.Div:
        """Create the file upload section"""
        return html.Div([
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select a Dexcom/Libre CSV File')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                }
            ),
            html.Div(id='example-data-warning', style={'marginTop': '10px'})
        ])

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
                # Left column - Game description and instructions
                html.Div([
                    html.P([
                        'Test your glucose prediction skills! ',
                        html.Br(),
                        'Click on the graph or draw lines to predict future glucose values. ',
                        'The game will show you how accurate your predictions are compared to actual measurements.'
                    ], style={
                        'fontSize': '18px',
                        'color': '#4a5568',
                        'lineHeight': '1.5',
                        'marginBottom': '15px'
                    }),
                    html.P([
                        html.Strong('How to play:'),
                        html.Br(),
                        '1. Click and drag in the graph to add predictions',
                        html.Br(),
                        '2. Draw one line after another to create prediction curves',
                        html.Br(),
                        '3. Try to predict at least 5 points to see your accuracy metrics'
                    ], style={
                        'fontSize': '16px',
                        'color': '#4a5568',
                        'lineHeight': '1.8'
                    })
                ], style={'flex': '1', 'paddingRight': '20px'}),
                
                # Right column - Upload and controls
                html.Div([
                    self.create_controls(),
                    self.create_upload_section(),
                ], style={'flex': '1'})
            ], style={
                'display': 'flex',
                'flexDirection': 'row',
                'gap': '20px',
                'alignItems': 'start'
            })
        ]