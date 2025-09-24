from typing import Sequence, Optional, Any
from dash import dcc, html

from dash.html import Div
from sugar_sugar.config import DEFAULT_POINTS, MIN_POINTS, MAX_POINTS


class HeaderComponent(Div):
    def __init__(self, show_time_slider: bool = True, children: Optional[Sequence[Any]] = None, **kwargs: Any) -> None:
        self.show_time_slider = show_time_slider
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
        controls_children = [
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
            ], style={
                'display': 'flex',
                'flexDirection': 'row',
                'alignItems': 'center',
                'gap': '10px',
                'marginBottom': '10px'
            })
        ]

        # Always include the time slider for functionality, but conditionally hide it
        time_slider_div = html.Div([
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
        ], style={
            'flex': '1',
            'marginLeft': '20px',
            'display': 'block' if self.show_time_slider else 'none'  # Hide when show_time_slider is False
        })

        # Add the time slider to the first child's style
        if self.show_time_slider:
            controls_children[0]['props']['children'].append(time_slider_div)
        else:
            # Still include the slider but hidden for callback functionality
            controls_children.append(time_slider_div)

        return html.Div(controls_children)

    def create_upload_section(self) -> html.Div:
        """Create the file upload section"""
        return html.Div([
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Upload Your Data Here')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'marginBottom': '10px'
                }
            ),
            html.Button(
                'Use Example Data',
                id='use-example-data-button',
                style={
                    'width': '100%',
                    'height': '40px',
                    'backgroundColor': '#f8f9fa',
                    'border': '1px solid #dee2e6',
                    'borderRadius': '5px',
                    'cursor': 'pointer',
                    'fontSize': '14px',
                    'color': '#6c757d'
                }
            ),
            html.Div(id='example-data-warning', style={'marginTop': '10px'})
        ])

    def _create_header_content(self) -> Sequence[Any]:
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
                    # Add data source indicator with improved visibility
                    html.Div([
                        html.Label('Current Data Source:', style={
                            'fontWeight': 'bold',
                            'marginRight': '8px',
                            'color': '#4a5568',
                            'fontSize': '14px'
                        }),
                        html.Div(id='data-source-display', children="example.csv", style={
                            'color': '#2c5282',
                            'fontStyle': 'italic',
                            'fontSize': '14px',
                            'fontWeight': '500',
                            'overflow': 'hidden',
                            'textOverflow': 'ellipsis',
                            'maxWidth': '300px',
                            'whiteSpace': 'nowrap'
                        })
                    ], style={
                        'marginTop': '15px',
                        'padding': '10px',
                        'backgroundColor': '#f7fafc',
                        'borderRadius': '5px',
                        'display': 'flex',
                        'alignItems': 'center',
                        'width': '100%',
                        'boxShadow': '0 1px 2px rgba(0,0,0,0.05)'
                    })
                ], style={'flex': '1'})
            ], style={
                'display': 'flex',
                'flexDirection': 'row',
                'gap': '20px',
                'alignItems': 'start'
            })
        ]