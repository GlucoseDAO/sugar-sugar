from dash import html, dcc, dash_table
from .config import DEFAULT_POINTS, MIN_POINTS, MAX_POINTS

'''
Create the layout of the app, the base on which user will interact with
'''

def create_header() -> html.Div:
    """Create the header section with title and description"""
    return html.Div([
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
                create_controls(),
                create_upload_section(),
            ], style={'flex': '1'})
        ], style={
            'display': 'flex',
            'flexDirection': 'row',
            'gap': '20px',
            'alignItems': 'start'
        })
    ], style={
        'padding': '15px',
        'marginBottom': '15px',
        'backgroundColor': 'white',
        'borderRadius': '10px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    })

def create_controls() -> html.Div:
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

def create_upload_section() -> html.Div:
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

def create_graph_section() -> html.Div:
    """Create the main graph section"""
    return html.Div([
        dcc.Graph(
            id='glucose-graph',
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

def create_instructions() -> html.Div:
    """Create the game instructions section"""
    return html.Div([
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
    ], style={
        'padding': '15px',
        'backgroundColor': 'white',
        'borderRadius': '10px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
        'marginBottom': '20px'
    })

def create_predictions_table() -> html.Div:
    """Create the predictions table section"""
    return html.Div([
        html.H4('Glucose Values and Predictions', style={'textAlign': 'center'}),
        dash_table.DataTable(
            id='predictions-table',
            columns=[
                {'name': 'Metric', 'id': 'metric'},
                *[{'name': f'T{i}', 'id': f't{i}'} for i in range(24)]
            ],
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'center',
                'padding': '5px',
                'minWidth': '60px'
            },
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 0},
                    'backgroundColor': 'rgba(200, 240, 200, 0.5)'
                },
                {
                    'if': {'row_index': 1},
                    'backgroundColor': 'rgba(255, 200, 200, 0.5)'
                }
            ]
        )
    ], style={
        'padding': '20px',
        'backgroundColor': 'white',
        'borderRadius': '10px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
        'marginBottom': '20px'
    })

def create_layout() -> html.Div:
    """Create the main layout of the application"""
    return html.Div([
        # Header section
        create_header(),

        # Main content container
        html.Div([
            # Interactive glucose graph component
            create_graph_section(),

            # Game instructions
            create_instructions(),

            # Store component for click tracking
            dcc.Store(id='last-click-time', data=0),
            
            # Predictions table
            create_predictions_table(),
            
            # Metrics section
            html.Div([
                html.Div(id='error-metrics', style={
                    'marginBottom': '15px'
                })
            ], style={
                'padding': '15px',
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
            })
        ], style={
            'margin': '0 auto',
            'padding': '0 20px',
            'display': 'flex',
            'flexDirection': 'column',
            'gap': '20px'
        })
    ], style={
        'backgroundColor': '#f7fafc',
        'minHeight': '100vh',
        'padding': '20px'
    })