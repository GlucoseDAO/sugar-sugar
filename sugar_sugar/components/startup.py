from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import dash

class StartupPage:
    def __init__(self):
        self.id = 'startup-page'
        
    def __call__(self):
        return html.Div([
            html.H1("Sugar Sugar", 
                style={
                    'textAlign': 'center', 
                    'marginBottom': '30px', 
                    'fontSize': '48px',
                    'fontWeight': 'bold',
                    'color': '#2c5282'  # Match the prediction page color
                }
            ),
            html.Div([
                html.Div([
                    html.Label("Email", style={'fontSize': '24px', 'marginBottom': '10px'}),
                    dcc.Input(
                        id='email-input',
                        type='email',
                        placeholder='Enter your email',
                        style={'width': '100%', 'padding': '10px', 'fontSize': '20px', 'marginBottom': '20px'}
                    ),
                    
                    html.Label("Age", style={'fontSize': '24px', 'marginBottom': '10px'}),
                    dcc.Input(
                        id='age-input',
                        type='number',
                        placeholder='Enter your age',
                        min=0,
                        max=120,
                        style={'width': '100%', 'padding': '10px', 'fontSize': '20px', 'marginBottom': '20px'}
                    ),
                    
                    html.Label("Gender", style={'fontSize': '24px', 'marginBottom': '10px'}),
                    dcc.Dropdown(
                        id='gender-dropdown',
                        options=[
                            {'label': 'Male', 'value': 'M'},
                            {'label': 'Female', 'value': 'F'},
                            {'label': 'N/A', 'value': 'N/A'}
                        ],
                        placeholder='Select your gender',
                        style={'fontSize': '20px', 'marginBottom': '20px'}
                    ),
                    
                    html.Label("Are you diabetic?", style={'fontSize': '24px', 'marginBottom': '10px'}),
                    dcc.Dropdown(
                        id='diabetic-dropdown',
                        options=[
                            {'label': 'Yes', 'value': True},
                            {'label': 'No', 'value': False}
                        ],
                        placeholder='Select your diabetic status',
                        style={'fontSize': '20px', 'marginBottom': '20px'}
                    ),
                    
                    html.Div(id='diabetic-details', children=[
                        html.Label("Type of Diabetes", style={'fontSize': '24px', 'marginBottom': '10px'}),
                        dcc.Dropdown(
                            id='diabetic-type-dropdown',
                            options=[
                                {'label': 'Type 1', 'value': 'Type 1'},
                                {'label': 'Type 2', 'value': 'Type 2'},
                                {'label': 'N/A', 'value': 'N/A'}
                            ],
                            placeholder='Select type of diabetes',
                            style={'fontSize': '20px', 'marginBottom': '20px'}
                        ),
                        
                        html.Label("Duration of Diabetes (years)", style={'fontSize': '24px', 'marginBottom': '10px'}),
                        dcc.Input(
                            id='diabetes-duration-input',
                            type='number',
                            placeholder='Enter number of years',
                            min=0,
                            max=100,
                            style={'width': '100%', 'padding': '10px', 'fontSize': '20px', 'marginBottom': '20px'}
                        )
                    ]),
                    
                    html.Label("Do you have other medical conditions?", style={'fontSize': '24px', 'marginBottom': '10px'}),
                    dcc.Dropdown(
                        id='medical-conditions-dropdown',
                        options=[
                            {'label': 'Yes', 'value': True},
                            {'label': 'No', 'value': False}
                        ],
                        placeholder='Select if you have other medical conditions',
                        style={'fontSize': '20px', 'marginBottom': '20px'}
                    ),
                    
                    html.Div(id='medical-conditions-details', children=[
                        html.Label("Description of Medical Conditions", style={'fontSize': '24px', 'marginBottom': '10px'}),
                        dcc.Textarea(
                            id='medical-conditions-input',
                            placeholder='Describe your medical conditions',
                            style={'width': '100%', 'padding': '10px', 'fontSize': '20px', 'marginBottom': '20px', 'minHeight': '100px'}
                        )
                    ]),
                    
                    html.Label("Location", style={'fontSize': '24px', 'marginBottom': '10px'}),
                    dcc.Input(
                        id='location-input',
                        type='text',
                        placeholder='Enter your location',
                        style={'width': '100%', 'padding': '10px', 'fontSize': '20px', 'marginBottom': '20px'}
                    ),
                    
                    html.Div([
                        html.Button(
                            'Start Prediction',
                            id='start-button',
                            style={
                                'backgroundColor': '#4CAF50',
                                'color': 'white',
                                'padding': '20px 30px',
                                'border': 'none',
                                'borderRadius': '5px',
                                'fontSize': '24px',
                                'cursor': 'pointer',
                                'width': '100%',
                                'height': '80px',
                                'display': 'flex',
                                'alignItems': 'center',
                                'justifyContent': 'center',
                                'lineHeight': '1.2'
                            }
                        )
                    ], style={'textAlign': 'center', 'marginTop': '30px', 'marginBottom': '30px'})
                ], style={'maxWidth': '600px', 'margin': '0 auto', 'padding': '20px'})
            ], style={'backgroundColor': 'white', 'borderRadius': '10px', 'boxShadow': '0 0 10px rgba(0,0,0,0.1)'})
        ], style={
            'padding': '20px', 
            'backgroundColor': '#f5f5f5', 
            'minHeight': '100vh',
            'display': 'flex',
            'flexDirection': 'column'
        })

    def register_callbacks(self, app):
        @app.callback(
            [Output('diabetic-details', 'style'),
             Output('diabetic-type-dropdown', 'value'),
             Output('diabetes-duration-input', 'value')],
            [Input('diabetic-dropdown', 'value')]
        )
        def update_diabetic_details(is_diabetic):
            if is_diabetic is None:
                return {'display': 'none'}, None, None
            elif is_diabetic:
                return {'display': 'block'}, None, None
            else:
                return {'display': 'none'}, 'N/A', 0

        @app.callback(
            [Output('medical-conditions-details', 'style'),
             Output('medical-conditions-input', 'value')],
            [Input('medical-conditions-dropdown', 'value')]
        )
        def update_medical_conditions_details(has_conditions):
            if has_conditions is None:
                return {'display': 'none'}, None
            elif has_conditions:
                return {'display': 'block'}, None
            else:
                return {'display': 'none'}, 'N/A' 