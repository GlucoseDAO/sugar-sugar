from dash import html, dcc
from dash.dependencies import Input, Output, State
from dash import no_update
import dash
from typing import Any, Optional
# DEBUG_MODE will be imported dynamically to get the latest value
from sugar_sugar.i18n import t

class StartupPage(html.Div):
    def __init__(self, *, locale: str = "en") -> None:
        self.component_id: str = 'startup-page'
        self._locale: str = locale
        
        # Create the layout
        layout = [
            html.A(
                t("ui.startup.fork_me", locale=locale),
                href="https://github.com/GlucoseDAO/sugar-sugar",
                target="_blank",
                rel="noopener noreferrer",
                className="github-fork-ribbon github-fork-ribbon-right-bottom fixed",
                **{"data-ribbon": "Fork me on GitHub"}
            ),
            html.H1(t("ui.common.app_title", locale=locale), 
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
                    html.Div([
                        html.P(t("ui.startup.required_fields_note", locale=locale), style={'color': '#666', 'fontSize': '16px', 'fontStyle': 'italic', 'marginBottom': '20px', 'textAlign': 'right'})
                    ]),
                    
                    html.Div([
                        html.Label(t("ui.startup.email_label", locale=locale), style={'fontSize': '24px', 'marginBottom': '10px', 'color': '#333', 'display': 'inline-block'}),
                        html.Span(id='email-required', children=' *', style={'color': '#d32f2f', 'fontSize': '24px', 'fontWeight': 'bold'})
                    ], style={'marginBottom': '10px'}),
                    dcc.Input(
                        id='email-input',
                        type='email',
                        placeholder=t("ui.startup.email_placeholder", locale=locale),
                        style={'width': '100%', 'padding': '10px', 'fontSize': '20px', 'marginBottom': '20px'}
                    ),
                    
                    html.Div([
                        html.Label(t("ui.startup.age_label", locale=locale), style={'fontSize': '24px', 'marginBottom': '10px', 'color': '#333', 'display': 'inline-block'}),
                        html.Span(id='age-required', children=' *', style={'color': '#d32f2f', 'fontSize': '24px', 'fontWeight': 'bold'})
                    ], style={'marginBottom': '10px'}),
                    dcc.Input(
                        id='age-input',
                        type='number',
                        placeholder=t("ui.startup.age_placeholder", locale=locale),
                        min=0,
                        max=120,
                        style={'width': '100%', 'padding': '10px', 'fontSize': '20px', 'marginBottom': '20px'}
                    ),
                    
                    html.Div([
                        html.Label(t("ui.startup.gender_label", locale=locale), style={'fontSize': '24px', 'marginBottom': '10px', 'color': '#333', 'display': 'inline-block'}),
                        html.Span(id='gender-required', children=' *', style={'color': '#d32f2f', 'fontSize': '24px', 'fontWeight': 'bold'})
                    ], style={'marginBottom': '10px'}),
                    dcc.Dropdown(
                        id='gender-dropdown',
                        options=[
                            {'label': t("ui.startup.gender_male", locale=locale), 'value': 'M'},
                            {'label': t("ui.startup.gender_female", locale=locale), 'value': 'F'},
                            {'label': t("ui.startup.gender_na", locale=locale), 'value': 'N/A'}
                        ],
                        placeholder=t("ui.startup.gender_placeholder", locale=locale),
                        style={'fontSize': '20px', 'marginBottom': '20px'}
                    ),

                    html.Label(t("ui.startup.cgm_label", locale=locale), style={'fontSize': '24px', 'marginBottom': '10px', 'color': '#333'}),
                    dcc.Dropdown(
                        id='cgm-dropdown',
                        options=[
                            {'label': t("ui.startup.yes", locale=locale), 'value': True},
                            {'label': t("ui.startup.no", locale=locale), 'value': False}
                        ],
                        placeholder=t("ui.startup.cgm_placeholder", locale=locale),
                        style={'fontSize': '20px', 'marginBottom': '20px'}
                    ),

                    html.Div(id='cgm-details', children=[
                        html.Label(t("ui.startup.cgm_duration_label", locale=locale), style={'fontSize': '24px', 'marginBottom': '10px', 'color': '#333'}),
                        dcc.Input(
                            id='cgm-duration-input',
                            type='number',
                            placeholder=t("ui.startup.cgm_duration_placeholder", locale=locale),
                            min=0,
                            max=100,
                            style={'width': '100%', 'padding': '10px', 'fontSize': '20px', 'marginBottom': '20px'}
                        )
                    ]),
                    
                    html.Div([
                        html.Label(t("ui.startup.diabetic_label", locale=locale), style={'fontSize': '24px', 'marginBottom': '10px', 'color': '#333', 'display': 'inline-block'}),
                        html.Span(id='diabetic-required', children=' *', style={'color': '#d32f2f', 'fontSize': '24px', 'fontWeight': 'bold'})
                    ], style={'marginBottom': '10px'}),
                    dcc.Dropdown(
                        id='diabetic-dropdown',
                        options=[
                            {'label': t("ui.startup.yes", locale=locale), 'value': True},
                            {'label': t("ui.startup.no", locale=locale), 'value': False}
                        ],
                        placeholder=t("ui.startup.diabetic_placeholder", locale=locale),
                        style={'fontSize': '20px', 'marginBottom': '20px'}
                    ),
                    
                    html.Div(id='diabetic-details', children=[
                        html.Div([
                            html.Label(t("ui.startup.diabetes_type_label", locale=locale), style={'fontSize': '24px', 'marginBottom': '10px', 'color': '#333', 'display': 'inline-block'}),
                            html.Span(id='diabetic-type-required', children=' *', style={'color': '#d32f2f', 'fontSize': '24px', 'fontWeight': 'bold'})
                        ], style={'marginBottom': '10px'}),
                        dcc.Dropdown(
                            id='diabetic-type-dropdown',
                            options=[
                                {'label': t("ui.startup.diabetes_type_1", locale=locale), 'value': 'Type 1'},
                                {'label': t("ui.startup.diabetes_type_2", locale=locale), 'value': 'Type 2'},
                                {'label': t("ui.startup.gender_na", locale=locale), 'value': 'N/A'}
                            ],
                            placeholder=t("ui.startup.diabetes_type_placeholder", locale=locale),
                            style={'fontSize': '20px', 'marginBottom': '20px'}
                        ),
                        
                        html.Div([
                            html.Label(t("ui.startup.diabetes_duration_label", locale=locale), style={'fontSize': '24px', 'marginBottom': '10px', 'color': '#333', 'display': 'inline-block'}),
                            html.Span(id='diabetes-duration-required', children=' *', style={'color': '#d32f2f', 'fontSize': '24px', 'fontWeight': 'bold'})
                        ], style={'marginBottom': '10px'}),
                        dcc.Input(
                            id='diabetes-duration-input',
                            type='number',
                            placeholder=t("ui.startup.diabetes_duration_placeholder", locale=locale),
                            min=0,
                            max=100,
                            style={'width': '100%', 'padding': '10px', 'fontSize': '20px', 'marginBottom': '20px'}
                        )
                    ]),
                    
                    html.Div([
                        html.Label(t("ui.startup.location_label", locale=locale), style={'fontSize': '24px', 'marginBottom': '10px', 'color': '#333', 'display': 'inline-block'}),
                        html.Span(id='location-required', children=' *', style={'color': '#d32f2f', 'fontSize': '24px', 'fontWeight': 'bold'})
                    ], style={'marginBottom': '10px'}),
                    dcc.Input(
                        id='location-input',
                        type='text',
                        placeholder=t("ui.startup.location_placeholder", locale=locale),
                        style={'width': '100%', 'padding': '10px', 'fontSize': '20px', 'marginBottom': '20px'}
                    ),
                    
                    html.Div(
                        [
                            html.Hr(style={'margin': '30px 0', 'border': '1px solid #ddd'}),
                            html.H3(
                                t("ui.startup.contact_prefs_title", locale=locale),
                                style={'fontSize': '24px', 'marginBottom': '12px', 'color': '#2c5282'}
                            ),
                            html.P(
                                t("ui.startup.contact_prefs_text", locale=locale),
                                style={'fontSize': '18px', 'lineHeight': '1.6', 'marginBottom': '0', 'color': '#555'}
                            ),
                        ],
                        style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '8px', 'marginBottom': '20px'}
                    ),
                    
                    # <!-- START INSERTION: Just Test Me Button (Debug Mode Only) --> 
                    html.Div([
                        html.Button(
                            t("ui.startup.just_test_me", locale=locale),
                            id='test-me-button',
                            style={
                                'backgroundColor': '#2196F3',
                                'color': 'white',
                                'padding': '15px 25px',
                                'border': 'none',
                                'borderRadius': '5px',
                                'fontSize': '18px',
                                'cursor': 'pointer',
                                'width': '100%',
                                'height': '60px',
                                'display': 'flex',
                                'alignItems': 'center',
                                'justifyContent': 'center',
                                'lineHeight': '1.2',
                                'marginBottom': '15px'
                            }
                        )
                    ], style={
                        'textAlign': 'center', 
                        'marginTop': '30px',
                        'display': 'block' if self._get_debug_mode() else 'none'
                    }),
                    # <!-- END INSERTION: Just Test Me Button (Debug Mode Only) -->
                    
                    html.Div([
                        html.Button(
                            t("ui.startup.start_prediction", locale=locale),
                            id='start-button',
                            disabled=True,  # Initially disabled until consent is given
                            style={
                                'backgroundColor': '#cccccc',  # Gray when disabled
                                'color': 'white',
                                'padding': '20px 30px',
                                'border': 'none',
                                'borderRadius': '5px',
                                'fontSize': '24px',
                                'cursor': 'not-allowed',  # Show not-allowed cursor when disabled
                                'width': '100%',
                                'height': '80px',
                                'display': 'flex',
                                'alignItems': 'center',
                                'justifyContent': 'center',
                                'lineHeight': '1.2'
                            }
                        )
                    ], style={'textAlign': 'center', 'marginBottom': '30px'}),
                    

                ], style={'maxWidth': '600px', 'margin': '0 auto', 'padding': '20px'})
            ], style={'backgroundColor': 'white', 'borderRadius': '10px', 'boxShadow': '0 0 10px rgba(0,0,0,0.1)'})
        ]
        
        # Initialize the parent html.Div with the layout and styling
        super().__init__(
            children=layout,
            id=self.component_id,
            style={
                'padding': '20px', 
                'backgroundColor': '#f5f5f5', 
                'minHeight': '100vh',
                'display': 'flex',
                'flexDirection': 'column'
            }
        )

    def _get_debug_mode(self) -> bool:
        """Dynamically get the current DEBUG_MODE value."""
        try:
            from sugar_sugar.config import DEBUG_MODE
            return DEBUG_MODE
        except ImportError:
            return False

    def register_callbacks(self, app: dash.Dash) -> None:
        @app.callback(
            [Output('diabetic-details', 'style'),
             Output('diabetic-type-dropdown', 'value'),
             Output('diabetes-duration-input', 'value')],
            [Input('diabetic-dropdown', 'value')],
            [State('test-me-button', 'n_clicks'),
             State('email-input', 'value')]
        )
        def update_diabetic_details(
            is_diabetic: Optional[bool],
            test_clicks: Optional[int],
            email: Optional[str]
        ) -> tuple[dict[str, str], Optional[str], Optional[int]]:
            if is_diabetic is None:
                return {'display': 'none'}, None, None
            elif is_diabetic:
                # Check if this is from the test button (email will be test email)
                if test_clicks and email and 'test.user@example.com' in str(email):
                    return {'display': 'block'}, 'Type 1', 5
                else:
                    return {'display': 'block'}, None, None
            else:
                return {'display': 'none'}, 'N/A', 0

        @app.callback(
            [Output('cgm-details', 'style'),
             Output('cgm-duration-input', 'value')],
            [Input('cgm-dropdown', 'value')],
            [State('test-me-button', 'n_clicks'),
             State('email-input', 'value')]
        )
        def update_cgm_details(
            uses_cgm: Optional[bool],
            test_clicks: Optional[int],
            email: Optional[str],
        ) -> tuple[dict[str, str], Optional[float]]:
            if uses_cgm is True:
                if test_clicks and email and 'test.user@example.com' in str(email):
                    return {'display': 'block'}, 3
                return {'display': 'block'}, None
            return {'display': 'none'}, None

        @app.callback(
            [Output('start-button', 'disabled'),
             Output('start-button', 'style'),
             Output('email-required', 'style'),
             Output('age-required', 'style'),
             Output('gender-required', 'style'),
             Output('diabetic-required', 'style'),
             Output('diabetic-type-required', 'style'),
             Output('diabetes-duration-required', 'style'),
             Output('location-required', 'style')],
            [Input('email-input', 'value'),
             Input('age-input', 'value'),
             Input('gender-dropdown', 'value'),
             Input('diabetic-dropdown', 'value'),
             Input('diabetic-type-dropdown', 'value'),
             Input('diabetes-duration-input', 'value'),
             Input('location-input', 'value'),
             Input('user-info-store', 'data')]
        )
        def update_form_validation(
            email: Optional[str], 
            age: Optional[int | float], 
            gender: Optional[str], 
            is_diabetic: Optional[bool], 
            diabetic_type: Optional[str], 
            diabetes_duration: Optional[int | float], 
            location: Optional[str],
            user_info: Optional[dict[str, Any]]
        ) -> tuple[
            bool,
            dict[str, str | int],
            dict[str, str | int],
            dict[str, str | int],
            dict[str, str | int],
            dict[str, str | int],
            dict[str, str | int],
            dict[str, str | int],
            dict[str, str | int]
        ]:
            # Base asterisk style (hidden when field is filled, red when empty)
            hidden_style = {'display': 'none'}
            required_style = {'color': '#d32f2f', 'fontSize': '24px', 'fontWeight': 'bold'}

            info: dict[str, Any] = dict(user_info or {})
            wants_contact = bool(
                info.get('consent_receive_results_later') or
                info.get('consent_keep_up_to_date')
            )
            
            # Check each required field and set asterisk visibility
            email_asterisk = hidden_style if (not wants_contact or email) else required_style
            age_asterisk = hidden_style if age else required_style
            gender_asterisk = hidden_style if gender else required_style
            diabetic_asterisk = hidden_style if is_diabetic is not None else required_style
            diabetic_type_asterisk = hidden_style if (not is_diabetic or diabetic_type) else required_style
            diabetes_duration_asterisk = hidden_style if (not is_diabetic or diabetes_duration is not None) else required_style
            location_asterisk = hidden_style if location else required_style
            
            # Check if all required fields are filled
            all_required_filled = (
                (email if wants_contact else True) and
                age and gender and is_diabetic is not None and location and
                (not is_diabetic or (diabetic_type and diabetes_duration is not None))
            )
            
            # Enable button only if all required fields are filled
            if all_required_filled:
                button_style = {
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
                return False, button_style, email_asterisk, age_asterisk, gender_asterisk, diabetic_asterisk, diabetic_type_asterisk, diabetes_duration_asterisk, location_asterisk
            else:
                button_style = {
                    'backgroundColor': '#cccccc',
                    'color': 'white',
                    'padding': '20px 30px',
                    'border': 'none',
                    'borderRadius': '5px',
                    'fontSize': '24px',
                    'cursor': 'not-allowed',
                    'width': '100%',
                    'height': '80px',
                    'display': 'flex',
                    'alignItems': 'center',
                    'justifyContent': 'center',
                    'lineHeight': '1.2'
                }
                return True, button_style, email_asterisk, age_asterisk, gender_asterisk, diabetic_asterisk, diabetic_type_asterisk, diabetes_duration_asterisk, location_asterisk 

        # <!-- START INSERTION: Test Me Button Callback -->
        # Callback for "Just Test Me" button
        # Note: diabetic-type-dropdown and diabetes-duration-input are handled
        # by their respective callback when diabetic-dropdown changes
        @app.callback(
            [Output('email-input', 'value'),
             Output('age-input', 'value'),
             Output('gender-dropdown', 'value'),
             Output('cgm-dropdown', 'value'),
             Output('diabetic-dropdown', 'value'),
             Output('location-input', 'value')],
            [Input('test-me-button', 'n_clicks')],
            prevent_initial_call=True
        )
        def fill_form_data(n_clicks: Optional[int]) -> tuple[str, int, str, bool, bool, str]:
            if n_clicks:
                # Fill the form with realistic test data and tick consent checkbox
                # Note: diabetic-type and diabetes-duration will be auto-filled by existing callbacks
                return (
                    'test.user@example.com',  # email
                    28,                       # age
                    'F',                      # gender (Female)
                    True,                     # uses_cgm
                    True,                     # is_diabetic (Yes) - this will trigger diabetic details callback
                    'San Francisco, CA'       # location
                )
            
            return no_update, no_update, no_update, no_update, no_update, no_update 

 