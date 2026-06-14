from dash import html, dcc
from dash.dependencies import Input, Output, State
from dash import no_update
import dash
from typing import Any, Optional
# DEBUG_MODE will be imported dynamically to get the latest value
from sugar_sugar.components.landing import consent_controls_children
from sugar_sugar.i18n import t
from sugar_sugar.config import STORAGE_TYPE



def _compute_format_options(
    uses_cgm: Optional[bool],
    interface_language: Optional[str],
    current_format: Optional[str],
) -> tuple[list[dict[str, Any]], Optional[str]]:
    """Return the dropdown options list and the desired selected value.

    Keeping the option ordering consistent (A, B, C) is important for the
dropdown scroller.  Formats B and C are disabled unless ``uses_cgm`` is True.
    The returned ``value`` is used to update the component's value according to
eligibility and previous selection.
    """
    allow_all = uses_cgm is True
    options: list[dict[str, Any]] = [
        {
            'label': t("ui.startup.format_a_label", locale=interface_language),
            'value': 'A',
        },
        {
            'label': t("ui.startup.format_b_label", locale=interface_language),
            'value': 'B',
            'disabled': not allow_all,
        },
        {
            'label': t("ui.startup.format_c_label", locale=interface_language),
            'value': 'C',
            'disabled': not allow_all,
        },
    ]

    if not current_format:
        return options, ('C' if allow_all else 'A')
    if allow_all and current_format == 'A':
        # Encourage option C once eligible.
        return options, 'C'
    if not allow_all and current_format in ('B', 'C'):
        return options, 'A'
    return options, current_format


class StartupPage(html.Div):
    def __init__(self, *, locale: str = "en") -> None:
        self.component_id: str = 'startup-page'
        self._locale: str = locale
        
        # Create the layout
        layout = [
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
                        html.Label(t("ui.startup.email_label", locale=locale), style={'fontSize': '22px', 'fontWeight': '800', 'marginBottom': '10px', 'color': '#0f172a', 'display': 'inline-block'}),
                        html.Span(id='email-required', children=' *', style={'color': '#d32f2f', 'fontSize': '22px', 'fontWeight': 'bold'})
                    ], style={'marginBottom': '10px'}),
                    dcc.Input(
                        id='email-input',
                        type='email',
                        placeholder=t("ui.startup.email_placeholder", locale=locale),
                        persistence=True,
                        persistence_type=STORAGE_TYPE,
                        style={'width': '100%', 'padding': '10px', 'fontSize': '20px', 'marginBottom': '20px'}
                    ),
                    
                    html.Div([
                        html.Label(t("ui.startup.age_label", locale=locale), style={'fontSize': '22px', 'fontWeight': '800', 'marginBottom': '10px', 'color': '#0f172a', 'display': 'inline-block'}),
                        html.Span(id='age-required', children=' *', style={'color': '#d32f2f', 'fontSize': '22px', 'fontWeight': 'bold'})
                    ], style={'marginBottom': '10px'}),
                    dcc.Input(
                        id='age-input',
                        type='number',
                        placeholder=t("ui.startup.age_placeholder", locale=locale),
                        min=0,
                        max=120,
                        persistence=True,
                        persistence_type=STORAGE_TYPE,
                        style={'width': '100%', 'padding': '10px', 'fontSize': '20px', 'marginBottom': '20px'}
                    ),
                    html.Div(
                        id='age-error',
                        children='',
                        style={'color': '#d32f2f', 'fontSize': '16px', 'marginTop': '-12px', 'marginBottom': '20px'}
                    ),
                    
                    html.Div([
                        html.Label(t("ui.startup.gender_label", locale=locale), style={'fontSize': '22px', 'fontWeight': '800', 'marginBottom': '10px', 'color': '#0f172a', 'display': 'inline-block'}),
                        html.Span(id='gender-required', children=' *', style={'color': '#d32f2f', 'fontSize': '22px', 'fontWeight': 'bold'})
                    ], style={'marginBottom': '10px'}),
                    dcc.Dropdown(
                        id='gender-dropdown',
                        options=[
                            {'label': t("ui.startup.gender_male", locale=locale), 'value': 'M'},
                            {'label': t("ui.startup.gender_female", locale=locale), 'value': 'F'},
                            {'label': t("ui.startup.gender_na", locale=locale), 'value': 'N/A'}
                        ],
                        placeholder=t("ui.startup.gender_placeholder", locale=locale),
                        persistence=True,
                        persistence_type=STORAGE_TYPE,
                        style={'fontSize': '20px', 'marginBottom': '20px'}
                    ),

                    html.Label(t("ui.startup.cgm_label", locale=locale), style={'fontSize': '22px', 'fontWeight': '800', 'marginBottom': '10px', 'color': '#0f172a'}),
                    dcc.Dropdown(
                        id='cgm-dropdown',
                        options=[
                            {'label': t("ui.startup.yes", locale=locale), 'value': True},
                            {'label': t("ui.startup.no", locale=locale), 'value': False}
                        ],
                        placeholder=t("ui.startup.cgm_placeholder", locale=locale),
                        persistence=True,
                        persistence_type=STORAGE_TYPE,
                        style={'fontSize': '20px', 'marginBottom': '20px'}
                    ),

                    html.Div(id='cgm-details', children=[
                        html.Label(t("ui.startup.cgm_duration_label", locale=locale), style={'fontSize': '22px', 'fontWeight': '800', 'marginBottom': '10px', 'color': '#0f172a'}),
                        dcc.Input(
                            id='cgm-duration-input',
                            type='number',
                            placeholder=t("ui.startup.cgm_duration_placeholder", locale=locale),
                            min=0,
                            max=100,
                            persistence=True,
                            persistence_type=STORAGE_TYPE,
                            style={'width': '100%', 'padding': '10px', 'fontSize': '20px', 'marginBottom': '20px'}
                        )
                    ]),

                    html.Div([
                        html.Div([
                            html.Label(t("ui.startup.format_label", locale=locale), style={'fontSize': '22px', 'fontWeight': '800', 'marginBottom': '10px', 'color': '#0f172a', 'display': 'inline-block'}),
                            html.Span(id='format-required', children=' *', style={'color': '#d32f2f', 'fontSize': '22px', 'fontWeight': 'bold'})
                        ], style={'marginBottom': '10px'}),
                        dcc.Dropdown(
                            id='format-dropdown',
                            options=[
                                {'label': t("ui.startup.format_a_label", locale=locale), 'value': 'A'},
                                {'label': t("ui.startup.format_b_label", locale=locale), 'value': 'B', 'disabled': True},
                                {'label': t("ui.startup.format_c_label", locale=locale), 'value': 'C', 'disabled': True},
                            ],
                            placeholder=t("ui.startup.format_placeholder", locale=locale),
                            persistence=True,
                            persistence_type=STORAGE_TYPE,
                            style={'fontSize': '20px', 'marginBottom': '10px'}
                        ),
                        html.Div(
                            [
                                html.Small(t("ui.startup.format_help_a", locale=locale)),
                                html.Br(),
                                html.Small(t("ui.startup.format_help_b", locale=locale)),
                                html.Br(),
                                html.Small(t("ui.startup.format_help_c", locale=locale)),
                            ],
                            style={'color': '#666', 'fontSize': '14px', 'marginBottom': '20px', 'lineHeight': '1.4'}
                        ),
                        html.Div(
                            id='data-usage-consent-container',
                            children=[
                                dcc.Checklist(
                                    id='data-usage-consent',
                                    options=[{'label': t("ui.startup.data_usage_consent_label", locale=locale), 'value': 'agree'}],
                                    value=[],
                                    persistence=True,
                                    persistence_type=STORAGE_TYPE,
                                    style={'fontSize': '16px'}
                                ),
                                html.Div(id='data-usage-error', style={'marginTop': '8px', 'color': '#d32f2f', 'fontSize': '16px'})
                            ],
                            style={'display': 'none', 'marginBottom': '20px'}
                        ),
                    ], style={'marginBottom': '10px'}),
                    
                    html.Div([
                        html.Label(t("ui.startup.diabetic_label", locale=locale), style={'fontSize': '22px', 'fontWeight': '800', 'marginBottom': '10px', 'color': '#0f172a', 'display': 'inline-block'}),
                        html.Span(id='diabetic-required', children=' *', style={'color': '#d32f2f', 'fontSize': '22px', 'fontWeight': 'bold'})
                    ], style={'marginBottom': '10px'}),
                    dcc.Dropdown(
                        id='diabetic-dropdown',
                        options=[
                            {'label': t("ui.startup.yes", locale=locale), 'value': True},
                            {'label': t("ui.startup.no", locale=locale), 'value': False}
                        ],
                        placeholder=t("ui.startup.diabetic_placeholder", locale=locale),
                        persistence=True,
                        persistence_type=STORAGE_TYPE,
                        style={'fontSize': '20px', 'marginBottom': '20px'}
                    ),
                    
                    html.Div(id='diabetic-details', children=[
                        html.Div([
                            html.Label(t("ui.startup.diabetes_type_label", locale=locale), style={'fontSize': '22px', 'fontWeight': '800', 'marginBottom': '10px', 'color': '#0f172a', 'display': 'inline-block'}),
                            html.Span(id='diabetic-type-required', children=' *', style={'color': '#d32f2f', 'fontSize': '22px', 'fontWeight': 'bold'})
                        ], style={'marginBottom': '10px'}),
                        dcc.Dropdown(
                            id='diabetic-type-dropdown',
                            options=[
                                {'label': t("ui.startup.diabetes_type_1", locale=locale), 'value': 'Type 1'},
                                {'label': t("ui.startup.diabetes_type_2", locale=locale), 'value': 'Type 2'},
                                {'label': t("ui.startup.diabetes_type_gestational", locale=locale), 'value': 'Gestational'},
                                {'label': t("ui.startup.diabetes_type_lada", locale=locale), 'value': 'LADA'},
                                {'label': t("ui.startup.gender_na", locale=locale), 'value': 'N/A'}
                            ],
                            placeholder=t("ui.startup.diabetes_type_placeholder", locale=locale),
                            persistence=True,
                            persistence_type=STORAGE_TYPE,
                            style={'fontSize': '20px', 'marginBottom': '20px'}
                        ),
                        
                        html.Div([
                            html.Label(t("ui.startup.diabetes_duration_label", locale=locale), style={'fontSize': '22px', 'fontWeight': '800', 'marginBottom': '10px', 'color': '#0f172a', 'display': 'inline-block'}),
                            html.Span(id='diabetes-duration-required', children=' *', style={'color': '#d32f2f', 'fontSize': '22px', 'fontWeight': 'bold'})
                        ], style={'marginBottom': '10px'}),
                        dcc.Input(
                            id='diabetes-duration-input',
                            type='number',
                            placeholder=t("ui.startup.diabetes_duration_placeholder", locale=locale),
                            min=0,
                            max=100,
                            persistence=True,
                            persistence_type=STORAGE_TYPE,
                            style={'width': '100%', 'padding': '10px', 'fontSize': '20px', 'marginBottom': '20px'}
                        )
                    ]),
                    
                    html.Div([
                        html.Label(t("ui.startup.location_label", locale=locale), style={'fontSize': '22px', 'fontWeight': '800', 'marginBottom': '10px', 'color': '#0f172a', 'display': 'inline-block'}),
                        html.Span(id='location-required', children=' *', style={'color': '#d32f2f', 'fontSize': '22px', 'fontWeight': 'bold'})
                    ], style={'marginBottom': '10px'}),
                    dcc.Input(
                        id='location-input',
                        type='text',
                        placeholder=t("ui.startup.location_placeholder", locale=locale),
                        persistence=True,
                        persistence_type=STORAGE_TYPE,
                        style={'width': '100%', 'padding': '10px', 'fontSize': '20px', 'marginBottom': '20px'}
                    ),
                    
                    html.Div(
                        [
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
                            className="ui blue-action button",
                            style={
                                'backgroundColor': '#1976D2',
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
                            className="ui green button",
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
            [Output('format-dropdown', 'options'),
             Output('format-dropdown', 'value')],
            [Input('cgm-dropdown', 'value'),
             Input('interface-language', 'data')],
            [State('format-dropdown', 'value')]
        )
        def update_format_options(
            uses_cgm: Optional[bool],
            interface_language: Optional[str],
            current_format: Optional[str],
        ) -> tuple[list[dict[str, Any]], Optional[str]]:
            # delegate to helper so we can unit-test behaviour independently
            return _compute_format_options(uses_cgm, interface_language, current_format)

        @app.callback(
            [Output('data-usage-consent-container', 'style'),
             Output('data-usage-consent', 'value')],
            [Input('format-dropdown', 'value')],
            [State('data-usage-consent', 'value')],
        )
        def toggle_data_usage_consent(
            format_value: Optional[str],
            current_value: Optional[list[str]],
        ) -> tuple[dict[str, str], list[str]]:
            if format_value in ('B', 'C'):
                return {'display': 'block', 'marginBottom': '20px'}, list(current_value or [])
            return {'display': 'none', 'marginBottom': '20px'}, []

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
        ) -> tuple[dict[str, str], Any, Any]:
            if is_diabetic is None:
                return {'display': 'none'}, dash.no_update, dash.no_update
            elif is_diabetic:
                # Check if this is from the test button (email will be test email)
                if test_clicks and email and 'test.user@example.com' in str(email):
                    return {'display': 'block'}, 'Type 1', 5
                else:
                    return {'display': 'block'}, dash.no_update, dash.no_update
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
        ) -> tuple[dict[str, str], Any]:
            if uses_cgm is True:
                if test_clicks and email and 'test.user@example.com' in str(email):
                    return {'display': 'block'}, 3
                return {'display': 'block'}, dash.no_update
            return {'display': 'none'}, dash.no_update

        @app.callback(
            [Output('start-button', 'disabled'),
             Output('start-button', 'style'),
             Output('email-required', 'style'),
             Output('age-required', 'style'),
             Output('gender-required', 'style'),
             Output('diabetic-required', 'style'),
             Output('diabetic-type-required', 'style'),
             Output('diabetes-duration-required', 'style'),
             Output('location-required', 'style'),
             Output('format-required', 'style'),
             Output('age-error', 'children'),
             Output('data-usage-error', 'children')],
            [Input('email-input', 'value'),
             Input('age-input', 'value'),
             Input('gender-dropdown', 'value'),
             Input('format-dropdown', 'value'),
             Input('data-usage-consent', 'value'),
             Input('diabetic-dropdown', 'value'),
             Input('diabetic-type-dropdown', 'value'),
             Input('diabetes-duration-input', 'value'),
             Input('location-input', 'value'),
             Input('user-info-store', 'data'),
             Input('consent-receive-results', 'value'),
             Input('consent-keep-updated', 'value'),
             Input('interface-language', 'data')]
        )
        def update_form_validation(
            email: Optional[str], 
            age: Optional[int | float], 
            gender: Optional[str], 
            format_value: Optional[str],
            data_usage_consent: Optional[list[str]],
            is_diabetic: Optional[bool], 
            diabetic_type: Optional[str], 
            diabetes_duration: Optional[int | float], 
            location: Optional[str],
            user_info: Optional[dict[str, Any]],
            consent_receive_results: Optional[list[str]],
            consent_keep_updated: Optional[list[str]],
            interface_language: Optional[str],
        ) -> tuple[
            bool,
            dict[str, str | int],
            dict[str, str | int],
            dict[str, str | int],
            dict[str, str | int],
            dict[str, str | int],
            dict[str, str | int],
            dict[str, str | int],
            dict[str, str | int],
            dict[str, str | int],
            str,
            str
        ]:
            # Base asterisk style (hidden when field is filled, red when empty)
            hidden_style = {'display': 'none'}
            required_style = {'color': '#d32f2f', 'fontSize': '24px', 'fontWeight': 'bold'}

            info: dict[str, Any] = dict(user_info or {})
            wants_contact = bool(
                info.get('consent_receive_results_later') or
                info.get('consent_keep_up_to_date') or
                (consent_receive_results and 'receive_results' in consent_receive_results) or
                (consent_keep_updated and 'keep_updated' in consent_keep_updated)
            )
            
            # Check each required field and set asterisk visibility
            email_asterisk = hidden_style if (not wants_contact or email) else required_style
            age_asterisk = hidden_style if age else required_style
            gender_asterisk = hidden_style if gender else required_style
            format_asterisk = hidden_style if format_value else required_style
            diabetic_asterisk = hidden_style if is_diabetic is not None else required_style
            diabetic_type_asterisk = hidden_style if (not is_diabetic or diabetic_type) else required_style
            diabetes_duration_asterisk = hidden_style if (not is_diabetic or diabetes_duration is not None) else required_style
            location_asterisk = hidden_style if location else required_style

            is_adult = (age is not None) and (float(age) >= 18)
            age_error = t("ui.startup.age_must_be_18_error", locale=interface_language) if (age is not None and not is_adult) else ""

            needs_data_consent = format_value in ("B", "C")
            has_data_consent = bool(data_usage_consent and "agree" in data_usage_consent)
            data_usage_error = (
                t("ui.startup.data_usage_consent_required", locale=interface_language)
                if (needs_data_consent and not has_data_consent)
                else ""
            )
            
            # Check if all required fields are filled
            all_required_filled = (
                (email if wants_contact else True) and
                age and is_adult and gender and format_value and is_diabetic is not None and location and
                (not needs_data_consent or has_data_consent) and
                (not is_diabetic or (diabetic_type and diabetes_duration is not None))
            )
            
            # Enable button only if all required fields are filled
            if all_required_filled:
                button_style = {
                    'backgroundColor': '#4CBB17',
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
                return (
                    False,
                    button_style,
                    email_asterisk,
                    age_asterisk,
                    gender_asterisk,
                    diabetic_asterisk,
                    diabetic_type_asterisk,
                    diabetes_duration_asterisk,
                    location_asterisk,
                    format_asterisk,
                    age_error,
                    data_usage_error,
                )
            else:
                button_style = {
                    'backgroundColor': '#555555',
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
                return (
                    True,
                    button_style,
                    email_asterisk,
                    age_asterisk,
                    gender_asterisk,
                    diabetic_asterisk,
                    diabetic_type_asterisk,
                    diabetes_duration_asterisk,
                    location_asterisk,
                    format_asterisk,
                    age_error,
                    data_usage_error,
                )

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

        # ---- Mobile startup wizard step navigation (StartupPageMobile) ----
        # The `mobile-step-*`, `startup-prev`, `startup-next`, `startup-progress`
        # ids exist ONLY in the mobile builder.  On desktop these buttons are
        # absent, so with prevent_initial_call=True this callback never fires
        # there.  Initial step visibility is baked into the mobile layout, so no
        # initial call is needed.  This callback only toggles the `mobile-step-*`
        # wrappers -- it never touches `cgm-details` / `diabetic-details` /
        # `data-usage-consent-container`, which keep their own conditional
        # callbacks (disjoint Outputs, no races).
        @app.callback(
            [Output('startup-step', 'data'),
             *[Output(f'mobile-step-{i}', 'style') for i in range(WIZARD_STEPS)],
             Output('startup-prev', 'style'),
             Output('startup-next', 'style'),
             Output('startup-progress', 'children')],
            [Input('startup-prev', 'n_clicks'),
             Input('startup-next', 'n_clicks')],
            [State('startup-step', 'data'),
             State('interface-language', 'data')],
            prevent_initial_call=True,
        )
        def navigate_startup_wizard(
            prev_clicks: Optional[int],
            next_clicks: Optional[int],
            current_step: Optional[int],
            interface_language: Optional[str],
        ) -> tuple[Any, ...]:
            trigger = dash.callback_context.triggered_id
            step = int(current_step or 0)
            if trigger == 'startup-next':
                step = min(step + 1, WIZARD_STEPS - 1)
            elif trigger == 'startup-prev':
                step = max(step - 1, 0)
            step_styles = [
                ({'display': 'block'} if i == step else {'display': 'none'})
                for i in range(WIZARD_STEPS)
            ]
            return (
                step,
                *step_styles,
                _wizard_nav_btn_style(visible=step > 0),
                _wizard_nav_btn_style(visible=step < WIZARD_STEPS - 1),
                _wizard_progress_children(step, interface_language),
            )

        # The consent gate must run on initial mobile render so the first Next
        # button starts disabled until the required consent actions are complete.
        @app.callback(
            Output('startup-next', 'disabled'),
            [Input('consent-scroll-complete', 'data'),
             Input('consent-acknowledge', 'value'),
             Input('consent-gdpr', 'value'),
             Input('startup-step', 'data')],
            prevent_initial_call=False,
        )
        def gate_mobile_consent_step(
            scroll_complete: Optional[bool],
            acknowledge_value: Optional[list[str]],
            gdpr_value: Optional[list[str]],
            current_step: Optional[int],
        ) -> bool:
            step = int(current_step or 0)
            if step != 0:
                return False
            return not (
                bool(scroll_complete) and
                bool(acknowledge_value and 'ack' in acknowledge_value) and
                bool(gdpr_value and 'gdpr' in gdpr_value)
            )


# ---------------------------------------------------------------------------
# Mobile startup wizard (StartupPageMobile)
# ---------------------------------------------------------------------------
# Number of wizard steps.  Must match the number of `mobile-step-{i}` wrappers
# the builder renders and the Outputs in `navigate_startup_wizard`.
WIZARD_STEPS: int = 6

# Mobile field styling: big tap targets, 16px+ to avoid iOS zoom-on-focus.
_M_LABEL = {'fontSize': '18px', 'fontWeight': '800', 'marginBottom': '8px', 'color': '#0f172a', 'display': 'inline-block'}
_M_REQ = {'color': '#d32f2f', 'fontSize': '18px', 'fontWeight': 'bold'}
_M_INPUT = {'width': '100%', 'padding': '14px', 'fontSize': '17px', 'marginBottom': '6px', 'boxSizing': 'border-box'}
_M_DROPDOWN = {'fontSize': '17px', 'marginBottom': '6px'}
_M_ERROR = {'color': '#d32f2f', 'fontSize': '15px', 'marginTop': '2px', 'marginBottom': '10px', 'minHeight': '18px'}


def _wizard_nav_btn_style(*, visible: bool) -> dict[str, str]:
    """Style for a wizard Back/Next button; hidden via visibility to keep layout."""
    return {
        'flex': '1',
        'padding': '16px',
        'fontSize': '18px',
        'fontWeight': '700',
        'borderRadius': '10px',
        'border': 'none',
        'cursor': 'pointer',
        'visibility': 'visible' if visible else 'hidden',
    }


def _wizard_progress_children(step: int, locale: Optional[str]) -> html.Div:
    """Progress indicator: a row of dots plus 'Step X of N' text."""
    dots = [
        html.Span(
            style={
                'display': 'inline-block',
                'width': '10px',
                'height': '10px',
                'borderRadius': '50%',
                'margin': '0 4px',
                'backgroundColor': '#2c5282' if i <= step else '#cbd5e1',
            }
        )
        for i in range(WIZARD_STEPS)
    ]
    return html.Div(
        [
            html.Div(dots, style={'textAlign': 'center', 'marginBottom': '6px'}, disable_n_clicks=True),
            html.Div(
                t("ui.startup.wizard_step", locale=locale, current=step + 1, total=WIZARD_STEPS),
                style={'textAlign': 'center', 'color': '#64748b', 'fontSize': '14px'},
                disable_n_clicks=True,
            ),
        ],
        disable_n_clicks=True,
    )


def _m_label(text: str, required_id: Optional[str] = None) -> html.Div:
    """A field label, optionally with a managed required-asterisk span."""
    children: list[Any] = [html.Label(text, style=_M_LABEL)]
    if required_id:
        children.append(html.Span(id=required_id, children=' *', style=_M_REQ))
    return html.Div(children, style={'marginBottom': '8px'}, disable_n_clicks=True)


class StartupPageMobile(html.Div):
    """Portrait-first multi-step wizard for the startup form.

    Renders EVERY input id of the desktop ``StartupPage`` (same ids, same
    persistence) so the existing validation / conditional callbacks in
    ``StartupPage.register_callbacks`` drive it unchanged -- and so no callback
    ever targets a missing component (the "nonexistent object" crash class).
    The fields are grouped into ``mobile-step-{i}`` wrappers shown one at a time;
    ``navigate_startup_wizard`` toggles their ``display``.  Each conditional
    parent lives in the SAME step as its dependents (CGM->duration,
    diabetic->type+duration, format B/C->data-usage-consent) so a hidden step
    never strands a half-revealed cascade.

    Only the first step is visible initially (baked into the layout); the step
    store is memory-backed and resets to 0 on load.  No callbacks are registered
    here.
    """

    def __init__(self, *, locale: str = "en") -> None:
        self.component_id = 'startup-page'
        self._locale = locale

        # --- Step 0: consent (mandatory gate before form fields) ---
        step_consent = [
            html.H2(
                t("ui.landing.patient_consent_form_title", locale=locale),
                style={'fontSize': '22px', 'fontWeight': '800', 'color': '#2c5282', 'marginBottom': '12px'},
                disable_n_clicks=True,
            ),
            html.Div(
                consent_controls_children(locale),
                id='consent-notice-scroll',
                disable_n_clicks=True,
            ),
            html.Div(
                t("ui.landing.next_hint", locale=locale),
                style={'color': '#64748b', 'marginTop': '10px', 'fontSize': '13px'},
                disable_n_clicks=True,
            ),
            dcc.Store(id='consent-scroll-complete', data=False, storage_type=STORAGE_TYPE),
            dcc.Interval(id='consent-scroll-poll', interval=500, n_intervals=0),
        ]

        # --- Step 1: identity (pure required fields, no conditional cascade) ---
        step0 = [
            _m_label(t("ui.startup.email_label", locale=locale), 'email-required'),
            dcc.Input(
                id='email-input', type='email',
                placeholder=t("ui.startup.email_placeholder", locale=locale),
                persistence=True, persistence_type=STORAGE_TYPE, style=_M_INPUT,
            ),
            _m_label(t("ui.startup.age_label", locale=locale), 'age-required'),
            dcc.Input(
                id='age-input', type='number',
                placeholder=t("ui.startup.age_placeholder", locale=locale),
                min=0, max=120, persistence=True, persistence_type=STORAGE_TYPE, style=_M_INPUT,
            ),
            html.Div(id='age-error', children='', style=_M_ERROR, disable_n_clicks=True),
            _m_label(t("ui.startup.gender_label", locale=locale), 'gender-required'),
            dcc.Dropdown(
                id='gender-dropdown',
                options=[
                    {'label': t("ui.startup.gender_male", locale=locale), 'value': 'M'},
                    {'label': t("ui.startup.gender_female", locale=locale), 'value': 'F'},
                    {'label': t("ui.startup.gender_na", locale=locale), 'value': 'N/A'},
                ],
                placeholder=t("ui.startup.gender_placeholder", locale=locale),
                persistence=True, persistence_type=STORAGE_TYPE, style=_M_DROPDOWN,
            ),
            html.Div(style={'height': '12px'}, disable_n_clicks=True),
            _m_label(t("ui.startup.location_label", locale=locale), 'location-required'),
            dcc.Input(
                id='location-input', type='text',
                placeholder=t("ui.startup.location_placeholder", locale=locale),
                persistence=True, persistence_type=STORAGE_TYPE, style=_M_INPUT,
            ),
        ]

        # --- Step 2: CGM (cgm-dropdown -> cgm-details/duration) ---
        step1 = [
            _m_label(t("ui.startup.cgm_label", locale=locale)),
            dcc.Dropdown(
                id='cgm-dropdown',
                options=[
                    {'label': t("ui.startup.yes", locale=locale), 'value': True},
                    {'label': t("ui.startup.no", locale=locale), 'value': False},
                ],
                placeholder=t("ui.startup.cgm_placeholder", locale=locale),
                persistence=True, persistence_type=STORAGE_TYPE, style=_M_DROPDOWN,
            ),
            html.Div(id='cgm-details', children=[
                html.Div(style={'height': '12px'}, disable_n_clicks=True),
                _m_label(t("ui.startup.cgm_duration_label", locale=locale)),
                dcc.Input(
                    id='cgm-duration-input', type='number',
                    placeholder=t("ui.startup.cgm_duration_placeholder", locale=locale),
                    min=0, max=100, persistence=True, persistence_type=STORAGE_TYPE, style=_M_INPUT,
                ),
            ]),
        ]

        # --- Step 3: diabetes (diabetic-dropdown -> type + duration) ---
        step2 = [
            _m_label(t("ui.startup.diabetic_label", locale=locale), 'diabetic-required'),
            dcc.Dropdown(
                id='diabetic-dropdown',
                options=[
                    {'label': t("ui.startup.yes", locale=locale), 'value': True},
                    {'label': t("ui.startup.no", locale=locale), 'value': False},
                ],
                placeholder=t("ui.startup.diabetic_placeholder", locale=locale),
                persistence=True, persistence_type=STORAGE_TYPE, style=_M_DROPDOWN,
            ),
            html.Div(id='diabetic-details', children=[
                html.Div(style={'height': '12px'}, disable_n_clicks=True),
                _m_label(t("ui.startup.diabetes_type_label", locale=locale), 'diabetic-type-required'),
                dcc.Dropdown(
                    id='diabetic-type-dropdown',
                    options=[
                        {'label': t("ui.startup.diabetes_type_1", locale=locale), 'value': 'Type 1'},
                        {'label': t("ui.startup.diabetes_type_2", locale=locale), 'value': 'Type 2'},
                        {'label': t("ui.startup.diabetes_type_gestational", locale=locale), 'value': 'Gestational'},
                        {'label': t("ui.startup.diabetes_type_lada", locale=locale), 'value': 'LADA'},
                        {'label': t("ui.startup.gender_na", locale=locale), 'value': 'N/A'},
                    ],
                    placeholder=t("ui.startup.diabetes_type_placeholder", locale=locale),
                    persistence=True, persistence_type=STORAGE_TYPE, style=_M_DROPDOWN,
                ),
                html.Div(style={'height': '12px'}, disable_n_clicks=True),
                _m_label(t("ui.startup.diabetes_duration_label", locale=locale), 'diabetes-duration-required'),
                dcc.Input(
                    id='diabetes-duration-input', type='number',
                    placeholder=t("ui.startup.diabetes_duration_placeholder", locale=locale),
                    min=0, max=100, persistence=True, persistence_type=STORAGE_TYPE, style=_M_INPUT,
                ),
            ]),
        ]

        # --- Step 4: format & data-usage consent ---
        step3 = [
            _m_label(t("ui.startup.format_label", locale=locale), 'format-required'),
            dcc.Dropdown(
                id='format-dropdown',
                options=[
                    {'label': t("ui.startup.format_a_label", locale=locale), 'value': 'A'},
                    {'label': t("ui.startup.format_b_label", locale=locale), 'value': 'B', 'disabled': True},
                    {'label': t("ui.startup.format_c_label", locale=locale), 'value': 'C', 'disabled': True},
                ],
                placeholder=t("ui.startup.format_placeholder", locale=locale),
                persistence=True, persistence_type=STORAGE_TYPE, style=_M_DROPDOWN,
            ),
            html.Div(
                [
                    html.Small(t("ui.startup.format_help_a", locale=locale)), html.Br(),
                    html.Small(t("ui.startup.format_help_b", locale=locale)), html.Br(),
                    html.Small(t("ui.startup.format_help_c", locale=locale)),
                ],
                style={'color': '#666', 'fontSize': '14px', 'margin': '8px 0 16px', 'lineHeight': '1.4'},
                disable_n_clicks=True,
            ),
            html.Div(
                id='data-usage-consent-container',
                children=[
                    dcc.Checklist(
                        id='data-usage-consent',
                        options=[{'label': t("ui.startup.data_usage_consent_label", locale=locale), 'value': 'agree'}],
                        value=[], persistence=True, persistence_type=STORAGE_TYPE, style={'fontSize': '16px'},
                    ),
                    html.Div(id='data-usage-error', style={'marginTop': '8px', 'color': '#d32f2f', 'fontSize': '15px'}, disable_n_clicks=True),
                ],
                style={'display': 'none', 'marginBottom': '20px'},
            ),
        ]

        # --- Step 5: contact prefs + submit (start-button driven by validation) ---
        step4 = [
            html.Div(
                [
                    html.H3(t("ui.startup.contact_prefs_title", locale=locale), style={'fontSize': '20px', 'marginBottom': '10px', 'color': '#2c5282'}),
                    html.P(t("ui.startup.contact_prefs_text", locale=locale), style={'fontSize': '15px', 'lineHeight': '1.6', 'margin': '0', 'color': '#555'}),
                ],
                style={'backgroundColor': '#f8f9fa', 'padding': '16px', 'borderRadius': '8px', 'marginBottom': '20px'},
                disable_n_clicks=True,
            ),
            html.Button(
                t("ui.startup.just_test_me", locale=locale),
                id='test-me-button',
                className="ui blue-action button",
                style={
                    'backgroundColor': '#1976D2', 'color': 'white', 'padding': '14px',
                    'border': 'none', 'borderRadius': '8px', 'fontSize': '16px',
                    'cursor': 'pointer', 'width': '100%', 'marginBottom': '14px',
                    'display': 'block' if self._get_debug_mode() else 'none',
                },
            ),
            html.Button(
                t("ui.startup.start_prediction", locale=locale),
                id='start-button',
                className="ui green button",
                disabled=True,
                style={
                    'backgroundColor': '#cccccc', 'color': 'white', 'padding': '18px',
                    'border': 'none', 'borderRadius': '8px', 'fontSize': '20px',
                    'cursor': 'not-allowed', 'width': '100%',
                },
            ),
        ]

        steps = [step_consent, step0, step1, step2, step3, step4]
        step_divs = [
            html.Div(
                children=content,
                id=f'mobile-step-{i}',
                style={'display': 'block' if i == 0 else 'none'},
                disable_n_clicks=True,
            )
            for i, content in enumerate(steps)
        ]

        nav = html.Div(
            [
                html.Button(
                    t("ui.startup.wizard_back", locale=locale),
                    id='startup-prev',
                    className="ui button",
                    style=_wizard_nav_btn_style(visible=False),
                ),
                html.Button(
                    t("ui.startup.wizard_next", locale=locale),
                    id='startup-next',
                    className="ui blue button",
                    disabled=True,
                    style=_wizard_nav_btn_style(visible=True),
                ),
            ],
            style={'display': 'flex', 'gap': '12px', 'marginTop': '20px'},
            disable_n_clicks=True,
        )

        card = html.Div(
            [
                html.H1(
                    t("ui.common.app_title", locale=locale),
                    style={'textAlign': 'center', 'marginBottom': '14px', 'fontSize': '28px', 'fontWeight': 'bold', 'color': '#2c5282'},
                    disable_n_clicks=True,
                ),
                html.Div(id='startup-progress', children=_wizard_progress_children(0, locale), disable_n_clicks=True),
                html.Div(
                    t("ui.startup.required_fields_note", locale=locale),
                    style={'color': '#666', 'fontSize': '13px', 'fontStyle': 'italic', 'margin': '12px 0', 'textAlign': 'right'},
                    disable_n_clicks=True,
                ),
                *step_divs,
                nav,
            ],
            style={
                'backgroundColor': 'white', 'borderRadius': '12px',
                'boxShadow': '0 0 10px rgba(0,0,0,0.1)', 'padding': '18px 16px',
            },
            disable_n_clicks=True,
        )

        super().__init__(
            children=[card],
            id=self.component_id,
            style={'padding': '14px 12px 32px', 'backgroundColor': '#f5f5f5', 'minHeight': '100vh'},
            disable_n_clicks=True,
        )

    def _get_debug_mode(self) -> bool:
        try:
            from sugar_sugar.config import DEBUG_MODE
            return DEBUG_MODE
        except ImportError:
            return False

 