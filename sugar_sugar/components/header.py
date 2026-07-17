from typing import Sequence, Optional, Any

from dash import dcc, html
from dash.html import Div

from sugar_sugar.config import STORAGE_TYPE
from sugar_sugar.i18n import t


def make_csv_upload(
    locale: str, *, style: Optional[dict[str, Any]] = None, className: Optional[str] = None
) -> dcc.Upload:
    """Build the single CSV ``dcc.Upload`` (id='upload-data').

    Shared between the desktop/portrait header and the prediction action strip so
    the component (and its ``header-upload-prompt`` child, referenced by the
    language-change callback) exists in exactly one place per page.
    """
    return dcc.Upload(
        id='upload-data',
        className=className,
        children=html.Div(
            id='header-upload-prompt',
            children=t("ui.header.upload_button", locale=locale),
        ),
        style=style if style is not None else {
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'marginBottom': '10px',
        },
    )


class HeaderComponent(Div):
    def __init__(
        self,
        *,
        locale: str = "en",
        show_time_slider: bool = True,
        show_upload_section: bool = True,
        show_example_button: bool = True,
        show_data_source_section: bool = True,
        render_csv_upload: bool = True,
        children: Optional[Sequence[Any]] = None,
        initial_slider_value: int = 0,
        data_source_name: str = "example.csv",
        **kwargs: Any
    ) -> None:
        self._locale: str = locale
        self.show_time_slider = show_time_slider
        self.show_upload_section = show_upload_section
        self.show_example_button = show_example_button
        # When False the CSV dcc.Upload (id='upload-data') is NOT rendered in the
        # header -- the prediction page relocates it into the always-visible action
        # strip for formats B/C so it stays reachable in landscape. The single
        # dcc.Upload must live in exactly one place, so the header drops it here.
        self.render_csv_upload = render_csv_upload
        self.show_data_source_section = show_data_source_section
        self.initial_slider_value = initial_slider_value
        self.data_source_name = data_source_name
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
        """Create the header controls section.

        Note: the prediction window size is fixed, so we no longer expose a
        "number of points to show" control in the UI.
        """
        # Always include the time slider for callback wiring, but conditionally hide it.
        time_slider_div = html.Div(
            [
                html.Label(
                    t("ui.header.time_window_label", locale=self._locale),
                    id='header-time-window-label',
                    style={"marginRight": "10px"},
                ),
                dcc.Slider(
                    id="time-slider",
                    min=0,
                    max=100,  # Updated by callback
                    value=self.initial_slider_value,
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True},
                    updatemode="mouseup",
                    included=True,
                    step=1,
                    persistence=True,
                    persistence_type=STORAGE_TYPE,
                ),
            ],
            style={
                "flex": "1",
                "display": "block" if self.show_time_slider else "none",
            },
        )

        return html.Div([time_slider_div])

    def create_upload_section(self, *, visible: bool = True) -> html.Div:
        """Create the file upload and Nightscout data source section."""
        _input_style: dict[str, str] = {
            'width': '100%',
            'marginBottom': '8px',
            'padding': '8px 10px',
            'borderRadius': '4px',
            'border': '1px solid #ccc',
            'fontSize': '14px',
            'boxSizing': 'border-box',
        }

        csv_tab = dcc.Tab(
            label=t("ui.header.csv_tab_label", locale=self._locale),
            value="csv",
            children=html.Div(
                [make_csv_upload(self._locale)] if self.render_csv_upload else [],
                style={'paddingTop': '10px'},
            ),
        )

        nightscout_tab = dcc.Tab(
            label=t("ui.header.nightscout_tab_label", locale=self._locale),
            value="nightscout",
            children=html.Div([
                dcc.Input(
                    id="nightscout-url-input",
                    type="url",
                    placeholder=t("ui.header.nightscout_url_placeholder", locale=self._locale),
                    debounce=False,
                    style=_input_style,
                ),
                dcc.Input(
                    id="nightscout-token-input",
                    type="text",
                    placeholder=t("ui.header.nightscout_token_placeholder", locale=self._locale),
                    debounce=False,
                    style=_input_style,
                ),
                html.Button(
                    id="nightscout-load-button",
                    children=t("ui.header.nightscout_load_button", locale=self._locale),
                    className="ui blue button",
                    style={'width': '100%', 'marginBottom': '8px'},
                ),
                html.Div(id="nightscout-status"),
            ], style={'paddingTop': '10px'}),
        )

        children: list[Any] = [
            dcc.Tabs(
                id="data-input-tabs",
                value="csv",
                children=[csv_tab, nightscout_tab],
                style={'marginBottom': '4px'},
            ),
        ]

        # Always render the example button + warning so their ids stay in the DOM
        # (the prediction language-change callback outputs to them on every format);
        # hide them when they don't apply.
        _example_display = 'block' if self.show_example_button else 'none'
        children.append(
            html.Button(
                id='use-example-data-button',
                children=t("ui.header.use_example_data", locale=self._locale),
                style={
                    'width': '100%',
                    'height': '40px',
                    'backgroundColor': '#f8f9fa',
                    'border': '1px solid #dee2e6',
                    'borderRadius': '5px',
                    'cursor': 'pointer',
                    'fontSize': '14px',
                    'color': '#6c757d',
                    'display': _example_display,
                }
            )
        )
        children.append(html.Div(id='example-data-warning', style={'marginTop': '10px', 'display': _example_display}))

        return html.Div(
            children,
            style={'display': 'block' if visible else 'none'},
        )

    def _create_header_content(self) -> Sequence[Any]:
        """Create the header section content with title and description"""
        right_column_children: list[Any] = [
            self.create_controls(),
        ]

        right_column_children.append(self.create_upload_section(visible=self.show_upload_section))
        if self.show_data_source_section:
            right_column_children.append(self.create_data_source_section())

        return [
            html.H1(t("ui.common.app_title", locale=self._locale),
                    id='header-app-title',
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
                    html.P(id='header-description', children=[
                        t("ui.header.description_1", locale=self._locale),
                    ], style={
                        'fontSize': '18px',
                        'color': '#4a5568',
                        'lineHeight': '1.5',
                        'marginBottom': '15px'
                    }),
                    html.Div(id='header-how-to-play', children=[
                        html.Button(
                            t("ui.header.how_to_play", locale=self._locale),
                            id="header-how-to-play-toggle",
                            className="header-how-to-play-toggle",
                            type="button",
                        ),
                        html.Div(
                            [
                                html.Button("×", id="header-how-to-play-close", className="header-how-to-play-close", type="button"),
                                html.Div(
                                    [
                                        t("ui.header.description_2", locale=self._locale) + " ",
                                        t("ui.header.description_3", locale=self._locale),
                                        html.Br(),
                                        t("ui.header.how_to_play_1", locale=self._locale),
                                        html.Br(),
                                        t("ui.header.how_to_play_2", locale=self._locale),
                                        html.Br(),
                                        t("ui.header.how_to_play_3", locale=self._locale),
                                    ],
                                    className="header-how-to-play-body",
                                ),
                            ],
                            id="header-how-to-play-bubble",
                            className="header-how-to-play-bubble",
                            style={"display": "none"},
                        ),
                    ], style={
                        'fontSize': '16px',
                        'color': '#4a5568',
                        'lineHeight': '1.8'
                    })
                ], style={'flex': '1', 'paddingRight': '20px'}),

                # Right column - Upload and controls
                html.Div([
                    *right_column_children
                ], style={'flex': '1'})
            ], style={
                'display': 'flex',
                'flexDirection': 'row',
                'gap': '20px',
                'alignItems': 'start'
            })
        ]

    def create_data_source_section(self) -> html.Div:
        """Create the data source indicator used by prediction/startup headers."""
        return html.Div(
            [
                html.Div(
                    [
                        html.Label(
                            t("ui.header.current_data_source", locale=self._locale),
                            id='header-data-source-label',
                            style={
                                'fontWeight': 'bold',
                                'marginRight': '8px',
                                'color': '#4a5568',
                                'fontSize': '14px',
                            },
                        ),
                        html.Div(
                            id='data-source-display',
                            children=self.data_source_name,
                            style={
                                'color': '#2c5282',
                                'fontStyle': 'italic',
                                'fontSize': '14px',
                                'fontWeight': '500',
                                'overflow': 'hidden',
                                'textOverflow': 'ellipsis',
                                'maxWidth': '300px',
                                'whiteSpace': 'nowrap',
                            },
                        ),
                    ],
                    style={
                        'display': 'flex',
                        'alignItems': 'center',
                        'width': '100%',
                    },
                ),
                html.Div(
                    id='generic-source-metadata-display',
                    children="",
                    style={
                        'color': '#2c5282',
                        'fontStyle': 'italic',
                        'fontSize': '14px',
                        'fontWeight': '500',
                        'maxWidth': '100%',
                        'whiteSpace': 'normal',
                        'overflowWrap': 'anywhere',
                        'marginTop': '4px',
                    },
                ),
            ],
            style={
                'marginTop': '15px',
                'padding': '10px',
                'backgroundColor': '#f7fafc',
                'borderRadius': '5px',
                'display': 'flex',
                'flexDirection': 'column',
                'alignItems': 'flex-start',
                'width': '100%',
                'boxShadow': '0 1px 2px rgba(0,0,0,0.05)',
            },
        )