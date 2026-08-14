import base64
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

import plotly.graph_objs as go
import polars as pl
from dash import ALL, dcc, Output, Input, State
from dash import Dash, html
from dash.exceptions import PreventUpdate
from eliot import start_action

from sugar_sugar.cgmacros import cgmacros_photo_url, visible_food_photo_events
from sugar_sugar.config import PREDICTION_HOUR_OFFSET, STORAGE_TYPE
from sugar_sugar.i18n import normalize_locale, t

_FOOD_LINE_COLOR: str = "#2e7d32"
_APPLE_ICON_SRC: str = "/assets/images/apple.svg"


def event_x_index(window_df: pl.DataFrame, event_time: datetime) -> float:
    """Map an event timestamp onto the chart's integer x index."""
    df_times = window_df.get_column("time")
    before_idx: Optional[int] = None
    after_idx: Optional[int] = None
    for i, time_val in enumerate(df_times):
        if time_val <= event_time:
            before_idx = i
        if time_val >= event_time and after_idx is None:
            after_idx = i
    if before_idx is None:
        before_idx = 0
    if after_idx is None:
        after_idx = len(df_times) - 1
    if df_times[before_idx] == event_time or before_idx == after_idx:
        return float(before_idx)
    before_time = df_times[before_idx].timestamp()
    after_time = df_times[after_idx].timestamp()
    factor = (event_time.timestamp() - before_time) / (after_time - before_time)
    return float(before_idx) + factor


def meal_food_bubble_children(
    window_df: pl.DataFrame,
    events_df: pl.DataFrame,
    *,
    source_name: str,
    hide_last_hour: bool,
) -> list[html.Button]:
    """HTML speech bubbles above the plot, one per visible meal photo."""
    if window_df.height == 0:
        return []
    meals = visible_food_photo_events(
        window_df, events_df, hide_last_hour=hide_last_hour
    )
    n_points = float(len(window_df))
    buttons: list[html.Button] = []
    for meal in meals:
        event_time = meal["time"]
        photo_path = str(meal.get("photo_path") or "")
        if not isinstance(event_time, datetime) or not photo_path:
            continue
        x_pos = event_x_index(window_df, event_time)
        left_pct = 100.0 * (x_pos + 0.5) / n_points
        photo_url = cgmacros_photo_url(source_name, photo_path)
        buttons.append(
            html.Button(
                html.Img(
                    src=_APPLE_ICON_SRC,
                    className="meal-food-bubble-apple",
                    alt="",
                    disable_n_clicks=True,
                ),
                id={"type": "meal-food-bubble", "index": photo_url},
                className="meal-food-speech-bubble",
                type="button",
                n_clicks=0,
                style={"left": f"{left_pct:.3f}%"},
            )
        )
    return buttons

_ASSETS_IMAGES = Path(__file__).resolve().parents[2] / "assets" / "images"

# Insulin / carbs: SVG layout images (plotly.js ignores custom path:// markers).
_ICON_EVENT_TYPES: frozenset[str] = frozenset({"Insulin", "Carbohydrates"})


@lru_cache(maxsize=4)
def _svg_data_uri(filename: str) -> str:
    path = _ASSETS_IMAGES / filename
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


class GlucoseChart(html.Div):
    RANGE_COLORS = {
        "dangerous_low": {"fill": "rgba(255, 200, 200, 0.5)", "line": "rgba(200, 0, 0, 0.5)"},
        "normal": {"fill": "rgba(200, 240, 200, 0.5)", "line": "rgba(0, 100, 0, 0.5)"},
        "high": {"fill": "rgba(255, 255, 200, 0.5)", "line": "rgba(200, 200, 0, 0.5)"},
        "dangerous_high": {"fill": "rgba(255, 200, 200, 0.5)", "line": "rgba(200, 0, 0, 0.5)"}
    }

    EVENT_STYLES: dict[str, dict[str, Any]] = {
        "Insulin": {
            "icon": "syringe.svg",
            "color": "#7b1fa2",
            # Data-coordinate icon box (~2× the earlier layout-image size).
            "icon_size_x": 2.8,
            "icon_size_y_frac": 0.12,
        },
        "Exercise": {
            "symbol": "star",
            "color": "orange",
            "size": 20,
        },
        "Carbohydrates": {
            "icon": "apple.svg",
            "color": "#2e7d32",
            "icon_size_x": 2.6,
            "icon_size_y_frac": 0.11,
        },
    }

    def __init__(
        self,
        id: str = "glucose-chart",
        hide_last_hour: bool = False,
    ) -> None:
        super().__init__(
            [
                dcc.Store(id=f"{id}-df-store", data=None, storage_type=STORAGE_TYPE),
                dcc.Store(id=f"{id}-events-store", data=None, storage_type=STORAGE_TYPE),
                dcc.Store(id=f"{id}-source-store", data=None, storage_type=STORAGE_TYPE),
                html.Div(
                    id=f"{id}-food-bubbles",
                    className="meal-food-bubble-strip",
                    children=[],
                    disable_n_clicks=True,
                ),
                dcc.Graph(
                    id=f"{id}-graph",
                    figure=self._create_empty_figure(),
                    config={
                        'displayModeBar': False,
                        'scrollZoom': False,
                        'doubleClick': 'reset',
                        'showAxisDragHandles': False,
                        'showAxisRangeEntryBoxes': False,
                        'displaylogo': False,
                        'modeBarButtonsToAdd': ['drawopenpath', 'eraseshape'],
                        'editable': False,
                        'edits': {
                            'shapePosition': False,
                            'annotationPosition': False
                        }
                    },
                    # `touchAction: none` prevents the browser from intercepting
                    # touch gestures (pinch-zoom, pan) on the chart, which
                    # otherwise fights with Plotly's drawline handler on mobile.
                    style={'height': '100%', 'touchAction': 'none', 'flex': '1', 'minHeight': '0'}
                )
            ],
            className="glucose-chart-shell",
            style={'height': '100%', 'display': 'flex', 'flexDirection': 'column'},
        )
        self.id = id
        self.hide_last_hour = hide_last_hour
        self._display_unit: str = "mg/dL"
        self._display_factor: float = 1.0

    def _create_empty_figure(self) -> go.Figure:
        """Create an empty figure with basic layout"""
        fig = go.Figure()
        fig.update_layout(
            title='Glucose Levels',
            autosize=True,
            xaxis=dict(title='Time'),
            yaxis=dict(title='Glucose Level (mg/dL)'),
            margin=dict(l=50, r=20, t=80, b=50),
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5,
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        return fig

    def register_callbacks(self, app: Dash) -> None:
        """Register all glucose chart related callbacks"""
        
        @app.callback(
            [Output(f'{self.id}-df-store', 'data'),
             Output(f'{self.id}-events-store', 'data'),
             Output(f'{self.id}-source-store', 'data')],
            [Input('current-window-df', 'data'),
             Input('events-df', 'data'),
             Input('data-source-name', 'data')],
            [State('url', 'pathname')]
        )
        def store_chart_data(
            df_data: Optional[dict[str, Any]],
            events_data: Optional[dict[str, Any]],
            source_name: Optional[str],
            pathname: Optional[str],
        ) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]], Optional[str]]:
            """Store the current DataFrame and events data when they change"""
            if pathname != '/prediction':
                raise PreventUpdate
            with start_action(
                action_type=u"glucose_store_chart_data",
                source=source_name
            ):
                return df_data, events_data, source_name

        @app.callback(
            [Output(f'{self.id}-graph', 'figure'),
             Output(f'{self.id}-food-bubbles', 'children')],
            [Input(f'{self.id}-df-store', 'data'),
             Input(f'{self.id}-events-store', 'data'),
             Input(f'{self.id}-source-store', 'data'),
             Input('glucose-chart-mode', 'data'),
             Input('glucose-unit', 'data'),
             Input('interface-language', 'data')],
            [State('url', 'pathname')]
        )
        def update_chart_figure(
            df_data: Optional[dict[str, Any]],
            events_data: Optional[dict[str, Any]],
            source_name: Optional[str],
            mode_data: Optional[dict[str, Any]],
            glucose_unit: Optional[str],
            interface_language: Optional[str],
            pathname: Optional[str],
        ) -> tuple[go.Figure, list[html.Button]]:
            """Update the chart figure when data changes"""
            if pathname != '/prediction':
                raise PreventUpdate
            if not df_data:
                return self._create_empty_figure(), []
            locale = normalize_locale(interface_language)
            
            # Reconstruct DataFrames from stored data
            df = self._reconstruct_dataframe_from_dict(df_data)
            events_df = self._reconstruct_events_dataframe_from_dict(events_data) if events_data else pl.DataFrame()
            hide_mode = mode_data or {'hide_last_hour': self.hide_last_hour}
            hide_last_hour_flag = hide_mode.get('hide_last_hour', self.hide_last_hour)
            self.hide_last_hour = hide_last_hour_flag
            self._display_unit = glucose_unit if glucose_unit in ("mg/dL", "mmol/L") else "mg/dL"
            self._display_factor = (1.0 / 18.0) if self._display_unit == "mmol/L" else 1.0
            
            with start_action(
                action_type=u"glucose_update_figure",
                points=len(df),
                gl_min=df.get_column('gl').min(),
                gl_max=df.get_column('gl').max(),
                source=source_name,
                hide_last_hour=hide_last_hour_flag
            ):
                figure = self._build_figure(df, events_df, source_name, locale=locale)
                bubbles = meal_food_bubble_children(
                    df,
                    events_df,
                    source_name=str(source_name or ""),
                    hide_last_hour=hide_last_hour_flag,
                )
                return figure, bubbles

        app.clientside_callback(
            """
            function(nClicks) {
                const cs = window.dash_clientside;
                if (!nClicks || !nClicks.some(Boolean)) {
                    return [cs.no_update, cs.no_update, cs.no_update];
                }
                const triggered = cs.callback_context.triggered_id;
                if (!triggered || triggered.type !== 'meal-food-bubble' || !triggered.index) {
                    return [cs.no_update, cs.no_update, cs.no_update];
                }
                return ['meal-food-lightbox is-open', triggered.index, 'false'];
            }
            """,
            [
                Output('meal-food-lightbox', 'className'),
                Output('meal-food-lightbox-image', 'src'),
                Output('meal-food-lightbox', 'aria-hidden'),
            ],
            Input({'type': 'meal-food-bubble', 'index': ALL}, 'n_clicks'),
            prevent_initial_call=True,
        )
        app.clientside_callback(
            """
            function(nClicks) {
                const cs = window.dash_clientside;
                if (!nClicks) {
                    return [cs.no_update, cs.no_update];
                }
                return ['meal-food-lightbox', 'true'];
            }
            """,
            [
                Output('meal-food-lightbox', 'className', allow_duplicate=True),
                Output('meal-food-lightbox', 'aria-hidden', allow_duplicate=True),
            ],
            Input('meal-food-lightbox-backdrop', 'n_clicks'),
            prevent_initial_call=True,
        )

    def _reconstruct_dataframe_from_dict(self, df_data: dict[str, list[Any]]) -> pl.DataFrame:
        """Reconstruct a Polars DataFrame from stored dictionary data"""
        return pl.DataFrame({
            'time': pl.Series(df_data['time']).str.strptime(pl.Datetime, format='%Y-%m-%dT%H:%M:%S'),
            'gl': pl.Series(df_data['gl'], dtype=pl.Float64),
            'prediction': pl.Series(df_data['prediction'], dtype=pl.Float64),
            'age': pl.Series([int(float(x)) for x in df_data['age']], dtype=pl.Int64),
            'user_id': pl.Series([int(float(x)) for x in df_data['user_id']], dtype=pl.Int64)
        })

    def _reconstruct_events_dataframe_from_dict(self, events_data: dict[str, list[Any]]) -> pl.DataFrame:
        """Reconstruct the events DataFrame from stored data"""
        n_rows = len(events_data.get('time') or [])
        photo_raw = events_data.get('photo_path')
        photo_paths = (
            [str(value or '') for value in photo_raw]
            if photo_raw is not None and len(photo_raw) == n_rows
            else [''] * n_rows
        )
        return pl.DataFrame({
            'time': pl.Series(events_data['time'], dtype=pl.String).str.strptime(pl.Datetime, format='%Y-%m-%dT%H:%M:%S'),
            'event_type': pl.Series(events_data['event_type'], dtype=pl.String),
            'event_subtype': pl.Series(events_data['event_subtype'], dtype=pl.String),
            # Coerce numeric strings and mixed integer/float values to Float64.
            'insulin_value': pl.Series(events_data['insulin_value'], dtype=pl.Float64, strict=False),
            'photo_path': pl.Series(photo_paths, dtype=pl.String),
        })

    def _build_figure(self, df: pl.DataFrame, events_df: pl.DataFrame, source_name: Optional[str] = None, *, locale: str = "en") -> go.Figure:
        """Build complete figure with all components"""
        figure = go.Figure()
        
        # Store data for internal methods
        self._current_df = df
        self._current_events = events_df
        self._current_source = source_name
        
        # Build all components
        self._add_range_rectangles(figure)
        self._add_glucose_trace(figure, locale=locale)
        self._add_prediction_traces(figure, locale=locale)
        self._add_event_markers(figure, locale=locale)
        self._add_food_photo_guides(figure, locale=locale)
        self._update_layout(figure, locale=locale)
        
        return figure

    def _add_range_rectangles(self, figure: go.Figure) -> None:
        """Add colored range rectangles to indicate glucose ranges."""
        f = self._display_factor
        # Add rectangle for high range (>180 mg/dL)
        figure.add_hrect(
            y0=180 * f, y1=400 * f,
            fillcolor="rgba(255, 0, 0, 0.1)",
            line_width=0,
            xref='x',
            yref='y'
        )
        
        # Add rectangle for low range (<70 mg/dL)
        figure.add_hrect(
            y0=0 * f, y1=70 * f,
            fillcolor="rgba(255, 0, 0, 0.1)",
            line_width=0,
            xref='x',
            yref='y'
        )
        
        # Add rectangle for target range (70-180 mg/dL)
        figure.add_hrect(
            y0=70 * f, y1=180 * f,
            fillcolor="rgba(0, 255, 0, 0.1)",
            line_width=0,
            xref='x',
            yref='y'
        )

    def _calculate_y_axis_range(self) -> tuple[float, float]:
        """Calculates the y-axis range based on glucose and prediction values."""
        f = self._display_factor
        STANDARD_MIN = 40 * f  # Standard lower bound for CGM charts
        STANDARD_MAX = 300 * f  # Upper bound for CGM chart
        
        line_points = self._current_df.filter(pl.col("prediction") != 0.0)
        
        # Get actual data ranges
        data_min = float(self._current_df.get_column("gl").min()) * f
        data_max = float(self._current_df.get_column("gl").max()) * f
        
        # Include prediction values in range calculation if they exist
        if line_points.height > 0:
            pred_max = float(line_points.get_column("prediction").max()) * f
            data_max = max(data_max, pred_max)
        
        # Set bounds with some padding
        lower_bound = min(STANDARD_MIN, max(0, data_min * 0.9))
        upper_bound = max(STANDARD_MAX, data_max * 1.1)
        
        return lower_bound, upper_bound

    def _add_glucose_trace(self, figure: go.Figure, *, locale: str) -> None:
        """Adds the main glucose data line to the figure."""
        f = self._display_factor
        # Determine how many points to show based on hide_last_hour setting
        if self.hide_last_hour:
            # Show only data points minus PREDICTION_HOUR_OFFSET (hide last hour)
            visible_points = len(self._current_df) - PREDICTION_HOUR_OFFSET +1
            visible_df = self._current_df.slice(0, visible_points)
            x_indices = list(range(visible_points))
            glucose_values = visible_df['gl'] * f
        else:
            # Show all data points
            x_indices = list(range(len(self._current_df)))
            glucose_values = self._current_df['gl'] * f
        
        figure.add_trace(go.Scatter(
            x=x_indices,
            y=glucose_values,
            mode='lines+markers',
            name=t("ui.chart.trace_glucose", locale=locale),
            line=dict(color='blue'),
        ))


    def _get_time_position(self, time_point: datetime) -> float:
        """Converts a datetime to its corresponding x-axis position."""
        time_series = self._current_df.get_column("time")
        for idx, time_val in enumerate(time_series):
            if time_val == time_point:
                return idx
        return 0

    def _add_prediction_traces(self, figure: go.Figure, *, locale: str) -> None:
        """Adds prediction points and connecting lines to the figure."""
        f = self._display_factor
        line_points = self._current_df.filter(pl.col("prediction") != 0.0)
        if line_points.height > 0:
            x_positions = [self._get_time_position(t) for t in line_points.get_column("time")]
            
            # Filter predictions to only show in allowed area when hiding last hour
            if self.hide_last_hour:
                visible_points = len(self._current_df) - PREDICTION_HOUR_OFFSET
                # Only show predictions in the hidden area (after PREDICTION_HOUR_OFFSET)
                filtered_data = []
                seen_positions = set()  # Track seen positions to avoid duplicates
                
                for i, (pos, pred, time_val) in enumerate(zip(x_positions, line_points.get_column("prediction"), line_points.get_column("time"))):
                    if pos >= visible_points and pos not in seen_positions:  # Only show unique predictions in the hidden area
                        filtered_data.append((pos, pred, time_val))
                        seen_positions.add(pos)
                
                # Sort by position to ensure proper order for line drawing
                filtered_data.sort(key=lambda x: x[0])
                
                if filtered_data:
                    x_positions, predictions, custom_data = zip(*filtered_data)
                    x_positions = list(x_positions)
                    predictions = [float(p) * f for p in list(predictions)]
                    custom_data = list(custom_data)
                else:
                    x_positions, predictions, custom_data = [], [], []
            else:
                # Remove duplicates even when showing all predictions
                unique_data = []
                seen_positions = set()
                
                for i, (pos, pred, time_val) in enumerate(zip(x_positions, line_points.get_column("prediction"), line_points.get_column("time"))):
                    if pos not in seen_positions:
                        unique_data.append((pos, pred, time_val))
                        seen_positions.add(pos)
                
                # Sort by position to ensure proper order
                unique_data.sort(key=lambda x: x[0])
                
                if unique_data:
                    x_positions, predictions, custom_data = zip(*unique_data)
                    x_positions = list(x_positions)
                    predictions = [float(p) * f for p in list(predictions)]
                    custom_data = list(custom_data)
                else:
                    predictions = (line_points.get_column("prediction") * f).to_list()
                    custom_data = line_points.get_column("time").to_list()
            
            if x_positions:  # Only add traces if we have data to show
                # Add prediction points
                figure.add_trace(go.Scatter(
                    x=x_positions,
                    y=predictions,
                    mode='markers',
                    name=t("ui.chart.trace_predictions", locale=locale),
                    marker=dict(
                        color='red',
                        size=8,
                        symbol='circle'
                    ),
                    hoverinfo='x+y',
                    hoverlabel=dict(bgcolor='white'),
                    customdata=custom_data
                ))

                # Always join the prediction path to the last known glucose point.
                # `anchor_predictions_at_boundary` normally fills the boundary slot
                # while drawing, but restored / staging / prefilled windows can still
                # start later, and a detached red line reads as a broken prediction.
                boundary_idx = len(self._current_df) - PREDICTION_HOUR_OFFSET
                if 0 <= boundary_idx < len(self._current_df) and float(x_positions[0]) > boundary_idx:
                    known_value = self._current_df.get_column("gl")[boundary_idx]
                    if known_value is not None:
                        figure.add_trace(go.Scatter(
                            x=[boundary_idx, x_positions[0]],
                            y=[float(known_value) * f, predictions[0]],
                            mode='lines',
                            line=dict(color='red', width=2),
                            showlegend=False,
                            hoverinfo='skip'
                        ))

                # Add connecting lines between predictions
                if len(predictions) >= 2:
                    for i in range(len(predictions) - 1):
                        figure.add_trace(go.Scatter(
                            x=[x_positions[i], x_positions[i + 1]],
                            y=[predictions[i], predictions[i + 1]],
                            mode='lines',
                            line=dict(color='red', width=2),
                            showlegend=False,
                            hoverinfo='skip'
                        ))

    def _event_xy_for_time(self, event_time: datetime) -> tuple[float, float]:
        """Map an event timestamp onto chart (x index, y glucose) coordinates."""
        f = self._display_factor
        df_times = self._current_df.get_column("time")
        before_idx: Optional[int] = None
        after_idx: Optional[int] = None
        for i, time_val in enumerate(df_times):
            if time_val <= event_time:
                before_idx = i
            if time_val >= event_time and after_idx is None:
                after_idx = i
        if before_idx is None:
            before_idx = 0
        if after_idx is None:
            after_idx = len(df_times) - 1

        if df_times[before_idx] == event_time or before_idx == after_idx:
            x_pos = float(before_idx)
            glucose_value = float(self._current_df.get_column("gl")[before_idx]) * f
        else:
            before_time = df_times[before_idx].timestamp()
            after_time = df_times[after_idx].timestamp()
            factor = (event_time.timestamp() - before_time) / (after_time - before_time)
            x_pos = float(before_idx) + factor
            before_glucose = float(self._current_df.get_column("gl")[before_idx])
            after_glucose = float(self._current_df.get_column("gl")[after_idx])
            glucose_value = (before_glucose + (after_glucose - before_glucose) * factor) * f
        return x_pos, glucose_value

    def _add_event_markers(self, figure: go.Figure, *, locale: str) -> None:
        """Adds event markers (insulin syringe, exercise, carb apple) to the figure.

        Insulin/carbs use SVG ``layout_image`` markers (plotly.js does not render
        custom ``path://`` symbols from Python). On the prediction page
        (``hide_last_hour``), those icons appear only in known history — never
        in the last hour — so they cannot tip the draw. Results show all.
        """
        if self._current_events.height == 0:
            return

        start_time = self._current_df.get_column("time")[0]
        end_time = self._current_df.get_column("time")[-1]
        window_events = self._current_events.filter(
            (pl.col("time") >= start_time) & (pl.col("time") <= end_time)
        )

        # First index of the hidden / predicted hour (same as prediction_boundary).
        known_end_idx = len(self._current_df) - PREDICTION_HOUR_OFFSET

        legend_name_by_type: dict[str, str] = {
            "Insulin": t("ui.chart.event_insulin", locale=locale),
            "Exercise": t("ui.chart.event_exercise", locale=locale),
            "Carbohydrates": t("ui.chart.event_carbohydrates", locale=locale),
        }

        y_min, y_max = self._calculate_y_axis_range()
        y_span = max(y_max - y_min, 1.0)

        # Collect insulin/carb icons first so near-identical x positions can stack.
        icon_markers: list[dict[str, Any]] = []
        for event_type in ("Insulin", "Carbohydrates"):
            style = self.EVENT_STYLES[event_type]
            events = window_events.filter(pl.col("event_type") == event_type)
            if event_type == "Insulin":
                events = events.filter(
                    pl.col("insulin_value").is_not_null() & (pl.col("insulin_value") != 0)
                )
            if events.height == 0:
                continue
            for event_time in events.get_column("time"):
                x_pos, glucose_value = self._event_xy_for_time(event_time)
                if self.hide_last_hour and x_pos > float(known_end_idx):
                    continue
                event_row = events.filter(pl.col("time") == event_time)
                if event_type == "Carbohydrates" and "photo_path" in events.columns:
                    photo = str(event_row.get_column("photo_path")[0] or "").strip()
                    if photo:
                        continue
                if event_type == "Insulin":
                    hover = t(
                        "ui.chart.hover_insulin",
                        locale=locale,
                        value=event_row.get_column("insulin_value")[0],
                        time=event_time.strftime("%H:%M"),
                    )
                else:
                    hover = (
                        f"{legend_name_by_type[event_type]}"
                        f"<br>{event_time.strftime('%H:%M')}"
                    )
                icon_markers.append(
                    {
                        "event_type": event_type,
                        "x": x_pos,
                        "y": glucose_value,
                        "hover": hover,
                        "style": style,
                    }
                )

        self._stack_icon_markers(icon_markers, y_span=y_span, y_max=y_max)
        icon_legend_entries = self._draw_icon_markers(
            figure, icon_markers, legend_name_by_type=legend_name_by_type
        )

        # Exercise keeps a normal Plotly marker (no SVG stacking).
        exercise_style = self.EVENT_STYLES["Exercise"]
        exercise_events = window_events.filter(pl.col("event_type") == "Exercise")
        if exercise_events.height > 0:
            x_positions = []
            y_positions = []
            hover_texts = []
            for event_time in exercise_events.get_column("time"):
                x_pos, glucose_value = self._event_xy_for_time(event_time)
                x_positions.append(x_pos)
                y_positions.append(glucose_value)
                hover_texts.append(
                    f"{legend_name_by_type['Exercise']}"
                    f"<br>{event_time.strftime('%H:%M')}"
                )
            figure.add_trace(
                go.Scatter(
                    x=x_positions,
                    y=y_positions,
                    mode="markers",
                    name=legend_name_by_type["Exercise"],
                    marker=dict(
                        symbol=str(exercise_style["symbol"]),
                        size=int(exercise_style["size"]),
                        color=str(exercise_style["color"]),
                        line=dict(width=2, color="white"),
                        opacity=0.9,
                    ),
                    text=hover_texts,
                    hoverinfo="text",
                )
            )

        self._add_icon_legend(figure, icon_legend_entries)

    def _add_food_photo_guides(self, figure: go.Figure, *, locale: str) -> None:
        """Thin green dotted meal line + translated FOOD label along it."""
        meals = visible_food_photo_events(
            self._current_df,
            self._current_events,
            hide_last_hour=self.hide_last_hour,
        )
        if not meals:
            return
        food_label = t("ui.chart.food_label", locale=locale)
        for meal in meals:
            event_time = meal["time"]
            if not isinstance(event_time, datetime):
                continue
            x_pos, _glucose = self._event_xy_for_time(event_time)
            figure.add_shape(
                type="line",
                x0=x_pos,
                x1=x_pos,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line=dict(color=_FOOD_LINE_COLOR, width=1.5, dash="dot"),
                layer="below",
            )
            figure.add_annotation(
                x=x_pos,
                y=0.52,
                xref="x",
                yref="paper",
                text=food_label,
                textangle=-90,
                showarrow=False,
                font=dict(size=11, color=_FOOD_LINE_COLOR),
                xanchor="left",
                yanchor="middle",
                xshift=7,
            )

    @staticmethod
    def _stack_icon_markers(
        markers: list[dict[str, Any]],
        *,
        y_span: float,
        y_max: float,
    ) -> None:
        """Offset overlapping insulin/carb icons so they stack vertically in place.

        Mutates each marker's ``y`` (display position). Markers whose x positions
        fall within ~half an icon width are treated as one stack.
        """
        if len(markers) < 2:
            return

        # Sort left-to-right, stable type order (Insulin below Carbs when tied).
        type_order = {"Insulin": 0, "Carbohydrates": 1}
        markers.sort(
            key=lambda m: (m["x"], type_order.get(str(m["event_type"]), 9))
        )

        overlap_x = 1.2  # ~half of typical icon_size_x — treat as same column
        stacks: list[list[dict[str, Any]]] = []
        for marker in markers:
            if stacks and abs(marker["x"] - stacks[-1][0]["x"]) <= overlap_x:
                stacks[-1].append(marker)
            else:
                stacks.append([marker])

        for stack in stacks:
            if len(stack) == 1:
                continue
            # Step ≈ one icon height so markers sit fully above each other.
            step = y_span * max(
                float(m["style"]["icon_size_y_frac"]) for m in stack
            )
            base_y = float(stack[0]["y"])
            stack_x = sum(float(m["x"]) for m in stack) / len(stack)
            # Prefer stacking upward; flip the whole column if it would clip.
            direction = 1.0
            if base_y + (len(stack) - 1) * step + step * 0.5 > y_max:
                direction = -1.0
            for i, marker in enumerate(stack):
                marker["x"] = stack_x
                marker["y"] = base_y + direction * i * step

    def _draw_icon_markers(
        self,
        figure: go.Figure,
        markers: list[dict[str, Any]],
        *,
        legend_name_by_type: dict[str, str],
    ) -> list[tuple[str, str]]:
        """Render stacked SVG icons + hover targets; return legend entries."""
        if not markers:
            return []

        y_min, y_max = self._calculate_y_axis_range()
        y_span = max(y_max - y_min, 1.0)
        legend_entries: list[tuple[str, str]] = []
        seen_types: set[str] = set()

        by_type: dict[str, list[dict[str, Any]]] = {}
        for marker in markers:
            by_type.setdefault(str(marker["event_type"]), []).append(marker)

        for event_type in ("Insulin", "Carbohydrates"):
            group = by_type.get(event_type)
            if not group:
                continue
            style = group[0]["style"]
            icon_uri = _svg_data_uri(str(style["icon"]))
            icon_size_x = float(style["icon_size_x"])
            icon_size_y = y_span * float(style["icon_size_y_frac"])
            x_positions = [float(m["x"]) for m in group]
            y_positions = [float(m["y"]) for m in group]
            hover_texts = [str(m["hover"]) for m in group]
            for x_pos, y_pos in zip(x_positions, y_positions):
                figure.add_layout_image(
                    dict(
                        source=icon_uri,
                        x=x_pos,
                        y=y_pos,
                        xref="x",
                        yref="y",
                        sizex=icon_size_x,
                        sizey=icon_size_y,
                        xanchor="center",
                        yanchor="middle",
                        sizing="contain",
                        layer="above",
                    )
                )
            legend_name = legend_name_by_type.get(event_type, event_type)
            figure.add_trace(
                go.Scatter(
                    x=x_positions,
                    y=y_positions,
                    mode="markers",
                    name=legend_name,
                    showlegend=False,
                    marker=dict(
                        symbol="circle",
                        size=28,
                        color=str(style["color"]),
                        opacity=0.0,
                        line=dict(width=0),
                    ),
                    text=hover_texts,
                    hoverinfo="text",
                )
            )
            if event_type not in seen_types:
                legend_entries.append((icon_uri, legend_name))
                seen_types.add(event_type)

        return legend_entries

    @staticmethod
    def _add_icon_legend(
        figure: go.Figure,
        entries: list[tuple[str, str]],
    ) -> None:
        """Paper-coord SVG + label so the legend matches the chart icons."""
        if not entries:
            return
        # Right-aligned row above the plot (alongside the horizontal Plotly legend).
        slot = 0.16
        right = 0.98
        start_x = right - slot * (len(entries) - 1)
        for i, (uri, label) in enumerate(entries):
            x = start_x + i * slot
            figure.add_layout_image(
                dict(
                    source=uri,
                    xref="paper",
                    yref="paper",
                    x=x - 0.06,
                    y=1.08,
                    sizex=0.035,
                    sizey=0.07,
                    xanchor="center",
                    yanchor="middle",
                    sizing="contain",
                    layer="above",
                )
            )
            figure.add_annotation(
                xref="paper",
                yref="paper",
                x=x - 0.038,
                y=1.08,
                text=label,
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font=dict(size=12, color="#333"),
            )

    @classmethod
    def build_static_figure(
        cls,
        df: "pl.DataFrame",
        events_df: "pl.DataFrame",
        source_name: Optional[str] = None,
        *,
        unit: str = "mg/dL",
        locale: str = "en",
        prediction_boundary: Optional[int] = None,
    ) -> go.Figure:
        """Build a complete figure from given data without touching any instance state.

        Args:
            df: Window DataFrame with ``gl`` and ``prediction`` columns.
            events_df: Events DataFrame (may be empty).
            source_name: Human-readable data source label.
            unit: ``"mg/dL"`` or ``"mmol/L"`` – controls y-axis scaling.
            locale: UI locale string.
            prediction_boundary: Index of the first *predicted* point. When
                supplied a vertical dashed line is drawn there and both regions
                are labelled. Results figures keep ``hide_last_hour=False`` so
                insulin/carb markers appear across the full window.
        """
        instance = cls.__new__(cls)
        instance.hide_last_hour = False
        instance._display_unit = unit if unit in ("mg/dL", "mmol/L") else "mg/dL"
        instance._display_factor = (1.0 / 18.0) if instance._display_unit == "mmol/L" else 1.0
        instance._current_df = df
        instance._current_events = events_df
        instance._current_source = source_name
        figure = instance._build_figure(df, events_df, source_name, locale=locale)

        if prediction_boundary is not None and 0 <= prediction_boundary <= len(df):
            x_pos = float(prediction_boundary)
            x_pos = max(-0.5, min(float(len(df)) - 0.5, x_pos))

            figure.add_shape(
                type="line",
                x0=x_pos,
                x1=x_pos,
                y0=0,
                y1=1,
                line=dict(color="orange", width=2, dash="dash"),
                xref="x",
                yref="paper",
            )
            figure.add_annotation(
                x=x_pos,
                y=0.98,
                text=f"← {t('ui.chart.known_label', locale=locale)} | {t('ui.chart.predicted_label', locale=locale)} →",
                showarrow=False,
                font=dict(size=11, color="orange"),
                bgcolor="white",
                bordercolor="orange",
                borderwidth=1,
                xref="x",
                yref="paper",
                xanchor="center",
                yanchor="top",
            )

        figure.update_layout(dragmode=False)
        return figure

    def _update_layout(self, figure: go.Figure, *, locale: str) -> None:
        """Updates the figure layout with axes, margins, and interaction settings."""
        y_range = self._calculate_y_axis_range()
        
        figure.update_layout(
            title="",
            autosize=True,
            xaxis=dict(
                title=t("ui.chart.x_axis", locale=locale),
                tickmode='array',
                tickvals=list(range(len(self._current_df))),
                ticktext=[time_val.strftime('%H:%M') for time_val in self._current_df.get_column("time")],
                fixedrange=True,
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                gridcolor='rgba(128, 128, 128, 0.2)',
                showgrid=True,
                range=[-0.5, len(self._current_df) - 0.5]
            ),
            yaxis=dict(
                # On /prediction the HTML unit chip beside the axis owns the label;
                # results/static figures keep the classic "Glucose Level (unit)" title.
                title=(
                    ""
                    if self.hide_last_hour
                    else t("ui.chart.y_axis", locale=locale, unit=self._display_unit)
                ),
                fixedrange=True,
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                gridcolor='rgba(128, 128, 128, 0.2)',
                showgrid=True,
                range=y_range
            ),
            # Extra top margin so insulin/carb SVG legend icons (paper y≈1.08) fit.
            # Prediction leaves room on the left for the unit chip near the axis.
            margin=dict(
                l=56 if self.hide_last_hour else 50,
                r=20,
                t=72,
                b=50,
            ),
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='top',
                y=1.08,
                xanchor='left',
                x=0.0,
            ),
            dragmode='drawline',
            hovermode='closest',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

