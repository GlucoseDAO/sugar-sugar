import base64
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Sequence
from datetime import datetime

import plotly.graph_objs as go
import polars as pl
from dash import ALL, dcc, Output, Input, State
from dash import Dash, html
from dash.exceptions import PreventUpdate
from eliot import start_action

from urllib.parse import quote

from flask import has_request_context, request as flask_request

from sugar_sugar.cgmacros import cgmacros_photo_url, visible_food_photo_events
from sugar_sugar.d1namo import d1namo_photo_url, is_d1namo_source_name
from sugar_sugar.food_note_i18n import translate_food_note
from sugar_sugar.components.submit import hidden_area_is_complete
from sugar_sugar.config import PREDICTION_HOUR_OFFSET, STORAGE_TYPE
from sugar_sugar.i18n import normalize_locale, t

# Same tokens as app._MOBILE_UA_KEYWORDS — kept here to avoid a circular import.
_MOBILE_UA_KEYWORDS: tuple[str, ...] = (
    "iphone", "android", "ipad", "mobile", "mobi", "opera mini",
)


def _chart_request_is_mobile() -> bool:
    """True when the current Flask request looks like a phone/tablet.

    Picks the compact Plotly layout (tighter margins, smaller hour ticks).
    Tests and CLI builds have no request and stay on the desktop layout.
    """
    if not has_request_context():
        return False
    ua = (flask_request.headers.get("User-Agent") or "").lower()
    return any(keyword in ua for keyword in _MOBILE_UA_KEYWORDS)

_FOOD_LINE_COLOR: str = "#2e7d32"
_APPLE_ICON_SRC: str = "/assets/images/apple.svg"
_FOOD_CLUSTER_X_GAP: float = 1.8
# Where a meal marker sits when it falls inside the hidden hour, as a fraction of
# the y-axis span. Deliberately not the meal's real glucose value -- that is the
# number the player is being asked to predict.
_HIDDEN_MARKER_Y_FRAC: float = 0.88
# Dose-scaled insulin circle on the glucose curve. 1 U → min px, 10 U+ → max px.
_INSULIN_CIRCLE_MIN_PX: int = 8
_INSULIN_CIRCLE_MAX_PX: int = 22
_INSULIN_DOSE_MIN_U: float = 1.0
_INSULIN_DOSE_MAX_U: float = 10.0
# Vertical gap (as a fraction of the y-span) when two doses share a timestamp.
# Circles and syringes stack one above the other; x stays on the injection time.
_INSULIN_STACK_Y_FRAC: float = 0.06
# Same 5-minute cell only. 1.0 is the next CGM reading (5 min later); a wider
# window stacked 19:31 onto 19:36 and left the later circle floating off the line.
_INSULIN_OVERLAP_X: float = 0.5
# Neighbours this close (5–7 min) share a label column — alternate above/below.
_INSULIN_LABEL_CLUSTER_X: float = 1.5
FOOD_COMPOSITE_MAX: int = 6
_FOOD_COMPOSITE_PREFIX: str = "composite:"

# Paper margins with automargin OFF — these are the real reserved strips.
# Plotly's default automargin grows t/b to fit the legend and rotated HH:MM
# ticks, which left the cartesian (drawn) area a short band in a tall white
# card on /prediction and /ending. Turning automargin off and keeping these
# values tiny is what actually enlarges the line. Numbers must stay in sync
# with assets/compact-chart.js (that file also Plotly.Plots.resize()s so the
# SVG fills the flex card in every run mode: debug, staging, production).
# Desktop `b` has to clear horizontal HH:MM; compact `b` has to clear
# the same labels rotated -90 (8px "HH:MM" is ~28px tall plus tick marks).
_COMPACT_MARGIN: dict[str, int] = {"l": 36, "r": 4, "t": 2, "b": 40, "pad": 0}
_DESKTOP_MARGIN: dict[str, int] = {"l": 50, "r": 8, "t": 8, "b": 36, "pad": 0}
_DESKTOP_PREDICTION_LEFT: int = 56


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


@dataclass(frozen=True, slots=True)
class FoodEventCluster:
    x_pos: float
    photo_urls: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _meal_photo_url(source_name: str, photo_path: str) -> str:
    if is_d1namo_source_name(source_name):
        return d1namo_photo_url(source_name, photo_path)
    return cgmacros_photo_url(source_name, photo_path)


def cluster_visible_food_events(
    window_df: pl.DataFrame,
    events_df: pl.DataFrame,
    *,
    source_name: str,
) -> list[FoodEventCluster]:
    """Group meal markers that would overlap into one clickable cluster.

    Meals in the predicted hour are included: see `visible_food_photo_events`.
    """
    if window_df.height == 0:
        return []
    meals = visible_food_photo_events(window_df, events_df)
    items: list[tuple[float, str, str]] = []
    for meal in meals:
        event_time = meal["time"]
        photo_path = str(meal.get("photo_path") or "").strip()
        food_note = str(meal.get("food_note") or "").strip()
        if not isinstance(event_time, datetime) or (not photo_path and not food_note):
            continue
        photo_url = _meal_photo_url(source_name, photo_path) if photo_path else ""
        items.append((event_x_index(window_df, event_time), photo_url, food_note))
    items.sort(key=lambda item: item[0])

    clusters: list[FoodEventCluster] = []
    for x_pos, photo_url, food_note in items:
        if clusters and abs(x_pos - clusters[-1].x_pos) <= _FOOD_CLUSTER_X_GAP:
            previous = clusters[-1]
            photos = list(previous.photo_urls)
            notes = list(previous.notes)
            if photo_url and photo_url not in photos:
                photos.append(photo_url)
            if food_note and food_note not in notes:
                notes.append(food_note)
            n_items = max(1, len(photos) + len(notes))
            merged_x = (previous.x_pos * (n_items - 1) + x_pos) / n_items
            clusters[-1] = FoodEventCluster(
                x_pos=merged_x,
                photo_urls=photos[:FOOD_COMPOSITE_MAX],
                notes=notes,
            )
            continue
        clusters.append(
            FoodEventCluster(
                x_pos=x_pos,
                photo_urls=[photo_url] if photo_url else [],
                notes=[food_note] if food_note else [],
            )
        )
    return clusters


def _bubble_index_for_cluster(cluster: FoodEventCluster) -> str:
    if len(cluster.photo_urls) > 1:
        return _FOOD_COMPOSITE_PREFIX + "|".join(cluster.photo_urls)
    if cluster.photo_urls:
        return cluster.photo_urls[0]
    return f"note:{quote(chr(10).join(cluster.notes), safe='')}"


def meal_food_bubble_children(
    window_df: pl.DataFrame,
    events_df: pl.DataFrame,
    *,
    source_name: str,
    locale: str = "en",
) -> list[html.Button]:
    """HTML speech bubbles above the plot, one per meal cluster in the window."""
    if window_df.height == 0:
        return []
    n_points = float(len(window_df))
    buttons: list[html.Button] = []
    for cluster in cluster_visible_food_events(
        window_df,
        events_df,
        source_name=source_name,
    ):
        cluster = FoodEventCluster(
            x_pos=cluster.x_pos,
            photo_urls=cluster.photo_urls,
            notes=[translate_food_note(note, locale) for note in cluster.notes],
        )
        left_pct = 100.0 * (cluster.x_pos + 0.5) / n_points
        count = max(len(cluster.photo_urls), 1 if cluster.notes else 0)
        extra: dict[str, str] = {}
        if count > 1:
            extra["data-count"] = str(count)
        food_label = t("ui.chart.food_label", locale=locale)
        accessible_label = t("ui.chart.event_carbohydrates", locale=locale)
        buttons.append(
            html.Button(
                [
                    html.Img(
                        src=_APPLE_ICON_SRC,
                        className="meal-food-bubble-apple",
                        alt="",
                        disable_n_clicks=True,
                    ),
                    html.Span(food_label, className="meal-food-bubble-label"),
                    html.Span(
                        "›",
                        className="meal-food-bubble-open",
                        **{"aria-hidden": "true"},
                    ),
                ],
                id={"type": "meal-food-bubble", "index": _bubble_index_for_cluster(cluster)},
                className="meal-food-speech-bubble",
                type="button",
                n_clicks=0,
                title=accessible_label,
                **{"aria-label": accessible_label},
                style={"left": f"{left_pct:.3f}%"},
                **extra,
            )
        )
    return buttons

_ASSETS_IMAGES = Path(__file__).resolve().parents[2] / "assets" / "images"

# Carb apples still use SVG layout images (plotly.js ignores custom path://
# markers). Insulin is a filled circle on the curve plus a smaller syringe at
# the plot base — see ``_add_insulin_markers``.
_ICON_EVENT_TYPES: frozenset[str] = frozenset({"Carbohydrates"})


def _insulin_circle_size(dose: float) -> int:
    """Map an insulin dose (U) onto a clamped circle diameter in pixels."""
    span = _INSULIN_DOSE_MAX_U - _INSULIN_DOSE_MIN_U
    fraction = (float(dose) - _INSULIN_DOSE_MIN_U) / span
    fraction = max(0.0, min(1.0, fraction))
    size = _INSULIN_CIRCLE_MIN_PX + fraction * (
        _INSULIN_CIRCLE_MAX_PX - _INSULIN_CIRCLE_MIN_PX
    )
    return int(round(size))


def _insulin_compact_label(dose: float) -> str:
    """Short mobile label next to the dose circle, e.g. ``4u`` / ``2.5u``."""
    return f"{dose:g}u"


def _insulin_label_positions(
    marks: Sequence[dict[str, Any]],
    *,
    y_min: float,
    y_max: float,
) -> list[str]:
    """Place ``Nu`` above or below the circle so neighbours stay readable.

    Isolated doses sit above unless they are near the y ceiling. A run of
    neighbours (Δx ≤ ``_INSULIN_LABEL_CLUSTER_X``) alternates, same idea as
    stacked circles: if there is no room above, start below.
    """
    if not marks:
        return []
    y_span = max(y_max - y_min, 1.0)
    indexed = list(enumerate(marks))
    ordered = sorted(indexed, key=lambda item: float(item[1]["x"]))
    clusters: list[list[tuple[int, dict[str, Any]]]] = []
    for item in ordered:
        if (
            clusters
            and abs(float(item[1]["x"]) - float(clusters[-1][-1][1]["x"]))
            <= _INSULIN_LABEL_CLUSTER_X
        ):
            clusters[-1].append(item)
        else:
            clusters.append([item])

    positions: list[str] = [""] * len(marks)
    for cluster in clusters:
        first_y = float(cluster[0][1]["glucose_y"])
        start_below = first_y > y_max - 0.18 * y_span
        for i, (orig_i, _mark) in enumerate(cluster):
            below = start_below if len(cluster) == 1 else ((i % 2 == 1) != start_below)
            positions[orig_i] = "bottom center" if below else "top center"
    return positions

# Finish-line marker on the playing chart: amber while the hidden hour is still
# unfinished, green once the drawn line reaches its last point.
_FINISH_PENDING_COLOR = "#b45309"
_FINISH_DONE_COLOR = "#15803d"
# Flag width in x-index units. Wide enough to read on a phone, narrow enough not
# to cover the points immediately before the finish line.
_FINISH_FLAG_SIZE_X = 2.2


@lru_cache(maxsize=8)
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
            # Base-of-plot syringe. The dose itself is the filled circle on the
            # glucose curve; this icon only marks the end of the dotted line.
            "icon_size_x": 2.4,
            "icon_size_y_frac": 0.09,
            "legend_sizex": 0.07,
            "legend_sizey": 0.14,
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
                    style={'height': '100%', 'touchAction': 'none', 'flex': '1', 'minHeight': '0'},
                    responsive=True,
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
            xaxis=dict(title='Time', automargin=False),
            yaxis=dict(title='Glucose Level (mg/dL)', automargin=False),
            margin=dict(_DESKTOP_MARGIN),
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
                figure = self._build_figure(
                    df, events_df, source_name, locale=locale,
                )
                bubbles = meal_food_bubble_children(
                    df,
                    events_df,
                    source_name=str(source_name or ""),
                    locale=locale,
                )
                return figure, bubbles

        _empty_tiles_js = "[" + ", ".join(["''"] * FOOD_COMPOSITE_MAX) + "]"
        app.clientside_callback(
            f"""
            function(nClicks) {{
                const cs = window.dash_clientside;
                const emptyTiles = {_empty_tiles_js};
                const skip = [cs.no_update, cs.no_update, cs.no_update, cs.no_update, cs.no_update];
                if (!nClicks || !nClicks.some(Boolean)) {{
                    return skip;
                }}
                const triggered = cs.callback_context.triggered_id;
                if (!triggered || triggered.type !== 'meal-food-bubble' || !triggered.index) {{
                    return skip;
                }}
                const index = String(triggered.index);
                if (index.startsWith('note:')) {{
                    let note = index.slice(5);
                    try {{ note = decodeURIComponent(note); }} catch (err) {{ /* keep raw */ }}
                    return ['meal-food-lightbox is-open is-note', '', note, 'false', emptyTiles];
                }}
                if (index.startsWith('composite:')) {{
                    const urls = index.slice(10).split('|').filter(Boolean).slice(0, {FOOD_COMPOSITE_MAX});
                    const tiles = emptyTiles.slice();
                    urls.forEach(function(src, i) {{ tiles[i] = src; }});
                    return ['meal-food-lightbox is-open is-composite', '', '', 'false', tiles];
                }}
                return ['meal-food-lightbox is-open', index, '', 'false', emptyTiles];
            }}
            """,
            [
                Output('meal-food-lightbox', 'className'),
                Output('meal-food-lightbox-image', 'src'),
                Output('meal-food-lightbox-note', 'children'),
                Output('meal-food-lightbox', 'aria-hidden'),
                Output({'type': 'meal-food-lightbox-tile', 'index': ALL}, 'src'),
            ],
            Input({'type': 'meal-food-bubble', 'index': ALL}, 'n_clicks'),
            prevent_initial_call=True,
        )
        app.clientside_callback(
            f"""
            function(nClicks) {{
                const cs = window.dash_clientside;
                if (!nClicks) {{
                    return [cs.no_update, cs.no_update, cs.no_update, cs.no_update, cs.no_update];
                }}
                return ['meal-food-lightbox', '', '', 'true', {_empty_tiles_js}];
            }}
            """,
            [
                Output('meal-food-lightbox', 'className', allow_duplicate=True),
                Output('meal-food-lightbox-image', 'src', allow_duplicate=True),
                Output('meal-food-lightbox-note', 'children', allow_duplicate=True),
                Output('meal-food-lightbox', 'aria-hidden', allow_duplicate=True),
                Output({'type': 'meal-food-lightbox-tile', 'index': ALL}, 'src', allow_duplicate=True),
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
        note_raw = events_data.get('food_note')
        food_notes = (
            [str(value or '') for value in note_raw]
            if note_raw is not None and len(note_raw) == n_rows
            else [''] * n_rows
        )
        return pl.DataFrame({
            'time': pl.Series(events_data['time'], dtype=pl.String).str.strptime(pl.Datetime, format='%Y-%m-%dT%H:%M:%S'),
            'event_type': pl.Series(events_data['event_type'], dtype=pl.String),
            'event_subtype': pl.Series(events_data['event_subtype'], dtype=pl.String),
            # Coerce numeric strings and mixed integer/float values to Float64.
            'insulin_value': pl.Series(events_data['insulin_value'], dtype=pl.Float64, strict=False),
            'photo_path': pl.Series(photo_paths, dtype=pl.String),
            'food_note': pl.Series(food_notes, dtype=pl.String),
        })

    def _build_figure(
        self,
        df: pl.DataFrame,
        events_df: pl.DataFrame,
        source_name: Optional[str] = None,
        *,
        locale: str = "en",
        compact: Optional[bool] = None,
    ) -> go.Figure:
        """Build complete figure with all components"""
        figure = go.Figure()

        # Store data for internal methods. Compact must be set before event
        # markers so the insulin/carb icon legend can sit in the thinner strip.
        self._current_df = df
        self._current_events = events_df
        self._current_source = source_name
        self._compact_layout = _chart_request_is_mobile() if compact is None else compact

        # Build all components
        self._add_range_rectangles(figure)
        self._add_glucose_trace(figure, locale=locale)
        self._add_prediction_traces(figure, locale=locale)
        self._add_event_markers(figure, locale=locale)
        self._add_food_photo_guides(figure)
        self._add_prediction_finish_line(figure, locale=locale)
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
        """Adds event markers (insulin, exercise, carb apple) to the figure.

        Carbs use SVG ``layout_image`` apples (plotly.js does not render custom
        ``path://`` symbols from Python). Insulin is a dose-scaled filled circle
        on the glucose curve plus a smaller syringe at the plot base, joined by
        a dotted line — see ``_add_insulin_markers``.

        On the prediction page (``hide_last_hour``) **meal and exercise markers
        stay visible in the predicted hour**. A player always knows when they
        ate or exercised; hiding those made them draw a flat line into an
        excursion they had no way to see coming.

        Insulin is the exception: the circle sits on the true glucose value, so
        drawing it in the hidden hour would hand over the answer. Doses past
        the boundary wait for the results chart (``hide_last_hour=False``).

        A meal/exercise marker past the boundary is pinned to a neutral rail
        near the top of the plot instead of sitting at its true glucose height,
        and gets a dotted guide line in its own event colour so its timing is
        unambiguous without the glucose trace behind it.
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
        # Neutral rail for markers past the prediction boundary: high enough to
        # read as "not a glucose value", low enough to leave room for stacking.
        hidden_marker_y = y_min + _HIDDEN_MARKER_Y_FRAC * y_span

        insulin_legend = self._add_insulin_markers(
            figure,
            window_events,
            known_end_idx=known_end_idx,
            locale=locale,
            legend_name=legend_name_by_type["Insulin"],
        )

        # Collect carb apples so near-identical x positions can stack.
        icon_markers: list[dict[str, Any]] = []
        hidden_marker_guides: list[tuple[float, str]] = []
        style = self.EVENT_STYLES["Carbohydrates"]
        events = window_events.filter(pl.col("event_type") == "Carbohydrates")
        for event_time in events.get_column("time") if events.height > 0 else []:
            x_pos, glucose_value = self._event_xy_for_time(event_time)
            past_boundary = self.hide_last_hour and x_pos > float(known_end_idx)
            if past_boundary:
                # Never place it at the hidden glucose value -- that is the answer.
                glucose_value = hidden_marker_y
            event_row = events.filter(pl.col("time") == event_time)
            photo = (
                str(event_row.get_column("photo_path")[0] or "").strip()
                if "photo_path" in events.columns
                else ""
            )
            note = (
                str(event_row.get_column("food_note")[0] or "").strip()
                if "food_note" in events.columns
                else ""
            )
            if photo or note:
                continue
            hover = (
                f"{legend_name_by_type['Carbohydrates']}"
                f"<br>{event_time.strftime('%H:%M')}"
            )
            if past_boundary:
                hidden_marker_guides.append((x_pos, str(style["color"])))
            icon_markers.append(
                {
                    "event_type": "Carbohydrates",
                    "x": x_pos,
                    "y": glucose_value,
                    "hover": hover,
                    "style": style,
                }
            )

        self._draw_hidden_marker_guides(figure, hidden_marker_guides)
        self._stack_icon_markers(icon_markers, y_span=y_span, y_max=y_max)
        icon_legend_entries = self._draw_icon_markers(
            figure, icon_markers, legend_name_by_type=legend_name_by_type
        )

        # Exercise keeps a normal Plotly marker (no SVG stacking).
        exercise_style = self.EVENT_STYLES["Exercise"]
        exercise_events = window_events.filter(pl.col("event_type") == "Exercise")
        hidden_exercise_guides: list[tuple[float, str]] = []
        if exercise_events.height > 0:
            x_positions = []
            y_positions = []
            hover_texts = []
            for event_time in exercise_events.get_column("time"):
                x_pos, glucose_value = self._event_xy_for_time(event_time)
                # Exercise was never gated by the boundary at all, so a star in
                # the predicted hour used to sit at the hidden glucose value and
                # read straight off the y-axis. Same rail as the other markers.
                if self.hide_last_hour and x_pos > float(known_end_idx):
                    glucose_value = hidden_marker_y
                    hidden_exercise_guides.append(
                        (x_pos, str(exercise_style["color"]))
                    )
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

        self._draw_hidden_marker_guides(figure, hidden_exercise_guides)
        self._add_icon_legend(figure, insulin_legend + icon_legend_entries)

    def _add_insulin_markers(
        self,
        figure: go.Figure,
        window_events: pl.DataFrame,
        *,
        known_end_idx: int,
        locale: str,
        legend_name: str,
    ) -> list[tuple[str, str, str]]:
        """Dose-scaled circle on the curve, syringe at the base, dotted connector.

        Circles sit at the interpolated glucose value so they mark the exact
        injection. On the prediction page doses past ``known_end_idx`` are
        omitted — that y *is* the answer. Results charts draw every dose.
        """
        style = self.EVENT_STYLES["Insulin"]
        events = window_events.filter(pl.col("event_type") == "Insulin")
        events = events.filter(
            pl.col("insulin_value").is_not_null() & (pl.col("insulin_value") != 0)
        )
        if events.height == 0:
            return []

        y_min, _y_max = self._calculate_y_axis_range()
        y_span = max(_y_max - y_min, 1.0)
        icon_size_x = float(style["icon_size_x"])
        icon_size_y = y_span * float(style["icon_size_y_frac"])
        base_y = y_min + 0.5 * icon_size_y
        color = str(style["color"])
        icon_uri = _svg_data_uri(str(style["icon"]))
        compact = bool(getattr(self, "_compact_layout", False))

        marks: list[dict[str, Any]] = []
        times = events.get_column("time")
        doses = events.get_column("insulin_value")
        for i in range(events.height):
            event_time = times[i]
            x_pos, glucose_y = self._event_xy_for_time(event_time)
            if self.hide_last_hour and x_pos > float(known_end_idx):
                continue
            dose = float(doses[i])
            hover = t(
                "ui.chart.hover_insulin",
                locale=locale,
                value=dose,
                time=event_time.strftime("%H:%M"),
            )
            marks.append(
                {
                    "x": x_pos,
                    "glucose_y": glucose_y,
                    "syringe_x": x_pos,
                    "syringe_y": base_y,
                    "dose": dose,
                    "hover": hover,
                }
            )

        if not marks:
            return []

        self._stack_overlapping_insulin(
            marks,
            circle_step=y_span * _INSULIN_STACK_Y_FRAC,
            syringe_step=icon_size_y,
            y_max=_y_max,
        )

        for mark in marks:
            figure.add_shape(
                type="line",
                x0=float(mark["x"]),
                x1=float(mark["syringe_x"]),
                y0=float(mark["glucose_y"]),
                y1=float(mark["syringe_y"]),
                xref="x",
                yref="y",
                line=dict(color=color, width=1.5, dash="dot"),
                layer="below",
            )
            figure.add_layout_image(
                dict(
                    source=icon_uri,
                    x=float(mark["syringe_x"]),
                    y=float(mark["syringe_y"]),
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

        x_positions = [float(m["x"]) for m in marks]
        y_positions = [float(m["glucose_y"]) for m in marks]
        sizes = [_insulin_circle_size(float(m["dose"])) for m in marks]
        hover_texts = [str(m["hover"]) for m in marks]
        labels = [_insulin_compact_label(float(m["dose"])) for m in marks]
        text_positions = _insulin_label_positions(marks, y_min=y_min, y_max=_y_max)
        figure.add_trace(
            go.Scatter(
                x=x_positions,
                y=y_positions,
                mode="markers+text" if compact else "markers",
                name=legend_name,
                showlegend=False,
                marker=dict(
                    symbol="circle",
                    size=sizes,
                    color=color,
                    line=dict(width=1, color="white"),
                ),
                text=labels if compact else None,
                textposition=text_positions if compact else None,
                textfont=dict(size=8, color=color) if compact else None,
                hovertext=hover_texts,
                hoverinfo="text",
            )
        )
        return [(icon_uri, legend_name, "Insulin")]

    @staticmethod
    def _stack_overlapping_insulin(
        marks: list[dict[str, Any]],
        *,
        circle_step: float,
        syringe_step: float,
        y_max: float,
    ) -> None:
        """Stack overlapping dose circles and syringes one above the other.

        Mutates ``glucose_y`` and ``syringe_y``. Circle ``x`` stays on the
        injection time so the column still reads as that moment.
        """
        if len(marks) < 2:
            return
        ordered = sorted(marks, key=lambda m: float(m["x"]))
        stacks: list[list[dict[str, Any]]] = []
        for mark in ordered:
            if stacks and abs(float(mark["x"]) - float(stacks[-1][0]["x"])) <= _INSULIN_OVERLAP_X:
                stacks[-1].append(mark)
            else:
                stacks.append([mark])
        for stack in stacks:
            if len(stack) < 2:
                continue
            curve_y = float(stack[0]["glucose_y"])
            base_syringe_y = float(stack[0]["syringe_y"])
            direction = 1.0
            if curve_y + (len(stack) - 1) * circle_step + circle_step * 0.5 > y_max:
                direction = -1.0
            for i, mark in enumerate(stack):
                mark["glucose_y"] = curve_y + direction * i * circle_step
                mark["syringe_y"] = base_syringe_y + i * syringe_step
                mark["syringe_x"] = float(mark["x"])

    @staticmethod
    def _draw_hidden_marker_guides(
        figure: go.Figure,
        guides: list[tuple[float, str]],
    ) -> None:
        """Dotted verticals for markers sitting in the hidden hour.

        Without the glucose trace behind it an icon on the neutral rail reads as
        floating; the line ties it to a time on the axis. Each is drawn in its
        own event colour so an apple is not announced by an orange exercise line.
        """
        for x_pos, color in guides:
            figure.add_shape(
                type="line",
                x0=x_pos,
                x1=x_pos,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line=dict(color=color, width=1.5, dash="dot"),
                layer="below",
            )

    def _add_food_photo_guides(self, figure: go.Figure) -> None:
        """Thin green dotted meal line. The FOOD label lives in the HTML bubble."""
        clusters = cluster_visible_food_events(
            self._current_df,
            self._current_events,
            source_name=str(self._current_source or ""),
        )
        if not clusters:
            return
        for cluster in clusters:
            x_pos = cluster.x_pos
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

    @staticmethod
    def _stack_icon_markers(
        markers: list[dict[str, Any]],
        *,
        y_span: float,
        y_max: float,
    ) -> None:
        """Offset overlapping carb apples so they stack vertically in place.

        Mutates each marker's ``y`` (display position). Markers whose x positions
        fall within ~half an icon width are treated as one stack. Insulin is
        drawn separately (circle on the curve, syringe at the base).
        """
        if len(markers) < 2:
            return

        # Sort left-to-right, stable type order (Insulin below Carbs when tied).
        type_order = {"Insulin": 0, "Carbohydrates": 1}
        markers.sort(
            key=lambda m: (m["x"], type_order.get(str(m["event_type"]), 9))
        )

        overlap_x = 1.35  # ~half of typical icon_size_x — treat as same column
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
    ) -> list[tuple[str, str, str]]:
        """Render stacked SVG icons + hover targets; return legend entries."""
        if not markers:
            return []

        y_min, y_max = self._calculate_y_axis_range()
        y_span = max(y_max - y_min, 1.0)
        legend_entries: list[tuple[str, str, str]] = []
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
                        size=int(style.get("hover_size", 28)),
                        color=str(style["color"]),
                        opacity=0.0,
                        line=dict(width=0),
                    ),
                    text=hover_texts,
                    hoverinfo="text",
                )
            )
            if event_type not in seen_types:
                legend_entries.append((icon_uri, legend_name, event_type))
                seen_types.add(event_type)

        return legend_entries

    def _add_icon_legend(
        self,
        figure: go.Figure,
        entries: list[tuple[str, str, str]],
    ) -> None:
        """Paper-coord SVG + label so the legend matches the chart icons."""
        if not entries:
            return
        compact = bool(getattr(self, "_compact_layout", False))
        # Right-aligned row inside the plot so the top paper margin can stay thin.
        slot = 0.16
        right = 0.98
        start_x = right - slot * (len(entries) - 1)
        legend_y = 0.97
        label_size = 10 if compact else 12
        for i, (uri, label, event_type) in enumerate(entries):
            x = start_x + i * slot
            style = self.EVENT_STYLES.get(event_type, {})
            sizex = float(style.get("legend_sizex", 0.035))
            sizey = float(style.get("legend_sizey", 0.07))
            if compact:
                sizex *= 0.72
                sizey *= 0.72
            icon_x = x - 0.06
            # Label starts just after the icon, same gap as the original 0.035-wide slot.
            label_x = icon_x + sizex / 2.0 + 0.0045
            figure.add_layout_image(
                dict(
                    source=uri,
                    xref="paper",
                    yref="paper",
                    x=icon_x,
                    y=legend_y,
                    sizex=sizex,
                    sizey=sizey,
                    xanchor="center",
                    yanchor="middle",
                    sizing="contain",
                    layer="above",
                )
            )
            figure.add_annotation(
                xref="paper",
                yref="paper",
                x=label_x,
                y=legend_y,
                text=label,
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font=dict(size=label_size, color="#333"),
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
        compact: Optional[bool] = None,
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
                insulin circles and carb markers appear across the full window.
            compact: Mobile-tight margins and hour ticks. ``None`` follows the
                current request User-Agent; tests should pass explicitly.
        """
        instance = cls.__new__(cls)
        instance.hide_last_hour = False
        instance._display_unit = unit if unit in ("mg/dL", "mmol/L") else "mg/dL"
        instance._display_factor = (1.0 / 18.0) if instance._display_unit == "mmol/L" else 1.0
        instance._current_df = df
        instance._current_events = events_df
        instance._current_source = source_name
        figure = instance._build_figure(
            df, events_df, source_name, locale=locale, compact=compact,
        )
        compact_layout = bool(instance._compact_layout)

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
                font=dict(size=9 if compact_layout else 11, color="orange"),
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

    def _hour_ticks(self) -> tuple[list[int], list[str]]:
        """Return x tick indexes and HH:MM labels, thinned on compact layouts."""
        times = self._current_df.get_column("time")
        count = len(times)
        compact = bool(getattr(self, "_compact_layout", False))
        if compact and count > 18:
            step = 3 if count > 40 else 2
            indexes = list(range(0, count, step))
            if indexes[-1] != count - 1:
                indexes.append(count - 1)
        else:
            indexes = list(range(count))
        return indexes, [times[index].strftime("%H:%M") for index in indexes]

    def _add_prediction_finish_line(self, figure: go.Figure, *, locale: str) -> None:
        """Mark the last point a prediction has to reach, on the playing chart.

        Reported August 2026: a player stopped drawing partway through the
        hidden hour, found Submit did nothing and took it for a bug. Nothing on
        the chart said where the line had to end -- the glucose trace simply
        stopped and the rest of the plot was empty -- and the one piece of copy
        that said so (``prediction-progress-label``) is ``display: none`` on
        mobile, where most rounds are played.

        So the target gets a marker in the same vocabulary as the meal and
        insulin events: a flag on a dashed vertical at the final x index, with a
        label that turns green once the line reaches it. Results charts never
        show it -- there the hour is already revealed.
        """
        if not self.hide_last_hour or self._current_df.height == 0:
            return

        finish_x = float(self._current_df.height - 1)
        complete = hidden_area_is_complete(self._current_df)
        color = _FINISH_DONE_COLOR if complete else _FINISH_PENDING_COLOR
        label = (
            t("ui.chart.prediction_complete", locale=locale)
            if complete
            else t("ui.chart.draw_to_here", locale=locale)
        )
        compact = bool(getattr(self, "_compact_layout", False))

        figure.add_shape(
            type="line",
            x0=finish_x,
            x1=finish_x,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color=color, width=2, dash="dash"),
            layer="below",
        )
        # The flag sits half a point from the plot's right edge, so both it and
        # the label hang leftwards -- centred on the line they would be clipped.
        size_y = 0.13 if compact else 0.11
        figure.add_layout_image(
            dict(
                source=_svg_data_uri("finish-flag.svg"),
                xref="x",
                yref="paper",
                x=finish_x,
                y=0.9,
                sizex=_FINISH_FLAG_SIZE_X,
                sizey=size_y,
                xanchor="right",
                yanchor="middle",
                sizing="contain",
                layer="above",
            )
        )
        figure.add_annotation(
            x=finish_x,
            y=0.9,
            text=label,
            showarrow=False,
            xref="x",
            yref="paper",
            xanchor="right",
            yanchor="top",
            # A couple of pixels clear of the plot border, or the label's own
            # frame is clipped by it.
            xshift=-5,
            yshift=-14,
            font=dict(size=10 if compact else 12, color=color),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=color,
            borderwidth=1,
            borderpad=2,
        )

    def _layout_margin(self) -> dict[str, int]:
        """Return Plotly paper margins. Compact and desktop both stay tight."""
        compact = bool(getattr(self, "_compact_layout", False))
        if compact:
            return dict(_COMPACT_MARGIN)
        margin = dict(_DESKTOP_MARGIN)
        if self.hide_last_hour:
            margin["l"] = _DESKTOP_PREDICTION_LEFT
        return margin

    def _update_layout(self, figure: go.Figure, *, locale: str) -> None:
        """Updates the figure layout with axes, margins, and interaction settings."""
        y_range = self._calculate_y_axis_range()
        compact = bool(getattr(self, "_compact_layout", False))
        tickvals, ticktext = self._hour_ticks()
        # HH:MM ticks already say this is time. A "Time"/"Timp" title sits in
        # the same bottom paper strip and overlaps the values on /ending.
        hide_y_title = compact or self.hide_last_hour

        figure.update_layout(
            title="",
            autosize=True,
            xaxis=dict(
                title="",
                title_standoff=0,
                tickmode="array",
                tickvals=tickvals,
                ticktext=ticktext,
                tickangle=-90 if compact else None,
                tickfont=dict(size=8 if compact else 11),
                ticks="outside",
                ticklen=2 if compact else 4,
                automargin=False,
                fixedrange=True,
                showspikes=True,
                spikemode="across",
                spikesnap="cursor",
                gridcolor="rgba(128, 128, 128, 0.2)",
                showgrid=True,
                range=[-0.5, len(self._current_df) - 0.5],
            ),
            yaxis=dict(
                # Compact and /prediction let ticks (and the HTML unit chip)
                # stand alone so the cartesian area can grow vertically.
                title=dict(
                    text=(
                        ""
                        if hide_y_title
                        else t("ui.chart.y_axis", locale=locale, unit=self._display_unit)
                    ),
                    font=dict(size=11 if compact else 14),
                    standoff=4,
                ),
                tickfont=dict(size=10 if compact else 12),
                automargin=False,
                fixedrange=True,
                showspikes=True,
                spikemode="across",
                spikesnap="cursor",
                gridcolor="rgba(128, 128, 128, 0.2)",
                showgrid=True,
                range=y_range,
            ),
            # Legend and insulin icons sit inside the plot, so the top strip
            # only needs a few pixels. automargin is off: these values stick.
            margin=self._layout_margin(),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.0,
                font=dict(size=10 if compact else 12),
                bgcolor="rgba(255,255,255,0.85)",
            ),
            dragmode="drawline",
            hovermode="closest",
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

