from __future__ import annotations

from typing import Any, Iterable, Optional

import dash_ag_grid as dag


READONLY_GRID_OPTIONS: dict[str, Any] = {
    "domLayout": "autoHeight",
    "suppressCellFocus": True,
    "suppressRowClickSelection": True,
    # Explicit row height. With domLayout="autoHeight", ag-grid otherwise measures
    # each row's content height; under the mobile.css cascade (html.mobile-device)
    # that measurement collapses to ~1px on phones, so only one row paints and the
    # /final per-round and /ending prediction tables look empty. A fixed rowHeight
    # makes the grid deterministic on every device (desktop already worked).
    "rowHeight": 42,
    "headerHeight": 42,
}

READONLY_DEFAULT_COL_DEF: dict[str, Any] = {
    "editable": False,
    "filter": False,
    "resizable": False,
    "sortable": False,
    "suppressMovable": True,
}

RESULT_ROW_STYLE: dict[str, Any] = {
    "styleConditions": [
        {
            "condition": "params.node.rowIndex === 0",
            "style": {"backgroundColor": "rgba(200, 240, 200, 0.5)"},
        },
        {
            "condition": "params.node.rowIndex === 1",
            "style": {"backgroundColor": "rgba(255, 200, 200, 0.5)"},
        },
    ]
}

RESULT_ROW_STYLE_WITH_AI: dict[str, Any] = {
    "styleConditions": [
        {
            "condition": "params.node.rowIndex === 0",
            "style": {"backgroundColor": "rgba(200, 240, 200, 0.5)"},
        },
        {
            "condition": "params.node.rowIndex === 1",
            "style": {"backgroundColor": "rgba(255, 200, 200, 0.5)"},
        },
        {
            "condition": "params.node.rowIndex >= 4",
            "style": {"backgroundColor": "rgba(200, 220, 255, 0.5)"},
        },
    ]
}


def build_readonly_column_defs(
    columns: Iterable[dict[str, Any]],
    *,
    metric_field: str = "metric",
    fixed_decimal_fields: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    """Convert simple Dash table column metadata into AG Grid column definitions."""
    fixed_decimal_fields = fixed_decimal_fields or set()
    column_defs: list[dict[str, Any]] = []
    for column in columns:
        field = str(column["id"])
        col_def: dict[str, Any] = {
            "field": field,
            "headerName": str(column["name"]),
            "minWidth": 80,
            "flex": 1,
            "cellStyle": {"textAlign": "center"},
        }
        if field == metric_field:
            col_def.update(
                {
                    "minWidth": 150,
                    "pinned": "left",
                    "cellStyle": {
                        "textAlign": "left",
                        "fontWeight": "bold",
                        "backgroundColor": "#f8fafc",
                    },
                }
            )
        if field in fixed_decimal_fields:
            col_def["valueFormatter"] = {
                "function": "params.value == null ? '' : Number(params.value).toFixed(2)"
            }
        column_defs.append(col_def)
    return column_defs


def build_readonly_ag_grid(
    *,
    table_id: str,
    row_data: list[dict[str, Any]],
    column_defs: list[dict[str, Any]],
    style: Optional[dict[str, Any]] = None,
    highlight_first_two_rows: bool = False,
    has_ai_rows: bool = False,
) -> dag.AgGrid:
    """Create a non-editable AG Grid for display-only result tables."""
    grid_style: dict[str, Any] = {"width": "100%"}
    if style:
        grid_style.update(style)

    if has_ai_rows:
        row_style = RESULT_ROW_STYLE_WITH_AI
    elif highlight_first_two_rows:
        row_style = RESULT_ROW_STYLE
    else:
        row_style = None

    return dag.AgGrid(
        id=table_id,
        rowData=row_data,
        columnDefs=column_defs,
        defaultColDef=READONLY_DEFAULT_COL_DEF,
        dashGridOptions=READONLY_GRID_OPTIONS,
        getRowStyle=row_style,
        dangerously_allow_code=any("valueFormatter" in column_def for column_def in column_defs),
        className="ag-theme-alpine",
        style=grid_style,
    )
