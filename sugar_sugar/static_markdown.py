"""Server-side Markdown rendering to avoid Dash's async dcc.Markdown (react-markdown) warnings."""

from __future__ import annotations

import re
from typing import Any, Final

import markdown
from dash import html

_MARKDOWN_EXTENSIONS: Final[tuple[str, ...]] = (
    "extra",
    "nl2br",
    "sane_lists",
    "tables",
    "fenced_code",
)

# Styles aligned with prior dcc.Markdown + landing/about cards (readable body text).
_SRCDOC_CSS: Final[str] = """
html { box-sizing: border-box; background: transparent !important; }
*, *::before, *::after { box-sizing: inherit; }
body {
  margin: 0;
  padding: 0;
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  color: #334155;
  line-height: 1.6;
  font-size: 16px;
  background: transparent !important;
}
h1, h2, h3 { color: #0f172a; margin: 1em 0 0.5em; font-weight: 800; }
h1 { font-size: 1.35rem; }
h2 { font-size: 1.2rem; }
h3 { font-size: 1.05rem; }
p { margin: 0 0 0.85em; }
a { color: #1e88e5; }
ul, ol { margin: 0 0 0.85em 1.1em; padding-left: 1.2em; }
table { border-collapse: collapse; width: 100%; margin: 0.5em 0; }
th, td { border: 1px solid rgba(15,23,42,0.12); padding: 6px 8px; text-align: left; }
code { font-size: 0.92em; background: transparent; padding: 0.1em 0.35em; border-radius: 4px; }
pre { background: transparent; padding: 10px; overflow: auto; border-radius: 8px; font-size: 0.92em; }
blockquote { margin: 0.5em 0; padding-left: 12px; border-left: 3px solid rgba(30,136,229,0.35); }

/* Dark mode overrides */
body.dark-mode {
    color: #ffffff;
}
body.dark-mode h1, body.dark-mode h2, body.dark-mode h3 {
    color: #ffffff;
}
body.dark-mode a {
    color: #7dd3fc;
}
body.dark-mode th, body.dark-mode td {
    border-color: #333;
}
body.dark-mode code, body.dark-mode pre {
    background: transparent;
    color: #ffffff;
}
body.dark-mode blockquote {
    border-left-color: rgba(99, 179, 237, 0.35);
}
"""


def _ensure_link_targets(html_fragment: str, *, target: str = "_blank") -> str:
    rel = "noopener noreferrer"

    def inject(m: re.Match[str]) -> str:
        tag = m.group(0)
        if re.search(r"\btarget\s*=", tag, flags=re.IGNORECASE):
            return tag
        return re.sub(
            r"<a\s+",
            f'<a target="{target}" rel="{rel}" ',
            tag,
            count=1,
            flags=re.IGNORECASE,
        )

    return re.sub(r"<a\s[^>]+>", inject, html_fragment, flags=re.IGNORECASE)


def _sanitize_srcdoc_fragment(fragment: str) -> str:
    # Prevent premature </iframe> or script injection from rare markdown edge cases.
    return fragment.replace("</script>", r"<\/script>")


def markdown_to_html_fragment(md: str) -> str:
    raw = markdown.markdown(md, extensions=list(_MARKDOWN_EXTENSIONS))
    return _ensure_link_targets(raw)


def html_fragment_to_srcdoc(html_fragment: str, *, theme: str = "light") -> str:
    safe = _sanitize_srcdoc_fragment(html_fragment)
    body_class = ' class="dark-mode"' if theme == "dark" else ""
    return (
        "<!DOCTYPE html><html><head>"
        '<meta charset="utf-8">'
        f"<style>{_SRCDOC_CSS}</style>"
        "</head><body" + body_class + ">"
        f"{safe}"
        "</body></html>"
    )


def static_markdown_iframe(
    md: str,
    *,
    title: str,
    iframe_style: dict[str, Any] | None = None,
    theme: str = "light",
) -> html.Iframe:
    """
    Render Markdown on the server and display it via iframe srcDoc.

    Avoids dcc.Markdown's async react-markdown bundle, which can log React 18
    warnings on first paint.
    """
    fragment = markdown_to_html_fragment(md)
    src_doc = html_fragment_to_srcdoc(fragment, theme=theme)
    style: dict[str, Any] = {
        "width": "100%",
        "border": "none",
        "background": "transparent !important",
        "boxShadow": "none",
        "display": "block",
        "height": "min(58vh, 780px)",
    }
    if iframe_style:
        style.update(iframe_style)
    return html.Iframe(
        srcDoc=src_doc,
        title=title,
        style=style,
        disable_n_clicks=True,
    )


def static_markdown_autosize_iframe(
    md: str,
    *,
    title: str,
    iframe_id: str = "study-design-iframe",
    theme: str = "light",
) -> html.Iframe:
    """Same as *static_markdown_iframe* but the iframe auto-expands to fit
    its content — no inner scrollbar.

    Relies on ``assets/autosize-iframe.js`` which reads
    ``contentDocument.scrollHeight`` from any iframe with
    ``data-autosize="true"`` (same-origin access is allowed for srcdoc).
    """
    src_doc = html_fragment_to_srcdoc(markdown_to_html_fragment(md), theme=theme)
    return html.Iframe(
        id=iframe_id,
        srcDoc=src_doc,
        title=title,
        style={
            "width": "100%",
            "border": "none",
            "background": "transparent !important",
            "boxShadow": "none",
            "display": "block",
            "overflow": "hidden",
            "height": "0",
        },
        disable_n_clicks=True,
        **{"data-autosize": "true"},
    )
