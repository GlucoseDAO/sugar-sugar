import os
from typing import Union


def _env_bool(name: str, default: Union[str, bool]) -> bool:
    if isinstance(default, str):
        return os.getenv(name, default).lower() in ("1", "true", "yes")
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes")


# Add this near the top with other type aliases
# represents the number of points to show in the graph and it's min and max (going from 2h to 4h)
DEFAULT_POINTS: int = int(os.getenv("DEFAULT_POINTS", "36"))
MIN_POINTS: int = int(os.getenv("MIN_POINTS", "24"))
MAX_POINTS: int = int(os.getenv("MAX_POINTS", "60"))

# Number of points (equivalent to hours) to subtract for prediction area
# 12 points = 1 hour (assuming 5-minute intervals)
PREDICTION_HOUR_OFFSET: int = int(os.getenv("PREDICTION_HOUR_OFFSET", "12"))
DOUBLE_CLICK_THRESHOLD: int = int(os.getenv("DOUBLE_CLICK_THRESHOLD", "500"))  # milliseconds

# A gap longer than this between consecutive readings means the sensor stopped:
# a playable window must not straddle one (see subject_sources.window_is_continuous).
# Matches cgm-format's own `FormatProcessor.small_gap_max_minutes` default, so the
# app and the library agree on what counts as continuous data.
SEQUENCE_GAP_MINUTES: int = int(os.getenv("SEQUENCE_GAP_MINUTES", "15"))

# Dash server (see README / .env.template)
DASH_HOST: str = os.getenv("DASH_HOST", "127.0.0.1")
DASH_PORT: int = int(os.getenv("DASH_PORT", "8050"))

# Public canonical origin for crawler-facing URLs and share metadata.
# In production set this to e.g. https://sugar-sugar.study.
DEPLOY_URL: str = os.getenv("DEPLOY_URL", "").strip().rstrip("/")

# Dash dcc.Store / component persistence type: 'local' (survives browser restart),
# 'session' (cleared when tab closes), or 'memory' (cleared on page refresh).
STORAGE_TYPE: str = os.getenv("STORAGE_TYPE", "local")

# Maximum number of prediction rounds per session
MAX_ROUNDS: int = int(os.getenv("MAX_ROUNDS", "12"))
MIN_USEFUL_ROUNDS: int = int(os.getenv("MIN_USEFUL_ROUNDS", str(max(1, MAX_ROUNDS // 2))))

# Umami analytics. Defaults use the same-domain Caddy proxy so common
# cross-domain analytics blocklists do not match the tracker URL.
UMAMI_SCRIPT_URL: str = os.getenv("UMAMI_SCRIPT_URL", "https://sugar-sugar.study/stats/script.js").strip()
UMAMI_WEBSITE_ID: str = os.getenv("UMAMI_WEBSITE_ID", "7c6fb178-d8ff-439e-a9f3-e289d9ec7e97").strip()
UMAMI_DOMAINS: str = os.getenv("UMAMI_DOMAINS", "sugar-sugar.study").strip()
UMAMI_HOST_URL: str = os.getenv("UMAMI_HOST_URL", "https://sugar-sugar.study/stats").strip()

# Share-mode defaults (used by `uv run share` dev shortcut)
SHARE_ROUNDS: int = int(os.getenv("SHARE_ROUNDS", str(MAX_ROUNDS)))
SHARE_NOISE: float = float(os.getenv("SHARE_NOISE", "0.30"))
SHARE_FORMATS: str = os.getenv("SHARE_FORMATS", "A")
SHARE_NAME: str = os.getenv("SHARE_NAME", "Dev Tester")

# Salt for the one-way `email_key` that groups a player's ranking rows across devices.
# The ranking CSVs store this hash, never the address. MUST stay stable for the lifetime
# of a deployment -- changing it re-splits every existing player into a new identity.
# Leave unset and sugar_sugar/nickname.py persists a random salt to data/.ranking_salt.
RANKING_EMAIL_SALT: str = os.getenv("RANKING_EMAIL_SALT", "").strip()

# Round labels on the synthesis chart: "single" (only when one format),
# "none" (never), "all" (always show, even with multiple formats).
SHARE_ROUND_LABELS: str = os.getenv("SHARE_ROUND_LABELS", "single").strip().lower()

# Application debug (e.g. test button); startup reads this dynamically after CLI may update it
DEBUG_MODE: bool = _env_bool("DEBUG_MODE", "false")
DASH_DEBUG: bool = _env_bool("DASH_DEBUG", DEBUG_MODE)

# Bump this integer on every deploy that changes clientside callback JS, OR that
# changes the Output list of an existing callback / removes one -- a browser still
# holding the old /_dash-dependencies POSTs the old output key and the server 500s
# with "Callback function not found for output ...".
# Dash computes its client-side fingerprint from the layout JSON, NOT from
# clientside callback content, so browsers cache old JS and survive server
# restarts without re-fetching /_dash-dependencies. Including this value in
# the layout as a dcc.Store forces the fingerprint to change and triggers a
# full client reload for every connected browser on the next server restart.
# 17: handle_time_slider gained an events-df Output (window-trimmed events store).
# 18: navbar gained the Highscore item (/highscore page) on desktop + mobile.
# 19: optional leaderboard nicknames -- fill_form_data gained a nickname-input Output.
# 21: CGMacros meal speech-bubble clientside open/close callbacks.
# 22: BIG IDEAs food-note notepad lightbox (extra Output on meal-food-lightbox).
# 23: meal-food-lightbox close also clears the image src (same 4 Outputs as open).
# 24: clustered meal bubbles open a composite gallery (extra gallery children Output).
# 25: composite gallery uses fixed img slots (clientside cannot create Img children).
# 26: challenge-unknown slider removed; paper-mention fields added to handle_start_button.
# 27: Results loading overlay (clientside) so /final's slow display_page is not a second click.
DEPLOY_BUILD: int = int(os.getenv("DEPLOY_BUILD", "31"))

