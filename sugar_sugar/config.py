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

# Dash server (see README / .env.template)
DASH_HOST: str = os.getenv("DASH_HOST", "127.0.0.1")
DASH_PORT: int = int(os.getenv("DASH_PORT", "8050"))

# Dash dcc.Store / component persistence type: 'local' (survives browser restart),
# 'session' (cleared when tab closes), or 'memory' (cleared on page refresh).
STORAGE_TYPE: str = os.getenv("STORAGE_TYPE", "local")

# Maximum number of prediction rounds per session
MAX_ROUNDS: int = int(os.getenv("MAX_ROUNDS", "12"))

# Application debug (e.g. test button); startup reads this dynamically after CLI may update it
DEBUG_MODE: bool = _env_bool("DEBUG_MODE", "false")
DASH_DEBUG: bool = _env_bool("DASH_DEBUG", DEBUG_MODE)

# Bump this integer on every deploy that changes clientside callback JS.
# Dash computes its client-side fingerprint from the layout JSON, NOT from
# clientside callback content, so browsers cache old JS and survive server
# restarts without re-fetching /_dash-dependencies. Including this value in
# the layout as a dcc.Store forces the fingerprint to change and triggers a
# full client reload for every connected browser on the next server restart.
DEPLOY_BUILD: int = int(os.getenv("DEPLOY_BUILD", "2"))

