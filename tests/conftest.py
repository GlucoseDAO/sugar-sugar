from __future__ import annotations

import os


# The app's pretty Eliot renderer goes through eliottree, which currently emits
# Python 3.14 deprecation warnings for every rendered action. Tests assert
# behavior, not terminal log formatting, so keep the renderer out of pytest.
os.environ.setdefault("SUGAR_SUGAR_DISABLE_NICE_LOGS", "1")
