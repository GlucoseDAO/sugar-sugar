"""WSGI entry point for production gunicorn serving."""
from __future__ import annotations

from typing import Any

from sugar_sugar.app import bootstrap_wsgi_application


application: Any = bootstrap_wsgi_application()
