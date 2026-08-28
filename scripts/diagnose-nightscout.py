#!/usr/bin/env python3
"""Find out why a Nightscout import failed: the network, or the parser.

    uv run python scripts/diagnose-nightscout.py https://their-site.example.com

Run this **on the server that failed**, not on a laptop. It walks the import
stage by stage -- DNS, TCP, TLS, HTTP, then parse -- and reports where it stops.

There are two independent known failure modes and this tells them apart:

* **Network.** The instance is hosted in .ru. Outbound blocking in either
  direction (RKN on their side, a hosting provider's filtering on ours) shows up
  here as a DNS or connect failure while the control hosts still work. Two
  controls run for exactly this reason: a non-.ru host proves the server has
  egress at all, and a second .ru host proves whether the block is regional or
  specific to this instance.
* **Parser.** cgm-format 0.12.0 could not parse Nightscout treatments whose
  ``carbs``/``rate``/``insulin``/``duration`` are null for the first 100 records
  -- polars inferred dtype ``Null`` and then could not append a number. Fixed in
  0.12.2 (``FEEDBACK.md`` issue 1), but the check stays: it is how you tell a
  too-old library on a deployed host apart from a network fault. The OFFLINE
  PARSE stage reproduces it with synthetic data and **needs no network**, so it
  answers even on a server with no route to the instance at all.

Prints no glucose values, timestamps or tokens -- only counts, dtypes and status
codes -- so the output is safe to paste into an issue.
"""

from __future__ import annotations

import json
import socket
import ssl
import time
from typing import Any, Optional
from urllib.parse import urlparse

import typer

# Non-.ru control: proves this server has outbound internet at all.
CONTROL_HOST: str = "https://pypi.org"
# .ru control: separates "all of .ru is blocked" from "this instance is blocked".
CONTROL_RU_HOST: str = "https://ya.ru"

app = typer.Typer(add_completion=False)

# Set once from the CLI option. httpx.HTTPStatusError puts the full request URL
# -- query string and all -- into its message, so any exception text printed
# here can carry the access token. That is FEEDBACK.md's "no exception wrapping"
# item biting the diagnostic tool itself, and this output is meant to be pasted
# into issues.
_TOKEN: Optional[str] = None


def _scrub(text: str) -> str:
    """Remove the access token from anything about to be printed."""
    if not _TOKEN:
        return text
    return text.replace(_TOKEN, "<token-redacted>")


def _say(status: str, label: str, detail: str = "", elapsed: Optional[float] = None) -> None:
    """Print one stage result. ``status`` is OK / FAIL / WARN / INFO."""
    mark = {"OK": "  ok  ", "FAIL": " FAIL ", "WARN": " warn ", "INFO": " info "}.get(status, status)
    timing = f"  [{elapsed * 1000:.0f} ms]" if elapsed is not None else ""
    print(f"[{mark}] {_scrub(label)}{timing}")
    if detail:
        for line in _scrub(str(detail)).splitlines():
            print(f"         {line}")


def _stage(title: str) -> None:
    print(f"\n--- {title} " + "-" * max(0, 58 - len(title)))


def _resolve(host: str, port: int) -> Optional[list[str]]:
    """DNS only. A failure here is resolution, not reachability."""
    started = time.monotonic()
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except OSError as exc:
        _say("FAIL", f"DNS resolve {host}", f"{type(exc).__name__}: {exc}", time.monotonic() - started)
        return None
    addresses = sorted({str(info[4][0]) for info in infos})
    _say("OK", f"DNS resolve {host}", ", ".join(addresses), time.monotonic() - started)
    return addresses


def _tcp_then_tls(host: str, port: int, use_tls: bool, timeout: float) -> bool:
    """TCP connect, then a TLS handshake. Split so a TLS-only failure is visible."""
    started = time.monotonic()
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
    except OSError as exc:
        _say("FAIL", f"TCP connect {host}:{port}", f"{type(exc).__name__}: {exc}", time.monotonic() - started)
        return False
    _say("OK", f"TCP connect {host}:{port}", elapsed=time.monotonic() - started)

    if not use_tls:
        sock.close()
        return True

    started = time.monotonic()
    try:
        context = ssl.create_default_context()
        with context.wrap_socket(sock, server_hostname=host) as tls_sock:
            cipher = tls_sock.cipher()
            detail = f"{tls_sock.version()} {cipher[0] if cipher else ''}"
        _say("OK", f"TLS handshake {host}", detail, time.monotonic() - started)
        return True
    except (OSError, ssl.SSLError) as exc:
        _say("FAIL", f"TLS handshake {host}", f"{type(exc).__name__}: {exc}", time.monotonic() - started)
        return False
    finally:
        try:
            sock.close()
        except OSError:
            pass


def _http_get(client: Any, url: str, label: str, params: Optional[dict[str, str]] = None) -> Optional[Any]:
    """One GET, reporting status without following redirects, then following them."""
    import httpx

    started = time.monotonic()
    try:
        response = client.get(url, params=params, follow_redirects=False)
    except Exception as exc:  # httpx network errors are not a single base class
        _say("FAIL", label, f"{type(exc).__name__}: {exc}", time.monotonic() - started)
        return None
    elapsed = time.monotonic() - started

    if response.is_redirect:
        location = response.headers.get("location", "?")
        _say(
            "WARN",
            label,
            f"HTTP {response.status_code} redirect -> {location}\n"
            f"cgm-format does NOT follow redirects (FEEDBACK.md issue 2), so this alone\n"
            f"breaks the import. Retrying with redirects followed:",
            elapsed,
        )
        started = time.monotonic()
        try:
            response = client.get(url, params=params, follow_redirects=True)
        except Exception as exc:
            _say("FAIL", f"{label} (followed)", f"{type(exc).__name__}: {exc}", time.monotonic() - started)
            return None
        elapsed = time.monotonic() - started

    status = "OK" if response.status_code < 400 else "FAIL"
    _say(status, label, f"HTTP {response.status_code}, {len(response.content)} bytes", elapsed)
    return response if response.status_code < 400 else None


def _check_controls(timeout: float) -> None:
    """Prove whether this host has egress at all, and whether .ru is reachable."""
    import httpx

    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        for label, url in (("control (non-.ru)", CONTROL_HOST), ("control (.ru)", CONTROL_RU_HOST)):
            started = time.monotonic()
            try:
                response = client.get(url)
                _say("OK", f"{label} {url}", f"HTTP {response.status_code}", time.monotonic() - started)
            except Exception as exc:
                _say("FAIL", f"{label} {url}", f"{type(exc).__name__}: {exc}", time.monotonic() - started)


def _report_treatment_dtypes(payload: Any) -> None:
    """Show whether this instance triggers the null-run parser bug.

    Counts only -- no values, no timestamps.
    """
    if not isinstance(payload, list):
        _say("WARN", "treatments shape", f"expected a JSON list, got {type(payload).__name__}")
        return

    print(f"         {len(payload)} treatment records")
    triggers: list[str] = []
    for field in ("insulin", "carbs", "rate", "duration"):
        kinds: dict[str, int] = {}
        first_non_null: Optional[int] = None
        for index, record in enumerate(payload):
            value = record.get(field) if isinstance(record, dict) else None
            kinds[type(value).__name__] = kinds.get(type(value).__name__, 0) + 1
            if value is not None and first_non_null is None:
                first_non_null = index
        shown = "none" if first_non_null is None else str(first_non_null)
        print(f"         {field:9s} types={kinds}  first non-null at row {shown}")
        # polars infers from the first 100 rows; a longer null run means dtype Null.
        if first_non_null is not None and first_non_null >= 100:
            triggers.append(f"{field} (first value at row {first_non_null})")

    if triggers:
        _say(
            "INFO",
            "this payload has the shape that broke cgm-format 0.12.0",
            "Null past polars' 100-row inference window:\n"
            + "\n".join(f"  - {item}" for item in triggers)
            + "\nHarmless from 0.12.2 on (FEEDBACK.md issue 1). Whether THIS build handles it\n"
            "is decided by the offline check below, not by this line.",
        )
    else:
        _say("INFO", "this payload does not have the 0.12.0 null-run shape")


def _offline_parse_repro() -> None:
    """Reproduce the parser bug with synthetic data. Needs no network."""
    try:
        import importlib.metadata

        from cgm_format import FormatParser
    except ImportError as exc:
        _say("WARN", "offline parse repro", f"cgm-format not importable: {exc}")
        return

    try:
        _say("INFO", f"cgm-format {importlib.metadata.version('cgm-format')} installed here")
    except importlib.metadata.PackageNotFoundError:
        pass

    entries = [
        {"type": "sgv", "dateString": f"2026-08-28T{hour:02d}:00:00.000Z", "sgv": 120}
        for hour in range(5)
    ]
    treatments: list[dict[str, Any]] = [
        {
            "eventType": "Temp Basal",
            "created_at": "2026-08-28T00:00:00.000Z",
            "rate": 0.5,
            "duration": 30,
            "carbs": None,
            "insulin": None,
        }
        for _ in range(105)
    ]
    treatments.append(
        {
            "eventType": "Meal Bolus",
            "created_at": "2026-08-28T05:00:00.000Z",
            "carbs": 47,
            "insulin": 2.5,
        }
    )

    try:
        FormatParser.parse_nightscout(json.dumps(entries), json.dumps(treatments))
    except Exception as exc:
        _say(
            "FAIL",
            "offline parse repro (synthetic data, no network)",
            f"{type(exc).__name__}: {str(exc).splitlines()[0]}\n"
            "The parser bug is present in this environment, independent of any network issue.",
        )
        return
    _say("OK", "offline parse repro", "This cgm-format build parses the null-run case -- bug is fixed here.")


def _live_import(base_url: str, token: Optional[str], timeout: float) -> None:
    """The real thing: exactly what the app calls."""
    try:
        from cgm_format import FormatParser
    except ImportError as exc:
        _say("WARN", "live import", f"cgm-format not importable: {exc}")
        return

    started = time.monotonic()
    try:
        unified = FormatParser.from_nightscout_url(base_url, token=token, timeout=timeout)
    except Exception as exc:
        _say(
            "FAIL",
            "FormatParser.from_nightscout_url",
            f"{type(exc).__name__}: {str(exc).splitlines()[0]}",
            time.monotonic() - started,
        )
        return
    _say(
        "OK",
        "FormatParser.from_nightscout_url",
        f"{unified.height} unified rows -- the import works from this host.",
        time.monotonic() - started,
    )


@app.command()
def main(
    url: str = typer.Argument(..., help="Nightscout base URL, e.g. https://site.example.com"),
    token: Optional[str] = typer.Option(None, "--token", help="Nightscout access token, if the site needs one."),
    count: int = typer.Option(10_000, "--count", help="Records to request, matching what the app asks for."),
    timeout: float = typer.Option(30.0, "--timeout", help="Per-request timeout in seconds."),
    skip_controls: bool = typer.Option(False, "--skip-controls", help="Skip the reachability control hosts."),
) -> None:
    """Diagnose a failing Nightscout import from the host that failed."""
    global _TOKEN
    _TOKEN = token

    parsed = urlparse(url if "://" in url else f"https://{url}")
    host = parsed.hostname
    if host is None:
        raise typer.BadParameter(f"Could not read a hostname out of {url!r}")
    use_tls = parsed.scheme != "http"
    port = parsed.port or (443 if use_tls else 80)
    base = f"{parsed.scheme}://{parsed.netloc}".rstrip("/")

    print(f"Nightscout import diagnosis for {host}")
    print(f"cgm-format expectation: base URL is the bare origin, so it will call {base}/api/v1/entries.json")

    _stage("1. Can this host reach the internet at all?")
    if skip_controls:
        _say("INFO", "controls skipped")
    else:
        _check_controls(timeout)
        print("\n         Read this as:")
        print("           both ok         -> egress is fine, the instance itself is the problem")
        print("           non-.ru ok only -> .ru appears blocked from this host (the RKN/provider hypothesis)")
        print("           both fail       -> this host has no outbound internet; nothing else below is meaningful")

    _stage("2. Is the instance reachable?")
    if _resolve(host, port) is not None:
        _tcp_then_tls(host, port, use_tls, timeout)

    _stage("3. Do the Nightscout endpoints answer?")
    try:
        import httpx
    except ImportError:
        _say("FAIL", "httpx not installed", "Nightscout import cannot work at all without it.")
        httpx = None  # type: ignore[assignment]

    treatments_payload: Optional[Any] = None
    if httpx is not None:
        params: dict[str, str] = {"count": str(count)}
        if token:
            params["token"] = token
        with httpx.Client(timeout=timeout) as client:
            _http_get(client, f"{base}/api/v1/status.json", "GET /api/v1/status.json",
                      {"token": token} if token else None)
            _http_get(client, f"{base}/api/v1/entries.json", "GET /api/v1/entries.json", params)
            treatments = _http_get(client, f"{base}/api/v1/treatments.json", "GET /api/v1/treatments.json", params)
            if treatments is not None:
                try:
                    treatments_payload = treatments.json()
                except ValueError as exc:
                    _say("FAIL", "treatments JSON decode", f"{type(exc).__name__}: {exc}")

    _stage("4. What shape is this instance's treatment data?")
    if treatments_payload is None:
        _say("INFO", "no treatments payload to inspect (see above)")
    else:
        _report_treatment_dtypes(treatments_payload)

    _stage("5. Parser check that needs no network")
    _offline_parse_repro()

    _stage("6. The real import, exactly as the app calls it")
    _live_import(base, token, timeout)

    print("\nDone. Paste this whole output into the issue -- it contains no patient data or tokens.")


if __name__ == "__main__":
    app()
