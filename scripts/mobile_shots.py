"""Mobile screenshot harness.

Renders the app's pages in a *nauseatingly narrow* phone viewport (and the
chart page additionally in landscape) and saves full-page PNGs, so obvious
visual artifacts -- overlaps, out-of-bounds content, clipped/oversized
elements, misalignments -- can be caught without deploying to staging and
opening on a real phone.

It drives Chromium over the Chrome DevTools Protocol via ``choreographer``
(already a transitive dependency through Plotly's kaleido), emulating a real
mobile device: ``mobile=true`` device metrics + touch emulation, which makes
``(pointer: coarse)`` / ``(hover: none)`` media queries match, plus a phone
``User-Agent`` override so the server-side mobile-layout branch
(``_is_mobile_request``) returns the mobile builders.

Usage::

    uv run python scripts/mobile_shots.py                     # all pages, default device
    uv run python scripts/mobile_shots.py --device iphone-se  # a specific device preset
    uv run python scripts/mobile_shots.py --only chart        # just the prediction group
    uv run python scripts/mobile_shots.py --only result       # ending/final/share (staging)
    uv run python scripts/mobile_shots.py --language-set babylon  # all locales, nested by language
    uv run python scripts/mobile_shots.py --base-url http://127.0.0.1:8050  # running server

Output goes to ``data/output/mobile_shots/`` by default.
"""

from __future__ import annotations

import asyncio
import base64
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import typer

# A phone-ish User-Agent so the server returns the mobile builders.
_IPHONE_UA: str = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
)
# Desktop UA so the server returns the desktop (non-wizard) builders.
_DESKTOP_UA: str = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_DESKTOP_VIEWPORT: tuple[int, int] = (1280, 1000)


@dataclass(frozen=True)
class Device:
    """A viewport preset.  Dimensions are CSS pixels (the layout viewport)."""

    name: str
    width: int
    height: int
    scale: int = 2


# Deliberately narrow presets.  `android-narrow` is the default torture test.
DEVICES: dict[str, Device] = {
    "android-narrow": Device("android-narrow", 360, 740),
    "iphone-se": Device("iphone-se", 320, 568),
    "iphone-13": Device("iphone-13", 390, 844),
    "pixel-7": Device("pixel-7", 412, 915),
}


# Screenshot language sets.  Keep the historic English-only flat output as the
# default; "babylon" is the expanded all-language mode and writes one directory
# per locale under the output root.
LANGUAGE_SETS: dict[str, tuple[str, ...]] = {
    "english": ("en",),
    "babylon": ("en", "de", "uk", "ro", "ru", "zh", "fr", "es"),
}


@dataclass(frozen=True)
class Shot:
    """A single screenshot job: a route plus optional landscape capture."""

    label: str
    path: str
    landscape: bool = False  # also capture rotated (long edge horizontal)
    settle_s: float = 1.5    # extra wait after hydration (charts need more)
    # Element ids to .click() in order before capturing (each followed by a
    # short wait for the Dash callback + DOM update).  Used to walk the mobile
    # startup wizard to a given step.
    clicks: tuple[str, ...] = ()
    # Arbitrary JS evaluated before clicks.  Used by gated flows where the
    # harness must satisfy a required checkbox/scroll condition before clicking.
    pre_click_js: Optional[str] = None
    # Arbitrary JS evaluated just before capture (e.g. scroll an inner iframe
    # to its bottom so the end-of-form + button is visible).
    pre_capture_js: Optional[str] = None
    # Grow the viewport height to the page scrollHeight before capturing.
    # Disable for viewport-sized pages (consent reader, chart).
    grow_height: bool = True
    # Landscape /prediction only: pin the LAYOUT viewport to each of these fixed
    # widths and emit one PNG per width. The app serves a responsive
    # `width=device-width` meta (correct for real devices), so to inspect the
    # immersive landscape chart at a representative desktop-ish width we drive the
    # CDP device metrics to a fixed width instead of the phone's ~800px. 1280
    # reproduces the historically-correct scale; 1920 shows how it scales wider.
    # Empty tuple -> use the device's own rotated landscape size.
    landscape_widths: tuple[int, ...] = ()
    # Optional pre-navigation seeding: load `seed_path` first, run `seed_js` (e.g.
    # set user-info-store consent so desktop /startup renders instead of redirecting
    # to landing), then navigate to `path`.
    seed_path: str = "/"
    seed_js: Optional[str] = None


@dataclass
class ServerGroup:
    """A set of routes served by one spawned server invocation."""

    name: str
    cmd: list[str]
    env: dict[str, str] = field(default_factory=dict)
    shots: list[Shot] = field(default_factory=list)
    # Render with a desktop UA + wide viewport (the non-wizard desktop builders)
    # instead of the phone emulation. Landscape variants are skipped for desktop.
    desktop: bool = False


def _show_mobile_step_js(step: int, fmt: Optional[str] = None) -> str:
    """JS to reveal one mobile wizard step directly (bypassing the per-step Next
    gate, which blocks blind click-through) and optionally select a data format
    (B/C reveals the data-usage consent + import block)."""
    set_fmt = ""
    if fmt:
        set_fmt = (
            "if(window.dash_clientside&&window.dash_clientside.set_props){"
            f"window.dash_clientside.set_props('format-dropdown',{{value:'{fmt}'}});}}"
        )
    return (
        "(function(){for(var k=0;k<6;k++){var d=document.getElementById('mobile-step-'+k);"
        f"if(d){{d.style.display=(k==={step}?'block':'none');}}}}" + set_fmt + "})()"
    )


def _server_groups(port: int, *, locale: str) -> list[ServerGroup]:
    """Define which routes are captured under which server invocation."""
    base = ["uv", "run"]
    # Seed consent so the DESKTOP /startup renders (it redirects to landing without
    # it; mobile /startup is exempt as the consent entry point).
    seed_desktop_consent = (
        "window.dash_clientside.set_props('user-info-store',"
        "{data:{consent_completed:true,consent_no_selection:true,study_id:'shot'}})"
    )
    return [
        ServerGroup(
            name="entry",
            cmd=base + ["start", "--port", str(port)],
            shots=[
                Shot("landing", "/"),
                Shot("consent-form", "/consent-form", grow_height=False),
                # Same page scrolled to the end of the consent text so the
                # bottom of the form + the back button are captured.
                Shot(
                    "consent-form-bottom",
                    "/consent-form",
                    grow_height=False,
                    pre_capture_js=(
                        "(function(){var f=document.querySelector("
                        "'#consent-form-scroll iframe');"
                        "if(f&&f.contentWindow&&f.contentWindow.document){"
                        "var d=f.contentWindow.document;"
                        "var el=d.scrollingElement||d.documentElement||d.body;"
                        "el.scrollTop=el.scrollHeight;}})()"
                    ),
                ),
                # Mobile startup wizard. Per-step Next-gating blocks blind
                # click-through, so each step is revealed directly for a layout shot.
                # Format A = the full wizard (all 6 steps, no upload gate); B/C only
                # need the format step, where the data-usage consent + import block
                # (upload + Nightscout) appear.
                Shot("startup-a-step1-consent", "/startup", pre_capture_js=_show_mobile_step_js(0)),
                Shot("startup-a-step2-identity", "/startup", pre_capture_js=_show_mobile_step_js(1)),
                Shot("startup-a-step3-cgm", "/startup", pre_capture_js=_show_mobile_step_js(2)),
                Shot("startup-a-step4-diabetes", "/startup", pre_capture_js=_show_mobile_step_js(3)),
                Shot("startup-a-step5-format", "/startup", pre_capture_js=_show_mobile_step_js(4, 'A')),
                Shot("startup-a-step6-contact", "/startup", pre_capture_js=_show_mobile_step_js(5)),
                Shot("startup-bc-step5-format-import", "/startup", pre_capture_js=_show_mobile_step_js(4, 'B')),
                Shot("about", "/about"),
                Shot("faq", "/faq"),
                Shot("contact", "/contact"),
                Shot("demo", "/demo"),
            ],
        ),
        ServerGroup(
            name="result",
            # Staging mode (prod+) exposes prefilled /staging/ending, /staging/final
            # and a /staging/share redirect, so the result/share pages are
            # screenshot-able without a full playthrough (the project forbids LLM
            # click-through; staging nodes replace it). `_STAGING_MODE=1` activates
            # the /staging/* routes; everything else is the normal `uv run start`
            # server. These are display-only pages -> device-width, grow-height.
            cmd=base + ["start", "--port", str(port)],
            env={"_STAGING_MODE": "1"},
            shots=[
                Shot("ending", "/staging/ending", settle_s=3.0),
                Shot("final", "/staging/final", settle_s=3.0),
                # /staging/share 302-redirects to a freshly generated /share/<id>;
                # the synthetic record cycles formats A/B/C so the multi-panel
                # synthesis graph is exercised.
                Shot("share", "/staging/share", settle_s=3.0),
            ],
        ),
        ServerGroup(
            name="chart",
            # --prefill fills the prediction region so submit/ending are reachable.
            # --no-debug/--no-reloader keeps the harness independent of
            # Werkzeug's debug fork while chart-mode env is still seeded by the
            # chart entry point before app import.
            # --format C ("mixed"): the prediction page shows the relocated Upload
            # button (left of Submit) that formats B/C use -- so the default
            # prediction shots now exercise that control.
            cmd=base + [
                "chart",
                "--prefill",
                "--format",
                "C",
                "--no-debug",
                "--no-reloader",
                "--locale",
                locale,
                "--port",
                str(port),
            ],
            shots=[
                # Capture both closed and open How-to-play states in portrait
                # and landscape; the same Shot with landscape=True writes both
                # orientation PNGs. Landscape is pinned to 1280 (primary, matches
                # the historically-correct scale) and 1920 (wide, to see scaling).
                Shot("prediction", "/prediction", landscape=True, settle_s=3.0, landscape_widths=(1280, 1920)),
                Shot(
                    "prediction-help-open",
                    "/prediction",
                    landscape=True,
                    settle_s=3.0,
                    clicks=("header-how-to-play-toggle",),
                    landscape_widths=(1280, 1920),
                ),
            ],
        ),
        ServerGroup(
            name="chart-a",
            # --format A ("generic"): no upload button -- the action strip keeps the
            # original Fullscreen(2x) + Submit/Finish layout. Captured so that
            # layout is validated to stay unchanged by the B/C button rework.
            cmd=base + [
                "chart",
                "--prefill",
                "--format",
                "A",
                "--no-debug",
                "--no-reloader",
                "--locale",
                locale,
                "--port",
                str(port),
            ],
            shots=[
                Shot("prediction-a", "/prediction", landscape=True, settle_s=3.0, landscape_widths=(1280, 1920)),
            ],
        ),
        ServerGroup(
            name="chart-b",
            # --format B ("my data only"): no data is loaded until the user uploads,
            # so /prediction renders the upload gate (message + Upload button) in
            # place of the chart. No --prefill (nothing to fill without a dataset).
            cmd=base + [
                "chart",
                "--format",
                "B",
                "--no-debug",
                "--no-reloader",
                "--locale",
                locale,
                "--port",
                str(port),
            ],
            shots=[
                Shot("prediction-b", "/prediction", landscape=True, settle_s=3.0, landscape_widths=(1280, 1920)),
            ],
        ),
        ServerGroup(
            name="desktop",
            # Desktop (non-wizard) builders: the whole startup form on one page,
            # which is where the roomy CGM import block lives. Consent is seeded so
            # /startup renders (it redirects to landing without it on desktop).
            cmd=base + ["start", "--port", str(port)],
            desktop=True,
            shots=[
                Shot("desktop-landing", "/", settle_s=2.0),
                Shot(
                    "desktop-startup-import",
                    "/startup",
                    seed_js=seed_desktop_consent,
                    settle_s=2.0,
                    pre_capture_js=(
                        "(function(){if(window.dash_clientside&&window.dash_clientside.set_props){"
                        "window.dash_clientside.set_props('format-dropdown',{value:'B'});"
                        "window.dash_clientside.set_props('data-usage-consent',{value:['agree']});}})()"
                    ),
                ),
            ],
        ),
    ]


def _wait_for_server(base_url: str, timeout_s: float = 60.0) -> bool:
    """Poll the server root until it responds or the timeout elapses."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(base_url, timeout=2) as resp:
                if resp.status < 500:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


async def _hydrate_wait(tab, timeout_s: float = 20.0) -> None:
    """Wait until Dash has rendered page-content (or timeout)."""
    deadline = time.monotonic() + timeout_s
    expr = (
        "(function(){var c=document.getElementById('page-content');"
        "return !!(c && c.children && c.children.length>0);})()"
    )
    while time.monotonic() < deadline:
        res = await tab.send_command(
            "Runtime.evaluate", {"expression": expr, "returnByValue": True}
        )
        if res.get("result", {}).get("result", {}).get("value") is True:
            return
        await asyncio.sleep(0.25)


async def _apply_locale(tab, locale: str) -> None:
    """Switch the page to a non-default locale using the existing Dash callback."""
    if locale == "en":
        return

    res = await tab.send_command(
        "Runtime.evaluate",
        {
            "expression": (
                f"(function(){{var e=document.getElementById('lang-{locale}');"
                "if(!e){return false;} e.click(); return true;})()"
            ),
            "returnByValue": True,
        },
    )
    clicked = res.get("result", {}).get("result", {}).get("value") is True
    if not clicked:
        raise RuntimeError(f"Could not find language switcher for locale '{locale}'")
    await asyncio.sleep(0.9)


async def _capture(
    tab,
    *,
    base_url: str,
    shot: Shot,
    device: Device,
    out_dir: Path,
    landscape: bool,
    locale: str,
    landscape_layout_width: Optional[int] = None,
    suffix: str = "portrait",
    desktop: bool = False,
) -> Path:
    """Emulate the device, navigate, and write a full-page PNG."""
    ua = _DESKTOP_UA if desktop else _IPHONE_UA
    if desktop:
        w, h = _DESKTOP_VIEWPORT
        scale = 1
    elif landscape and landscape_layout_width:
        # Pin the LAYOUT viewport to a fixed width (e.g. 1280 / 1920). The height
        # keeps the phone's rotated aspect so the immersive 100dvh chart still
        # reads as a landscape phone. scale=1 keeps the output width == layout
        # width (no 2x blow-up at these already-wide sizes).
        w = landscape_layout_width
        h = max(1, round(landscape_layout_width * device.width / device.height))
        scale = 1
    elif landscape:
        w, h = device.height, device.width
        scale = device.scale
    else:
        w, h = device.width, device.height
        scale = device.scale

    await tab.send_command(
        "Emulation.setTouchEmulationEnabled", {"enabled": True, "maxTouchPoints": 5}
    )
    await tab.send_command("Emulation.setUserAgentOverride", {"userAgent": ua})

    # NOTE: mobile=False everywhere.  The app now serves a responsive
    # `width=device-width` viewport meta on EVERY route (including /prediction --
    # it no longer force-switches to width=1280).  With mobile=False the CDP
    # metrics width IS the layout viewport (deterministic), so pinning width to
    # 360/740/1280/1920 renders the page at exactly that CSS width -- which is
    # precisely how we want to inspect the responsive layout.  With mobile=True
    # Chromium would instead honour the meta and expand/zoom to fit overflowing
    # content, which made landscape /prediction captures flaky.  We still send a
    # phone User-Agent (so the server returns the mobile builders) and enable
    # touch (so `pointer: coarse` / immersive-landscape media queries match).
    is_chart = shot.path == "/prediction"
    metrics = {
        "width": w,
        "height": h,
        "deviceScaleFactor": scale,
        "mobile": False,
        "screenWidth": w,
        "screenHeight": h,
    }
    await tab.send_command("Emulation.setDeviceMetricsOverride", metrics)
    # Optional pre-navigation seeding (e.g. set consent so desktop /startup renders).
    if shot.seed_js:
        await tab.send_command("Page.navigate", {"url": base_url.rstrip("/") + shot.seed_path})
        await _hydrate_wait(tab)
        await asyncio.sleep(1.0)
        await tab.send_command("Runtime.evaluate", {"expression": shot.seed_js, "returnByValue": True})
        await asyncio.sleep(1.0)
    url = base_url.rstrip("/") + shot.path
    await tab.send_command("Page.navigate", {"url": url})
    await _hydrate_wait(tab)
    await asyncio.sleep(shot.settle_s)
    await _apply_locale(tab, locale)

    if shot.pre_click_js:
        await tab.send_command(
            "Runtime.evaluate",
            {"expression": shot.pre_click_js, "returnByValue": True},
        )
        await asyncio.sleep(1.0)

    # Walk wizard steps (or any sequence of clicks) before capturing.  Each
    # click fires a Dash callback (server round-trip), so wait for the DOM to
    # settle between clicks.
    for el_id in shot.clicks:
        await tab.send_command(
            "Runtime.evaluate",
            {
                "expression": (
                    f"(function(){{var e=document.getElementById('{el_id}');"
                    "if(e){e.click();}})()"
                ),
                "returnByValue": True,
            },
        )
        await asyncio.sleep(0.7)

    if shot.pre_capture_js:
        await tab.send_command(
            "Runtime.evaluate",
            {"expression": shot.pre_capture_js, "returnByValue": True},
        )
        await asyncio.sleep(0.4)

    # Plotly only re-fits its SVG on a window `resize` event, not on a
    # CSS-driven container resize (the clientside route-prediction class +
    # immersive-landscape CSS reshape the chart container after Plotly's first
    # render).  A bare window-resize event races the layout, so also call
    # Plotly.Plots.resize() on every plot directly -- without this the
    # landscape chart keeps its initial, oversized height and the x-axis +
    # Submit button fall off the bottom of the capture.
    await tab.send_command(
        "Runtime.evaluate",
        {
            "expression": (
                "(function(){window.dispatchEvent(new Event('resize'));"
                "if(window.Plotly){document.querySelectorAll('.js-plotly-plot')"
                ".forEach(function(g){try{window.Plotly.Plots.resize(g);}catch(e){}});}"
                "return true;})()"
            ),
            "returnByValue": True,
        },
    )
    await asyncio.sleep(0.8)

    # Capture the FULL page without `captureBeyondViewport` (which, under
    # mobile=False, re-lays-out at a ~1280 fallback width and ruins the shot).
    # Instead grow the viewport height to the page's scroll height and take a
    # plain viewport screenshot -- this keeps the true device width.
    #
    # EXCEPTIONS (capture the viewport as-is, don't grow):
    #  - landscape shots: growing height flips orientation back to portrait,
    #    re-triggering the rotate overlay and breaking landscape CSS.
    #  - /prediction: portrait shows a fixed full-screen rotate overlay and
    #    landscape is an immersive viewport-filling chart -- both are viewport-sized.
    if shot.grow_height and not landscape and not is_chart:
        sh = await tab.send_command(
            "Runtime.evaluate",
            {
                "expression": "Math.max(document.body.scrollHeight, document.documentElement.scrollHeight)",
                "returnByValue": True,
            },
        )
        full_h = int(sh.get("result", {}).get("result", {}).get("value") or h)
        full_h = max(h, min(full_h, 20000))  # clamp to a sane maximum
        tall = dict(metrics)
        tall["height"] = full_h
        tall["screenHeight"] = full_h
        await tab.send_command("Emulation.setDeviceMetricsOverride", tall)
        await asyncio.sleep(0.3)

    res = await tab.send_command(
        "Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False}
    )
    data = res.get("result", {}).get("data")
    if not data:
        raise RuntimeError(f"No screenshot data for {url} ({res})")

    out_path = out_dir / f"{shot.label}-{device.name}-{suffix}.png"
    out_path.write_bytes(base64.b64decode(data))
    return out_path


async def _run_group(
    group: ServerGroup, *, base_url: str, device: Device, out_dir: Path, locale: str
) -> list[Path]:
    """Screenshot every shot in a group against an already-running server."""
    from choreographer import Browser
    from choreographer.cli import get_chrome_sync

    chrome = get_chrome_sync()
    written: list[Path] = []
    async with Browser(path=chrome, headless=True) as browser:
        tab = await browser.create_tab("")
        await tab.send_command("Page.enable")
        await tab.send_command("Runtime.enable")
        for shot in group.shots:
            # (suffix, is_landscape, pinned_layout_width). Portrait always; then
            # one landscape capture per pinned width (first width keeps the plain
            # "landscape" suffix; extras get "-{width}").
            suffix0 = "desktop" if group.desktop else "portrait"
            specs: list[tuple[str, bool, Optional[int]]] = [(suffix0, False, None)]
            # Landscape variants are mobile-only (desktop is a single wide capture).
            if shot.landscape and not group.desktop:
                if shot.landscape_widths:
                    for i, lw in enumerate(shot.landscape_widths):
                        specs.append(("landscape" if i == 0 else f"landscape-{lw}", True, lw))
                else:
                    specs.append(("landscape", True, None))
            for suffix, is_landscape, layout_width in specs:
                try:
                    p = await _capture(
                        tab,
                        base_url=base_url,
                        shot=shot,
                        device=device,
                        out_dir=out_dir,
                        landscape=is_landscape,
                        locale=locale,
                        landscape_layout_width=layout_width,
                        suffix=suffix,
                        desktop=group.desktop,
                    )
                    written.append(p)
                    typer.echo(f"  ✓ {p.name}")
                except Exception as exc:  # keep going; one bad page shouldn't abort
                    typer.echo(f"  ✗ {shot.label} ({suffix}): {exc}")
    return written


def _spawn_server(group: ServerGroup, *, port: int, log_path: Path) -> subprocess.Popen:
    """Launch a server group in its own process group, logging to a file."""
    env = {**os.environ, "DASH_DEBUG": "0", "DEBUG_MODE": "0", **group.env}
    log = log_path.open("w")
    return subprocess.Popen(
        group.cmd,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # own process group so we can kill children too
    )


def _port_is_free(port: int) -> bool:
    """True when nothing accepts a TCP connection on the port (i.e. it is free)."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) != 0


def _wait_for_port_free(port: int, timeout_s: float = 20.0) -> None:
    """Block until the port is free (or timeout).

    `uv run start` auto-increments the bind port when the requested one is busy,
    so if the previous group's server is not fully dead the next server quietly
    relocates to port+1 while the harness keeps polling the original port and
    reports "server did not come up". Waiting for the port to free up before the
    next spawn keeps every server on its requested port (this is the root cause of
    the flaky babylon failures on sequential same-port spawns).
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _port_is_free(port):
            return
        time.sleep(0.25)


def _kill_server(proc: subprocess.Popen, *, port: int) -> None:
    """Terminate the server and any stragglers on its port."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=8)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            proc.kill()
    # Belt-and-suspenders: clear anything still bound to the port.
    if shutil.which("fuser"):
        subprocess.run(
            ["fuser", "-k", f"{port}/tcp"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    # Wait for the socket to be released so the next spawn binds this exact port.
    _wait_for_port_free(port)


def main(
    device: str = typer.Option("android-narrow", "--device", "-d", help=f"Viewport preset: {', '.join(DEVICES)}"),
    out: Path = typer.Option(Path("data/output/mobile_shots"), "--out", "-o", help="Output directory for PNGs"),
    only: Optional[str] = typer.Option(None, "--only", help="Only this server group: entry | result | chart"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Use an already-running server instead of spawning one"),
    port: int = typer.Option(8099, "--port", "-p", help="Port to spawn servers on"),
    language_set: str = typer.Option(
        "english",
        "--language-set",
        "--variant",
        help=f"Screenshot language set: {', '.join(LANGUAGE_SETS)}",
    ),
) -> None:
    """Render the app's pages on a narrow phone viewport and save screenshots."""
    if device not in DEVICES:
        typer.echo(f"Unknown device '{device}'. Choices: {', '.join(DEVICES)}")
        raise typer.Exit(1)
    if language_set not in LANGUAGE_SETS:
        typer.echo(f"Unknown language set '{language_set}'. Choices: {', '.join(LANGUAGE_SETS)}")
        raise typer.Exit(1)
    dev = DEVICES[device]
    out_root = Path(out)
    out_root.mkdir(parents=True, exist_ok=True)
    locales = LANGUAGE_SETS[language_set]

    all_written: list[Path] = []

    for locale in locales:
        out_dir = out_root if language_set == "english" else out_root / locale
        out_dir.mkdir(parents=True, exist_ok=True)

        groups = _server_groups(port, locale=locale)
        if only:
            groups = [g for g in groups if g.name == only]
            if not groups:
                typer.echo(f"No group named '{only}'.")
                raise typer.Exit(1)

        if base_url:
            # Caller manages the server; just screenshot every selected group's shots.
            for group in groups:
                typer.echo(f"[{locale}:{group.name}] using {base_url}")
                all_written += asyncio.run(
                    _run_group(group, base_url=base_url, device=dev, out_dir=out_dir, locale=locale)
                )
            continue

        for group in groups:
            typer.echo(f"[{locale}:{group.name}] starting server on :{port} ...")
            log_path = out_dir / f"_server-{group.name}.log"
            proc = _spawn_server(group, port=port, log_path=log_path)
            try:
                url = f"http://127.0.0.1:{port}/"
                if not _wait_for_server(url):
                    typer.echo(f"  ✗ server did not come up (see {log_path})")
                    continue
                all_written += asyncio.run(
                    _run_group(group, base_url=url, device=dev, out_dir=out_dir, locale=locale)
                )
            finally:
                _kill_server(proc, port=port)

    typer.echo(f"\nWrote {len(all_written)} screenshot(s) to {out_root}/")
    for p in all_written:
        typer.echo(f"  {p}")


def cli() -> None:
    """Console-script entry point."""
    typer.run(main)


if __name__ == "__main__":
    cli()
