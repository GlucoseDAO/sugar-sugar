# Mobile version

This document describes the mobile-first experience of Sugar Sugar: why it exists,
how it is built, how it differs from the desktop web app, how to test it, and —
most importantly — the pitfalls and lessons learned while building it (June 2026).

If you only read one section, read **[Pitfalls & lessons learned](#pitfalls--lessons-learned)**.

---

## 1. Why this exists (the problem)

The app was originally desktop-only: it forced a fixed `width=1280` layout viewport
on every device (`meta_tags` on the `Dash()` constructor) so phones rendered the
desktop layout scaled down, like "Request desktop site". On real phones this produced
four reported failures:

1. **Portrait was blocked entirely** by a full-screen "PLEASE USE LANDSCAPE MODE"
   plaque — the site was unusable in the orientation people actually hold phones in.
2. **Landscape wasted its limited height** — the navbar/footer/controls stretched to
   1280-scaled widths and ate the short edge, leaving a thin chart band.
3. **The on-screen keyboard covered ~80% of the form pages** (startup, consent), hiding
   the active input/dropdown.
4. The overall impression was "this site doesn't work on mobile".

The goal: a genuine mobile-first experience that **works in portrait** for every page
except the one step that physically needs width — line-drawing on the glucose chart.

## 2. Design choices

- **Hybrid architecture (not pure CSS).** CSS can only rearrange/resize/hide elements
  that already exist; it cannot build a multi-step wizard, a burger menu, or any
  stateful interaction. So pages that need *structurally different* markup get
  **separate mobile builders** (new Python + callbacks); display-only pages get
  **responsive CSS** on the existing builders. The split:
  - Separate mobile builders: **landing/consent**, **startup (wizard)**, **navbar (burger)**.
  - Responsive CSS only: ending, final, share, faq, about, contact, demo.
  - `/prediction`: mostly CSS (immersive landscape) + a portrait rotate prompt.
- **`width=device-width` by default, `1280` only on `/prediction`.** Mobile-first means
  the default viewport is the real device width; the chart page is the single
  exception because Plotly's `drawline` needs horizontal space and is unusable at
  ~390px portrait width.
- **Rotate-to-draw, not rotate-to-use.** The old landscape plaque blocked the whole
  site. Now portrait works everywhere; the rotate prompt appears **only on
  `/prediction`**, where the user rotates to landscape for an **immersive full-screen
  chart** (chrome collapses). We never CSS-rotate the chart — it breaks touch mapping.
- **Wizard for the startup form.** 11+ inputs on one page means the keyboard hides the
  active field. The mobile startup is a 5-step wizard (1–3 fields per step) so the
  keyboard never covers what you're typing.

## 3. Implementation

### 3.1 Viewport + route class (`sugar_sugar/app.py`)

- Static meta: `width=device-width, initial-scale=1, maximum-scale=5, user-scalable=yes`.
- A clientside callback keyed on `url.pathname` (a) adds/removes `route-prediction` on
  `<html>` and (b) rewrites the `<meta viewport>` content to `width=1280` on
  `/prediction`, `width=device-width` elsewhere. Output is a throwaway `viewport-sink`
  div. `prevent_initial_call=False` so it runs on first load. Bump `DEPLOY_BUILD` in
  `config.py` when changing this JS (clientside JS is not fingerprinted, so browsers
  cache it across restarts).

### 3.2 Device detection + builder selection

- `_is_mobile_ua(ua)` / `_is_mobile_request()` read the live Flask request User-Agent
  (request-scoped, correct on first render). `display_page` and
  `update_on_language_change` call `_startup_builder()`, `_landing_builder()`, and
  `_navbar()` which return the mobile or desktop variant.
- The existing clientside callback still adds `html.mobile-device` (UA keywords +
  `(pointer:coarse) and (max-device-width:1024px)` fallback). This class scopes all of
  `assets/mobile.css`. The two detectors are intentionally separate: server-side UA
  picks the *layout*; the class drives *CSS*.

### 3.3 Mobile builders

- **`MobileNavBar`** (`components/navbar.py`): a compact bar — burger button (`☰`,
  `mobile-nav-toggle`) + title + language dropdown — plus a hidden `mobile-nav-drawer`
  listing the 5 destinations. A clientside callback toggles the drawer (n_clicks
  parity); navigating via `dcc.Link` re-renders the navbar fresh, which closes the
  drawer automatically. The shared `build_language_dropdown(locale)` is reused by both
  navbars.
- **`StartupPageMobile`** (`components/startup.py`): renders **every** input id of the
  desktop `StartupPage` (same ids, same persistence, plus all `*-required`/`*-error`
  Output elements) grouped into `mobile-step-{0..4}` divs. `navigate_startup_wizard`
  (registered inside `StartupPage.register_callbacks`, `prevent_initial_call=True`)
  toggles each step's `display` and the Back/Next buttons + progress dots. Conditional
  parents live in the SAME step as their dependents (CGM→duration, diabetic→type+
  duration, format B/C→data-usage-consent) so a hidden step never strands a
  half-revealed cascade. The existing validation/conditional callbacks are unchanged.
- **`LandingPageMobile`** (`components/landing.py`): single-column landing + consent
  reusing `consent-notice-scroll`, the six consent checklist ids, `landing-error`,
  `landing-continue`, and the scroll-poll store/interval. The consent iframe owns the
  only scrollbar (single-scrollbar rule). `consent_controls_children(locale)` is shared
  with the desktop builder so the consent controls never drift.

### 3.4 CSS (`assets/mobile.css`, `assets/orientation.css`)

- `mobile.css` is scoped under `html.mobile-device`. Global rules release the
  `min-width:1280` anchor, cap form controls, and style the burger navbar + drawer.
  The 1280-scaling compensations (chart fonts, blanket text bump, immersive landscape)
  are scoped to `html.route-prediction` only.
- `orientation.css` overlay is scoped to `html.route-prediction` — a responsive
  "rotate to draw" prompt (`font-size: clamp(34px, 11vw, 150px)`), portrait + coarse
  pointer only.

### 3.5 Screenshot harness (`scripts/mobile_shots.py`)

Renders every page on a narrow phone viewport (and `/prediction` also in landscape) and
saves full-page PNGs to `data/output/mobile_shots/`, so visual artifacts can be caught
without deploying to staging. It drives Chromium over the DevTools Protocol via
`choreographer` (a transitive dep of Plotly's kaleido). Options: `--device`,
`--only entry|chart`, `--base-url`, `--port`, `--out`. See the harness notes in
[Pitfalls](#pitfalls--lessons-learned) for the non-obvious CDP emulation choices.

## 4. Differences from the desktop web app

| Aspect | Desktop | Mobile |
|---|---|---|
| Layout viewport | `device-width` (meta ignored by desktop browsers) | `device-width`; `1280` only on `/prediction` |
| Navbar | Fomantic `massive tabular menu` (one row) | `MobileNavBar` burger + drawer |
| Startup form | One long `StartupPage` | `StartupPageMobile` 5-step wizard |
| Landing/consent | Two-column hero + card | `LandingPageMobile` single column |
| `/prediction` portrait | Chart | "Rotate to draw" overlay |
| `/prediction` landscape | Chart + full chrome | Immersive: chrome hidden, chart fills screen |
| `min-width:1280` (lang.css) | Active | Released except on `/prediction` |
| Display pages | As-is | Reflow to single column via CSS |

## 5. Testing

- `uv run python scripts/mobile_shots.py` — screenshots all pages (entry group + chart
  group) at 360px. `uv run python scripts/mobile_shots.py --only chart --port <free>`
  for just the prediction page (portrait overlay + landscape immersive). Run the two
  groups on different ports if the OS is slow to release the socket.
- `uv run chart --prefill` — manual prediction/submit/ending flow.
- Real-device check still recommended for touch drawing and keyboard behaviour, which
  the harness cannot fully reproduce.

---

## Pitfalls & lessons learned

These are the traps that cost the most time. The same list is mirrored in `CLAUDE.md`.

- **`html, body { min-width: 1280px }` in `assets/lang.css` is the master kill-switch.**
  A leftover from the forced-desktop strategy, it silently pins the entire page to 1280
  regardless of the viewport meta. Released in `mobile.css` via `min-width:0 !important`
  on `html.mobile-device:not(.route-prediction)`. **If mobile pages render desktop-width
  again, check this first.** This single rule was the root cause behind hours of
  confusion where the navbar, inputs, and everything else "inherited" 1280.

- **Under `width=device-width`, any element wider than the screen makes the browser
  expand the layout viewport to fit it and zoom the whole page out.** Real phones do
  this too (Safari/Chrome), not just Chromium. So one overflowing element breaks
  mobile-first for the WHOLE page, and `width:100%` on inner elements is futile because
  `100%` resolves against the already-expanded width. Mobile work is largely a hunt for
  overflow sources, from the outside in.

- **CSS alone cannot fix structural problems.** The `massive tabular menu` navbar is
  ~1280px in one row and cannot be squeezed to fit — it needs a different component
  (`MobileNavBar`). A multi-step wizard, a burger drawer, and added inputs are all
  state/markup, not CSS. This is why the architecture is hybrid, not "responsive CSS
  everywhere".

- **`dcc.Input` applies the Python `style=` to the wrapper `div.dash-input-container`,
  not the real `<input class="dash-input-element">`.** The inner input keeps the classic
  chriddyp Dash CSS (wide default + its own border), so it renders as an inset
  double-box and overflows. Fix: style `.dash-input-element` via CSS and flatten the
  wrapper (`padding:0; border:none`).

- **Mobile builders MUST render every id their callbacks target** (same failure class as
  the documented `ending-*` rule — a missing Output id crashes the callback at runtime,
  and `suppress_callback_exceptions` does NOT save you there). The wizard keeps all
  inputs mounted in `display:none` step divs rather than conditionally rendering only
  the active step (which would also lose `persistence`).

- **New callbacks for mobile-only ids go in the desktop component's `register_callbacks`
  with `prevent_initial_call=True`** so they register once and stay inert on desktop
  (where the ids are absent). Toggle only the `mobile-step-{i}` wrappers — never the
  conditional sub-sections (`cgm-details` etc.) that already own their own `display`,
  or the two callbacks fight.

- **Server-side UA detection (`flask_request.headers`) must choose the layout, not the
  `user-agent` dcc.Store** — the store hydrates async from localStorage and is `None` or
  stale on the first render.

- **Per-page viewport switching is clientside (rewrite `<meta viewport>` content).** It
  works on real browsers but won't relayout while a CDP device-metrics override is
  active (a harness-only quirk). Bump `DEPLOY_BUILD` when changing clientside JS.

- **Screenshot-harness CDP emulation has sharp edges:**
  - Use `mobile:false` for device-width pages — it makes the metrics width the literal
    layout viewport (deterministic). `mobile:true` honours the meta AND triggers the
    flaky expand-to-fit behaviour, so the same page can come out 360 or 1280 between
    runs.
  - Use `mobile:true` for `/prediction` in **landscape** so the `1280` meta is honoured
    and scaled (matches a real phone's immersive chart); use `mobile:false` for
    `/prediction` **portrait** so the fixed full-screen overlay is centred in the 360
    viewport (with `mobile:true` it centres in 1280 and lands off the crop).
  - **Landscape `/prediction` needs a metrics re-apply, or it crops the chart bottom.**
    The `1280` meta switch is clientside and fires *after* first paint; under a CDP
    device-metrics override Chromium does NOT re-fit the page scale on that swap, so the
    capture shows the unscaled top-left 740px slice of the 1280 layout and the x-axis +
    Submit fall off the bottom. Fix: after hydration+settle, `Emulation.clearDeviceMetricsOverride`
    then re-apply the **same** `mobile:true` 740×360 metrics — this forces a full
    re-emulation that re-reads the now-`1280` meta and scales the whole layout into the
    device viewport. **Do NOT "fix" it by setting `screenWidth:1280`/`mobile:false`** —
    that makes *device-width* 1280, which fails the `max-device-width:1024` gate on the
    immersive landscape CSS, so the navbar + instructions reappear and the chart drops
    below the fold. Device-width must stay ≤1024; only the *layout* viewport is 1280.
  - **Plotly does not re-fit on a CSS-driven container resize, only on a window
    `resize`.** A bare `window.dispatchEvent(new Event('resize'))` races the layout, so
    also call `Plotly.Plots.resize(gd)` on every `.js-plotly-plot` just before capturing
    the chart; otherwise the SVG keeps its initial (oversized) height.
  - **`uv run chart` runs Dash with the debug reloader, whose forked child re-imports the
    module and loses the chart-mode prefill** — so `/prediction` intermittently redirects
    to the landing page and the harness captures the wrong page. If the chart shots show
    the landing/consent content, this (not the CSS) is why; ensure the chart entry point
    keeps the prefill alive across the reloader fork (env-var pattern), as with `--prefill`.
  - **Do NOT use `captureBeyondViewport:true`** — under `mobile:false` it re-lays-out at
    a ~1280 fallback and ruins the shot. Instead grow the viewport *height* to the
    page's `scrollHeight` and take a normal viewport capture. Skip the height-grow for
    landscape and `/prediction` (growing height flips orientation to portrait, breaking
    landscape CSS and the overlay; their content is viewport-sized anyway).

- **Number inputs: hide Dash's own `.dash-input-stepper` buttons on mobile, NOT the
  native webkit spinner.** Newer Dash `dcc.Input(type="number")` renders its own `−`/`+`
  stepper `<button>`s *inside* the `.dash-input-container`. Once the input is forced
  full-width (`display:block`) on mobile, those buttons wrap to a second line and render
  as stray `−`/`+` controls below the field. The a11y tree labels them "Decrease value" /
  "Increase value", which looks like a native spinner — but `::-webkit-inner-spin-button {
  appearance:none }` will NOT remove them. Hide `html.mobile-device .dash-input-stepper {
  display:none }` (the touch numeric keypad is the input method anyway). **Don't set
  `overflow:hidden` on the wrapper to clip them** — that re-clips the input's own bottom
  border (the original "editor boxes cropped on the bottom" bug).

- **Don't let the generic `html.mobile-device input { display:block; width:100% }` rule
  hit checkboxes/radios.** It stretches consent checkboxes to full width and pushes their
  label to the next line. Exclude them
  (`input:not([type="checkbox"]):not([type="radio"])`) and lay each `.form-check` out as a
  `display:flex` row so the fixed-size box stays inline with its wrapping label.

- **Consent reader (`/consent-form`): don't use a `height:100vh` flex shell.** `100vh`
  ignores the navbar above `#page-content`, so the shell overflows by the navbar height
  and the "Go to start" button is pushed below the fold (and a `min-height:100vh` shell
  adds a second, page-level scrollbar — the recurring double-scrollbar bug). Instead let
  the shell be normal flow and give the embedded iframe a reserved height
  (`calc(100vh - 190px)`, room for navbar + button + paddings) so the iframe owns the
  only scrollbar and the button stays on-screen. This is the full-bleed, single-box,
  TOS/EULA-style layout (no nested inner card).

- **Contact links: stack the tables into one column on mobile.** Wide multi-column tables
  truncate long emails/URLs into 1–2 character dangling overhangs that read as
  unprofessional. Collapse `thead`/`tr`/`td` to `display:block; width:100%`, bold the
  first cell as a heading, and apply a narrower font with `overflow-wrap:anywhere;
  word-break:normal` to links so they wrap cleanly at full width.

- **Don't CSS-rotate the chart** (`transform: rotate(90deg)`) — it desyncs Plotly's
  drawline touch coordinates. The immersive landscape is achieved by hiding chrome and
  letting the native chart fill the viewport, not by rotation. **Don't use
  `screen.orientation.lock()`** — it needs fullscreen and is unsupported on iOS Safari.
