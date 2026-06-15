# Mobile version

This document describes the mobile-first experience of Sugar Sugar: why it exists,
how it is built, how it differs from the desktop web app, how to test it, and —
most importantly — the pitfalls and lessons learned while building it (June 2026).

If you only read one section, read **[Pitfalls & lessons learned](#pitfalls--lessons-learned)**.

---

## 0. Present-state snapshot (flowpath reference)

A verified baseline of how the mobile/desktop flow is wired as of June 2026, so
future changes have a reference point. All line numbers are approximate.

### Route map (`display_page`, `sugar_sugar/app.py`)

| Pathname | Builder / handler | Device split | Guard |
|---|---|---|---|
| `/` (default) | `_landing_builder` → `LandingPage` (desktop) / `LandingPageMobile` | server UA | — |
| `/consent-form` | `ConsentFormPage` | CSS only | — |
| `/startup` | `_startup_builder` → `StartupPage` / `StartupPageMobile` (6-step wizard) | server UA | **desktop:** requires `consent_completed`, else landing |
| `/prediction` | `create_prediction_layout` | CSS (immersive landscape) | requires `user_info` **and** `consent_completed`, else landing |
| `/ending` | `create_ending_layout` | CSS | needs `full_df` + `prediction_table_data`, else session-expired |
| `/final` | `create_final_layout` | CSS | needs `user_info`, else session-expired |
| `/share/<id>` | `create_share_layout` (from disk record) | CSS (`.share-page`) | record must exist, else `create_expired_layout` |
| `/about` `/faq` `/contact` `/demo` | info-page builders | CSS | — |
| `/staging`, `/staging/{ending,final,share,prediction}` | `_staging_display` / staging routes | CSS | **only when `_STAGING_MODE=1`** (else fall through) |

Builder selection is **server-side** (`_is_mobile_request()` reads the live Flask
request User-Agent). Only **one** builder (mobile *or* desktop) is ever in the DOM
per request — so the two consent component trees (landing vs wizard step 0) never
coexist; the "duplicate consent ids" worry is moot.

### Consent enforcement points

1. **Desktop landing `/`** — `update_continue_button` gates the Continue button on
   scroll+ack+gdpr; `handle_landing_continue` (`landing.py:295-416`) writes consent +
   `consent_completed=True` and redirects to `/startup`.
2. **Mobile wizard step 0** — `gate_mobile_consent_step` (`startup.py:661-682`)
   disables `startup-next` while on step 0 until scroll+ack+gdpr. You cannot advance
   past step 0 without consent, and returning to step 0 re-applies the gate.
3. **The real gate — `handle_start_button`** (`app.py:3656-3779`) — on mobile
   (`has_mobile_consent` True, because the wizard renders the consent fields) it
   re-checks ack+gdpr and writes `consent_completed=True`. On desktop the consent
   fields are absent (`has_mobile_consent` False) so it trusts the landing-page gate.
4. **`display_page` consent guard** (`app.py`) — closes the residual desktop hole:
   `/prediction` (both devices) and desktop `/startup` require `consent_completed` in
   `user_info`; without it they render the landing/consent page. This stops a
   direct-URL / burger-menu visit from reaching the game unconsented.

Chart mode's synthetic user and every staging node set `consent_completed=True` so the
guard lets them through.

### Cross-device matrix

| Scenario | Works? | Why |
|---|---|---|
| Share on desktop → open link on mobile (and vice-versa) | **Yes** | `/share/<id>` renders from `data/shares/<id>.json` on disk; responsive CSS; crawler OG via `/share/<id>/og`; the 1200×630 card is device-independent |
| Resume a game session: mobile → desktop (or vice-versa) | **Yes, via resume code** | Session state is still `STORAGE_TYPE=local` (per-device), but a server-side snapshot (`resume_store.py`, `data/resume/<code>.json`) keyed by a short `resume_code` is auto-saved at every meaningful boundary. Re-enter the code on another device (`?resume=<code>` URL, the landing-page "resume code" box, or read it off the resume dialog) to restore the session. See "Cross-device resume" below. |
| Same-device, close tab, reopen later | Yes | localStorage persists; `restore_page_on_load` shows the resume dialog |

### Screenshot coverage matrix (`scripts/mobile_shots.py`)

| Group | Server | Pages |
|---|---|---|
| `entry` | `uv run start` | `/`, `/consent-form` (top+bottom), `/startup` step 1–6, `/about`, `/faq`, `/contact`, `/demo` |
| `result` | `uv run start` + `_STAGING_MODE=1` | `/staging/ending`, `/staging/final`, `/staging/share` (→ `/share/<id>`) |
| `chart` | `uv run chart --prefill` | `/prediction` portrait + landscape |

Together these cover every onboarding, gameplay, result, and share surface in at least
one form.

### Cross-device resume

localStorage is per-device, so the game does not follow a user to another device on its
own. `sugar_sugar/resume_store.py` bridges that with a server-side snapshot:

- **Code** — a short, human-typeable `resume_code` (no ambiguous glyphs) is assigned to
  `user_info` at consent (`handle_landing_continue` desktop, `handle_start_button`
  mobile).
- **Auto-save** — `auto_snapshot_session` writes `data/resume/<code>.json` (atomic, like
  `share_store`) at meaningful boundaries: it triggers on `user-info-store`,
  `last-visited-page`, `glucose-unit`, `interface-language` changes and pulls the
  dataframes in via `State`, so it does **not** fire on every in-progress drawline.
- **Redeem** — three entry points, all calling `_restore_outputs_from_code`:
  `?resume=<code>` on any URL (the universal link; works on first paint via
  `initial_duplicate`), the "resume code" box on the landing page (fresh-device entry),
  and the code displayed on the resume dialog for returning users. After a URL redeem a
  clientside callback strips `?resume=` from the address bar.
- **Payload** — `_resume_payload` snapshots `user_info`, `full_df`,
  `current_window_df`, `events_df`, `last_visited_page`, `glucose_unit`,
  `interface_language`. Add any new game-state store here AND in
  `_restore_outputs_from_code`, or it won't transfer.

Treat resume codes as session-transfer tokens (anyone with the code resumes the
session) — like a login link, not a public id. `data/resume/` is gitignored.

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
  - Separate mobile builders: **landing entry**, **startup + consent wizard**, **navbar (burger)**.
  - Responsive CSS only: ending, final, share, faq, about, contact, demo.
  - `/prediction`: CSS immersive landscape + a Start-button fullscreen/landscape-lock
    request (the portrait rotate-nag overlay was removed).
- **`width=device-width` by default, `1280` only on `/prediction`.** Mobile-first means
  the default viewport is the real device width; the chart page is the single
  exception because Plotly's `drawline` needs horizontal space and is unusable at
  ~390px portrait width.
- **Immersive landscape is the single mobile drawing mode.** The old landscape plaque
  blocked the whole site; portrait now works everywhere except drawing. On
  `/prediction` the user is in landscape (the wizard Start button best-effort enters
  fullscreen + landscape lock; otherwise they rotate) for an **immersive full-screen
  chart** (chrome collapses, controls pinned within `100dvh`). There is no portrait
  rotate-nag overlay — it was a second, non-playable mode. We never CSS-rotate the
  chart — it breaks touch mapping.
- **Wizard for consent + startup.** 11+ inputs on one page means the keyboard hides the
  active field. Mobile now keeps the landing page short and puts consent in the
  wizard as mandatory step 1, followed by the startup form steps (1–3 fields per
  step) so the keyboard never covers what you're typing.

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
  Output elements) plus the shared consent checklist ids in `mobile-step-{0..5}`
  divs. Step 0 reuses `consent_controls_children()` inside `#consent-notice-scroll`,
  and `startup-next.disabled` is gated **only on the two mandatory consent checkboxes**
  (acknowledge + GDPR). It is deliberately **not** gated on scroll-to-end: that
  detection watches the outer div, but on real mobile browsers the user scrolls the
  inner consent iframe, so the div never registers "scrolled" and the user got
  hard-locked with a dead Next button even after ticking both boxes (see pitfalls).
  `navigate_startup_wizard`
  (registered inside `StartupPage.register_callbacks`, `prevent_initial_call=True`)
  toggles each step's `display` and the Back/Next buttons + progress dots. Conditional
  parents live in the SAME step as their dependents (CGM→duration, diabetic→type+
  duration, format B/C→data-usage-consent) so a hidden step never strands a
  half-revealed cascade. The existing validation/conditional callbacks are unchanged.
- **`LandingPageMobile`** (`components/landing.py`): short single-column entry page:
  hero / how-it-works, about-the-study summary, and one full-width "Take me in" link
  to `/startup`. It deliberately does not render consent ids; mobile consent is
  non-skippable because the wizard's first Next button is the gate and the final
  Start button is the only route to `/prediction`.

### 3.4 CSS (`assets/mobile.css`, `assets/orientation.css`)

- `mobile.css` is scoped under `html.mobile-device`. Global rules release the
  `min-width:1280` anchor, cap form controls, and style the burger navbar + drawer.
  The 1280-scaling compensations (chart fonts, blanket text bump, immersive landscape)
  are scoped to `html.route-prediction` only.
- `orientation.css` is **retired** — the portrait "rotate to draw" nag overlay was
  removed (it was a second, non-playable mode the user could only dismiss). The single
  mobile `/prediction` flowpath is now: the immersive landscape CSS applies the moment
  the phone is in landscape, and the wizard's Start button best-effort requests
  fullscreen + landscape lock (clientside, mobile-only) so the user lands straight in
  the immersive chart. Immersive landscape is gated on
  `@media (orientation: landscape) and (pointer: coarse)` (NO `max-device-width` — see
  pitfalls) and uses `100dvh` (not `100vh`) so the bottom controls stay on-screen under
  the browser chrome.

### 3.5 Screenshot harness (`scripts/mobile_shots.py`)

Renders every mobile-relevant page on a narrow phone viewport and saves PNGs to
`data/output/mobile_shots/`, so visual artifacts can be caught without deploying to
staging. It drives Chromium over the DevTools Protocol via `choreographer` (a transitive
dependency of Plotly's kaleido), sets a mobile Safari User-Agent so server-side mobile
builders are selected, enables touch emulation so coarse-pointer CSS applies, and then
uses carefully chosen device-metrics overrides per page.

The harness is intentionally split into three server groups:

- **`entry`** starts `uv run start --port <port>` and captures display/onboarding pages:
  landing (`/`), consent form top and bottom (`/consent-form`), all six startup wizard
  states (`/startup` plus repeated `startup-next` clicks), about, FAQ, contact, and demo.
- **`result`** starts `uv run start --port <port>` with `_STAGING_MODE=1` and captures
  the result/share surfaces via the prod+ staging nodes: `/staging/ending`,
  `/staging/final`, and `/staging/share` (which 302-redirects to a freshly generated
  `/share/<id>` whose synthetic record cycles formats A/B/C, exercising the multi-panel
  synthesis graph). This is the deterministic way to screenshot `/ending`, `/final`, and
  `/share` without a full playthrough — the project forbids LLM click-through, so the
  staging nodes replace it. (Without `_STAGING_MODE` these routes don't exist.)
- **`chart`** starts `uv run chart --prefill --no-debug --no-reloader --locale <locale>
  --port <port>` and captures `/prediction` in portrait and landscape. `--prefill`
  avoids browser automation for drawing predictions, and `--no-reloader` avoids the
  Werkzeug fork losing chart-mode environment.

Device presets:

- `android-narrow` (default): 360x740, the torture-test phone width.
- `iphone-se`: 320x568.
- `iphone-13`: 390x844.
- `pixel-7`: 412x915.

Language sets:

- `english` (default): preserves the historic behaviour. It renders English only and
  writes flat files directly under `data/output/mobile_shots/`, e.g.
  `prediction-android-narrow-landscape.png`.
- `babylon`: renders every supported locale (`en`, `de`, `uk`, `ro`, `ru`, `zh`, `fr`,
  `es`) and writes one folder per language, e.g. `data/output/mobile_shots/ro/*.png`.
  The normal pages switch locale by clicking the existing `lang-<code>` navbar element;
  the chart server also receives `--locale <code>` at startup.

Common commands:

```bash
uv run python scripts/mobile_shots.py
uv run python scripts/mobile_shots.py --only chart
uv run python scripts/mobile_shots.py --device iphone-se
uv run python scripts/mobile_shots.py --language-set babylon
uv run python scripts/mobile_shots.py --language-set babylon --only entry --port 8101
uv run python scripts/mobile_shots.py --base-url http://127.0.0.1:8050
uv run python scripts/mobile_shots.py --out /tmp/mobile-shots
```

The important options are:

- `--device` / `-d`: choose a viewport preset.
- `--only entry|chart`: run one server group.
- `--language-set english|babylon` (alias `--variant`): choose default English-only or
  all-locales output.
- `--out` / `-o`: choose the output root.
- `--port` / `-p`: choose the port used for spawned servers.
- `--base-url`: use an already-running server instead of spawning one. This is useful
  when debugging a single page manually, but remember that the spawned-mode chart group
  normally starts with `uv run chart --prefill`.

Each spawned group writes `_server-entry.log` or `_server-chart.log` beside its PNGs.
If a shot fails, the harness logs the failure and keeps going so one broken page does not
erase evidence from the rest of the run. See the harness notes in
[Pitfalls](#pitfalls--lessons-learned) for the non-obvious CDP emulation choices.

## 4. Differences from the desktop web app

| Aspect | Desktop | Mobile |
|---|---|---|
| Layout viewport | `device-width` (meta ignored by desktop browsers) | `device-width`; `1280` only on `/prediction` |
| Navbar | Fomantic `massive tabular menu` (one row) | `MobileNavBar` burger + drawer |
| Startup form | One long `StartupPage` | `StartupPageMobile` 6-step wizard, starting with consent |
| Landing/consent | Two-column hero + consent card | Short `LandingPageMobile` entry; consent is wizard step 1 |
| `/prediction` portrait | Chart | Chart (no overlay); rotate or use Start-button fullscreen for the immersive landscape |
| `/prediction` landscape | Chart + full chrome | Immersive: chrome hidden, chart fills screen |
| `min-width:1280` (lang.css) | Active | Released except on `/prediction` |
| Display pages | As-is | Reflow to single column via CSS |

## 5. Testing

- `uv run python scripts/mobile_shots.py` — screenshots all pages (entry + result +
  chart groups) at 360px. `uv run python scripts/mobile_shots.py --only chart --port
  <free>` for just the prediction page (portrait chart + landscape immersive);
  `--only result` for the ending/final/share pages (via the staging nodes). Run the
  groups on different ports if the OS is slow to release the socket.
- `uv run python scripts/mobile_shots.py --language-set babylon` — screenshots the same
  page set in every supported language under `data/output/mobile_shots/<locale>/`. Use
  this before shipping translation, navbar, consent, startup, or share/display-page
  layout changes; long German/Romanian/Spanish labels and CJK wrapping catch different
  bugs from English.
- `uv run chart --prefill` — manual prediction/submit/ending flow.
- Real-device check still recommended for touch drawing and keyboard behaviour, which
  the harness cannot fully reproduce.
- For consent changes, manually verify the mobile path **on a real touch device**
  (the headless harness cannot reproduce iframe touch-scroll): `/` → "Take me in" →
  `/startup`; the Step 1 Next button stays disabled until **both required boxes**
  (acknowledge + GDPR) are ticked, then enables (no scroll-to-end requirement on
  mobile — see pitfalls); completing the wizard reaches `/prediction` and writes a
  consent agreement row.

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

- **New callbacks for mobile-only ids go in the desktop component's `register_callbacks`**
  so they register once and stay inert on desktop (where the ids are absent). Use
  `prevent_initial_call=True` for navigation callbacks whose initial state is baked
  into the layout; use `prevent_initial_call=False` only for initial gates that must
  be active immediately, such as the consent step disabling `startup-next`. Toggle only
  the `mobile-step-{i}` wrappers — never the conditional sub-sections (`cgm-details`
  etc.) that already own their own `display`, or the two callbacks fight.

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

- **Mobile landscape chart controls should reclaim every non-drawing pixel.** Treat the
  short edge as the budget: remove unneeded wording ("Glucose Levels", repeated
  "Source:" labels, standalone "Ready to submit!" status text), move metadata into the
  bottom control strip, and group related controls into compact clusters. Prefer a
  single bottom row with source/time, units, Submit, and Finish rather than stacked
  rows. Keep visual heights matched across chips/buttons, use subtle borders or bevels
  to delimit non-button metadata, and make fonts as large as comfortably possible after
  the row still fits on the narrow landscape harness.

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

- **Immersive landscape must NOT be gated on `max-device-width`.** The immersive
  `/prediction` landscape CSS was gated on
  `@media (orientation: landscape) and (pointer: coarse) and (max-device-width: 1024px)`.
  On a real high-DPI phone the landscape device-width can EXCEED 1024 (and this page
  forces a 1280 layout viewport), so the query failed on rotation and the immersive
  layout never applied — the chart stayed full-size and Submit/Finish fell off the
  bottom (reported: "turning the phone doesn't trigger immersive"; the controls were
  unreachable because the chart eats touch-scroll). The `html.mobile-device` selector
  scope already restricts this to phones, so `max-device-width` was redundant AND the
  bug. Gate on `(orientation: landscape) and (pointer: coarse)` only.

- **Use `100dvh`, not `100vh`, for the immersive shell.** `100vh` is the chrome-excluded
  tall viewport, so with the address bar visible the bottom control strip sits below the
  fold and `overflow:hidden` blocks scrolling to it. `100dvh` (dynamic viewport height)
  tracks the actually-visible area and keeps the controls on-screen. Fullscreen (which
  hides chrome) makes `vh`/`dvh` equal, but we can't rely on fullscreen always engaging.

- **Fullscreen needs a user-gesture, so trigger it from the entering CLICK.** A
  `requestFullscreen` from a route-change/store callback is rejected (no user
  activation). It's wired to the wizard Start-button `n_clicks` (clientside, gated on
  the `mobile-device` class). Burger "Game" → `/` → redirect loses the gesture, so that
  path relies on the (now playable) `100dvh` landscape layout + manual rotate instead.
  `screen.orientation.lock('landscape')` works on Android Chrome/Vivaldi after
  fullscreen and rejects on iOS Safari (caught; user rotates manually).

- **Mobile buttons/links need `touch-action: manipulation` or taps get swallowed.**
  The viewport allows zoom (`user-scalable=yes, maximum-scale=5`), so mobile browsers
  wait ~300 ms after each tap for a possible double-tap-zoom and treat rapid taps as
  zoom gestures — which **drops clicks**, so a button or link only fires after several
  taps (reported on Vivaldi Android: the wizard Next "worked on the 4th click", the
  landing link "fired on the third tap"). It is NOT a Dash/callback bug and does NOT
  reproduce in the headless harness (synthetic `.click()`/touch events bypass the
  gesture wait). Fix: `touch-action: manipulation` on `a`/`button`/`.ui.button`/
  `[role=button]`/`label`/`input`/`.form-check` (scoped to
  `html.mobile-device:not(.route-prediction)`; the chart manages its own
  `touch-action` for drawline). Pinch-zoom is preserved.

- **Don't CSS-rotate the chart** (`transform: rotate(90deg)`) — it desyncs Plotly's
  drawline touch coordinates. The immersive landscape is achieved by hiding chrome and
  letting the native chart fill the viewport, not by rotation. **Don't use
  `screen.orientation.lock()`** — it needs fullscreen and is unsupported on iOS Safari.

- **Consent enforcement is asymmetric between desktop and mobile, and the Start
  handler can't see it.** `handle_start_button` only re-checks consent when the consent
  *fields* are present (`has_mobile_consent`), which is true on mobile (wizard step 0
  renders them) but false on desktop (consent lives on the landing page). So a
  direct-URL / burger visit to desktop `/startup` — bypassing the landing gate — would
  reach `/prediction` unconsented. The fix is the `display_page` guard on the
  `consent_completed` flag (`/prediction` both devices; `/startup` desktop only — mobile
  `/startup` IS the consent entry and must stay reachable). **If you add a new synthetic
  user path (chart mode, a staging node, a test), set `consent_completed=True` or the
  guard will bounce it to landing.**

- **localStorage is device-local; cross-device resume goes through the resume code.**
  Every session store is `STORAGE_TYPE=local` (per-device), so nothing follows the user
  to another device automatically. The bridge is `resume_store.py`: a server-side
  snapshot keyed by `user_info['resume_code']`, auto-saved at meaningful boundaries
  (`auto_snapshot_session`, triggered by user_info / navigation / unit / language —
  **not** every drawline; the dataframes come in via `State`). Redeem on another device
  via `?resume=<code>`, the landing-page box, or the code shown on the resume dialog.
  **If you add a new game-state store, add it to the resume payload** (`_resume_payload`
  / `_restore_outputs_from_code`) or it won't transfer.

- **Screenshot `/ending`, `/final`, `/share` via the staging nodes, not click-through.**
  The harness `result` group runs `uv run start` with `_STAGING_MODE=1` and hits
  `/staging/ending`, `/staging/final`, `/staging/share`. These render the *real*
  builders with synthetic data, so they're deterministic and need no drawing/submit
  automation. The synthetic rounds intentionally don't fill the per-round metrics table
  on `/final` (or "Prediction Results" on `/ending`) — those boxes look empty in the
  shots; that's a data quirk, not a layout bug. The aggregate metrics, charts, share
  panels, and buttons all render and are what these shots verify.
