## Project overview

This project is a Sugar-Sugar game where the user gets the glucose value for some timespan and has to predict by drawing the lines on the chart. He is given a sequence that he has to prolong. The aim of the study is to measure human accuracy of the glucose predictions.
The project is a DASH app, with app.py being main, while glucose, metrics, prediction and startup are dash components. I has default example csv file to play and debug with but provide an option to upload your own csv files from dexcom, libre and other CGM-s.
We use session storage to allow multiple users workin on the same app. Predictions are stored in polars dataframe, there is also a dataframe for current prediction window and scrolling positions.
When the user draws the line it interpolates the position to detect closes glucose and time value (time measurements are done every 5 minutes) and then updates the dataframe with the prediction values.

## Build and test commands

uv is used as the package manager for the project.
uv run start is used to run the dash app.
uv run chart is the fast dev shortcut: it starts Dash with data pre-loaded and routes straight to the prediction chart (bypasses landing, startup, and consent). Use this whenever the user asks to debug or test the chart in the browser. Only fall back to uv run start when the user explicitly needs the startup/landing/consent screens. uv run chart accepts --file, --points, --start, --unit, --locale, --host, --port options. Use --prefill to pre-fill the prediction region with noisy ground-truth values so the submit/ending/metrics flow can be tested without drawing (--noise controls the noise level, default 5%). Always prefer uv run chart --prefill over attempting browser automation for testing submit or ending pages.
uv run share is the share page dev shortcut: it generates fake rounds with synthetic prediction data, saves a share record to disk, and opens the browser at `/share/<id>`. Use this whenever debugging the share page, share card PNG, OG tags, social buttons, or share-related styling. Accepts --port, --locale, --host, --formats, --rounds options. Use `--formats "A,B,C"` to simulate multi-format play (Generic + My Data + Mixed); rounds cycle through formats evenly. Default is 12 rounds with format A only. Example: `uv run share --formats "A,B,C"` generates 12 rounds (4 per format) with distinct colour palettes per panel.
uv run serve runs gunicorn (production). uv run serve-staging (= uv run serve --staging) sets `_STAGING_MODE=1` and adds prod+ test routes under `/staging/*` (`/staging/ending`, `/staging/final`, `/staging/share`, `/staging/prediction`, `/staging` index) that jump to prefilled states for remote testing **without altering production logic** (flag off = byte-identical). Staging deploy: `https://vanilla-sugar.glucosedao.org/`. See `docs/share-ops.md` → "Staging Mode".

## Known Dash pitfalls

### n_clicks corruption on static pages (issue #29)

In Dash 4 (also reproduced in Dash 3), every `html.*` component tracks `n_clicks` by default. Clicking anywhere on a page — text, background, wrapper divs, flex gaps — increments `n_clicks` on the clicked element. This triggers a React re-render that corrupts the component tree: children below the click target silently disappear from the DOM. No server-side callback fires; this is purely a client-side renderer bug.

**Symptoms:** On `/ending` or `/final`, clicking any non-button area causes metrics, buttons, and other sections to vanish. The outer container's padding changes and content is truncated.

**Root cause:** Dash's React wrapper re-renders the component when `n_clicks` changes. During reconciliation of complex static layouts, the renderer drops child components.

**Fix applied:** `disable_n_clicks=True` on every non-interactive element:
- Main layout: `page-content`, `navbar-container`
- `create_ending_layout`: outer wrapper, disclaimer, round info, units, graph section, chart container, metrics, buttons container, switch-format section
- `create_final_layout`: outer wrapper, disclaimer, rounds played, ranking, played formats, overall metrics, per-round metrics table wrapper, switch-format section, restart button container

**Rule for new pages:** When building layouts that are primarily display-only (no drawing/click interactions), add `disable_n_clicks=True` to all `html.Div` and similar wrapper elements. Only omit it on elements that need click tracking (buttons, links, interactive graphs).

**What did NOT work:** CSS `pointer-events: none` on containers, global JS click interceptors in `assets/` (broke the prediction chart), pathname guards on callbacks, making DataTables non-interactive.

### `ending-*` IDs must always be in the DOM on `/ending`

`create_ending_layout` must unconditionally render the full skeleton with every `ending-*` ID (`ending-title`, `ending-disclaimer-*`, `ending-round-info`, etc.). Never early-return a plain "session expired" fallback div — any callback targeting those IDs (e.g. `update_ending_text_on_language_change`, metrics updates) immediately crashes with `A nonexistent object was used in an Output`. If the user has no data, render the skeleton with placeholder/empty content; put the "session expired" handling at the `display_page` level for pathname `/ending` only when you also skip every `ending-*`-targeted callback via a `pathname != '/ending' or not user_info or 'prediction_table_data' not in user_info` guard.

### Consent notice: single scrollbar rule

`consent_notice_children()` is shared between the landing page (`/`) and the `/consent-form` page. It renders the long consent markdown via `static_markdown_iframe` with a fixed height (e.g. `min(55vh, 480px)`) so the iframe owns the scrollbar. **Never wrap it in an outer `overflowY: auto` container** — that creates the infamous double scrollbar bug the user has reported repeatedly. Do not use `static_markdown_autosize_iframe` here either; autosize makes the iframe so tall it forces a second page-level scrollbar. Also do not try to flex-collapse the landing page to `height: 100vh` + `overflow: hidden` to avoid the page scrollbar; that collapses the consent section entirely. Let the landing page scroll normally; the iframe scrolls its own content.

## Code style guidelines

Always use type-hints. 
For file pathes prefer to use pathlib, for cli - typer, for dataframes - polars. 
We try to split logic into components and use functional style when possible, avoiding unneccesary mutability and duplication.
We use eliot logging library with with start_action(action_type=u"action_name") as action pattern to log results to logs folder. We use to_nice_file, to_nice_stdout from pycomfort logging to tell where to save files
Avoid excessive try-catch blocks

### Dash debug reloader caveat

Dash `debug=True` uses Werkzeug's auto-reloader, which forks a child process that re-imports the entire module. Any runtime mutations to `app.layout` are lost on reload. To pass configuration that must survive the fork (e.g. `uv run chart --prefill`), use environment variables read at module-level import time, not post-layout mutations.

### localStorage hydration race condition

`dcc.Store` with `storage_type='local'` hydrates **asynchronously** after the initial server render. Each store hydrates independently — there is no guaranteed order. A callback triggered by one store hydrating as `Input` may read other stores via `State` before they have hydrated, seeing the server-default value (`None` or whatever `data=` was in the layout) instead of the persisted value.

**Rule:** When a callback needs data from multiple localStorage-backed stores to make a correct decision (e.g. `restore_page_on_load` needs both `last-visited-page` and `user-info-store`), make **all** of them `Input` — not `State`. Use a one-shot memory flag (`page-restore-done`) to prevent the callback from acting more than once. If a required store hasn't hydrated yet (`data` is still `None`), `raise PreventUpdate` to wait for the next firing.

**Corollary — don't clobber stores on `/`:** Callbacks like `initialize_data_on_url_change` that write to `full-df` / `current-window-df` must **not** load fresh data when `pathname` is `/` or any non-prediction page. The URL-change callback fires before stores hydrate; overwriting them destroys the persisted session that the resume flow needs.

### Slider and component persistence

Interactive Dash components (sliders, dropdowns, inputs) that are destroyed and recreated on page navigation lose their value unless `persistence=True` and `persistence_type=STORAGE_TYPE` are set. The `time-slider` on the prediction page is recreated every time `create_prediction_layout` runs (e.g. on resume). Without persistence it mounts with the layout-default value, which triggers `handle_time_slider` and re-slices `current-window-df` at the wrong position.

**Rule:** Any interactive component whose value must survive a layout rebuild (page navigation, resume, language change) needs `persistence=True, persistence_type=STORAGE_TYPE`.

### resume-dialog-target must be cleared after dismissal

`render_resume_dialog` has `Input('interface-language', 'data')` so the dialog text updates when language changes. But `resume-dialog-target` is a memory store — if it is not set to `None` when the dialog is dismissed, any later `interface-language` change (e.g. clicking a flag on `/ending`) will re-render the stale dialog on top of the current page.

**Rule:** Every callback that dismisses the resume dialog (`handle_resume_continue`, `handle_resume_start_over`) must set `resume-dialog-target` to `None` in addition to clearing `resume-dialog-container`.

### Mobile version

The app is **mobile-first** (rebuilt June 2026). **Read `docs/mobile-version.md` before touching anything mobile** — it has the full design rationale, a present-state flowpath reference, the screenshot-harness internals, and the complete DON'T list.

Architecture in one paragraph: the static viewport meta is **`width=device-width`** (NOT the old forced `1280`). A clientside callback in `app.py` flips it to `width=1280` **only on `/prediction`** (the chart-drawing page genuinely needs the width) and stamps a `route-prediction` class on `<html>` so CSS can target that page. Detection is two-pronged: a server-side `_is_mobile_ua()` check on the Flask request User-Agent picks **separate mobile builders** in `display_page` / `update_on_language_change` (`StartupPageMobile` wizard, `LandingPageMobile`, `MobileNavBar` burger menu); the existing clientside callback still adds the `html.mobile-device` class (UA + `(pointer:coarse)` fallback) used to scope `assets/mobile.css`. Display-only pages (ending, final, share, faq, about, contact, demo) just reflow via CSS once `device-width` is active. Hard rules: **do not CSS-rotate the chart** (`transform: rotate(90deg)` breaks Plotly drawline touch mapping); `render_mobile_warning()` returns `None` and `mobile-warning` is a throwaway Output sink. **The portrait "rotate to draw" nag overlay was REMOVED** (second non-playable mode). Single mobile `/prediction` flowpath: immersive landscape CSS applies on rotation (gated `@media (orientation: landscape) and (pointer: coarse)` — NO `max-device-width`; uses `100dvh` not `100vh`), and the wizard Start button best-effort requests fullscreen + `screen.orientation.lock('landscape')` (clientside, mobile-only; works on Android after fullscreen, rejects-caught on iOS). Use `uv run python scripts/mobile_shots.py` to screenshot every page.

**`html, body { min-width: 1280px }` in `assets/lang.css` is the master mobile kill-switch** — a leftover that pins the whole page to 1280; `mobile.css` releases it with `min-width:0 !important` on `html.mobile-device:not(.route-prediction)`. If mobile pages render desktop-width again, check this first.

### Mobile pitfalls & lessons learned

**Read `docs/mobile-version.md` before touching anything mobile** — it has the full design rationale, the screenshot-harness internals, and the complete DON'T list. The high-value DON'Ts:

- **Number inputs: hide Dash's own `.dash-input-stepper` `−`/`+` buttons on mobile, NOT the native webkit spinner.** Newer `dcc.Input(type="number")` renders its own stepper `<button>`s inside `.dash-input-container`; once the input is full-width on mobile they wrap below as stray `−`/`+` (a11y-labelled "Decrease/Increase value", so they look native, but `::-webkit-inner-spin-button{appearance:none}` won't remove them). Use `html.mobile-device .dash-input-stepper { display:none }`. Don't clip them with `overflow:hidden` on the wrapper — that re-clips the input's own bottom border ("editor boxes cropped on bottom").
- **Don't let the generic `html.mobile-device input { display:block; width:100% }` rule hit checkboxes/radios** — it stretches consent checkboxes full-width and breaks the label onto the next line. Exclude them (`:not([type="checkbox"]):not([type="radio"])`) and lay each `.form-check` out as a `display:flex` row.
- **Consent reader (`/consent-form`): don't use a `height:100vh`/`min-height:100vh` flex shell.** `100vh` ignores the navbar above `#page-content` and pushes the "Go to start" button below the fold; `min-height:100vh` adds a second page-level scrollbar (double-scrollbar bug). Let the shell be normal flow and give the embedded iframe `height: calc(100vh - 190px)` so the iframe owns the only scrollbar — a full-bleed single box, no nested inner card.
- **Contact links: stack the tables into one column on mobile** (`thead`/`tr`/`td` → `display:block; width:100%`, first cell bold, narrow font + `overflow-wrap:anywhere; word-break:normal` on links) so long emails/URLs don't leave 1–2 char dangling overhangs.
- **Landscape `/prediction` crops the chart bottom in the screenshot harness unless you re-apply the device metrics after the `1280` meta switch.** The clientside meta swap fires after first paint and Chromium doesn't re-fit the page scale under a CDP override → it captures the unscaled top-left slice. Fix: `clearDeviceMetricsOverride` then re-apply the **same** `mobile:true` 740×360 metrics. Don't "fix" it with `screenWidth:1280`/`mobile:false` — that makes device-width 1280 and fails the `max-device-width:1024` gate, so the navbar + instructions reappear. Also call `Plotly.Plots.resize(gd)` (a bare window `resize` event races the layout).
- **Mobile landscape chart controls should reclaim every non-drawing pixel.** Remove unneeded wording ("Glucose Levels", repeated "Source:" labels, standalone "Ready to submit!" spacers), group related controls into compact clusters, and prefer one bottom strip for source/time, units, Submit, and Finish. Keep chip/button heights matched, use subtle bevels/borders to mark metadata, and push fonts as large as comfortably possible after confirming the narrow landscape screenshot still fits.
- **`uv run chart` runs the Dash debug reloader, whose forked child loses the chart-mode prefill, so `/prediction` can redirect to landing** — if the harness chart shots show landing/consent, that (not the CSS) is the cause.
- **Mobile buttons/links need `touch-action: manipulation`** or the browser's double-tap-zoom wait swallows rapid taps (button fires only after several taps; seen on Vivaldi Android). Not a callback bug, not reproducible headless. Fix in `mobile.css` on `a`/`button`/`.ui.button`/`[role=button]`/`label`/`input`/`.form-check`, scoped `html.mobile-device:not(.route-prediction)`.
- **The mobile wizard consent gate must NOT depend on scroll-to-end** (`consent-scroll-complete` watches the outer div, but mobile users scroll the inner consent iframe → never registers → Next hard-locks; headless masks it). Gate `gate_mobile_consent_step` on acknowledge+GDPR only. A gated mobile button must look disabled (grey `startup-next-disabled` class, not dimmed-blue) and show a "why" hint.
- **Consent enforcement is asymmetric desktop vs mobile.** `handle_start_button` only re-checks consent when the consent *fields* are in the DOM (`has_mobile_consent`) — true on mobile (wizard step 0), false on desktop (consent lives on landing). A `display_page` guard on a `consent_completed` flag closes the desktop bypass: `/prediction` (both) and desktop `/startup` require it, else they render landing; mobile `/startup` stays reachable (it IS the consent entry). **Any new synthetic user (chart mode, staging node, test) must set `consent_completed=True`.**
- **localStorage is device-local; cross-device resume goes through the resume code.** All session stores are `STORAGE_TYPE=local`. `resume_store.py` (`data/resume/<code>.json`) keeps a server-side snapshot keyed by `user_info['resume_code']`, auto-saved by `auto_snapshot_session` at meaningful boundaries (not every drawline; dataframes via `State`). Redeem via `?resume=<code>`, the landing-page resume box, or the code shown on the resume dialog. Add new game-state stores to `_resume_payload`/`_restore_outputs_from_code` or they won't transfer.
- **Screenshot `/ending`, `/final`, `/share` via the staging `result` group** (`uv run start` + `_STAGING_MODE=1` → `/staging/ending|final|share`), not LLM click-through. Synthetic rounds leave the per-round metrics table on `/final` empty — a data quirk, not a layout bug.

## Session persistence & navigation contract

These are the expected behaviours that every change must preserve. Treat regressions here as bugs.

1. **First visit → consent form.** A new user lands on `/` (landing page with embedded consent form). She fills it in, proceeds to `/startup` → `/prediction`. No resume dialog, no redirect.
2. **Cross-session resume (localStorage).** The game can span many rounds. All session state (`user-info-store`, `full-df`, `last-visited-page`, etc.) lives in localStorage. If the user closes the browser and reopens hours later, `restore_page_on_load` detects the persisted state, and because `session-active` (sessionStorage) is gone the **resume dialog** appears asking "Continue" or "Start Over".
3. **In-session tab switching (no dialog).** While mid-game the user can click "The Study", "FAQ", "Contact us", etc. and then click "Game" to return. Navbar links use `dcc.Link` (client-side routing, no page reload), so all stores stay populated. `redirect_landing_to_game` silently redirects `/` → the last game page. **No resume dialog must appear in this flow.**
4. **Explicit exit / Start Over cleans storage.** Both the "Finish / Exit" button, the restart button on `/final`, and the "Start Over" button in the resume dialog set `last-visited-page=None` and `clean-storage-flag=True`, which wipes localStorage via a clientside callback. After cleanup the user lands on `/` as a fresh visitor.
5. **`uv run start --clean`.** Sets `_CLEAN_STORAGE=1` env var → `clean-storage-flag=True` in the layout. The clientside callback clears localStorage once on first connect. Subsequent interactions use localStorage normally. Every new browser tab connecting to the same running server also cleans once (stop the server to stop cleaning).
6. **No spurious resume dialogs.** The resume dialog must only appear on genuine fresh sessions (scenario 2). It must never pop up when switching navbar tabs (scenario 3), pressing F5 within an active session, or changing language.

### Key stores involved

| Store | `storage_type` | Purpose |
|---|---|---|
| `last-visited-page` | `local` | Last game-flow page (`/startup`, `/prediction`, `/ending`, `/final`). Never stores `/` or non-game pages. |
| `session-active` | `session` | `True` once the user interacts. Survives in-tab reloads (F5) but clears on tab close, distinguishing fresh sessions from reloads. |
| `page-restore-done` | `memory` | One-shot flag preventing `restore_page_on_load` from acting more than once per page load. Resets on every full reload. |
| `clean-storage-flag` | `memory` | When `True`, a clientside callback wipes localStorage and resets the flag to `False`. |
| `resume-dialog-target` | `memory` | Holds target page + round info for the resume dialog. Must be set to `None` when the dialog is dismissed. |

### How each callback participates

- **Clientside persist callback** — writes the current pathname to `last-visited-page` only for persistable game pages (`/startup`, `/prediction`, `/ending`, `/final`). Never writes `/`.
- **`restore_page_on_load`** — fires on full page loads as localStorage stores hydrate. If `session-active` is `True` (same tab, e.g. F5), silently redirects. If `False` (fresh session), shows resume dialog. Waits for both `user-info-store` and `full-df` before deciding the target for `/ending`.
- **`redirect_landing_to_game`** — fires on in-session client-side navigation to `/`. Reads the already-populated stores and redirects to the last game page. Does nothing on fresh page loads (stores are `None`).

## Learned User Preferences

- Never attempt browser automation (drawing predictions, clicking through multi-step forms) with LLM agents — it fails; always use `uv run chart --prefill` instead
- Use `fuser -k PORT/tcp` to kill stray Dash processes on a busy port
- Keep `logs/*` with `!logs/.gitkeep` in `.gitignore` to preserve the directory in git while ignoring log files; `.cursor/` must be fully gitignored
- The UI uses Fomantic UI (Semantic UI fork) classes alongside Dash — prefix interactive classes with `ui` (e.g. `ui green button`)
- Do **not** rewrite the landing page into a flex-only `height: 100vh; overflow: hidden` shell to eliminate the double scrollbar — past attempts collapsed the consent section entirely. Fix double-scrollbar issues by choosing a single owner of the scroll (usually the iframe) and removing `overflowY: auto` from the others
- When a fix regresses or layout breaks, check `git stash list` / `git stash show -p stash@{N}` for a prior working version before re-designing from scratch; the user has stashed working fixes in the past
- "Start Over" must reset the app to a truly fresh state: clear `user-info-store`, consent selections, `last-visited-page`, and any other localStorage-backed stores. A partial clear that leaves consent checkboxes ticked is a bug
- Do not introduce image libraries like PIL/Pillow for chart/share rendering — the project already has Plotly + kaleido and must reuse them for any PNG/OG-card output
- On `/final` the exit button is labelled "Exit" (not "Start Over") and routes to landing (`/`); the share page's "Play again" button uses the same landing-redirect contract as the final "Exit" button

## Browser automation tips (cursor-ide-browser MCP)

- Elements with `disable_n_clicks=True` (including language flags and navbar wrappers) do **not** appear as interactive refs in `browser_snapshot`. You cannot click them by ref.
- CSS-selector-based clicks (`browser_click` with `selector: "#some-id"`) also fail on elements with `disable_n_clicks=True` — the Dash attribute strips the React event handlers the browser tool relies on.
- **Workaround that works:** Use `browser_navigate` with a `javascript:void(...)` URL to programmatically click the element via the DOM: `javascript:void(document.getElementById('lang-de').click())`. This bypasses the missing React handlers and fires the Dash callback correctly.
- Coordinate-based clicks (`browser_click` with `coordinates`) fail when the element is outside the default viewport (1024 px wide). Use `browser_resize` first, or prefer the JS workaround above.
- `browser_screenshot` does not exist; the correct tool name is `browser_take_screenshot`.

## Learned Workspace Facts

- The app uses Fomantic UI CSS/JS loaded via `external_stylesheets` and `external_scripts` (jQuery is loaded first as a dependency)
- GitHub repo is GlucoseDAO/sugar-sugar; issues are tracked there
- `suppress_callback_exceptions=True` is set on the Dash app to allow callbacks referencing components not yet in the layout
- The navbar is a Fomantic UI `massive blue inverted tabular menu` (`NavBar` class in `sugar_sugar/components/navbar.py`). Left items: Game, The Study, FAQ, Video instructions, Contact us. Right side: a Fomantic `ui simple dropdown item` (`lang-dropdown`) — the trigger shows the active language's flag+label and a dropdown caret; the menu lists all 8 languages from the module-level `LANGUAGES` constant. Use the **`simple` dropdown class** (CSS-only hover) because Fomantic's JS dropdown requires jQuery init which doesn't play well with Dash. Each dropdown item is an `html.A` with `id="lang-{code}"`, so the existing `set_interface_language` callback works unchanged. Wrapper divs inside the dropdown have `disable_n_clicks=True`; the `lang-*` links do not. Navbar uses `dcc.Link` for navigation (client-side routing, no full page reload) — this preserves all `dcc.Store` values and avoids hydration races. A `redirect_landing_to_game` callback redirects `/` → last game page when the user clicks "Game" mid-session.
- `STORAGE_TYPE` env var controls `dcc.Store` `storage_type` and input `persistence_type` across the app; defaults to `local` (localStorage persists across sessions)
- When using `dcc.Store` with `storage_type='local'`, the store hydrates from localStorage client-side **asynchronously** after initial render; use it as callback `Input` (not `State`) to react to hydration — see "localStorage hydration race condition" pitfall above
- A `last-visited-page` store + `restore_page_on_load` callback restores the user's last page when `STORAGE_TYPE=local`; a resume dialog (continue / start over) appears for returning users. Page flow: `/` → `/startup` → `/prediction` → `/ending` → `/final`. The callback uses `user-info-store` and `full-df` as Inputs (not State) to avoid the hydration race
- `page-restore-done` uses `storage_type='memory'` — it resets on every full page reload. `session-active` (sessionStorage) is the store that distinguishes a genuine new session (show resume dialog) from an in-tab reload (silent redirect). See "Session persistence & navigation contract" above.
- `initialize_data_on_url_change` must only load fresh data when `pathname == '/prediction'` and `full-df` is empty. For all other pathnames it returns `no_update` to avoid clobbering persisted stores during resume
- `dcc.Location` must NOT have a hardcoded `pathname="/"` — it overrides the actual browser URL and breaks direct navigation to `/about`, `/contact`, etc. Omit `pathname` so it reads from the browser.
- Dash clientside callbacks cannot use the same `dcc.Store` as both Input and Output — causes `dc[namespace][function_name] is not a function` JS error. Use a separate store or `State` instead.
- `uv run start --clean` clears all browser localStorage on first connect via `clean-storage-flag` store + clientside callback; "Start Over" in the resume dialog reuses the same `clean-storage-flag` mechanism
- `_STATEFUL_PAGES` (`/prediction`, `/ending`) skip full `page-content` re-renders on language change to preserve interactive/chart state. Each stateful page needs its own `update_*_text_on_language_change` callback that targets individual element IDs. `/final` is **not** stateful — it re-renders fully via `update_on_language_change`.
- When adding a new stateful page or translatable text to an existing one, every translatable element needs a stable `id` and a corresponding `Output` in the page's language-change callback. Otherwise the text stays in the old language.
- Large static markdown documents (study design, consent-style content) should keep using the server-rendered `static_markdown.py` iframe path; `dcc.Markdown` can misrender or fail on the 100KB+ study document because it loads asynchronously via `react-markdown`.
- The prediction area is 12 points (1 hour at 5-min intervals); the game requires predictions drawn to the end of the hidden area before submit. `MAX_ROUNDS` is configurable via `.env` (defaults to 12).
- CGM file uploads are parsed in `sugar_sugar/data.py` via `cgm-format` (`FormatParser.parse_file` + `FormatProcessor.split_glucose_events`) and then adapted to the app's existing glucose/events store schemas.
- Plotly charts on `/prediction` (`GlucoseChart` in `sugar_sugar/components/glucose.py`) and `/ending` (`ending-static-graph` in `app.py`) use `config={'displayModeBar': False, ...}` — the Plotly toolbar (camera/zoom/pan icons) is hidden on purpose. The chart's outer div and inner `dcc.Graph` both set `style={'touchAction': 'none'}` so browser pinch/pan gestures don't fight Plotly's `drawline` handler on mobile.
- The clientside persist callback never writes `/` to `last-visited-page` — only `/startup`, `/prediction`, `/ending`, `/final`. Writing `/` would clobber a deeper stored page and break the resume dialog. Exit from `/prediction` always goes to `/ending` (never directly `/final`); exit from `/ending` with no completed rounds goes to `/` (landing), otherwise to `/final`.

## Share page (`/share/<share_id>`)

A public, read-only page that lets a user broadcast their Sugar Sugar performance. Key invariants:

- **Share records live on disk**, not in `dcc.Store`. The store at `data/shares/<share_id>.json` is written atomically by `sugar_sugar/share_store.py`. This is deliberate: the URL must work for anyone, across devices, after localStorage is wiped.
- **`/share/<share_id>` must render without any `dcc.Store` data**. `display_page` and `update_on_language_change` both load the record from disk via `share_store.load_share` and pass it into `create_share_layout`. If the id is missing/corrupt they show `create_expired_layout`, never a crashed page.
- **Two sibling Flask routes**, not Dash pages:
 - `GET /share/<id>/image.png` — `sugar_sugar.share_png.render_share_card_png_bytes` renders `build_share_card_figure` to a 1200×630 PNG (kaleido/Chrome-for-Testing primary; static branded `assets/og-card.png` last resort). Results are cached in `_SHARE_PNG_CACHE` so repeated loads don't respawn Chromium.
 - `GET /share/<id>/og` — crawler-only minimal HTML with Open Graph + Twitter Card meta tags and no meta-refresh. Humans hitting `/og` are redirected server-side (`302`) to the real Dash page. Needed because FB/X/WhatsApp/LinkedIn crawlers don't execute JS and would otherwise see the Dash shell with no OG tags. Social share buttons always link to the regular `/share/<id>` URL — crawler user agents get the OG response at that URL via `before_request`.
- **`kaleido`** is used for the primary share PNG (pixel-perfect card). The card is 1200x630 (1.91:1) so Facebook/LinkedIn/X do not crop it. Kaleido/Choreographer downloads Chrome for Testing via `uv run setup-chrome`, but the downloaded binary still needs Chromium runtime shared libraries on bare Linux hosts. If kaleido cannot export, `share_png.py` retries after ensuring managed Chrome exists, then serves the static branded `assets/og-card.png` as the last-resort fallback. Do not reintroduce a separate matplotlib redraw path; it drifted from the real card design. First kaleido render takes ~1 s; subsequent are served from `_SHARE_PNG_CACHE`. Do NOT hot-reload the server while kaleido is rendering — it can leave orphaned Chromium processes on Windows.
- **Share button wiring**: `share-results-button` on `/final` fires `handle_share_results_button`, which builds a lean JSON-safe record (rounds, limited `user_info` keys, locale, timestamp), persists it, and returns the new `/share/<id>` URL as the `url.pathname`. The record intentionally drops heavyweight stores (`full-df`, `events-df`) — everything the share page needs already lives in `prediction_table_data`.
- **Encouragement text** (`sugar_sugar/encouragement.py`) is template-based today, keyed by a score bracket derived from overall MAE. A module-level `LLM_BACKEND: Optional[Callable]` is the swap point if you want to plug in a real LLM later; do not sprinkle LLM calls elsewhere.
- **`data/shares/` is gitignored** — share records are session data, not source code.
- **Synthesis graph on the share page** is one **stacked Plotly row per data-source format the user played (A / B / C)**. The **x-axis** is **time in the next hour** with a tick for **every 5-min step** (count = `PREDICTION_HOUR_OFFSET`, default 12). **Y = percent off actual** — `(pred − actual) / actual × 100` (skip if actual is 0). If the **first** value in the next-hour window has **no prediction**, **actual** at that step is imputed for display so the series starts. Each format row has a **tinted panel**; **colours are data-scaled per row**: `ref = max |% error|` over all rounds in that format, and each point blends its line colour toward **neutral grey (128,128,128)** by `(|y|/ref)^γ` (γ≈0.38), so the **worst |error| in that subplot** reads as **literal grey**. **Stacked fill bands** keep more intensity near the 0% axis. A **continuous solid black, thick 0%** line sits **above fills, below lines/markers**. The old gradient bar and the "accuracy %" stat tile are gone; MAE/RMSE stay on the card.
- **Rankings are shown per data-source category (example/generic, mixed, own) AND overall** on the share page, derived from `is_example_data` / `data_source_name` on each round record. The overall ranking comes first in the layout, followed by per-category rankings. The redundant per-round metrics table below the synthesis graph was removed on purpose — do not reintroduce it.
- **Clientside persist allowlist** in `app.py` only writes `/startup`, `/prediction`, `/ending`, `/final` to `last-visited-page`. `/share/*` is automatically excluded by the allowlist; do not add it.
- **Mobile**: the share page reuses `.info-page` + scoped `.mobile-device .share-page` rules in `assets/share.css` so the download/copy buttons stay readable on narrow viewports. The orientation overlay from `mobile-support` still fires if the user loads `/share/<id>` on a phone in portrait.
