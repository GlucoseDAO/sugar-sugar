## Project overview

This project is a Sugar-Sugar game where the user gets the glucose value for some timespan and has to predict by drawing the lines on the chart. He is given a sequence that he has to prolong. The aim of the study is to measure human accuracy of the glucose predictions.
The project is a DASH app, with app.py being main, while glucose, metrics, prediction and startup are dash components. I has default example csv file to play and debug with but provide an option to upload your own csv files from dexcom, libre and other CGM-s.
We use session storage to allow multiple users workin on the same app. Predictions are stored in polars dataframe, there is also a dataframe for current prediction window and scrolling positions.
When the user draws the line it interpolates the position to detect closes glucose and time value (time measurements are done every 5 minutes) and then updates the dataframe with the prediction values.

## Build and test commands

uv is used as the package manager for the project.
uv run start is used to run the dash app.
uv run chart is the fast dev shortcut: it starts Dash with data pre-loaded and routes straight to the prediction chart (bypasses landing, startup, and consent). Use this whenever the user asks to debug or test the chart in the browser. Only fall back to uv run start when the user explicitly needs the startup/landing/consent screens. uv run chart accepts --file, --points, --start, --unit, --locale, --host, --port options. Use --prefill to pre-fill the prediction region with noisy ground-truth values so the submit/ending/metrics flow can be tested without drawing (--noise controls the noise level, default 5%). Always prefer uv run chart --prefill over attempting browser automation for testing submit or ending pages.
uv run share is the share page dev shortcut: it generates fake rounds with synthetic prediction data, saves a share record to disk, and opens the browser at `/share/<id>`. Use this whenever debugging the share page, share card PNG, OG tags, social buttons, or share-related styling. Accepts --port, --locale, --host, --formats, --rounds options. Use `--formats "A,B,C"` to simulate multi-format play (Public + My Data + Mixed); rounds cycle through formats evenly. Default is 12 rounds with format A only. Example: `uv run share --formats "A,B,C"` generates 12 rounds (4 per format) with distinct colour palettes per panel.
uv run build-locations regenerates `assets/location-suggestions.<locale>.json` from `sugar_sugar/location_catalog.py`. Run it after editing the catalog; the tests fail if the shipped files drift from it.
uv run download fetches the Format A corpora (BIG IDEAs + D1NAMO). Already-present copies are skipped; `--force` re-downloads; `--all` also pulls CGMacros (unused in Format A). `uv run start` works without them — Format A falls back to `data/example.csv`. Per-dataset commands remain.
uv run download-bigideas fetches PhysioNet BIG IDEAs Dexcom + food logs into `data/bigideas/` (gitignored). Empatica files are skipped. CGMacros is unused in Format A.
uv run download-d1namo fetches the public D1NAMO (Dubosson) T1D subset into `data/d1namo/` (gitignored). Default extract keeps meal photos; `--no-photos` skips JPEGs.

Format A source policy (`generic_intervention`): no diabetes / gestational → BIG IDEAs; type 1 → D1NAMO; type 2 → 50/50 mix each round; prediabetes → 75% BIG IDEAs / 25% D1NAMO; LADA → 75% D1NAMO / 25% BIG IDEAs. BIG IDEAs meals have no photos — the apple icon opens a text notepad (backdrop click closes, same as the D1NAMO photo lightbox).
uv run serve runs gunicorn (production). uv run serve-staging (= uv run serve --staging) is the same but sets `_STAGING_MODE=1`, exposing prod+ test routes under `/staging/*` (`/staging/ending`, `/staging/final`, `/staging/share`, `/staging/prediction`, and a `/staging` index) that jump straight to prefilled states for remote/visual testing **without altering any production logic** — when the flag is off the app is byte-identical. The staging deployment `https://vanilla-sugar.glucosedao.org/` hosts the dev branch. See `docs/share-ops.md` → "Staging Mode".
The public FAQ ask/reply board at the bottom of `/faq` is **off by default** (`FAQ_BOARD_ENABLED=0` in `.env.template`) until it has bot protection; the flag hides the post form *and* the list of existing questions, and `add_faq_question` / `add_faq_reply` refuse writes while it is off. The curated FAQ entries always render.

## Data ingest: everything parseable goes through `cgm-format`

`sugar_sugar/corpus.py` is the **single boundary** between the library's unified frames and the app's
`(time, gl, prediction, age, user_id)` / `(time, event_type, event_subtype, insulin_value, …)` stores.
`load_glucose_data` (`data.py`) routes to five loaders; only LOOP still parses by hand.
The library floor is **cgm-format 0.12.2** (`pyproject.toml`). 0.11 was the release that added
BIG IDEAs, the last corpus the app parsed itself; 0.12 added grid re-timing for training alignment;
0.12.2 fixed the Nightscout defects reported in `FEEDBACK.md` and is a **hard** floor, not a
preference — on 0.12.0 the Nightscout URL import is dead for any real instance.

**What 0.12 changed for this app: the frame got one column wider, and nothing else.**
`original_glucose` (the device's own reading, before any re-timing) joins both unified schemas, and
`Quality.GRID_RETIMED = 64` joins the flag vocabulary. Every loader's `(time, gl, …)` /
`(time, event_type, …)` output is **byte-identical to 0.11** across all five routes — verified over
example.csv, Dexcom/Libre/Medtronic/Nightscout exports, `loop_467`, pre-0.12 unified CSVs, and the
D1NAMO / BIG IDEAs / CGMacros corpora. Two reasons it is inert here:

- **The app never runs the grid stages.** `synchronize_timestamps`, `interpolate_gaps`,
  `prepare_for_inference` and `to_ml_format` are where 0.12's re-timing lives, and the app calls
  none of them — only `split_glucose_events`. Glucose values reaching the chart are the device's,
  never re-timed. If you ever do call `synchronize_timestamps`, note it now **rewrites glucose** at
  each grid instant by default; pass `retime_glucose=False` for the old behaviour.
- **`corpus.adapt_*` selects columns by name**, so the extra column is dropped at the boundary and
  never reaches a `dcc.Store`. (This is what keeps the store-size contract below intact.)

Pre-0.12 unified CSVs on disk stay readable: the parser backfills `original_glucose` from `glucose`
per row, leaving it null only on rows that carry no glucose.

**Why adopt a release the app is inert to:** those grid stages plus `to_ml_ready_df` *are* the
inference path — they exist to align a trace with the shape the SugarOne model was trained on. Sugar
Sugar has no AI player yet; when it gets one, that is the code it will run on, so the floor is here
waiting instead of being raised under pressure later. Nothing needs to change for the human game.

| Source | Parser | Why |
|---|---|---|
| Vendor exports (Dexcom/Libre/Medtronic) | `FormatParser.parse_file` | — |
| **Nightscout `entries.json`** | `FormatParser.parse_nightscout` (routed in `data.py`) | JSON, which `detect_format` deliberately does not sniff |
| **D1NAMO** | `FormatParser.parse_subject_directory` | subject = a *bundle* of glucose/insulin/food CSVs |
| **BIG IDEAs** | `FormatParser.parse_subject_directory` | subject = a Clarity export + a food log (0.11+) |
| **CGMacros** | `FormatParser.parse_tracks` | subject = one CSV with two sensor tracks |
| LOOP `*_chronological.csv` | in-repo (`data.py`) | no library counterpart |

Hard-won rules:

- **Nightscout uploads are `entries.json`, never a Nightscout CSV.** `/api/v1/entries.csv` is
  headerless with five hardcoded columns and there is no treatments CSV at all, so cgm-format
  cannot detect it and the upload hints must not offer it. `is_nightscout_entries_json` sniffs
  *content*, not the file name, because both upload handlers rewrite the extension. It never
  looks for a sibling `treatments.json`: every upload lands in one shared `data/input/users/`
  under a timestamped name, so guessing at a neighbour would eventually pair one player's
  entries with another's treatments. `load_nightscout_json_data` takes an explicit optional
  treatments path instead, and the profile file is not a third input — cgm-format downloads it
  and discards it.
- **The Nightscout *URL* import needs cgm-format >= 0.12.2, which is the floor.** On 0.12.0 it
  was dead for any real instance: `_parse_nightscout_treatments_json` built a frame with no
  schema, so a treatment field null for the first 100 records inferred as dtype `Null` and the
  import died with `ComputeError` — which is what a player actually hit. It was deliberately
  never worked around in `data.py`; the fix belongs upstream and shipped there. Full history,
  a synthetic repro and the four other 0.12.0 defects it travelled with are in `FEEDBACK.md`,
  along with the two items still open (neither affects this app — both are on the exporter-CSV
  path, which Nightscout data never takes here). `tests/test_nightscout_json_upload.py`
  ::`test_treatments_with_a_long_null_run_are_loaded` was written red against 0.12.0 and is now
  a regression guard — if it goes red again, fix it upstream, not here.
- **`scripts/diagnose-nightscout.py` is the tool for "my Nightscout import failed".** Run it on
  the host that failed, not a laptop: it walks DNS → TCP → TLS → the three endpoints, then runs
  a synthetic parse repro that needs no network, so it separates a network block (the instances
  are often .ru-hosted, and egress is filtered in both directions) from a too-old library on the
  deployed host. Two control hosts tell "no egress" apart from ".ru blocked" apart from "this
  instance is the problem". It scrubs the `--token` from its own output because
  `httpx.HTTPStatusError` embeds the full query string.

- **`ExtendedFormatProcessor`, not `FormatProcessor`, for corpora.** Corpora target the wide
  `CGM_SCHEMA_EXTENDED`; `FormatProcessor.schema` is the narrow `CGM_SCHEMA` and dies with
  `MalformedDataError: Schema has N columns, dataframe has M columns`. Never hardcode either width —
  call `corpus.unified_processor(df)`, which dispatches on the frame's column tuple, read off the
  library. **The widths grow between releases:** 0.11 was 10 and 22 columns, 0.12 added
  `original_glucose` and made them **11 and 23**. Any hardcoded count silently misroutes every corpus
  frame to the narrow processor.
- **Meal detail rides in the extended schema's JSON `annotations` column**, not in real columns:
  D1NAMO uses `picture`/`description`, CGMacros `image_path`/`meal_type_raw`. `corpus.adapt_events_df`
  coalesces both vocabularies into `photo_path`/`meal_type`/`carbs_g`.
- **Those three columns appear only for corpora.** A vendor export has no meal photos, so it gets the
  bare 4-column events frame. Widening it would ship an array of empty strings to the browser on every
  upload — `events-df` is a localStorage store the client re-uploads each callback (see the 2026-07-28
  freeze note below).
- **`synchronize_timestamps` is not a downsampler.** Asking it for 5 minutes on 1-minute CGMacros
  *snaps* readings onto the grid, yielding five rows on one timestamp. `_downsample_glucose_5min`
  in `cgmacros.py` still owns cadence reduction.
- **CGMacros: play Dexcom, never the `mean` track.** Each of `libre`/`dexcom` is a complete view of the
  same days with meals replicated into both, so concatenating double-counts every meal, and the
  library's synthetic `mean` track looks like the obvious fix — `mean_horizontal` ignores nulls, so it
  averages where both sensors read and passes a lone reading through untouched, reproducing the old
  hand-written coalesce exactly (138,168 rows, identical in all 45 subjects).
  **Do not use it: the two sensors disagree.** Fitting `libre = slope·dexcom + intercept` over all 45
  published subjects gives median slope **0.70** (range 0.10–1.16) with a small intercept — Libre is
  not offset from Dexcom, it *compresses* the excursion by a subject-specific factor. Median
  correlation is 0.82 and 15/45 subjects fall below 0.7, so a third of the corpus disagrees in shape
  too. Libre is the implausible one: it reads below 70 mg/dL for 82% and 86% of subjects 007 and 015,
  against 0.4% of all Dexcom readings. Averaging assumes unbiased independent error from two views of
  one quantity; all three conditions fail, and the mean would carry ~85% of true excursion amplitude
  into a study scoring human error in mg/dL.
  `_playable_track` therefore takes Dexcom, falling back to Libre (with a warning) only when a subject
  has no Dexcom readings at all. The 8.4% of rows this gives up are Libre-only ones, 10.1% of which are
  sub-70 — a contaminant the old coalesce silently mixed in, not good data lost. Span is not binding:
  ten Dexcom days is ~2850 five-minute readings, ~79 non-overlapping 36-point windows per subject.
- **cgm-format emits one event per *labelled* meal row.** CGMacros also has ~1553 unlabelled
  photo-only rows (a follow-up shot minutes after the meal); the library drops them, which is why the
  old bridging behaviour that merged distinct meals into one marker is gone.
- **D1NAMO is always mmol/L** and is converted through the declared unit (18.0182), never a
  "max < 40" heuristic. Fingersticks become `CALIBRAT` events and so stay out of the glucose trace.
  `carbs` is genuinely null — D1NAMO has no carbohydrate column anywhere.
- Photo *resolution and serving* stay in-app: the library only carries the reference it was given.
  Real D1NAMO subjects store JPEGs in `food_pictures/`; `resolve_photo_path` also tries
  `pictures/photos/food`. `subject_format()` returns `None` for a directory the library cannot
  identify (e.g. `glucose.csv` with no `insulin.csv`/`annotations.csv`), and discovery skips it so the
  round picker is never handed an unparseable source.
- **BIG IDEAs is a *directory* shape, never a header sniff.** Its glucose half genuinely is a Dexcom
  Clarity export — `parse_file` on it alone detects DEXCOM and returns glucose with no meals — so only
  the `Food_Log_*.csv` beside it makes the folder a subject. `bigideas.subject_format()` asks the
  library (`detect_subject_format`), and `is_bigideas_path` routes a `Dexcom_NNN.csv` to the corpus
  loader only when that probe answers. The probe is conjunctive: a subject directory holding a Dexcom
  export and *no* food log is not parseable at all (`parse_subject_directory` raises
  `UnknownFormatError`), so discovery skips it rather than handing the round picker a source that dies
  in front of a player.
- **The food log is per *item*; the chart wants per *sitting*.** cgm-format emits one `CARBS_IN` event
  per logged food ("clustering items into a sitting is a consumer concern"), so
  `bigideas.cluster_food_events` folds rows within 30 minutes of the previous one — chained — into one
  marker, sums their carbohydrates, and joins their names into the `food_note`. Two items logged at the
  same minute are *not* ordered: `_postprocess_unified` ends in a plain `sort("datetime")`, which polars
  does not promise is stable, so never assert the order of lines inside one note.
- **`food_note` is BIG IDEAs-only** (`corpus.FOOD_NOTE_EVENTS_SCHEMA` = the corpus seven plus it). The
  corpus has no meal photographs, so the apple icon opens a notepad instead of a lightbox; `photo_path`
  stays empty for every row. Same store-size reasoning as the three meal columns — do not widen the
  other loaders' frames with it.
- **Test fixtures must be real-shaped now that the library parses them.** `tests/testdata/bigideas/`
  holds proper Clarity exports (10 metadata rows, no `Transmitter ID` — which is what the published 16
  actually ship) rather than the trimmed 5-column tables the in-repo parser tolerated. `001` carries the
  extra `PatientIdentifier` metadata row that makes the library warn about metadata drift (expected, and
  it skips what the file has, so no data row is lost); `002` covers the headerless 11-column food log and
  the blank-`time_begin` → `date` + `time` fallback.

### Event markers are visible in the predicted hour; glucose is not

`hide_last_hour` withholds the **glucose trace** of the hour being predicted. It
must not withhold the **event markers** in it — meals, insulin or exercise. A
BIG IDEAs window put a meal a few minutes past the divider, the marker was
clipped, and the player drew a flat line into a post-meal rise they had no way to
see coming. The clipping did not withhold a hint, it made the displayed history
misleading, and it biased the very error the study measures. Nobody predicts
their own glucose without knowing when they ate, dosed or exercised.

`visible_food_photo_events` (`cgmacros.py`) therefore spans the whole window, and
`cluster_visible_food_events` / `meal_food_bubble_children` no longer take a
`hide_last_hour` argument at all — an ignored parameter is how the clipping would
come back. All three marker kinds show during the round: the FOOD speech bubble
plus guide line (photo/note meals), the apple icon (plain carb events), the
syringe, and the exercise star.

**What must stay hidden is the y-value.** Markers are normally placed at the
event's glucose height; past the boundary that *is* the answer.
`_add_event_markers` pins them to `_HIDDEN_MARKER_Y_FRAC` of the y-span instead —
a single "topside" rail they stack along — and `_draw_hidden_marker_guides` draws
a dotted vertical **in each event's own colour** (a syringe must not be announced
by a green meal line) so the timing still reads off the axis. Note exercise was
never boundary-gated at all, so before this its star sat at the true hidden
glucose value and could be read straight off the y-axis. Locked down by
`tests/test_food_marker_prediction_area.py`.

### The playing chart says where the line has to end

A player stopped drawing partway through the hidden hour, pressed Submit, got
nothing, and reported it as a bug. Three things had to line up, and all three
were true:

- **Nothing on the chart marked the target.** The glucose trace simply stopped
  and the rest of the plot was empty. `_add_prediction_finish_line`
  (`glucose.py`) now puts a checkered flag on a dashed vertical at the *last* x
  index whenever `hide_last_hour` is on, labelled "Draw to here" and turning
  green ("Line complete") once `hidden_area_is_complete` passes. Results charts
  never get it -- there the hour is already revealed. The flag and its label are
  **right-anchored**: the x-axis range is `[-0.5, n-0.5]`, so anything centred on
  the last index is clipped by the plot border.
- **The copy that says so is hidden where it matters.** `mobile.css` has
  `#prediction-progress-label { display: none !important }` on mobile
  `/prediction` -- the landscape control strip has no room for it -- so on the
  device most rounds are played, the button label is the only surface left. The
  disabled Submit therefore reads `Submit (6/12)`
  (`ui.submit.submit_remaining`). Do not "fix" this by unhiding the label.
- **Fomantic colour classes beat inline styles.** `.ui.green.button` sets its
  background `!important`, so the gate's inline `#999999` never applied: a
  disabled Submit rendered as *green at 45% opacity* and looked pressable. The
  gate now swaps the class (`SUBMIT_ENABLED_CLASS` / `SUBMIT_DISABLED_CLASS`),
  same as the mobile wizard's Next button. **Any Fomantic-coloured button with a
  disabled state has this bug until the class is swapped.**

Locked down by `tests/test_prediction_finish_marker.py`.

### Windows must not straddle a sensor gap

A CGM trace is not one continuous run — `data/example.csv` alone breaks into 11 stretches. Windows are
sliced by *row index*, so before this check 1.4–7.6% of random 36-point windows spanned an outage,
asking the player to continue an hour that was days away from the one shown.

`subject_sources.window_is_continuous(window_df)` reads the largest gap straight off the `time` column
(threshold `config.SEQUENCE_GAP_MINUTES = 15`, matching cgm-format's `small_gap_max_minutes`). Both
pickers — `pick_unique_generic_window` and `app.get_random_data_window` — reject discontinuous
candidates, ranking continuity **above** the food-photo preference (a photo-less window is duller; a
discontinuous one is broken), and fall back to the old choice rather than failing a round.

Continuity is deliberately **derived from timestamps, not a stamped `sequence_id` column**: the glucose
frame must stay exactly `["time", "gl", "prediction", "age", "user_id"]`, or frames loaded from disk
and frames reconstructed from a `dcc.Store` would have different shapes.

## Known Dash pitfalls

### `allow_duplicate` hashes the INPUTS, so same-trigger writers collide

Dash derives the duplicate-output suffix from the callback's inputs alone
(`create_callback_id` → `_hash_inputs`, `dash/_utils.py`). Two callbacks writing the same
property off the *same* `Input` therefore hash to the **identical** output id and the
renderer aborts the whole page with "Duplicate callback outputs" — `uv run chart` served a
blank `/prediction`. Without `debug` the page rendered but one writer was silently moot, and
which one won was never defined.

`compact_events_store` and `initialize_data_on_url_change` both wrote `events-df` off
`Input('url', 'pathname')`, which is how this shipped. Two writers of one store on one trigger
were also a last-writer-wins race.

**Rule:** `allow_duplicate=True` is not a licence to add another writer. When the trigger is the
same, fold the work into the existing callback (`compacted_events_store` is now a plain helper
called from `initialize_data_on_url_change`). Any future pathname-keyed `events-df` writer belongs
there too. `tests/test_callback_output_ids.py` fails the build if two callbacks ever share an
output id again.

### A callback fires only when EVERY Input and State is mounted

`suppress_callback_exceptions=True` lets you *register* callbacks for ids that are not always in
the tree; at fire time Dash still needs all of them present. A confirm handler that listed both
`finish-study-button` (`/prediction` only) and `finish-study-button-ending` (`/ending` only) could
never fire on either page — the exit button did nothing. Same rule as the `/startup` vs `consent-*`
trap below.

**Rule:** a callback may only span ids that render together. Page-scoped flows get page-scoped ids:
`finish-confirm-*-prediction` (may read `time-slider`) and `finish-confirm-*-ending` (must not).
Keep each overlay and its context store in the same layout builder so a callback can never see half
of them. Locked down by `tests/test_callback_output_ids.py`; full postmortem in `docs/known-issues.md`.

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

### Never put a whole dataset in a `dcc.Store` — the client re-uploads it every callback

A store's value is not just "state on the client": the browser sends the value of **every**
`Input`/`State` store a callback declares along with each `_dash-update-component` POST. A big
store therefore costs its full size *in upload bandwidth, per interaction*, on top of the
localStorage write (synchronous, ~5 MB quota per origin).

**Production incident 2026-07-28:** `events-df` was filled with the *whole subject's* event log.
Rounds 1–6 landed on small generic sources and felt fine; round 7 drew `loop_467`
(62,308 events → **3.4 MB** of JSON) and the game became unusable — "extremely slow, it takes
many seconds for every click", then a 48 s timeout on the public monitor. The CSV parse itself was
never the problem (0.2 s, cached in `_load_dataset_cached`); the per-click 3.4 MB upload was.
Fixed by `events_within_window` / `events_store_for_window` in `app.py`: every writer of
`events-df` trims events to the current window's first/last timestamp. This is lossless —
`GlucoseChart._add_event_markers`, `window_has_carb_events` and `create_ending_layout` all filter
to that same span anyway. `compact_events_store` (fires on navigation) shrinks oversized stores
left in localStorage by an older build. `handle_time_slider` re-trims whenever the window moves,
so window and events never drift apart.

**Rules:** keep client stores window-sized; the full dataset stays server-side behind
`load_dataset` (per-worker LRU) and is re-sliced on demand — this is the same reasoning that
removed the `full-df` store. When adding a store, check its worst-case size against the *largest*
generic subject (`data/subjects/loop_467`, 9 MB CSV), not `example.csv` (260 KB). And remember
sync gunicorn workers hold a worker for the whole request body read, so one slow client's upload
blocks other players: `serve --threads N` (or `GUNICORN_THREADS`) switches to gthread if needed.

### Nothing loads eagerly on `/assets` that only one page needs

Everything in `assets/` is served on every page, so a cost paid at script-eval
time is paid by the consent step, the chart and `/faq` alike. Two shapes of waste
both shipped to production and both read as "the app froze" on a low-spec phone:

- **A big payload fetched before its page exists.** `location-suggestions.json`
  was 824 KB of all-locale labels, fetched, parsed and re-mapped at script eval,
  on `DOMContentLoaded` *and* on every navigation, for a field on wizard step 3.
  It is now one compact file per locale (~88 KB, ~29 KB gzipped) fetched on the
  first keystroke in the field. `uv run build-locations` regenerates them from
  `sugar_sugar/location_catalog.py` (still the only source of truth — **no
  generated corpus is kept on disk**), and `tests/test_location_suggestions.py`
  fails if the shipped assets drift from the catalog or if the all-locales file
  reappears in `assets/`. Rows ship as `[label, rank]` / `[label, rank, extras]`;
  the JS derives the lowercase and ASCII-folded tokens from the label, so only
  what it cannot derive (alternate spellings, aliases like "peking") is stored.
- **A timer or observer that outlives its page.** `autosize-iframe.js` ran
  `setInterval(resizeAll, 500)` forever on every page and re-added a `load`
  listener to every frame on every DOM mutation — an unbounded listener leak on
  an app that re-renders on each callback — to serve one iframe that only
  `/about` renders. Sizing is event-driven now (`load` + `ResizeObserver` on the
  inner document + window resize) with a bounded ~2 s settle poll.

**Rules for new asset JS:** fetch data at the moment it is used, not at import;
never leave an unbounded `setInterval`; a `MutationObserver` on
`document.body`/`documentElement` with `subtree: true` must collapse a burst into
at most one scan per animation frame (`requestAnimationFrame` guard) — Dash fires
mutations on every callback response. Prefer delegated `focusin`/`input`
listeners over an observer that exists only to re-attach handlers after a
re-render (`number-inputs.js` dropped its observer that way). And bump
`DEPLOY_BUILD` — clientside assets are not fingerprinted, so open tabs keep the
old file until forced to reload.

**Not on this list:** `consent-scroll-poll` (the 500 ms `dcc.Interval` behind
`consent-scroll-complete`) *looks* immortal but self-disables on its first tick.
`#consent-notice-scroll` has no `overflow` — the iframe owns the scrollbar, per
the single-scrollbar rule — so `scrollHeight == clientHeight`, `atEnd` is
trivially true, and the callback returns `disabled=True`. A side effect worth
knowing: the desktop landing page's "scroll to the end" gate is therefore
satisfied immediately, and the mobile wizard ignores the store entirely.

### localStorage hydration race condition

`dcc.Store` with `storage_type='local'` hydrates **asynchronously** after the initial server render. Each store hydrates independently — there is no guaranteed order. A callback triggered by one store hydrating as `Input` may read other stores via `State` before they have hydrated, seeing the server-default value (`None` or whatever `data=` was in the layout) instead of the persisted value.

**Rule:** When a callback needs data from multiple localStorage-backed stores to make a correct decision (e.g. `restore_page_on_load` needs both `last-visited-page` and `user-info-store`), make **all** of them `Input` — not `State`. Use a one-shot memory flag (`page-restore-done`) to prevent the callback from acting more than once. If a required store hasn't hydrated yet (`data` is still `None`), `raise PreventUpdate` to wait for the next firing.

**Corollary — "all Inputs" is not enough; never burn a one-shot guard mid-hydration.** `restore_page_on_load` had all three stores as `Input` and still failed, because hydration has an *order*: `last-visited-page` sits **after** `user-info-store` / `current-window-df` in the layout, so the first firing routinely arrives with a populated session and `last_page` still `None`. The `if not last_page: return no_update, True, ...` branch then set `page-restore-done=True`, and when `last-visited-page` finally hydrated the callback was already spent — **no resume dialog, no redirect**, the player left on the landing page whose only mobile CTA walks into the consent wizard. Intermittent by nature (it depends which store's dispatch lands first), which is why resume "worked yesterday". The guard now `raise PreventUpdate`s whenever any other session store already holds data, and only treats an all-empty firing as a genuinely fresh visitor. **Rule:** a one-shot flag may only be set on a firing that had every store it needs; "value is None" means "wait", not "absent", unless *nothing* has hydrated.

**Corollary — don't clobber stores on `/`:** Callbacks like `initialize_data_on_url_change` that write to `full-df` / `current-window-df` must **not** load fresh data when `pathname` is `/` or any non-prediction page. The URL-change callback fires before stores hydrate; overwriting them destroys the persisted session that the resume flow needs.

**Corollary — a cold load *on* a game URL renders before hydration (`display_page`).** `display_page` cannot follow the rule above: its store reads must stay `State`, because making `user-info-store` an `Input` would re-render — and so destroy — the live chart on every round. So a **full page load whose URL is already `/prediction`, `/ending` or `/final`** (Android tab restore days later, pull-to-refresh on the chart, a bookmark, F5) runs with `user_info=None`. `/prediction` then fell through every branch to the **default route: the landing page** — which on mobile leads straight into the consent wizard — and nothing ever re-rendered, because `page-content` only changes on `url.pathname` and `restore_page_on_load` bails out for any pathname other than `/` (`update_on_language_change` can't rescue it either: `/prediction` and `/ending` are in `_STATEFUL_PAGES`). **Reported August 2026: a player resumed 3 days later on Samsung/Android portrait and was dumped back on the consent form with her session intact in localStorage.**

The fix is the **placeholder-scoped poll** pattern (`_game_stores_ready` / `_restoring_layout` / `resolve_session_restore` in `app.py`):
- `_game_stores_ready(pathname, user_info, current_df)` decides whether the route can be built yet (`/ending` also needs the window store). `display_page` renders `_restoring_layout()` when it can't — never landing/consent, never "session expired".
- The `dcc.Interval` that resolves it (`session-restore-poll`) lives **inside that placeholder only**. Because a Dash callback fires only while every component it references is mounted, this re-render is *structurally* impossible mid-game — do not move that interval into the base layout, or every hydration event becomes a chart reset.
- It flips the one-shot memory store `game-stores-hydrated` (an `Input` of `display_page`), or, after `_RESTORE_GIVE_UP_TICKS`, routes to `/` so an empty-localStorage deep link doesn't spin forever.
- **Rule for new game routes:** add them to `_GAME_ROUTES` and teach `_game_stores_ready` which stores they need, or they will render from empty state on a cold load.

**Corollary — `route-prediction` must follow the *render*, not the URL.** The `<html>` class was stamped from `pathname === '/prediction'` alone, so during the bug above (URL `/prediction`, content = landing/consent) every prediction-only rule in `mobile.css` applied to that foreign content — including the two `:not(.route-prediction)` **releases**: the `#page-content * { max-width: 100% }` overflow cap (without it a form page overflows and the browser zooms the whole page out) and `touch-action: manipulation` (without it Android waits ~300 ms per tap for a double-tap-zoom and swallows taps — the documented "Next worked on the 4th click"). Net effect: a consent form the player could tick but not submit. The class is now keyed on `prediction-chart-rendered`, written by `mark_prediction_chart_rendered` from `_renders_prediction_chart(pathname, user_info)` — the same predicate `display_page` uses for its consent bounce. Keep those two in step, and keep the clientside check as `pathname === '/prediction' && chartRendered` (the pathname half drops the class instantly on navigation away; the flag half withholds it until the chart is really up).

### Slider and component persistence

Interactive Dash components (sliders, dropdowns, inputs) that are destroyed and recreated on page navigation lose their value unless `persistence=True` and `persistence_type=STORAGE_TYPE` are set. The `time-slider` on the prediction page is recreated every time `create_prediction_layout` runs (e.g. on resume). Without persistence it mounts with the layout-default value, which triggers `handle_time_slider` and re-slices `current-window-df` at the wrong position.

**Rule:** Any interactive component whose value must survive a layout rebuild (page navigation, resume, language change) needs `persistence=True, persistence_type=STORAGE_TYPE`.

### resume-dialog-target must be cleared after dismissal

`render_resume_dialog` has `Input('interface-language', 'data')` so the dialog text updates when language changes. But `resume-dialog-target` is a memory store — if it is not set to `None` when the dialog is dismissed, any later `interface-language` change (e.g. clicking a flag on `/ending`) will re-render the stale dialog on top of the current page.

**Rule:** Every callback that dismisses the resume dialog (`handle_resume_continue`, `handle_resume_start_over`) must set `resume-dialog-target` to `None` in addition to clearing `resume-dialog-container`.

### Mobile version

The app now has a genuine **mobile-first** experience (rebuilt June 2026). **Full design rationale, implementation, and the screenshot harness are documented in `docs/mobile-version.md` — read it before touching anything mobile.** The condensed pitfalls/lessons are in the "Mobile pitfalls & lessons learned" section below.

Architecture in one paragraph: the static viewport meta is **`width=device-width`** (NOT the old forced `1280`), and it now stays device-width on **every** page **including `/prediction`** — that route used to force `width=1280` for Plotly drawline, but 1280 overflowed/cropped the right ~30% (incl. Submit) in real fullscreen landscape, so it was dropped (landscape uses the real ~800-900px device width; portrait puts the chart in a horizontal scroller). A clientside callback in `app.py` keeps the meta at device-width and stamps a `route-prediction` class on `<html>` so CSS can target that page. Detection is two-pronged: a server-side `_is_mobile_ua()` check on the Flask request User-Agent picks **separate mobile builders** in `display_page` / `update_on_language_change` (`StartupPageMobile` wizard, `LandingPageMobile`, `MobileNavBar` burger menu); the existing clientside callback still adds the `html.mobile-device` class (UA + `(pointer:coarse)` fallback) used to scope `assets/mobile.css`. Display-only pages (ending, final, share, faq, about, contact, demo) just reflow via CSS once `device-width` is active. Hard rules that still hold: **do not CSS-rotate the chart** (`transform: rotate(90deg)` breaks Plotly drawline touch mapping); `render_mobile_warning()` returns `None` and `mobile-warning` is a throwaway Output sink. **The portrait "rotate to draw" nag overlay (`assets/orientation.css`, `#orientation-overlay`) was REMOVED** — it was a second, non-playable mode. The single mobile `/prediction` flowpath: the immersive landscape CSS applies the moment the phone is in landscape (gated `@media (orientation: landscape) and (pointer: coarse)` — NO `max-device-width`, which broke it on wide phones — and using `100dvh` not `100vh`), and fullscreen is entered by a user GESTURE (clientside, mobile-only) via two buttons — the wizard **Start button** and a persistent **"Fullscreen mode" button** on `/prediction` portrait — each calling `requestFullscreen(documentElement)` + best-effort `screen.orientation.lock('landscape')` (works on Android after fullscreen, rejects-caught on iOS). Fullscreen is an enhancement; the `100dvh`/device-width landscape is playable without it. Use `uv run python scripts/mobile_shots.py` to screenshot every page on a narrow phone viewport (see the doc).

### Mobile pitfalls & lessons learned

- **`html, body { min-width: 1280px }` in `assets/lang.css` is the master kill-switch for mobile.** It was a fallback for the old forced-desktop strategy and silently pins the ENTIRE page to 1280 regardless of the viewport meta. `assets/mobile.css` releases it with `min-width: 0 !important` on `html.mobile-device` (now **all** mobile pages — the `:not(.route-prediction)` carve-out was removed when `/prediction` went device-width). If mobile pages ever render desktop-width again, check this first.
- **"Where does 1280 keep popping up?" — there are TWO independent sources; kill both.** `/prediction` rendered at 1280 from (1) the clientside `<meta viewport>` switch and (2) the `lang.css` `min-width:1280px` anchor. Fixing the meta alone leaves the body pinned by `min-width` (the meta says `device-width` but `innerWidth` stays 1280). Both had to be released together for `/prediction` to actually use the device width.
- **Under `width=device-width`, any element wider than the screen makes the browser expand the layout viewport to fit it and zoom the whole page out** (real phones do this too, not just Chromium). So a single overflowing element (the navbar, a wide input, a fixed-width child) breaks mobile-first for the WHOLE page. Mobile work is largely a hunt for overflow sources.
- **The Fomantic `massive tabular menu` navbar is ~1280px wide in one row** and cannot be CSS-squeezed to fit (`width:100%` resolves against the already-expanded 1280). It needs a structurally different component — hence `MobileNavBar` (burger menu). CSS-only could never have fixed this.
- **`dcc.Input` applies the Python `style=` to the WRAPPER `div.dash-input-container`, not the real `<input class="dash-input-element">`.** The inner input keeps the classic chriddyp Dash CSS (wide default + border), rendering as an inset double-box and overflowing. Style `.dash-input-element` via CSS and flatten the wrapper.
- **Mobile builders MUST reuse every id of their desktop counterpart** (same class as the documented `ending-*` rule). The startup wizard renders ALL the same input ids grouped into `mobile-step-{i}` divs toggled by `display`; the existing validation/conditional callbacks drive it unchanged. New wizard callbacks (`navigate_startup_wizard`) live in the desktop `StartupPage.register_callbacks` so they register once; `prevent_initial_call=True` keeps them inert on desktop where the ids are absent.
- **Server-side UA detection (`flask_request.headers`) must drive layout choice, not the `user-agent` dcc.Store** — the store hydrates async from localStorage and is `None`/stale on first render.
- **Per-page viewport switching is a clientside callback that rewrites the `<meta viewport>` content**; it can't relayout reliably while a CDP device-metrics override is active (a screenshot-harness quirk), but works on real browsers. Keep `prevent_initial_call=False` and bump `DEPLOY_BUILD` when changing the JS (clientside JS isn't fingerprinted).
- **Screenshot harness (`uv run python scripts/mobile_shots.py`):** emulate with CDP `mobile:false` for device-width pages (deterministic — `mobile:true` triggers the flaky expand-to-fit). Do NOT use `captureBeyondViewport:true` (it re-lays-out at a ~1280 fallback) — instead grow the viewport height to `scrollHeight` and take a normal viewport capture; skip the height-grow for landscape and `/prediction` (their content is viewport-sized).
- **HISTORICAL / obsolete harness trap:** `/prediction` used to force a `width=1280` meta and the immersive landscape CSS was gated on `max-device-width:1024`, which needed a delicate `clearDeviceMetricsOverride` + re-apply dance to re-scale the 1280 meta and keep device-width ≤1024. **None of that applies now** — `/prediction` is device-width and the `max-device-width` gate is gone, so emulate it like any other page. If old shots crop the chart bottom, it's stale cache/CSS, not the meta trick. **Plotly only re-fits on a window `resize`** — also call `Plotly.Plots.resize(gd)` on every `.js-plotly-plot` before capturing.
- **Plotly re-fits only on a window `resize`, not on a CSS-driven container resize.** A bare `window.dispatchEvent(new Event('resize'))` races the layout — also call `Plotly.Plots.resize(gd)` on every `.js-plotly-plot` just before capturing the chart, or the SVG keeps its initial oversized height.
- **`uv run chart` runs the Dash debug reloader; its forked child re-imports the module and loses the chart-mode prefill, so `/prediction` intermittently redirects to landing.** If chart shots show landing/consent content, that (not the CSS) is the cause — keep the prefill alive across the reloader fork via the env-var pattern.
- **Number inputs: hide Dash's own `.dash-input-stepper` `−`/`+` buttons on mobile, NOT the native webkit spinner.** Newer `dcc.Input(type="number")` renders its own stepper `<button>`s inside `.dash-input-container`; once the input is full-width on mobile they wrap below as stray `−`/`+` (a11y-labelled "Decrease/Increase value", so they look native, but `::-webkit-inner-spin-button{appearance:none}` won't remove them). Use `html.mobile-device .dash-input-stepper { display:none }`. Do NOT clip them with `overflow:hidden` on the wrapper — that re-clips the input's own bottom border (the "editor boxes cropped on bottom" bug).
- **Don't let the generic `html.mobile-device input { display:block; width:100% }` rule hit checkboxes/radios** — it stretches consent checkboxes full-width and breaks the label onto the next line. Exclude them (`:not([type="checkbox"]):not([type="radio"])`) and lay each `.form-check` out as a `display:flex` row.
- **Consent reader (`/consent-form`): don't use a `height:100vh` (or `min-height:100vh`) flex shell.** `100vh` ignores the navbar above `#page-content`, pushing the "Go to start" button below the fold; `min-height:100vh` adds a second page-level scrollbar (the recurring double-scrollbar bug). Let the shell be normal flow and give the embedded iframe `height: calc(100vh - 190px)` (room for navbar + button + paddings) so the iframe owns the only scrollbar — a full-bleed single box, no nested inner card.
- **Contact links: stack the tables into one column on mobile** (`thead`/`tr`/`td` → `display:block; width:100%`, first cell bold as heading, narrow font + `overflow-wrap:anywhere; word-break:normal` on links). Wide multi-column tables truncate long emails/URLs into 1–2 char dangling overhangs that look unprofessional.
- **A hidden flex sibling is a silent dead zone — hide with `display:none`, never `visibility:hidden`.** The wizard nav row is two `flex: 1` buttons; hiding Back with `visibility` kept its box in the row while excluding it from hit testing, so on step 0 Next was only the right ~50% of the bar and every tap on the left half fell through to the `disable_n_clicks=True` container: no callback, no `:active` flash, **nothing in the server log**. A thumb aimed at the middle of the bar misses every time. Reported August 2026 as "Next is blue, pressed it a few times, zero reaction, then it worked on the first press minutes later" — and the log proves it was tap loss, not lag: zero POSTs during the gap and exactly one `navigate_startup_wizard` response when it finally moved, so at most one click was ever dispatched. **Diagnostic rule: zero POSTs in the gap means the click never happened; server latency and callback bugs both leave traces.** Related: never let a control's own activation move it — `startup-consent-hint` collapsed from `display:block` to `none` in the same repaint that turned Next blue, shifting the row up under the user's thumb; it keeps its box via `visibility` on step 0. And give every mobile button an `:active` state, or a missed tap and a slow one look identical.
- **Mobile buttons/links need `touch-action: manipulation` or taps get swallowed.** The viewport allows zoom (`user-scalable=yes`), so mobile browsers wait ~300ms per tap for a double-tap-zoom and drop rapid taps as zoom gestures — a button/link then only fires after several taps (seen on Vivaldi Android: Next "worked on the 4th click"). NOT a callback bug and NOT reproducible headless (synthetic clicks bypass the gesture wait). Fix in `mobile.css`: `touch-action: manipulation` on `a`/`button`/`.ui.button`/`[role=button]`/`label`/`input`/`.form-check`, scoped to `html.mobile-device:not(.route-prediction)` (the chart owns its own touch-action for drawline). Pinch-zoom preserved.
- **Fullscreen/immersive entry must be a clientside callback fired by a real GESTURE** — `requestFullscreen` from a route-change/store callback is rejected (no user activation). Wired to two gesture buttons (wizard `start-button` + persistent "Fullscreen mode" button on `/prediction`), each `requestFullscreen(documentElement)` + best-effort `screen.orientation.lock('landscape')` (the OLD "never use orientation.lock" rule is **superseded** — it's used after fullscreen, Android works, iOS rejects-caught) + `Plotly.Plots.resize`. Reuse the demo-video fullscreen path (proven). Don't rely on it for playability — `100dvh`/device-width landscape stands alone. Localize clientside button feedback via a `data-*` attr (`t()`-rendered server-side), e.g. `data-copied-text`.
- **Landscape `/prediction` header chips are absolute-positioned — rebalance edges together.** Round (`left+width`), Units (`right+width`), Source (`left`/`right`-pinned, so its width = `screenW − left − right`). Shrinking Round/Units does NOT widen Source unless you also move Source's `left`/`right` in. mobile.css has near-duplicate landscape blocks — append a final `@media (orientation: landscape) and (pointer: coarse)` override so it wins.
- **Secondary/occasional actions go on the between-rounds `/ending` page, not in-round `/prediction`** — the chart page has zero spare screen budget (chart + control strip fill `100dvh`). The cross-device "copy resume link" button lives on `/ending`.
- **Removing/renaming a clientside callback breaks open tabs until refresh** — a stale tab POSTs the old callback id and the server 500s with `KeyError: "Callback function not found for output '..<id>..'"`. NOT a server bug; bump `DEPLOY_BUILD` (forces fresh loads) + hard-refresh. If "new behaviour doesn't work" AND the console shows that 500, it's a stale client.
- **`/prediction` is device-width now → size its elements at native px.** During the brief 1280-scaled era, fixed elements had to be ~3.3× larger to be tappable; that trap is gone. An oddly huge/tiny `/prediction` element is likely tuned for the old 1280-scaled assumption.
- **The mobile wizard consent gate must NOT depend on scroll-to-end detection.** `consent-scroll-complete` watches the outer `#consent-notice-scroll` div, but on real mobile the user scrolls the inner consent iframe, so it never registers and Next stays hard-locked even after ticking the boxes (headless masks it — the div fits its content so `atEnd` is trivially true). Gate `gate_mobile_consent_step` on the two mandatory checkboxes only (acknowledge — which already says "18+ and have read the terms" — + GDPR). Also: a gated mobile button must LOOK disabled (the gate swaps `startup-next` to a grey `startup-next-disabled` class, not a dimmed-but-blue Fomantic button) and show a "why" hint (`startup-consent-hint`).
- **Consent enforcement is asymmetric desktop vs mobile, but `handle_start_button` must NOT depend on the consent components.** A Dash callback only fires when *every* one of its `Input`/`State` components is present in the current layout. The five `consent-*` checkboxes (`consent-acknowledge`, `consent-gdpr`, …) live in `landing.py` and render on the **desktop landing page** and inside the **mobile wizard step 0** (imported via `consent_controls_children`) — but **never on desktop `/startup`**. So if `handle_start_button` (or the form-validation callback `update_form_validation`) takes any `consent-*` as Input/State, the callback silently never fires on desktop `/startup`: the Start button activates but navigates nowhere, and the form's validation/asterisks go dead. **This bit twice (June 2026).** Rule: callbacks on `/startup` may only reference components that exist on BOTH desktop and mobile `/startup` (the demographics fields + `user-info-store`). Consent is recorded *before* Start and read from `user-info-store`: on desktop by `handle_landing_continue`, on mobile by `record_mobile_consent` (Input on the consent checkboxes, **UA-guarded to mobile** so it doesn't race `handle_landing_continue` on the desktop landing page). Both write `consent_completed=True` + the `consent_*` flags; `handle_start_button` just carries them forward and writes the consent CSV row once on Start. The `display_page` guard on `consent_completed` then gates `/prediction` (both) and desktop `/startup`; mobile `/startup` is exempt (it IS the consent entry). **Any new synthetic user (chart mode, staging node, test) must set `consent_completed=True` or the guard bounces it to landing.**
- **localStorage is device-local; cross-device resume goes through the resume code.** Session stores are all `STORAGE_TYPE=local` (per-device). The bridge is `resume_store.py` (`data/resume/<code>.json`): a server-side snapshot keyed by `user_info['resume_code']`, auto-saved at meaningful boundaries by `auto_snapshot_session` (triggers on user_info/navigation/unit/language — **not** every drawline; dataframes come via `State`). Redeem on another device via `?resume=<code>` (universal), the landing-page resume box, or the code shown on the resume dialog. The code is assigned at consent (`handle_landing_continue` / `handle_start_button`). **If you add a new game-state store, add it to `_resume_payload` / `_restore_outputs_from_code`** or it won't transfer. Resume codes are session-transfer tokens — treat like a login link, not a public id.
- **Screenshot `/ending`, `/final`, `/share` via the staging nodes, not click-through.** The harness `result` group runs `uv run start` with `_STAGING_MODE=1` and hits `/staging/ending`, `/staging/final`, `/staging/share` (→ a synthetic `/share/<id>`). These render the real builders with synthetic data — deterministic, no drawing/submit automation. The synthetic rounds don't populate the per-round metrics table on `/final` (or "Prediction Results" on `/ending`); those boxes read empty in the shots — a data quirk, not a layout bug.

## Session persistence & navigation contract

These are the expected behaviours that every change must preserve. Treat regressions here as bugs.

1. **First visit → consent form.** A new user lands on `/` (landing page with embedded consent form). She fills it in, proceeds to `/startup` → `/prediction`. No resume dialog, no redirect.
2. **Cross-session resume (localStorage).** The game can span many rounds. All session state (`user-info-store`, `full-df`, `last-visited-page`, etc.) lives in localStorage. If the user closes the browser and reopens hours later, `restore_page_on_load` detects the persisted state, and because `session-active` (sessionStorage) is gone the **resume dialog** appears asking "Continue" or "Start Over".
3. **In-session tab switching (no dialog).** While mid-game the user can click "The Study", "FAQ", "Contact us", etc. and then click "Game" to return. Navbar links use `dcc.Link` (client-side routing, no page reload), so all stores stay populated. `redirect_landing_to_game` silently redirects `/` → the last game page. **No resume dialog must appear in this flow.**
4. **Explicit exit / Start Over cleans storage.** Both the "Finish / Exit" button, the restart button on `/final`, and the "Start Over" button in the resume dialog set `last-visited-page=None` and `clean-storage-flag=True`, which wipes localStorage via a clientside callback. After cleanup the user lands on `/` as a fresh visitor.
5. **`uv run start --clean`.** Sets `_CLEAN_STORAGE=1` env var → `clean-storage-flag=True` in the layout. The clientside callback clears localStorage once on first connect. Subsequent interactions use localStorage normally. Every new browser tab connecting to the same running server also cleans once (stop the server to stop cleaning).
6. **No spurious resume dialogs.** The resume dialog must only appear on genuine fresh sessions (scenario 2). It must never pop up when switching navbar tabs (scenario 3), pressing F5 within an active session, or changing language.
7. **A cold load on a game URL resumes that page — it never falls back to consent.** Re-opening `/prediction`, `/ending` or `/final` with a full page load (Android tab restore, pull-to-refresh, bookmark, F5) shows the "restoring your game" placeholder for the moment localStorage takes to hydrate, then the real page. It must never show the landing page, the consent form or "session expired" while a session exists in localStorage. See the "cold load *on* a game URL" pitfall above; `tests/test_session_restore_hydration.py` locks this down.
8. **Consent guard.** `user_info.consent_completed` must be `True` before the game flow is reachable. `display_page` redirects `/prediction` (both devices) and desktop `/startup` to landing when it is missing. It is set in `handle_landing_continue` (desktop) and `record_mobile_consent` (mobile wizard, UA-guarded). Mobile `/startup` is exempt (it IS the consent entry). `window.localStorage.clear()` on Start Over wipes it (and the persisted consent checkboxes), forcing fresh consent.

### Key stores involved

| Store | `storage_type` | Purpose |
|---|---|---|
| `last-visited-page` | `local` | Last game-flow page (`/startup`, `/prediction`, `/ending`, `/final`). Never stores `/` or non-game pages. |
| `session-active` | `session` | `True` once the user interacts. Survives in-tab reloads (F5) but clears on tab close, distinguishing fresh sessions from reloads. |
| `page-restore-done` | `memory` | One-shot flag preventing `restore_page_on_load` from acting more than once per page load. Resets on every full reload. |
| `clean-storage-flag` | `memory` | When `True`, a clientside callback wipes localStorage and resets the flag to `False`. |
| `resume-dialog-target` | `memory` | Holds target page + round info for the resume dialog. Must be set to `None` when the dialog is dismissed. |
| `game-stores-hydrated` | `memory` | One-shot: flips `True` when the stores a game route needs have hydrated, so `display_page` (which reads them as `State`) can re-render a cold load. Written **only** by `resolve_session_restore`, whose interval lives inside `_restoring_layout`. |
| `prediction-chart-rendered` | `memory` | Server-side truth about whether the drawing chart is on screen. Gates the `route-prediction` `<html>` class so prediction-only CSS never lands on other content. |

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
- The navbar is a Fomantic UI `massive blue inverted tabular menu` (`NavBar` class in `sugar_sugar/components/navbar.py`). Left items: Game, Highscore, The Study, FAQ, Video instructions, Contact us. The mobile burger drawer (`MobileNavBar.LINKS`) must carry the same destinations — add every new nav item to **both**, plus `PUBLIC_ROUTES` (sitemap/llms.txt) and `tests/test_navbar.py` (it asserts the exact item count and order). Right side: a Fomantic `ui simple dropdown item` (`lang-dropdown`) — the trigger shows the active language's flag+label and a dropdown caret; the menu lists all 8 languages from the module-level `LANGUAGES` constant. Use the **`simple` dropdown class** (CSS-only hover) because Fomantic's JS dropdown requires jQuery init which doesn't play well with Dash. Each dropdown item is an `html.A` with `id="lang-{code}"`, so the existing `set_interface_language` callback works unchanged. Wrapper divs inside the dropdown have `disable_n_clicks=True`; the `lang-*` links do not. Navbar uses `dcc.Link` for navigation (client-side routing, no full page reload) — this preserves all `dcc.Store` values and avoids hydration races. A `redirect_landing_to_game` callback redirects `/` → last game page when the user clicks "Game" mid-session.
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

## Highscore page (`/highscore`) and player pages (`/player/<public_id>`)

A public, session-free leaderboard reachable from the navbar (desktop) and the burger drawer (mobile).

- **Two main class boards, split by the data being predicted** (`sugar_sugar/scoreboard.py`): non-diabetic data (BIG IDEAs, or a non-diabetic player's own upload) and diabetic data (D1NAMO, legacy LOOP `*_chronological.csv`, or a diabetic player's own upload). Non-diabetic traces are flatter and objectively easier, so the two classes are never ranked against each other. Rounds classify individually via `classify_round_source` (`per_round_metrics` carries each round's `data_source_name`), so a mixed run — challenge-the-unknown, type 2 / prediabetes / LADA pools, Format C — earns one slot on *each* board it has enough rounds of. `example.csv` rounds classify nowhere and stay off both boards.
- **Score = mean per-round MAE over the best `CLASS_SCORE_ROUNDS` (= `MIN_USEFUL_ROUNDS`) rounds of that class.** A cumulative 12-round MAE cannot get lucky the way a 6-round one can, so judging everyone on the same round count removes the long-run handicap (playing more rounds can only help). Runs with fewer in-class rounds stay off that board; legacy statistics rows without `per_round_metrics` fall back to `overall_mae_mgdl` when their run-level source classifies whole.
- **Badges** (rendered inside the player cell so the five-column `final-leaderboard-row` grid contract is untouched): ⚔️ **hard mode** — the entry's player predicted data foreign to their own condition (`diabetic` column of `prediction_statistics.csv` vs the board's class); 🏅 **veteran** — the identity has more than one finished game, and the name becomes a `dcc.Link` to `/player/<public_id>`.
- **`/player/<public_id>` is a public per-player statistics page**: badges, best score per class, and every finished game with timestamp, format, per-class MAE and overall MAE. `public_id` is a domain-prefixed HMAC of the leaderboard identity under `deployment_salt()` (like share ids), so the URL exposes neither `study_id` nor `email_key`; an unknown id renders a not-found state. Wired in both `display_page` and `_update_language_page`; not in `_STATEFUL_PAGES` and never in the `last-visited-page` allowlist.
- **Source of truth for the class boards is `prediction_statistics.csv`** (one row per `study_id` + `run_id`, `per_round_metrics` parsed with `ast.literal_eval` — it is a Python-repr string, not JSON), joined with the ranking CSVs for nicknames only. The raw email is reduced to `email_key()` inside `scoreboard.py` and never leaves it. `/final` still reads the ranking CSVs through `_leaderboard_snapshot` unchanged. All these CSVs are **gitignored** and header-only in a fresh checkout, so the page legitimately renders its "no ranked games yet" empty state locally — seed synthetic rows if you need to look at a populated board. The landing "games played so far" counter reads **all** starters from `prediction_statistics.csv`, then "out of which N completed the task" (unique people with at least `MIN_USEFUL_ROUNDS` in any category).
- **`/final` keeps the old shared building blocks** — `_leaderboard_hero_children` and `_leaderboard_board` now render only there; `/highscore` uses `_scoreboard_class_board` (same `final-leaderboard-*` CSS classes and cell grid). Players who picked a nickname are shown by it; the rest stay anonymous (`Player N`). Never render `study_id` or `email_key`.
- **Every format run is stored**, even at 2 rounds. `save_statistics` upserts `prediction_statistics.csv` on ``study_id`` + ``run_id`` (not study_id alone) so switching A→B cannot erase A. Archived ``runs_by_format`` runs are rewritten on later saves. Ranking CSVs get the same rows for bookkeeping; **the public board still hides runs below `MIN_USEFUL_ROUNDS`** (default 6). A Start-only stub (0 rounds) stays in statistics only. Chart Exit (`write_ranking=False`) still skips ranking.
- **Arcade placement: one slot per finished game, among completers.** Both the class boards and `/final`'s `_ranking_entries` do no grouping at all — every rankable run is its own board slot, so beating your own score leaves the old one standing below it, and one player can hold several slots. Ties break on the earlier timestamp (whoever got there first sits higher). Each slot keeps the nickname stored per `study_id` — the name that score was set under, not the player's current name. Do not "tidy" this by deduping; hiding a player's earlier *ranked* runs is the confusing behaviour this replaced. **Display filter:** runs with fewer than `MIN_USEFUL_ROUNDS` rounds (of the class, for class boards) are excluded and show as unranked on `/final` and the share card/image. Rank is lowest score; the floor is what stops a 1-round lucky score beating a full run.
  - On the class boards `players` = distinct identities across both boards (`Scoreboard.player_count()`); the board subtitle uses `ui.highscore.scores_count`.
  - The `Rounds` column (`ui.final.col_rounds`) on a class board shows the run's rounds *of that class* — the pool the best-N score was drawn from. The `When` column (`ui.final.col_when`) is the statistics `timestamp` stacked as date + `HH:MM`. `grid-template-columns` in `lang.css` and `mobile.css` must stay in sync with the five cells.
  - `mode=` survives on `_leaderboard_snapshot` / `_rank_from_ranking_csv` as an ignored no-op so older call sites still type-check; placement no longer depends on it.
- **`_ranking_identities` / `_match_identity` are still NOT wired into any page.** The individual stats page they were parked for landed as `/player/<public_id>`, but it is built on `scoreboard.py` (per-run rows from the statistics CSV) rather than these ranking-CSV rollups. `tests/test_ranking_identity.py` still exercises them so they cannot rot; delete or reuse them only as a deliberate decision, not as dead-code cleanup.

### Nicknames and the `email_key` (`sugar_sugar/nickname.py`)

- **The nickname is NOT study data.** It is a public display label and lives **only** in `prediction_ranking*.csv` — deliberately absent from `prediction_statistics.csv` and `consent_agreement.csv`, and unmentioned in `consent_notice_text.py`. That is why it needs no consent wording. Do not "helpfully" add it to the research export; `ui.highscore.privacy_note` promises otherwise on the page itself.
- **The ranking CSVs store `email_key`, never the address.** `email_key()` is a salted HMAC-SHA256 of the casefolded address, truncated to 16 hex chars, used purely to merge one player's rows across devices. The raw address stays in `prediction_statistics.csv` as before, matching the consent notice's promise that the study-ID↔email mapping lives in a separate encrypted file.
- **The salt must never be rotated.** `RANKING_EMAIL_SALT` (env) wins; otherwise a random salt is generated once into `data/.ranking_salt` (gitignored, `0600`). Changing it changes every `email_key` and re-splits existing players into new identities.
- **One nickname per `study_id`; earlier studies are history.** `SubmitComponent.set_study_nickname` matches on `study_id` **only** — never on `email_key` — so a returning player picking a new name does not rewrite their older rows, and each board slot keeps the name it was set under. The `/final` box is *seeded* from `stored_nickname()` (the newest name recorded against the identity) as a suggestion, which is what makes a name follow a player onto a new device.
- **`backfill_leaderboard_identity` converts pre-nickname CSVs on first boot** (`SubmitComponent.__init__`, gated by the module-level `_IDENTITY_BACKFILL_DONE` so the per-render `SubmitComponent(locale=...)` in `create_prediction_layout` does not re-read five CSVs). It adds the two columns and derives each historical row's `email_key` from the address `prediction_statistics.csv` already holds for that `study_id`. Idempotent, atomic (`.tmp` + `replace`), and it never merges or removes a slot — it only links a player's own history so a new device highlights it and `stored_nickname` can find it. Rows whose study_id has no recorded address keep a blank key.
  - **It snapshots each file to `<name>.pre-nickname.bak` before its first write** (gitignored via `*.pre-nickname.bak`). `_backup_before_conversion` deliberately **never overwrites an existing backup** — a later boot would copy already-converted content over the only pristine record. Rollback is `cp data/input/prediction_ranking.csv.pre-nickname.bak data/input/prediction_ranking.csv`.
- **`nickname-input` (on `/startup`) vs `final-nickname-*` (on `/final`) are deliberately different ids.** A Dash callback only fires when every one of its components is in the layout, so reusing `nickname-input` on `/final` would drag the startup validation callback there and crash on its missing Outputs. It is also why `nickname-input` had to be added to **both** startup builders (desktop `StartupPage` and mobile `step0` → `mobile-step-1`) — `handle_start_button` takes it as `State`, and a missing id makes Start activate but navigate nowhere. `tests/test_startup_nickname_field.py` asserts exactly that. The field is optional, so it is intentionally **not** an Input of `update_form_validation` and `validate_startup_form` is untouched.
- On mobile the nickname belongs to `mobile-step-1` (identity), never `mobile-step-0` (consent) — it must not read as a consent item.
- **Styling reuses the `final-leaderboard-*` classes** (`assets/lang.css`) with a `highscore-page` wrapper for page-level layout and `assets/mobile.css` for the narrow-viewport column widths. Note the global `body :not(h1..h6) { font-size: 16px !important }` rule beats inherited font sizes on the header cells — the highscore page re-asserts 12px via `.highscore-page .final-leaderboard-row.head .final-leaderboard-cell`.
- The route is wired in **both** `display_page` and `update_on_language_change` (language switches re-render it), and it is *not* in `_STATEFUL_PAGES` nor in the `last-visited-page` allowlist — it is an info page, not part of the game flow.

## Share page (`/share/<share_id>`)

A public, read-only page that lets a user broadcast their Sugar Sugar performance. Key invariants:

- **Share records live on disk**, not in `dcc.Store`. The store at `data/shares/<share_id>.json` is written atomically by `sugar_sugar/share_store.py`. This is deliberate: the URL must work for anyone, across devices, after localStorage is wiped.
- **`/share/<share_id>` must render without any `dcc.Store` data**. `display_page` and `update_on_language_change` both load the record from disk via `share_store.load_share` and pass it into `create_share_layout`. If the id is missing/corrupt they show `create_expired_layout`, never a crashed page.
- **Two sibling Flask routes**, not Dash pages:
  - `GET /share/<id>/image.png` — kaleido renders `build_share_card_figure` to a 1200x630 PNG. Results are cached in `_SHARE_PNG_CACHE` (a module-level dict) so repeated loads don't respawn Chromium.
 - `GET /share/<id>/og` — crawler-only minimal HTML with Open Graph + Twitter Card meta tags and no meta-refresh. Humans hitting `/og` are redirected server-side (`302`) to the real Dash page. Needed because FB/X/WhatsApp/LinkedIn crawlers don't execute JS and would otherwise see the Dash shell with no OG tags. Social share buttons always link to the regular `/share/<id>` URL — crawler user agents get the OG response at that URL via `before_request`.
- **Twitter/X OG footguns (full detail in `docs/share-ops.md` → "Twitter/X OG Footguns"):** X is the strictest, most opaque OG consumer; every item here bit production. (1) **Twitterbot obeys `robots.txt`** — FB/WhatsApp/LinkedIn/Telegram ignore it for OG fetches — so `Disallow: /share/*/image.png` makes the card image vanish **on X only**, reproducibly. Never disallow the card image; keep per-share PNGs out of search via `X-Robots-Tag: noindex` on the image route instead. (2) **`Content-Disposition: attachment` kills the card** — the image route serves `as_attachment=False` (inline); the Download button forces download client-side via the HTML `download` attr. (3) **X share URL must be `twitter.com/intent/tweet?text=…&url=…`, never `/intent/post`** — `/intent/post` isn't a real intent path, so the mobile X app opens then bounces ("share opens app then closes"). (4) **X retired the Card Validator (2022)** — no official re-scrape; third-party validators bypass robots.txt AND X's cache, so a green preview does NOT prove the live card works. (5) **X caches a card per-URL ~7 days** — force a fresh scrape by posting `/share/<id>?r=1` (X keys cache on full URL; crawler hook matches on path so OG is identical). (6) **New shares are inherently fresh URLs** (unique ids) → scrape clean on first post; `SHARE_CARD_IMAGE_VERSION` (`?v=` on the image) only refreshes *already-posted* URLs after a card redesign. Diagnosis: "works everywhere except X, reproducibly" → robots.txt or attachment; "validator green but tweet blank" → stale per-URL cache, re-share with `?r=1`.
- **`kaleido`** is a hard dependency (see `pyproject.toml`). First render takes ~1 s (spawns Chromium); subsequent are served from the cache. Do NOT hot-reload the server while kaleido is rendering — it can leave orphaned Chromium processes on Windows.
- **Share flow is part of `/final` — no button, no callback, no navigation.** `create_final_layout` builds a lean JSON-safe record via `build_final_share_record` (rounds across all formats, frozen rankings, limited `user_info` keys, locale, timestamp), persists it with `share_store.ensure_share`, and renders the share panel (`build_share_panel` in `components/share.py`: download PNG, copy link, social buttons) as a regular section right after the leaderboard (the share impulse peaks at the ranking; graph and metric detail follow). The share id is **content-addressed** — a salted HMAC (`nickname.deployment_salt()`, domain-prefixed `share-id:`) of rounds + trimmed user_info, excluding `created_at`/`locale`/`rankings` — so re-renders (language change, revisits) reuse the same file/URL instead of minting one per render, while a new round (or nickname change) yields a fresh id; `created_at` and rankings freeze at the first render of that game state. `/share/<id>` remains the recipient-facing public page the social links point at; its "Play again" button is excluded from the `/final` panel (`include_play_again=False`). The record intentionally drops heavyweight stores (`full-df`, `events-df`) — everything the share page needs already lives in `prediction_table_data`. Tests: `tests/conftest.py` autouse-redirects `SUGAR_SHARE_DIR` to tmp because any test rendering `/final` now writes a record.
- **Encouragement text** (`sugar_sugar/encouragement.py`) is template-based today, keyed by a score bracket derived from overall MAE. A module-level `LLM_BACKEND: Optional[Callable]` is the swap point if you want to plug in a real LLM later; do not sprinkle LLM calls elsewhere.
- **`data/shares/` is gitignored** — share records are session data, not source code.
- **Synthesis graph on the share page** is one **stacked Plotly row per data-source format the user played (A / B / C)**. The **x-axis** is **time in the next hour** with a tick for **every 5-min step** (count = `PREDICTION_HOUR_OFFSET`, default 12). **Y = percent off actual** — `(pred − actual) / actual × 100` (skip if actual is 0). If the **first** value in the next-hour window has **no prediction**, **actual** at that step is imputed for display so the series starts. Each format row has a **tinted panel**; **colours are data-scaled per row**: `ref = max |% error|` over all rounds in that format, and each point blends its line colour toward **neutral grey (128,128,128)** by `(|y|/ref)^γ` (γ≈0.38), so the **worst |error| in that subplot** reads as **literal grey**. **Stacked fill bands** keep more intensity near the 0% axis. A **continuous solid black, thick 0%** line sits **above fills, below lines/markers**. The old gradient bar and the "accuracy %" stat tile are gone; MAE/RMSE stay on the card.
- **Rankings are shown per data-source category (example/generic, mixed, own) AND overall** on the share page, derived from `is_example_data` / `data_source_name` on each round record. The overall ranking comes first in the layout, followed by per-category rankings. The redundant per-round metrics table below the synthesis graph was removed on purpose — do not reintroduce it.
- **Clientside persist allowlist** in `app.py` only writes `/startup`, `/prediction`, `/ending`, `/final` to `last-visited-page`. `/share/*` is automatically excluded by the allowlist; do not add it.
- **Mobile**: the share page reuses `.info-page` + scoped `.mobile-device .share-page` rules in `assets/share.css` so the download/copy buttons stay readable on narrow viewports. The portrait "rotate" overlay was removed (it was `route-prediction`-only anyway), so `/share/<id>` just works in portrait.
