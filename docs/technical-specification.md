# Technical specification

Maintainer-facing reference for how Sugar Sugar is wired, where each piece runs,
and how to regenerate bundled assets. For participant-facing setup see the root
[README](../README.md). For mobile layout and share/OG details see the linked
topic docs at the end.

This document is organised **by where a feature acts** (page or layer), not by
Python module name.

---

## Document map

| Topic | Where it acts | Deep dive |
|-------|---------------|-----------|
| Location autocomplete | `/startup` form (`#location-input`) | [§ Startup form](#startup-form-location-autocomplete) |
| Session resume / navigation | All game pages (`/startup` … `/final`) | [§ Session & navigation](#session--navigation) |
| Landing & consent | `/`, `/consent-form` | [§ Landing & consent](#landing--consent) |
| Prediction chart | `/prediction` | [§ Prediction chart](#prediction-chart) |
| Share page & OG cards | `/share/<id>`, Flask OG routes | [docs/share-ops.md](share-ops.md) |
| Mobile layout | All routes (UA + CSS) | [docs/mobile-version.md](mobile-version.md) |
| Bundled browser assets | `assets/` (auto-served by Dash) | [§ Asset build & cache busting](#asset-build--cache-busting) |
| Study CSVs | `data/input/*.csv` (server disk) | [§ Study CSVs](#study-csvs) |

---

## Startup form — location autocomplete

### What the user sees

On **desktop and mobile `/startup`**, the **Location** field (`#location-input`)
shows a dropdown of matching countries and cities as the user types. Suggestions
appear in the active UI language (e.g. `München, Deutschland`, `Москва, Россия`).
If nothing matches, the user can still type any free-text location — autocomplete
is optional; validation only requires a non-empty value.

Autocomplete is **client-side only**: no Python callback runs on keystrokes.

### Where each layer lives

| Layer | Path | Role |
|-------|------|------|
| Form field + host wrapper | `sugar_sugar/components/startup.py` | Renders `#location-input` inside `.location-autocomplete-host` (desktop layout and mobile wizard step with location). |
| Route init ping | `startup.py` → clientside callback | On `url.pathname` change, calls `window.sugarSugarLocationAutocomplete.refresh(pathname)` so autocomplete attaches after navigation/resume. |
| Memory store sink | `sugar_sugar/app.py` | `dcc.Store(id='location-autocomplete-ping')` — clientside output only. |
| Browser logic | `assets/location-autocomplete.js` | Debounced filter, dropdown UI, keyboard navigation. **Fetches the corpus only once the user types in the field** — never at import or on navigation. |
| Dropdown styling | `assets/location-autocomplete.css` | Host `overflow: visible`, z-index, mobile overrides. |
| Suggestion data (browser) | `assets/location-suggestions.<locale>.json` | One compact file per locale (~88 KB, ~29 KB gzipped): ~2k rows of countries + up to 10 cities per country, labelled in that locale only. |
| Catalog builder (Python) | `sugar_sugar/location_catalog.py` | Merges country i18n, city lists, and per-city locale overrides into `CITY_SPECS`. |
| Country list | `sugar_sugar/location_countries.py` | Canonical `COUNTRY_NAMES` tuple (197 countries). |
| City source data | `sugar_sugar/build_city_data.py` | Embedded `TOP_CITIES_BY_COUNTRY` (~10 cities per country). |
| Generated city JSON | `sugar_sugar/data/top_cities_by_country.json` | Written by `build_city_data`; read at import (with embedded fallback). |
| City locale overrides | `sugar_sugar/location_city_i18n.py` | Native spellings + extra search tokens (e.g. Kyiv/Kiev, München). |
| Filter + all-locale dump | `sugar_sugar/location_suggestions.py` | `filter_location_suggestions()` (server-side filter), `write_suggestions_asset()` (debug dump to `data/`, **not** a shipped asset). |
| Per-locale asset build | `sugar_sugar/build_locations.py` | `uv run build-locations` → `assets/location-suggestions.<locale>.json`. |
| Tests | `tests/test_location_suggestions.py` | Filter logic, per-locale asset sync, per-country city counts, and a guard that the 824 KB all-locales file never returns to `assets/`. |

### Data model

Each row of `location-suggestions.<locale>.json` is a compact array —
`[label, rank]`, or `[label, rank, extra_tokens]`:

```json
[
  ["Berlin, Deutschland", 0],
  ["München, Deutschland", 1, ["munich, germany", "munchen", "munich"]]
]
```

- **`label`**: the only spelling shipped for this locale. The runtime derives the
  lowercase and ASCII-folded search tokens from it, which is why they are not
  stored.
- **`extra_tokens`**: only what the label cannot yield — alternate spellings and
  aliases ("munchen", "peking", "nyc"). Other locales' full "City, Country"
  labels are deliberately dropped: they were the bulk of the old corpus, and the
  label's own prefix already matches the city half. Replayed over 5636 realistic
  queries, the top suggestion is unchanged and the 8-item tail differs in 5.8%.
- **`rank`**: city position within its country (0 = largest/capital first). Used to
  sort prefix matches so `ber` → Berlin before Berat.
- **Countries** use `rank: 1000` so city matches surface first when both match.

**Why per-locale and lazy:** everything in `assets/` is served on every page. The
old single 824 KB all-locale file was fetched, parsed and re-mapped at script
eval, on `DOMContentLoaded` *and* on every navigation — on `/faq`, on the chart,
on the consent step — for a field that lives on wizard step 3 and that most
players never reach. On a low-spec Android that is seconds of main thread and a
large object graph bought before the user has consented.

### Commands — edit city / country data

**1. Edit city lists** (add cities, reorder by importance):

Edit `TOP_CITIES_BY_COUNTRY` in `sugar_sugar/build_city_data.py`, then:

```bash
uv run python -m sugar_sugar.build_city_data
```

Writes `sugar_sugar/data/top_cities_by_country.json` (197 countries, capped at 10
cities each).

**2. Add localized city names or search aliases:**

Edit `CITY_I18N` in `sugar_sugar/location_city_i18n.py` (keyed by
`(city_en, country_en)`).

**3. Add or fix country translations:**

Edit `COUNTRY_I18N` in `sugar_sugar/location_catalog.py`. Countries without an
entry fall back to the English name in all locales.

**4. Regenerate the browser bundles** (required after any catalog change):

```bash
uv run build-locations
```

Writes `assets/location-suggestions.<locale>.json` for all eight locales. The
on-disk assets must match Python
(`tests/test_location_suggestions.py::test_per_locale_assets_match_python_source`).

Do **not** run `uv run python -m sugar_sugar.location_suggestions` to produce a
shipped asset — that dumps the whole all-locales catalog, and it writes to
`data/` precisely so it cannot end up on the wire again.

**5. Run tests:**

```bash
uv run pytest tests/test_location_suggestions.py -q
```

**6. Bump cache buster and restart:**

After changing JS or JSON assets, bump `DEPLOY_BUILD` in `sugar_sugar/config.py`
and hard-refresh open tabs (stale clients can POST obsolete callback ids).

### Troubleshooting — location autocomplete

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| No dropdown when typing | Not on `/startup`, or consent not completed (`display_page` redirects to landing) | Complete consent → `/startup`; or use `uv run chart` only for chart debugging (no startup form). |
| No dropdown after code change | Stale browser cache / stale per-locale asset | Hard refresh; bump `DEPLOY_BUILD`; type 2+ characters and confirm `/assets/location-suggestions.<locale>.json` appears in the Network tab (it is not requested before you type). |
| Dropdown clipped / hidden | Parent `overflow: hidden` | Host must have `location-autocomplete-host`; CSS sets `overflow: visible`. |
| Works in EN but not DE/RU/ZH | Missing `CITY_I18N` / `COUNTRY_I18N` entry, or that locale's asset was not rebuilt | Add locale in `location_city_i18n.py` or `COUNTRY_I18N`; `uv run build-locations`. |
| `ber` shows obscure cities first | Missing or wrong `rank` in asset | `uv run build-locations`; cities are ranked 0–9 within each country in `build_city_data` order. |
| Server 500 on navigation after JS change | Stale tab with old clientside callback id | Bump `DEPLOY_BUILD`, hard-refresh all tabs. |
| Tests fail on asset sync | Forgot to rebuild after a catalog edit | Run step 4 above. |
| Import error on fresh clone | Missing `top_cities_by_country.json` | Run `uv run python -m sugar_sugar.build_city_data` once (catalog falls back to embedded dict but JSON should be committed). |

### Manual smoke test

1. `uv run start` (or normal flow through landing → startup).
2. Open `/startup`, focus **Location**, type at least 2 characters (`mun`, `ber`, `моск`).
3. Confirm localized suggestions; pick one or leave custom text.
4. Change language via navbar — suggestion labels should follow on next keystroke.

---

## Session & navigation

### Where it acts

| Page / flow | Key stores | Callbacks (indicative) |
|-------------|------------|-------------------------|
| All game pages | `last-visited-page`, `session-active`, `user-info-store`, `full-df`, … | `restore_page_on_load`, clientside persist |
| Resume dialog | `resume-dialog-target`, `page-restore-done` | `handle_resume_continue`, `handle_resume_start_over` |
| Cross-device resume | `data/resume/<code>.json` | `resume_store.py`, `?resume=<code>` on landing |

### Commands

```bash
uv run start --clean          # wipe localStorage on first connect (every tab once)
```

### Troubleshooting

| Symptom | Fix |
|---------|-----|
| Resume dialog on every navbar click | Bug — should only appear on fresh sessions; check `session-active` and persist allowlist. |
| Resume dialog never appears | Stores not hydrating; check `STORAGE_TYPE=local` and that game pages were visited. |
| State lost on `/` visit | `initialize_data_on_url_change` must not clobber stores on non-`/prediction` paths. |

Full contract: root [README § Resume and study integrity](../README.md#resume-and-study-integrity) and `AGENTS.md` → Session persistence.

---

## Landing & consent

### Where it acts

| Route | Component | Notes |
|-------|-----------|-------|
| `/` | `landing.py` / `LandingPageMobile` | Consent checkboxes, resume code entry |
| `/consent-form` | Consent reader iframe | Single scrollbar owned by iframe — do not wrap in outer scroll |

### Troubleshooting

| Symptom | Fix |
|---------|-----|
| Double scrollbar on landing | Remove outer `overflowY: auto` around consent iframe |
| Start button dead on desktop `/startup` | Callback must not reference `consent-*` ids absent on that layout |
| Mobile wizard Next locked on consent | Gate on checkboxes only, not scroll-to-end of outer div |

---

## Prediction chart

### Where it acts

| Route | Component | Notes |
|-------|-----------|-------|
| `/prediction` | `glucose.py`, `predictions.py`, `app.py` | Drawline chart, time slider, submit |

### Commands

```bash
uv run chart                              # skip landing/startup/consent
uv run chart --prefill                    # pre-fill prediction region (test submit flow)
uv run chart --prefill --noise 0.10
uv run chart --file /path/to/export.csv --unit mmol/L --locale de
```

### Troubleshooting

| Symptom | Fix |
|---------|-----|
| `uv run chart` lands on landing | Debug reloader fork — env-var prefill pattern; restart once |
| Time slider wrong after resume | `time-slider` needs `persistence=True` |
| Mobile drawline offset | Do not CSS-rotate chart; see mobile doc |

---

## Share page & social previews

See **[docs/share-ops.md](share-ops.md)** for record lifecycle, OG routes, kaleido,
Twitter/X footguns, and staging share nodes.

### Commands (quick)

```bash
uv run share
uv run share --formats "A,B,C" --rounds 12 --locale de
uv run python scripts/render_share_card_previews.py
uv run setup-chrome
```

---

## Mobile layout

See **[docs/mobile-version.md](mobile-version.md)** for viewport meta, immersive
landscape `/prediction`, wizard steps, screenshot harness, and pitfalls.

### Commands (quick)

```bash
uv run python scripts/mobile_shots.py
uv run python scripts/mobile_shots.py --only chart --device iphone-se
```

---

## Asset build & cache busting

Dash serves everything under `assets/` at `/assets/<filename>`. Files changed at
runtime (location JSON, autocomplete JS/CSS) are **not** fingerprinted by webpack;
browsers cache them aggressively.

| Asset | Regenerate with |
|-------|-----------------|
| `assets/location-suggestions.<locale>.json` (8 files) | `uv run build-locations` |
| `sugar_sugar/data/top_cities_by_country.json` | `uv run python -m sugar_sugar.build_city_data` |
| Clientside JS behaviour | Edit `assets/*.js`, bump `DEPLOY_BUILD` in `config.py` |

`DEPLOY_BUILD` is injected into the app shell so clients pick up new asset/callback
versions after deploy.

---

## Study CSVs

Server-side research exports live under `data/input/` (gitignored except empty
headers). They are **not** `dcc.Store` state: a visitor's browser never uploads
these files. Writers: `sugar_sugar/consent.py` (consent) and
`SubmitComponent.save_statistics` (stats + ranking). Chart-mode
(`uv run chart`) skips writes.

One **person** is a `study_id`. One **format run** (A, then later B, …) is a
`run_id`. Stats and ranking upsert on `study_id` + `run_id`, so switching
format must not erase the previous run.

| File | One row is | Written when |
|------|------------|--------------|
| `consent_agreement.csv` | one person (`study_id`) | Consent / Start; later upload-consent upserts the same row |
| `prediction_statistics.csv` | one format run | Start (0-round stub), every Submit, Finish/Exit |
| `prediction_ranking.csv` | one finished game, `format=ALL` (cumulative) | Submit / Finish from `/ending` (not chart Exit) |
| `prediction_ranking_{A,B,C}.csv` | one format run | Same as ranking, only that format |

The public `/highscore` board reads the ranking CSVs and hides runs below
`MIN_USEFUL_ROUNDS` (default 6). Statistics keep every submitted round,
including 1-round and Start-only stubs.

### `consent_agreement.csv`

Consent flags only. No predictions, no nickname, no email. `paper_full_name` is
the publication-opt-in exception (acknowledgments), not a leaderboard label.

| Column | Meaning |
|--------|---------|
| `study_id` | Stable person id (UUID). Join key to the other CSVs. |
| `number` | Sequential study number (from stats `max(number)+1`). |
| `timestamp` | Consent (or last upsert) time, `YYYY-MM-DD HH:MM:SS`. |
| `gdpr_consent` | Mandatory GDPR box. |
| `upload_own_data` | Player ticked “I will upload my CGM”. |
| `play_only` | Always `False` now (legacy column; old sessions may still say `True`). |
| `participate_in_study` | Always `True` after 18+ + GDPR. |
| `receive_results_later` | Optional: email results later. |
| `keep_up_to_date` | Optional: project updates. |
| `no_selection` | True when neither optional email box was ticked. |
| `consent_use_uploaded_data` | Late consent that the uploaded file may be used. |
| `consent_use_uploaded_data_timestamp` | When that late consent was given. |
| `paper_mention` | True when the player asked to be named in a later paper and entered a full name. |
| `paper_full_name` | Name for the acknowledgments list. Use only with `paper_mention` and ≥12 rounds. |

### `prediction_statistics.csv`

The research record. Metrics are always **mg/dL**, regardless of the UI unit.

| Column | Meaning |
|--------|---------|
| `study_id` | Person. |
| `run_id` | This format run. Empty on some pre-`run_id` rows. |
| `number` | Sequential study number. |
| `timestamp` | Last save of this run. |
| `email` | Address as entered (separate from ranking; see consent notice). |
| `format` | `A` public (anonymized traces from other people), `B` own data, `C` mixed (odd public / even own). |
| `is_example_data` | Run-level flag: last round only. **Do not use this to classify Format C rounds.** |
| `data_source_name` | Run-level source: last round only. Format A may be `BIGIDEAS-001.csv` / `D1NAMO-002.csv`; B is the upload filename; C is whichever side played last. Use `per_round_metrics` for the real list. |
| `age`, `user_id`, `gender`, `uses_cgm`, `cgm_duration_years`, `diabetic`, `diabetic_type`, `diabetes_duration`, `location` | Demographics from `/startup`. `user_id` is the adapter default (`1`), not a public id. |
| `generic_intervention` | Format A source policy for this player (`bigideas`, `d1namo`, `mix_t2`, or `mix:bigideas=0.50,d1namo=0.50`). Empty on older rows. |
| `challenge_unknown` | True when the player opted into Challenge the unknown (formats A/C, non-diabetic or type 1 only). |
| `challenge_unknown_pct` | Opposite-pool share. Always `50` when the challenge is on; empty when it is off. Older rows may still hold a slider value (10–100). |
| `paper_mention` | True when the player asked to be named in a later paper and entered a full name. |
| `paper_full_name` | Full name for the acknowledgments list. Only use it when `paper_mention` is true and the player completed at least 12 rounds. |
| `rounds_played` | Count of completed rounds in this run (`0` = Start stub). |
| `predicted_values` | Python-literal list of `{version, round, value}` (prediction, mg/dL). |
| `real_values` | Same shape: ground truth, mg/dL. |
| `prediction_times` | Same shape: window timestamps (`YYYY-MM-DD HH:MM:SS`). |
| `overall_mae_mgdl`, `overall_mse_mgdl`, `overall_rmse_mgdl`, `overall_mape_pct` | Aggregate over every point in the run. |
| `per_round_metrics` | Python-literal list — **this is the per-round trace**. |

`predicted_values` / `real_values` / `prediction_times` are aligned by index.
Each item looks like `{'version': 'A', 'round': 2, 'value': '119.6'}`. Parse
with `ast.literal_eval` (they are `str(list)`, not JSON).

#### `per_round_metrics` (the per-round source of truth)

```python
{
  "round_number": 2,
  "mae": 8.4,
  "mse": 93.4,
  "rmse": 9.7,
  "mape": 7.7,
  "data_source_name": "BIGIDEAS-001.csv",  # or the upload filename
  "is_example_data": True,                 # False = own uploaded file
  "generic_slice_key": "a1b2c3…",          # SHA-256 of time+glucose in the window
}
```

| Format | Typical `data_source_name` | `is_example_data` | `generic_slice_key` |
|--------|----------------------------|-------------------|---------------------|
| A | `BIGIDEAS-{id}.csv` or `D1NAMO-{id}.csv` (one subject per round) | `True` | Window fingerprint; used to avoid repeating the same generic slice |
| B | Upload basename (e.g. `Clarity_Export.csv`) — same file every round | `False` | Distinguishes **which window** of that file |
| C | Alternates: odd = generic subject, even = upload basename | `True` / `False` | Same meaning as A or B for that round |

The slice key is content-based (`time` + `gl` rounded to 0.1), not a file
offset. Two rounds with the same key used the same glucose window.

**Older rows** (before this field was persisted) have metrics only, or a
run-level `data_source_name` of `example.csv`. Those cannot be backfilled.

### `prediction_ranking.csv` and `prediction_ranking_{A,B,C}.csv`

Leaderboard bookkeeping. **No email, no nickname in the study/consent files.**
`nickname` and `email_key` live only here.

| Column | Meaning |
|--------|---------|
| `study_id`, `run_id`, `number`, `timestamp` | Same as stats. |
| `email_key` | Salted HMAC-SHA256 of the casefolded address, 16 hex chars. Merges one player across devices. Never rotate `RANKING_EMAIL_SALT` / `data/.ranking_salt`. |
| `nickname` | Optional public display label. Absent from stats and consent on purpose. |
| `format` | `A` / `B` / `C` on the per-source files; `ALL` on the overall file (cumulative across formats). |
| `rounds_played` | Overall file is cumulative (12 then 24 if they played two formats). |
| `is_example_data`, `data_source_name` | Run-level only (last source, or `multiple` on `ALL` when sources differ). |
| `overall_mae_mgdl` … `overall_mape_pct` | Same units as stats. |

Arcade placement: one slot per finished game among completers; ties break on
earlier timestamp. Display filter: hide `rounds_played < MIN_USEFUL_ROUNDS`.

### Related writers

```
sugar_sugar/consent.py              # consent_agreement.csv
sugar_sugar/components/submit.py    # stats + ranking (save_statistics)
sugar_sugar/nickname.py             # email_key / nickname rules
sugar_sugar/app.py                  # append_round_from_window (per-round fields)
```

---

## Command cheat sheet (by task)

| Task | Command |
|------|---------|
| Run app (dev) | `uv run start` |
| Run app (production) | `uv run serve --host 0.0.0.0 --port 8050` |
| Staging routes | `uv run serve-staging` |
| Clear localStorage (dev) | `uv run start --clean` |
| Chart only | `uv run chart [--prefill]` |
| Share page dev | `uv run share [--formats "A,B,C"]` |
| Regenerate city JSON | `uv run python -m sugar_sugar.build_city_data` |
| Regenerate location autocomplete JSON | `uv run python -m sugar_sugar.location_suggestions` |
| Location autocomplete tests | `uv run pytest tests/test_location_suggestions.py -q` |
| Mobile screenshots | `uv run python scripts/mobile_shots.py` |
| Share card PNG previews | `uv run python scripts/render_share_card_previews.py` |
| Install Chrome for kaleido | `uv run setup-chrome` |
| Format A public datasets | `uv run download` (`--all` adds CGMacros) |
| Full test suite | `uv run pytest` |

---

## Related files (quick index)

```
assets/
  location-autocomplete.js      # browser autocomplete
  location-autocomplete.css
  location-suggestions.*.json   # generated per locale — commit after regen

sugar_sugar/
  location_countries.py         # COUNTRY_NAMES
  location_catalog.py           # COUNTRY_I18N + CITY_SPECS builder
  location_city_i18n.py         # per-city locale overrides
  build_city_data.py            # TOP_CITIES_BY_COUNTRY source
  location_suggestions.py       # filter + write_suggestions_asset()
  data/top_cities_by_country.json

sugar_sugar/components/startup.py   # #location-input, clientside init
tests/test_location_suggestions.py
```
