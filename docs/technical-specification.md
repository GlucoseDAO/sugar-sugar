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
| Browser logic | `assets/location-autocomplete.js` | Debounced filter, dropdown UI, keyboard navigation; loads JSON once via `fetch`. |
| Dropdown styling | `assets/location-autocomplete.css` | Host `overflow: visible`, z-index, mobile overrides. |
| Suggestion data (browser) | `assets/location-suggestions.json` | ~2k rows: countries + up to 10 cities per country, 8 locale labels each. |
| Catalog builder (Python) | `sugar_sugar/location_catalog.py` | Merges country i18n, city lists, and per-city locale overrides into `CITY_SPECS`. |
| Country list | `sugar_sugar/location_countries.py` | Canonical `COUNTRY_NAMES` tuple (197 countries). |
| City source data | `sugar_sugar/build_city_data.py` | Embedded `TOP_CITIES_BY_COUNTRY` (~10 cities per country). |
| Generated city JSON | `sugar_sugar/data/top_cities_by_country.json` | Written by `build_city_data`; read at import (with embedded fallback). |
| City locale overrides | `sugar_sugar/location_city_i18n.py` | Native spellings + extra search tokens (e.g. Kyiv/Kiev, München). |
| Filter + asset export | `sugar_sugar/location_suggestions.py` | `filter_location_suggestions()`, `write_suggestions_asset()`. |
| Tests | `tests/test_location_suggestions.py` | Filter logic, asset sync, per-country city counts. |

### Data model

Each autocomplete row in `location-suggestions.json`:

```json
{
  "canonical": "Berlin, Germany",
  "labels": { "en": "Berlin, Germany", "de": "Berlin, Deutschland", ... },
  "search": ["berlin, germany", "berlin, deutschland", ...],
  "rank": 0
}
```

- **`rank`**: city position within its country (0 = largest/capital first). Used to
  sort prefix matches so `ber` → Berlin before Berat.
- **Countries** use `rank: 1000` so city matches surface first when both match.

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

**4. Regenerate the browser bundle** (required after any catalog change):

```bash
uv run python -m sugar_sugar.location_suggestions
```

Writes `assets/location-suggestions.json`. The on-disk asset must match Python
(`tests/test_location_suggestions.py::test_asset_matches_python_source`).

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
| No dropdown after code change | Stale browser cache / old `location-suggestions.json` | Hard refresh; bump `DEPLOY_BUILD`; confirm `/assets/location-suggestions.json` loads in Network tab. |
| Dropdown clipped / hidden | Parent `overflow: hidden` | Host must have `location-autocomplete-host`; CSS sets `overflow: visible`. |
| Works in EN but not DE/RU/ZH | Missing `CITY_I18N` / `COUNTRY_I18N` entry | Add locale in `location_city_i18n.py` or `COUNTRY_I18N`; regenerate asset. |
| `ber` shows obscure cities first | Missing or wrong `rank` in asset | Regenerate asset; cities are ranked 0–9 within each country in `build_city_data` order. |
| Server 500 on navigation after JS change | Stale tab with old clientside callback id | Bump `DEPLOY_BUILD`, hard-refresh all tabs. |
| Tests fail on asset sync | Forgot to run `location_suggestions` after catalog edit | Run step 4 above. |
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
| `/` | `landing.py` / `LandingPageMobile` | Consent checkboxes, play-only mode, resume code entry |
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
| `assets/location-suggestions.json` | `uv run python -m sugar_sugar.location_suggestions` |
| `sugar_sugar/data/top_cities_by_country.json` | `uv run python -m sugar_sugar.build_city_data` |
| Clientside JS behaviour | Edit `assets/*.js`, bump `DEPLOY_BUILD` in `config.py` |

`DEPLOY_BUILD` is injected into the app shell so clients pick up new asset/callback
versions after deploy.

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
| Full test suite | `uv run pytest` |

---

## Related files (quick index)

```
assets/
  location-autocomplete.js      # browser autocomplete
  location-autocomplete.css
  location-suggestions.json     # generated — commit after regen

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
