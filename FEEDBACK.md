# Upstream feedback for `cgm-format`

Issues found in [`cgm-format`](https://github.com/GlucoseDAO/cgm_format) while working on
Sugar Sugar. Filed here rather than fixed in this repo: `sugar_sugar/corpus.py` /
`sugar_sugar/data.py` are the single boundary to the library, and working around a parser
defect on this side of it is exactly the drift that boundary exists to prevent.

**Status: everything below was reported against 0.12.0 and all of it but §5 shipped in
0.12.2**, which is now the floor in `pyproject.toml`. Re-verified by running the code, not
by reading the changelog. The fixed sections are kept rather than deleted — they record why
the floor is where it is, and what to re-test if any of it regresses.

---

## 1. Nightscout URL import failed outright on any real instance — FIXED in 0.12.2

**Was: blocking.** This is the bug a player actually hit; it took down the whole "paste your
Nightscout URL" feature. Reported against a live Nightscout 15.0.3 instance (URL withheld —
it serves a real patient's data).

### Symptom

`FormatParser.from_nightscout_url(base_url)` raised, before any app code saw a frame:

```
polars.exceptions.ComputeError: could not append value: 47 of type: i64 to the builder;
make sure that all rows have the same schema or consider increasing `infer_schema_length`
```

Traceback bottomed out at `format_parser.py:1480`, in `_parse_nightscout_treatments_json`:
`return pl.DataFrame(rows)`.

### Root cause

`_parse_nightscout_treatments_json` built `rows` as a list of dicts and handed it to
`pl.DataFrame` with **no schema**, so polars inferred dtypes from the first
`infer_schema_length=100` rows only.

Nightscout treatments are heterogeneous by design — a `Temp Basal` carries `rate`/`duration`
and no `carbs`, a `Meal Bolus` carries `carbs`/`insulin` and no `rate`. On the reporting
instance, `carbs` was `null` for the first **167** rows — past the 100-row inference window
— so the column inferred as dtype **`Null`**, and appending the `i64` at row 167 failed.

Two things worth stressing, because they ruled out the obvious near-miss diagnoses:

- **It was not the int/float mix.** `insulin` mixed int and float and appended fine — polars
  takes the supertype. The failure was specifically *any* value into a `Null`-dtype builder.
  Coercing the JSON's numbers to float did **not** help; it only changed the message to
  `could not append value: 47.0 of type: f64`.
- **It was not exotic data.** Any Nightscout user who runs a closed loop and logs carbs
  rarely — a large fraction of them — has a null run longer than 100 rows in at least one of
  these four columns. Bolus-only users hit it via `rate`, pump-only users via `carbs`.

### Minimal repro (no network, no patient data)

```python
import json
from cgm_format import FormatParser

entries = [{"type": "sgv", "dateString": f"2026-08-28T{h:02d}:00:00.000Z", "sgv": 120}
           for h in range(5)]
treatments = [{"eventType": "Temp Basal", "created_at": "2026-08-28T00:00:00.000Z",
               "rate": 0.5, "duration": 30, "carbs": None, "insulin": None}
              for _ in range(105)]
treatments.append({"eventType": "Meal Bolus", "created_at": "2026-08-28T05:00:00.000Z",
                   "carbs": 47, "insulin": 2.5})

FormatParser.parse_nightscout(json.dumps(entries), json.dumps(treatments))
```

106 records was enough. Dropping the null run below 100 made it pass, which was the whole bug.

### Resolution

Fixed in 0.12.2. Verified against the reporting instance: the import now returns **1378
unified rows**, and numeric-string treatment values (`"carbs": "47"`, which some uploaders
emit) are coerced rather than silently dropped — that detail was worth 3 events on the
reporting instance and is covered by the repro above.

Pinned here by `tests/test_nightscout_json_upload.py`
::`test_treatments_with_a_long_null_run_are_loaded`, which was written to fail on 0.12.0 and
now passes. It stays as a regression guard.

---

## 2. Redirects were not followed — FIXED in 0.12.2

`nightscout_downloader.py:115` constructed the client as
`httpx.Client(timeout=timeout, headers=headers)`.

httpx differs from `requests` here — on the pinned httpx 0.28.1, `httpx/_client.py:197` is
`follow_redirects: bool = False`, and `download_nightscout` never passed the argument. So any
instance that redirected — `http://` → `https://`, apex → `www`, a Heroku/Fly/Netlify
hostname 301ing to its canonical domain, a trailing-slash normalisation — failed at the
`raise_for_status()` on the next line with `httpx.HTTPStatusError: Redirect response '301
Moved Permanently'`. Users type bare hostnames constantly.

`download_nightscout` now passes `follow_redirects`.

---

## 3. Empty or non-JSON API responses were misrouted into the CSV parser — FIXED in 0.12.2

`parse_nightscout` picked its branch with `_is_nightscout_entries_json`, which tested
`stripped.startswith("[") and '"sgv"' in stripped[:2000]`. A Nightscout account with no data
returns `200` with body `[]` — starts with `[`, contains no `"sgv"` — so the JSON fell
through to `_parse_nightscout_entries_csv`. The results were:

```
"[]"                                      -> MalformedDataError: Missing required column:
                                             'Glucose (mg/dL)'. Got columns: ['[]']
'{"status":401,"message":"Unauthorized"}'  -> MalformedDataError: Missing required column:
                                             'Glucose (mg/dL)'. Got columns:
                                             ['{"status":401', 'message":"Unauthorized"']
```

So the purpose-built `ZeroValidInputError("Nightscout entries JSON is empty")` guard was
**unreachable** from a real API response, and a server's own error text survived only as a
mangled polars column name.

Both now report themselves properly:

```
"[]"        -> ZeroValidInputError: Nightscout entries JSON is empty
auth object -> MalformedDataError: Nightscout returned an error object instead of
               entries: Unauthorized
```

---

## 4. Entries were dropped on falsy and missing fields — FIXED in 0.12.2

- `entry.get("sgv") or entry.get("glucose")` — an `sgv` of `0` is falsy, so the row fell
  through to `glucose`/`None` and was dropped. `0` is not physiological, but it is a real
  sensor-error value and dropping it silently differed from what the code read as intending.
- `if entry.get("type") != "sgv": continue` — uploaders that omit `type` (some xDrip+/Loop
  configurations) had every row skipped, producing `ZeroValidInputError("No SGV entries
  found")` on a payload carrying `sgv` and `dateString` on every record.

Both now keep their rows: a two-entry payload with `sgv: 0` yields 2 rows, and a two-entry
payload with no `type` field yields 2 rows.

---

## 5. Still open: UK/EU exporter CSV dates, and one error-contract asymmetry

**Neither is blocking, and neither affects this app** — Sugar Sugar takes Nightscout data as
JSON, by URL or as an uploaded `entries.json`, and never through the exporter CSV path.

- **UK/EU-locale nightscout-exporter CSVs still do not parse.** `_EXPORTER_DATETIME_FORMATS`
  tries `%m/%d/%Y` before `%d/%m/%Y` with `strict=False`, and the probe returns on the first
  format that does not *throw* rather than the first that *parses*, so an unambiguous UK date
  never reaches the UK format. 0.12.2 improved this from **silent** data loss to a loud
  failure — an all-UK file now raises `ZeroValidInputError: No valid data rows found after
  processing` instead of returning a frame with the rows quietly missing — which is the more
  important half. But a UK exporter CSV is still unreadable, and a *mixed* file would still
  lose only the ambiguous rows. Probing which format parses the most rows, rather than which
  one throws last, would settle it.
- **`parse_nightscout` still lacks the `try/except -> MalformedDataError` wrapper that
  `_process_nightscout` has.** The two public paths into the same parsing code therefore have
  different error contracts. This is what made issue 1 surface as a raw polars `ComputeError`
  rather than a library error; with issue 1 fixed it is cosmetic, but the asymmetry remains.

---

## 6. Resolved housekeeping

- **The temp directory is now cleaned up.** `from_nightscout_url` with the default
  `output_dir=None` used to `tempfile.mkdtemp(prefix="nightscout_")` and never remove it,
  leaving raw entries + treatments + profile JSON — a patient's full CGM record — in `/tmp`
  indefinitely on every call. It now uses a `TemporaryDirectory`.
- **The package now ships repository metadata.** `Project-URL` gives Homepage, Repository,
  Changelog and Issues; `pip show cgm-format` previously gave no link at all and the only
  pointer was a CI badge in the README body.

Still worth considering upstream, though it costs this app nothing today: **no library
exception type wraps network failures**, so connect errors, timeouts, HTTP status errors and
`json.JSONDecodeError` all escape raw and a caller must import `httpx` itself to catch them.
`sugar_sugar/components/startup.py` does exactly that, matching on `type(exc).__name__`
substrings to classify. Relatedly, `HTTPStatusError` messages embed the full request URL
including the `?token=` query parameter, so any caller that logs the exception leaks the
Nightscout read token — `scripts/diagnose-nightscout.py` has to scrub its own output for
precisely this reason.
