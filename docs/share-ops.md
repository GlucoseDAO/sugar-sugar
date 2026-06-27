# Sugar Sugar Sharing Operations

This document summarizes the sharing and production-mode behavior implemented in
this repository. It is intentionally specific to Sugar Sugar, not a generic guide
for social previews.

## What Sharing Means Here

Sugar Sugar has two crawler-facing sharing surfaces:

- Site-wide previews for public informational routes, using the static
  `assets/og-card.png` image.
- Per-result previews for `/share/<share_id>`, using a generated 1200x630 PNG
  card based on a stored share record.

The share page is a live Dash page. Social crawlers do not execute Dash
JavaScript, so the Flask server serves crawler-ready metadata before Dash routing
gets involved.

## Share Record Lifecycle

The final page's `share-results-button` is handled by
`handle_share_results_button()` in `sugar_sugar/app.py`.

That callback:

1. Merges current rounds with archived `runs_by_format`.
2. Tags every round with its data-source format (`A`, `B`, or `C`).
3. Computes and freezes overall plus per-format rankings.
4. Writes a small JSON-safe share record.
5. Redirects the browser to `/share/<share_id>`.

Share records are stored by `sugar_sugar/share_store.py` as one JSON file per
share:

```text
data/shares/<share_id>.json
```

The directory can be changed with `SUGAR_SHARE_DIR`. Use that in multi-worker or
multi-container deployments if workers need shared storage. Writes are atomic:
the record is written to a temp file and renamed into place, so readers never see
partial JSON.

Share records deliberately contain only the data needed by the public share page:
round summaries, `prediction_table_data`, played formats, rankings, locale, and a
small subset of `user_info`. They do not depend on browser `dcc.Store` data and
do not keep heavyweight full CGM data.

## Public Routes

The implemented routes are:

- `/share/<id>`: live Dash share page for humans.
- `/share/<id>/image.png`: generated PNG card for the share record.
- `/share/<id>/og`: crawler metadata HTML, with server-side redirect for humans.

There is also a `before_request` hook. If a crawler requests `/share/<id>`
directly, the app serves the same OG HTML it would serve from `/share/<id>/og`.
This is necessary because platforms usually scrape the exact URL the user shares,
not the helper `/og` URL.

Crawler user-agent tokens currently include Facebook, X/Twitter, LinkedIn,
WhatsApp, Slack, Telegram, Discord, Pinterest, and Skype preview bots.

## Share Card Image Pipeline

The visual card is built once in Plotly by
`sugar_sugar.components.share.build_share_card_figure()`.

The export path is in `sugar_sugar/share_png.py`:

1. Export the Plotly figure with Kaleido.
2. If export fails, ensure the managed Chrome for Testing binary exists and retry
   once.
3. If export still fails and fallback is allowed, serve `assets/og-card.png`.

There is intentionally no matplotlib fallback. A second drawing implementation
drifts from the real card and previously caused production previews with tiny
fonts, overlapping labels, and wrong aspect ratio.

The canonical card size is:

```text
1200x630
```

This 1.91:1 format is the safe large-card size for Facebook, LinkedIn, X,
Telegram, Slack, Discord, and most messaging apps. The OG metadata declares the
same dimensions.

`/share/<id>/image.png` caches rendered PNG bytes in the worker process by
`(share_id, locale)`. The HTTP response also sets `Cache-Control:
public, max-age=86400`.

When the card design changes, bump `SHARE_CARD_IMAGE_VERSION` in
`sugar_sugar/app.py`. The OG image URL includes this as:

```text
/share/<id>/image.png?v=<version>
```

Social platforms cache aggressively, so a version bump plus platform re-scrape is
part of the deploy process for preview changes.

## Open Graph And Share Intents

Crawler HTML is built by `_share_card_og_response()` in `sugar_sugar/app.py`.

It includes:

- `og:title`, `og:description`, `og:url`
- `og:image`, `og:image:secure_url`, `og:image:type`
- `og:image:width=1200`, `og:image:height=630`
- `twitter:card=summary_large_image`
- `twitter:title`, `twitter:description`, `twitter:image`

The crawler response does not use a meta refresh. Humans who open
`/share/<id>/og` are redirected server-side with HTTP 302 to `/share/<id>`.
Meta refresh on crawler responses confused some platforms into scraping the
generic Dash shell instead of the item-specific card.

The share-page buttons in `sugar_sugar/components/share.py` use platform-specific
intent URLs:

- X: `https://twitter.com/intent/tweet?text=...&url=...`
- Facebook: `https://www.facebook.com/sharer/sharer.php?u=...&quote=...`
- WhatsApp: `https://api.whatsapp.com/send?text=...`
- LinkedIn: `https://www.linkedin.com/feed/?shareActive=true&shareUrl=...`
- Telegram: `https://t.me/share/url?url=...&text=...`
- Discord: no reliable URL-prefill intent, so the UI provides a separate action.

LinkedIn intentionally uses `feed/?shareActive=true&shareUrl=` rather than the
older `sharing/share-offsite/?url=` form, which has been flaky in production.

### Pre-filled share TEXT support is platform-specific (not a bug)

Only some platforms let a share URL pre-fill the user's message text:

- **Telegram & WhatsApp** honour `text=` → the composer opens with the invite
  text **and** the link. Working as intended.
- **Facebook & LinkedIn** strip pre-filled share text by design (an anti-spam
  decision both made years ago). `sharer.php` only reliably consumes `u=`;
  LinkedIn's feed share only consumes the URL. We pass `&quote=` to Facebook as a
  best-effort (it *may* surface the text in some surfaces, but FB usually ignores
  it without a registered `app_id` + the JS Share Dialog). **On FB/LinkedIn the
  message is carried by the OG card (`og:title` / `og:description`), not by
  pre-filled post text** — so "no text in the FB/LinkedIn composer" is expected,
  not a regression. The only way to truly pre-fill FB text is the JS SDK Share
  Dialog with an app_id (out of scope; not worth a Facebook app registration).

X intentionally uses the canonical `/intent/tweet` Web Intent path with separate
`text` and `url` params. Do **not** switch it back to `/intent/post`: that is not
a real intent path, so on mobile the X app claims the `x.com` universal link,
fails to resolve `/intent/post` as an in-app route, and bounces straight back out
("share opens the app and then closes"). See the Twitter/X footguns below.

## Twitter/X OG Footguns

X/Twitter is the strictest and most opaque OG consumer. Every item below has
bitten this project in production; read it before touching the share card,
`robots.txt`, the image route, or the X intent URL.

**1. Twitterbot obeys `robots.txt`; most others do not.** `facebookexternalhit`,
WhatsApp, LinkedIn, and Telegram ignore `robots.txt` for OG fetches, but
Twitterbot respects it. So `Disallow: /share/*/image.png` makes the card image
vanish **on X only**, reproducibly, while every other platform looks fine. Never
disallow the card image. To keep per-share PNGs out of search indexes, use the
`X-Robots-Tag: noindex` header on the image response instead (permits fetching,
blocks indexing).

**2. `Content-Disposition: attachment` silently kills the card.** Serving the PNG
with `as_attachment=True` (a forced download) makes X refuse to render it inline.
The image route serves **inline** (`as_attachment=False`). The human "Download"
button still works because it forces the download client-side via the HTML
`download` attribute, not the disposition header.

**3. Use `/intent/tweet`, never `/intent/post`.** `/intent/post` is not a real
Web Intent path. Desktop happens to redirect it to the composer, but on mobile
the installed X app claims the `x.com` universal link, can't resolve
`/intent/post` in-app, and bounces back out — the user sees the share "open the
app then close". Canonical: `https://twitter.com/intent/tweet?text=...&url=...`.

**4. X retired the Card Validator (2022). There is no official re-scrape button.**
Any "card validator" you find today is third-party — it fetches the live tags
fresh (bypassing both `robots.txt` *and* X's cache), so it can show a green
preview while real tweets still show nothing. A green third-party validator does
**not** prove the live card works.

**5. X caches a card per-URL for up to ~7 days.** If X scraped a URL while it was
broken (e.g. before a fix deployed), it serves that stale "no image" card until
the cache expires. Because there's no re-scrape tool (#4), the only ways to force
a fresh card are:
- **Share a URL X has never seen** — append a throwaway query param to the *page*
  URL you post: `…/share/<id>?r=1`. X keys its cache on the full URL, so this is a
  brand-new entry → fresh scrape. The crawler hook matches on the path, so the OG
  response is byte-identical; only X's cache key changes.
- Wait out the ~7-day cache on the bare URL.

**6. Newly-generated shares are inherently fresh URLs.** Each share has a unique
id (`/share/ytTfKQ65Zb` ≠ `/share/bbOyjxLhX8`), so any *new* share scrapes clean
on its first post — no cache-bust param needed. `SHARE_CARD_IMAGE_VERSION` (baked
into the image URL as `?v=`) only matters when you **redesign the card art** and
need *already-posted* URLs to refresh their image; it does nothing for new shares.

**7. The compose/DM inline preview is cached harder than posted tweets.** When
testing, an actual post (to a throwaway/private account), or a `?r=1` URL, is the
real test — the draft-composer preview can stay stale even after the live card is
fixed.

**Diagnosing "image works everywhere except X, reproducibly":** suspect #1
(robots.txt) or #2 (attachment) first — both are Twitter-specific and systematic.
**"Third-party validator green but the tweet is blank":** suspect #5 (stale
per-URL cache) — re-share with `?r=1`.

## Site-Wide Crawler Metadata

The Dash app shell includes site-wide OG/Twitter metadata in the `Dash()`
constructor. It points to:

```text
/assets/og-card.png?v=1
```

In production, set `DEPLOY_URL` so this becomes an absolute URL. The site-wide
preview is used for informational routes and any platform that scrapes the
generic app shell.

The Flask server also exposes:

- `/robots.txt`
- `/sitemap.xml`
- `/llms.txt`

These are generated dynamically from the current request base or `DEPLOY_URL`.
`robots.txt` allows the public site, advertises the sitemap and `llms.txt`, and
disallows Dash internals (`/_dash-`, `/_reload-hash`).

It must **not** disallow `/share/*/image.png`. Twitterbot honors robots.txt, so a
Disallow there makes X skip the card image entirely (other platforms ignore
robots.txt for OG fetches, so the failure looks Twitter-specific). The per-share
PNGs are instead kept out of search indexes via an `X-Robots-Tag: noindex`
response header on the image route, which permits crawler *fetching* while
blocking *indexing*. See the Twitter/X footguns below.

## Canonical URL Resolution

`DEPLOY_URL` is the preferred production source of truth for absolute public
URLs:

```bash
DEPLOY_URL=https://sugar-sugar.study
```

It is used for:

- site-wide OG/Twitter image URLs
- share URLs
- crawler helper files
- `og:url`
- share-card image URLs

If `DEPLOY_URL` is blank, the app falls back to `SUGAR_SUGAR_PUBLIC_BASE_URL`,
then `X-Forwarded-Host` / `X-Forwarded-Proto`, then the Flask request host.

For production, prefer setting `DEPLOY_URL` explicitly. If it changes, restart
the app so all process-level configuration and generated responses use the new
origin.

## Production Mode

Use:

```bash
uv run serve --host 0.0.0.0 --port 8050 --workers 2
```

`uv run serve` runs gunicorn against `sugar_sugar.wsgi:application`.

Important behavior:

- `serve()` calls `_ensure_chrome()` before `os.execvp("gunicorn", ...)`.
- Each gunicorn worker imports `sugar_sugar.wsgi`, which calls
  `bootstrap_wsgi_application()`.
- `bootstrap_wsgi_application()` registers callbacks and calls `_ensure_chrome()`
  again in the worker.
- Gunicorn reads `GUNICORN_WORKERS`, `WEB_CONCURRENCY`,
  `GUNICORN_TIMEOUT`, and `GUNICORN_FORWARDED_ALLOW_IPS`.
- The Flask app respects forwarded host/proto headers when `DEPLOY_URL` is not
  set, but `DEPLOY_URL` is still preferred.

Typical production env:

```bash
DEPLOY_URL=https://sugar-sugar.study
DASH_HOST=0.0.0.0
DASH_PORT=8050
GUNICORN_WORKERS=2
GUNICORN_TIMEOUT=120
GUNICORN_FORWARDED_ALLOW_IPS=*
STORAGE_TYPE=local
```

Serve the app behind a TLS reverse proxy such as Caddy or nginx. The public
origin should be HTTPS because social platforms require or strongly prefer HTTPS
for images and metadata.

## Staging Mode (prod+)

The staging deployment at `https://vanilla-sugar.glucosedao.org/` hosts the dev
branch. Run it like production but with the staging test routes enabled:

```bash
uv run serve-staging                 # = uv run serve --staging
uv run serve --staging --workers 2   # equivalent, explicit
```

`--staging` sets the `_STAGING_MODE=1` environment variable before `os.execvp`,
so every gunicorn worker re-reads it at import. When the flag is **off**
(production default), none of the routes below exist and the app is byte-identical
to plain `serve` — this is **prod+**: extra routes layered on top of the real
pathways, with no change to production logic.

The staging nodes let you reach prefilled states **remotely, page-by-page, without
a playthrough and without restarting the server** (handy because simulating mobile
is tedious):

- `GET /staging` — index page linking the nodes.
- `GET /staging/ending` — `create_ending_layout` with a synthetic prefilled round.
- `GET /staging/final` — `create_final_layout` with several synthetic rounds
  (formats A/B/C).
- `GET /staging/share?formats=A,B,C&lang=de` — generates a synthetic share record
  on disk and 302-redirects to `/share/<id>` (reuses `share_store.save_share`).
- `GET /staging/prediction` — seeds the prediction stores with a prefilled window
  (server callback) and routes to the real `/prediction`.

Implementation notes:

- The nodes reuse the **real** layout builders and `share_store`; only the synthetic
  *input* data is generated (`_staging_*` helpers in `app.py`). They are defined at
  module scope but invoked only from `_STAGING_MODE`-gated handlers, so they never run
  at import when the flag is off.
- Every synthetic `user_info` sets `consent_completed=True` so the `display_page`
  consent guard lets the nodes render.
- The screenshot harness `result` group (`scripts/mobile_shots.py`) uses these same
  nodes to capture `/ending`, `/final`, and `/share` on a phone viewport.
- **Security:** set `STAGING_AUTH="user:password"` on a publicly reachable staging
  origin to require HTTP Basic Auth (over HTTPS) on every `/staging/*` request. A
  `before_request` hook (registered only in staging mode) enforces it and reads the
  credential live, so it can be rotated without a code change. When `STAGING_AUTH`
  is unset the routes are open — which is what keeps local `serve --staging` and the
  screenshot harness working. (A reverse-proxy IP allowlist is a fine alternative.)
- Set `DEPLOY_URL=https://vanilla-sugar.glucosedao.org` on the staging host so share
  URLs, OG tags, and card image URLs resolve to the staging origin.

## Chrome And Kaleido Requirements

Share-card PNG export uses Kaleido, and Kaleido needs a Chromium browser.

The project provides:

```bash
uv run setup-chrome
```

This provisions Chrome for Testing into the choreographer cache. The app also
calls `_ensure_chrome()` during `uv run start`, `uv run share`, `uv run chart`,
`uv run serve`, and WSGI worker bootstrap.

Chrome for Testing is not a fully static binary. On bare Linux hosts, install the
runtime libraries before running production:

```bash
sudo apt-get update
sudo apt-get install -y \
  libatk1.0-0t64 libatk-bridge2.0-0t64 libcups2t64 libdrm2 \
  libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
  libgbm1 libpango-1.0-0 libcairo2 libasound2t64 libatspi2.0-0t64 \
  libnss3 libnspr4
uv run setup-chrome
```

If Kaleido fails, check:

```text
logs/sugar_sugar.log
logs/sugar_sugar.json
```

Look for `share_png_kaleido_failed` and `kaleido_failed_after_retry`. If the app
falls back, the OG image still returns a valid PNG, but it will be the generic
static `assets/og-card.png`, not the personalized result card.

Smoke-test the managed Chrome directly:

```bash
CHROME=$(find ~/.local/share/choreographer -type f -name chrome -path '*chrome-linux64*' | head -1)
"$CHROME" --headless --no-sandbox --disable-gpu --disable-dev-shm-usage --dump-dom about:blank
```

DBus or UPower warnings are common on headless servers. An exit code of `0` with
basic HTML output means Chrome can launch. Missing `.so` errors mean the host
still lacks runtime libraries.

Docker is different: the included `Dockerfile` installs system Chromium, so image
export should work out of the box there.

## Local Share Testing

Use the share shortcut for page and card work:

```bash
uv run share
uv run share --formats "A,B,C"
uv run share --formats "A,B" --rounds 8
uv run share --locale de
```

The command generates synthetic prediction rounds, saves a share record to disk,
starts Dash, and opens `/share/<id>`. It bypasses the full game flow.

Use the deterministic preview renderer before shipping card layout changes:

```bash
uv run python scripts/render_share_card_previews.py
uv run python scripts/render_share_card_previews.py --locale de
uv run python scripts/render_share_card_previews.py --output-dir /tmp/share-card-previews
```

It writes single-format and multi-format card PNGs under
`data/output/share-card-previews/` by default. Inspect them across locales,
especially long European text, Ukrainian, Romanian, and any CJK/no-space strings.

For tests:

```bash
uv run pytest tests/test_share_png.py
```

`test_kaleido_direct_can_render_small_png` is the important low-level proof that
Kaleido can actually launch a browser on the current host.

## Crawler Verification

After deploying, verify what crawlers see with raw HTTP requests.

Site-wide:

```bash
curl -sL -A "Twitterbot/1.0" https://sugar-sugar.study/ | rg 'og:|twitter:'
curl -sI https://sugar-sugar.study/assets/og-card.png?v=1
```

Per-share:

```bash
curl -sL -A "facebookexternalhit/1.1" https://sugar-sugar.study/share/<id> | rg 'og:|twitter:'
curl -sI https://sugar-sugar.study/share/<id>/image.png?v=2
curl -sI https://sugar-sugar.study/share/<id>/og | rg 'HTTP|Location'
```

Expected results:

- Bot requests to `/share/<id>` return item-specific OG/Twitter tags.
- `og:image` points to `/share/<id>/image.png?v=<SHARE_CARD_IMAGE_VERSION>`.
- Image response is `200`, `Content-Type: image/png`, `Content-Disposition:
  inline` (NOT `attachment`), `X-Robots-Tag: noindex`, and not behind auth.
- `robots.txt` does **not** Disallow `/share/*/image.png`.
- Human requests to `/share/<id>/og` redirect with 302 to `/share/<id>`.
- `og:image:width` and `og:image:height` match the actual PNG dimensions.

Then force re-scrapes in platform tools:

- Facebook Sharing Debugger
- LinkedIn Post Inspector
- Telegram `@WebpageBot`
- X/Twitter: there is **no** official re-scrape tool (the Card Validator was
  retired in 2022). Force a fresh scrape by posting the page URL with a throwaway
  query param (`/share/<id>?r=1`) — see the Twitter/X footguns. Third-party OG
  previewers show the live tags but cannot clear X's own per-URL cache.
- a throwaway Discord or Slack channel

Most platforms cache previews for days. If an image changed but the URL did not,
you may keep seeing the old image until the version query changes or the
platform cache is cleared.

## Operational Pitfalls

- Do not store `/share/*` in `last-visited-page`. The persistence allowlist in
  `app.py` intentionally includes only game-flow pages.
- Do not make the share page depend on localStorage. `/share/<id>` must render
  from `data/shares/<id>.json` alone.
- Do not reintroduce a second image renderer. Keep one Plotly card and branch
  only in the export step.
- Do not use square OG cards. Use 1200x630 and declare matching dimensions.
- Do not add meta refresh to crawler OG responses. Redirect humans server-side.
- Do not share `/share/<id>/image.png` directly. Share `/share/<id>` so crawlers
  can read title, description, and image metadata.
- Do not `Disallow: /share/*/image.png` in `robots.txt` — it breaks the X card
  (Twitterbot obeys robots.txt). Use `X-Robots-Tag: noindex` on the image route.
- Do not serve the card PNG with `Content-Disposition: attachment` — X won't
  render it. Serve inline; the Download button uses the HTML `download` attribute.
- Do not use `x.com/intent/post` for the X share button. Use the canonical
  `twitter.com/intent/tweet?text=...&url=...` (post bounces the mobile X app).
- Do not assume a downloaded Chrome binary is enough. The host still needs
  Chromium shared libraries.
- Do not rely on `DASH_DEBUG=True` for production. Use `uv run serve` so the app
  runs under gunicorn rather than Werkzeug.
- Do not commit `data/shares/` contents. Share records are runtime session data.

## Quick Deploy Checklist

1. Set `DEPLOY_URL` to the public HTTPS origin.
2. Install Chromium runtime libraries on bare hosts.
3. Run `uv run setup-chrome`.
4. Start with `uv run serve`.
5. Generate a test share with real or synthetic data.
6. Verify `/share/<id>` as a bot and `/share/<id>/image.png` as an image
   (`200`, `image/png`, `Content-Disposition: inline`, `X-Robots-Tag: noindex`;
   confirm `robots.txt` does not Disallow the image).
7. Render locale stress previews if card layout changed.
8. Bump `SHARE_CARD_IMAGE_VERSION` if the share-card PNG design changed (this
   only refreshes *already-posted* URLs; new shares are fresh URLs already).
9. Re-scrape in Facebook, LinkedIn, Telegram, Discord, and Slack. For X there is
   no re-scrape tool — test with a fresh `/share/<id>?r=1` URL (see footguns).
10. Watch `logs/sugar_sugar.log` for Kaleido fallback messages.
