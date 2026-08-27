# Copying the "db" from prod into test

There is no database. Sugar-Sugar's state is **flat files under `data/`**, so copying prod into
test is a tar + scp, not a dump/restore. The whole job is knowing *which* files, and which ones
must not travel.

Prod is `https://sugar-sugar.study`; the staging box is `https://vanilla-sugar.glucosedao.org/`
(publicly reachable, runs `uv run serve-staging`). Everything below assumes you are copying
**prod → staging**, i.e. onto a machine that is less protected than the one the data came from.

## What the state actually is

| Path | Holds | Copy to test? |
|---|---|---|
| `data/input/prediction_statistics.csv` | the research record — **raw `email`** + age, gender, diabetes status, location | yes, **with the email column blanked** |
| `data/input/prediction_ranking{,_A,_B,_C}.csv` | leaderboard slots: `email_key` (hashed) + public nickname | yes, as-is |
| `data/input/consent_agreement.csv` | consent flags per `study_id` (no email, no name) | yes, as-is |
| `data/input/users/` | uploaded CGM exports — **real patient names in the filenames** | **no** |
| `data/resume/*.json` | cross-device savegames = **redeemable session tokens**, carry `user_info` incl. email | **no** |
| `data/shares/*.json` | public share records | only if testing `/share/<id>` |
| `data/faq/` | FAQ board posts | only if testing the board |
| `data/.ranking_salt` | per-deployment HMAC secret | **no** — see below |
| `data/bigideas/`, `data/d1namo/`, `data/subjects/` | corpora | no — `uv run download` on the test box |
| `data/output/`, `logs/` | generated artifacts | no |

## The quick way (recommended)

The privacy-safe copy is also the *sufficient* one for looking at the boards: `/highscore` reads
`prediction_statistics.csv` (for rounds and per-round MAEs) plus the ranking CSVs (for nicknames,
joined on `study_id`). Neither needs an email address, an upload, or a resume blob.

**On prod** — reuse the existing backup script rather than hand-rolling a tar; it takes a lock,
verifies the archive and writes it `0600`:

```bash
cd /path/to/sugar-sugar
./backup/backup-input.sh              # → backup/archives/sugar-data-input-<UTC stamp>.tar.gz
```

**Transfer** (the archive contains participant data until you scrub it — keep it off shared boxes):

```bash
scp prod:/path/to/sugar-sugar/backup/archives/sugar-data-input-<stamp>.tar.gz /tmp/
scp /tmp/sugar-data-input-<stamp>.tar.gz test:/tmp/
```

**On test** — extract at the repo root (archive paths are root-relative), dropping the uploads:

```bash
cd /path/to/sugar-sugar
tar -xzf /tmp/sugar-data-input-<stamp>.tar.gz --exclude='data/input/users/*'
```

**Scrub the emails immediately** (on the test copy — never run this against prod):

```bash
uv run python -c "
import polars as pl
p = 'data/input/prediction_statistics.csv'
df = pl.read_csv(p, infer_schema_length=0)
df.with_columns(pl.lit('').alias('email')).write_csv(p)
print('blanked', df.height, 'rows')
"
```

`infer_schema_length=0` reads every column as text, so numbers are not reformatted and the
comma-heavy quoted columns (`location`, `per_round_metrics`) survive the round-trip intact. It
rewrites the file in place, which is why it belongs on the copy only.

Then delete the archive from `/tmp` on both hops.

### What blanking the email costs you

Identity degrades to per-`study_id`: one person who played on two devices becomes two players in
test. Veteran badges still work for anyone whose runs share a `study_id`, and nicknames still
resolve (they join on `study_id`, not on the email). That is usually the right trade — but if
you are specifically testing **cross-device identity merging**, you need real emails and the salt,
which means you are testing on production data and should do it on a box you would trust with it.

This is pseudonymized, not anonymized: age, gender, diabetes type and location remain, and a
`study_id` still keys back to prod's records.

## The salt: don't copy it

`data/.ranking_salt` (or `RANKING_EMAIL_SALT`) is a per-deployment secret behind three HMACs:
`email_key`, the content-addressed share ids, and the `/player/<id>` URLs. Copying it puts a prod
secret on a public staging box, and rotating it is not an option (it re-splits every existing
player), so treat the copy as one-way.

With a *different* salt on test:

- **Works normally** — the `/highscore` class boards (they recompute `email_key` from the email
  column, so they are self-consistent under any salt) and nicknames (joined on `study_id`).
- **Breaks for a tester reusing a prod email** — `/final`'s "You" highlight and the `stored_nickname`
  prefill, which compare a locally computed hash against the `email_key` *stored* in the ranking
  CSVs. Copied rows will not match.
- **Changes** — `/player/<id>` URLs differ from prod's (the pages still work), and re-rendering a
  copied share mints a new id instead of reusing the file.

## Verify

```bash
uv run start                # or: uv run serve-staging
```

Open `/highscore` and confirm both class boards populate. If they are empty but the CSVs have rows,
the usual cause is the round floor: an entry needs at least `MIN_USEFUL_ROUNDS` (default 6) rounds
*of that data class*, and `example.csv` rounds classify to neither board.

The two-board layout only renders on a checkout that has the `scoreboard-redesign` branch — staging
tracks `development-ai`, so check out the branch there before expecting the new boards.

## Notes

- **Consistency:** CSV writes are atomic per file (`.tmp` + `replace`), so a tar taken while the app
  is live gets internally consistent files, with at most slight skew *between* files. Quiesce the
  service if that matters; for eyeballing a board it does not.
- **Never copy test → prod.** Nothing here is reversible in that direction; `save_statistics` upserts
  on `study_id` + `run_id`, so a synthetic row with a colliding id would overwrite a real one.
- `backup/backup-input.sh` covers `data/input/` only. `data/shares/`, `data/faq/` and `data/resume/`
  are separate trees — and can be pointed elsewhere on the test box with `SUGAR_SHARE_DIR`,
  `SUGAR_FAQ_DIR` and `SUGAR_RESUME_DIR` if you want them isolated from the copied state.
- Put `STAGING_AUTH="user:password"` on the staging origin before it holds anything derived from
  participant data (see `share-ops.md` → "Staging Mode").
