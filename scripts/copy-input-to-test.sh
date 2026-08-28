#!/usr/bin/env bash
# Copy the study CSVs from one data/input to another (prod -> test).
#
#   scripts/copy-input-to-test.sh <source> <dest> [options]
#
# <source> and <dest> are data/input directories, each either a local path or an
# rsync/ssh remote (`[user@]host:/path`). Both may be remote: the copy always
# stages through a local temp dir, which is also what guarantees the email scrub
# happens *before* anything lands on the destination.
#
# What travels: top-level `*.csv` only, non-recursively. Every directory under
# data/input therefore stays put by construction -- `users/` holds uploaded CGM
# exports whose filenames carry real patient names, and `patient_consent_form/`
# and `study_design/` come from the git checkout. The email column of
# prediction_statistics.csv is blanked unless --with-emails is passed.
#
# What this script deliberately does NOT copy (see docs/db-copy-prod-to-test.md):
# `data/resume/` (redeemable session tokens), `data/shares/`, `data/faq/`, and
# `data/.ranking_salt` (a per-deployment secret that must not reach a public box).
#
# The `email_key` column of the ranking CSVs is re-derived under the DESTINATION's
# own salt, so one person keeps one identity there instead of the source's digests
# (which the destination can never reproduce). The source's salt is not needed and
# never moves. This happens before the scrub, since it reads the addresses.
#
# Options:
#   --with-emails   keep the raw email column (real participant data -- only onto
#                   a box you would trust with prod)
#   --no-rekey      leave email_key exactly as the source wrote it
#   --keep-source-names  keep uploaded filenames verbatim; by default they are
#                   pseudonymised, since people name CGM exports after themselves
#   --dest-salt V   use V as the destination's salt instead of reading
#                   <dest>/../.ranking_salt -- needed when the destination sets
#                   RANKING_EMAIL_SALT in its environment, which overrides the file
#   --delete        remove CSVs in <dest> that are absent from <source>
#   --dry-run       show what would happen; write nothing to <dest>
#   -y, --yes       skip the confirmation prompt (required when not a TTY)
#   -h, --help      this text

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

WITH_EMAILS=0
DELETE=0
DRY_RUN=0
ASSUME_YES=0
REKEY=1
KEEP_SOURCE_NAMES=0
DEST_SALT=""
SRC=""
DST=""

log()  { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }
usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//; $d'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-emails) WITH_EMAILS=1; shift ;;
    --no-rekey)    REKEY=0; shift ;;
    --keep-source-names) KEEP_SOURCE_NAMES=1; shift ;;
    --dest-salt)   DEST_SALT="${2:-}"; [[ -n "${DEST_SALT}" ]] || die "--dest-salt needs a value"; shift 2 ;;
    --delete)      DELETE=1; shift ;;
    --dry-run)     DRY_RUN=1; shift ;;
    -y|--yes)      ASSUME_YES=1; shift ;;
    -h|--help)     usage; exit 0 ;;
    -*)            die "unknown option: $1 (try --help)" ;;
    *)
      if   [[ -z "${SRC}" ]]; then SRC="$1"
      elif [[ -z "${DST}" ]]; then DST="$1"
      else die "unexpected argument: $1"
      fi
      shift ;;
  esac
done

[[ -n "${SRC}" && -n "${DST}" ]] || { usage; exit 2; }
command -v rsync >/dev/null || die "rsync is required (local end)"
command -v python3 >/dev/null || die "python3 is required (local end, for the email scrub)"

# `host:/path` -- a colon before the first slash. Bare local paths never match.
is_remote() { [[ "$1" == *:* && "${1%%:*}" != */* ]]; }
host_of()   { echo "${1%%:*}"; }
path_of()   { if is_remote "$1"; then echo "${1#*:}"; else echo "$1"; fi; }

# Run a command where PATH lives, local or over ssh.
run_at() {
  local target="$1"; shift
  if is_remote "${target}"; then ssh "$(host_of "${target}")" "$@"; else bash -c "$*"; fi
}

SRC="${SRC%/}"
DST="${DST%/}"
DST_PATH="$(path_of "${DST}")"

run_at "${SRC}" "test -d '$(path_of "${SRC}")'" \
  || die "source is not a directory: ${SRC}"
run_at "${DST}" "test -d '$(path_of "${DST}")'" \
  || die "destination is not a directory: ${DST} (create it first -- refusing to invent a path)"

# Refuse a self-copy. `../sugar-sugar` from inside ~/staging/sugar-sugar is that
# same checkout, not the sibling one, and the run then looks entirely successful:
# identical row counts (they are the same rows), nothing re-keyed (already keyed
# under this box's salt), and a destination that never gains the players you came
# for. Comparing resolved paths is the only way this announces itself.
_resolved() {
  local target="$1"
  if is_remote "${target}"; then
    echo "$(host_of "${target}"):$(run_at "${target}" "cd '$(path_of "${target}")' && pwd -P")"
  else
    echo "local:$(cd "${target}" && pwd -P)"
  fi
}
SRC_REAL="$(_resolved "${SRC}")"
DST_REAL="$(_resolved "${DST}")"
if [[ "${SRC_REAL}" == "${DST_REAL}" ]]; then
  die "source and destination are the same directory (${SRC_REAL#*:}).
       Copying it onto itself would report success and change nothing.
       From a checkout at ~/staging/sugar-sugar, a sibling at ~/sugar-sugar is
       ../../sugar-sugar/data/input -- or just give an absolute path."
fi

STAGING="$(mktemp -d "${TMPDIR:-/tmp}/sugar-input-copy.XXXXXX")"
chmod 0700 "${STAGING}"
cleanup() { rm -rf "${STAGING}"; }
trap cleanup EXIT

# ---- 1. pull: top-level *.csv only, so no directory can ever travel ----------
log "source: ${SRC}"
# --no-recursive --dirs: descend into the source listing but never into a
# subdirectory, so `users/` and the git-tracked doc dirs cannot travel. (`-a`
# implies `-r`; without --dirs, rsync skips the source listing altogether.)
rsync -a --no-recursive --dirs --include='*.csv' --exclude='*' "${SRC}/" "${STAGING}/"

shopt -s nullglob
STAGED=("${STAGING}"/*.csv)
shopt -u nullglob
[[ ${#STAGED[@]} -gt 0 ]] || die "no .csv files found in ${SRC}"

# ---- 2. re-key: rebuild email_key under the DESTINATION's salt ---------------
# email_key is an HMAC, so the copied digests cannot be converted -- but the
# addresses they were derived from are sitting in prediction_statistics.csv,
# in this staging dir, right now. Re-deriving them here under the destination's
# own salt is what stops one person reading as two identities on the test box,
# and it does it without the source's salt ever moving. Must run before the
# scrub, which is what takes the addresses away.
REKEYED=0
if [[ ${REKEY} -eq 1 ]]; then
  if [[ -z "${DEST_SALT}" ]]; then
    DEST_SALT="$(run_at "${DST}" "cat '${DST_PATH}/../.ranking_salt' 2>/dev/null" || true)"
    DEST_SALT="$(printf '%s' "${DEST_SALT}" | tr -d '[:space:]')"
  fi
  if [[ -z "${DEST_SALT}" ]]; then
    log "NOTE: no readable ${DST_PATH}/../.ranking_salt -- skipping the re-key."
    log "      Start the app once on the destination to mint one, or pass --dest-salt."
    log "      Without it, copied rows keep the source's digests: same person, different"
    log "      identity, so /final's \"You\" highlight will not match them."
  else
    python3 - "${STAGING}" "${DEST_SALT}" <<'PY'
import csv, hashlib, hmac, sys
from pathlib import Path

staging, salt = Path(sys.argv[1]), sys.argv[2].encode("utf-8")

def email_key(email: str) -> str:
    # Mirrors sugar_sugar.nickname.email_key: strip + casefold, HMAC-SHA256, 16 hex.
    normalized = str(email or "").strip().casefold()
    if not normalized:
        return ""
    return hmac.new(salt, normalized.encode("utf-8"), hashlib.sha256).hexdigest()[:16]

def read(path):
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.reader(handle))

def write(path, rows):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    tmp.replace(path)

# study_id -> address, newest row wins (what a future run would hash).
addresses: dict[str, str] = {}
stats = staging / "prediction_statistics.csv"
if stats.exists():
    rows = read(stats)
    header = rows[0] if rows else []
    if "study_id" in header and "email" in header:
        study_i, email_i = header.index("study_id"), header.index("email")
        for row in rows[1:]:
            if len(row) > max(study_i, email_i) and row[email_i].strip():
                addresses[row[study_i].strip()] = row[email_i]

if not addresses:
    print("  no addresses in prediction_statistics.csv, nothing to re-key")
    sys.exit(0)

total = 0
for path in sorted(staging.glob("prediction_ranking*.csv")):
    rows = read(path)
    if not rows:
        continue
    header = rows[0]
    if "email_key" not in header or "study_id" not in header:
        print(f"  {path.name}: no email_key column (pre-nickname schema), left alone")
        continue
    key_i, study_i = header.index("email_key"), header.index("study_id")
    changed = orphans = 0
    for row in rows[1:]:
        if len(row) <= max(key_i, study_i):
            continue
        address = addresses.get(row[study_i].strip())
        if address:
            new = email_key(address)
            if new != row[key_i]:
                changed += 1
            row[key_i] = new
        elif row[key_i].strip():
            # No address on record for this study_id, so nothing to re-derive from.
            # Left as-is: those rows stay grouped with each other, they just cannot
            # match a session created on the destination.
            orphans += 1
    write(path, rows)
    total += changed
    note = f", {orphans} with no address left as-is" if orphans else ""
    print(f"  {path.name}: re-keyed {changed} rows{note}")
print(f"  re-keyed {total} ranking rows to the destination's salt")
PY
    REKEYED=1
  fi
fi

# ---- 3. scrub: blank the one column holding participant addresses ------------
STATS="${STAGING}/prediction_statistics.csv"
[[ ${WITH_EMAILS} -eq 1 ]] && log "WARNING: --with-emails -- real addresses will reach ${DST}"
if [[ -f "${STATS}" ]]; then
  python3 - "${STATS}" "${WITH_EMAILS}" "${KEEP_SOURCE_NAMES}" <<'PY'
import ast, csv, re, sys
from pathlib import Path

path, with_emails, keep_names = Path(sys.argv[1]), sys.argv[2] == "1", sys.argv[3] == "1"
with path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.reader(handle))
if not rows:
    sys.exit(0)
header = rows[0]

def column(name):
    return header.index(name) if name in header else None

# A player's own upload is stored by FILENAME, and people name CGM exports after
# themselves ("Surname Firstname 06.09.2021.csv"). Those names ride in
# data_source_name and in every per_round_metrics entry, so the uploads' contents
# staying put does not keep the patients' names out of the copy. Public sources
# are named for their corpus and are kept as-is; everything else becomes a stable
# pseudonym, which classifies identically (not a corpus, not example -> the
# player's own status decides).
PUBLIC = re.compile(r"^(example\.csv|BIGIDEAS-\d{3}\.csv|D1NAMO-\d{3}\.csv|.*_chronological\.csv)$", re.I)
pseudonyms: dict[str, str] = {}

def clean_source(name):
    text = str(name or "").strip()
    if not text or PUBLIC.match(text):
        return text
    if text not in pseudonyms:
        pseudonyms[text] = f"own-upload-{len(pseudonyms) + 1}.csv"
    return pseudonyms[text]

email_i, source_i, per_round_i = column("email"), column("data_source_name"), column("per_round_metrics")
blanked = 0
# csv.reader/writer (not DictReader) so a row with an unexpected field count keeps
# its shape instead of being silently reshaped to the header's width.
for row in rows[1:]:
    if not with_emails and email_i is not None and email_i < len(row) and row[email_i]:
        row[email_i] = ""
        blanked += 1
    if keep_names:
        continue
    if source_i is not None and source_i < len(row):
        row[source_i] = clean_source(row[source_i])
    if per_round_i is not None and per_round_i < len(row) and row[per_round_i].strip().startswith("["):
        try:
            entries = ast.literal_eval(row[per_round_i])
        except (ValueError, SyntaxError):
            continue
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, dict) and "data_source_name" in entry:
                    entry["data_source_name"] = clean_source(entry["data_source_name"])
            # The app reads this cell with ast.literal_eval, so repr round-trips.
            row[per_round_i] = str(entries)

tmp = path.with_suffix(path.suffix + ".tmp")
with tmp.open("w", newline="", encoding="utf-8") as handle:
    csv.writer(handle).writerows(rows)
tmp.replace(path)
if not with_emails:
    print(f"  blanked {blanked} email values in {len(rows) - 1} rows")
if pseudonyms:
    print(f"  renamed {len(pseudonyms)} uploaded-file names (they carry patient names)")
PY
fi

# ---- 4. report exactly what is about to be written --------------------------
echo
log "staged for ${DST}:"
for file in "${STAGED[@]}"; do
  rows=$(( $(wc -l < "${file}") - 1 ))
  (( rows < 0 )) && rows=0
  printf '    %-34s %6s data rows\n' "$(basename "${file}")" "${rows}"
done
echo

RSYNC_WRITE=(rsync -a)
[[ ${DELETE} -eq 1 ]] && RSYNC_WRITE+=(--delete --include='*.csv' --exclude='*')

if [[ ${DRY_RUN} -eq 1 ]]; then
  log "DRY RUN -- nothing written. Would rsync the above into ${DST}/"
  "${RSYNC_WRITE[@]}" --dry-run --itemize-changes "${STAGING}/" "${DST}/"
  exit 0
fi

if [[ ${ASSUME_YES} -ne 1 ]]; then
  # Read the answer from the controlling terminal rather than stdin: stdin may
  # have been redirected or already consumed, and an empty read there looks
  # exactly like the user declining. `|| reply=""` keeps a read failure (EOF)
  # from tripping `set -e` and exiting with no explanation at all.
  # Actually open it: /dev/tty passes a `-r` test even with no controlling
  # terminal, then fails at the redirect with a raw shell error.
  { : < /dev/tty; } 2>/dev/null \
    || die "no terminal to confirm on: pass --yes to write to ${DST}"
  reply=""
  read -r -p "Overwrite these CSVs in ${DST}? [y/N] " reply < /dev/tty || reply=""
  # Tolerate a trailing CR (terminals that send CRLF), stray whitespace and case.
  answer="$(printf '%s' "${reply}" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')"
  if [[ "${answer}" != "y" && "${answer}" != "yes" ]]; then
    # Echo the raw bytes: if a well-meant "y" is ever rejected again, this line
    # says why instead of leaving you to guess.
    log "aborted (read $(printf '%q' "${reply}")) -- use --yes to skip this prompt"
    exit 1
  fi
fi

# ---- 5. snapshot the destination before overwriting it ----------------------
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DST_PATH="$(path_of "${DST}")"
# Written from inside the destination directory, so the name stays relative to
# it (an absolute path built here would be re-resolved against the cd'd cwd).
# Lands beside data/input as data/input-before-copy-<stamp>.tar.gz.
SNAPSHOT_NAME="input-before-copy-${STAMP}.tar.gz"
if run_at "${DST}" "cd '${DST_PATH}' && ls *.csv >/dev/null 2>&1"; then
  run_at "${DST}" "cd '${DST_PATH}' && tar -czf '../${SNAPSHOT_NAME}' *.csv && chmod 0600 '../${SNAPSHOT_NAME}'"
  log "destination snapshot: ${DST_PATH}/../${SNAPSHOT_NAME}"
else
  log "destination has no CSVs yet, no snapshot needed"
fi

# ---- 6. push ----------------------------------------------------------------
"${RSYNC_WRITE[@]}" "${STAGING}/" "${DST}/"
log "OK: copied ${#STAGED[@]} CSV files into ${DST}"

echo
if [[ ${WITH_EMAILS} -eq 0 ]]; then
  log "Emails are blank, so identity in test falls back to study_id: one person"
  log "who played on two devices now reads as two players. Nicknames still resolve"
  log "(they join on study_id). Still pseudonymized, not anonymized -- demographics"
  log "and location remain, and study_id keys back to the source's records."
fi
if [[ ${REKEYED} -eq 1 ]]; then
  log "email_key was re-derived under the destination's own salt, so the same person"
  log "keeps one identity there. The source's salt never moved."
else
  log "The ranking salt was NOT copied (it never should be) and nothing was re-keyed,"
  log "so copied rows keep the source's digests: /final's \"You\" highlight will not"
  log "match them. See docs/db-copy-prod-to-test.md."
fi
