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
# Options:
#   --with-emails   keep the raw email column (real participant data -- only onto
#                   a box you would trust with prod)
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
SRC=""
DST=""

log()  { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }
usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//; $d'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-emails) WITH_EMAILS=1; shift ;;
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

run_at "${SRC}" "test -d '$(path_of "${SRC}")'" \
  || die "source is not a directory: ${SRC}"
run_at "${DST}" "test -d '$(path_of "${DST}")'" \
  || die "destination is not a directory: ${DST} (create it first -- refusing to invent a path)"

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

# ---- 2. scrub: blank the one column holding participant addresses ------------
STATS="${STAGING}/prediction_statistics.csv"
if [[ ${WITH_EMAILS} -eq 1 ]]; then
  log "WARNING: --with-emails -- real addresses will reach ${DST}"
elif [[ -f "${STATS}" ]]; then
  python3 - "${STATS}" <<'PY'
import csv, sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.reader(handle))
if not rows:
    sys.exit(0)

header = rows[0]
if "email" not in header:
    print("  note: no email column in prediction_statistics.csv, nothing to scrub")
    sys.exit(0)

index = header.index("email")
blanked = 0
# csv.reader/writer (not DictReader) so a row with an unexpected field count keeps
# its shape instead of being silently reshaped to the header's width.
for row in rows[1:]:
    if index < len(row) and row[index]:
        row[index] = ""
        blanked += 1

tmp = path.with_suffix(path.suffix + ".tmp")
with tmp.open("w", newline="", encoding="utf-8") as handle:
    csv.writer(handle).writerows(rows)
tmp.replace(path)
print(f"  blanked {blanked} email values in {len(rows) - 1} rows")
PY
fi

# ---- 3. report exactly what is about to be written --------------------------
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
  [[ -t 0 ]] || die "not a TTY: pass --yes to confirm writing to ${DST}"
  read -r -p "Overwrite these CSVs in ${DST}? [y/N] " reply
  [[ "${reply}" =~ ^[Yy]$ ]] || { log "aborted"; exit 1; }
fi

# ---- 4. snapshot the destination before overwriting it ----------------------
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

# ---- 5. push ----------------------------------------------------------------
"${RSYNC_WRITE[@]}" "${STAGING}/" "${DST}/"
log "OK: copied ${#STAGED[@]} CSV files into ${DST}"

echo
if [[ ${WITH_EMAILS} -eq 0 ]]; then
  log "Emails are blank, so identity in test falls back to study_id: one person"
  log "who played on two devices now reads as two players. Nicknames still resolve"
  log "(they join on study_id). Still pseudonymized, not anonymized -- demographics"
  log "and location remain, and study_id keys back to the source's records."
fi
log "The ranking salt was NOT copied (it never should be): the /highscore class"
log "boards work regardless, but /final's \"You\" highlight will not match copied"
log "rows for a tester reusing a source-side email. See docs/db-copy-prod-to-test.md."
