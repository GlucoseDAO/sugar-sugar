#!/usr/bin/env bash
# Archive data/input (study CSVs, consent, user uploads) with time-based rotation.
#
# Environment (optional):
#   SUGAR_BACKUP_DEST              Destination directory for .tar.gz files (default: backup/archives next to this script)
#   SUGAR_BACKUP_RETENTION_DAYS    Delete archives older than this many days (default: 30)
#   SUGAR_BACKUP_INCLUDE_LEGACY    If set to 1, include project-root prediction_statistics.csv when present
#   SUGAR_BACKUP_INCLUDE_LOGS      If set to 1, include logs/ directory when present
#   SUGAR_BACKUP_DRY_RUN           If set to 1, print actions only; do not write or delete

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEST="${SUGAR_BACKUP_DEST:-${SCRIPT_DIR}/archives}"
RETENTION_DAYS="${SUGAR_BACKUP_RETENTION_DAYS:-30}"
DRY_RUN="${SUGAR_BACKUP_DRY_RUN:-0}"

SOURCE_INPUT="${PROJECT_ROOT}/data/input"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ARCHIVE_NAME="sugar-data-input-${STAMP}.tar.gz"
ARCHIVE_PATH="${DEST}/${ARCHIVE_NAME}"
LOCK_FILE="${DEST}/.backup-input.lock"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

if [[ ! -d "${SOURCE_INPUT}" ]]; then
  log "ERROR: missing source directory: ${SOURCE_INPUT}" >&2
  exit 1
fi

TAR_PATHS=(data/input)

if [[ "${SUGAR_BACKUP_INCLUDE_LEGACY:-0}" == "1" ]]; then
  legacy="${PROJECT_ROOT}/prediction_statistics.csv"
  if [[ -f "${legacy}" ]]; then
    TAR_PATHS+=(prediction_statistics.csv)
  else
    log "WARN: SUGAR_BACKUP_INCLUDE_LEGACY=1 but ${legacy} not found, skipping"
  fi
fi

if [[ "${SUGAR_BACKUP_INCLUDE_LOGS:-0}" == "1" ]]; then
  if [[ -d "${PROJECT_ROOT}/logs" ]]; then
    TAR_PATHS+=(logs)
  else
    log "WARN: SUGAR_BACKUP_INCLUDE_LOGS=1 but ${PROJECT_ROOT}/logs/ not found, skipping"
  fi
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  log "DRY_RUN: would mkdir -p ${DEST}"
  log "DRY_RUN: would tar -C ${PROJECT_ROOT} -czf ${ARCHIVE_PATH} ${TAR_PATHS[*]}"
  log "DRY_RUN: would delete archives under ${DEST} older than ${RETENTION_DAYS} days matching sugar-data-input-*.tar.gz"
  exit 0
fi

mkdir -p "${DEST}"
chmod 0700 "${DEST}"

# Prevent overlapping runs (flock is part of util-linux, available on all major distros)
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  log "ERROR: another backup is already running (lock: ${LOCK_FILE})" >&2
  exit 1
fi

tar -C "${PROJECT_ROOT}" -czf "${ARCHIVE_PATH}" "${TAR_PATHS[@]}"
chmod 0600 "${ARCHIVE_PATH}"

# Verify archive integrity before declaring success
if ! gzip -t "${ARCHIVE_PATH}" 2>/dev/null; then
  log "ERROR: archive failed integrity check, removing corrupt file: ${ARCHIVE_PATH}" >&2
  rm -f "${ARCHIVE_PATH}"
  exit 1
fi

log "OK: wrote ${ARCHIVE_PATH} ($(du -h "${ARCHIVE_PATH}" | cut -f1))"

# Rotation: remove archives older than RETENTION_DAYS.
# Run outside set -e so a rotation failure doesn't mask a successful backup.
if ! find "${DEST}" -maxdepth 1 -type f -name 'sugar-data-input-*.tar.gz' \
     -mtime "+${RETENTION_DAYS}" -print -delete; then
  log "WARN: rotation encountered errors (backup itself succeeded)"
fi
