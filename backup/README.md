# Sugar-Sugar backups

`backup-input.sh` creates a gzip-compressed tar of `data/input/` (prediction CSVs, consent, uploaded CGM files under `users/`). Optional extras: legacy root `prediction_statistics.csv`, `logs/`.

Features: file-level locking (prevents overlapping runs), archive integrity verification, restrictive permissions (`0600`), timestamped log lines, and time-based rotation.

## Quick run

From the repository root:

```bash
chmod +x backup/backup-input.sh
./backup/backup-input.sh
```

Archives default to `backup/archives/sugar-data-input-YYYYMMDDTHHMMSSZ.tar.gz` (that directory is gitignored).

## Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `SUGAR_BACKUP_DEST` | `backup/archives` (next to the script) | Where to write `.tar.gz` files |
| `SUGAR_BACKUP_RETENTION_DAYS` | `30` | Delete matching archives older than this many days |
| `SUGAR_BACKUP_INCLUDE_LEGACY` | unset | Set to `1` to add root `prediction_statistics.csv` if it exists |
| `SUGAR_BACKUP_INCLUDE_LOGS` | unset | Set to `1` to add `logs/` if it exists |
| `SUGAR_BACKUP_DRY_RUN` | unset | Set to `1` to print planned actions only |

Example with a system backup location:

```bash
SUGAR_BACKUP_DEST=/var/backups/sugar-sugar SUGAR_BACKUP_RETENTION_DAYS=14 ./backup/backup-input.sh
```

## Restore

```bash
cd /path/to/sugar-sugar
tar -xzf /path/to/sugar-data-input-20250324T031700Z.tar.gz
```

Paths in the archive are relative to the project root (e.g. `data/input/...`).

## Cron (daily)

```cron
17 3 * * * /path/to/sugar-sugar/backup/backup-input.sh >> /var/log/sugar-sugar-backup.log 2>&1
```

## systemd timer

Example unit files live in `systemd/`. Edit the service file to set the correct `ExecStart` path and `User`, then:

```bash
sudo cp backup/systemd/backup-sugar-sugar.{service,timer} /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now backup-sugar-sugar.timer
```

The timer fires daily around 03:17 UTC with up to 15 minutes of randomized jitter. The service runs at idle I/O priority (`Nice=19`, `IOSchedulingClass=idle`) and retries once on transient failure.
