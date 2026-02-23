from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import csv
from eliot import start_action


def consent_csv_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "data"
        / "input"
        / "consent_agreement.csv"
    )


def prediction_statistics_csv_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "data"
        / "input"
        / "prediction_statistics.csv"
    )


def get_next_study_number() -> int:
    """
    Return the next sequential `number` used for study exports.

    This reads `data/input/prediction_statistics.csv` and returns max(number)+1, or 0 if missing/empty.
    """
    path = prediction_statistics_csv_path()
    if not path.exists():
        return 0

    with path.open("r", newline="", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        numbers: list[int] = []
        for row in reader:
            raw = (row.get("number") or "").strip()
            if raw.isdigit():
                numbers.append(int(raw))
        return (max(numbers) + 1) if numbers else 0


def consent_row_exists(study_id: str) -> bool:
    path = consent_csv_path()
    if not path.exists():
        return False
    with path.open("r", newline="", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        for row in reader:
            if (row.get("study_id") or "") == study_id:
                return True
    return False


def append_consent_agreement_row(row: dict[str, Any]) -> None:
    """
    Append a consent agreement row to `data/input/consent_agreement.csv`.

    The CSV schema is upgraded in-place if new columns appear.
    """
    path = consent_csv_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    normalized: dict[str, str] = {str(k): "" if v is None else str(v) for k, v in row.items()}
    if "timestamp" not in normalized:
        normalized["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    desired_fieldnames = list(normalized.keys())

    with start_action(action_type=u"append_consent_agreement_row", path=str(path)):
        if not path.exists():
            with path.open("w", newline="", encoding="utf-8") as file_handle:
                writer = csv.DictWriter(file_handle, fieldnames=desired_fieldnames)
                writer.writeheader()
                writer.writerow(normalized)
            return

        with path.open("r", newline="", encoding="utf-8") as file_handle:
            reader = csv.DictReader(file_handle)
            existing_fieldnames = list(reader.fieldnames or [])
            existing_rows = list(reader)

        if existing_fieldnames != desired_fieldnames:
            merged_fieldnames: list[str] = []
            seen: set[str] = set()
            for name in existing_fieldnames + desired_fieldnames:
                if name in seen:
                    continue
                merged_fieldnames.append(name)
                seen.add(name)

            with path.open("w", newline="", encoding="utf-8") as file_handle:
                writer = csv.DictWriter(file_handle, fieldnames=merged_fieldnames)
                writer.writeheader()
                for r in existing_rows:
                    writer.writerow({k: r.get(k, "") for k in merged_fieldnames})
                writer.writerow({k: normalized.get(k, "") for k in merged_fieldnames})
            return

        with path.open("a", newline="", encoding="utf-8") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=existing_fieldnames)
            writer.writerow(normalized)


def ensure_consent_agreement_row(row: dict[str, Any]) -> None:
    """
    Ensure there is at least one consent row for this `study_id`.

    If the row already exists, this is a no-op (prevents duplicates when users bypass the landing page).
    """
    study_id = str(row.get("study_id") or "")
    if not study_id:
        return
    if consent_row_exists(study_id):
        return
    append_consent_agreement_row(row)


def upsert_consent_agreement_fields(study_id: str, updates: dict[str, Any]) -> None:
    """Update fields for an existing consent row, or append a new row if missing.

    This is used for consents that can be given later in the session (e.g. uploaded CGM data usage).
    """
    sid = str(study_id or "").strip()
    if not sid:
        return

    path = consent_csv_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    normalized_updates: dict[str, str] = {str(k): "" if v is None else str(v) for k, v in updates.items()}

    with start_action(action_type=u"upsert_consent_agreement_fields", study_id=sid, path=str(path)):
        if not path.exists():
            append_consent_agreement_row({"study_id": sid, **normalized_updates})
            return

        with path.open("r", newline="", encoding="utf-8") as file_handle:
            reader = csv.DictReader(file_handle)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)

        if "study_id" not in fieldnames:
            fieldnames = ["study_id", *fieldnames]

        updated = False
        new_rows: list[dict[str, str]] = []
        for row in rows:
            if (row.get("study_id") or "") == sid:
                merged = dict(row)
                merged.update(normalized_updates)
                # Keep timestamp fresh for late consents.
                merged["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                new_rows.append(merged)
                updated = True
            else:
                new_rows.append(dict(row))

        if not updated:
            append_consent_agreement_row({"study_id": sid, **normalized_updates})
            return

        # Upgrade schema if needed.
        desired_fieldnames = list(fieldnames)
        for k in normalized_updates.keys():
            if k not in desired_fieldnames:
                desired_fieldnames.append(k)
        if "timestamp" not in desired_fieldnames:
            desired_fieldnames.append("timestamp")

        tmp_path = path.with_suffix(".tmp")
        with tmp_path.open("w", newline="", encoding="utf-8") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=desired_fieldnames)
            writer.writeheader()
            for r in new_rows:
                writer.writerow({k: r.get(k, "") for k in desired_fieldnames})
        tmp_path.replace(path)

