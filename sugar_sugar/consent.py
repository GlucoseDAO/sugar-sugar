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

