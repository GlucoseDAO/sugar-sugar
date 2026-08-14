from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import csv
from eliot import start_action


def resolve_optional_consents(
    *,
    receive_results: bool,
    keep_updated: bool,
) -> tuple[bool, bool, bool]:
    """Return ``(play_only, participate_in_study, no_selection)``.

    The play-only checkbox is gone: ticking 18+ and GDPR *is* study consent.
    Optional boxes (email results, updates, upload) are extras, not a gate.
    ``play_only`` stays in the return tuple / CSV column as always-False so
    existing writers and historical rows keep a stable schema.
    """
    play_only = False
    participate = True
    no_selection = not (receive_results or keep_updated)
    return play_only, participate, no_selection


def apply_optional_consent_choices(
    info: dict[str, Any],
    *,
    receive_results: bool,
    keep_updated: bool,
    upload_own_data: bool,
) -> dict[str, Any]:
    """Stamp resolved optional-consent flags onto ``info`` and return it."""
    play_only, participate, no_selection = resolve_optional_consents(
        receive_results=receive_results,
        keep_updated=keep_updated,
    )
    info["consent_play_only"] = play_only
    info["consent_participate_in_study"] = participate
    info["consent_receive_results_later"] = receive_results
    info["consent_keep_up_to_date"] = keep_updated
    info["consent_no_selection"] = no_selection
    info["consent_upload_own_data"] = upload_own_data
    if upload_own_data:
        info["consent_use_uploaded_data"] = True
    return info


def reconcile_stored_consents(user_info: dict[str, Any]) -> dict[str, Any]:
    """Force leftover ``consent_play_only=True`` off so old sessions still save.

    Sessions that consented when the play-only box existed can still have that
    flag in localStorage. Re-resolving at Start and at the save boundary
    writes those games without asking the player to re-tick boxes.
    """
    return apply_optional_consent_choices(
        user_info,
        receive_results=bool(user_info.get("consent_receive_results_later")),
        keep_updated=bool(user_info.get("consent_keep_up_to_date")),
        upload_own_data=bool(
            user_info.get("consent_upload_own_data")
            or user_info.get("consent_use_uploaded_data")
        ),
    )


def session_age(user_info: Optional[dict[str, Any]]) -> int:
    """Player age for dataframe/CSV writes. ``0`` until ``/profile`` fills it in."""
    if not user_info:
        return 0
    raw = user_info.get("age")
    if raw is None or str(raw).strip() == "":
        return 0
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return 0


def identity_is_complete(user_info: Optional[dict[str, Any]]) -> bool:
    """True when nickname/email/age/gender/location have already been collected.

    New sessions collect these on ``/profile`` after the player exits or
    finishes. Older sessions (and staging/chart synthetics) filled them on
    ``/startup``; treat those as complete so they are not asked again.
    """
    if not user_info:
        return False
    if user_info.get("identity_completed"):
        return True
    age = user_info.get("age")
    try:
        age_ok = age is not None and str(age).strip() != "" and float(age) >= 18
    except (TypeError, ValueError):
        age_ok = False
    return bool(age_ok and user_info.get("gender") and user_info.get("location"))


def stamp_identity_fields(
    info: dict[str, Any],
    *,
    nickname: Optional[str],
    email: Optional[str],
    age: Optional[int | float],
    gender: Optional[str],
    location: Optional[str],
    receive_results: bool,
    keep_updated: bool,
) -> dict[str, Any]:
    """Write end-of-game identity + email-pref fields onto ``info``."""
    from sugar_sugar.nickname import normalize_nickname

    info["nickname"] = normalize_nickname(nickname) or info.get("nickname") or ""
    info["email"] = (str(email).strip() if email else "") or info.get("email") or ""
    info["age"] = age
    info["gender"] = gender or ""
    info["location"] = location or ""
    apply_optional_consent_choices(
        info,
        receive_results=receive_results,
        keep_updated=keep_updated,
        upload_own_data=bool(
            info.get("consent_upload_own_data") or info.get("consent_use_uploaded_data")
        ),
    )
    info["identity_completed"] = True
    return info


def results_destination(user_info: Optional[dict[str, Any]]) -> str:
    """``/profile`` until identity is in, then ``/final``."""
    if identity_is_complete(user_info):
        return "/final"
    return "/profile"


def should_persist_study_data(user_info: Optional[dict[str, Any]]) -> bool:
    """True when this session should write (or update) stats + ranking CSVs.

    Written at Start (demographics, even with zero rounds) and again after
    every submitted round so a player who closes the tab without Exit is
    still in the study files. Same ``study_id`` upserts in place. There is
    no 6- or 12-round minimum and no play-only opt-out. ``uv run chart``
    skips writes via ``_CHART_MODE``.
    """
    if os.environ.get("_CHART_MODE") == "1":
        return False
    if not user_info:
        return False
    if not user_info.get("consent_completed"):
        return False
    reconcile_stored_consents(user_info)
    return True


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

    with path.open("r", newline="", encoding="utf-8", errors="replace") as file_handle:
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
    with path.open("r", newline="", encoding="utf-8", errors="replace") as file_handle:
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

        with path.open("r", newline="", encoding="utf-8", errors="replace") as file_handle:
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

        with path.open("r", newline="", encoding="utf-8", errors="replace") as file_handle:
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

