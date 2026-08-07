"""First-run conversion of pre-nickname ranking CSVs.

`backfill_leaderboard_identity` adds the `email_key`/`nickname` columns and derives
each historical row's `email_key` from the address `prediction_statistics.csv` already
holds for that `study_id`.

Placement is arcade-style, so the migration must never merge or remove a slot -- it
only *links* a player to their own history. Without it, a player returning on a new
device (fresh localStorage -> new study_id, same email) would not have their own past
scores highlighted, counted as their placement, or offered as a nickname suggestion.
A pristine `*.pre-nickname.bak` copy is taken before the first conversion.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any, Iterator

import pytest

from sugar_sugar import app as app_module
from sugar_sugar import nickname as nickname_module
from sugar_sugar.components import submit as submit_module
from sugar_sugar.components.submit import SubmitComponent

_OLD_HEADER = (
    "study_id,run_id,number,timestamp,format,rounds_played,is_example_data,"
    "data_source_name,overall_mae_mgdl,overall_mse_mgdl,overall_rmse_mgdl,"
    "overall_mape_pct\n"
)
_NEW_HEADER = _OLD_HEADER.rstrip("\n") + ",email_key,nickname\n"

_STATS_HEADER = "study_id,run_id,number,timestamp,email,format,overall_mae_mgdl\n"


@pytest.fixture()
def component(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[SubmitComponent]:
    submit = SubmitComponent()
    submit._stats_csv_path = tmp_path / "prediction_statistics.csv"
    submit._ranking_csv_path = tmp_path / "prediction_ranking.csv"
    submit._ranking_by_format_paths = {
        fmt: tmp_path / f"prediction_ranking_{fmt}.csv" for fmt in ("A", "B", "C")
    }
    monkeypatch.setattr(nickname_module, "RANKING_EMAIL_SALT", "test-salt")
    nickname_module._salt.cache_clear()
    yield submit
    nickname_module._salt.cache_clear()


def _old_row(study_id: str, mae: float, *, fmt: str = "ALL", ts: str = "2026-07-01 10:00:00") -> str:
    return f"{study_id},run1,1,{ts},{fmt},12,True,example,{mae},0,0,0\n"


def _new_row(
    study_id: str,
    mae: float,
    *,
    fmt: str = "ALL",
    ts: str = "2026-08-07 10:00:00",
    key: str = "",
    nickname: str = "",
) -> str:
    return f"{study_id},run1,1,{ts},{fmt},12,True,example,{mae},0,0,0,{key},{nickname}\n"


def _stats(*pairs: tuple[str, str]) -> str:
    body = "".join(
        f"{study_id},run1,1,2026-07-01 10:00:00,{email},A,20.0\n" for study_id, email in pairs
    )
    return _STATS_HEADER + body


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _header(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle).fieldnames or [])


# --- schema conversion -------------------------------------------------------


def test_old_schema_gains_the_identity_columns(component: SubmitComponent) -> None:
    component._ranking_csv_path.write_text(_OLD_HEADER + _old_row("s1", 20.0), encoding="utf-8")
    component.backfill_leaderboard_identity()
    assert _header(component._ranking_csv_path)[-2:] == ["email_key", "nickname"]


def test_all_four_ranking_files_are_converted(component: SubmitComponent) -> None:
    component._stats_csv_path.write_text(_stats(("s1", "ann@x.com")), encoding="utf-8")
    component._ranking_csv_path.write_text(_OLD_HEADER + _old_row("s1", 20.0), encoding="utf-8")
    for fmt, path in component._ranking_by_format_paths.items():
        path.write_text(_OLD_HEADER + _old_row("s1", 20.0, fmt=fmt), encoding="utf-8")

    assert component.backfill_leaderboard_identity() == 4
    for path in [component._ranking_csv_path, *component._ranking_by_format_paths.values()]:
        assert _rows(path)[0]["email_key"] == nickname_module.email_key("ann@x.com")


def test_header_only_csv_is_converted_without_error(component: SubmitComponent) -> None:
    """A fresh checkout's ranking CSVs are header-only."""
    component._ranking_csv_path.write_text(_OLD_HEADER, encoding="utf-8")
    assert component.backfill_leaderboard_identity() == 0
    assert "email_key" in _header(component._ranking_csv_path)


def test_missing_files_are_skipped(component: SubmitComponent) -> None:
    assert component.backfill_leaderboard_identity() == 0
    assert not component._ranking_csv_path.exists()


# --- key derivation ----------------------------------------------------------


def test_email_key_is_derived_from_the_statistics_csv(component: SubmitComponent) -> None:
    component._stats_csv_path.write_text(
        _stats(("s1", "Ann@X.com"), ("s2", "bob@x.com")), encoding="utf-8"
    )
    component._ranking_csv_path.write_text(
        _OLD_HEADER + _old_row("s1", 20.0) + _old_row("s2", 10.0), encoding="utf-8"
    )
    assert component.backfill_leaderboard_identity() == 2

    by_study = {row["study_id"]: row["email_key"] for row in _rows(component._ranking_csv_path)}
    assert by_study["s1"] == nickname_module.email_key("ann@x.com")  # casefolded
    assert by_study["s2"] == nickname_module.email_key("bob@x.com")


def test_the_address_itself_is_never_written(component: SubmitComponent) -> None:
    component._stats_csv_path.write_text(_stats(("s1", "ann@x.com")), encoding="utf-8")
    component._ranking_csv_path.write_text(_OLD_HEADER + _old_row("s1", 20.0), encoding="utf-8")
    component.backfill_leaderboard_identity()
    assert "ann@x.com" not in component._ranking_csv_path.read_text(encoding="utf-8")


def test_study_without_a_recorded_email_keeps_a_blank_key(component: SubmitComponent) -> None:
    """It then groups by study_id, exactly like an anonymous player."""
    component._stats_csv_path.write_text(_stats(("s1", "")), encoding="utf-8")
    component._ranking_csv_path.write_text(_OLD_HEADER + _old_row("s1", 20.0), encoding="utf-8")
    assert component.backfill_leaderboard_identity() == 0
    assert _rows(component._ranking_csv_path)[0]["email_key"] == ""


def test_missing_statistics_csv_only_upgrades_the_schema(component: SubmitComponent) -> None:
    component._ranking_csv_path.write_text(_OLD_HEADER + _old_row("s1", 20.0), encoding="utf-8")
    assert component.backfill_leaderboard_identity() == 0
    assert _rows(component._ranking_csv_path)[0]["email_key"] == ""


def test_most_recent_address_wins_for_one_study(component: SubmitComponent) -> None:
    """A corrected address is the one a future run would hash, so match it."""
    component._stats_csv_path.write_text(
        _stats(("s1", "old@x.com"), ("s1", "new@x.com")), encoding="utf-8"
    )
    component._ranking_csv_path.write_text(_OLD_HEADER + _old_row("s1", 20.0), encoding="utf-8")
    component.backfill_leaderboard_identity()
    assert _rows(component._ranking_csv_path)[0]["email_key"] == nickname_module.email_key("new@x.com")


def test_an_existing_key_is_never_relabelled(component: SubmitComponent) -> None:
    component._stats_csv_path.write_text(_stats(("s1", "ann@x.com")), encoding="utf-8")
    component._ranking_csv_path.write_text(
        _NEW_HEADER + _new_row("s1", 20.0, key="already-set"), encoding="utf-8"
    )
    assert component.backfill_leaderboard_identity() == 0
    assert _rows(component._ranking_csv_path)[0]["email_key"] == "already-set"


def test_existing_nicknames_survive_the_migration(component: SubmitComponent) -> None:
    component._stats_csv_path.write_text(_stats(("s1", "ann@x.com")), encoding="utf-8")
    component._ranking_csv_path.write_text(
        _NEW_HEADER + _new_row("s1", 20.0, nickname="Ninja"), encoding="utf-8"
    )
    component.backfill_leaderboard_identity()
    row = _rows(component._ranking_csv_path)[0]
    assert row["nickname"] == "Ninja"
    assert row["email_key"] == nickname_module.email_key("ann@x.com")


# --- backup before conversion ------------------------------------------------


def _backup(path: Path) -> Path:
    return path.with_name(path.name + ".pre-nickname.bak")


def test_a_pristine_copy_is_kept_before_converting(component: SubmitComponent) -> None:
    component._stats_csv_path.write_text(_stats(("s1", "ann@x.com")), encoding="utf-8")
    original = _OLD_HEADER + _old_row("s1", 20.0)
    component._ranking_csv_path.write_text(original, encoding="utf-8")

    component.backfill_leaderboard_identity()

    backup = _backup(component._ranking_csv_path)
    assert backup.exists()
    assert backup.read_text(encoding="utf-8") == original  # byte-for-byte pre-conversion
    assert component._ranking_csv_path.read_text(encoding="utf-8") != original


def test_every_converted_file_is_backed_up(component: SubmitComponent) -> None:
    component._stats_csv_path.write_text(_stats(("s1", "ann@x.com")), encoding="utf-8")
    component._ranking_csv_path.write_text(_OLD_HEADER + _old_row("s1", 20.0), encoding="utf-8")
    for fmt, path in component._ranking_by_format_paths.items():
        path.write_text(_OLD_HEADER + _old_row("s1", 20.0, fmt=fmt), encoding="utf-8")

    component.backfill_leaderboard_identity()

    for path in [component._ranking_csv_path, *component._ranking_by_format_paths.values()]:
        assert _backup(path).exists(), path.name


def test_a_later_boot_never_overwrites_the_pristine_backup(component: SubmitComponent) -> None:
    """Otherwise the second run would copy converted content over the only original."""
    component._stats_csv_path.write_text(_stats(("s1", "ann@x.com")), encoding="utf-8")
    original = _OLD_HEADER + _old_row("s1", 20.0)
    component._ranking_csv_path.write_text(original, encoding="utf-8")
    component.backfill_leaderboard_identity()

    # A new game lands, then another boot converts again.
    with component._ranking_csv_path.open("a", encoding="utf-8") as handle:
        handle.write(_old_row("s2", 11.0).rstrip("\n") + ",,\n")
    component._stats_csv_path.write_text(
        _stats(("s1", "ann@x.com"), ("s2", "bob@x.com")), encoding="utf-8"
    )
    component.backfill_leaderboard_identity()

    assert _backup(component._ranking_csv_path).read_text(encoding="utf-8") == original


def test_nothing_is_backed_up_when_nothing_is_converted(component: SubmitComponent) -> None:
    """An already-converted file is left alone, so no backup is taken."""
    component._ranking_csv_path.write_text(
        _NEW_HEADER + _new_row("s1", 20.0, key="k"), encoding="utf-8"
    )
    assert component.backfill_leaderboard_identity() == 0
    assert not _backup(component._ranking_csv_path).exists()


def test_the_backup_restores_the_original_board(component: SubmitComponent) -> None:
    """The point of the backup: `cp` it back and you are where you started."""
    component._stats_csv_path.write_text(_stats(("s1", "ann@x.com")), encoding="utf-8")
    original = _OLD_HEADER + _old_row("s1", 22.0) + _old_row("s7", 14.0)
    component._ranking_csv_path.write_text(original, encoding="utf-8")
    component.backfill_leaderboard_identity()

    shutil.copy2(_backup(component._ranking_csv_path), component._ranking_csv_path)

    assert component._ranking_csv_path.read_text(encoding="utf-8") == original
    restored = app_module._leaderboard_snapshot(
        component._ranking_csv_path, study_id="s1", format_filter="ALL"
    )
    assert restored is not None and restored["total"] == 2


# --- idempotency -------------------------------------------------------------


def test_running_twice_changes_nothing_the_second_time(component: SubmitComponent) -> None:
    component._stats_csv_path.write_text(_stats(("s1", "ann@x.com")), encoding="utf-8")
    component._ranking_csv_path.write_text(_OLD_HEADER + _old_row("s1", 20.0), encoding="utf-8")

    assert component.backfill_leaderboard_identity() == 1
    first = component._ranking_csv_path.read_text(encoding="utf-8")
    assert component.backfill_leaderboard_identity() == 0
    assert component._ranking_csv_path.read_text(encoding="utf-8") == first


def test_no_temp_file_is_left_behind(component: SubmitComponent) -> None:
    component._stats_csv_path.write_text(_stats(("s1", "ann@x.com")), encoding="utf-8")
    component._ranking_csv_path.write_text(_OLD_HEADER + _old_row("s1", 20.0), encoding="utf-8")
    component.backfill_leaderboard_identity()
    assert not component._ranking_csv_path.with_suffix(".tmp").exists()


def test_other_columns_are_preserved_verbatim(component: SubmitComponent) -> None:
    component._stats_csv_path.write_text(_stats(("s1", "ann@x.com")), encoding="utf-8")
    component._ranking_csv_path.write_text(_OLD_HEADER + _old_row("s1", 20.0), encoding="utf-8")
    component.backfill_leaderboard_identity()
    row = _rows(component._ranking_csv_path)[0]
    assert row["overall_mae_mgdl"] == "20.0"
    assert row["rounds_played"] == "12"
    assert row["data_source_name"] == "example"
    assert row["timestamp"] == "2026-07-01 10:00:00"


# --- wiring ------------------------------------------------------------------


def test_construction_runs_the_migration_once_per_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It must fire automatically at boot, but not on every prediction-page render
    (`create_prediction_layout` constructs a SubmitComponent too, and this reads
    five CSVs)."""
    calls: list[int] = []
    monkeypatch.setattr(submit_module, "_IDENTITY_BACKFILL_DONE", False)
    monkeypatch.setattr(
        SubmitComponent,
        "backfill_leaderboard_identity",
        lambda self: calls.append(1) or 0,
    )

    SubmitComponent()
    SubmitComponent()
    SubmitComponent()

    assert len(calls) == 1


def test_the_legacy_row_repair_survives_the_new_columns(component: SubmitComponent) -> None:
    """`_repair_misaligned_csv_rows` keys off a hardcoded 12-column order that
    deliberately excludes email_key/nickname; it must not scramble them."""
    # A corrupted row: values written in `ranking_desired` order into a header that
    # has run_id/format elsewhere -- detected via a non-numeric overall_mae_mgdl.
    corrupt = (
        "study_id,number,timestamp,rounds_played,is_example_data,data_source_name,"
        "overall_mae_mgdl,overall_mse_mgdl,overall_rmse_mgdl,overall_mape_pct,"
        "run_id,format,email_key,nickname\n"
        "s1,run1,1,2026-07-01 10:00:00,ALL,12,True,example,20.0,0,0,0,hash-ann,Ninja\n"
    )
    component._ranking_csv_path.write_text(corrupt, encoding="utf-8")
    component._repair_misaligned_csv_rows()

    row = _rows(component._ranking_csv_path)[0]
    assert row["email_key"] == "hash-ann"
    assert row["nickname"] == "Ninja"
    assert row["overall_mae_mgdl"] == "20.0"  # actually repaired


# --- the bug this exists to fix ---------------------------------------------


def _board(component: SubmitComponent, *, study_id: str, key: str) -> dict[str, Any]:
    snapshot = app_module._leaderboard_snapshot(
        component._ranking_csv_path,
        study_id=study_id,
        key=key,
        format_filter="ALL",
        top_n=10,
    )
    assert snapshot is not None
    return snapshot


def test_a_new_device_cannot_see_your_old_scores_as_yours_until_migrated(
    component: SubmitComponent,
) -> None:
    """The reason the backfill exists.

    Historical rows carry only a `study_id`. On a new device (fresh localStorage ->
    new study_id, same email) they match neither the study_id nor the email key, so
    the player's own past scores read as somebody else's.
    """
    key = nickname_module.email_key("ann@x.com")
    component._stats_csv_path.write_text(_stats(("s1", "ann@x.com")), encoding="utf-8")
    component._ranking_csv_path.write_text(
        _OLD_HEADER + _old_row("s1", 25.0) + _old_row("s1", 20.0), encoding="utf-8"
    )

    before = _board(component, study_id="s99", key=key)
    assert before["total"] == 2
    assert [entry["is_you"] for entry in before["top"]] == [False, False]
    assert before["rank"] is None

    assert component.backfill_leaderboard_identity() == 2

    after = _board(component, study_id="s99", key=key)
    # Arcade rules: still two slots, nothing merged away -- but now they are HERS.
    assert after["total"] == 2
    assert [entry["is_you"] for entry in after["top"]] == [True, True]
    assert after["rank"] == 1 and after["mae"] == 20.0


def test_migration_does_not_merge_or_remove_any_slot(component: SubmitComponent) -> None:
    """Backfilling identities must not cost anyone their place on the board."""
    component._stats_csv_path.write_text(
        _stats(("s1", "ann@x.com"), ("s7", "ann@x.com"), ("s3", "bob@x.com")), encoding="utf-8"
    )
    component._ranking_csv_path.write_text(
        _OLD_HEADER + _old_row("s1", 22.0) + _old_row("s7", 14.0) + _old_row("s3", 19.0),
        encoding="utf-8",
    )

    before = _board(component, study_id="", key="")
    assert before["total"] == 3

    assert component.backfill_leaderboard_identity() == 3

    after = _board(component, study_id="", key="")
    assert after["total"] == 3
    assert [entry["mae"] for entry in after["top"]] == [14.0, 19.0, 22.0]
    # Ann's two devices are now recognised as one person for the stat chip only.
    assert before["players"] == 3 and after["players"] == 2


def test_migration_enables_the_cross_device_nickname_suggestion(
    component: SubmitComponent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`stored_nickname` reads the real data/input tree, so point it at the fixture."""
    monkeypatch.setattr(app_module, "project_root", tmp_path.parent)
    input_dir = tmp_path.parent / "data" / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    component._stats_csv_path = input_dir / "prediction_statistics.csv"
    component._ranking_csv_path = input_dir / "prediction_ranking.csv"
    component._ranking_by_format_paths = {
        fmt: input_dir / f"prediction_ranking_{fmt}.csv" for fmt in ("A", "B", "C")
    }

    key = nickname_module.email_key("ann@x.com")
    component._stats_csv_path.write_text(_stats(("s1", "ann@x.com")), encoding="utf-8")
    component._ranking_csv_path.write_text(
        _NEW_HEADER + _new_row("s1", 20.0, nickname="Ninja"), encoding="utf-8"
    )

    # New device: the old row is not yet linked to her email.
    assert app_module.stored_nickname(study_id="s99", key=key) == ""
    component.backfill_leaderboard_identity()
    assert app_module.stored_nickname(study_id="s99", key=key) == "Ninja"
