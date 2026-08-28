"""`scripts/copy-input-to-test.sh` re-keys ranking rows to the destination's salt.

The script re-implements `sugar_sugar.nickname.email_key` in stdlib python (it has
to run on a box with no synced venv), so the digests it writes must be identical
to the ones the app computes for the same address under the same salt. If those
two ever drift, every copied row silently stops matching its own player -- which
is the exact bug re-keying exists to prevent, reintroduced invisibly.

Also pins the safety properties the script is responsible for: no directory under
data/input travels, and addresses are gone unless explicitly kept.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from sugar_sugar.nickname import email_key

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "copy-input-to-test.sh"

STATS_HEADER = (
    "study_id,run_id,number,timestamp,email,format,is_example_data,data_source_name,"
    "age,user_id,gender,uses_cgm,cgm_duration_years,diabetic,diabetic_type,"
    "diabetes_duration,location,rounds_played,predicted_values,real_values,"
    "prediction_times,overall_mae_mgdl,overall_mse_mgdl,overall_rmse_mgdl,"
    "overall_mape_pct,per_round_metrics"
).split(",")
RANK_HEADER = (
    "study_id,run_id,number,timestamp,format,rounds_played,is_example_data,"
    "data_source_name,overall_mae_mgdl,overall_mse_mgdl,overall_rmse_mgdl,"
    "overall_mape_pct,email_key,nickname"
).split(",")

DEST_SALT = "destination-salt-not-the-source-one"
PEOPLE = {"s1": "Ann@Example.com", "s2": "bob@example.com", "s3": "cara@example.com"}

pytestmark = pytest.mark.skipif(
    shutil.which("rsync") is None, reason="rsync not available"
)


def _write_csv(path: Path, header: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{column: "" for column in header}, **row})


@pytest.fixture()
def tree(tmp_path: Path) -> tuple[Path, Path]:
    """A source data/input with real addresses, and a destination with its own salt."""
    source = tmp_path / "prod" / "data" / "input"
    dest = tmp_path / "test" / "data" / "input"
    (source / "users").mkdir(parents=True)
    (source / "study_design").mkdir()
    dest.mkdir(parents=True)

    _write_csv(
        source / "prediction_statistics.csv",
        STATS_HEADER,
        [
            {
                "study_id": study_id,
                "run_id": "r1",
                "timestamp": "2026-08-01 10:00:00",
                "email": address,
                "format": "A",
                "rounds_played": "12",
                "location": "Berlin, Germany",  # a comma inside a quoted field
                "overall_mae_mgdl": "20.0",
                "per_round_metrics": str(
                    [{"round_number": 1, "mae": 20.0, "data_source_name": "BIGIDEAS-001.csv"}]
                ),
            }
            for study_id, address in PEOPLE.items()
        ],
    )
    # Ranking rows carrying digests from some other deployment's salt.
    _write_csv(
        source / "prediction_ranking.csv",
        RANK_HEADER,
        [
            {
                "study_id": study_id,
                "run_id": "r1",
                "timestamp": "2026-08-01 10:00:00",
                "format": "ALL",
                "rounds_played": "12",
                "overall_mae_mgdl": "20.0",
                "email_key": "0000deadbeef0000",
                "nickname": "Ninja" if study_id == "s1" else "",
            }
            for study_id in PEOPLE
        ],
    )
    # Things that must not travel.
    (source / "users" / "Clarity_Export_Real_Patient_2026.csv").write_text("gl\n120\n")
    (source / "study_design" / "guide.md").write_text("# guide\n")
    (dest.parent / ".ranking_salt").write_text(DEST_SALT, encoding="utf-8")
    return source, dest


def _run(source: Path, dest: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(SCRIPT), str(source), str(dest), "--yes", *args],
        capture_output=True,
        text=True,
        check=True,
    )


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_rekey_matches_the_apps_email_key(tree: tuple[Path, Path]) -> None:
    """The whole point: digests the destination can reproduce for itself."""
    source, dest = tree
    _run(source, dest)

    # email_key() reads the salt at import via an lru_cache, so ask for the value
    # the app *would* compute under the destination's salt in a clean process.
    expected = {
        study_id: subprocess.run(
            [
                "python3", "-c",
                "import sys;from sugar_sugar.nickname import email_key;"
                "print(email_key(sys.argv[1]))",
                address,
            ],
            capture_output=True, text=True, check=True,
            env={**os.environ, "RANKING_EMAIL_SALT": DEST_SALT},
            cwd=str(SCRIPT.parent.parent),
        ).stdout.strip()
        for study_id, address in PEOPLE.items()
    }

    for row in _rows(dest / "prediction_ranking.csv"):
        assert row["email_key"] == expected[row["study_id"]], row["study_id"]
        assert row["email_key"] != "0000deadbeef0000"


def test_same_address_yields_one_identity(tree: tuple[Path, Path]) -> None:
    """Two study_ids, one person: after the re-key they share a key, as on prod."""
    source, dest = tree
    with (source / "prediction_statistics.csv").open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATS_HEADER)
        writer.writerow({**{c: "" for c in STATS_HEADER}, "study_id": "s4",
                         "run_id": "r1", "timestamp": "2026-08-02 10:00:00",
                         # same person as s1, different device: different case and spacing
                         "email": "  ANN@example.com ", "rounds_played": "12"})
    with (source / "prediction_ranking.csv").open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RANK_HEADER)
        writer.writerow({**{c: "" for c in RANK_HEADER}, "study_id": "s4", "run_id": "r1",
                         "timestamp": "2026-08-02 10:00:00", "format": "ALL",
                         "rounds_played": "12", "overall_mae_mgdl": "18.0",
                         "email_key": "1111deadbeef1111"})

    _run(source, dest)
    keys = {row["study_id"]: row["email_key"] for row in _rows(dest / "prediction_ranking.csv")}
    assert keys["s1"] == keys["s4"] != ""


def test_no_rekey_leaves_the_source_digests(tree: tuple[Path, Path]) -> None:
    source, dest = tree
    _run(source, dest, "--no-rekey")
    assert all(
        row["email_key"] == "0000deadbeef0000"
        for row in _rows(dest / "prediction_ranking.csv")
    )


def test_missing_destination_salt_copies_without_rekeying(tree: tuple[Path, Path]) -> None:
    """A destination that never ran the app must not fail the copy."""
    source, dest = tree
    (dest.parent / ".ranking_salt").unlink()
    result = _run(source, dest)
    assert "skipping the re-key" in result.stdout
    assert (dest / "prediction_ranking.csv").exists()


def test_addresses_and_directories_never_reach_the_destination(tree: tuple[Path, Path]) -> None:
    source, dest = tree
    _run(source, dest)

    assert [row["email"] for row in _rows(dest / "prediction_statistics.csv")] == ["", "", ""]
    # Non-recursive copy: no directory travels, so uploads cannot leak.
    assert [path.name for path in dest.iterdir() if path.is_dir()] == []
    blob = " ".join(path.read_text(encoding="utf-8") for path in dest.glob("*.csv"))
    for address in PEOPLE.values():
        assert address not in blob
    assert "Real_Patient" not in blob
    # The quoted comma-bearing column survived the rewrite.
    assert _rows(dest / "prediction_statistics.csv")[0]["location"] == "Berlin, Germany"


def test_uploaded_filenames_are_pseudonymised(tree: tuple[Path, Path]) -> None:
    """People name CGM exports after themselves, and that name rides in
    data_source_name and in every per_round_metrics entry -- so keeping the
    uploads themselves out of the copy does not keep the patients' names out.
    The pseudonym must classify identically: not a corpus, not example, so the
    player's own status still decides which board the round counts for."""
    source, dest = tree
    patient = "Surname Firstname 06.09.2021 (1)01.05-30.07.2025.csv"
    with (source / "prediction_statistics.csv").open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=STATS_HEADER).writerow(
            {**{c: "" for c in STATS_HEADER}, "study_id": "own", "run_id": "r1",
             "timestamp": "2026-08-05 10:00:00", "email": "own@example.com",
             "format": "B", "diabetic": "True", "rounds_played": "6",
             "data_source_name": patient,
             "per_round_metrics": str(
                 [{"round_number": i + 1, "mae": 20.0, "data_source_name": patient,
                   "is_example_data": False} for i in range(6)]
             )}
        )

    _run(source, dest)
    blob = (dest / "prediction_statistics.csv").read_text(encoding="utf-8")
    assert "Surname Firstname" not in blob
    row = [r for r in _rows(dest / "prediction_statistics.csv") if r["study_id"] == "own"][0]
    assert row["data_source_name"] == "own-upload-1.csv"
    # The name is gone from the embedded per-round copies too, not just the column.
    assert "own-upload-1.csv" in row["per_round_metrics"]
    # Public corpus names are recognised and kept, so they still classify.
    kept = [r for r in _rows(dest / "prediction_statistics.csv") if r["study_id"] == "s1"][0]
    assert kept["data_source_name"] in ("", "example.csv")

    # Opting out is possible for data that is knowingly public.
    _run(source, dest, "--keep-source-names")
    assert patient in (dest / "prediction_statistics.csv").read_text(encoding="utf-8")


def test_with_emails_keeps_them(tree: tuple[Path, Path]) -> None:
    source, dest = tree
    _run(source, dest, "--with-emails")
    addresses = {row["email"] for row in _rows(dest / "prediction_statistics.csv")}
    assert addresses == set(PEOPLE.values())


def test_refuses_to_copy_a_directory_onto_itself(tree: tuple[Path, Path]) -> None:
    """`../sugar-sugar` from inside ~/staging/sugar-sugar is that same checkout.
    The self-copy that follows looks entirely successful -- identical row counts,
    nothing re-keyed, a destination that never gains the players you came for --
    so it has to be refused rather than reported as done."""
    source, _ = tree
    result = subprocess.run(
        [str(SCRIPT), str(source), str(source), "--yes"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "same directory" in result.stderr
