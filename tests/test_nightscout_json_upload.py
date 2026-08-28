"""Nightscout entries.json is an uploadable CGM export.

Nightscout's own CSV export is not usable -- ``/api/v1/entries.csv`` is
headerless with five hardcoded columns and there is no treatments CSV at all --
so the JSON the site serves at ``/api/v1/entries.json`` is what the upload hint
asks for and what these tests pin down.

``test_treatments_with_a_long_null_run_are_loaded`` was written red, against
cgm-format 0.12.0, and went green on its own when the floor moved to 0.12.2 --
which is exactly why it was never marked ``xfail``. It stays as a regression
guard: the shape it covers is what broke a real player's import.
"""

from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path

import polars as pl
import pytest

from sugar_sugar.data import (
    classify_nightscout_uploads,
    decode_upload_bytes,
    decode_upload_files,
    is_nightscout_entries_json,
    load_glucose_data,
    load_nightscout_json_data,
    load_nightscout_uploads,
    nightscout_json_kind,
    safe_upload_filename,
)

# All three fixtures are synthetic and committed. The real Nightscout exports
# beside them are gitignored like every other real export in this directory, so
# a test that depended on them would error out on a fresh checkout.
TESTDATA_DIR = Path(__file__).parent / "testdata"
ENTRIES_JSON = TESTDATA_DIR / "Nightscout_entries_synthetic.json"
TREATMENTS_JSON = TESTDATA_DIR / "Nightscout_treatments_synthetic.json"
SPARSE_TREATMENTS_JSON = TESTDATA_DIR / "Nightscout_treatments_sparse_carbs_synthetic.json"

GLUCOSE_COLUMNS = ["time", "gl", "prediction", "age", "user_id"]


def test_entries_json_is_detected() -> None:
    """The fixture opens with a `cal` and an `mbg` record on purpose.

    cgm-format's own sniff reads a 2000-character prefix and would miss a real
    export whose first records are not sensor readings; this detector scans the
    whole array so a leading calibration cannot hide the file from it.
    """
    assert is_nightscout_entries_json(ENTRIES_JSON)


@pytest.mark.parametrize(
    "name",
    [
        "Nightscout_treatments_synthetic.json",  # JSON, but carries no glucose
        "Clarity_Export_synthetic.csv",          # a vendor CSV -- the library's job
        "Unified_synthetic.csv",                 # an already-unified frame -- the library's job
    ],
)
def test_non_entries_files_are_not_claimed(name: str) -> None:
    """The detector must not swallow files the library should handle."""
    assert not is_nightscout_entries_json(TESTDATA_DIR / name)


def test_detection_reads_content_not_the_file_name(tmp_path: Path) -> None:
    """Both upload handlers rewrite the extension, so the name proves nothing."""
    misnamed = tmp_path / "20260828_120000_entries.json.csv"
    misnamed.write_bytes(ENTRIES_JSON.read_bytes())
    assert is_nightscout_entries_json(misnamed)

    glucose_df, _events_df = load_glucose_data(misnamed)
    assert glucose_df.height > 0


def test_router_loads_entries_json() -> None:
    """load_glucose_data routes a Nightscout export past FormatParser.parse_file.

    ``detect_format`` pattern-matches CSV headers and raises UnknownFormatError
    on JSON, which is why the detector has to run ahead of it.
    """
    glucose_df, events_df = load_glucose_data(ENTRIES_JSON)

    assert glucose_df.columns == GLUCOSE_COLUMNS
    assert glucose_df.height > 0
    assert glucose_df["gl"].null_count() == 0
    assert glucose_df["time"].is_sorted()
    # Entries alone carry no treatments, so no event markers.
    assert events_df.height == 0


def test_entries_json_survives_the_upload_transport() -> None:
    """The gzip+base64 payload the browser sends must round-trip byte-identically."""
    raw = ENTRIES_JSON.read_bytes()
    payload = "gzip:" + base64.b64encode(gzip.compress(raw)).decode("ascii")

    decoded = decode_upload_bytes(payload)

    assert decoded == raw
    assert json.loads(decoded.decode("utf-8"))


def test_treatments_are_loaded_when_supplied() -> None:
    """Passing treatments alongside entries produces event markers.

    This fixture's `carbs` column has a value inside polars' 100-row inference
    window, so it passed even on 0.12.0. That is the point: it isolated the
    wiring from the library bug, so the sparse-carbs test failing next door was
    attributable to cgm-format rather than to this repo.
    """
    glucose_df, events_df = load_nightscout_json_data(ENTRIES_JSON, TREATMENTS_JSON)

    assert glucose_df.height > 0
    assert events_df.height > 0
    assert set(events_df["event_type"].unique()) <= {"Insulin", "Carbohydrates", "Exercise"}


def test_treatments_with_a_long_null_run_are_loaded() -> None:
    """Regression guard for the bug that broke a real import -- FEEDBACK.md issue 1.

    A closed-loop user whose pump writes a Temp Basal every few minutes but who
    logs carbs rarely has `carbs` null for far more than the 100 rows polars
    samples to infer dtypes. On cgm-format 0.12.0 the column inferred as `Null`,
    appending the first real value raised ComputeError, and the whole import
    died. Fixed in 0.12.2, which is the floor in ``pyproject.toml``.

    If this ever goes red again, the fix belongs upstream -- not in ``data.py``.
    """
    glucose_df, events_df = load_nightscout_json_data(ENTRIES_JSON, SPARSE_TREATMENTS_JSON)

    assert glucose_df.height > 0
    carb_events = events_df.filter(pl.col("event_type") == "Carbohydrates")
    assert carb_events.height == 2


# --------------------------------------------------------------------------
# Multi-file upload: a Nightscout export is up to three sibling files.
# --------------------------------------------------------------------------


def _packed(path: Path) -> str:
    """The gzip+base64 envelope the browser's clientside compressor produces."""
    return "gzip:" + base64.b64encode(gzip.compress(path.read_bytes())).decode("ascii")


def test_kinds_are_told_apart_by_content() -> None:
    assert nightscout_json_kind(ENTRIES_JSON.read_bytes()) == "entries"
    assert nightscout_json_kind(TREATMENTS_JSON.read_bytes()) == "treatments"
    assert nightscout_json_kind((TESTDATA_DIR / "nightscout_profile.json").read_bytes()) == "profile"
    assert nightscout_json_kind((TESTDATA_DIR / "Clarity_Export_synthetic.csv").read_bytes()) is None


def test_profile_is_accepted_and_discarded() -> None:
    """profile.json is part of the same download, so users include it.

    It carries settings, not readings -- cgm-format fetches and discards it on
    the URL path, and an upload has to do the same rather than rejecting the
    whole selection over a file that was reasonable to attach.
    """
    files = [
        (name, (TESTDATA_DIR / name).read_bytes())
        for name in (
            "Nightscout_entries_synthetic.json",
            "Nightscout_treatments_synthetic.json",
            "nightscout_profile.json",
        )
    ]
    bundle = classify_nightscout_uploads(files)

    assert bundle is not None and bundle.is_usable
    assert bundle.entries.filename == "Nightscout_entries_synthetic.json"
    assert bundle.treatments.filename == "Nightscout_treatments_synthetic.json"
    assert bundle.discarded == ("nightscout_profile.json",)


def test_order_does_not_matter() -> None:
    """Users select files in whatever order the file picker gives them."""
    names = [
        "nightscout_profile.json",
        "Nightscout_treatments_synthetic.json",
        "Nightscout_entries_synthetic.json",
    ]
    bundle = classify_nightscout_uploads([(n, (TESTDATA_DIR / n).read_bytes()) for n in names])

    assert bundle is not None
    assert bundle.entries.filename == "Nightscout_entries_synthetic.json"
    assert bundle.treatments.filename == "Nightscout_treatments_synthetic.json"


def test_a_bundle_without_entries_is_not_usable() -> None:
    """treatments + profile is a real mistake to make, and needs its own message."""
    names = ["Nightscout_treatments_synthetic.json", "nightscout_profile.json"]
    bundle = classify_nightscout_uploads([(n, (TESTDATA_DIR / n).read_bytes()) for n in names])

    assert bundle is not None
    assert not bundle.is_usable
    with pytest.raises(ValueError):
        load_nightscout_uploads(bundle, TESTDATA_DIR)


def test_ordinary_csv_uploads_are_not_claimed_as_a_bundle() -> None:
    """Returning None is how the caller falls back to the single-file path."""
    csv_bytes = (TESTDATA_DIR / "Clarity_Export_synthetic.csv").read_bytes()
    assert classify_nightscout_uploads([("export.csv", csv_bytes)]) is None


def test_two_file_upload_produces_event_markers(tmp_path: Path) -> None:
    """The point of the whole feature: treatments become chart markers."""
    files = [
        (name, (TESTDATA_DIR / name).read_bytes())
        for name in ("Nightscout_entries_synthetic.json", "Nightscout_treatments_synthetic.json")
    ]
    bundle = classify_nightscout_uploads(files)
    glucose_df, events_df, save_path = load_nightscout_uploads(bundle, tmp_path)

    assert glucose_df.height > 0
    assert events_df.height > 0
    assert save_path.parent == tmp_path


def test_saved_bundle_keeps_its_events_when_reloaded(tmp_path: Path) -> None:
    """Later rounds reload from `uploaded_data_path`, so the merge must persist.

    Saving the entries file instead of the unified frame would drop every event
    marker after round one -- the meals and boluses would silently vanish
    mid-game, which is worse than never having shown them.
    """
    files = [
        (name, (TESTDATA_DIR / name).read_bytes())
        for name in ("Nightscout_entries_synthetic.json", "Nightscout_treatments_synthetic.json")
    ]
    bundle = classify_nightscout_uploads(files)
    glucose_df, events_df, save_path = load_nightscout_uploads(bundle, tmp_path)

    reloaded_glucose, reloaded_events = load_glucose_data(save_path)

    assert reloaded_glucose.height == glucose_df.height
    assert reloaded_events.height == events_df.height
    assert reloaded_events.height > 0


def test_transport_round_trips_several_files() -> None:
    """Dash sends a list for a multiple=True Upload; the compressor keeps the shape."""
    payloads = [_packed(ENTRIES_JSON), _packed(TREATMENTS_JSON)]
    names = ["entries.json", "treatments.json"]

    decoded = decode_upload_files(payloads, names)

    assert [name for name, _ in decoded] == names
    assert decoded[0][1] == ENTRIES_JSON.read_bytes()
    assert decoded[1][1] == TREATMENTS_JSON.read_bytes()


def test_transport_still_accepts_the_single_file_shape() -> None:
    """A single-file Upload hands over a bare string, not a one-element list."""
    decoded = decode_upload_files(_packed(ENTRIES_JSON), "entries.json")

    assert len(decoded) == 1
    assert decoded[0] == ("entries.json", ENTRIES_JSON.read_bytes())


def test_an_undecodable_member_does_not_sink_the_others() -> None:
    decoded = decode_upload_files([_packed(ENTRIES_JSON), "not-a-payload"], ["entries.json", "junk"])

    assert [name for name, _ in decoded] == ["entries.json"]


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("entries.json", "entries.json"),
        ("my export.csv", "my_export.csv"),
        ("../../etc/passwd", ".._.._etc_passwd.csv"),
        (None, "uploaded.csv"),
    ],
)
def test_saved_names_keep_json_as_json(given: object, expected: str) -> None:
    """Forcing .csv onto everything used to store entries.json as entries.json.csv."""
    assert safe_upload_filename(given) == expected
