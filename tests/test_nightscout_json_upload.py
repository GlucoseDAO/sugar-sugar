"""Nightscout entries.json is an uploadable CGM export.

Nightscout's own CSV export is not usable -- ``/api/v1/entries.csv`` is
headerless with five hardcoded columns and there is no treatments CSV at all --
so the JSON the site serves at ``/api/v1/entries.json`` is what the upload hint
asks for and what these tests pin down.

One test in here is expected to FAIL until cgm-format ships the fix reported in
``FEEDBACK.md`` issue 1. It is deliberately not marked ``xfail``: it is a live
tripwire that turns green on its own the day the library is bumped, and an
``xfail`` would then quietly flip to ``XPASS`` instead of telling anyone.
"""

from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path

import polars as pl
import pytest

from sugar_sugar.data import (
    decode_upload_bytes,
    is_nightscout_entries_json,
    load_glucose_data,
    load_nightscout_json_data,
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

    Green today: this fixture's `carbs` column has a value inside polars'
    100-row inference window, so it dodges the library bug. It exists to prove
    the wiring itself is right, which is what makes the next test's failure
    attributable to cgm-format rather than to this repo.
    """
    glucose_df, events_df = load_nightscout_json_data(ENTRIES_JSON, TREATMENTS_JSON)

    assert glucose_df.height > 0
    assert events_df.height > 0
    assert set(events_df["event_type"].unique()) <= {"Insulin", "Carbohydrates", "Exercise"}


def test_treatments_with_a_long_null_run_are_loaded() -> None:
    """EXPECTED RED until cgm-format > 0.12.0 -- see FEEDBACK.md issue 1.

    A closed-loop user whose pump writes a Temp Basal every few minutes but who
    logs carbs rarely has `carbs` null for far more than the 100 rows polars
    samples to infer dtypes. The column infers as `Null` and appending the first
    real value raises ComputeError, so the whole import dies.

    Do not xfail this and do not work around it in ``data.py``: the fix belongs
    upstream, and this test should start passing on its own once the floor in
    ``pyproject.toml`` moves.
    """
    glucose_df, events_df = load_nightscout_json_data(ENTRIES_JSON, SPARSE_TREATMENTS_JSON)

    assert glucose_df.height > 0
    carb_events = events_df.filter(pl.col("event_type") == "Carbohydrates")
    assert carb_events.height == 2
