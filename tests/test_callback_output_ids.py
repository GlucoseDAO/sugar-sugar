"""No two callbacks may resolve to the same output id.

Dash builds the ``allow_duplicate`` suffix from the callback's INPUTS alone::

    def _hash_inputs():
        return hashlib.sha256(".".join(str(x) for x in inputs).encode()).hexdigest()

    if x.allow_duplicate:
        _id += f"@{hashed_inputs}"

(`dash/_utils.py`, ``create_callback_id``). So ``allow_duplicate=True`` only
separates two writers of one property when their input lists differ. Two
callbacks writing ``events-df`` off the same ``Input('url', 'pathname')`` hashed
to the identical id, and the renderer aborted the page with "Duplicate callback
outputs" -- ``uv run chart`` served a blank ``/prediction`` (debug shows the
error; without debug the page rendered but one of the two writers was moot, and
which one won was never defined).

`allow_duplicate` is not a licence to add another writer: when the trigger is the
same, fold the work into the existing callback instead (see
`initialize_data_on_url_change`, which absorbed `compact_events_store`).
"""

from __future__ import annotations

import collections

from sugar_sugar.app import app


def _output_ids(callback_key: str) -> list[str]:
    """Split a callback_map key into its individual ``id.property[@hash]`` outputs."""
    if callback_key.startswith(".."):
        return [part for part in callback_key.strip(".").split("...") if part]
    return [callback_key]


def test_no_two_callbacks_share_an_output_id() -> None:
    counts: collections.Counter[str] = collections.Counter()
    for key in app.callback_map:
        counts.update(_output_ids(key))

    duplicates = {out: n for out, n in counts.items() if n > 1}
    assert not duplicates, (
        "Two callbacks resolve to the same output id, which blanks the page with "
        f"'Duplicate callback outputs': {duplicates}. allow_duplicate hashes the "
        "inputs only, so same-trigger writers collide -- merge them instead."
    )


def _callback_input_ids(callback_key: str) -> set[str]:
    return {
        str(dep["id"])
        for dep in app.callback_map[callback_key]["inputs"]
        if isinstance(dep.get("id"), str)
    }


def _callback_state_ids(callback_key: str) -> set[str]:
    return {
        str(dep["id"])
        for dep in app.callback_map[callback_key]["state"]
        if isinstance(dep.get("id"), str)
    }


def test_finish_confirm_callbacks_do_not_mix_page_buttons() -> None:
    """A callback that lists both finish buttons never fires on either page.

    ``finish-study-button`` lives only on ``/prediction`` and
    ``finish-study-button-ending`` only on ``/ending``. Dash skips the
    callback unless every Input is mounted, so mixing them made the red
    close control do nothing.
    """
    mixed = [
        key
        for key in app.callback_map
        if _callback_input_ids(key)
        >= {"finish-study-button", "finish-study-button-ending"}
    ]
    assert mixed == [], (
        "finish-study-button and finish-study-button-ending must not share a "
        f"callback (the confirm card would never open): {mixed}"
    )


def test_finish_confirm_flows_are_page_local() -> None:
    prediction_toggle = next(
        key for key in app.callback_map
        if "finish-confirm-overlay-prediction.style" in key
    )
    ending_toggle = next(
        key for key in app.callback_map
        if "finish-confirm-overlay-ending.style" in key
    )
    prediction_confirm = next(
        key for key in app.callback_map
        if _callback_input_ids(key) == {"finish-confirm-button-prediction"}
    )
    ending_confirm = next(
        key for key in app.callback_map
        if "finish-confirm-button-ending" in _callback_input_ids(key)
        and "url.pathname" in key
    )

    assert _callback_input_ids(prediction_toggle) == {
        "finish-study-button",
        "finish-keep-playing-button-prediction",
    }
    assert _callback_input_ids(ending_toggle) == {
        "finish-study-button-ending",
        "finish-keep-playing-button-ending",
    }
    assert "time-slider" in _callback_state_ids(prediction_confirm)
    assert "time-slider" not in _callback_state_ids(ending_confirm)


def test_finish_language_callbacks_only_update_text() -> None:
    for source in ("prediction", "ending"):
        callback_key = next(
            key for key in app.callback_map
            if f"finish-confirm-title-{source}.children" in key
        )
        assert _callback_input_ids(callback_key) == {
            "interface-language",
            f"finish-confirm-context-{source}",
        }
        assert set(_output_ids(callback_key)) == {
            f"finish-confirm-title-{source}.children",
            f"finish-confirm-message-{source}.children",
            f"finish-confirm-button-{source}.children",
            f"finish-keep-playing-button-{source}.children",
        }


def test_events_df_has_a_single_pathname_writer() -> None:
    """The specific collision that shipped, kept honest by name."""
    writers = [
        key
        for key in app.callback_map
        if any(out.startswith("events-df.data") for out in _output_ids(key))
        and [
            f"{dep['id']}.{dep['property']}"
            for dep in app.callback_map[key]["inputs"]
        ] == ["url.pathname"]
    ]
    assert len(writers) == 1, (
        f"expected exactly one url.pathname-triggered writer of events-df, got {writers}"
    )
