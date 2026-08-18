# Known issues

Open problems we have not decided to fix yet. Add a short repro and the Dash IDs involved so a later pass can judge whether the cost is worth it.

## Finish confirm: page-only button IDs in a shared callback

**Status:** resolved by page-local confirmation flows.

**Symptom:** Dash raises

```text
A nonexistent object was used in an Input of a Dash callback.
The id of this object is `finish-study-button-ending` and the property is `n_clicks`.
```

The listed layout was the prediction page (`finish-study-button` was present; `finish-study-button-ending` was not). The confirm overlay did not open and the game did not exit.

**Why:** `suppress_callback_exceptions=True` only lets you *register* callbacks for IDs that are not always in the tree. At fire time Dash still requires **every** `Input` and `State` of that callback to exist in the current layout. Same rule as the `/startup` vs `consent-*` pitfall in `AGENTS.md`.

The old confirm overlay lived in the **base layout**, while the two Finish triggers did not:

| ID | Where it is rendered |
|---|---|
| `finish-study-button` | `/prediction` only (`SubmitComponent`) |
| `finish-study-button-ending` | `/ending` only (`create_ending_layout`) |
| `time-slider` | `/prediction` only |

The shared callback took **both** Finish buttons as `Input`s:

```text
Input('finish-study-button', 'n_clicks')
Input('finish-study-button-ending', 'n_clicks')
Input('finish-keep-playing-button', 'n_clicks')
Input('finish-confirm-button', 'n_clicks')
```

Clicking Finish on `/prediction` therefore asked Dash for `finish-study-button-ending`, which was not in the tree. The mirror failure happened on `/ending` (`finish-study-button` missing).

There was a second cross-page dependency: the shared confirm handler read `State('time-slider', 'value')`, although that slider exists only on `/prediction`. The `/ending` path does not need the slider.

### Resolution

Each page now owns a complete confirmation flow with source-specific IDs:

- `/prediction`: `finish-confirm-*-prediction`; its confirm handler may read `time-slider`.
- `/ending`: `finish-confirm-*-ending`; its confirm handler never references `time-slider`.

The overlays are children of their page roots, outside the clipped action rows. There is no shared `finish-confirm-source` dispatch store and no duplicate overlay Output.

Language is deliberately separate from behavior. Page-local text callbacks consume only `interface-language` plus an inert confirmation-context store and output only title, message, and button text. They cannot open, close, confirm, or navigate.

**Regression tests:** `tests/test_callback_output_ids.py`, `tests/test_exit_saves_complete_round.py`, and `tests/test_ending_switch_format_row.py`.
