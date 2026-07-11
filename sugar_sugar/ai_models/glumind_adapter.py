from __future__ import annotations

from pathlib import Path

import torch

from .base import BaseGlucoseModel
from .normalization import normalize_glucose, denormalize_glucose

CHECKPOINT_PATH = Path(__file__).parent / "checkpoints" / "best_model.pt"

# GluMind was trained with 80 steps of history (see tuning_meta.json).
# The game only provides 24. Missing steps are padded by repeating the
# earliest known value. This is a known approximation, not a perfect
# solution -- see PR discussion / mentor sign-off for context.
REQUIRED_INPUT_STEPS = 80


class GluMindAdapter(BaseGlucoseModel):
    """Real GluMind model adapter: loads the trained checkpoint once,
    pads short histories to the required window, normalizes, runs
    inference, and denormalizes the output.
    """

    def __init__(self) -> None:
        from .glumind_arch.glumind_model import GluMindModel

        self._device = torch.device("cpu")
        self._model = GluMindModel(
            n_time_steps=REQUIRED_INPUT_STEPS,
            n_features=3,
            d_model=32,
            n_heads=4,
            ff_units=128,
            n_blocks=3,
            prediction_horizon=12,
            dropout=0.1,
        )
        state = torch.load(CHECKPOINT_PATH, map_location=self._device, weights_only=False)
        if any(k.startswith("_orig_mod.") for k in state):
            state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
        self._model.load_state_dict(state)
        self._model.eval()

    def predict(self, history: list[float], prediction_steps: int) -> list[float]:
        if not history:
            return [100.0] * prediction_steps

        padded_history = list(history)
        if len(padded_history) < REQUIRED_INPUT_STEPS:
            pad_value = padded_history[0]
            pad_count = REQUIRED_INPUT_STEPS - len(padded_history)
            padded_history = [pad_value] * pad_count + padded_history
        else:
            padded_history = padded_history[-REQUIRED_INPUT_STEPS:]

        glucose_norm = [normalize_glucose(v) for v in padded_history]
        hr_norm = [0.0] * REQUIRED_INPUT_STEPS
        steps_norm = [0.0] * REQUIRED_INPUT_STEPS

        input_tensor = torch.tensor(
            [list(triple) for triple in zip(glucose_norm, hr_norm, steps_norm)],
            dtype=torch.float32,
        ).unsqueeze(0)

        with torch.no_grad():
            output = self._model(input_tensor)

        predictions_norm = output.squeeze(0).tolist()
        predictions = [denormalize_glucose(v) for v in predictions_norm]

        if len(predictions) >= prediction_steps:
            return [round(v, 1) for v in predictions[:prediction_steps]]
        last = predictions[-1] if predictions else history[-1]
        padded_predictions = predictions + [last] * (prediction_steps - len(predictions))
        return [round(v, 1) for v in padded_predictions]