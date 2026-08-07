"""Load blue-intensity SSVEP stimulus levels for the Metamers blue-test experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_blue_levels(path: Path) -> np.ndarray:
    """Load per-trial blue PWM intensity levels from a `*BlueTest.npz` file.

    Returns shape (20,): blue_levels[i] is the blue intensity used in the
    i-th stimulus trial, in the same order as the "stimulus" trials in the
    corresponding R01.mat run (EEG.data trials 4:23 / EpochsArray
    STIMULUS_EVENT_ID).
    """
    data = np.load(path, allow_pickle=True)
    return data["blueArray"].ravel()
