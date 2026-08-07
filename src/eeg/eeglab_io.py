"""Load EEG epochs from EEGLAB-format .mat files (BCI2000-independent SSVEP runs)."""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import scipy.io as sio

BASELINE_EVENT_ID = 1
STIMULUS_EVENT_ID = 2
N_BASELINE_TRIALS = 3


def load_eeglab_epochs(path: Path) -> mne.EpochsArray:
    """Load an EEGLAB R0*.mat run into an EpochsArray.

    The first N_BASELINE_TRIALS trials are tagged event id BASELINE_EVENT_ID
    ("baseline"); the remaining trials are tagged STIMULUS_EVENT_ID ("stimulus"),
    per the experiment protocol (plan.md M2).
    """
    mat = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    eeg = mat["EEG"]

    data = eeg.data.astype(np.float64)  # (n_channels, n_times, n_trials)
    data = np.moveaxis(data, -1, 0)  # -> (n_trials, n_channels, n_times), MNE convention

    ch_names = [ch.labels for ch in eeg.chanlocs]
    info = mne.create_info(ch_names, sfreq=float(eeg.srate), ch_types="eeg")

    n_trials = data.shape[0]
    codes = np.full(n_trials, STIMULUS_EVENT_ID)
    codes[:N_BASELINE_TRIALS] = BASELINE_EVENT_ID
    events = np.column_stack(
        [np.arange(n_trials), np.zeros(n_trials, dtype=int), codes]
    )
    event_id = {"baseline": BASELINE_EVENT_ID, "stimulus": STIMULUS_EVENT_ID}

    epochs = mne.EpochsArray(
        data, info, events=events, tmin=float(eeg.xmin), event_id=event_id
    )
    montage = mne.channels.make_standard_montage("standard_1020")
    epochs.set_montage(montage, on_missing="warn")

    return epochs
