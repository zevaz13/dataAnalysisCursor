"""Load EEG data from file."""

from pathlib import Path

import mne
import numpy as np

from eeg.bci2k_meta import decode_states, parse_channel_names

STIM_CHANNEL_NAME = "STI 014"


def load_bci2k(path: Path, stim_state: str = "DigitalInput1") -> mne.io.BaseRaw:
    """Load a BCI2000 .dat file with channel names, montage, and digital stim channel."""
    raw = mne.io.read_raw_bci2k(path, preload=True)

    channel_names = parse_channel_names(path)
    raw.rename_channels(dict(zip(raw.ch_names[: len(channel_names)], channel_names)))

    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="warn")

    states = decode_states(path)
    if stim_state not in states:
        raise ValueError(
            f"State {stim_state!r} not found. Available: {sorted(states)}"
        )

    stim_data = states[stim_state].astype(np.float64)[np.newaxis, :]
    stim_info = mne.create_info(
        ch_names=[STIM_CHANNEL_NAME],
        sfreq=raw.info["sfreq"],
        ch_types=["stim"],
    )
    stim_raw = mne.io.RawArray(stim_data, stim_info)
    raw.add_channels([stim_raw], force_update_info=True)

    return raw
