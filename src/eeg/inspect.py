"""Print metadata and extract events from MNE Raw objects."""

import mne
import numpy as np


def _stim_channel(raw: mne.io.BaseRaw) -> str | None:
    picks = mne.pick_types(raw.info, stim=True, exclude=[])
    if len(picks) == 0:
        return None
    return raw.ch_names[picks[0]]


def _rising_edge_count(raw: mne.io.BaseRaw, stim_ch: str) -> int:
    pick = raw.ch_names.index(stim_ch)
    data = raw[pick, :][0][0]
    return int(np.sum((data[1:] == 1) & (data[:-1] == 0)))


def print_summary(raw: mne.io.BaseRaw) -> None:
    """Print channel info, sampling rate, montage, and events."""
    montage = raw.get_montage()
    stim_ch = _stim_channel(raw)
    eeg_chs = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, exclude=[])]

    print(f"EEG channels ({len(eeg_chs)}): {eeg_chs}")
    print(f"All channels ({raw.info['nchan']}): {raw.ch_names}")
    print(f"Sampling rate: {raw.info['sfreq']} Hz")
    print(f"Duration: {raw.times[-1]:.2f} s")
    print(f"Montage: {montage if montage else 'none'}")
    print(f"Stimulus channel: {stim_ch if stim_ch else 'none'} (from DigitalInput1)")

    if stim_ch:
        rising_edges = _rising_edge_count(raw, stim_ch)
        events = mne.find_events(raw, stim_channel=stim_ch, shortest_event=1)
        print(f"DigitalInput1 rising edges: {rising_edges}")
        print(f"Events (find_events): {len(events)}")
        if len(events) > 0:
            print(f"Event counts: {mne.count_events(events)}")
    else:
        print("Events: none (no stimulus channel)")
