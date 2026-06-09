"""Plotting helpers for MNE Raw objects."""

import matplotlib.pyplot as plt
import mne


def plot_raw_traces(
    raw: mne.io.BaseRaw, duration: float = 10.0, n_channels: int = 20
) -> plt.Figure:
    """Plot multichannel raw EEG traces."""
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    picks = eeg_picks[: min(n_channels, len(eeg_picks))]
    n_samples = int(duration * raw.info["sfreq"])
    data, times = raw[picks, :n_samples]

    fig, axes = plt.subplots(len(picks), 1, figsize=(12, 0.8 * len(picks)), sharex=True)
    if len(picks) == 1:
        axes = [axes]

    ch_names = [raw.ch_names[p] for p in picks]
    for ax, ch_name, row in zip(axes, ch_names, data):
        ax.plot(times, row)
        ax.set_ylabel(ch_name, fontsize=8)
        ax.tick_params(labelsize=7)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Raw EEG traces")
    fig.tight_layout()
    return fig


def plot_stim_channel(
    raw: mne.io.BaseRaw, duration: float = 60.0
) -> plt.Figure | None:
    """Plot the stimulus channel time series."""
    picks = mne.pick_types(raw.info, stim=True, exclude=[])
    if len(picks) == 0:
        print("No stimulus channel to plot.")
        return None

    ch_name = raw.ch_names[picks[0]]
    n_samples = int(duration * raw.info["sfreq"])
    data, times = raw[picks[0], :n_samples]

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(times, data[0])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Stimulus channel: {ch_name}")
    fig.tight_layout()
    return fig


def plot_psd(raw: mne.io.BaseRaw) -> plt.Figure:
    """Plot power spectral density."""
    return raw.compute_psd().plot()
