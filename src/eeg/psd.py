"""Power spectral density via Welch's method, per channel and per trial."""

from __future__ import annotations

import numpy as np
from scipy.signal import welch


def compute_psd(
    data: np.ndarray, sfreq: float, nperseg: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD for every channel and trial independently.

    Parameters
    ----------
    data : np.ndarray, shape (n_trials, n_channels, n_times)
        Matches mne's `epochs.get_data()` layout.
    sfreq : float
        Sampling rate in Hz.
    nperseg : int | None
        Welch window length in samples. Defaults to 2 seconds (`2 * sfreq`).

    Returns
    -------
    freqs : np.ndarray, shape (n_freqs,)
    psd : np.ndarray, shape (n_channels, n_freqs, n_trials)
    """
    if nperseg is None:
        nperseg = int(2 * sfreq)

    freqs, psd = welch(data, fs=sfreq, nperseg=nperseg, axis=-1)  # (n_trials, n_channels, n_freqs)
    return freqs, np.moveaxis(psd, 0, -1)


def combine_channels(
    psd: np.ndarray,
    channel_names: list[str],
    channels: list[str] | None = None,
) -> np.ndarray:
    """Mean spectrum across a set of channels.

    Parameters
    ----------
    psd : np.ndarray, shape (n_channels, n_freqs, n_trials)
        Output of compute_psd.
    channel_names : list[str]
        Full channel name list, in the same order as psd's channel axis.
    channels : list[str] | None
        Channels to average. Defaults to ["Oz"].

    Returns
    -------
    np.ndarray, shape (n_freqs, n_trials)
    """
    if channels is None:
        channels = ["Oz"]
    idx = [channel_names.index(ch) for ch in channels]
    return psd[idx].mean(axis=0)
