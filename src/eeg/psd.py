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


def band_power(
    freqs: np.ndarray,
    spectrum: np.ndarray,
    freq_range: float | tuple[float, float],
) -> np.ndarray:
    """Spectrum value at one frequency, or the mean over a frequency range.

    Frequency must be axis 0; any trailing shape works (e.g. (n_freqs, n_trials)
    from combine_channels, or (n_freqs, n_times, n_trials) to collapse the
    frequency axis of a time-frequency array while keeping time).

    Parameters
    ----------
    freqs : np.ndarray, shape (n_freqs,)
    spectrum : np.ndarray, shape (n_freqs, ...)
        E.g. output of combine_channels, or a single channel's slice of a
        time_frequency_decompose array.
    freq_range : float | tuple[float, float]
        A single frequency snaps to the nearest bin. A (low, high) tuple
        averages all bins in that inclusive range.

    Returns
    -------
    np.ndarray, shape matching spectrum with axis 0 removed
    """
    if isinstance(freq_range, tuple):
        lo, hi = freq_range
        mask = (freqs >= lo) & (freqs <= hi)
        if not mask.any():
            raise ValueError(f"no frequency bins found in range {freq_range}")
        return spectrum[mask].mean(axis=0)
    idx = np.argmin(np.abs(freqs - freq_range))
    return spectrum[idx]
