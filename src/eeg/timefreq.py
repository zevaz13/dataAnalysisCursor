"""Time-frequency decomposition via Morlet wavelet convolution, and baseline normalization.

Port of templateCode/timeFreq (MATLAB, Cohen-style wavelet convolution), vectorized with
scipy.signal.fftconvolve instead of the original manual FFT/trial-concatenation trick.
Numerically equivalent up to the wavelet's overall normalization constant, which cancels
out in every baseline-normalized output (percent change, dB, z-score) since it is a
per-frequency scalar shared by the baseline and the activity it's compared against.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve


def _morlet_wavelet(freq: float, n_cycles: float, sfreq: float) -> np.ndarray:
    """Build one complex Morlet wavelet, normalized to unit energy.

    wavtime spans -2 to 2 seconds (matches the MATLAB template), so half_wave
    edge effects are confined to `wavtime` seconds at each trial boundary.
    """
    n_wave = int(4 * sfreq) + 1
    wavtime = np.linspace(-2, 2, n_wave)
    s = n_cycles / (2 * np.pi * freq)
    wavelet = np.exp(2j * np.pi * freq * wavtime) * np.exp(-(wavtime**2) / (2 * s**2))
    return wavelet / np.sqrt(np.sum(np.abs(wavelet) ** 2))


def time_frequency_decompose(
    data: np.ndarray,
    sfreq: float,
    freq_range: tuple[float, float] = (2, 30),
    n_freqs: int = 50,
    n_cycles_range: tuple[float, float] = (4, 10),
    include_freq: float | None = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Morlet wavelet convolution, vectorized over trials and channels.

    Parameters
    ----------
    data : np.ndarray, shape (n_trials, n_channels, n_times)
        Matches mne's `epochs.get_data()` layout.
    sfreq : float
        Sampling rate in Hz.
    freq_range : tuple[float, float]
        (min_freq, max_freq) in Hz, linearly spaced into n_freqs bins.
    n_freqs : int
        Number of frequency bins.
    n_cycles_range : tuple[float, float]
        (min_cycles, max_cycles), log-spaced across frequencies (fewer cycles
        at low frequencies for time resolution, more cycles at high
        frequencies for frequency resolution).
    include_freq : float | None
        If set, the nearest frequency bin is snapped to exactly this value
        (e.g. a known SSVEP stimulus frequency), regardless of freq_range/n_freqs.

    Returns
    -------
    tf : np.ndarray, shape (n_channels, n_freqs, n_times, n_trials)
        Power (|analytic signal|^2) at each channel, frequency, time, and trial.
    frex : np.ndarray, shape (n_freqs,)
        Center frequency of each bin, in Hz.
    """
    frex = np.linspace(freq_range[0], freq_range[1], n_freqs)
    if include_freq is not None:
        if not freq_range[0] <= include_freq <= freq_range[1]:
            raise ValueError(
                f"include_freq={include_freq} is outside freq_range={freq_range}"
            )
        frex[np.argmin(np.abs(frex - include_freq))] = include_freq

    cycles = np.logspace(
        np.log10(n_cycles_range[0]), np.log10(n_cycles_range[1]), n_freqs
    )

    n_trials, n_channels, n_times = data.shape
    tf = np.empty((n_channels, n_freqs, n_times, n_trials))

    for fi, (freq, n_cyc) in enumerate(zip(frex, cycles)):
        wavelet = _morlet_wavelet(freq, n_cyc, sfreq)
        analytic = fftconvolve(data, wavelet[None, None, :], mode="same", axes=-1)
        power = np.abs(analytic) ** 2  # (n_trials, n_channels, n_times)
        tf[:, fi, :, :] = np.moveaxis(power, 0, -1)

    return tf, frex


def compute_baseline_stats(
    tf: np.ndarray, baseline_trial_idx: np.ndarray | slice
) -> tuple[np.ndarray, np.ndarray]:
    """Baseline mean and std per (channel, freq), pooled over time and baseline trials.

    Parameters
    ----------
    tf : np.ndarray, shape (n_channels, n_freqs, n_times, n_trials)
    baseline_trial_idx : indices or slice selecting the baseline trials on the
        last axis of `tf`.

    Returns
    -------
    baseline_mean, baseline_std : np.ndarray, shape (n_channels, n_freqs)
    """
    baseline = tf[..., baseline_trial_idx]  # (n_channels, n_freqs, n_times, n_baseline_trials)
    baseline_mean = baseline.mean(axis=(-2, -1))
    baseline_std = baseline.std(axis=(-2, -1))
    return baseline_mean, baseline_std


def baseline_normalize(
    tf: np.ndarray,
    baseline_mean: np.ndarray,
    baseline_std: np.ndarray,
    method: str = "db",
) -> np.ndarray:
    """Normalize tf against a per-(channel, freq) baseline.

    Parameters
    ----------
    tf : np.ndarray, shape (n_channels, n_freqs, n_times, n_trials)
    baseline_mean, baseline_std : np.ndarray, shape (n_channels, n_freqs)
        Output of compute_baseline_stats.
    method : {"percent", "db", "zscore"}
        percent : (tf - mean) / mean * 100
        db      : 10 * log10(tf / mean)
        zscore  : (tf - mean) / std
    """
    mean = baseline_mean[:, :, None, None]
    if method == "percent":
        return (tf - mean) / mean * 100
    if method == "db":
        return 10 * np.log10(tf / mean)
    if method == "zscore":
        std = baseline_std[:, :, None, None]
        return (tf - mean) / std
    raise ValueError(f"Unknown method {method!r}, expected 'percent', 'db', or 'zscore'")
