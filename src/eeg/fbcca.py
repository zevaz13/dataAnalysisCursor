"""Filter-bank CCA (FBCCA) for SSVEP frequency detection.

Reference: Chen et al. (2015) — filter banks defined as in the original MATLAB template.
"""

from __future__ import annotations

import numpy as np
import mne
from scipy.signal import butter, filtfilt

# 5 sub-bands, all sharing the same upper edge (55 Hz), lower edge increases by 10 Hz
FILTER_BANKS: list[tuple[float, float]] = [
    (5, 55),
    (15, 55),
    (25, 55),
    (35, 55),
    (45, 55),
]


def _make_reference(n_samples: int, freq: float, sfreq: float, n_harmonics: int = 3) -> np.ndarray:
    """Build sin/cos reference matrix for a single target frequency.

    Returns shape (n_samples, 2 * n_harmonics):
        [sin(f), cos(f), sin(2f), cos(2f), sin(3f), cos(3f), ...]
    """
    t = np.arange(n_samples) / sfreq
    cols = []
    for h in range(1, n_harmonics + 1):
        cols.append(np.sin(h * 2 * np.pi * freq * t))
        cols.append(np.cos(h * 2 * np.pi * freq * t))
    return np.column_stack(cols)  # (n_samples, 2*n_harmonics)


def _max_cca(X: np.ndarray, Y: np.ndarray) -> float:
    """Return the largest canonical correlation between X (T×p) and Y (T×q).

    Uses QR decomposition followed by SVD for numerical stability.
    X is demeaned; Y (sin/cos) is already zero-mean.
    """
    Qx, _ = np.linalg.qr(X - X.mean(axis=0))
    Qy, _ = np.linalg.qr(Y)
    _, s, _ = np.linalg.svd(Qx.T @ Qy, full_matrices=False)
    return float(s[0])


def run_fbcca(
    epochs: mne.Epochs,
    freq: float,
    channels: list[str] | None = None,
    n_harmonics: int = 3,
    filter_order: int = 4,
) -> np.ndarray:
    """Compute the FBCCA score for each epoch.

    Parameters
    ----------
    epochs : mne.Epochs
        Preprocessed, artifact-cleaned epochs (trials × channels × samples).
    freq : float
        Target SSVEP frequency in Hz.
    channels : list[str] | None
        Channel names to use. None uses all EEG channels.
    n_harmonics : int
        Number of harmonics included in the CCA reference signal (default 3).
    filter_order : int
        Order of the Butterworth bandpass filters (default 4).

    Returns
    -------
    np.ndarray, shape (n_trials,)
        Weighted FBCCA score per epoch: sum_j( w_j * rho_j^2 )
        where rho_j is the max canonical correlation for filter bank j and
        w_j = j^(-1.25) + 0.25  (1-indexed, as in Chen 2015).
    """
    if channels is None:
        picks = mne.pick_types(epochs.info, eeg=True, exclude=[])
    else:
        picks = [epochs.ch_names.index(ch) for ch in channels]

    data = epochs.get_data(picks=picks)          # (n_trials, n_ch, n_samples)
    sfreq = epochs.info["sfreq"]
    n_trials, _, n_samples = data.shape

    ref = _make_reference(n_samples, freq, sfreq, n_harmonics)  # (n_samples, 2*n_harmonics)

    n_banks = len(FILTER_BANKS)
    # weights: w_i = i^(-1.25) + 0.25, i is 1-indexed (Chen 2015 eq. 7)
    weights = np.array([(i + 1) ** (-1.25) + 0.25 for i in range(n_banks)])

    corr_matrix = np.zeros((n_trials, n_banks))

    for j, (lf, hf) in enumerate(FILTER_BANKS):
        b, a = butter(filter_order, [lf, hf], btype="bandpass", fs=sfreq)
        fb_data = filtfilt(b, a, data, axis=-1)  # (n_trials, n_ch, n_samples)
        for i in range(n_trials):
            X = fb_data[i].T                     # (n_samples, n_ch)
            corr_matrix[i, j] = _max_cca(X, ref)

    # Weighted sum of squared canonical correlations across filter banks
    return (weights * corr_matrix ** 2).sum(axis=1)  # (n_trials,)
