"""EEG preprocessing: filtering, re-referencing, ICA, and ICLabel."""

from __future__ import annotations

import mne
import mne_icalabel


def bandpass_filter(
    raw: mne.io.BaseRaw,
    l_freq: float = 1.0,
    h_freq: float = 55.0,
    order: int = 4,
) -> mne.io.BaseRaw:
    """Bandpass filter with a Butterworth IIR filter, in-place."""
    raw.filter(
        l_freq,
        h_freq,
        method="iir",
        iir_params={"order": order, "ftype": "butter"},
    )
    return raw


def rereference_average(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Re-reference EEG to the average of all EEG channels, in-place."""
    raw.set_eeg_reference("average", projection=False)
    return raw


def run_ica(
    raw: mne.io.BaseRaw,
    n_components: int = 24,
    method: str = "infomax",
) -> mne.preprocessing.ICA:
    """Fit ICA on the raw data and return the fitted ICA object."""
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method=method,
        fit_params={"extended": True},
        random_state=42,
    )
    ica.fit(raw, picks="eeg")
    return ica


def label_components(
    raw: mne.io.BaseRaw,
    ica: mne.preprocessing.ICA,
) -> dict:
    """Run ICLabel on the fitted ICA and return the labels dict."""
    return mne_icalabel.label_components(raw, ica, method="iclabel")
