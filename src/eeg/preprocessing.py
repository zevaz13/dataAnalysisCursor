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


def remove_artifacts(
    raw: mne.io.BaseRaw,
    ica: mne.preprocessing.ICA,
    labels: dict,
    eye_threshold: float = 0.7,
    muscle_threshold: float = 0.5,
) -> tuple[mne.io.BaseRaw, list[int]]:
    """Exclude artifact ICs and return a cleaned copy of raw plus the excluded indices."""
    exclude = [
        i
        for i, (label, proba) in enumerate(zip(labels["labels"], labels["y_pred_proba"]))
        if (label == "eye blink" and proba.max() > eye_threshold)
        or (label == "muscle artifact" and proba.max() > muscle_threshold)
    ]
    ica.exclude = exclude
    return ica.apply(raw.copy()), exclude


def make_epochs(
    raw: mne.io.BaseRaw,
    tmin: float = 0.0,
    tmax: float = 3.0,
    event_id: int = 1,
    baseline: tuple | None = None,
) -> mne.Epochs:
    """Create epochs locked to rising edges of STI 014, default 0–3 s."""
    events = mne.find_events(raw, stim_channel="STI 014", shortest_event=1)
    return mne.Epochs(
        raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
        baseline=baseline, preload=True,
    )
