"""Plotting helpers for MNE Raw objects."""

from __future__ import annotations

import matplotlib.pyplot as plt
import mne
import numpy as np


def plot_montage(
    raw: mne.io.BaseRaw,
    show_names: bool = True,
) -> plt.Figure:
    """Plot channel locations from the montage.

    show_names=True labels each sensor with its channel name.
    show_names=False labels with channel indices instead.
    """
    return raw.plot_sensors(show_names=show_names)


def plot_channels(
    raw: mne.io.BaseRaw,
    channel_names: list[str],
    duration: float = 10.0,
) -> plt.Figure:
    """Plot one or more channels by name."""
    picks = [raw.ch_names.index(ch) for ch in channel_names]
    n_samples = int(duration * raw.info["sfreq"])
    data, times = raw[picks, :n_samples]

    fig, axes = plt.subplots(len(picks), 1, figsize=(12, 0.8 * len(picks) + 1), sharex=True)
    if len(picks) == 1:
        axes = [axes]

    for ax, ch_name, row in zip(axes, channel_names, data):
        ax.plot(times, row)
        ax.set_ylabel(ch_name, fontsize=8)
        ax.tick_params(labelsize=7)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Channel traces")
    fig.tight_layout()
    return fig


def plot_channels_psd(
    raw: mne.io.BaseRaw,
    channel_names: list[str],
    fmax: float = 100.0,
) -> plt.Figure:
    """Plot PSD for one or more channels by name."""
    return raw.compute_psd(picks=channel_names, fmax=fmax).plot()


def plot_ica_components(
    raw: mne.io.BaseRaw,
    ica: mne.preprocessing.ICA,
    component_indices: list[int],
    labels: dict | None = None,
) -> list[plt.Figure]:
    """Plot properties for each ICA component, optionally annotating ICLabel results.

    labels: dict returned by mne_icalabel.label_components, keys 'labels' and 'y_pred_proba'.
    """
    figures = []
    for idx in component_indices:
        figs = ica.plot_properties(raw, picks=[idx], show=False)
        if labels is not None:
            label = labels["labels"][idx]
            prob = labels["y_pred_proba"][idx].max()
            for fig in figs:
                fig.suptitle(f"IC {idx}  |  {label} ({prob:.0%})", fontsize=10)
        figures.extend(figs)
    # plt.show() with the inline backend displays-and-closes all pending figures,
    # preventing the post-cell flush_figures hook from rendering them a second time.
    # Skip on Agg (scripts) where show() would warn and figures stay open for savefig.
    if plt.get_backend().lower() != "agg":
        plt.show()
    return figures


def plot_before_after(
    raw_before: mne.io.BaseRaw,
    raw_after: mne.io.BaseRaw,
    channel_names: list[str],
    duration: float = 10.0,
) -> plt.Figure:
    """Two-column time-domain comparison: before ICA (left) vs after ICA (right)."""
    n_ch = len(channel_names)
    picks = [raw_before.ch_names.index(ch) for ch in channel_names]
    n_samples = int(duration * raw_before.info["sfreq"])

    data_before, times = raw_before[picks, :n_samples]
    data_after, _ = raw_after[picks, :n_samples]

    fig, axes = plt.subplots(n_ch, 2, figsize=(14, 0.9 * n_ch + 1), sharex=True)
    if n_ch == 1:
        axes = axes[None, :]  # ensure 2-D

    for row, (ch_name, before, after) in enumerate(zip(channel_names, data_before, data_after)):
        axes[row, 0].plot(times, before)
        axes[row, 1].plot(times, after)
        axes[row, 0].set_ylabel(ch_name, fontsize=8)
        for col in range(2):
            axes[row, col].tick_params(labelsize=7)

    axes[0, 0].set_title("Before ICA", fontsize=9)
    axes[0, 1].set_title("After ICA", fontsize=9)
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


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


def plot_baseline_boxplot(baseline_scores: np.ndarray) -> plt.Figure:
    """Boxplot of the 4 baseline FBCCA epochs with individual points annotated.

    Parameters
    ----------
    baseline_scores : np.ndarray, shape (4,)
        FBCCA scores for epochs 1, 2, 103, 104 (in that order).
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    bp = ax.boxplot(
        baseline_scores,
        widths=0.4,
        patch_artist=True,
        boxprops=dict(facecolor="lightsteelblue", alpha=0.6),
        medianprops=dict(color="navy", linewidth=2),
        whiskerprops=dict(linestyle="--"),
    )
    labels = ["B1", "B2", "B3", "B4"]
    x_jitter = np.array([1] * 4) + np.random.default_rng(0).uniform(-0.05, 0.05, 4)
    for i, (x, y) in enumerate(zip(x_jitter, baseline_scores)):
        ax.scatter(x, y, s=60, color="steelblue", zorder=4)
        ax.annotate(labels[i], (x, y), xytext=(6, 2), textcoords="offset points", fontsize=8)
    ax.set_xticks([1])
    ax.set_xticklabels(["Baseline epochs"])
    ax.set_ylabel("FBCCA score")
    ax.set_title("Baseline FBCCA scores (n=4)")
    fig.tight_layout()
    return fig


def plot_stimulus_heatmap(
    matrix: np.ndarray,
    red_array: np.ndarray,
    green_array: np.ndarray,
    title: str = "FBCCA score — red × green intensity grid",
) -> plt.Figure:
    """Heatmap of FBCCA scores on the 10×10 red × green PWM intensity grid.

    Parameters
    ----------
    matrix : np.ndarray, shape (10, 10)
        Output of build_stimulus_matrix. Rows = green intensity (low→high from bottom),
        columns = red intensity (low→high from left).
    red_array : np.ndarray, shape (10,)
        Red channel PWM intensity tick labels (D/A units).
    green_array : np.ndarray, shape (10,)
        Green channel PWM intensity tick labels (D/A units).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(im, ax=ax, label="FBCCA score")

    red_labels = [f"{v:.0f}" for v in red_array]
    green_labels = [f"{v:.0f}" for v in green_array]

    ax.set_xticks(np.arange(10))
    ax.set_xticklabels(red_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(10))
    ax.set_yticklabels(green_labels, fontsize=8)
    ax.set_xlabel("Red intensity (D/A units)")
    ax.set_ylabel("Green intensity (D/A units)")
    ax.set_title(title)

    # Annotate each cell with its score
    vmin, vmax = matrix.min(), matrix.max()
    midpoint = (vmin + vmax) / 2
    for row in range(10):
        for col in range(10):
            val = matrix[row, col]
            color = "white" if val < midpoint else "black"
            ax.text(col, row, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color=color)

    fig.tight_layout()
    return fig


def plot_fbcca_stream(
    scores: np.ndarray,
    freq: float | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Plot the FBCCA score stream across epochs.

    Parameters
    ----------
    scores : np.ndarray, shape (n_epochs,)
        Output of run_fbcca — one weighted score per epoch.
    freq : float | None
        Target frequency in Hz; included in the title if given.
    title : str | None
        Override the auto-generated title.
    """
    n = len(scores)
    x = np.arange(1, n + 1)

    if title is None:
        title = f"FBCCA score stream — {n} epochs" + (f" (target {freq} Hz)" if freq is not None else "")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, scores, linewidth=1, color="steelblue")
    ax.scatter(x, scores, s=18, color="steelblue", zorder=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("FBCCA score")
    ax.set_title(title)
    ax.set_xlim(0.5, n + 0.5)
    fig.tight_layout()
    return fig
