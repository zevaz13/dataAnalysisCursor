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


def plot_time_frequency(
    tf: np.ndarray,
    frex: np.ndarray,
    times: np.ndarray,
    channel_names: list[str],
    channel: str,
    trial: int | None = None,
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Spectrogram (freq x time) for one channel.

    Parameters
    ----------
    tf : np.ndarray, shape (n_channels, n_freqs, n_times, n_trials)
        Output of eeg.timefreq.time_frequency_decompose (optionally baseline-normalized).
    frex : np.ndarray, shape (n_freqs,)
    times : np.ndarray, shape (n_times,)
        Time vector in seconds.
    channel_names : list[str]
        Full channel name list, in the same order as tf's channel axis.
    channel : str
        Channel to plot.
    trial : int | None
        Trial index to plot. None averages across all trials.
    cmap : str
        Colormap name.
    vmin, vmax : float | None
        Color axis limits. Default: symmetric around 0 (vmin=-vmax=-|data|.max()),
        matching the diverging colormap's default. Pass either or both to override.
    """
    ch_idx = channel_names.index(channel)
    data = tf[ch_idx]  # (n_freqs, n_times, n_trials)
    plot_data = data.mean(axis=-1) if trial is None else data[:, :, trial]

    if title is None:
        trial_label = "mean across trials" if trial is None else f"trial {trial}"
        title = f"Time-frequency — {channel} ({trial_label})"

    if vmin is None or vmax is None:
        auto = np.abs(plot_data).max()
        vmin = -auto if vmin is None else vmin
        vmax = auto if vmax is None else vmax

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(times, frex, plot_data, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Power")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_tf_grid(
    tf: np.ndarray,
    frex: np.ndarray,
    times: np.ndarray,
    channel_names: list[str],
    channel: str,
    trial_idx: np.ndarray | None = None,
    symmetric: bool = False,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    n_cols: int = 5,
    title: str | None = None,
) -> plt.Figure:
    """Grid of per-trial time-frequency spectrograms for one channel.

    Parameters
    ----------
    tf : np.ndarray, shape (n_channels, n_freqs, n_times, n_trials)
        Output of eeg.timefreq.time_frequency_decompose (raw or baseline-normalized).
    frex, times : np.ndarray
    channel_names : list[str]
    channel : str
    trial_idx : np.ndarray | None
        Trial indices to include (e.g. stimulus-only). None includes every trial.
    symmetric : bool
        True for baseline-normalized data: diverging colormap (default "RdBu_r"),
        auto vmin/vmax symmetric around 0. False (default) for raw power:
        sequential colormap (default "viridis"), auto vmin=0.
    cmap, vmin, vmax : color axis controls
        All three override the `symmetric`-based defaults; pass any subset.
    n_cols : int
    title : str | None
    """
    ch_idx = channel_names.index(channel)
    data = tf[ch_idx]  # (n_freqs, n_times, n_trials)
    if trial_idx is not None:
        data = data[:, :, trial_idx]
    n_trials = data.shape[-1]
    n_rows = int(np.ceil(n_trials / n_cols))
    trial_labels = trial_idx if trial_idx is not None else np.arange(n_trials)

    if cmap is None:
        cmap = "RdBu_r" if symmetric else "viridis"
    if vmin is None or vmax is None:
        if symmetric:
            auto = np.abs(data).max()
            vmin = -auto if vmin is None else vmin
            vmax = auto if vmax is None else vmax
        else:
            vmin = 0.0 if vmin is None else vmin
            vmax = data.max() if vmax is None else vmax

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.8 * n_cols, 2.2 * n_rows), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes).ravel()

    im = None
    for i in range(n_trials):
        ax = axes[i]
        im = ax.pcolormesh(
            times, frex, data[:, :, i], cmap=cmap, vmin=vmin, vmax=vmax, shading="auto"
        )
        ax.set_title(f"trial {trial_labels[i]}", fontsize=8)
        ax.tick_params(labelsize=6)

    for ax in axes[n_trials:]:
        ax.axis("off")

    fig.suptitle(title or f"Time-frequency per trial — {channel}")
    fig.supxlabel("Time (s)")
    fig.supylabel("Frequency (Hz)")
    fig.colorbar(im, ax=axes[:n_trials].tolist(), shrink=0.6, label="Power")
    return fig


def plot_time_course_grid(
    time_courses: np.ndarray,
    times: np.ndarray,
    channel: str,
    freq_label: str,
    trial_idx: np.ndarray | None = None,
    n_cols: int = 5,
) -> plt.Figure:
    """Grid of per-trial time courses (e.g. frequency-collapsed TF power), one subplot per trial.

    Parameters
    ----------
    time_courses : np.ndarray, shape (n_times, n_trials)
        E.g. output of eeg.psd.band_power applied to a channel's TF slice.
    times : np.ndarray, shape (n_times,)
    channel : str
        Used only in the title.
    freq_label : str
        Used only in the title, e.g. "10 Hz" or "9-11 Hz".
    trial_idx : np.ndarray | None
        Trial indices corresponding to time_courses' trial axis, used as subplot
        titles. None labels them 0..n_trials-1.
    n_cols : int
    """
    n_trials = time_courses.shape[-1]
    n_rows = int(np.ceil(n_trials / n_cols))
    trial_labels = trial_idx if trial_idx is not None else np.arange(n_trials)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.6 * n_cols, 1.8 * n_rows), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes).ravel()

    for i in range(n_trials):
        ax = axes[i]
        ax.plot(times, time_courses[:, i], linewidth=1)
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.set_title(f"trial {trial_labels[i]}", fontsize=8)
        ax.tick_params(labelsize=6)

    for ax in axes[n_trials:]:
        ax.axis("off")

    fig.suptitle(f"{channel} — {freq_label} vs. time")
    fig.supxlabel("Time (s)")
    fig.supylabel("Power")
    fig.tight_layout()
    return fig


def plot_time_course_overlay(
    time_courses: np.ndarray,
    times: np.ndarray,
    channel: str,
    freq_label: str,
    trial_idx: np.ndarray | None = None,
    cmap: str = "viridis",
) -> plt.Figure:
    """All trials' time courses overlaid on one axes, colored by trial index.

    Parameters
    ----------
    time_courses : np.ndarray, shape (n_times, n_trials)
        E.g. output of eeg.psd.band_power applied to a channel's TF slice.
    times : np.ndarray, shape (n_times,)
    channel : str
        Used only in the title.
    freq_label : str
        Used only in the title, e.g. "10 Hz" or "9-11 Hz".
    trial_idx : np.ndarray | None
        Trial indices corresponding to time_courses' trial axis, used for the
        color scale and colorbar. None uses 0..n_trials-1.
    cmap : str
        Colormap used for the trial-index color gradient.
    """
    n_trials = time_courses.shape[-1]
    trial_labels = trial_idx if trial_idx is not None else np.arange(n_trials)
    colormap = plt.get_cmap(cmap)
    norm = plt.Normalize(vmin=trial_labels.min(), vmax=trial_labels.max())

    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(n_trials):
        ax.plot(times, time_courses[:, i], color=colormap(norm(trial_labels[i])), linewidth=1)

    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    fig.colorbar(sm, ax=ax, label="Trial")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power")
    ax.set_title(f"{channel} — {freq_label} vs. time, all trials")
    fig.tight_layout()
    return fig


def plot_psd_grid(
    psd: np.ndarray,
    freqs: np.ndarray,
    channel_names: list[str],
    channel: str = "Oz",
    freq_max: float = 40.0,
    n_cols: int = 5,
) -> plt.Figure:
    """Grid of per-trial PSD subplots for one channel, one subplot per trial.

    Parameters
    ----------
    psd : np.ndarray, shape (n_channels, n_freqs, n_trials)
        Output of eeg.psd.compute_psd.
    freqs : np.ndarray, shape (n_freqs,)
    channel_names : list[str]
        Full channel name list, in the same order as psd's channel axis.
    channel : str
        Channel to plot.
    freq_max : float
        Upper x-axis limit (Hz); also excludes higher bins from the y-axis autoscale.
    n_cols : int
        Number of subplot columns.
    """
    ch_idx = channel_names.index(channel)
    spectra = psd[ch_idx]  # (n_freqs, n_trials)
    n_trials = spectra.shape[1]
    n_rows = int(np.ceil(n_trials / n_cols))

    mask = freqs <= freq_max
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.6 * n_cols, 2.0 * n_rows), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes).ravel()

    for trial in range(n_trials):
        ax = axes[trial]
        ax.semilogy(freqs[mask], spectra[mask, trial])
        ax.set_title(f"trial {trial}", fontsize=8)
        ax.tick_params(labelsize=6)

    for ax in axes[n_trials:]:
        ax.axis("off")

    fig.suptitle(f"PSD per trial — {channel}")
    fig.supxlabel("Frequency (Hz)")
    fig.supylabel("PSD")
    fig.tight_layout()
    return fig


def plot_band_power_vs_trial(
    band_values: np.ndarray,
    freq_label: str,
    channel: str = "Oz",
    baseline_trial_idx: np.ndarray | None = None,
) -> plt.Figure:
    """Band power at one frequency (or range) as a function of trial number.

    Parameters
    ----------
    band_values : np.ndarray, shape (n_trials,)
        Output of eeg.psd.band_power.
    freq_label : str
        Label for the title, e.g. "10 Hz" or "9-11 Hz".
    channel : str
    baseline_trial_idx : np.ndarray | None
        Trial indices to highlight as baseline trials (boolean mask or integer array).
    """
    n = len(band_values)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, band_values, color="steelblue", linewidth=1, zorder=2)
    ax.scatter(x, band_values, color="steelblue", s=25, zorder=3, label="stimulus")
    if baseline_trial_idx is not None:
        ax.scatter(
            x[baseline_trial_idx], band_values[baseline_trial_idx],
            color="firebrick", s=40, zorder=4, label="baseline",
        )
        ax.legend()
    ax.set_xlabel("Trial")
    ax.set_ylabel("PSD")
    ax.set_title(f"{channel} power at {freq_label} vs. trial")
    ax.set_xticks(x)
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
