"""EEG data loading, preprocessing, inspection, and visualization."""

from eeg.bci2k_meta import digital_input_to_events
from eeg.blue_stimulus import load_blue_levels
from eeg.eeglab_io import load_eeglab_epochs
from eeg.fbcca import run_fbcca
from eeg.io import load_bci2k
from eeg.stimulus import build_stimulus_matrix, parse_sequence, split_baseline_task
from eeg.inspect import print_summary
from eeg.preprocessing import (
    bandpass_filter,
    label_components,
    make_epochs,
    remove_artifacts,
    rereference_average,
    run_ica,
)
from eeg.psd import combine_channels, compute_psd
from eeg.timefreq import baseline_normalize, compute_baseline_stats, time_frequency_decompose
from eeg.viz import (
    plot_baseline_boxplot,
    plot_before_after,
    plot_channels,
    plot_montage,
    plot_channels_psd,
    plot_fbcca_stream,
    plot_ica_components,
    plot_psd,
    plot_raw_traces,
    plot_stim_channel,
    plot_stimulus_heatmap,
    plot_time_frequency,
)

__all__ = [
    "digital_input_to_events",
    "load_blue_levels",
    "load_eeglab_epochs",
    "run_fbcca",
    "load_bci2k",
    "print_summary",
    "parse_sequence",
    "split_baseline_task",
    "build_stimulus_matrix",
    "bandpass_filter",
    "rereference_average",
    "run_ica",
    "label_components",
    "remove_artifacts",
    "make_epochs",
    "combine_channels",
    "compute_psd",
    "time_frequency_decompose",
    "compute_baseline_stats",
    "baseline_normalize",
    "plot_baseline_boxplot",
    "plot_before_after",
    "plot_channels",
    "plot_fbcca_stream",
    "plot_montage",
    "plot_channels_psd",
    "plot_ica_components",
    "plot_psd",
    "plot_raw_traces",
    "plot_stim_channel",
    "plot_stimulus_heatmap",
    "plot_time_frequency",
]
