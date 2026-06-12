# EEG Library — API Reference

All symbols below are importable from the `eeg` package directly:

```python
from eeg import load_bci2k, print_summary, bandpass_filter, ...
```

---

## `eeg.io`

### `load_bci2k(path, stim_state="DigitalInput1") -> mne.io.BaseRaw`

Loads a BCI2000 `.dat` file and returns a fully configured MNE `Raw` object.

**Steps performed internally:**
1. Reads raw data with `mne.io.read_raw_bci2k` (preloaded into memory).
2. Parses channel names from the BCI2000 header and renames MNE's default `ch0 … chN` labels.
3. Attaches the `standard_1020` montage. Channels not in the montage trigger a warning but do not raise.
4. Decodes the requested state vector field (`stim_state`) and injects it as a stim channel named `"STI 014"`.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `path` | `Path` | — | Path to the `.dat` file |
| `stim_state` | `str` | `"DigitalInput1"` | BCI2000 state field to use as the trigger line |

**Raises:** `ValueError` if `stim_state` is not present in the file's state vector.

**Returns:** `mne.io.BaseRaw` with EEG channels + `"STI 014"` stim channel, preloaded.

---

### `STIM_CHANNEL_NAME`

String constant `"STI 014"`. The stim channel is always named this throughout the pipeline.

---

## `eeg.bci2k_meta`

Low-level header parsing. Normally called internally by `load_bci2k`; exposed for advanced use.

### `parse_channel_names(path) -> list[str]`

Reads the `ChannelNames` parameter from the BCI2000 header and returns the list of channel name strings. Raises `ValueError` if the field is missing or the count does not match `n_channels`.

### `decode_states(path) -> dict[str, np.ndarray]`

Decodes all state vector fields from the `.dat` file. Returns a dict mapping state name → 1-D integer array of length `n_samples`. Common keys: `DigitalInput1`, `StimulusCode`, `Running`.

### `digital_input_to_events(digital_input, sfreq, event_id=1) -> np.ndarray`

Converts a binary state array to an MNE-format events array by detecting rising edges (0→1 transitions).

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `digital_input` | `np.ndarray` | 1-D binary integer array (e.g. from `decode_states`) |
| `sfreq` | `float` | Sampling frequency (used implicitly for sample indices) |
| `event_id` | `int` | Event ID to fill in column 3 (default `1`) |

**Returns:** Array of shape `(n_events, 3)` — `[sample, 0, event_id]` — compatible with `mne.Epochs`.

---

## `eeg.inspect`

### `print_summary(raw) -> None`

Prints a human-readable summary of the `Raw` object to stdout. Always call this after `load_bci2k` to verify the data loaded correctly.

**Output includes:**
- EEG channel list and count
- All channel names and total channel count
- Sampling rate (Hz) and recording duration (s)
- Montage name (or "none")
- Stimulus channel name and source
- Number of rising edges detected on the stim channel
- Event count and breakdown from `mne.find_events`

---

## `eeg.preprocessing`

All functions except `make_epochs` and `remove_artifacts` operate in-place and return `self` for optional chaining.

### `bandpass_filter(raw, l_freq=1.0, h_freq=55.0, order=4) -> mne.io.BaseRaw`

Applies a Butterworth IIR bandpass filter in-place.

| Parameter | Default | Notes |
|-----------|---------|-------|
| `l_freq` | `1.0` Hz | High-pass cutoff; removes slow drift |
| `h_freq` | `55.0` Hz | Low-pass cutoff; stays below 60 Hz line noise |
| `order` | `4` | Filter order; higher = steeper roll-off, more ringing |

### `rereference_average(raw) -> mne.io.BaseRaw`

Re-references EEG to the average of all EEG channels in-place. Uses `projection=False` (applies immediately rather than storing as a projector).

### `run_ica(raw, n_components=24, method="infomax") -> mne.preprocessing.ICA`

Fits ICA on EEG channels and returns the fitted `ICA` object. Uses extended Infomax (`fit_params={"extended": True}`) with `random_state=42` for reproducibility.

The returned object is not yet applied to the data; call `remove_artifacts` to apply it.

| Parameter | Default | Notes |
|-----------|---------|-------|
| `n_components` | `24` | Number of independent components to extract |
| `method` | `"infomax"` | ICA algorithm; MNE also supports `"fastica"`, `"picard"` |

### `label_components(raw, ica) -> dict`

Runs ICLabel on the fitted ICA and returns a labels dict with keys:

- `"labels"` — list of `str` category names per component (e.g. `"brain"`, `"eye blink"`, `"muscle artifact"`, `"heart beat"`, `"line noise"`, `"channel noise"`, `"other"`)
- `"y_pred_proba"` — list of `np.ndarray` of shape `(7,)`, one probability per category

### `remove_artifacts(raw, ica, labels, eye_threshold=0.7, muscle_threshold=0.5) -> tuple[mne.io.BaseRaw, list[int]]`

Selects ICA components classified as eye blink or muscle artifact above the given probability thresholds, subtracts them, and returns a cleaned copy of the data.

**Returns:** `(raw_clean, excluded_indices)`
- `raw_clean` — new `Raw` object with artifact ICs subtracted
- `excluded_indices` — list of IC indices that were removed

The original `raw` object is not modified.

### `make_epochs(raw, tmin=0.0, tmax=3.0, event_id=1, baseline=None) -> mne.Epochs`

Creates epochs locked to rising edges of `"STI 014"`. Uses `mne.find_events` with `shortest_event=1`.

| Parameter | Default | Notes |
|-----------|---------|-------|
| `tmin` | `0.0` s | Epoch start relative to event |
| `tmax` | `3.0` s | Epoch end relative to event |
| `event_id` | `1` | Event ID to epoch on |
| `baseline` | `None` | Baseline correction window; `None` disables it |

---

## `eeg.viz`

All functions return a `matplotlib.figure.Figure` (or a list of figures for `plot_ica_components`). In scripts, call `fig.savefig(path)` then `plt.close(fig)`. In notebooks, the figure renders automatically.

### `plot_montage(raw, show_names=True) -> Figure`

Plots 2-D electrode positions from the attached montage.

### `plot_raw_traces(raw, duration=10.0, n_channels=20) -> Figure`

Plots up to `n_channels` EEG channels for the first `duration` seconds. One subplot per channel, shared x-axis.

### `plot_channels(raw, channel_names, duration=10.0) -> Figure`

Plots a specific list of channels by name for the first `duration` seconds.

### `plot_stim_channel(raw, duration=60.0) -> Figure | None`

Plots the first stim channel (`"STI 014"`) as a time series. Returns `None` and prints a message if no stim channel exists.

### `plot_psd(raw) -> Figure`

Plots the power spectral density of all channels using MNE's default PSD method.

### `plot_channels_psd(raw, channel_names, fmax=100.0) -> Figure`

Plots PSD for a specific list of channels, up to `fmax` Hz.

### `plot_ica_components(raw, ica, component_indices, labels=None) -> list[Figure]`

Plots ICA component properties (time series, spectrum, topomap, etc.) for each index in `component_indices`. If `labels` is provided (the dict from `label_components`), each figure title is annotated with the ICLabel category and probability.

Returns a list of all figures produced (MNE may generate multiple figures per component).

### `plot_before_after(raw_before, raw_after, channel_names, duration=10.0) -> Figure`

Two-column time-domain comparison: left column shows `raw_before`, right column shows `raw_after`, one row per channel in `channel_names`. Useful for visualizing the effect of ICA artifact removal.
