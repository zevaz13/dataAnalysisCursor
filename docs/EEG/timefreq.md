# Time-Frequency Decomposition — Morlet Wavelet Convolution

## Background

Time-frequency decomposition tracks how spectral power at each frequency evolves over time within a trial — unlike a single Welch PSD per trial, it resolves the timing of an oscillatory response (e.g. an SSVEP ramping up after stimulus onset).

This is a Python port of `templateCode/timeFreq` (MATLAB, Cohen-style wavelet convolution: *Analyzing Neural Time Series Data*, M.X. Cohen). The convolution itself is vectorized with `scipy.signal.fftconvolve` instead of the original manual FFT/trial-concatenation trick — numerically equivalent in shape (verified against a literal port on synthetic data: constant-ratio match to ~1e-7 relative precision), differing only in the wavelet's overall normalization constant. That constant is a per-frequency scalar shared between the baseline and the activity compared against it, so it cancels out exactly in every baseline-normalized output (percent change, dB, z-score).

Baseline normalization differs from the MATLAB template on purpose: the template baselines against a **time window** within the same trial. Here, baseline stats come from the **3 dedicated baseline trials** (no stimulus) of the experiment, pooled across all their time points — see plan.md M4.

---

## Implementation

### `src/eeg/eeglab_io.py`

#### `load_eeglab_epochs(path) -> mne.EpochsArray`

Loads an EEGLAB `R0*.mat` run (as saved by the Metamers SSVEP experiment) and returns an `mne.EpochsArray`:

1. `scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)` — reads the `EEG` struct.
2. `EEG.data` (`channels x times x trials`, `float32`) is cast to `float64` and transposed to `(trials, channels, times)` — MNE's convention, matching `epochs.get_data()` elsewhere in the codebase (e.g. `fbcca.py`).
3. Channel names come from `EEG.chanlocs[i].labels` (standard 10-20 extended names — `PO7`, `Oz`, `AF4`, etc.).
4. The first `N_BASELINE_TRIALS` (3) trials are tagged event id `1` (`"baseline"`); the rest are tagged event id `2` (`"stimulus"`), so callers can do `epochs["baseline"]` / `epochs["stimulus"]`.
5. A `standard_1020` montage is attached (`on_missing="warn"`, same pattern as `io.py::load_bci2k`).

### `src/eeg/timefreq.py`

#### `_morlet_wavelet(freq, n_cycles, sfreq) -> np.ndarray`

Builds one complex Morlet wavelet over a fixed `-2` to `2` second window (matches the MATLAB template), normalized to unit energy:

```
wavelet = exp(2i*pi*freq*t) * exp(-t^2 / (2*s^2))    where s = n_cycles / (2*pi*freq)
```

#### `time_frequency_decompose(data, sfreq, freq_range=(2, 30), n_freqs=50, n_cycles_range=(4, 10), include_freq=10.0) -> (tf, frex)`

- `data`: `(n_trials, n_channels, n_times)`.
- Frequencies are linearly spaced across `freq_range`; if `include_freq` is set, the nearest bin is snapped to exactly that value (e.g. a known SSVEP stimulus frequency), regardless of `n_freqs`. Raises `ValueError` if `include_freq` falls outside `freq_range` — pass `include_freq=None` to disable snapping, e.g. when scanning a range that doesn't contain your target frequency.
- Number of wavelet cycles is log-spaced across `n_cycles_range` (fewer cycles → better time resolution at low frequencies; more cycles → better frequency resolution at high frequencies).
- For each frequency: build the wavelet, convolve with `scipy.signal.fftconvolve(data, wavelet, mode="same", axes=-1)` — vectorized over trials and channels simultaneously (only the frequency loop remains explicit), then take `|analytic signal|^2` for power.
- **Returns** `tf` with shape `(n_channels, n_freqs, n_times, n_trials)` and `frex` (Hz).

#### `compute_baseline_stats(tf, baseline_trial_idx) -> (baseline_mean, baseline_std)`

Pools over **time and the baseline trials** (last two axes after selecting `baseline_trial_idx`), giving one scalar mean/std per `(channel, frequency)`, shape `(n_channels, n_freqs)`.

#### `baseline_normalize(tf, baseline_mean, baseline_std, method="db") -> tf_norm`

| `method` | Formula |
|----------|---------|
| `"percent"` | `(tf - mean) / mean * 100` |
| `"db"` | `10 * log10(tf / mean)` |
| `"zscore"` | `(tf - mean) / std` |

### `src/eeg/viz.py`

#### `plot_time_frequency(tf, frex, times, channel_names, channel, trial=None, cmap="RdBu_r") -> Figure`

Spectrogram (`pcolormesh`, frequency x time) for one channel. `trial=None` averages across all trials; pass an integer to plot a single trial. Diverging colormap centered at 0 — intended for baseline-normalized `tf`, not raw power.

---

## Usage

### Script

```bash
uv run python scripts/eeg/timefreq.py
uv run python scripts/eeg/timefreq.py --participant MET003TESTBLUE --run R01 --channel O1 --method zscore
uv run python scripts/eeg/timefreq.py --min-freq 4 --max-freq 40 --n-freqs 80 --include-freq 10
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--participant` | `MET004TESTBLUE` | Folder under `--data-root` |
| `--run` | `R01` | `.mat` file stem |
| `--data-root` | Metamers `compsRem` share (WSL mount) | Root containing per-participant folders |
| `--min-freq` / `--max-freq` | `2.0` / `30.0` | Frequency range, Hz |
| `--n-freqs` | `50` | Number of frequency bins |
| `--min-cycles` / `--max-cycles` | `4.0` / `10.0` | Wavelet cycle range |
| `--include-freq` | `10.0` | Frequency bin snapped exactly (SSVEP target) |
| `--method` | `db` | `percent`, `db`, or `zscore` |
| `--channel` | `Oz` | Channel to plot |

**Output:** `outputs/{participant}_{run}_timefreq_{channel}_{method}.png`

### Notebook

`notebooks/eeg/timefreq.ipynb` — validates the decomposition on a synthetic 10 Hz burst (asserts the recovered peak frequency and timing match what was injected) before running on real data.

### Python API

```python
from eeg import load_eeglab_epochs, time_frequency_decompose, compute_baseline_stats, baseline_normalize, plot_time_frequency
from eeg.eeglab_io import BASELINE_EVENT_ID

epochs = load_eeglab_epochs(path)
data, sfreq = epochs.get_data(), epochs.info["sfreq"]

tf, frex = time_frequency_decompose(data, sfreq, include_freq=10.0)

baseline_idx = epochs.events[:, 2] == BASELINE_EVENT_ID
mean, std = compute_baseline_stats(tf, baseline_idx)
tf_db = baseline_normalize(tf, mean, std, method="db")

fig = plot_time_frequency(tf_db, frex, epochs.times, epochs.ch_names, channel="Oz")
```
