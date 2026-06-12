# FBCCA — Filter-Bank CCA for SSVEP Detection

## Background

FBCCA (Filter-Bank Canonical Correlation Analysis) is a frequency detection method for Steady-State Visual Evoked Potentials (SSVEP). A stimulus flickering at a known frequency `f` drives the visual cortex to oscillate at `f` and its harmonics. CCA measures the linear relationship between the observed EEG and a reference signal built from those harmonics.

Using multiple bandpass filter banks — each retaining a different subset of harmonics — and combining the resulting correlations with decreasing weights gives better detection than single-band CCA, because it leverages information at each harmonic selectively.

Reference: Chen et al. (2015), *Filter Bank Canonical Correlation Analysis for Implementing a High-Speed SSVEP-Based Brain–Computer Interface*.

---

## Implementation

### `src/eeg/fbcca.py`

#### `FILTER_BANKS`

```python
FILTER_BANKS = [(5, 55), (15, 55), (25, 55), (35, 55), (45, 55)]
```

Five sub-bands, all sharing the upper edge (55 Hz). The lower cutoff rises by 10 Hz each step, progressively excluding the fundamental and lower harmonics so that each bank captures progressively higher-order harmonic content.

For a 10 Hz target:
- Bank 1 `[5–55]` passes fundamental (10 Hz) + harmonics 2 (20) + 3 (30) + 4 (40) + 5 (50)
- Bank 2 `[15–55]` passes harmonics 2, 3, 4, 5
- Bank 3 `[25–55]` passes harmonics 3, 4, 5
- Bank 4 `[35–55]` passes harmonics 4, 5
- Bank 5 `[45–55]` passes harmonic 5 only

#### `_make_reference(n_samples, freq, sfreq, n_harmonics=3)`

Builds the sine/cosine reference matrix for one target frequency. Returns shape `(n_samples, 2 × n_harmonics)`:

```
[sin(2π·f·t), cos(2π·f·t), sin(2·2π·f·t), cos(2·2π·f·t), sin(3·2π·f·t), cos(3·2π·f·t)]
```

Time axis: `t = [0, 1/fs, 2/fs, ..., (n_samples-1)/fs]` — identical to the MATLAB template.

#### `_max_cca(X, Y)`

Computes the largest canonical correlation between `X` (T × p) and `Y` (T × q) via QR decomposition + SVD:

1. Demean `X` (demeaning `Y` is unnecessary — sin/cos have zero mean over the epoch).
2. Thin QR: `X = Qx @ Rx`, `Y = Qy @ Ry`.
3. SVD of `Qx.T @ Qy` → singular values are the canonical correlations.
4. Return the largest singular value.

This is numerically stable and avoids inverting the covariance matrices directly.

#### `run_fbcca(epochs, freq, channels=None, n_harmonics=3, filter_order=4)`

Main function. See the API reference for the full signature. Internal steps:

1. **Extract data** — `epochs.get_data(picks)` → shape `(n_trials, n_ch, n_samples)`.
2. **Build reference** — one call to `_make_reference`; shape `(n_samples, 6)` for 3 harmonics.
3. **Filter bank loop** — for each of the 5 sub-bands:
   - Design a 4th-order Butterworth bandpass with `scipy.signal.butter(order, [lf, hf], fs=sfreq)`.
   - Apply zero-phase forward-backward filtering with `scipy.signal.filtfilt` across the samples axis (vectorised over all trials and channels at once).
4. **CCA loop** — for each trial × filter bank: call `_max_cca(X, ref)` where `X` is `(n_samples, n_ch)`.
5. **Weight and combine** — weights follow Chen 2015 eq. 7:

   ```
   w_j = j^(−1.25) + 0.25,  j = 1 … 5
   ```

   Values: `[1.25, 0.670, 0.503, 0.427, 0.384]`

   Final score per trial:
   ```
   score = sum_j( w_j × rho_j² )
   ```

---

## Pipeline position

FBCCA is applied **after** artifact removal and **after** epoching:

```
load_bci2k()
  → bandpass_filter()
  → rereference_average()
  → run_ica() → label_components() → remove_artifacts()   ← IC-cleaned continuous raw
  → make_epochs()                                           ← 3 s epochs at STI 014 onsets
  → run_fbcca()                                             ← one score per epoch
```

---

## Usage

### Script

```bash
# Default: 10 Hz, all EEG channels
uv run python scripts/eeg/fbcca.py

# Custom frequency and channel subset
uv run python scripts/eeg/fbcca.py --freq 12 --channels Oz O1 O2 Pz

# Adjust thresholds and epoch window
uv run python scripts/eeg/fbcca.py --freq 10 --tmin 0 --tmax 3 \
    --eye-threshold 0.8 --muscle-threshold 0.6

# Full argument list
uv run python scripts/eeg/fbcca.py --help
```

**CLI arguments specific to FBCCA:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--freq` | `10.0` Hz | Target SSVEP frequency |
| `--channels` | all EEG | Channel name(s) to include in CCA |
| `--n-harmonics` | `3` | Number of sin/cos harmonics in the reference |
| `--filter-order` | `4` | Butterworth order for filter banks |
| `--tmin` | `0.0` s | Epoch start relative to stimulus |
| `--tmax` | `3.0` s | Epoch end relative to stimulus |

The script also accepts all preprocessing arguments from `preprocess_pipeline.py` (`--l-freq`, `--h-freq`, `--n-components`, etc.).

**Outputs:**
- `outputs/{stem}_fbcca_{freq}Hz.png` — scatter+line plot of FBCCA score per epoch
- Per-epoch scores printed to stdout

### Python API

```python
from eeg import run_fbcca, plot_fbcca_stream, make_epochs

# After preprocessing and running ICA artifact removal:
epochs = make_epochs(raw_clean, tmin=0.0, tmax=3.0)

# All EEG channels
scores = run_fbcca(epochs, freq=10.0)

# Occipital channels only
scores = run_fbcca(epochs, freq=10.0, channels=["Oz", "O1", "O2"])

print(scores)         # shape (n_trials,)
print(scores.mean())  # average FBCCA score

# Plot
fig = plot_fbcca_stream(scores, freq=10.0)
fig.savefig("outputs/fbcca_stream.png", dpi=150)
```

---

## `plot_fbcca_stream`

In `src/eeg/viz.py`. Plots a scatter+line chart of FBCCA scores over epochs.

```python
plot_fbcca_stream(scores, freq=10.0, title=None) -> Figure
```

- x-axis: epoch number (1-indexed)
- y-axis: weighted FBCCA score
- Title is auto-generated as `"FBCCA score stream — N epochs (target F Hz)"` unless overridden.

---

## Interpreting the score

Higher FBCCA score = stronger correlation between the epoched EEG and the target frequency reference. The score is not bounded to [0, 1] because it is a weighted sum of squared canonical correlations:

- Each `rho_j²` is in `[0, 1]`
- The weights sum to `1.25 + 0.670 + 0.503 + 0.427 + 0.384 = 3.234`
- Theoretical maximum per trial ≈ 3.23 (perfect correlation at all filter banks)
- In practice, scores are well below 1.0 for noise-like signals and rise toward 1–2 for strong SSVEP responses

For classification: compare the score at the expected stimulus frequency against scores computed at other candidate frequencies; the highest score indicates the perceived frequency.

---

## Adapting for multiple frequencies

The current API computes FBCCA for one target frequency at a time. To evaluate all candidate frequencies and classify each epoch:

```python
candidate_freqs = [8.0, 10.0, 12.0, 15.0]

freq_scores = {
    f: run_fbcca(epochs, freq=f, channels=["Oz", "O1", "O2"])
    for f in candidate_freqs
}

# Classify each epoch as the frequency with the highest score
import numpy as np
score_matrix = np.stack([freq_scores[f] for f in candidate_freqs], axis=1)  # (n_trials, n_freqs)
predicted = [candidate_freqs[i] for i in score_matrix.argmax(axis=1)]
```
