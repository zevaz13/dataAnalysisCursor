# EEG Pipeline — Usage Guide

## Setup

```bash
uv sync
```

Place BCI2000 `.dat` files in `data/eeg/raw/`. The default file used by all scripts is:

```
data/eeg/raw/MET000bGridFixedS001R02.dat
```

---

## Scripts

All scripts use `matplotlib.use("Agg")` so they run headlessly and save figures to `outputs/` without opening any windows.

### 1. Load and inspect

Loads a file, prints metadata to stdout, and saves three plots to `outputs/`.

```bash
# Default file
uv run python scripts/eeg/load_and_inspect.py

# Specify a file
uv run python scripts/eeg/load_and_inspect.py --path data/eeg/raw/MET000bGridFixedS001R02.dat

# Save plots elsewhere
uv run python scripts/eeg/load_and_inspect.py --output-dir /tmp/eeg_out
```

**Outputs:**
- `{stem}_raw_traces.png` — first 10 s of up to 20 EEG channels
- `{stem}_stim_channel.png` — first 60 s of the `STI 014` stim channel
- `{stem}_psd.png` — power spectral density of all channels

---

### 2. Explore specific channels

Plots time-domain traces and PSD for a named subset of channels.

```bash
# Default: Cz, Fz, Pz
uv run python scripts/eeg/explore_channels.py

# Custom channels
uv run python scripts/eeg/explore_channels.py --channels C3 C4 Oz

# Control window and frequency axis
uv run python scripts/eeg/explore_channels.py --channels Cz --duration 20 --fmax 60
```

**CLI arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--path` | default `.dat` | Path to BCI2000 file |
| `--channels` | `Cz Fz Pz` | One or more channel names |
| `--duration` | `10.0` s | Time window for the trace plot |
| `--fmax` | `100.0` Hz | Upper frequency limit for PSD |
| `--output-dir` | `outputs/` | Where to save figures |

**Outputs:** `{stem}_{ch1}_{ch2}..._traces.png` and `{stem}_{ch1}_{ch2}..._psd.png`

---

### 3. Full preprocessing pipeline

Runs the complete pipeline: load → filter → reref → ICA → ICLabel → artifact removal → epochs.

```bash
# Default settings
uv run python scripts/eeg/preprocess_pipeline.py

# Custom thresholds and filter
uv run python scripts/eeg/preprocess_pipeline.py \
    --l-freq 0.5 --h-freq 40 --filter-order 5 \
    --n-components 20 \
    --eye-threshold 0.8 --muscle-threshold 0.6
```

**CLI arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--path` | default `.dat` | Path to BCI2000 file |
| `--l-freq` | `1.0` Hz | Bandpass high-pass cutoff |
| `--h-freq` | `55.0` Hz | Bandpass low-pass cutoff |
| `--filter-order` | `4` | Butterworth filter order |
| `--n-components` | `24` | Number of ICA components |
| `--eye-threshold` | `0.7` | Min probability to flag as eye blink |
| `--muscle-threshold` | `0.5` | Min probability to flag as muscle artifact |
| `--output-dir` | `outputs/` | Where to save all outputs |

**Outputs:**
- `{stem}_ica.fif` — fitted ICA object (reloadable with `mne.preprocessing.read_ica`)
- `{stem}_iclabel.json` — ICLabel results (labels + probabilities for all components)
- `{stem}_ica_comp0.png`, `{stem}_ica_comp1.png` — properties of first 2 ICA components
- `{stem}_before_after.png` — Cz/Fz/Pz time traces before and after artifact removal

**Stdout per step:**
```
Loading: data/eeg/raw/MET000bGridFixedS001R02.dat
EEG channels (64): ['Fp1', 'Fp2', ...]
Sampling rate: 256.0 Hz
Duration: 300.12 s
...
Bandpass filter [1.0–55.0 Hz], order 4
Re-referencing to average
Running ICA (infomax, n_components=24)
ICA saved -> outputs/MET000bGridFixedS001R02_ica.fif
Labeling components with ICLabel
  IC 00: brain (82%)
  IC 01: eye blink (94%)
  ...
Removing artifacts (eye>70%, muscle>50%)
  Excluded ICs: [1, 5]
Creating epochs [0–3 s from stim onset]
  Epochs: <Epochs | 48 events>
  Shape: (48, 64, 769)  (trials × channels × samples)
```

---

## Notebooks

Notebooks mirror the scripts but allow interactive inspection. Launch JupyterLab with:

```bash
uv run jupyter lab
```

| Notebook | Equivalent script |
|----------|------------------|
| `notebooks/eeg/load_and_inspect.ipynb` | `scripts/eeg/load_and_inspect.py` |
| `notebooks/eeg/explore_channels.ipynb` | `scripts/eeg/explore_channels.py` |
| `notebooks/eeg/preprocess_pipeline.ipynb` | `scripts/eeg/preprocess_pipeline.py` |

In notebooks, figures render inline automatically. You do not need to call `fig.savefig`.

---

## Python API

Use the `eeg` package directly for custom pipelines:

```python
from pathlib import Path
from eeg import (
    load_bci2k,
    print_summary,
    bandpass_filter,
    rereference_average,
    run_ica,
    label_components,
    remove_artifacts,
    make_epochs,
)
from eeg.viz import plot_before_after

path = Path("data/eeg/raw/MET000bGridFixedS001R02.dat")

# Load
raw = load_bci2k(path)
print_summary(raw)

# Preprocess
bandpass_filter(raw, l_freq=1.0, h_freq=55.0)
rereference_average(raw)

# ICA
ica = run_ica(raw, n_components=24)
labels = label_components(raw, ica)

# Artifact removal
raw_clean, excluded = remove_artifacts(raw, ica, labels)
print("Excluded ICs:", excluded)

# Epochs
epochs = make_epochs(raw_clean, tmin=0.0, tmax=3.0)
X = epochs.get_data()  # shape: (trials, channels, samples)

# Optional: plot before/after
fig = plot_before_after(raw, raw_clean, ["Cz", "Fz", "Pz"])
fig.savefig("outputs/before_after.png", dpi=150)
```

### Reloading saved ICA

```python
import mne

ica = mne.preprocessing.read_ica("outputs/MET000bGridFixedS001R02_ica.fif")
```

### Loading ICLabel results from JSON

```python
import json

with open("outputs/MET000bGridFixedS001R02_iclabel.json") as f:
    labels = json.load(f)
# labels["labels"]       → list of str
# labels["y_pred_proba"] → list of list[float]
```

### Using raw BCI2000 state vectors

```python
from eeg.bci2k_meta import decode_states, digital_input_to_events

states = decode_states(path)
print(list(states.keys()))  # ['DigitalInput1', 'StimulusCode', ...]

# Convert to MNE events manually
raw_states = load_bci2k(path)
events = digital_input_to_events(states["DigitalInput1"], sfreq=raw_states.info["sfreq"])
```

---

## Adding a new file

The default data file is hardcoded in each script's `DEFAULT_PATH`. Pass `--path` to override:

```bash
uv run python scripts/eeg/preprocess_pipeline.py --path data/eeg/raw/NewSubject.dat
```

Channel names, montage, and stim channel are all derived from the file's own header, so no code changes are needed for a new file — provided the recording uses the same BCI2000 state fields.

---

## Troubleshooting

**`ValueError: State 'DigitalInput1' not found`**
The recording used a different state field name for the trigger. List available states:
```python
from eeg.bci2k_meta import decode_states
print(list(decode_states(path).keys()))
```
Then pass the correct name: `load_bci2k(path, stim_state="StimulusCode")`.

**`ValueError: Expected N channel names, got M from header`**
The `ChannelNames` field in the BCI2000 header does not match `n_channels`. This usually means the file is truncated or used a non-standard recording configuration.

**MNE private API errors after upgrading MNE**
`bci2k_meta.py` imports `_parse_bci2k_header`, `_read_bci2k_data`, and `_decode_bci2k_states` from `mne.io.bci2k.bci2k`. If MNE renames or removes these, update the import path in `src/eeg/bci2k_meta.py`.

**ICA produces fewer components than requested**
MNE silently reduces `n_components` if it exceeds the rank of the data (e.g. after average re-referencing, rank drops by 1). This is expected behavior.
