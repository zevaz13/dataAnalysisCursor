# EEG Analysis Pipeline

Modular Python pipeline for loading, preprocessing, and analysing EEG data recorded with BCI2000. Implements the full preprocessing chain (filtering, ICA artifact removal) and FBCCA-based SSVEP frequency detection.

---

## Features

- Load BCI2000 `.dat` files via MNE-Python with correct channel names, standard 10-20 montage, and a digital stimulus channel
- Bandpass filtering and average re-referencing
- ICA decomposition with automated artifact labelling via ICLabel
- Filter-bank CCA (FBCCA) for SSVEP detection, following Chen et al. (2015)
- Plotting: raw traces, PSDs, ICA components, before/after artifact removal, FBCCA score stream
- CLI scripts and Jupyter notebooks for every stage

---

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

---

## Setup

```bash
git clone <repo-url>
cd dataAnalysisCursor
uv sync
```

Place BCI2000 `.dat` files in `data/eeg/raw/`. The default file used by all scripts is:

```
data/eeg/raw/MET000bGridFixedS001R02.dat
```

---

## Quick start

```bash
# Load a file, print metadata, save raw traces + PSD plots
uv run python scripts/eeg/load_and_inspect.py

# Run the full preprocessing pipeline
uv run python scripts/eeg/preprocess_pipeline.py

# Run FBCCA on the cleaned epochs (10 Hz target, all channels)
uv run python scripts/eeg/fbcca.py --freq 10

# Open interactive notebooks
uv run jupyter lab
```

All output figures and saved ICA files go to `outputs/` (auto-created).

---

## Pipeline overview

```
BCI2000 .dat
    │
    ▼
load_bci2k()          parse channel names, attach montage, inject STI 014 stim channel
    │
    ▼
bandpass_filter()     1–55 Hz, 4th-order Butterworth IIR, zero-phase
    │
    ▼
rereference_average() subtract mean of all EEG channels
    │
    ▼
run_ica()             extended Infomax, 24 components
label_components()    ICLabel neural-network classifier
remove_artifacts()    subtract eye blink and muscle artifact ICs
    │
    ▼
make_epochs()         lock to STI 014 rising edges, default 0–3 s windows
    │
    ▼
run_fbcca()           5 filter banks × CCA × weighted combination → score per epoch
```

---

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/eeg/load_and_inspect.py` | Load a file, print metadata, save raw traces / stim channel / PSD plots |
| `scripts/eeg/explore_channels.py` | Plot time traces and PSD for a named subset of channels |
| `scripts/eeg/preprocess_pipeline.py` | Full preprocessing: filter → reref → ICA → ICLabel → artifact removal → epochs |
| `scripts/eeg/fbcca.py` | Full pipeline + FBCCA scoring + stream plot |

All scripts accept `--path` to specify a different `.dat` file and `--output-dir` to change where figures are saved. Run any script with `--help` for the full argument list.

### Example: FBCCA on occipital channels

```bash
uv run python scripts/eeg/fbcca.py \
    --freq 10 \
    --channels Oz O1 O2 \
    --eye-threshold 0.8 \
    --muscle-threshold 0.6
```

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `notebooks/eeg/load_and_inspect.ipynb` | Interactive load and metadata inspection |
| `notebooks/eeg/explore_channels.ipynb` | Per-channel time and frequency domain plots |
| `notebooks/eeg/preprocess_pipeline.ipynb` | Step-through preprocessing with inline plots |
| `notebooks/eeg/full_pipeline_fbcca.ipynb` | Complete pipeline including FBCCA and score stream plot |

```bash
uv run jupyter lab
```

---

## Library — `src/eeg/`

All reusable logic lives here. Scripts and notebooks are thin wrappers.

| Module | Key functions |
|--------|--------------|
| `io.py` | `load_bci2k(path, stim_state)` |
| `bci2k_meta.py` | `parse_channel_names`, `decode_states`, `digital_input_to_events` |
| `inspect.py` | `print_summary(raw)` |
| `preprocessing.py` | `bandpass_filter`, `rereference_average`, `run_ica`, `label_components`, `remove_artifacts`, `make_epochs` |
| `fbcca.py` | `run_fbcca(epochs, freq, channels, ...)` |
| `viz.py` | `plot_raw_traces`, `plot_channels`, `plot_stim_channel`, `plot_psd`, `plot_channels_psd`, `plot_montage`, `plot_ica_components`, `plot_before_after`, `plot_fbcca_stream` |

### Python API example

```python
from pathlib import Path
from eeg import (
    load_bci2k, print_summary,
    bandpass_filter, rereference_average,
    run_ica, label_components, remove_artifacts,
    make_epochs, run_fbcca, plot_fbcca_stream,
)

path = Path("data/eeg/raw/MET000bGridFixedS001R02.dat")

raw = load_bci2k(path)
print_summary(raw)

bandpass_filter(raw)
rereference_average(raw)

ica    = run_ica(raw)
labels = label_components(raw, ica)
raw_clean, excluded = remove_artifacts(raw, ica, labels)

epochs = make_epochs(raw_clean)                        # (trials, channels, samples)
scores = run_fbcca(epochs, freq=10.0)                  # (n_trials,)

fig = plot_fbcca_stream(scores, freq=10.0)
fig.savefig("outputs/fbcca_stream.png", dpi=150)
```

---

## FBCCA

FBCCA (Filter-Bank Canonical Correlation Analysis) detects SSVEP responses by measuring the canonical correlation between the observed EEG and a sin/cos reference at the target frequency and its harmonics. Five sub-bands are used, each retaining a progressively higher set of harmonics, and their squared correlations are combined with decreasing weights.

Filter banks (same as Chen 2015):

| Bank | Pass-band | Harmonics passed (10 Hz target) |
|------|-----------|----------------------------------|
| 1 | 5–55 Hz | 1st, 2nd, 3rd, 4th, 5th |
| 2 | 15–55 Hz | 2nd, 3rd, 4th, 5th |
| 3 | 25–55 Hz | 3rd, 4th, 5th |
| 4 | 35–55 Hz | 4th, 5th |
| 5 | 45–55 Hz | 5th |

Weights: `w_j = j^(−1.25) + 0.25`, giving `[1.25, 0.67, 0.50, 0.43, 0.38]`.

Score per epoch: `Σ w_j · ρ_j²`, where `ρ_j` is the maximum canonical correlation for filter bank `j`.

---

## Project layout

```
dataAnalysisCursor/
├── configs/
│   └── eeg_preprocessing.yaml   # raw_dir, default_file
├── data/
│   └── eeg/raw/                  # BCI2000 .dat files (not committed)
├── docs/
│   └── EEG/
│       ├── overview.md           # architecture and data flow
│       ├── api_reference.md      # full function reference
│       ├── usage_guide.md        # usage guide for scripts, notebooks, API
│       └── fbcca.md              # FBCCA algorithm and usage
├── notebooks/
│   └── eeg/
│       ├── load_and_inspect.ipynb
│       ├── explore_channels.ipynb
│       ├── preprocess_pipeline.ipynb
│       └── full_pipeline_fbcca.ipynb
├── outputs/                      # figures and saved ICA files (auto-created)
├── scripts/
│   └── eeg/
│       ├── load_and_inspect.py
│       ├── explore_channels.py
│       ├── preprocess_pipeline.py
│       └── fbcca.py
├── src/
│   └── eeg/
│       ├── __init__.py
│       ├── bci2k_meta.py
│       ├── fbcca.py
│       ├── inspect.py
│       ├── io.py
│       ├── preprocessing.py
│       └── viz.py
├── pyproject.toml
└── README.md
```

---

## Dependencies

| Package | Version | Role |
|---------|---------|------|
| `mne` | ≥ 1.12 | EEG I/O, filtering, ICA, epoching |
| `mne-icalabel` | ≥ 0.9 | ICLabel neural-network component classifier |
| `scipy` | ≥ 1.7 | Butterworth filter design and zero-phase filtering |
| `onnxruntime` | ≥ 1.26 | Runtime for the ICLabel ONNX model |
| `matplotlib` | ≥ 3.10 | Plotting |
| `pandas` | ≥ 3.0 | Data manipulation |
| `jupyter` | — | Interactive notebooks |

---

## Documentation

Detailed documentation lives in `docs/EEG/`:

- **[overview.md](docs/EEG/overview.md)** — architecture, data flow diagram, design decisions
- **[api_reference.md](docs/EEG/api_reference.md)** — every public function with parameters and return values
- **[usage_guide.md](docs/EEG/usage_guide.md)** — step-by-step guide for scripts, notebooks, and the Python API
- **[fbcca.md](docs/EEG/fbcca.md)** — FBCCA algorithm, filter bank design, score interpretation, multi-frequency classification
