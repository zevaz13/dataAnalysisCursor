# EEG Pipeline — Architecture Overview

## Purpose

This project loads EEG data recorded with BCI2000, runs a preprocessing pipeline (filtering, re-referencing, ICA artifact removal), and produces clean epochs ready for analysis. All reusable logic lives in `src/eeg/`; scripts and notebooks are thin wrappers.

---

## Module map

```
src/eeg/
  bci2k_meta.py   — low-level BCI2000 header parsing (private MNE APIs)
  io.py           — public load function, stim channel injection
  inspect.py      — metadata printer
  preprocessing.py — filter, reref, ICA, ICLabel, epochs
  viz.py          — all plotting functions
  __init__.py     — re-exports every public symbol
```

---

## Data flow

```
BCI2000 .dat file
       |
       v
  bci2k_meta.py
    parse_channel_names()   — reads ChannelNames from raw header bytes
    decode_states()         — decodes all state vector fields
       |
       v
  io.py
    load_bci2k()
      mne.io.read_raw_bci2k()   — MNE reader (channels, sfreq, data)
      rename_channels()          — applies parsed names
      set_montage()              — standard_1020 layout
      inject STI 014             — DigitalInput1 → stim channel
       |
       v
  mne.io.BaseRaw  (EEG channels + STI 014)
       |
    +--+-----------------------------------------------+
    |                                                   |
inspect.py                                        preprocessing.py
print_summary()                              bandpass_filter()
  channel list                               rereference_average()
  sampling rate                              run_ica()
  montage info                               label_components()
  stim events                                remove_artifacts()
                                             make_epochs()
    |                                              |
    v                                              v
  stdout                                    mne.Epochs (clean)


viz.py  (called at any stage)
  plot_montage()
  plot_raw_traces()
  plot_channels()
  plot_stim_channel()
  plot_psd()
  plot_channels_psd()
  plot_ica_components()
  plot_before_after()
```

---

## Key design decisions

### BCI2000 header parsing

MNE's `read_raw_bci2k` does not expose channel names or state vectors through its public API. `bci2k_meta.py` bypasses this by importing three private functions directly from `mne.io.bci2k.bci2k`:

- `_parse_bci2k_header` — returns a dict with `n_channels`, `header_len`, `state_defs`, `sfreq`
- `_read_bci2k_data` — returns the raw data matrix and the raw state bytes
- `_decode_bci2k_states` — unpacks the state bytes into named arrays

If a future MNE upgrade moves or renames these, `bci2k_meta.py` is the first place to fix.

### Stimulus channel

The digital input line (`DigitalInput1` by default) is extracted from the BCI2000 state vector and injected as a dedicated MNE stim channel named `"STI 014"`. This name is defined as the constant `STIM_CHANNEL_NAME` in `io.py` and imported everywhere it is needed, so there is a single source of truth.

MNE's event utilities (`find_events`, `Epochs`) then treat this channel as a standard trigger line.

### ICA artifact removal

`run_ica` fits extended Infomax ICA. `label_components` then runs ICLabel (a neural-network classifier) to assign each independent component a category (`brain`, `eye blink`, `muscle artifact`, etc.). `remove_artifacts` applies thresholds on those probabilities to decide which components to subtract before returning a cleaned copy of the raw data.

### In-place vs. copy

Filtering and re-referencing mutate the `Raw` object in place (MNE default) and return `self` for chaining. `remove_artifacts` returns a new copy via `raw.copy()`, preserving the pre-removal data for the before/after comparison plot.

---

## Directory layout

```
dataAnalysisCursor/
  configs/
    eeg_preprocessing.yaml    — raw_dir and default_file paths
  data/
    eeg/raw/                  — BCI2000 .dat files (not committed)
  docs/
    EEG/                      — this documentation
  notebooks/
    eeg/
      load_and_inspect.ipynb
      explore_channels.ipynb
      preprocess_pipeline.ipynb
  outputs/                    — saved figures and ICA files (auto-created)
  scripts/
    eeg/
      load_and_inspect.py
      explore_channels.py
      preprocess_pipeline.py
  src/
    eeg/                      — library package
  pyproject.toml
```

---

## Dependencies

| Package | Role |
|---------|------|
| `mne >= 1.12` | EEG I/O, filtering, ICA, epoching |
| `mne-icalabel >= 0.9` | ICLabel neural-network component classifier |
| `onnxruntime >= 1.26` | Runtime for the ICLabel ONNX model |
| `matplotlib >= 3.10` | All plotting |
| `pandas >= 3.0` | Data manipulation (MNE dependency) |
| `jupyter` | Interactive notebooks |

Package manager: `uv` only. Never use pip directly.
