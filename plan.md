# EEG Analysis — Working Plan

## Current milestone — COMPLETE

All three steps implemented and tested against `MET000bGridFixedS001R02.dat`.

### What was built

**Step 1 — Granular signal exploration**
- `src/eeg/viz.py`: `plot_channels(raw, channel_names, duration)`, `plot_channels_psd(raw, channel_names, fmax)`
- `scripts/eeg/explore_channels.py` — CLI: `--path`, `--channels`, `--duration`, `--fmax`, `--output-dir`
- `notebooks/eeg/explore_channels.ipynb`

**Step 2 — Preprocessing pipeline**
- `src/eeg/preprocessing.py`: `bandpass_filter`, `rereference_average`, `run_ica`, `label_components`
  - ICA: extended Infomax, `n_components=24`, `random_state=42`
  - ICLabel backend: `onnxruntime`
- `scripts/eeg/preprocess_pipeline.py` — saves ICA to `outputs/<stem>_ica.fif`, labels to `outputs/<stem>_iclabel.json`
- `notebooks/eeg/preprocess_pipeline.ipynb`

**Step 3 — ICA component visualization**
- `src/eeg/viz.py`: `plot_ica_components(raw, ica, component_indices, labels=None)`
  - annotates figure title with ICLabel label and confidence
  - wired into `preprocess_pipeline.py` for components [0, 1]

**Dependencies added**: `mne-icalabel==0.9.0`, `onnxruntime==1.26.0`

---

## Next milestone

TBD.
