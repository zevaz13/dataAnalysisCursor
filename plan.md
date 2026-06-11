# EEG Analysis — Working Plan

## Current milestone — COMPLETE

### What was built

**Double-plot fix (`src/eeg/viz.py`)**
- `plot_ica_components` now calls `plt.show()` after annotating figures (skipped on Agg).
- With `%matplotlib inline`: plt.show() triggers flush_figures which displays and closes all
  figures; the post-cell hook finds nothing left to render. One display per figure.
- With Agg (scripts): plt.show() is skipped; figures stay open for savefig.

**Artifact removal (`src/eeg/preprocessing.py`)**
- `remove_artifacts(raw, ica, labels, eye_threshold=0.7, muscle_threshold=0.5)`
  → returns `(raw_clean, excluded_indices)`
- On the test file: excludes IC 01 (eye blink 96%), IC 07 (muscle 52%), IC 09 (eye blink 85%)

**Before/after visualization (`src/eeg/viz.py`)**
- `plot_before_after(raw_before, raw_after, channel_names, duration=10.0)`
  → two-column figure, one row per channel

**Epoch creation (`src/eeg/preprocessing.py`)**
- `make_epochs(raw, tmin=0.0, tmax=3.0, event_id=1, baseline=None)`
  → uses STI 014 rising edges; yields 104 × 33 × 1537 on test file

**Updated pipeline**
- `scripts/eeg/preprocess_pipeline.py` — full pipeline through epochs
- `notebooks/eeg/preprocess_pipeline.ipynb` — steps 7–9 added

---

## Next milestone

TBD.
