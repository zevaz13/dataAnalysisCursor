# EEG Data Analysis

Modular EEG pipeline using MNE-Python. Loads BCI2000 `.dat` files from `data/eeg/raw/`.

## Setup

```bash
uv sync
```

## Load and inspect

```bash
uv run python scripts/eeg/load_and_inspect.py
uv run python scripts/eeg/load_and_inspect.py --path data/eeg/raw/MET000bGridFixedS001R03.dat
```

Interactive exploration: `notebooks/eeg/load_and_inspect.ipynb`

## Layout

- `src/eeg/` — library functions
- `scripts/eeg/` — CLI scripts
- `notebooks/eeg/` — test notebooks
- `configs/` — pipeline configuration
- `plan.md` — current task tracker
