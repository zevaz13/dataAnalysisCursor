# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                                              # install dependencies
uv run python scripts/eeg/load_and_inspect.py        # run the load+inspect script
uv run python scripts/eeg/load_and_inspect.py --path data/eeg/raw/<file>.dat
uv run jupyter lab                                   # open notebooks
```

Plots are saved to `outputs/` (auto-created). The default data file is `data/eeg/raw/MET000bGridFixedS001R02.dat`.

## Architecture

All reusable logic lives in `src/eeg/`. Scripts in `scripts/eeg/` are thin CLI wrappers that call those functions. Notebooks in `notebooks/eeg/` mirror each script for interactive use.

### src/eeg modules

| Module | Purpose |
|--------|---------|
| `io.py` | `load_bci2k(path, stim_state)` — loads `.dat`, renames channels, attaches montage, injects digital input as stim channel `"STI 014"` |
| `bci2k_meta.py` | Parses BCI2000 header internals not exposed by MNE: `parse_channel_names`, `decode_states`, `digital_input_to_events`. Uses MNE private APIs (`_parse_bci2k_header`, `_read_bci2k_data`, `_decode_bci2k_states`). |
| `inspect.py` | `print_summary(raw)` — prints channel count, sfreq, montage, stim channel, and event counts |
| `viz.py` | `plot_raw_traces`, `plot_stim_channel`, `plot_psd` — all return `plt.Figure` |

### Key conventions

- The stimulus channel is always named `"STI 014"` (constant in `io.py`) and built from `DigitalInput1` by default.
- `bci2k_meta.py` directly imports from `mne.io.bci2k.bci2k` (private). If MNE updates break this, that is the first place to check.
- New functionality: add function(s) to the relevant `src/eeg/` module, then add a script in `scripts/eeg/` and a notebook in `notebooks/eeg/`.
- Always call `print_summary(raw)` after loading to validate channel names, sfreq, montage, and events.
- Package manager: `uv` only. Never use pip directly.
- Track current work in `plan.md` at the project root.

## Project rules

- No emojis, ever.
- Small, modular functions. No over-engineering.
- When hitting issues: identify root cause with evidence before fixing. Do not guess.
- Configs live in `configs/eeg_preprocessing.yaml`.
- Planning docs live in `docs/EEG/`.
