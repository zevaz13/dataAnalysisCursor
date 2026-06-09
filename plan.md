# EEG Analysis — Working Plan

## Milestone 1: Load and inspect

- [x] uv project + dependencies
- [x] `src/eeg/` modules (io, inspect, viz)
- [x] `scripts/eeg/load_and_inspect.py`
- [x] `notebooks/eeg/load_and_inspect.ipynb`
- [x] Validate run on `MET000bGridFixedS001R02.dat`

## Milestone 1.5: BCI2000 metadata

- [x] `src/eeg/bci2k_meta.py` — ChannelNames, state decoding, events helper
- [x] `load_bci2k` — rename channels, standard_1020 montage, DigitalInput1 as STI 014
- [x] Validated on R02: 32 named EEG ch, montage set, 104 DigitalInput1 events

## Backlog (post-milestone 1.5)

1. ~~Channel typing and montage~~ (done via BCI2000 ChannelNames + standard_1020)
2. Basic preprocessing — filter, notch, re-reference (from `configs/eeg_preprocessing.yaml`)
3. ~~Annotations and events~~ (DigitalInput1 as stim channel; trial-count logic TBD)
4. Epoching around events
5. ICA artifact rejection
6. Script + notebook per stage
