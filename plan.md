# EEG Analysis — Working Plan

## milestone 1 (M1) — Make plan [DONE]
- I want to translate the code in /templateCode/timeFreq written in matlab to python.
- This code implements the morlet wavelet convolution for EEG time frequency Analysis. 
- This code also has multiple ways of baseline normalization that need to be clearly presented and shown to the user. 
- Implemented in `src/eeg/timefreq.py`: `time_frequency_decompose` (Morlet wavelet convolution,
  vectorized with `scipy.signal.fftconvolve`), `compute_baseline_stats`, `baseline_normalize`
  (percent / db / zscore, baselined against the 3 baseline trials rather than a time window —
  see M4). CLI: `scripts/eeg/timefreq.py`. Notebook: `notebooks/eeg/timefreq.ipynb` (includes a
  synthetic-signal sanity check). Docs: `docs/EEG/timefreq.md`.

## milestone 2 (M2) — Explore first dataset [R01.mat loading DONE, BlueArray stimulus info pending]
- The datasets will be located in /mnt/c/Users/zevaz/OneDrive/Escritorio/Metamers/eegExp/compsRem/
    - There are 4 different participants of our interest there. MET000TESTBLUE, MET003TESTBLUE, MET004TESTBLUE, AND MET004TESTBLUEb
- The default participant for exploration should be MET004TESTBLUE, we will expand this to the other participants.
- From these folders we only care initially about R01.mat
- Inside these files there should be an EEG structure that follows eeglab conventions.
- The dimensions of EEG.data are (channels, times, trials). There should be 32 channels, montage is found too in the structure, a time vector [0 2.9980] s, and 23 trials. Trials will be [1 2 3] baseline trials, [4:23] stimulus trials.
- The data we will be analyzing is EEG.data, a single that should be converted to a double. 
- find a correspondence between channel name and channel index. 
- plot the spectrum for all trials combined for channel Oz, next to O1, O2 
- Information about blue array stimulus is found in ~/projects/metamers/eeg/ssvepBLUETEST/MET000TESTBLUEBlueTest.npz. The name of the variable is BlueArray
- Implemented in `src/eeg/eeglab_io.py`: `load_eeglab_epochs(path) -> mne.EpochsArray`. Verified
  against real MET004TESTBLUE/R01.mat: 32 channels (standard 10-20 labels), 1536 samples/trial,
  512 Hz, 23 trials (3 baseline + 20 stimulus, tagged via event id). Data cast float32->float64.
  BlueArray stimulus `.npz` integration not yet done.

## milestone 3 (M3) frequency analysis
- we can use psd (welch method) to find the spectrum for each channel, each trial.
- plot in a grid the 23 pds for Oz. 
- plot the psd precisely at 10 Hz (or a range, in which case we take the mean). For each trial we find the value at the frequency, or range. Then we plot them as a function of the trial number.
- allow to combine spectrums by taking the mean across channels, if not specified, we use channel Oz

## milestone 4 (M4) time-frequency analysis. [DONE]
- we can use the morlet wavelet convolution we just coded to check these data
- lets find the time-frequency transformation of these data for channels O1 and Oz.
    -   First plot each channel maps separated, for trial, on a grid. No normalization
    -   Separate the baseline time-frequency transformed runs and find a mean time frequency representation for the baseline
    -   For each trial, I want to normalize each pixel of the time frequency transform (same frequency, same time) using the percentage change, the db model and using logarithmic transofmations.
    -   Put those on a grid of trials. Maybe put this on a function so first call is for channel OZ and second for O1. 
- Lets look, for the resulting last set the time realization for the signal at 10 Hz, and 9 to 11 Hz (Collapse the frequency dimension of each map, by taking the mean)
    - Plot all the trials in a grid. One for each trial
    - Plot all the trials on the same plot, same time. This shows the evolution of 10Hz power over time (if any)
- Implemented: `baseline_normalize` (`src/eeg/timefreq.py`) gained a `"log"` method (natural log
  of the baseline ratio). `eeg.psd.band_power` turned out to generalize as-is to TF arrays
  (frequency must be axis 0; works for any trailing shape), so it's reused unmodified to collapse
  the frequency axis of a channel's TF slice into a `(n_times, n_trials)` time course.
  `src/eeg/viz.py` gained `plot_tf_grid` (per-trial spectrogram grid, raw or normalized),
  `plot_time_course_grid`, and `plot_time_course_overlay` (colored by trial index via colorbar);
  `plot_time_frequency` (M1) retrofitted with explicit `vmin`/`vmax` overrides. All
  spectrogram/heatmap plots take `cmap`/`vmin`/`vmax` to control the color axis explicitly.
  Notebook: `notebooks/eeg/timefreq_analysis.ipynb` — walks through every feature above for
  Oz and O1 on MET004TESTBLUE/R01 (raw grids, baseline, percent/db/log normalized grids, 10 Hz
  and 9-11 Hz time courses as grids and overlays, plus a custom-color-axis demo).
