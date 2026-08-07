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
- allow to combine spectrums by taking the mean across channels, if not specified, we use channel Oz
- we will expand this more.

## milestone 4 (M4) time-frequency analysis. 
- we can use the morlet wavelet convolution we just coded to check these data
- time frequency should be baseline normalized to the results for the baseline trials (different to the template implementation). Should allow percentage change, decibel norm, z-score (lets discuss)
- We will end up with an array for numChannels, times, frequencies, trials.
