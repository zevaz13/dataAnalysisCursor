# EEG data analysis

## Folder structure
dataAnalysisCursor/
  data/
    eeg/
      raw/            # raw BCI2000 or other EEG files
  src/
    eeg/
  scripts/
    eeg/              #.py files 
  notebooks/
    eeg/              #.ipynb
  configs/
    eeg_preprocessing.yaml
  docs/
    EEG/

## Requirements

Build a modular Python codebase for loading, preprocessing, visualizing, and analyzing EEG data using MNE-Python and python in general. Start small with a dummy dataset (BCI2000 format), then extend to more complex pipelines:

- user sets a path for raw data or data file name. 
- we design functionalities to process that data and plot it at certain stages
- Prefer small, modular functions over large scripts.
- When adding new functionality, create a script in scripts/eeg/ that calls functions from src/eeg/.
- Use MNE-Python idioms (Raw, Epochs, ICA, events, montages).
- Always validate imports by printing: number of channels, sampling rate, channel names, montage (if available),event/stimulus channel
- Always create a notebook for the user to test the code and outputs you create. for this use the notebooks/eeg directory
- Always keep track of current working plan using a file called plan.md in the project root directory

## Technical Decisions

- IMPORTANT: Use "uv" as the package manager for python. ONLY UV.
- create a new virtual environment that adds the needed packages to check the data. mne, matplotlib, pandas etc..
- create notebooks that allow the user to test functionality
- The user should be able to explore data by themselves looking at the documentation and creating scripts or notebooks of their own, calling the module fdunctions exposed in src/eeg

## Starting Point

The projec starts with basic loading functionality to load data (in BCI2000 format), with digital inputs, plot complete datasets in the time domain, and basic frequency domain plotting. 
## Coding standards

1. Use latest versions of libraries and idiomatic approaches as of today
2. Keep it simple - NEVER over-engineer, ALWAYS simplify, NO unnecessary defensive programming. No extra features - focus on simplicity.
3. Be concise. Keep README minimal. IMPORTANT: no emojis ever
4. When hitting issues, always identify root cause before trying a fix. Do not guess. Prove with evidence, then fix the root cause.

## Working documentation

All documents for planning and executing this project will be in the docs/ directory. Create if it doesn't exist
Please review plan.md in the project root before proceeding.

## Current milestone
1. Create scripts and functions for more granular signal exploration:
    - plot one or multiple channels, using the channel names as inputs
    - plot one or multiple channels psd, using the chanel name as input. Use mne functions for this.
    - plot information about one component, after ICA is used
2. have a script and notebook that performs the following pipeline on the data:
    - loads the dataset to mne using load_bci2k, assigning the stimulus channel as appropiate
    - bandpass filter the data using the following cutoffs [1 55] (must be an input parameter in the function), as well as the filter order (4 as default)
    - re reference to the average of the channels
    - compute independent component analysis, using infomax, pca = 24. 
    - use iclabel (mne has something similar to this.)
3. Plot the information for the first 2 components. show the components labeled by iclabel