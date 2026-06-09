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
- we design functionalities to process that data and plot it at certains tages
- Prefer small, modular functions over large scripts.
- When adding new functionality, create a script in scripts/eeg/ that calls functions from src/eeg/.
- Use MNE-Python idioms (Raw, Epochs, ICA, events, montages).
- Always validate imports by printing: number of channels, sampling rate, channel names, montage (if available),event/stimulus channel
- Always create a notebook for the user to test the code and outputs you create. for this use the notebooks/eeg directory
- Always keep track of current working plan using a file called plan.md in the project root directory

## Technical Decisions

- Use "uv" as the package manager for python
- create a new virtual environment that adds the needed packages to check the data. mne, matplotlib, pandas etc..
- create notebooks that allow the user to test functionality
- The user should be able to explore data by themselves looking at the documentation and creating scripts or notebooks of their own

## Starting Point

We are starting with an almost empty project folder, only with data/eeg/raw in it

## Coding standards

1. Use latest versions of libraries and idiomatic approaches as of today
2. Keep it simple - NEVER over-engineer, ALWAYS simplify, NO unnecessary defensive programming. No extra features - focus on simplicity.
3. Be concise. Keep README minimal. IMPORTANT: no emojis ever
4. When hitting issues, always identify root cause before trying a fix. Do not guess. Prove with evidence, then fix the root cause.

## Working documentation

All documents for planning and executing this project will be in the docs/ directory.
Please review plan.md in the project root before proceeding.

## First milestone
create a script scripts/eeg/load_and_inspect.py, and notebooks/eeg/load_and_inspect.ipynb that 
1. Loads a bci2000 file from data/eeg/raw (uses mne.io.read_raw_bci2k to load the data)
2. Converts it to an MNE Raw object
3. prints metadata (channels, sfreq, events)
4. plots:
   - raw traces,
   - stimulus channel
   - PSD