#!/usr/bin/env python3
"""Load a BCI2000 file, print metadata, and plot raw traces, stim channel, and PSD."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eeg.inspect import print_summary
from eeg.io import load_bci2k
from eeg.viz import plot_psd, plot_raw_traces, plot_stim_channel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = PROJECT_ROOT / "data/eeg/raw/MET000bGridFixedS001R02.dat"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and inspect BCI2000 EEG data")
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_PATH,
        help="Path to BCI2000 .dat file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory to save plot images",
    )
    args = parser.parse_args()

    if not args.path.exists():
        raise FileNotFoundError(f"Data file not found: {args.path}")

    print(f"Loading: {args.path}")
    raw = load_bci2k(args.path)

    print("\n--- Metadata ---")
    print_summary(raw)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.path.stem

    print("\n--- Plots (saved to outputs/) ---")
    fig = plot_raw_traces(raw)
    fig.savefig(args.output_dir / f"{stem}_raw_traces.png", dpi=150)
    plt.close(fig)
    print(f"  raw traces -> {args.output_dir / f'{stem}_raw_traces.png'}")

    fig = plot_stim_channel(raw)
    if fig is not None:
        fig.savefig(args.output_dir / f"{stem}_stim_channel.png", dpi=150)
        plt.close(fig)
        print(f"  stim channel -> {args.output_dir / f'{stem}_stim_channel.png'}")

    psd_fig = plot_psd(raw)
    psd_fig.savefig(args.output_dir / f"{stem}_psd.png", dpi=150)
    plt.close(psd_fig)
    print(f"  PSD -> {args.output_dir / f'{stem}_psd.png'}")


if __name__ == "__main__":
    main()
