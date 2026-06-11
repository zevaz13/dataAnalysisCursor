#!/usr/bin/env python3
"""Plot individual channels in time and frequency domains."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eeg.inspect import print_summary
from eeg.io import load_bci2k
from eeg.viz import plot_channels, plot_channels_psd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = PROJECT_ROOT / "data/eeg/raw/MET000bGridFixedS001R02.dat"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot selected EEG channels")
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    parser.add_argument(
        "--channels",
        nargs="+",
        default=["Cz", "Fz", "Pz"],
        metavar="CH",
        help="Channel names to plot (e.g. C3 C4 Fz)",
    )
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--fmax", type=float, default=100.0)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    print(f"Loading: {args.path}")
    raw = load_bci2k(args.path)
    print_summary(raw)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.path.stem
    ch_tag = "_".join(args.channels)

    fig = plot_channels(raw, args.channels, duration=args.duration)
    out = args.output_dir / f"{stem}_{ch_tag}_traces.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"traces -> {out}")

    fig = plot_channels_psd(raw, args.channels, fmax=args.fmax)
    out = args.output_dir / f"{stem}_{ch_tag}_psd.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"PSD    -> {out}")


if __name__ == "__main__":
    main()
