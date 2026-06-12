#!/usr/bin/env python3
"""FBCCA pipeline: load → filter → reref → ICA → artifact removal → epochs → FBCCA → plot."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eeg.fbcca import run_fbcca
from eeg.inspect import print_summary
from eeg.io import load_bci2k
from eeg.preprocessing import (
    bandpass_filter,
    label_components,
    make_epochs,
    remove_artifacts,
    rereference_average,
    run_ica,
)
from eeg.viz import plot_fbcca_stream

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = PROJECT_ROOT / "data/eeg/raw/MET000bGridFixedS001R02.dat"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def main() -> None:
    parser = argparse.ArgumentParser(description="FBCCA on preprocessed EEG epochs")
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--freq", type=float, default=10.0,
                        help="Target SSVEP frequency in Hz (default: 10.0)")
    parser.add_argument("--channels", nargs="+", default=None, metavar="CH",
                        help="Channel names to use (default: all EEG channels)")
    parser.add_argument("--n-harmonics", type=int, default=3,
                        help="Number of CCA harmonics (default: 3)")
    parser.add_argument("--filter-order", type=int, default=4,
                        help="Butterworth filter order for filter banks (default: 4)")
    # preprocessing knobs
    parser.add_argument("--l-freq", type=float, default=1.0)
    parser.add_argument("--h-freq", type=float, default=55.0)
    parser.add_argument("--n-components", type=int, default=24)
    parser.add_argument("--eye-threshold", type=float, default=0.7)
    parser.add_argument("--muscle-threshold", type=float, default=0.5)
    parser.add_argument("--tmin", type=float, default=0.0,
                        help="Epoch start relative to stim onset (s)")
    parser.add_argument("--tmax", type=float, default=3.0,
                        help="Epoch end relative to stim onset (s)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.path.stem

    # 1. Load
    print(f"Loading: {args.path}")
    raw = load_bci2k(args.path)
    print_summary(raw)

    # 2. Filter + reref
    print(f"\nBandpass [{args.l_freq}–{args.h_freq} Hz]")
    bandpass_filter(raw, args.l_freq, args.h_freq)
    print("Re-referencing to average")
    rereference_average(raw)

    # 3. ICA
    print(f"\nRunning ICA (n_components={args.n_components})…")
    ica = run_ica(raw, n_components=args.n_components)
    labels = label_components(raw, ica)
    for i, (label, proba) in enumerate(zip(labels["labels"], labels["y_pred_proba"])):
        print(f"  IC {i:02d}: {label} ({proba.max():.0%})")

    # 4. Artifact removal
    print(f"\nRemoving artifacts (eye>{args.eye_threshold:.0%}, muscle>{args.muscle_threshold:.0%})")
    raw_clean, excluded = remove_artifacts(
        raw, ica, labels,
        eye_threshold=args.eye_threshold,
        muscle_threshold=args.muscle_threshold,
    )
    print(f"  Excluded ICs: {excluded}")

    # 5. Epochs
    print(f"\nCreating epochs [{args.tmin}–{args.tmax} s]")
    epochs = make_epochs(raw_clean, tmin=args.tmin, tmax=args.tmax)
    print(f"  {epochs}")

    # 6. FBCCA
    ch_label = "all EEG channels" if args.channels is None else str(args.channels)
    print(f"\nRunning FBCCA @ {args.freq} Hz on {ch_label}")
    scores = run_fbcca(
        epochs,
        freq=args.freq,
        channels=args.channels,
        n_harmonics=args.n_harmonics,
        filter_order=args.filter_order,
    )
    print(f"  n_epochs={len(scores)}, min={scores.min():.4f}, max={scores.max():.4f}, "
          f"mean={scores.mean():.4f}")

    # Print per-epoch scores
    print("\nEpoch-by-epoch FBCCA scores:")
    for i, s in enumerate(scores, start=1):
        print(f"  [{i:3d}] {s:.6f}")

    # 7. Plot
    fig = plot_fbcca_stream(scores, freq=args.freq)
    out = args.output_dir / f"{stem}_fbcca_{args.freq}Hz.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFBCCA stream plot -> {out}")


if __name__ == "__main__":
    main()
