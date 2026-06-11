#!/usr/bin/env python3
"""Full preprocessing pipeline: filter → reref → ICA → ICLabel → artifact removal → epochs."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
from eeg.viz import plot_before_after, plot_ica_components

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = PROJECT_ROOT / "data/eeg/raw/MET000bGridFixedS001R02.dat"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOT_CHANNELS = ["Cz", "Fz", "Pz"]


def main() -> None:
    parser = argparse.ArgumentParser(description="EEG preprocessing pipeline")
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--l-freq", type=float, default=1.0)
    parser.add_argument("--h-freq", type=float, default=55.0)
    parser.add_argument("--filter-order", type=int, default=4)
    parser.add_argument("--n-components", type=int, default=24)
    parser.add_argument("--eye-threshold", type=float, default=0.7)
    parser.add_argument("--muscle-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.path.stem

    print(f"Loading: {args.path}")
    raw = load_bci2k(args.path)
    print_summary(raw)

    print(f"\nBandpass filter [{args.l_freq}–{args.h_freq} Hz], order {args.filter_order}")
    bandpass_filter(raw, args.l_freq, args.h_freq, args.filter_order)

    print("Re-referencing to average")
    rereference_average(raw)

    print(f"Running ICA (infomax, n_components={args.n_components})")
    ica = run_ica(raw, n_components=args.n_components)
    ica_path = args.output_dir / f"{stem}_ica.fif"
    ica.save(ica_path, overwrite=True)
    print(f"ICA saved -> {ica_path}")

    print("Labeling components with ICLabel")
    labels = label_components(raw, ica)
    labels_path = args.output_dir / f"{stem}_iclabel.json"
    labels_path.write_text(
        json.dumps(
            {
                "labels": labels["labels"],
                "y_pred_proba": [p.tolist() for p in labels["y_pred_proba"]],
            },
            indent=2,
        )
    )
    print(f"ICLabel saved -> {labels_path}")
    for i, (label, proba) in enumerate(zip(labels["labels"], labels["y_pred_proba"])):
        print(f"  IC {i:02d}: {label} ({proba.max():.0%})")

    print("\nPlotting first 2 components")
    figs = plot_ica_components(raw, ica, [0, 1], labels=labels)
    for i, fig in enumerate(figs):
        out = args.output_dir / f"{stem}_ica_comp{i}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  comp {i} -> {out}")

    print(f"\nRemoving artifacts (eye>{args.eye_threshold:.0%}, muscle>{args.muscle_threshold:.0%})")
    raw_clean, excluded = remove_artifacts(
        raw, ica, labels,
        eye_threshold=args.eye_threshold,
        muscle_threshold=args.muscle_threshold,
    )
    print(f"  Excluded ICs: {excluded}")

    print(f"Before/after traces: {PLOT_CHANNELS}")
    fig = plot_before_after(raw, raw_clean, PLOT_CHANNELS)
    out = args.output_dir / f"{stem}_before_after.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  before/after -> {out}")

    print("\nCreating epochs [0–3 s from stim onset]")
    epochs = make_epochs(raw_clean)
    print(f"  Epochs: {epochs}")
    print(f"  Shape: {epochs.get_data().shape}  (trials × channels × samples)")


if __name__ == "__main__":
    main()
