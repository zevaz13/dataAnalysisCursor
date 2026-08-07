#!/usr/bin/env python3
"""Time-frequency pipeline: load EEGLAB run -> Morlet wavelet decomposition -> baseline normalize -> plot."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eeg.eeglab_io import BASELINE_EVENT_ID, load_eeglab_epochs
from eeg.timefreq import baseline_normalize, compute_baseline_stats, time_frequency_decompose
from eeg.viz import plot_time_frequency

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/mnt/c/Users/zevaz/OneDrive/Escritorio/Metamers/eegExp/compsRem")
DEFAULT_PARTICIPANT = "MET004TESTBLUE"
DEFAULT_RUN = "R01"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def main() -> None:
    parser = argparse.ArgumentParser(description="Morlet wavelet time-frequency decomposition")
    parser.add_argument("--participant", default=DEFAULT_PARTICIPANT)
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--min-freq", type=float, default=2.0)
    parser.add_argument("--max-freq", type=float, default=30.0)
    parser.add_argument("--n-freqs", type=int, default=50)
    parser.add_argument("--min-cycles", type=float, default=4.0)
    parser.add_argument("--max-cycles", type=float, default=10.0)
    parser.add_argument("--include-freq", type=float, default=10.0,
                        help="Frequency bin snapped exactly, e.g. a known SSVEP frequency (default: 10.0)")
    parser.add_argument("--method", choices=["percent", "db", "zscore"], default="db")
    parser.add_argument("--channel", default="Oz")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    path = args.data_root / args.participant / f"{args.run}.mat"
    print(f"Loading: {path}")
    epochs = load_eeglab_epochs(path)
    print(f"  {epochs}")

    data = epochs.get_data()
    sfreq = epochs.info["sfreq"]

    print(f"\nDecomposing [{args.min_freq}-{args.max_freq} Hz, {args.n_freqs} bins, "
          f"cycles {args.min_cycles}-{args.max_cycles}]")
    tf, frex = time_frequency_decompose(
        data, sfreq,
        freq_range=(args.min_freq, args.max_freq),
        n_freqs=args.n_freqs,
        n_cycles_range=(args.min_cycles, args.max_cycles),
        include_freq=args.include_freq,
    )
    print(f"  tf shape (channels, freqs, times, trials): {tf.shape}")

    baseline_idx = epochs.events[:, 2] == BASELINE_EVENT_ID
    print(f"\nBaseline trials: {baseline_idx.sum()} of {len(baseline_idx)}")
    baseline_mean, baseline_std = compute_baseline_stats(tf, baseline_idx)

    print(f"Normalizing (method={args.method})")
    tf_norm = baseline_normalize(tf, baseline_mean, baseline_std, method=args.method)

    fig = plot_time_frequency(
        tf_norm, frex, epochs.times, epochs.ch_names, channel=args.channel,
        title=f"{args.participant} {args.run} — {args.channel} ({args.method}, mean across trials)",
    )
    out = args.output_dir / f"{args.participant}_{args.run}_timefreq_{args.channel}_{args.method}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSpectrogram -> {out}")


if __name__ == "__main__":
    main()
