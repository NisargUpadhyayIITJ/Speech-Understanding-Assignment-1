from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from speech_utils import MFCCConfig, compute_manual_mfcc, load_audio


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual MFCC/Cepstrum extraction without librosa.feature.mfcc.")
    parser.add_argument("--audio-path", required=True, help="Path to the input WAV/FLAC file.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate for analysis.")
    parser.add_argument("--frame-ms", type=float, default=25.0, help="Frame length in milliseconds.")
    parser.add_argument("--hop-ms", type=float, default=10.0, help="Hop size in milliseconds.")
    parser.add_argument("--n-fft", type=int, default=512, help="FFT size.")
    parser.add_argument("--n-mels", type=int, default=40, help="Number of Mel filters.")
    parser.add_argument("--n-ceps", type=int, default=13, help="Number of cepstral coefficients to retain.")
    parser.add_argument("--window", default="hamming", choices=["rectangular", "hamming", "hanning", "hann"])
    parser.add_argument("--pre-emphasis", type=float, default=0.97, help="Pre-emphasis coefficient.")
    parser.add_argument("--fmin", type=float, default=0.0, help="Lowest Mel filter frequency.")
    parser.add_argument("--fmax", type=float, default=None, help="Highest Mel filter frequency.")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "mfcc_manual"),
        help="Directory for plots and compressed feature arrays.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Prefix for output files. Defaults to the input filename stem.",
    )
    return parser.parse_args()


def build_figure(results: dict, output_path: Path) -> None:
    timestamps = results["timestamps_seconds"]
    frequencies = results["frequencies_hz"]
    quefrency = results["quefrency_seconds"] * 1000.0
    power_spectrum = np.maximum(results["power_spectrum"], 1e-10)

    figure, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    axes[0].imshow(
        10.0 * np.log10(power_spectrum.T),
        aspect="auto",
        origin="lower",
        extent=[timestamps[0], timestamps[-1], frequencies[0], frequencies[-1]],
        cmap="magma",
    )
    axes[0].set_title("Manual Short-Time Power Spectrum")
    axes[0].set_ylabel("Frequency (Hz)")

    axes[1].imshow(
        results["log_mel_energies"].T,
        aspect="auto",
        origin="lower",
        extent=[timestamps[0], timestamps[-1], 1, results["log_mel_energies"].shape[1]],
        cmap="viridis",
    )
    axes[1].set_title("Log Mel Filterbank Energies")
    axes[1].set_ylabel("Mel Filter Index")

    representative_index = int(np.argmax(np.mean(np.abs(results["windowed_frames"]), axis=1)))
    axes[2].plot(quefrency[:160], results["cepstrum"][representative_index, :160], linewidth=1.5)
    axes[2].set_title("Representative Real Cepstrum")
    axes[2].set_xlabel("Quefrency (ms)")
    axes[2].set_ylabel("Amplitude")
    axes[2].grid(alpha=0.3)

    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    audio_path = Path(args.audio_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = args.output_prefix or audio_path.stem

    signal, sample_rate = load_audio(audio_path, target_sr=args.sample_rate)
    config = MFCCConfig(
        sample_rate=sample_rate,
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        n_fft=args.n_fft,
        n_mels=args.n_mels,
        n_ceps=args.n_ceps,
        pre_emphasis=args.pre_emphasis,
        window=args.window,
        fmin=args.fmin,
        fmax=args.fmax,
    )
    results = compute_manual_mfcc(signal, sample_rate, config)

    npz_path = output_dir / f"{output_prefix}_mfcc_features.npz"
    np.savez_compressed(npz_path, **results)

    summary = {
        "audio_path": str(audio_path),
        "output_features_path": str(npz_path),
        "num_frames": int(results["mfcc"].shape[0]),
        "num_coefficients": int(results["mfcc"].shape[1]),
        "sample_rate": sample_rate,
        "frame_length_samples": int(results["frame_length"]),
        "hop_length_samples": int(results["hop_length"]),
        "n_fft": int(results["n_fft"]),
        "config": results["config"],
    }
    summary_path = output_dir / f"{output_prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_path = output_dir / f"{output_prefix}_overview.png"
    build_figure(results, plot_path)

    print(json.dumps(summary, indent=2))
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
