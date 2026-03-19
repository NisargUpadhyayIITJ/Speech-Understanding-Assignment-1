from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from speech_utils import compute_voicing_features, detect_voiced_frames, frames_to_segments, load_audio


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cepstrum-driven voiced/unvoiced boundary detection.")
    parser.add_argument("--audio-path", required=True, help="Input waveform path.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate.")
    parser.add_argument("--frame-ms", type=float, default=30.0, help="Frame length for cepstral analysis.")
    parser.add_argument("--hop-ms", type=float, default=10.0, help="Hop size for cepstral analysis.")
    parser.add_argument("--n-fft", type=int, default=512, help="FFT size used for cepstrum estimation.")
    parser.add_argument("--window", default="hamming", choices=["rectangular", "hamming", "hanning", "hann"])
    parser.add_argument("--pre-emphasis", type=float, default=0.97, help="Pre-emphasis coefficient.")
    parser.add_argument("--threshold", type=float, default=None, help="Manual voicing threshold; defaults to median score.")
    parser.add_argument("--smoothing-frames", type=int, default=5, help="Median filter width for frame decisions.")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "voiced_unvoiced"),
        help="Directory for JSON and plots.",
    )
    return parser.parse_args()


def save_plot(
    signal: np.ndarray,
    sample_rate: int,
    features: dict[str, np.ndarray],
    voiced_mask: np.ndarray,
    threshold: float,
    segments: list[dict[str, float | str]],
    output_path: Path,
) -> None:
    times = np.arange(signal.size) / float(sample_rate)
    figure, axes = plt.subplots(3, 1, figsize=(13, 9), constrained_layout=True)

    axes[0].plot(times, signal, color="black", linewidth=0.8)
    for segment in segments:
        color = "#3ba55c" if segment["label"] == "voiced" else "#f2c14e"
        axes[0].axvspan(segment["start_sec"], segment["end_sec"], color=color, alpha=0.22)
    axes[0].set_title("Waveform with Voiced / Unvoiced Segments")
    axes[0].set_ylabel("Amplitude")

    frame_times = features["timestamps_seconds"]
    axes[1].plot(frame_times, features["low_energy"], label="Low-quefrency energy", linewidth=1.4)
    axes[1].plot(frame_times, features["high_energy"], label="High-quefrency energy", linewidth=1.4)
    axes[1].set_title("Cepstral Energy Bands")
    axes[1].set_ylabel("Energy")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(frame_times, features["voicing_score"], color="#2c7fb8", linewidth=1.6, label="Voicing score")
    axes[2].axhline(threshold, color="#d95f02", linestyle="--", linewidth=1.4, label=f"Threshold = {threshold:.3f}")
    axes[2].fill_between(frame_times, voiced_mask.astype(float), alpha=0.15, color="#3ba55c", step="mid", label="Voiced mask")
    axes[2].set_title("Decision Function")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Score")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    audio_path = Path(args.audio_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    signal, sample_rate = load_audio(audio_path, target_sr=args.sample_rate)
    features = compute_voicing_features(
        signal=signal,
        sample_rate=sample_rate,
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        n_fft=args.n_fft,
        window=args.window,
        pre_emphasis_coefficient=args.pre_emphasis,
    )
    voiced_mask, threshold = detect_voiced_frames(
        features["voicing_score"],
        threshold=args.threshold,
        smoothing_frames=args.smoothing_frames,
    )
    segments = frames_to_segments(
        frame_mask=voiced_mask,
        sample_rate=sample_rate,
        frame_length=int(features["frame_length"]),
        hop_length=int(features["hop_length"]),
        signal_length=len(signal),
    )

    summary = {
        "audio_path": str(audio_path),
        "sample_rate": sample_rate,
        "threshold": threshold,
        "voiced_ratio": float(np.mean(voiced_mask)),
        "segments": segments,
        "boundaries_sec": [segment["end_sec"] for segment in segments[:-1]],
        "frame_level": {
            "timestamps_seconds": features["timestamps_seconds"].tolist(),
            "low_energy": features["low_energy"].tolist(),
            "high_energy": features["high_energy"].tolist(),
            "pitch_period_ms": features["pitch_period_ms"].tolist(),
            "voicing_score": features["voicing_score"].tolist(),
            "voiced_mask": voiced_mask.astype(int).tolist(),
        },
    }
    json_path = output_dir / f"{audio_path.stem}_voiced_unvoiced.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_path = output_dir / f"{audio_path.stem}_voiced_unvoiced.png"
    save_plot(signal, sample_rate, features, voiced_mask, threshold, segments, plot_path)

    print(json.dumps({"segments_path": str(json_path), "plot_path": str(plot_path), "threshold": threshold}, indent=2))


if __name__ == "__main__":
    main()
