from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from speech_utils.dsp import EPSILON, frame_signal, get_window, load_audio, select_high_energy_segment


WINDOWS = ("rectangular", "hamming", "hanning")
PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure spectral leakage and SNR for different window functions.")
    parser.add_argument("--audio-path", required=True, help="Input waveform path.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate.")
    parser.add_argument("--segment-start", type=float, default=None, help="Optional segment start time in seconds.")
    parser.add_argument("--segment-duration-ms", type=float, default=250.0, help="Analysis segment duration.")
    parser.add_argument("--frame-ms", type=float, default=30.0, help="Frame length for per-window analysis.")
    parser.add_argument("--hop-ms", type=float, default=10.0, help="Hop length for per-window analysis.")
    parser.add_argument("--n-fft", type=int, default=1024, help="FFT size for analysis.")
    parser.add_argument("--peak-bins", type=int, default=5, help="Number of dominant bins treated as signal.")
    parser.add_argument("--main-lobe-bins", type=int, default=2, help="Bins kept around each dominant peak.")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "leakage_snr"),
        help="Directory for tables and plots.",
    )
    return parser.parse_args()


def extract_segment(signal: np.ndarray, sample_rate: int, start_sec: float | None, duration_ms: float) -> tuple[np.ndarray, float, float]:
    segment_length = int(round(sample_rate * duration_ms / 1000.0))
    if start_sec is None:
        return select_high_energy_segment(signal, sample_rate, duration_ms=duration_ms)
    start_index = max(0, int(round(start_sec * sample_rate)))
    end_index = min(signal.size, start_index + segment_length)
    return signal[start_index:end_index].copy(), start_index / sample_rate, end_index / sample_rate


def analyze_window(
    segment: np.ndarray,
    sample_rate: int,
    window_name: str,
    frame_ms: float,
    hop_ms: float,
    n_fft: int,
    peak_bins: int,
    main_lobe_bins: int,
) -> dict[str, np.ndarray | float | str]:
    frame_length = int(round(sample_rate * frame_ms / 1000.0))
    hop_length = int(round(sample_rate * hop_ms / 1000.0))
    frames = frame_signal(segment, frame_length=frame_length, hop_length=hop_length)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    representative_index = int(np.argmax(rms))

    window_values = get_window(window_name, frame_length)
    representative_frame = frames[representative_index] * window_values
    power_spectrum = np.abs(np.fft.rfft(representative_frame, n=n_fft)) ** 2

    leakage_values = []
    snr_values = []
    for frame in frames:
        spectrum = np.abs(np.fft.rfft(frame * window_values, n=n_fft)) ** 2
        if np.allclose(spectrum.sum(), 0.0):
            continue
        peak_indices = np.argpartition(spectrum, -peak_bins)[-peak_bins:]
        signal_mask = np.zeros_like(spectrum, dtype=bool)
        for peak_index in peak_indices:
            start = max(0, peak_index - main_lobe_bins)
            end = min(len(signal_mask), peak_index + main_lobe_bins + 1)
            signal_mask[start:end] = True
        signal_energy = float(np.sum(spectrum[signal_mask]))
        leakage_energy = float(np.sum(spectrum[~signal_mask]))
        leakage_values.append(100.0 * leakage_energy / max(signal_energy + leakage_energy, EPSILON))
        snr_values.append(10.0 * np.log10((signal_energy + EPSILON) / (leakage_energy + EPSILON)))

    return {
        "window": window_name,
        "frequencies_hz": np.fft.rfftfreq(n_fft, d=1.0 / sample_rate),
        "power_spectrum": power_spectrum,
        "leakage_percent": float(np.mean(leakage_values)) if leakage_values else float("nan"),
        "snr_db": float(np.mean(snr_values)) if snr_values else float("nan"),
    }


def save_plot(results: list[dict], start_sec: float, end_sec: float, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for result in results:
        axes[0].plot(
            result["frequencies_hz"],
            10.0 * np.log10(np.maximum(result["power_spectrum"], EPSILON)),
            label=result["window"].title(),
            linewidth=1.6,
        )
    axes[0].set_title(f"Representative Spectrum ({start_sec:.2f}s - {end_sec:.2f}s)")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Power (dB)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    x_positions = np.arange(len(results))
    leakage = [result["leakage_percent"] for result in results]
    snr = [result["snr_db"] for result in results]
    axes[1].bar(x_positions - 0.15, leakage, width=0.3, label="Leakage (%)")
    axes[1].bar(x_positions + 0.15, snr, width=0.3, label="SNR (dB)")
    axes[1].set_xticks(x_positions, [result["window"].title() for result in results])
    axes[1].set_title("Window Comparison")
    axes[1].grid(alpha=0.3, axis="y")
    axes[1].legend()

    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    audio_path = Path(args.audio_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    signal, sample_rate = load_audio(audio_path, target_sr=args.sample_rate)
    segment, start_sec, end_sec = extract_segment(signal, sample_rate, args.segment_start, args.segment_duration_ms)
    results = [
        analyze_window(
            segment=segment,
            sample_rate=sample_rate,
            window_name=window_name,
            frame_ms=args.frame_ms,
            hop_ms=args.hop_ms,
            n_fft=args.n_fft,
            peak_bins=args.peak_bins,
            main_lobe_bins=args.main_lobe_bins,
        )
        for window_name in WINDOWS
    ]

    csv_path = output_dir / f"{audio_path.stem}_window_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["window", "leakage_percent", "snr_db"])
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "window": result["window"],
                    "leakage_percent": f"{result['leakage_percent']:.4f}",
                    "snr_db": f"{result['snr_db']:.4f}",
                }
            )

    json_path = output_dir / f"{audio_path.stem}_window_metrics.json"
    serializable = {
        "audio_path": str(audio_path),
        "segment_start_sec": round(start_sec, 6),
        "segment_end_sec": round(end_sec, 6),
        "results": [
            {
                "window": result["window"],
                "leakage_percent": result["leakage_percent"],
                "snr_db": result["snr_db"],
            }
            for result in results
        ],
    }
    json_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    plot_path = output_dir / f"{audio_path.stem}_window_comparison.png"
    save_plot(results, start_sec, end_sec, plot_path)
    print(json.dumps(serializable, indent=2))
    print(f"Saved comparison plot to {plot_path}")


if __name__ == "__main__":
    main()
