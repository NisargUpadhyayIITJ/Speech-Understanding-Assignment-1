from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


EPSILON = 1e-8


def signal_to_noise_ratio(reference: np.ndarray, transformed: np.ndarray) -> float:
    reference = reference[: min(len(reference), len(transformed))]
    transformed = transformed[: len(reference)]
    noise = reference - transformed
    return float(10.0 * np.log10((np.mean(reference**2) + EPSILON) / (np.mean(noise**2) + EPSILON)))


def log_spectral_distance(reference: np.ndarray, transformed: np.ndarray, sample_rate: int, n_fft: int = 512) -> float:
    def stft_magnitude(signal: np.ndarray) -> np.ndarray:
        frames = []
        hop = n_fft // 4
        for start in range(0, max(1, len(signal) - n_fft + 1), hop):
            frame = signal[start : start + n_fft]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            frames.append(np.abs(np.fft.rfft(frame * np.hanning(n_fft), n=n_fft)))
        return np.stack(frames, axis=0)

    min_len = min(len(reference), len(transformed))
    reference_mag = stft_magnitude(reference[:min_len])
    transformed_mag = stft_magnitude(transformed[:min_len])
    distance = np.sqrt(np.mean((np.log(reference_mag + EPSILON) - np.log(transformed_mag + EPSILON)) ** 2))
    return float(distance)


def acceptability_proxy(reference_path: str | Path, transformed_path: str | Path) -> dict[str, float]:
    reference, reference_sr = sf.read(reference_path)
    transformed, transformed_sr = sf.read(transformed_path)
    if reference.ndim > 1:
        reference = reference.mean(axis=1)
    if transformed.ndim > 1:
        transformed = transformed.mean(axis=1)
    if reference_sr != transformed_sr:
        raise ValueError("Sample rates must match for proxy evaluation")
    return {
        "snr_db": signal_to_noise_ratio(reference.astype(np.float32), transformed.astype(np.float32)),
        "log_spectral_distance": log_spectral_distance(reference.astype(np.float32), transformed.astype(np.float32), reference_sr),
    }
