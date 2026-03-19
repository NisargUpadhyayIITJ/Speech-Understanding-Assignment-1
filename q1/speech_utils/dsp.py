from __future__ import annotations

from dataclasses import asdict, dataclass
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import medfilt, resample_poly


EPSILON = 1e-10


@dataclass(slots=True)
class MFCCConfig:
    sample_rate: int = 16000
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    n_fft: int = 512
    n_mels: int = 40
    n_ceps: int = 13
    pre_emphasis: float = 0.97
    window: str = "hamming"
    fmin: float = 0.0
    fmax: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_audio(path: str | Path, target_sr: int | None = None, mono: bool = True) -> tuple[np.ndarray, int]:
    path = Path(path)
    audio, sample_rate = sf.read(path, always_2d=False)
    if audio.ndim > 1 and mono:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)
    if target_sr is not None and sample_rate != target_sr:
        divisor = gcd(int(sample_rate), int(target_sr))
        audio = resample_poly(audio, target_sr // divisor, sample_rate // divisor).astype(np.float32)
        sample_rate = target_sr
    return audio, int(sample_rate)


def pre_emphasis(signal: np.ndarray, coefficient: float = 0.97) -> np.ndarray:
    if signal.size == 0:
        return signal
    emphasized = np.empty_like(signal, dtype=np.float32)
    emphasized[0] = signal[0]
    emphasized[1:] = signal[1:] - coefficient * signal[:-1]
    return emphasized


def frame_signal(signal: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length and hop_length must be positive")
    if signal.ndim != 1:
        raise ValueError("frame_signal expects a mono waveform")

    if signal.size <= frame_length:
        padded = np.pad(signal, (0, frame_length - signal.size))
        return padded[np.newaxis, :].copy()

    num_frames = 1 + int(np.ceil((signal.size - frame_length) / hop_length))
    padded_length = (num_frames - 1) * hop_length + frame_length
    padded = np.pad(signal, (0, max(0, padded_length - signal.size)))
    strides = (padded.strides[0] * hop_length, padded.strides[0])
    frames = np.lib.stride_tricks.as_strided(
        padded,
        shape=(num_frames, frame_length),
        strides=strides,
        writeable=False,
    )
    return frames.copy()


def get_window(name: str, size: int) -> np.ndarray:
    normalized = name.lower()
    if normalized in {"rect", "rectangular", "boxcar"}:
        return np.ones(size, dtype=np.float32)
    if normalized in {"hamming"}:
        return np.hamming(size).astype(np.float32)
    if normalized in {"hanning", "hann"}:
        return np.hanning(size).astype(np.float32)
    raise ValueError(f"Unsupported window type: {name}")


def hz_to_mel(frequency_hz: np.ndarray | float) -> np.ndarray:
    frequency_hz = np.asarray(frequency_hz, dtype=np.float64)
    return 2595.0 * np.log10(1.0 + frequency_hz / 700.0)


def mel_to_hz(mel_value: np.ndarray | float) -> np.ndarray:
    mel_value = np.asarray(mel_value, dtype=np.float64)
    return 700.0 * (10 ** (mel_value / 2595.0) - 1.0)


def mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    if fmax is None:
        fmax = sample_rate / 2
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)

    for index in range(n_mels):
        start = bins[index]
        center = bins[index + 1]
        stop = bins[index + 2]
        if center == start:
            center += 1
        if stop == center:
            stop += 1
        for bin_index in range(start, min(center, filters.shape[1])):
            filters[index, bin_index] = (bin_index - start) / max(center - start, 1)
        for bin_index in range(center, min(stop, filters.shape[1])):
            filters[index, bin_index] = (stop - bin_index) / max(stop - center, 1)

    energy = filters.sum(axis=1, keepdims=True)
    return filters / np.maximum(energy, EPSILON)


def manual_dct_type_ii(values: np.ndarray, n_ceps: int) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError("manual_dct_type_ii expects a 2D array")
    n_features = values.shape[1]
    basis_rows = []
    sample_positions = np.arange(n_features, dtype=np.float64) + 0.5
    for coefficient_index in range(n_ceps):
        row = np.cos(np.pi * coefficient_index * sample_positions / n_features)
        scale = np.sqrt(1.0 / n_features) if coefficient_index == 0 else np.sqrt(2.0 / n_features)
        basis_rows.append(row * scale)
    basis = np.stack(basis_rows, axis=0)
    return values @ basis.T


def log_power(values: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(values, EPSILON))


def real_cepstrum_from_frames(windowed_frames: np.ndarray, n_fft: int) -> np.ndarray:
    spectrum = np.fft.rfft(windowed_frames, n=n_fft, axis=1)
    log_magnitude = np.log(np.maximum(np.abs(spectrum), EPSILON))
    cepstrum = np.fft.irfft(log_magnitude, n=n_fft, axis=1)
    return np.real(cepstrum).astype(np.float32)


def frame_timestamps(num_frames: int, hop_length: int, frame_length: int, sample_rate: int) -> np.ndarray:
    centers = np.arange(num_frames) * hop_length + frame_length / 2.0
    return centers / float(sample_rate)


def compute_manual_mfcc(signal: np.ndarray, sample_rate: int, config: MFCCConfig) -> dict[str, np.ndarray | dict[str, Any]]:
    frame_length = int(round(sample_rate * config.frame_ms / 1000.0))
    hop_length = int(round(sample_rate * config.hop_ms / 1000.0))
    n_fft = max(config.n_fft, frame_length)

    emphasized = pre_emphasis(signal, config.pre_emphasis)
    frames = frame_signal(emphasized, frame_length=frame_length, hop_length=hop_length)
    window = get_window(config.window, frame_length)
    windowed_frames = frames * window[None, :]
    spectrum = np.fft.rfft(windowed_frames, n=n_fft, axis=1)
    magnitude = np.abs(spectrum)
    power = (magnitude ** 2) / n_fft

    filters = mel_filterbank(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax,
    )
    mel_energies = power @ filters.T
    log_mel = log_power(mel_energies)
    mfcc = manual_dct_type_ii(log_mel, n_ceps=config.n_ceps)
    cepstrum = real_cepstrum_from_frames(windowed_frames=windowed_frames, n_fft=n_fft)
    frequencies_hz = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    quefrency_seconds = np.arange(n_fft) / float(sample_rate)
    timestamps_seconds = frame_timestamps(len(frames), hop_length, frame_length, sample_rate)

    return {
        "config": config.to_dict(),
        "signal": signal.astype(np.float32),
        "emphasized_signal": emphasized,
        "frames": frames.astype(np.float32),
        "window": window.astype(np.float32),
        "windowed_frames": windowed_frames.astype(np.float32),
        "magnitude_spectrum": magnitude.astype(np.float32),
        "power_spectrum": power.astype(np.float32),
        "mel_filterbank": filters.astype(np.float32),
        "mel_energies": mel_energies.astype(np.float32),
        "log_mel_energies": log_mel.astype(np.float32),
        "mfcc": mfcc.astype(np.float32),
        "cepstrum": cepstrum,
        "frequencies_hz": frequencies_hz.astype(np.float32),
        "quefrency_seconds": quefrency_seconds.astype(np.float32),
        "timestamps_seconds": timestamps_seconds.astype(np.float32),
        "frame_length": np.array(frame_length),
        "hop_length": np.array(hop_length),
        "n_fft": np.array(n_fft),
    }


def _safe_zscore(values: np.ndarray) -> np.ndarray:
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std < EPSILON:
        return np.zeros_like(values)
    return (values - mean) / std


def compute_voicing_features(
    signal: np.ndarray,
    sample_rate: int,
    frame_ms: float = 30.0,
    hop_ms: float = 10.0,
    n_fft: int | None = None,
    window: str = "hamming",
    pre_emphasis_coefficient: float = 0.97,
    low_quefrency_ms: tuple[float, float] = (0.5, 2.5),
    high_quefrency_ms: tuple[float, float] = (2.5, 16.0),
    pitch_quefrency_ms: tuple[float, float] = (2.5, 12.5),
) -> dict[str, np.ndarray]:
    frame_length = int(round(sample_rate * frame_ms / 1000.0))
    hop_length = int(round(sample_rate * hop_ms / 1000.0))
    fft_size = max(frame_length, int(n_fft or 0) or 512)

    emphasized = pre_emphasis(signal, pre_emphasis_coefficient)
    frames = frame_signal(emphasized, frame_length=frame_length, hop_length=hop_length)
    window_values = get_window(window, frame_length)
    windowed_frames = frames * window_values[None, :]
    cepstrum = real_cepstrum_from_frames(windowed_frames, n_fft=fft_size)
    quefrency_ms = (np.arange(fft_size) / float(sample_rate)) * 1000.0

    low_mask = (quefrency_ms >= low_quefrency_ms[0]) & (quefrency_ms < low_quefrency_ms[1])
    high_mask = (quefrency_ms >= high_quefrency_ms[0]) & (quefrency_ms <= high_quefrency_ms[1])
    pitch_mask = (quefrency_ms >= pitch_quefrency_ms[0]) & (quefrency_ms <= pitch_quefrency_ms[1])
    if not np.any(low_mask) or not np.any(high_mask) or not np.any(pitch_mask):
        raise ValueError("Quefrency masks are empty. Increase n_fft or widen the ranges.")

    absolute_cepstrum = np.abs(cepstrum)
    low_energy = absolute_cepstrum[:, low_mask].sum(axis=1)
    high_energy = absolute_cepstrum[:, high_mask].sum(axis=1)

    pitch_region = cepstrum[:, pitch_mask]
    pitch_peak_index = np.argmax(pitch_region, axis=1)
    pitch_peak_value = pitch_region[np.arange(len(pitch_region)), pitch_peak_index]
    pitch_period_ms = quefrency_ms[pitch_mask][pitch_peak_index]

    rms_energy = np.sqrt(np.mean(windowed_frames**2, axis=1))
    zero_crossings = np.mean(np.abs(np.diff(np.signbit(windowed_frames), axis=1)), axis=1)
    periodicity_ratio = (pitch_peak_value + EPSILON) / (low_energy + EPSILON)
    voicing_score = (
        0.65 * _safe_zscore(np.log(np.maximum(periodicity_ratio, EPSILON)))
        + 0.25 * _safe_zscore(np.log(np.maximum(high_energy, EPSILON)))
        + 0.10 * _safe_zscore(np.log(np.maximum(rms_energy, EPSILON)))
    )

    timestamps = frame_timestamps(len(frames), hop_length, frame_length, sample_rate)
    return {
        "frames": frames.astype(np.float32),
        "windowed_frames": windowed_frames.astype(np.float32),
        "cepstrum": cepstrum.astype(np.float32),
        "quefrency_ms": quefrency_ms.astype(np.float32),
        "timestamps_seconds": timestamps.astype(np.float32),
        "low_energy": low_energy.astype(np.float32),
        "high_energy": high_energy.astype(np.float32),
        "pitch_peak_value": pitch_peak_value.astype(np.float32),
        "pitch_period_ms": pitch_period_ms.astype(np.float32),
        "periodicity_ratio": periodicity_ratio.astype(np.float32),
        "rms_energy": rms_energy.astype(np.float32),
        "zero_crossings": zero_crossings.astype(np.float32),
        "voicing_score": voicing_score.astype(np.float32),
        "frame_length": np.array(frame_length),
        "hop_length": np.array(hop_length),
    }


def smooth_binary_mask(mask: np.ndarray, min_run_frames: int) -> np.ndarray:
    if min_run_frames <= 1 or mask.size == 0:
        return mask.astype(bool)
    smoothed = mask.astype(bool).copy()
    start = 0
    while start < len(smoothed):
        end = start + 1
        while end < len(smoothed) and smoothed[end] == smoothed[start]:
            end += 1
        if end - start < min_run_frames:
            left_value = smoothed[start - 1] if start > 0 else None
            right_value = smoothed[end] if end < len(smoothed) else None
            replacement = left_value if left_value is not None else right_value
            if replacement is not None:
                smoothed[start:end] = replacement
        start = end
    return smoothed


def detect_voiced_frames(
    voicing_score: np.ndarray,
    threshold: float | None = None,
    smoothing_frames: int = 5,
) -> tuple[np.ndarray, float]:
    decision_threshold = float(np.median(voicing_score) if threshold is None else threshold)
    voiced_mask = voicing_score >= decision_threshold
    if smoothing_frames > 1:
        kernel_size = smoothing_frames if smoothing_frames % 2 == 1 else smoothing_frames + 1
        voiced_mask = medfilt(voiced_mask.astype(float), kernel_size=kernel_size) > 0.5
    voiced_mask = smooth_binary_mask(voiced_mask.astype(bool), min_run_frames=max(2, smoothing_frames // 2))
    return voiced_mask.astype(bool), decision_threshold


def frames_to_segments(
    frame_mask: np.ndarray,
    sample_rate: int,
    frame_length: int,
    hop_length: int,
    signal_length: int,
    positive_label: str = "voiced",
    negative_label: str = "unvoiced",
) -> list[dict[str, float | str]]:
    if frame_mask.size == 0:
        return []
    segments: list[dict[str, float | str]] = []
    start_index = 0
    current_value = bool(frame_mask[0])
    for frame_index in range(1, len(frame_mask) + 1):
        boundary = frame_index == len(frame_mask) or bool(frame_mask[frame_index]) != current_value
        if not boundary:
            continue
        start_seconds = start_index * hop_length / float(sample_rate)
        end_sample = min(signal_length, (frame_index - 1) * hop_length + frame_length)
        end_seconds = end_sample / float(sample_rate)
        label = positive_label if current_value else negative_label
        segments.append(
            {
                "label": label,
                "start_sec": round(start_seconds, 6),
                "end_sec": round(end_seconds, 6),
                "duration_sec": round(end_seconds - start_seconds, 6),
            }
        )
        if frame_index < len(frame_mask):
            start_index = frame_index
            current_value = bool(frame_mask[frame_index])
    return segments


def select_high_energy_segment(
    signal: np.ndarray,
    sample_rate: int,
    duration_ms: float = 250.0,
    hop_ms: float = 10.0,
) -> tuple[np.ndarray, float, float]:
    segment_length = max(1, int(round(sample_rate * duration_ms / 1000.0)))
    hop_length = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    if signal.size <= segment_length:
        return signal.copy(), 0.0, signal.size / float(sample_rate)

    best_start = 0
    best_energy = -np.inf
    for start in range(0, signal.size - segment_length + 1, hop_length):
        chunk = signal[start : start + segment_length]
        energy = float(np.mean(chunk**2))
        if energy > best_energy:
            best_energy = energy
            best_start = start
    start_sec = best_start / float(sample_rate)
    end_sec = (best_start + segment_length) / float(sample_rate)
    return signal[best_start : best_start + segment_length].copy(), start_sec, end_sec


def nearest_boundary_rmse(manual_boundaries: list[float], reference_boundaries: list[float]) -> float:
    if not manual_boundaries or not reference_boundaries:
        return float("nan")
    manual = np.asarray(manual_boundaries, dtype=np.float64)
    reference = np.asarray(reference_boundaries, dtype=np.float64)
    squared_errors = []
    for boundary in manual:
        nearest_error = np.min((reference - boundary) ** 2)
        squared_errors.append(nearest_error)
    return float(np.sqrt(np.mean(squared_errors)))
