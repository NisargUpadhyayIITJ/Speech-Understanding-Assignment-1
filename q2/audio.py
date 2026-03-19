from __future__ import annotations

from math import gcd

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt, resample_poly


EPSILON = 1e-8


def pre_emphasis(signal: np.ndarray, coefficient: float = 0.97) -> np.ndarray:
    if signal.size == 0:
        return signal.astype(np.float32)
    emphasized = np.empty_like(signal, dtype=np.float32)
    emphasized[0] = signal[0]
    emphasized[1:] = signal[1:] - coefficient * signal[:-1]
    return emphasized


def resample_audio(signal: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    if original_sr == target_sr:
        return signal.astype(np.float32)
    divisor = gcd(int(original_sr), int(target_sr))
    return resample_poly(signal, target_sr // divisor, original_sr // divisor).astype(np.float32)


def normalize_audio(signal: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(signal))) if signal.size else 1.0
    if peak < EPSILON:
        return signal.astype(np.float32)
    return (signal / peak).astype(np.float32)


def crop_or_pad(signal: np.ndarray, target_length: int, rng: np.random.Generator | None, random_crop: bool) -> np.ndarray:
    if signal.size >= target_length:
        if random_crop and rng is not None:
            start = int(rng.integers(0, signal.size - target_length + 1))
        else:
            start = max(0, (signal.size - target_length) // 2)
        return signal[start : start + target_length].astype(np.float32)

    pad_total = target_length - signal.size
    if random_crop and rng is not None:
        left = int(rng.integers(0, pad_total + 1))
    else:
        left = pad_total // 2
    right = pad_total - left
    return np.pad(signal, (left, right)).astype(np.float32)


def _apply_butter_filter(signal: np.ndarray, sample_rate: int, cutoff_hz: float | tuple[float, float], btype: str) -> np.ndarray:
    nyquist = sample_rate / 2.0
    normalized = np.asarray(cutoff_hz, dtype=np.float64) / nyquist
    b, a = butter(4, normalized, btype=btype)
    return filtfilt(b, a, signal).astype(np.float32)


def _generate_reverb_impulse(sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    length = int(sample_rate * 0.18)
    times = np.linspace(0.0, 1.0, length, endpoint=False)
    decay = np.exp(-times * rng.uniform(6.0, 12.0))
    impulse = decay.copy()
    for _ in range(3):
        index = int(rng.integers(1, length))
        impulse[index] += rng.uniform(0.1, 0.35)
    impulse[0] = 1.0
    impulse /= np.sum(np.abs(impulse)) + EPSILON
    return impulse.astype(np.float32)


def build_environment_catalog() -> list[dict[str, int | str]]:
    return [
        {"id": 0, "name": "clean"},
        {"id": 1, "name": "white_noise"},
        {"id": 2, "name": "lowpass"},
        {"id": 3, "name": "telephone"},
        {"id": 4, "name": "reverb"},
    ]


def apply_environment(signal: np.ndarray, sample_rate: int, environment_name: str, rng: np.random.Generator) -> np.ndarray:
    processed = signal.astype(np.float32)
    if environment_name == "clean":
        return processed
    if environment_name == "white_noise":
        snr_db = float(rng.uniform(5.0, 20.0))
        power = np.mean(processed**2) + EPSILON
        noise_power = power / (10 ** (snr_db / 10.0))
        noise = rng.normal(0.0, np.sqrt(noise_power), size=processed.shape).astype(np.float32)
        return normalize_audio(processed + noise)
    if environment_name == "lowpass":
        cutoff = float(rng.uniform(1200.0, 3000.0))
        return normalize_audio(_apply_butter_filter(processed, sample_rate, cutoff, btype="lowpass"))
    if environment_name == "telephone":
        band = (300.0, float(rng.uniform(2600.0, 3400.0)))
        return normalize_audio(_apply_butter_filter(processed, sample_rate, band, btype="bandpass"))
    if environment_name == "reverb":
        impulse = _generate_reverb_impulse(sample_rate, rng)
        reverberant = np.convolve(processed, impulse, mode="full")[: processed.size]
        return normalize_audio(reverberant)
    raise ValueError(f"Unknown environment: {environment_name}")


def hz_to_mel(frequency_hz: np.ndarray | float) -> np.ndarray:
    frequency_hz = np.asarray(frequency_hz, dtype=np.float64)
    return 2595.0 * np.log10(1.0 + frequency_hz / 700.0)


def mel_to_hz(mel_value: np.ndarray | float) -> np.ndarray:
    mel_value = np.asarray(mel_value, dtype=np.float64)
    return 700.0 * (10 ** (mel_value / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
    if fmax is None:
        fmax = sample_rate / 2.0
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for index in range(n_mels):
        start, center, stop = bins[index], bins[index + 1], bins[index + 2]
        center = max(center, start + 1)
        stop = max(stop, center + 1)
        for bin_index in range(start, min(center, filters.shape[1])):
            filters[index, bin_index] = (bin_index - start) / max(center - start, 1)
        for bin_index in range(center, min(stop, filters.shape[1])):
            filters[index, bin_index] = (stop - bin_index) / max(stop - center, 1)
    filters /= np.maximum(filters.sum(axis=1, keepdims=True), EPSILON)
    return filters


class LogMelFrontend(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 40,
        pre_emphasis_coefficient: float = 0.97,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.pre_emphasis_coefficient = pre_emphasis_coefficient
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        mel = mel_filterbank(sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels)
        self.register_buffer("mel_filterbank", torch.tensor(mel), persistent=False)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        if waveforms.ndim != 2:
            raise ValueError("Expected waveforms with shape [batch, time]")
        emphasized = waveforms.clone()
        emphasized[:, 1:] = waveforms[:, 1:] - self.pre_emphasis_coefficient * waveforms[:, :-1]
        spectrum = torch.stft(
            emphasized,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            return_complex=True,
        )
        power = spectrum.abs().pow(2.0)
        mel = torch.matmul(self.mel_filterbank.to(power.dtype), power)
        log_mel = torch.log(torch.clamp(mel, min=EPSILON))
        return log_mel.transpose(1, 2).unsqueeze(1)
