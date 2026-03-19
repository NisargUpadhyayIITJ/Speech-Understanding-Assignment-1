from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


VOCAB = ["<blank>", " "] + list("abcdefghijklmnopqrstuvwxyz'")
TOKEN_TO_ID = {token: index for index, token in enumerate(VOCAB)}
ID_TO_TOKEN = {index: token for token, index in TOKEN_TO_ID.items()}
EPSILON = 1e-8


def encode_text(text: str) -> list[int]:
    return [TOKEN_TO_ID[char] for char in text if char in TOKEN_TO_ID]


def decode_tokens(token_ids: list[int]) -> str:
    tokens = []
    previous = None
    for token_id in token_ids:
        if token_id == 0 or token_id == previous:
            previous = token_id
            continue
        tokens.append(ID_TO_TOKEN[token_id])
        previous = token_id
    return "".join(tokens).strip()


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    dp = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=np.int32)
    dp[:, 0] = np.arange(len(ref_words) + 1)
    dp[0, :] = np.arange(len(hyp_words) + 1)
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + cost,
            )
    return float(dp[-1, -1] / max(len(ref_words), 1))


def collate_ctc(batch: list[dict[str, Any]]) -> dict[str, Any]:
    waveforms = pad_sequence([item["waveform"] for item in batch], batch_first=True)
    input_lengths = torch.tensor([len(item["waveform"]) for item in batch], dtype=torch.long)
    targets = [torch.tensor(item["target_ids"], dtype=torch.long) for item in batch]
    target_lengths = torch.tensor([len(target) for target in targets], dtype=torch.long)
    flattened_targets = torch.cat(targets) if targets else torch.empty(0, dtype=torch.long)
    return {
        "waveforms": waveforms,
        "input_lengths": input_lengths,
        "targets": flattened_targets,
        "target_lengths": target_lengths,
        "texts": [item["text"] for item in batch],
        "gender": [item["gender"] for item in batch],
        "sample_id": [item["sample_id"] for item in batch],
    }


class SpeechCTCDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], root: Path) -> None:
        self.rows = rows
        self.root = root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import soundfile as sf

        row = self.rows[index]
        waveform, sample_rate = sf.read(self.root / row["audio_path"])
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        return {
            "waveform": torch.tensor(waveform, dtype=torch.float32),
            "target_ids": encode_text(row["sentence"]),
            "text": row["sentence"],
            "gender": row["gender"],
            "sample_id": row["sample_id"],
        }


class LogMelFrontend(nn.Module):
    def __init__(self, sample_rate: int = 16000, n_fft: int = 400, hop_length: int = 160, n_mels: int = 40) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.register_buffer("mel_filter", torch.tensor(self._mel_filterbank(sample_rate, n_fft, n_mels)), persistent=False)

    @staticmethod
    def _mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
        def hz_to_mel(frequency_hz: np.ndarray | float) -> np.ndarray:
            frequency_hz = np.asarray(frequency_hz, dtype=np.float64)
            return 2595.0 * np.log10(1.0 + frequency_hz / 700.0)

        def mel_to_hz(mel_value: np.ndarray | float) -> np.ndarray:
            mel_value = np.asarray(mel_value, dtype=np.float64)
            return 700.0 * (10 ** (mel_value / 2595.0) - 1.0)

        mel_points = np.linspace(hz_to_mel(0.0), hz_to_mel(sample_rate / 2.0), n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
        filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
        for index in range(n_mels):
            start, center, stop = bins[index], bins[index + 1], bins[index + 2]
            center = max(center, start + 1)
            stop = max(stop, center + 1)
            for j in range(start, min(center, filters.shape[1])):
                filters[index, j] = (j - start) / max(center - start, 1)
            for j in range(center, min(stop, filters.shape[1])):
                filters[index, j] = (stop - j) / max(stop - center, 1)
        filters /= np.maximum(filters.sum(axis=1, keepdims=True), EPSILON)
        return filters

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        spectrum = torch.stft(
            waveforms,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            return_complex=True,
            center=True,
        )
        power = spectrum.abs().pow(2.0)
        mel = torch.matmul(self.mel_filter.to(power.dtype), power)
        log_mel = torch.log(torch.clamp(mel, min=EPSILON))
        return log_mel.transpose(1, 2)


class TinyCTCModel(nn.Module):
    def __init__(self, vocab_size: int, n_mels: int = 40, hidden_size: int = 128) -> None:
        super().__init__()
        self.frontend = LogMelFrontend(n_mels=n_mels)
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, hidden_size, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.encoder = nn.GRU(hidden_size, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, waveforms: torch.Tensor, input_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.frontend(waveforms)
        features = self.conv(features.transpose(1, 2)).transpose(1, 2)
        outputs, _ = self.encoder(features)
        logits = self.classifier(outputs)
        frame_lengths = torch.div(input_lengths + self.frontend.hop_length - 1, self.frontend.hop_length, rounding_mode="floor")
        frame_lengths = torch.clamp(frame_lengths, max=logits.shape[1])
        return logits, frame_lengths


def greedy_decode(logits: torch.Tensor) -> list[str]:
    token_ids = logits.argmax(dim=-1).cpu().tolist()
    return [decode_tokens(row) for row in token_ids]
