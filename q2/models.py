from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .audio import LogMelFrontend


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, lambda_value: float) -> torch.Tensor:
        ctx.lambda_value = lambda_value
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambda_value * grad_output, None


def gradient_reverse(inputs: torch.Tensor, lambda_value: float) -> torch.Tensor:
    return GradientReversalFunction.apply(inputs, lambda_value)


def mean_absolute_correlation(first: torch.Tensor, second: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    first = (first - first.mean(dim=0, keepdim=True)) / (first.std(dim=0, keepdim=True) + eps)
    second = (second - second.mean(dim=0, keepdim=True)) / (second.std(dim=0, keepdim=True) + eps)
    correlation = torch.matmul(first.transpose(0, 1), second) / max(first.shape[0], 1)
    return correlation.abs().mean()


class SmallSpeakerEncoder(nn.Module):
    def __init__(self, sample_rate: int, embedding_dim: int, n_mels: int, n_fft: int, hop_length: int) -> None:
        super().__init__()
        self.frontend = LogMelFrontend(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Linear(128, embedding_dim)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        features = self.frontend(waveforms)
        encoded = self.conv_blocks(features).flatten(start_dim=1)
        embeddings = self.projection(encoded)
        return F.normalize(embeddings, dim=-1)


class BaselineSpeakerNet(nn.Module):
    def __init__(self, sample_rate: int, embedding_dim: int, n_mels: int, n_fft: int, hop_length: int, num_speakers: int) -> None:
        super().__init__()
        self.encoder = SmallSpeakerEncoder(sample_rate, embedding_dim, n_mels, n_fft, hop_length)
        self.classifier = nn.Linear(embedding_dim, num_speakers)

    def forward(self, waveforms: torch.Tensor) -> dict[str, torch.Tensor]:
        embeddings = self.encoder(waveforms)
        logits = self.classifier(embeddings)
        return {"embeddings": embeddings, "speaker_logits": logits}


@dataclass(slots=True)
class DisentangledOutputs:
    embeddings: torch.Tensor
    speaker_code: torch.Tensor
    environment_code: torch.Tensor
    reconstructed_embedding: torch.Tensor
    speaker_logits: torch.Tensor
    environment_logits: torch.Tensor
    adversarial_environment_logits: torch.Tensor


class DisentangledSpeakerNet(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        embedding_dim: int,
        n_mels: int,
        n_fft: int,
        hop_length: int,
        num_speakers: int,
        num_environments: int,
        speaker_code_dim: int,
        environment_code_dim: int,
        adversarial_lambda: float,
    ) -> None:
        super().__init__()
        self.encoder = SmallSpeakerEncoder(sample_rate, embedding_dim, n_mels, n_fft, hop_length)
        self.speaker_projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, speaker_code_dim),
        )
        self.environment_projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, environment_code_dim),
        )
        self.reconstructor = nn.Sequential(
            nn.Linear(speaker_code_dim + environment_code_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.speaker_classifier = nn.Linear(speaker_code_dim, num_speakers)
        self.environment_classifier = nn.Linear(environment_code_dim, num_environments)
        self.adversarial_environment_classifier = nn.Linear(speaker_code_dim, num_environments)
        self.adversarial_lambda = adversarial_lambda

    def forward(self, waveforms: torch.Tensor) -> DisentangledOutputs:
        embeddings = self.encoder(waveforms)
        speaker_code = F.normalize(self.speaker_projector(embeddings), dim=-1)
        environment_code = self.environment_projector(embeddings)
        reconstructed_embedding = self.reconstructor(torch.cat([speaker_code, environment_code], dim=-1))
        speaker_logits = self.speaker_classifier(speaker_code)
        environment_logits = self.environment_classifier(environment_code)
        adversarial_environment_logits = self.adversarial_environment_classifier(
            gradient_reverse(speaker_code, self.adversarial_lambda)
        )
        return DisentangledOutputs(
            embeddings=embeddings,
            speaker_code=speaker_code,
            environment_code=environment_code,
            reconstructed_embedding=reconstructed_embedding,
            speaker_logits=speaker_logits,
            environment_logits=environment_logits,
            adversarial_environment_logits=adversarial_environment_logits,
        )


def build_model(config: dict[str, Any], num_speakers: int, num_environments: int) -> nn.Module:
    common_kwargs = {
        "sample_rate": int(config["audio"]["sample_rate"]),
        "embedding_dim": int(config["model"]["embedding_dim"]),
        "n_mels": int(config["audio"]["n_mels"]),
        "n_fft": int(config["audio"]["n_fft"]),
        "hop_length": int(config["audio"]["hop_length"]),
    }
    experiment_type = config["experiment"]["type"]
    if experiment_type == "baseline":
        return BaselineSpeakerNet(num_speakers=num_speakers, **common_kwargs)
    return DisentangledSpeakerNet(
        num_speakers=num_speakers,
        num_environments=num_environments,
        speaker_code_dim=int(config["model"]["speaker_code_dim"]),
        environment_code_dim=int(config["model"]["environment_code_dim"]),
        adversarial_lambda=float(config["loss"]["adversarial_lambda"]),
        **common_kwargs,
    )
