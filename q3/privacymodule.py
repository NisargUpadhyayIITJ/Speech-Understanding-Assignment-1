from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DemographicPreset:
    frequency_scale: float
    high_shelf_gain: float
    energy_scale: float
    blend: float


PRESETS = {
    "male_old": DemographicPreset(frequency_scale=0.96, high_shelf_gain=-0.04, energy_scale=1.00, blend=0.22),
    "male_young": DemographicPreset(frequency_scale=0.99, high_shelf_gain=0.00, energy_scale=1.00, blend=0.18),
    "female_old": DemographicPreset(frequency_scale=1.02, high_shelf_gain=0.02, energy_scale=0.99, blend=0.18),
    "female_young": DemographicPreset(frequency_scale=1.05, high_shelf_gain=0.05, energy_scale=0.98, blend=0.22),
}


class BiometricObfuscator(nn.Module):
    def __init__(self, n_fft: int = 512, hop_length: int = 128) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def _warp_frequency_axis(self, magnitude: torch.Tensor, scale: float) -> torch.Tensor:
        batch, freq_bins, frames = magnitude.shape
        source = magnitude.permute(0, 2, 1).reshape(batch * frames, 1, freq_bins)
        grid = torch.linspace(-1.0, 1.0, freq_bins, device=magnitude.device)
        grid = torch.clamp(grid / scale, -1.0, 1.0)
        grid = torch.stack([grid, torch.zeros_like(grid)], dim=-1).view(1, freq_bins, 1, 2)
        grid = grid.expand(batch * frames, freq_bins, 1, 2)
        sampled = F.grid_sample(source.unsqueeze(-1), grid, align_corners=True, mode="bilinear", padding_mode="border")
        warped = sampled.squeeze(-1).view(batch, frames, freq_bins).permute(0, 2, 1)
        return warped

    def _apply_spectral_tilt(self, magnitude: torch.Tensor, gain: float) -> torch.Tensor:
        freq_bins = magnitude.shape[1]
        curve = torch.linspace(0.0, 1.0, freq_bins, device=magnitude.device).view(1, freq_bins, 1)
        return magnitude * (1.0 + gain * curve)

    def forward(self, waveform: torch.Tensor, source_preset: str, target_preset: str) -> torch.Tensor:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        source = PRESETS[source_preset]
        target = PRESETS[target_preset]
        scale_ratio = target.frequency_scale / source.frequency_scale
        gain_delta = target.high_shelf_gain - source.high_shelf_gain
        energy_ratio = target.energy_scale / source.energy_scale
        blend = max(source.blend, target.blend)

        spectrum = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            return_complex=True,
            center=True,
        )
        magnitude = spectrum.abs()
        phase = torch.angle(spectrum)
        softened_scale = 1.0 + 0.35 * (scale_ratio - 1.0)
        warped = self._warp_frequency_axis(magnitude, softened_scale)
        tilted = self._apply_spectral_tilt(warped, gain_delta * 0.5)
        blended = (1.0 - blend) * magnitude + blend * tilted
        reconstructed = blended * torch.exp(1j * phase)
        waveform_out = torch.istft(
            reconstructed,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            length=waveform.shape[-1],
        )
        waveform_out = waveform_out * energy_ratio
        waveform_out = waveform_out / torch.clamp(waveform_out.abs().amax(dim=-1, keepdim=True), min=1e-6)
        return waveform_out.squeeze(0)
