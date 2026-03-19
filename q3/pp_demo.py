from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from .asr_utils import word_error_rate
from .common_voice import PROJECT_ROOT, materialize_common_voice_subset
from .evaluation_scripts.proxy_metrics import acceptability_proxy
from .privacymodule import BiometricObfuscator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run privacy-preserving voice transformation demos.")
    parser.add_argument("--subset-size", type=int, default=8, help="Number of examples to materialize for the demo subset.")
    parser.add_argument("--refresh", action="store_true", help="Refresh the local demo subset cache.")
    return parser.parse_args()


def build_transcriber(device: torch.device):
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    model_name = "openai/whisper-tiny.en"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name).to(device)
    model.eval()

    def transcribe(audio_path: Path) -> str:
        waveform, sample_rate = sf.read(audio_path)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
        input_features = inputs["input_features"].to(device)
        with torch.no_grad():
            generated = model.generate(input_features, max_new_tokens=96)
        text = processor.batch_decode(generated, skip_special_tokens=True)[0]
        return text.lower().strip()

    return transcribe


def main() -> None:
    args = parse_args()
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    subset = materialize_common_voice_subset(
        split="test",
        max_examples=args.subset_size,
        subset_name="privacy_demo",
        refresh=args.refresh,
        require_known_age=True,
        balance_by_gender=True,
    )
    examples_dir = PROJECT_ROOT / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transcribe = build_transcriber(device)
    obfuscator = BiometricObfuscator()
    results = []

    for row in subset.rows[:4]:
        source_gender = row["gender"]
        source_age_bucket = row["age_bucket"]
        source_preset = f"{source_gender}_{source_age_bucket if source_age_bucket in {'young', 'old'} else 'young'}"
        if source_gender == "male":
            target_preset = "female_young"
        else:
            target_preset = "male_old"

        waveform, sample_rate = sf.read(subset.root / row["audio_path"])
        waveform_tensor = torch.tensor(waveform, dtype=torch.float32)
        transformed = obfuscator(waveform_tensor, source_preset=source_preset, target_preset=target_preset).detach().cpu().numpy()

        original_path = examples_dir / f"{row['sample_id']}_original.wav"
        transformed_path = examples_dir / f"{row['sample_id']}_obfuscated.wav"
        sf.write(original_path, waveform, sample_rate)
        sf.write(transformed_path, transformed, sample_rate)

        reference_text = row["sentence"]
        original_asr = transcribe(original_path)
        transformed_asr = transcribe(transformed_path)
        proxy_metrics = acceptability_proxy(original_path, transformed_path)
        result = {
            "sample_id": row["sample_id"],
            "source_gender": source_gender,
            "source_age_bucket": source_age_bucket,
            "source_preset": source_preset,
            "target_preset": target_preset,
            "reference_text": reference_text,
            "original_asr": original_asr,
            "transformed_asr": transformed_asr,
            "original_wer": word_error_rate(reference_text, original_asr),
            "transformed_wer": word_error_rate(reference_text, transformed_asr),
            **proxy_metrics,
        }
        results.append(result)

    summary = {
        "num_examples": len(results),
        "examples": results,
        "mean_original_wer": float(np.mean([item["original_wer"] for item in results])),
        "mean_transformed_wer": float(np.mean([item["transformed_wer"] for item in results])),
        "mean_snr_db": float(np.mean([item["snr_db"] for item in results])),
        "mean_log_spectral_distance": float(np.mean([item["log_spectral_distance"] for item in results])),
    }
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    sample_labels = [item["sample_id"] for item in results]
    axes[0].bar(sample_labels, [item["transformed_wer"] for item in results], color="#4c78a8")
    axes[0].set_title("WER After Obfuscation")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(alpha=0.3, axis="y")
    axes[1].bar(sample_labels, [item["snr_db"] for item in results], color="#f58518")
    axes[1].set_title("Signal-to-Noise Ratio")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(alpha=0.3, axis="y")
    figure.savefig(results_dir / "privacy_demo_metrics.png", dpi=200)
    plt.close(figure)

    (results_dir / "privacy_demo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
