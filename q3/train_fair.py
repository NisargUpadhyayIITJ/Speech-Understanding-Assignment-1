from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCTC, Wav2Vec2Processor

from .asr_utils import word_error_rate
from .common_voice import PROJECT_ROOT, materialize_common_voice_subset


MODEL_NAME = "facebook/wav2vec2-base-960h"
TARGET_SAMPLE_RATE = 16000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a fairness-aware adapter on top of a frozen pretrained ASR model.")
    parser.add_argument("--refresh-data", action="store_true", help="Refresh the cached Common Voice subsets.")
    parser.add_argument("--device", default=None, help="Optional device override.")
    return parser.parse_args()


def normalize_asr_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fairness_penalty(sample_losses: torch.Tensor, genders: list[str]) -> torch.Tensor:
    grouped = {}
    for loss, gender in zip(sample_losses, genders, strict=True):
        grouped.setdefault(gender, []).append(loss)
    group_means = [torch.stack(values).mean() for values in grouped.values() if values]
    if len(group_means) < 2:
        return torch.tensor(0.0, device=sample_losses.device)
    return torch.stack(group_means).std(unbiased=False)


class CommonVoiceWaveformDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], root: Path) -> None:
        self.rows = rows
        self.root = root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        waveform, sample_rate = sf.read(self.root / row["audio_path"])
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        return {
            "waveform": waveform.astype(np.float32),
            "sample_rate": int(sample_rate),
            "text": row["sentence"].upper(),
            "gender": row["gender"],
            "sample_id": row["sample_id"],
        }


def build_collate(processor: Wav2Vec2Processor):
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        waveforms = [item["waveform"] for item in batch]
        encoded = processor(
            waveforms,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        label_lists = [processor.tokenizer(item["text"]).input_ids for item in batch]
        label_lengths = torch.tensor([len(item) for item in label_lists], dtype=torch.long)
        labels = torch.cat([torch.tensor(item, dtype=torch.long) for item in label_lists], dim=0)
        return {
            "input_values": encoded.input_values,
            "attention_mask": encoded.attention_mask,
            "labels": labels,
            "label_lengths": label_lengths,
            "texts": [item["text"] for item in batch],
            "gender": [item["gender"] for item in batch],
            "sample_id": [item["sample_id"] for item in batch],
        }

    return collate


class FrozenAdapterASR(nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.base_model = AutoModelForCTC.from_pretrained(model_name)
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False
        vocab_size = self.base_model.config.vocab_size
        self.adapter = nn.Linear(vocab_size, vocab_size, bias=False)
        with torch.no_grad():
            self.adapter.weight.zero_()

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            outputs = self.base_model(input_values=input_values, attention_mask=attention_mask)
        logits = outputs.logits + 0.05 * self.adapter(outputs.logits)
        input_lengths = attention_mask.sum(dim=-1)
        output_lengths = self.base_model._get_feat_extract_output_lengths(input_lengths).to(torch.long)
        output_lengths = torch.clamp(output_lengths, max=logits.shape[1])
        return logits, output_lengths

    def identity_regularizer(self) -> torch.Tensor:
        return F.mse_loss(self.adapter.weight, torch.zeros_like(self.adapter.weight))


def evaluate_model(
    model: FrozenAdapterASR,
    loader: DataLoader,
    processor: Wav2Vec2Processor,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    wers = []
    by_gender = {"male": [], "female": []}
    with torch.no_grad():
        for batch in loader:
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits, _ = model(input_values, attention_mask)
            prediction_ids = torch.argmax(logits, dim=-1)
            hypotheses = [normalize_asr_text(text) for text in processor.batch_decode(prediction_ids)]
            references = [normalize_asr_text(text) for text in batch["texts"]]
            for reference, hypothesis, gender in zip(references, hypotheses, batch["gender"], strict=True):
                wer = word_error_rate(reference, hypothesis)
                wers.append(wer)
                if gender in by_gender:
                    by_gender[gender].append(wer)
    male_wer = float(np.mean(by_gender["male"])) if by_gender["male"] else float("nan")
    female_wer = float(np.mean(by_gender["female"])) if by_gender["female"] else float("nan")
    return {
        "overall_wer": float(np.mean(wers)) if wers else float("nan"),
        "male_wer": male_wer,
        "female_wer": female_wer,
        "gender_gap": abs(male_wer - female_wer) if np.isfinite(male_wer) and np.isfinite(female_wer) else float("nan"),
    }


def selection_score(metrics: dict[str, Any]) -> float:
    overall_wer = metrics.get("overall_wer", float("inf"))
    gender_gap = metrics.get("gender_gap", float("inf"))
    if not np.isfinite(overall_wer):
        overall_wer = float("inf")
    if not np.isfinite(gender_gap):
        gender_gap = 1.0
    return float(overall_wer + 0.4 * gender_gap)


def balanced_capacity(rows: list[dict[str, Any]]) -> int:
    male = sum(1 for row in rows if row["gender"] == "male")
    female = sum(1 for row in rows if row["gender"] == "female")
    return min(male, female)


def stratified_gender_split(
    rows: list[dict[str, Any]],
    train_per_gender: int,
    val_per_gender: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped = {"male": [], "female": []}
    for row in rows:
        if row["gender"] in grouped:
            grouped[row["gender"]].append(row)
    required = train_per_gender + val_per_gender
    for gender, samples in grouped.items():
        if len(samples) < required:
            raise ValueError(f"Need {required} {gender} samples, found {len(samples)}")
    train_rows = grouped["male"][:train_per_gender] + grouped["female"][:train_per_gender]
    val_rows = grouped["male"][train_per_gender : train_per_gender + val_per_gender] + grouped["female"][train_per_gender : train_per_gender + val_per_gender]
    return train_rows, val_rows


def train_adapter_model(
    model_name: str,
    fairness_weight: float,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    processor: Wav2Vec2Processor,
    device: torch.device,
    results_dir: Path,
    blank_id: int,
) -> dict[str, Any]:
    model = FrozenAdapterASR(MODEL_NAME).to(device)
    optimizer = torch.optim.Adam(model.adapter.parameters(), lr=5e-5)
    ctc = torch.nn.CTCLoss(blank=blank_id, zero_infinity=True, reduction="none")

    checkpoint_dir = results_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{model_name}.pt"
    initial_metrics = evaluate_model(model, val_loader, processor, device)
    best_payload = {"epoch": 0, "train_loss": 0.0, **initial_metrics}
    best_score = selection_score(initial_metrics)
    history = [best_payload]
    torch.save({"state_dict": model.state_dict(), "best_payload": best_payload, "history": history}, checkpoint_path)

    for epoch in range(1, 4):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            label_lengths = batch["label_lengths"].to(device)

            logits, output_lengths = model(input_values, attention_mask)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
            sample_losses = ctc(log_probs, labels, output_lengths, label_lengths)
            sample_losses = torch.nan_to_num(sample_losses, nan=0.0, posinf=0.0, neginf=0.0)
            base_loss = sample_losses.mean()
            fair_loss = fairness_penalty(sample_losses, batch["gender"])
            regularizer = model.identity_regularizer()
            loss = base_loss + fairness_weight * fair_loss + 0.05 * regularizer
            if not torch.isfinite(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.adapter.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        val_metrics = evaluate_model(model, val_loader, processor, device)
        payload = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            **val_metrics,
        }
        history.append(payload)
        current_score = selection_score(val_metrics)
        if best_payload is None or current_score <= best_score:
            best_score = current_score
            best_payload = payload
            torch.save({"state_dict": model.state_dict(), "best_payload": best_payload, "history": history}, checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics = evaluate_model(model, test_loader, processor, device)
    result = {
        "model_name": model_name,
        "fairness_weight": fairness_weight,
        "best_validation": checkpoint["best_payload"],
        "test_metrics": test_metrics,
        "checkpoint_path": str(checkpoint_path),
        "history": checkpoint["history"],
    }
    (results_dir / f"{model_name}_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def evaluate_frozen_baseline(
    processor: Wav2Vec2Processor,
    test_loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model = FrozenAdapterASR(MODEL_NAME).to(device)
    metrics = evaluate_model(model, test_loader, processor, device)
    return {
        "model_name": "frozen_pretrained",
        "fairness_weight": 0.0,
        "test_metrics": metrics,
        "checkpoint_path": "identity_adapter",
    }


def main() -> None:
    args = parse_args()
    results_dir = PROJECT_ROOT / "results" / "fairness"
    results_dir.mkdir(parents=True, exist_ok=True)

    train_subset = materialize_common_voice_subset(
        split="dev",
        max_examples=160,
        subset_name="fair_asr_dev",
        balance_by_gender=True,
        max_word_count=6,
        refresh=args.refresh_data,
    )
    eval_subset = materialize_common_voice_subset(
        split="test",
        max_examples=64,
        subset_name="fair_asr_test",
        balance_by_gender=True,
        max_word_count=6,
        refresh=args.refresh_data,
    )

    capacity = balanced_capacity(train_subset.rows)
    val_per_gender = min(20, max(12, capacity // 4))
    train_per_gender = capacity - val_per_gender
    train_rows, val_rows = stratified_gender_split(train_subset.rows, train_per_gender=train_per_gender, val_per_gender=val_per_gender)
    test_rows = eval_subset.rows[:64]

    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    collate = build_collate(processor)
    train_loader = DataLoader(CommonVoiceWaveformDataset(train_rows, train_subset.root), batch_size=6, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(CommonVoiceWaveformDataset(val_rows, train_subset.root), batch_size=6, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(CommonVoiceWaveformDataset(test_rows, eval_subset.root), batch_size=6, shuffle=False, collate_fn=collate)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    blank_id = int(processor.tokenizer.pad_token_id)
    frozen_baseline = evaluate_frozen_baseline(processor, test_loader, device)
    adapter_baseline = train_adapter_model(
        "adapter_ctc",
        0.0,
        train_loader,
        val_loader,
        test_loader,
        processor,
        device,
        results_dir,
        blank_id,
    )
    fairness_model = train_adapter_model(
        "adapter_fair",
        0.8,
        train_loader,
        val_loader,
        test_loader,
        processor,
        device,
        results_dir,
        blank_id,
    )

    comparison = {
        "frozen_pretrained": frozen_baseline,
        "adapter_ctc": adapter_baseline,
        "adapter_fair": fairness_model,
        "train_examples": len(train_rows),
        "validation_examples": len(val_rows),
        "test_examples": len(test_rows),
    }
    (results_dir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    labels = ["frozen_pretrained", "adapter_ctc", "adapter_fair"]
    overall = [comparison[label]["test_metrics"]["overall_wer"] for label in labels]
    gaps = [comparison[label]["test_metrics"]["gender_gap"] for label in labels]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].bar(labels, overall, color=["#4c78a8", "#9ecae9", "#f58518"])
    axes[0].set_title("Overall WER")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(alpha=0.3, axis="y")
    axes[1].bar(labels, gaps, color=["#4c78a8", "#9ecae9", "#f58518"])
    axes[1].set_title("Gender WER Gap")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(alpha=0.3, axis="y")
    figure.savefig(results_dir / "fairness_comparison.png", dpi=200)
    plt.close(figure)

    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
