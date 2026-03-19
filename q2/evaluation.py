from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

from .data_utils import EvalSpeakerDataset, SplitBundle
from .utils import ensure_dir, save_json


def _extract_embeddings(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> list[dict[str, Any]]:
    model.eval()
    results: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            waveforms = batch["waveform"].to(device)
            outputs = model(waveforms)
            embeddings = outputs["embeddings"] if isinstance(outputs, dict) else outputs.speaker_code
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu().numpy()
            for index, embedding in enumerate(embeddings):
                results.append(
                    {
                        "embedding": embedding,
                        "speaker_label": int(batch["speaker_label"][index]),
                        "speaker_id": int(batch["speaker_id"][index]),
                        "sample_id": batch["sample_id"][index],
                        "environment_name": batch["environment_name"][index],
                    }
                )
    return results


def _build_centroids(records: list[dict[str, Any]]) -> dict[int, np.ndarray]:
    grouped: dict[int, list[np.ndarray]] = defaultdict(list)
    for record in records:
        grouped[int(record["speaker_label"])].append(record["embedding"])
    centroids = {}
    for label, embeddings in grouped.items():
        stacked = np.stack(embeddings, axis=0)
        centroid = stacked.mean(axis=0)
        centroids[label] = centroid / (np.linalg.norm(centroid) + 1e-8)
    return centroids


def _compute_accuracy(query_records: list[dict[str, Any]], centroids: dict[int, np.ndarray]) -> float:
    correct = 0
    for record in query_records:
        scores = {label: float(np.dot(record["embedding"], centroid)) for label, centroid in centroids.items()}
        prediction = max(scores.items(), key=lambda item: item[1])[0]
        correct += int(prediction == int(record["speaker_label"]))
    return correct / max(len(query_records), 1)


def _compute_eer(query_records: list[dict[str, Any]], centroids: dict[int, np.ndarray]) -> tuple[float, float]:
    scores = []
    labels = []
    for record in query_records:
        for speaker_label, centroid in centroids.items():
            scores.append(float(np.dot(record["embedding"], centroid)))
            labels.append(int(int(record["speaker_label"]) == int(speaker_label)))
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    index = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fnr[index] + fpr[index]) / 2.0)
    threshold = float(thresholds[index])
    return eer, threshold


def _plot_metric_bars(metrics: list[dict[str, Any]], results_dir: Path) -> None:
    model_names = [item["experiment_name"] for item in metrics]
    x_positions = np.arange(len(model_names))

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    clean_acc = [item["clean_top1_accuracy"] * 100.0 for item in metrics]
    aug_acc = [item["augmented_top1_accuracy"] * 100.0 for item in metrics]
    axes[0].bar(x_positions - 0.18, clean_acc, width=0.36, label="Clean")
    axes[0].bar(x_positions + 0.18, aug_acc, width=0.36, label="Augmented")
    axes[0].set_xticks(x_positions, model_names, rotation=15)
    axes[0].set_ylabel("Top-1 Accuracy (%)")
    axes[0].set_title("Speaker Identification Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis="y")

    clean_eer = [item["clean_eer"] * 100.0 for item in metrics]
    aug_eer = [item["augmented_eer"] * 100.0 for item in metrics]
    axes[1].bar(x_positions - 0.18, clean_eer, width=0.36, label="Clean")
    axes[1].bar(x_positions + 0.18, aug_eer, width=0.36, label="Augmented")
    axes[1].set_xticks(x_positions, model_names, rotation=15)
    axes[1].set_ylabel("EER (%)")
    axes[1].set_title("Speaker Verification EER")
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis="y")

    figure.savefig(results_dir / "model_comparison.png", dpi=200)
    plt.close(figure)


def _plot_embedding_pca(clean_records: list[dict[str, Any]], augmented_records: list[dict[str, Any]], results_dir: Path, experiment_name: str) -> None:
    merged = clean_records + augmented_records
    if len(merged) < 4:
        return
    embeddings = np.stack([record["embedding"] for record in merged], axis=0)
    pca = PCA(n_components=2)
    projected = pca.fit_transform(embeddings)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for axis, records, title in zip(
        axes,
        [clean_records, augmented_records],
        [f"{experiment_name}: Clean Embeddings", f"{experiment_name}: Augmented Embeddings"],
        strict=True,
    ):
        subset = projected[: len(records)] if records is clean_records else projected[len(clean_records) :]
        speaker_labels = [record["speaker_label"] for record in records]
        for speaker_label in sorted(set(speaker_labels)):
            mask = [index for index, label in enumerate(speaker_labels) if label == speaker_label]
            axis.scatter(subset[mask, 0], subset[mask, 1], label=f"spk {speaker_label}", s=25, alpha=0.8)
        axis.set_title(title)
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        axis.grid(alpha=0.3)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1, fontsize=8)
    figure.savefig(results_dir / f"{experiment_name}_embedding_pca.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def evaluate_checkpoint(
    model: torch.nn.Module,
    split_bundle: SplitBundle,
    config: dict[str, Any],
    checkpoint_name: str,
    device: torch.device,
    results_dir: Path,
) -> dict[str, Any]:
    batch_size = int(config["evaluation"]["batch_size"])
    enroll_loader = DataLoader(EvalSpeakerDataset(split_bundle.enroll_records, config, "clean"), batch_size=batch_size, shuffle=False)
    clean_loader = DataLoader(EvalSpeakerDataset(split_bundle.test_records, config, "clean"), batch_size=batch_size, shuffle=False)
    augmented_loader = DataLoader(
        EvalSpeakerDataset(split_bundle.test_records, config, str(config["evaluation"]["environment_name"])),
        batch_size=batch_size,
        shuffle=False,
    )

    enroll_records = _extract_embeddings(model, enroll_loader, device)
    clean_records = _extract_embeddings(model, clean_loader, device)
    augmented_records = _extract_embeddings(model, augmented_loader, device)

    centroids = _build_centroids(enroll_records)
    clean_accuracy = _compute_accuracy(clean_records, centroids)
    augmented_accuracy = _compute_accuracy(augmented_records, centroids)
    clean_eer, clean_threshold = _compute_eer(clean_records, centroids)
    augmented_eer, augmented_threshold = _compute_eer(augmented_records, centroids)

    metrics = {
        "experiment_name": checkpoint_name,
        "clean_top1_accuracy": clean_accuracy,
        "augmented_top1_accuracy": augmented_accuracy,
        "clean_eer": clean_eer,
        "augmented_eer": augmented_eer,
        "clean_threshold": clean_threshold,
        "augmented_threshold": augmented_threshold,
        "evaluation_environment": config["evaluation"]["environment_name"],
    }
    save_json(results_dir / f"{checkpoint_name}_metrics.json", metrics)
    _plot_embedding_pca(clean_records, augmented_records, results_dir, checkpoint_name)
    return metrics


def save_metrics_table(metrics: list[dict[str, Any]], results_dir: Path) -> None:
    results_dir = ensure_dir(results_dir)
    csv_path = results_dir / "comparison_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "experiment_name",
            "clean_top1_accuracy",
            "augmented_top1_accuracy",
            "clean_eer",
            "augmented_eer",
            "clean_threshold",
            "augmented_threshold",
            "evaluation_environment",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow(row)
    save_json(results_dir / "comparison_table.json", {"models": metrics})
    _plot_metric_bars(metrics, results_dir)
