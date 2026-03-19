from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .audio import build_environment_catalog
from .data_utils import BaselineSpeakerDataset, DisentangledTripletDataset, build_or_load_split_bundle
from .evaluation import evaluate_checkpoint
from .models import BaselineSpeakerNet, DisentangledSpeakerNet, build_model, mean_absolute_correlation
from .utils import ensure_dir, save_json, set_seed, load_config


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Question 2 speaker recognition experiments.")
    parser.add_argument("--config", required=True, help="Path to a YAML config under q2/configs/ or a custom config.")
    parser.add_argument("--device", default=None, help="Optional device override, e.g. cpu or cuda.")
    parser.add_argument("--refresh-splits", action="store_true", help="Rebuild the cached LibriSpeech split bundle.")
    return parser.parse_args()


def baseline_epoch(model: BaselineSpeakerNet, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for batch in loader:
        waveforms = batch["waveform"].to(device)
        speaker_labels = batch["speaker_label"].to(device)

        outputs = model(waveforms)
        loss = F.cross_entropy(outputs["speaker_logits"], speaker_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * len(speaker_labels)
        total_correct += int((outputs["speaker_logits"].argmax(dim=-1) == speaker_labels).sum().item())
        total_examples += len(speaker_labels)
    return {"loss": total_loss / max(total_examples, 1), "accuracy": total_correct / max(total_examples, 1)}


def disentangled_epoch(
    model: DisentangledSpeakerNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_config: dict[str, Any],
    experiment_type: str,
) -> dict[str, float]:
    model.train()
    total_examples = 0
    accumulated = {"loss": 0.0, "speaker_loss": 0.0, "env_loss": 0.0, "adv_loss": 0.0, "rec_loss": 0.0, "corr_loss": 0.0, "consistency_loss": 0.0, "accuracy": 0.0}

    for batch in loader:
        waveforms = batch["waveforms"].to(device)
        batch_size, triplet_size, waveform_length = waveforms.shape
        flat_waveforms = waveforms.view(batch_size * triplet_size, waveform_length)
        speaker_labels = batch["speaker_label"].to(device).repeat_interleave(triplet_size)
        environment_labels = batch["environment_labels"].to(device).reshape(-1)

        outputs = model(flat_waveforms)
        speaker_loss = F.cross_entropy(outputs.speaker_logits, speaker_labels)
        environment_loss = F.cross_entropy(outputs.environment_logits, environment_labels)
        adversarial_loss = F.cross_entropy(outputs.adversarial_environment_logits, environment_labels)
        reconstruction_loss = F.mse_loss(outputs.reconstructed_embedding, outputs.embeddings.detach())
        correlation_loss = mean_absolute_correlation(outputs.speaker_code, outputs.environment_code)

        consistency_loss = torch.tensor(0.0, device=device)
        if experiment_type == "improved":
            speaker_codes = outputs.speaker_code.view(batch_size, triplet_size, -1)
            anchor = speaker_codes[:, 0, :]
            different_environment = speaker_codes[:, 2, :]
            consistency_loss = (1.0 - F.cosine_similarity(anchor, different_environment, dim=-1)).mean()

        total_loss = (
            speaker_loss
            + float(loss_config["environment_weight"]) * environment_loss
            + float(loss_config["adversarial_weight"]) * adversarial_loss
            + float(loss_config["reconstruction_weight"]) * reconstruction_loss
            + float(loss_config["correlation_weight"]) * correlation_loss
            + float(loss_config.get("consistency_weight", 0.0)) * consistency_loss
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        predictions = outputs.speaker_logits.argmax(dim=-1)
        batch_examples = len(speaker_labels)
        accumulated["loss"] += float(total_loss.item()) * batch_examples
        accumulated["speaker_loss"] += float(speaker_loss.item()) * batch_examples
        accumulated["env_loss"] += float(environment_loss.item()) * batch_examples
        accumulated["adv_loss"] += float(adversarial_loss.item()) * batch_examples
        accumulated["rec_loss"] += float(reconstruction_loss.item()) * batch_examples
        accumulated["corr_loss"] += float(correlation_loss.item()) * batch_examples
        accumulated["consistency_loss"] += float(consistency_loss.item()) * batch_examples
        accumulated["accuracy"] += float((predictions == speaker_labels).sum().item())
        total_examples += batch_examples

    for key in accumulated:
        accumulated[key] /= max(total_examples, 1)
    return accumulated


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config["config_path"] = str(Path(args.config).resolve())
    set_seed(int(config["training"]["seed"]))

    experiment_name = str(config["experiment"]["name"])
    experiment_type = str(config["experiment"]["type"])
    split_bundle = build_or_load_split_bundle(config, refresh=args.refresh_splits)
    num_speakers = len(split_bundle.speaker_to_label)
    num_environments = len(build_environment_catalog())

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(config, num_speakers=num_speakers, num_environments=num_environments).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))

    checkpoints_dir = ensure_dir(PROJECT_ROOT / "results" / "checkpoints")
    experiment_dir = ensure_dir(checkpoints_dir / experiment_name)
    metrics_dir = ensure_dir(PROJECT_ROOT / "results" / "metrics")

    if experiment_type == "baseline":
        train_dataset = BaselineSpeakerDataset(split_bundle.train_records, config, training=True)
        train_loader = DataLoader(train_dataset, batch_size=int(config["training"]["batch_size"]), shuffle=True)
    else:
        train_dataset = DisentangledTripletDataset(split_bundle.train_records, config)
        train_loader = DataLoader(train_dataset, batch_size=int(config["training"]["batch_size"]), shuffle=True)

    history: list[dict[str, float]] = []
    best_augmented_eer = float("inf")
    best_augmented_accuracy = -1.0
    best_checkpoint_path = experiment_dir / "best.pt"

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        if experiment_type == "baseline":
            train_metrics = baseline_epoch(model, train_loader, optimizer, device)
        else:
            train_metrics = disentangled_epoch(model, train_loader, optimizer, device, config["loss"], experiment_type)

        evaluation_metrics = evaluate_checkpoint(
            model=model,
            split_bundle=split_bundle,
            config=config,
            checkpoint_name=f"{experiment_name}_epoch{epoch:02d}",
            device=device,
            results_dir=metrics_dir,
        )
        epoch_payload = {"epoch": epoch, **train_metrics, **evaluation_metrics}
        history.append(epoch_payload)

        augmented_accuracy = float(evaluation_metrics["augmented_top1_accuracy"])
        augmented_eer = float(evaluation_metrics["augmented_eer"])
        if (augmented_eer < best_augmented_eer) or (
            abs(augmented_eer - best_augmented_eer) < 1e-12 and augmented_accuracy >= best_augmented_accuracy
        ):
            best_augmented_eer = augmented_eer
            best_augmented_accuracy = augmented_accuracy
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": config,
                    "speaker_to_label": split_bundle.speaker_to_label,
                    "label_to_speaker": split_bundle.label_to_speaker,
                    "experiment_name": experiment_name,
                    "experiment_type": experiment_type,
                    "num_speakers": num_speakers,
                    "num_environments": num_environments,
                    "epoch": epoch,
                    "history": history,
                },
                best_checkpoint_path,
            )

    save_json(
        experiment_dir / "history.json",
        {
            "history": history,
            "best_augmented_accuracy": best_augmented_accuracy,
            "best_augmented_eer": best_augmented_eer,
        },
    )
    save_json(experiment_dir / "split_summary.json", {
        "num_speakers": num_speakers,
        "num_train_records": len(split_bundle.train_records),
        "num_enroll_records": len(split_bundle.enroll_records),
        "num_test_records": len(split_bundle.test_records),
    })
    print(f"Saved best checkpoint to {best_checkpoint_path}")


if __name__ == "__main__":
    main()
