from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .audio import (
    apply_environment,
    build_environment_catalog,
    crop_or_pad,
    normalize_audio,
    pre_emphasis,
    resample_audio,
)


@dataclass(slots=True)
class SplitBundle:
    speaker_to_label: dict[int, int]
    label_to_speaker: dict[int, int]
    train_records: list[dict[str, Any]]
    enroll_records: list[dict[str, Any]]
    test_records: list[dict[str, Any]]


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_hf_dataset(config: dict[str, Any]):
    from datasets import Audio, load_dataset

    dataset = load_dataset(
        config["dataset"]["hf_name"],
        config["dataset"]["hf_subset"],
        split=config["dataset"]["hf_split"],
    )
    return dataset.cast_column("audio", Audio(sampling_rate=int(config["audio"]["sample_rate"])))


def _serialize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in record.items() if key != "audio"} for record in records]


def _hydrate_records(records: list[dict[str, Any]], dataset: Any) -> list[dict[str, Any]]:
    hydrated = []
    for record in records:
        row = dataset[int(record["dataset_index"])]
        hydrated.append(
            {
                **record,
                "audio": {
                    "array": row["audio"]["array"],
                    "sampling_rate": int(row["audio"]["sampling_rate"]),
                },
            }
        )
    return hydrated


def build_or_load_split_bundle(config: dict[str, Any], refresh: bool = False) -> SplitBundle:
    cache_dir = _project_root() / "results" / "cache"
    _ensure_dir(cache_dir)
    cache_path = cache_dir / "split_bundle.json"
    if cache_path.exists() and not refresh:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        dataset = _load_hf_dataset(config)
        return SplitBundle(
            speaker_to_label={int(k): int(v) for k, v in payload["speaker_to_label"].items()},
            label_to_speaker={int(k): int(v) for k, v in payload["label_to_speaker"].items()},
            train_records=_hydrate_records(payload["train_records"], dataset),
            enroll_records=_hydrate_records(payload["enroll_records"], dataset),
            test_records=_hydrate_records(payload["test_records"], dataset),
        )

    dataset_name = config["dataset"]["hf_name"]
    subset = config["dataset"]["hf_subset"]
    split = config["dataset"]["hf_split"]
    sample_rate = int(config["audio"]["sample_rate"])
    num_speakers = int(config["dataset"]["num_speakers"])
    min_total = int(config["dataset"]["train_utterances_per_speaker"]) + int(config["dataset"]["enroll_utterances_per_speaker"]) + int(
        config["dataset"]["test_utterances_per_speaker"]
    )
    seed = int(config["training"]["seed"])

    dataset = _load_hf_dataset(config)

    speaker_ids = list(dataset["speaker_id"])
    counts = Counter(speaker_ids)
    eligible_speakers = [speaker_id for speaker_id, count in counts.items() if count >= min_total]
    eligible_speakers = sorted(eligible_speakers)[:num_speakers]
    if len(eligible_speakers) < num_speakers:
        raise RuntimeError(
            f"Requested {num_speakers} speakers, but only found {len(eligible_speakers)} with at least {min_total} utterances."
        )

    indices_by_speaker: dict[int, list[int]] = defaultdict(list)
    for index, speaker_id in enumerate(speaker_ids):
        if speaker_id in eligible_speakers:
            indices_by_speaker[int(speaker_id)].append(index)

    speaker_to_label = {speaker_id: label for label, speaker_id in enumerate(eligible_speakers)}
    label_to_speaker = {label: speaker for speaker, label in speaker_to_label.items()}
    rng = np.random.default_rng(seed)

    split_counts = {
        "train": int(config["dataset"]["train_utterances_per_speaker"]),
        "enroll": int(config["dataset"]["enroll_utterances_per_speaker"]),
        "test": int(config["dataset"]["test_utterances_per_speaker"]),
    }
    train_records: list[dict[str, Any]] = []
    enroll_records: list[dict[str, Any]] = []
    test_records: list[dict[str, Any]] = []

    for speaker_id in eligible_speakers:
        speaker_indices = indices_by_speaker[int(speaker_id)].copy()
        rng.shuffle(speaker_indices)
        train_end = split_counts["train"]
        enroll_end = train_end + split_counts["enroll"]
        selected = speaker_indices[: enroll_end + split_counts["test"]]
        for split_name, split_indices in {
            "train": selected[:train_end],
            "enroll": selected[train_end:enroll_end],
            "test": selected[enroll_end : enroll_end + split_counts["test"]],
        }.items():
            target = {"train": train_records, "enroll": enroll_records, "test": test_records}[split_name]
            for index in split_indices:
                row = dataset[int(index)]
                target.append(
                    {
                        "dataset_index": int(index),
                        "speaker_id": int(row["speaker_id"]),
                        "label": int(speaker_to_label[int(row["speaker_id"])]),
                        "sample_id": str(row["id"]),
                        "text": str(row["text"]),
                        "audio": {
                            "array": row["audio"]["array"],
                            "sampling_rate": int(row["audio"]["sampling_rate"]),
                        },
                    }
                )

    payload = {
        "speaker_to_label": speaker_to_label,
        "label_to_speaker": label_to_speaker,
        "train_records": _serialize_records(train_records),
        "enroll_records": _serialize_records(enroll_records),
        "test_records": _serialize_records(test_records),
    }
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return SplitBundle(
        speaker_to_label=speaker_to_label,
        label_to_speaker=label_to_speaker,
        train_records=train_records,
        enroll_records=enroll_records,
        test_records=test_records,
    )


def _prepare_audio(record: dict[str, Any], sample_rate: int, segment_seconds: float, rng: np.random.Generator | None, random_crop: bool) -> np.ndarray:
    waveform = np.asarray(record["audio"]["array"], dtype=np.float32)
    sr = int(record["audio"]["sampling_rate"])
    waveform = resample_audio(waveform, sr, sample_rate)
    waveform = pre_emphasis(normalize_audio(waveform))
    target_length = int(round(sample_rate * segment_seconds))
    waveform = crop_or_pad(waveform, target_length, rng=rng, random_crop=random_crop)
    return waveform.astype(np.float32)


class BaselineSpeakerDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], config: dict[str, Any], training: bool) -> None:
        self.records = records
        self.config = config
        self.training = training
        self.environments = build_environment_catalog()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        rng = np.random.default_rng(int(self.config["training"]["seed"]) * 10_000 + index + (1 if self.training else 9_999))
        waveform = _prepare_audio(
            record,
            sample_rate=int(self.config["audio"]["sample_rate"]),
            segment_seconds=float(self.config["audio"]["segment_seconds"]),
            rng=rng if self.training else None,
            random_crop=self.training,
        )
        environment = self.environments[int(rng.integers(0, len(self.environments)))] if self.training else self.environments[0]
        waveform = apply_environment(waveform, int(self.config["audio"]["sample_rate"]), str(environment["name"]), rng)
        return {
            "waveform": torch.tensor(waveform, dtype=torch.float32),
            "speaker_label": int(record["label"]),
            "environment_label": int(environment["id"]),
            "sample_id": record["sample_id"],
        }


class DisentangledTripletDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], config: dict[str, Any]) -> None:
        self.records = records
        self.config = config
        self.sample_rate = int(config["audio"]["sample_rate"])
        self.segment_seconds = float(config["audio"]["segment_seconds"])
        self.environments = build_environment_catalog()
        self.by_speaker: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            self.by_speaker[int(record["label"])].append(record)
        self.speakers = sorted(self.by_speaker)

    def __len__(self) -> int:
        return max(1, len(self.records))

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = np.random.default_rng(int(self.config["training"]["seed"]) * 31_337 + index)
        speaker_label = int(self.speakers[index % len(self.speakers)])
        speaker_records = self.by_speaker[speaker_label]
        chosen_indices = rng.choice(len(speaker_records), size=3, replace=len(speaker_records) < 3)
        selected_records = [speaker_records[int(i)] for i in chosen_indices]
        same_environment = self.environments[int(rng.integers(0, len(self.environments)))]
        different_choices = [item for item in self.environments if int(item["id"]) != int(same_environment["id"])]
        different_environment = different_choices[int(rng.integers(0, len(different_choices)))]

        waveforms = []
        env_labels = []
        for triplet_index, (record, environment) in enumerate(
            zip(selected_records, [same_environment, same_environment, different_environment], strict=True)
        ):
            local_rng = np.random.default_rng(int(rng.integers(0, 1_000_000)) + triplet_index)
            waveform = _prepare_audio(
                record,
                sample_rate=self.sample_rate,
                segment_seconds=self.segment_seconds,
                rng=local_rng,
                random_crop=True,
            )
            waveform = apply_environment(waveform, self.sample_rate, str(environment["name"]), local_rng)
            waveforms.append(torch.tensor(waveform, dtype=torch.float32))
            env_labels.append(int(environment["id"]))

        return {
            "waveforms": torch.stack(waveforms, dim=0),
            "speaker_label": speaker_label,
            "environment_labels": torch.tensor(env_labels, dtype=torch.long),
            "sample_ids": [record["sample_id"] for record in selected_records],
        }


class EvalSpeakerDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], config: dict[str, Any], environment_name: str) -> None:
        self.records = records
        self.config = config
        self.environment_name = environment_name
        self.sample_rate = int(config["audio"]["sample_rate"])
        self.segment_seconds = float(config["audio"]["segment_seconds"])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        rng = np.random.default_rng(int(self.config["training"]["seed"]) * 100_000 + index)
        waveform = _prepare_audio(
            record,
            sample_rate=self.sample_rate,
            segment_seconds=self.segment_seconds,
            rng=None,
            random_crop=False,
        )
        waveform = apply_environment(waveform, self.sample_rate, self.environment_name, rng)
        return {
            "waveform": torch.tensor(waveform, dtype=torch.float32),
            "speaker_label": int(record["label"]),
            "speaker_id": int(record["speaker_id"]),
            "sample_id": record["sample_id"],
            "environment_name": self.environment_name,
        }
