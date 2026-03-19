from __future__ import annotations

import csv
import io
import json
import random
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download
from scipy.signal import resample_poly


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_NAME = "fsicoli/common_voice_17_0"
DATASET_CONFIG = "en"
EPSILON = 1e-8
SPLIT_ALIASES = {
    "validation": "dev",
    "val": "dev",
    "dev": "dev",
    "test": "test",
    "validated": "validated",
}
AUDIO_TAR_FILES = {
    "dev": "audio/en/dev/en_dev_0.tar",
    "test": "audio/en/test/en_test_0.tar",
}


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_gender(value: str | None) -> str:
    value = (value or "").strip().lower()
    if value in {"male", "male_masculine", "man"}:
        return "male"
    if value in {"female", "female_feminine", "woman"}:
        return "female"
    return "unknown"


def normalize_age_bucket(value: str | None) -> str:
    value = (value or "").strip().lower()
    young_values = {"teens", "twenties", "thirties"}
    old_values = {"fourties", "fifties", "sixties", "seventies", "eighties", "nineties"}
    if value in young_values:
        return "young"
    if value in old_values:
        return "old"
    return "unknown"


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def resample_audio(signal: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    if original_sr == target_sr:
        return signal.astype(np.float32)
    ratio = np.gcd(original_sr, target_sr)
    return resample_poly(signal, target_sr // ratio, original_sr // ratio).astype(np.float32)


def resolve_split(split: str) -> str:
    key = (split or "").strip().lower()
    if key not in SPLIT_ALIASES:
        raise ValueError(f"Unsupported Common Voice split '{split}'. Expected one of {sorted(SPLIT_ALIASES)}.")
    return SPLIT_ALIASES[key]


def _download_transcript(split: str) -> Path:
    resolved_split = resolve_split(split)
    transcript_path = f"transcript/en/{resolved_split}.tsv"
    return Path(hf_hub_download(DATASET_NAME, transcript_path, repo_type="dataset"))


def _download_audio_tar(split: str) -> Path:
    resolved_split = resolve_split(split)
    if resolved_split not in AUDIO_TAR_FILES:
        raise ValueError(f"Split '{split}' does not have an audio tar configured for local materialization.")
    return Path(hf_hub_download(DATASET_NAME, AUDIO_TAR_FILES[resolved_split], repo_type="dataset"))


def iter_transcript_rows(split: str) -> Iterator[dict[str, str]]:
    transcript_path = _download_transcript(split)
    with transcript_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            yield {key: (value or "") for key, value in row.items()}


def build_audit_sample(
    split: str = "validated",
    max_examples: int = 1200,
    refresh: bool = False,
    seed: int = 7,
) -> list[dict[str, Any]]:
    resolved_split = resolve_split(split)
    cache_dir = _ensure_dir(PROJECT_ROOT / "cache")
    cache_path = cache_dir / f"audit_{resolved_split}_{max_examples}.json"
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    seen = 0
    for row in iter_transcript_rows(resolved_split):
        record = {
            "client_id": row.get("client_id", ""),
            "sentence": row.get("sentence", ""),
            "age": row.get("age", ""),
            "gender": row.get("gender", ""),
            "accent": row.get("accents", ""),
            "locale": row.get("locale", ""),
            "variant": row.get("variant", ""),
            "segment": row.get("segment", ""),
            "path": row.get("path", ""),
        }
        if len(records) < max_examples:
            records.append(record)
        else:
            swap_index = rng.randint(0, seen)
            if swap_index < max_examples:
                records[swap_index] = record
        seen += 1

    cache_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


@dataclass(slots=True)
class MaterializedSubset:
    root: Path
    metadata_path: Path
    rows: list[dict[str, Any]]


def _build_group_limits(max_examples: int, allowed_genders: tuple[str, ...], balance_by_gender: bool) -> dict[str, int] | None:
    if not balance_by_gender or not allowed_genders:
        return None
    base = max_examples // len(allowed_genders)
    remainder = max_examples % len(allowed_genders)
    limits = {}
    for index, gender in enumerate(allowed_genders):
        limits[gender] = base + (1 if index < remainder else 0)
    return limits


def materialize_common_voice_subset(
    split: str,
    max_examples: int,
    subset_name: str,
    refresh: bool = False,
    min_text_length: int = 5,
    max_word_count: int | None = None,
    allowed_genders: tuple[str, ...] = ("male", "female"),
    require_known_age: bool = False,
    balance_by_gender: bool = False,
    target_sr: int = 16000,
) -> MaterializedSubset:
    resolved_split = resolve_split(split)
    root = PROJECT_ROOT / "cache" / subset_name
    metadata_path = root / "metadata.csv"
    audio_dir = root / "audio"
    if metadata_path.exists() and not refresh:
        with metadata_path.open("r", newline="", encoding="utf-8") as csv_file:
            rows = list(csv.DictReader(csv_file))
        return MaterializedSubset(root=root, metadata_path=metadata_path, rows=rows)

    _ensure_dir(audio_dir)
    tar_path = _download_audio_tar(resolved_split)
    group_limits = _build_group_limits(max_examples, allowed_genders, balance_by_gender)
    group_counts = {gender: 0 for gender in allowed_genders}
    rows: list[dict[str, Any]] = []

    with tarfile.open(tar_path, "r") as archive:
        members = {
            Path(member.name).name: member
            for member in archive.getmembers()
            if member.isfile() and member.name.endswith(".mp3")
        }
        for example in iter_transcript_rows(resolved_split):
            normalized_text = normalize_text(example.get("sentence", ""))
            gender = normalize_gender(example.get("gender"))
            age_value = example.get("age", "") or "unknown"
            age_bucket = normalize_age_bucket(age_value)
            accent = (example.get("accents", "") or "unknown").strip() or "unknown"
            word_count = len(normalized_text.split())

            if len(normalized_text) < min_text_length or gender not in allowed_genders:
                continue
            if max_word_count is not None and word_count > max_word_count:
                continue
            if require_known_age and age_bucket == "unknown":
                continue
            if group_limits is not None and group_counts[gender] >= group_limits[gender]:
                continue

            clip_name = example.get("path", "")
            member = members.get(clip_name)
            if member is None:
                continue

            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            waveform, sample_rate = sf.read(io.BytesIO(extracted.read()))
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            waveform = resample_audio(np.asarray(waveform, dtype=np.float32), int(sample_rate), target_sr)
            peak = float(np.max(np.abs(waveform))) if waveform.size else 1.0
            waveform = waveform / max(peak, EPSILON)

            sample_id = f"{resolved_split}_{len(rows):05d}"
            audio_path = audio_dir / f"{sample_id}.wav"
            sf.write(audio_path, waveform, target_sr)
            row = {
                "sample_id": sample_id,
                "split": resolved_split,
                "audio_path": str(audio_path.relative_to(root)),
                "sentence": normalized_text,
                "gender": gender,
                "age": age_value,
                "age_bucket": age_bucket,
                "accent": accent,
                "locale": example.get("locale", "en"),
                "client_id": example.get("client_id", ""),
                "source_path": clip_name,
            }
            rows.append(row)
            group_counts[gender] = group_counts.get(gender, 0) + 1
            if len(rows) >= max_examples:
                break

    if not rows:
        raise RuntimeError(f"No materialized Common Voice rows were created for split '{resolved_split}'.")

    with metadata_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return MaterializedSubset(root=root, metadata_path=metadata_path, rows=rows)


def split_materialized_rows(
    rows: list[dict[str, Any]],
    train_count: int,
    val_count: int,
    test_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    total = train_count + val_count + test_count
    if len(rows) < total:
        raise ValueError(f"Need at least {total} rows, found {len(rows)}")
    return (
        rows[:train_count],
        rows[train_count : train_count + val_count],
        rows[train_count + val_count : total],
    )
