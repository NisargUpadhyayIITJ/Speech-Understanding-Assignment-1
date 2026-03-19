from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parent


def normalize_split_name(subset: str, split: str) -> str:
    split = split.strip()
    subset = subset.strip()

    if subset == "clean":
        mapping = {
            "test.clean": "test",
            "validation.clean": "validation",
            "dev.clean": "validation",
            "train.clean.100": "train.100",
            "train.clean.360": "train.360",
        }
        return mapping.get(split, split)

    if subset == "other":
        mapping = {
            "test.other": "test",
            "validation.other": "validation",
            "dev.other": "validation",
            "train.other.500": "train.500",
        }
        return mapping.get(split, split)

    if subset == "all":
        mapping = {
            "test": "test.clean",
            "validation": "validation.clean",
            "dev": "validation.clean",
        }
        return mapping.get(split, split)

    return split


def normalize_phone_tokens(tokens: list[str]) -> str:
    cleaned_tokens = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if token == " ":
            cleaned_tokens.append("|")
            continue
        if re.fullmatch(r"[A-Z]+[0-2]?", token):
            cleaned_tokens.append(token)
    return " ".join(cleaned_tokens)


def ensure_nltk_resource(resource_path: str, download_name: str) -> None:
    import nltk

    try:
        nltk.data.find(resource_path)
        return
    except LookupError:
        nltk.download(download_name, quiet=True)

    try:
        nltk.data.find(resource_path)
    except LookupError as exc:
        raise RuntimeError(
            f"Missing NLTK resource '{download_name}' required for auto phone sequence generation. "
            f"Please run: python3 -c \"import nltk; nltk.download('{download_name}')\""
        ) from exc


def ensure_g2p_dependencies() -> None:
    ensure_nltk_resource("corpora/cmudict", "cmudict")
    ensure_nltk_resource("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng")
    # Keep the legacy tagger available too because some g2p-en / NLTK combinations still reference it.
    ensure_nltk_resource("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger")


def build_phone_sequence(transcript: str, auto_phone_sequence: bool, g2p: object | None = None) -> str:
    if not auto_phone_sequence:
        return ""

    if g2p is None:
        raise RuntimeError("g2p instance is required when auto_phone_sequence is enabled")
    return normalize_phone_tokens(g2p(transcript))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a small subset from openslr/librispeech_asr and build a local manifest."
    )
    parser.add_argument(
        "--subset",
        default="clean",
        choices=["clean", "other", "all"],
        help="Dataset configuration to load from Hugging Face.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to sample from. Aliases like test.clean and validation.clean are normalized automatically.",
    )
    parser.add_argument("--num-samples", type=int, default=3, help="Number of clips to export locally.")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "librispeech_subset"),
        help="Directory where audio clips and manifests will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Shuffle seed used before selecting samples.",
    )
    parser.add_argument(
        "--auto-phone-sequence",
        action="store_true",
        help="Generate an ARPAbet-like phone sequence for each transcript using g2p-en.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    from datasets import Audio, get_dataset_split_names, load_dataset
    g2p = None
    if args.auto_phone_sequence:
        try:
            from g2p_en import G2p
        except ImportError as exc:
            raise RuntimeError(
                "Automatic phone-sequence generation requires g2p-en. "
                "Install dependencies from requirements.txt and retry with --auto-phone-sequence."
            ) from exc
        ensure_g2p_dependencies()
        g2p = G2p()

    resolved_split = normalize_split_name(args.subset, args.split)
    available_splits = get_dataset_split_names("openslr/librispeech_asr", args.subset)
    if resolved_split not in available_splits:
        raise ValueError(
            f'Unknown split "{args.split}" for subset "{args.subset}". '
            f'Resolved split "{resolved_split}" is not available. '
            f"Choose one of: {available_splits}."
        )

    dataset = load_dataset("openslr/librispeech_asr", args.subset, split=resolved_split)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    count = min(args.num_samples, len(dataset))
    if count <= 0:
        raise ValueError("num-samples must be positive")

    sampled = dataset.shuffle(seed=args.seed).select(range(count))
    manifest_path = output_dir / "manifest.csv"
    records = []

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=[
                "audio_path",
                "transcript",
                "phone_sequence",
                "split",
                "speaker_id",
                "chapter_id",
                "sample_id",
                "notes",
            ],
        )
        writer.writeheader()

        for example in sampled:
            audio = example["audio"]
            sample_id = example["id"]
            relative_audio_path = Path("audio") / f"{sample_id}.wav"
            local_audio_path = output_dir / relative_audio_path
            sf.write(local_audio_path, audio["array"], audio["sampling_rate"])
            phone_sequence = build_phone_sequence(example["text"], args.auto_phone_sequence, g2p=g2p)

            row = {
                "audio_path": str(relative_audio_path),
                "transcript": example["text"],
                "phone_sequence": phone_sequence,
                "split": resolved_split,
                "speaker_id": example["speaker_id"],
                "chapter_id": example["chapter_id"],
                "sample_id": sample_id,
                "notes": (
                    "phone_sequence generated with g2p-en"
                    if args.auto_phone_sequence
                    else "Fill phone_sequence if using a phoneme-tokenized alignment model."
                ),
            }
            writer.writerow(row)
            records.append(row)

    summary = {
        "dataset_name": "openslr/librispeech_asr",
        "subset": args.subset,
        "requested_split": args.split,
        "resolved_split": resolved_split,
        "auto_phone_sequence": args.auto_phone_sequence,
        "num_samples": count,
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "samples": records,
    }
    summary_path = output_dir / "manifest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
