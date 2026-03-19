from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from speech_utils import ctc_viterbi_align, decode_greedy_segments, load_audio, nearest_boundary_rmse, tokenize_alignment_text


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map cepstral segments to phones and compute boundary RMSE.")
    parser.add_argument("--audio-path", required=True, help="Input waveform path.")
    parser.add_argument("--segments-json", required=True, help="Output JSON produced by voiced_unvoiced.py.")
    parser.add_argument(
        "--model-name",
        default="facebook/wav2vec2-base-960h",
        help="Hugging Face CTC model. For phone-level alignment, prefer a phoneme-tokenized checkpoint.",
    )
    parser.add_argument("--transcript", default=None, help="Optional raw transcript for forced alignment.")
    parser.add_argument(
        "--phone-sequence",
        default=None,
        help="Optional phone sequence for forced alignment. This is preferred when using a phoneme model.",
    )
    parser.add_argument("--sample-rate", type=int, default=None, help="Override model sample rate if needed.")
    parser.add_argument("--device", default=None, help="Computation device, e.g. cpu or cuda.")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "phonetic_mapping"),
        help="Directory for JSON and plots.",
    )
    return parser.parse_args()


def load_segments(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_frame_step_seconds(model: Any, sample_rate: int, logits_steps: int, audio_num_samples: int) -> float:
    ratio = getattr(model.config, "inputs_to_logits_ratio", None)
    if ratio is not None:
        return float(ratio) / float(sample_rate)
    return (audio_num_samples / float(sample_rate)) / float(logits_steps)


def map_manual_segments_to_tokens(
    manual_segments: list[dict[str, Any]],
    token_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    mapped = []
    for segment in manual_segments:
        start = float(segment["start_sec"])
        end = float(segment["end_sec"])
        overlaps = []
        for token in token_segments:
            overlap = max(0.0, min(end, float(token["end_sec"])) - max(start, float(token["start_sec"])))
            if overlap > 0.0:
                overlaps.append((overlap, token))
        overlaps.sort(key=lambda item: item[0], reverse=True)
        mapped.append(
            {
                "label": segment["label"],
                "start_sec": start,
                "end_sec": end,
                "duration_sec": float(segment["duration_sec"]),
                "dominant_phone": overlaps[0][1]["label"] if overlaps else None,
                "overlap_labels": [token["label"] for _, token in overlaps],
            }
        )
    return mapped


def plot_alignment(
    signal: np.ndarray,
    sample_rate: int,
    manual_segments: list[dict[str, Any]],
    token_segments: list[dict[str, Any]],
    output_path: Path,
) -> None:
    times = np.arange(signal.size) / float(sample_rate)
    figure, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)

    axes[0].plot(times, signal, color="black", linewidth=0.8)
    for segment in manual_segments:
        color = "#3ba55c" if segment["label"] == "voiced" else "#f2c14e"
        axes[0].axvspan(segment["start_sec"], segment["end_sec"], color=color, alpha=0.18)
    axes[0].set_title("Manual Voiced / Unvoiced Segments")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(times, signal, color="#444444", linewidth=0.6)
    for token in token_segments:
        axes[1].axvspan(token["start_sec"], token["end_sec"], alpha=0.18, color="#2c7fb8")
        axes[1].text(
            (token["start_sec"] + token["end_sec"]) / 2.0,
            0.85 * np.max(np.abs(signal) + 1e-6),
            token["label"],
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=8,
        )
    axes[1].set_title("Model-Derived Token / Phone Alignment")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")

    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = Path(args.audio_path)
    segments_path = Path(args.segments_json)
    segments_payload = load_segments(segments_path)
    manual_segments = segments_payload["segments"]

    from transformers import AutoModelForCTC, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModelForCTC.from_pretrained(args.model_name)
    model.eval()

    model_sample_rate = int(getattr(processor.feature_extractor, "sampling_rate", args.sample_rate or 16000))
    signal, sample_rate = load_audio(audio_path, target_sr=model_sample_rate)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    inputs = processor(signal, sampling_rate=sample_rate, return_tensors="pt")
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits[0]
    log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()

    blank_id = processor.tokenizer.pad_token_id
    if blank_id is None:
        blank_id = getattr(model.config, "pad_token_id", 0) or 0
    frame_step_seconds = infer_frame_step_seconds(model, sample_rate, log_probs.shape[0], len(signal))

    target_text = args.phone_sequence or args.transcript
    alignment_mode = "greedy"
    if target_text:
        target_ids = tokenize_alignment_text(processor, target_text)
        alignment = ctc_viterbi_align(
            log_probs=log_probs,
            target_ids=target_ids,
            blank_id=int(blank_id),
            frame_step_seconds=frame_step_seconds,
            tokenizer=processor.tokenizer,
        )
        token_segments = alignment["segments"]
        alignment_mode = "forced"
        alignment_score = alignment["best_path_score"]
    else:
        greedy_ids = np.argmax(log_probs, axis=-1)
        token_segments = decode_greedy_segments(
            token_ids=greedy_ids,
            frame_step_seconds=frame_step_seconds,
            tokenizer=processor.tokenizer,
            blank_id=int(blank_id),
        )
        alignment_score = None

    manual_boundaries = [float(segment["end_sec"]) for segment in manual_segments[:-1]]
    reference_boundaries = [float(segment["end_sec"]) for segment in token_segments[:-1]]
    rmse_seconds = nearest_boundary_rmse(manual_boundaries, reference_boundaries)
    mapped_segments = map_manual_segments_to_tokens(manual_segments, token_segments)

    output_payload = {
        "audio_path": str(audio_path),
        "segments_json": str(segments_path),
        "model_name": args.model_name,
        "sample_rate": sample_rate,
        "alignment_mode": alignment_mode,
        "alignment_score": alignment_score,
        "manual_boundaries_sec": manual_boundaries,
        "reference_boundaries_sec": reference_boundaries,
        "boundary_rmse_sec": rmse_seconds,
        "token_segments": token_segments,
        "mapped_segments": mapped_segments,
    }
    json_path = output_dir / f"{audio_path.stem}_phonetic_mapping.json"
    json_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

    plot_path = output_dir / f"{audio_path.stem}_alignment.png"
    plot_alignment(signal, sample_rate, manual_segments, token_segments, plot_path)

    print(json.dumps({"output_json": str(json_path), "plot_path": str(plot_path), "boundary_rmse_sec": rmse_seconds}, indent=2))


if __name__ == "__main__":
    main()
