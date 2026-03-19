from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full Question 1 pipeline for every sample listed in a manifest."
    )
    parser.add_argument(
        "--manifest-path",
        default=str(PROJECT_ROOT / "data" / "librispeech_subset" / "manifest.csv"),
        help="CSV manifest produced by prepare_librispeech_subset.py.",
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "outputs"),
        help="Root directory containing the stage-wise output folders.",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="*",
        default=None,
        help="Optional subset of sample IDs to run. Defaults to all rows in the manifest.",
    )
    parser.add_argument(
        "--alignment-source",
        choices=["transcript", "phone_sequence", "auto", "none"],
        default="transcript",
        help="Text source passed to phonetic_mapping.py. 'auto' prefers phone_sequence when present.",
    )
    parser.add_argument(
        "--model-name",
        default="facebook/wav2vec2-base-960h",
        help="Hugging Face CTC model used by phonetic_mapping.py.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device passed to phonetic_mapping.py, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep processing later samples even if one stage fails.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a sample if all expected output files already exist.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as manifest_file:
        return list(csv.DictReader(manifest_file))


def resolve_audio_path(manifest_path: Path, audio_path_value: str) -> Path:
    audio_path = Path(audio_path_value)
    if audio_path.is_absolute():
        return audio_path
    return (manifest_path.parent / audio_path).resolve()


def choose_alignment_payload(row: dict[str, str], alignment_source: str) -> tuple[str | None, str | None]:
    transcript = (row.get("transcript") or "").strip()
    phone_sequence = (row.get("phone_sequence") or "").strip()

    if alignment_source == "none":
        return None, None
    if alignment_source == "transcript":
        return "--transcript", transcript or None
    if alignment_source == "phone_sequence":
        return "--phone-sequence", phone_sequence or None
    if phone_sequence:
        return "--phone-sequence", phone_sequence
    return "--transcript", transcript or None


def run_step(command: list[str], log_path: Path) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    log_path.write_text(
        f"$ {' '.join(command)}\n\nSTDOUT\n{result.stdout}\n\nSTDERR\n{result.stderr}",
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(command)}. "
            f"See log: {log_path}"
        )


def expected_outputs(output_root: Path, sample_id: str) -> dict[str, Path]:
    return {
        "mfcc_summary": output_root / "mfcc_manual" / f"{sample_id}_summary.json",
        "leakage_json": output_root / "leakage_snr" / f"{sample_id}_window_metrics.json",
        "voiced_json": output_root / "voiced_unvoiced" / f"{sample_id}_voiced_unvoiced.json",
        "phonetic_json": output_root / "phonetic_mapping" / f"{sample_id}_phonetic_mapping.json",
    }


def all_outputs_exist(paths: dict[str, Path], include_phonetic: bool) -> bool:
    required_keys = ["mfcc_summary", "leakage_json", "voiced_json"]
    if include_phonetic:
        required_keys.append("phonetic_json")
    return all(paths[key].exists() for key in required_keys)


def summarize_sample(
    row: dict[str, str],
    output_paths: dict[str, Path],
    phonetic_enabled: bool,
    status: str,
    error: str | None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "sample_id": row.get("sample_id") or Path(row["audio_path"]).stem,
        "audio_path": row["audio_path"],
        "status": status,
        "error": error,
        "mfcc_summary_path": str(output_paths["mfcc_summary"]),
        "leakage_json_path": str(output_paths["leakage_json"]),
        "voiced_json_path": str(output_paths["voiced_json"]),
        "phonetic_json_path": str(output_paths["phonetic_json"]) if phonetic_enabled else None,
        "boundary_rmse_sec": None,
    }
    if phonetic_enabled and output_paths["phonetic_json"].exists():
        payload = json.loads(output_paths["phonetic_json"].read_text(encoding="utf-8"))
        summary["boundary_rmse_sec"] = payload.get("boundary_rmse_sec")
    return summary


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_path).resolve()
    output_root = Path(args.output_root).resolve()
    logs_root = output_root / "pipeline_logs"
    output_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(manifest_path)
    selected_ids = set(args.sample_ids or [])
    if selected_ids:
        rows = [row for row in rows if (row.get("sample_id") or Path(row["audio_path"]).stem) in selected_ids]

    summaries: list[dict[str, Any]] = []
    for row in rows:
        sample_id = row.get("sample_id") or Path(row["audio_path"]).stem
        audio_path = resolve_audio_path(manifest_path, row["audio_path"])
        sample_logs_dir = logs_root / sample_id
        sample_logs_dir.mkdir(parents=True, exist_ok=True)

        alignment_flag, alignment_value = choose_alignment_payload(row, args.alignment_source)
        phonetic_enabled = args.alignment_source != "none"
        outputs = expected_outputs(output_root, sample_id)

        if args.skip_existing and all_outputs_exist(outputs, include_phonetic=phonetic_enabled):
            summaries.append(
                summarize_sample(row, outputs, phonetic_enabled=phonetic_enabled, status="skipped", error=None)
            )
            continue

        status = "completed"
        error = None
        try:
            run_step(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "mfcc_manual.py"),
                    "--audio-path",
                    str(audio_path),
                    "--output-dir",
                    str(output_root / "mfcc_manual"),
                ],
                sample_logs_dir / "01_mfcc_manual.log",
            )

            run_step(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "leakage_snr.py"),
                    "--audio-path",
                    str(audio_path),
                    "--output-dir",
                    str(output_root / "leakage_snr"),
                ],
                sample_logs_dir / "02_leakage_snr.log",
            )

            run_step(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "voiced_unvoiced.py"),
                    "--audio-path",
                    str(audio_path),
                    "--output-dir",
                    str(output_root / "voiced_unvoiced"),
                ],
                sample_logs_dir / "03_voiced_unvoiced.log",
            )

            if phonetic_enabled:
                phonetic_command = [
                    sys.executable,
                    str(PROJECT_ROOT / "phonetic_mapping.py"),
                    "--audio-path",
                    str(audio_path),
                    "--segments-json",
                    str(outputs["voiced_json"]),
                    "--model-name",
                    args.model_name,
                    "--output-dir",
                    str(output_root / "phonetic_mapping"),
                ]
                if args.device:
                    phonetic_command.extend(["--device", args.device])
                if alignment_flag and alignment_value:
                    phonetic_command.extend([alignment_flag, alignment_value])

                run_step(
                    phonetic_command,
                    sample_logs_dir / "04_phonetic_mapping.log",
                )
        except Exception as exc:
            status = "failed"
            error = str(exc)
            if not args.continue_on_error:
                summaries.append(
                    summarize_sample(row, outputs, phonetic_enabled=phonetic_enabled, status=status, error=error)
                )
                summary_path = output_root / "pipeline_summary.json"
                summary_path.write_text(json.dumps({"samples": summaries}, indent=2), encoding="utf-8")
                raise

        summaries.append(
            summarize_sample(row, outputs, phonetic_enabled=phonetic_enabled, status=status, error=error)
        )

    summary_payload = {
        "manifest_path": str(manifest_path),
        "output_root": str(output_root),
        "alignment_source": args.alignment_source,
        "model_name": args.model_name if args.alignment_source != "none" else None,
        "samples": summaries,
    }
    summary_json_path = output_root / "pipeline_summary.json"
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    summary_csv_path = output_root / "pipeline_summary.csv"
    with summary_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "sample_id",
            "audio_path",
            "status",
            "boundary_rmse_sec",
            "error",
            "mfcc_summary_path",
            "leakage_json_path",
            "voiced_json_path",
            "phonetic_json_path",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({field: row.get(field) for field in fieldnames})

    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
