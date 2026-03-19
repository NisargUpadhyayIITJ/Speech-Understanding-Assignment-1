from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .audio import build_environment_catalog
from .data_utils import build_or_load_split_bundle
from .evaluation import evaluate_checkpoint, save_metrics_table
from .models import build_model
from .utils import ensure_dir, load_config


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Question 2 checkpoints and aggregate result tables.")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="One or more checkpoint paths produced by q2/train.py.",
    )
    parser.add_argument("--device", default=None, help="Optional device override, e.g. cpu or cuda.")
    parser.add_argument(
        "--environment-name",
        default="telephone",
        help="Evaluation environment for the augmented condition.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    results_dir = ensure_dir(PROJECT_ROOT / "results")
    metrics_dir = ensure_dir(results_dir / "metrics")

    metrics = []
    valid_environment_names = {item["name"] for item in build_environment_catalog()}
    if args.environment_name not in valid_environment_names:
        raise ValueError(f"Unknown environment name '{args.environment_name}'. Choose from {sorted(valid_environment_names)}.")

    for checkpoint_path in args.checkpoints:
        payload = torch.load(checkpoint_path, map_location=device)
        config = dict(payload["config"])
        config["evaluation"]["environment_name"] = args.environment_name
        split_bundle = build_or_load_split_bundle(config, refresh=False)
        model = build_model(
            config,
            num_speakers=int(payload["num_speakers"]),
            num_environments=int(payload["num_environments"]),
        ).to(device)
        model.load_state_dict(payload["state_dict"])
        metrics.append(
            evaluate_checkpoint(
                model=model,
                split_bundle=split_bundle,
                config=config,
                checkpoint_name=str(payload["experiment_name"]),
                device=device,
                results_dir=metrics_dir,
            )
        )

    save_metrics_table(metrics, results_dir)
    print(f"Saved aggregate metrics to {results_dir}")


if __name__ == "__main__":
    main()
