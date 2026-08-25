"""Regenerate notebook-compatible performance reports from saved predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from utils.reporting import (
    PROFILE_IMAGE_POLICIES,
    REPORT_PROFILES,
    save_performance_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Build dataset-aware DeepMS performance artifacts from prediction_all_modalities.csv.")
    )
    parser.add_argument("--predictions", required=True, help="Scan-level prediction CSV.")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Destination directory; defaults to the prediction CSV directory.",
    )
    parser.add_argument(
        "--report_profile",
        choices=REPORT_PROFILES,
        default="generic",
        help="Dataset and cohort contract to apply.",
    )
    parser.add_argument(
        "--cohort_overrides",
        default=None,
        help="Optional m_id/include/label_override CSV.",
    )
    parser.add_argument(
        "--image_policy",
        default=None,
        help=("Provenance label recorded in the report; defaults to the named profile's required policy."),
    )
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--target_fpr", type=float, default=0.01)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    prediction_path = Path(args.predictions).resolve()
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Prediction CSV not found: {prediction_path}")
    output_dir = Path(args.output_dir).resolve() if args.output_dir is not None else prediction_path.parent
    predictions = pd.read_csv(prediction_path, dtype={"m_id": "string"}, low_memory=False)
    image_policy = args.image_policy or PROFILE_IMAGE_POLICIES[args.report_profile] or "unspecified"
    report = save_performance_report(
        predictions,
        output_dir=output_dir,
        profile=args.report_profile,
        image_policy=image_policy,
        threshold=args.threshold,
        target_fpr=args.target_fpr,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        cohort_overrides_csv=args.cohort_overrides,
    )
    recommended_selector = report["ablation_contract"]["recommended_selector"]
    selected_results = {
        "primary": report["primary"],
        "recommended_ablation_selector": recommended_selector,
        "recommended_ablation": report["ablation_results"][recommended_selector],
        "masking_comparison": report["masking_comparison"],
    }
    print(json.dumps(selected_results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(parse_args())
