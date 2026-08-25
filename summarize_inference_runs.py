"""Create one gated performance summary after a set of inference runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from utils.reporting import REPORT_PROFILES, build_performance_report

EVALUATION_MODES = ("dataset_calibrated", "masking_raw")
EVALUATION_CONTRACTS = {
    "dataset_calibrated": {
        "result_key": "ablation_results.smri_notebook_temperature",
        "cohort": "notebook_primary",
        "ensemble": "sMRI",
        "calibration": "notebook_temperature",
        "calibration_policy": "fixed temperatures from the reference notebook",
        "allowed_profiles": {
            "generic",
            "internal",
            "krakow",
            "public_external_unmasked",
        },
    },
    "masking_raw": {
        "result_key": "masking_comparison",
        "cohort": "masking_comparable",
        "ensemble": "FLAIR",
        "calibration": "raw",
        "calibration_policy": "none; raw logits",
        "allowed_profiles": {
            "public_external_unmasked",
            "public_external_masked",
        },
    },
}
MASKING_PROFILES = (
    "public_external_unmasked",
    "public_external_masked",
)
PAIRWISE_METRICS = (
    "roc_auc",
    "pr_auc",
    "average_precision",
    "accuracy",
    "sensitivity",
    "specificity",
    "partial_auc",
    "sensitivity_at_target_fpr",
    "specificity_at_target_fpr",
)
ABLATION_MAP_ORDER = (
    "Da_smi",
    "DePar_smi",
    "DePerp_smi",
    "f_smi",
    "p2_smi",
    "ad_dti",
    "fa_dti",
    "md_dti",
    "rd_dti",
    "ak_wdki",
    "mk_wdki",
    "rk_wdki",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover completed DeepMS inference runs, recompute their "
            "notebook-compatible metrics once, and write a gated final summary."
        )
    )
    parser.add_argument(
        "--runs_root",
        required=True,
        help="Root recursively containing completed inference output directories.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Destination for the final cross-run summary artifacts.",
    )
    parser.add_argument(
        "--evaluation_mode",
        choices=EVALUATION_MODES,
        default="dataset_calibrated",
        help=(
            "dataset_calibrated uses notebook-primary calibrated sMRI; "
            "masking_raw compares paired unmasked/masked raw FLAIR results."
        ),
    )
    parser.add_argument(
        "--expected_runs",
        type=int,
        default=None,
        help="Fail unless exactly this many completed inference runs are found.",
    )
    parser.add_argument(
        "--cohort_overrides",
        default=None,
        help="Optional shared m_id/include/label_override CSV for Internal reports.",
    )
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--target_fpr", type=float, default=0.01)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read valid JSON from {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def discover_inference_runs(
    runs_root: str | Path,
    *,
    expected_runs: int | None = None,
) -> list[dict[str, Any]]:
    """Discover only inference directories with predictions and completion metadata."""
    root = Path(runs_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Inference runs root not found: {root}")
    prediction_paths = sorted(root.rglob("prediction_all_modalities.csv"))
    if not prediction_paths:
        raise ValueError(f"No prediction_all_modalities.csv files found under {root}.")
    if expected_runs is not None:
        if expected_runs <= 0:
            raise ValueError("expected_runs must be positive.")
        if len(prediction_paths) != expected_runs:
            raise ValueError(f"Expected exactly {expected_runs} inference runs, but found {len(prediction_paths)}.")

    runs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for prediction_path in prediction_paths:
        run_dir = prediction_path.parent
        relative = run_dir.relative_to(root)
        run_id = relative.as_posix() if relative.parts else run_dir.name
        if run_id in seen_ids:
            raise ValueError(f"Duplicate inference run_id discovered: {run_id}")
        seen_ids.add(run_id)

        coverage_path = run_dir / "coverage.json"
        if not coverage_path.is_file():
            raise ValueError(f"Inference run {run_id!r} is incomplete: coverage.json is missing.")
        coverage = _load_json(coverage_path)
        if coverage.get("inference_complete") is not True:
            raise ValueError(
                f"Inference run {run_id!r} is incomplete: coverage.json does not contain inference_complete=true."
            )
        profile = coverage.get("performance_report_profile")
        image_policy = coverage.get("image_policy")
        if profile not in REPORT_PROFILES:
            raise ValueError(f"Inference run {run_id!r} has an invalid report profile: {profile!r}.")
        if not isinstance(image_policy, str) or not image_policy:
            raise ValueError(f"Inference run {run_id!r} does not record a valid image_policy.")
        runs.append(
            {
                "run_id": run_id,
                "run_dir": run_dir,
                "predictions": prediction_path,
                "coverage": coverage,
                "profile": profile,
                "image_policy": image_policy,
            }
        )
    return runs


def _evaluation_contract(evaluation_mode: str) -> dict[str, Any]:
    if evaluation_mode not in EVALUATION_CONTRACTS:
        raise ValueError(f"Unknown evaluation mode: {evaluation_mode}")
    return EVALUATION_CONTRACTS[evaluation_mode]


def _select_evaluation_result(
    report: dict[str, Any],
    *,
    evaluation_mode: str,
    run_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = _evaluation_contract(evaluation_mode)
    profile = report["profile"]
    if profile not in contract["allowed_profiles"]:
        raise ValueError(
            f"Inference run {run_id!r} uses profile {profile!r}, which is not "
            f"allowed for evaluation mode {evaluation_mode!r}."
        )

    if evaluation_mode == "dataset_calibrated":
        result = report["ablation_results"]["smri_notebook_temperature"]
    else:
        result = report["masking_comparison"]
        if result is None:
            raise ValueError(f"Inference run {run_id!r} does not provide the required masking_comparison result.")
    selected = dict(result)
    expected = {
        "cohort": contract["cohort"],
        "ensemble": contract["ensemble"],
        "calibration": contract["calibration"],
        "profile": profile,
        "image_policy": report["image_policy"],
    }
    mismatches = {key: (selected.get(key), value) for key, value in expected.items() if selected.get(key) != value}
    if mismatches:
        raise ValueError(f"Inference run {run_id!r} violates the {evaluation_mode!r} result contract: {mismatches}")
    return selected, contract


def _patient_fingerprint(
    patient_predictions: pd.DataFrame,
    selected: dict[str, Any],
) -> str:
    cohort_column = f"in_{str(selected['cohort']).replace(':', '_')}"
    if cohort_column not in patient_predictions:
        raise RuntimeError(f"Patient report does not contain the selected cohort column {cohort_column!r}.")
    mask = (
        patient_predictions["ensemble"].eq(selected["ensemble"])
        & patient_predictions["calibration"].eq(selected["calibration"])
        & patient_predictions[cohort_column].astype(bool)
    )
    patients = (
        patient_predictions.loc[mask, ["m_id", "ms"]].drop_duplicates().sort_values(["m_id", "ms"], kind="stable")
    )
    if len(patients) != int(selected["n_patients"]):
        raise RuntimeError("Selected metric count does not match its patient-level membership rows.")
    payload = "".join(f"{row.m_id}\t{int(row.ms)}\n" for row in patients.itertuples(index=False))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", double_precision=15))


def _metric(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.4f}"


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _masking_comparison_unit(run_id: str, profile: str) -> str:
    prefix = f"{profile}/"
    if not run_id.startswith(prefix):
        raise ValueError(
            "masking_raw expects runs_root to contain profile directories; "
            f"run {run_id!r} does not start with {prefix!r}."
        )
    comparison_unit = run_id[len(prefix) :]
    if not comparison_unit:
        raise ValueError(f"Could not derive a masking comparison unit from run {run_id!r}.")
    return comparison_unit


def _build_masking_pairwise_deltas(selected_frame: pd.DataFrame) -> pd.DataFrame:
    frame = selected_frame.copy()
    frame["comparison_unit"] = [
        _masking_comparison_unit(run_id, profile) for run_id, profile in zip(frame["run_id"], frame["profile"])
    ]
    observed_profiles = set(frame["profile"])
    if observed_profiles != set(MASKING_PROFILES):
        raise ValueError(
            "masking_raw requires both public_external_unmasked and "
            f"public_external_masked profiles; found {sorted(observed_profiles)}."
        )

    map_order = {name: index for index, name in enumerate(ABLATION_MAP_ORDER)}
    rows: list[dict[str, Any]] = []
    for comparison_unit, group in frame.groupby("comparison_unit", sort=False):
        profiles = set(group["profile"])
        if len(group) != 2 or profiles != set(MASKING_PROFILES):
            raise ValueError(
                f"masking_raw requires exactly one unmasked and one masked run for comparison unit {comparison_unit!r}."
            )
        if group["patient_label_fingerprint"].nunique() != 1:
            raise ValueError(
                "Masked and unmasked runs do not use the exact same patients "
                f"and labels for comparison unit {comparison_unit!r}."
            )

        indexed = group.set_index("profile")
        unmasked = indexed.loc["public_external_unmasked"]
        masked = indexed.loc["public_external_masked"]
        row: dict[str, Any] = {
            "comparison_unit": comparison_unit,
            "unmasked_run_id": unmasked["run_id"],
            "masked_run_id": masked["run_id"],
            "unmasked_image_policy": unmasked["image_policy"],
            "masked_image_policy": masked["image_policy"],
            "cohort": masked["cohort"],
            "ensemble": masked["ensemble"],
            "calibration": masked["calibration"],
            "n_patients": int(masked["n_patients"]),
            "n_positive": int(masked["n_positive"]),
            "n_negative": int(masked["n_negative"]),
            "patient_label_fingerprint": masked["patient_label_fingerprint"],
            "_map_order": map_order.get(
                Path(comparison_unit).name,
                len(map_order),
            ),
        }
        for metric_name in PAIRWISE_METRICS:
            unmasked_value = _optional_float(unmasked[metric_name])
            masked_value = _optional_float(masked[metric_name])
            row[f"{metric_name}_unmasked"] = unmasked_value
            row[f"{metric_name}_masked"] = masked_value
            row[f"{metric_name}_masked_minus_unmasked"] = (
                None if unmasked_value is None or masked_value is None else masked_value - unmasked_value
            )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["_map_order", "comparison_unit"], kind="stable").drop(columns="_map_order")


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# DeepMS final inference summary",
        "",
        f"- Completed runs: {report['n_runs']}",
        f"- Evaluation mode: {report['evaluation_mode']}",
        f"- Result key: {report['result_key']}",
        f"- Calibration policy: {report['calibration_policy']}",
        "- Completeness gate: passed",
        "- Exact patient/label cohort gate: passed",
        "",
    ]
    rows = report["runs"]
    profiles = sorted({row["profile"] for row in rows})
    for profile in profiles:
        lines.extend(
            [
                f"## {profile}",
                "",
                "| Run | N | Positive | ROC-AUC | PR-AUC | Average precision | Sensitivity | Specificity |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            if row["profile"] != profile:
                continue
            safe_run_id = str(row["run_id"]).replace("|", "\\|")
            lines.append(
                "| {run_id} | {n} | {positive} | {roc} | {pr} | {ap} | {sensitivity} | {specificity} |".format(
                    run_id=safe_run_id,
                    n=row["n_patients"],
                    positive=row["n_positive"],
                    roc=_metric(row["roc_auc"]),
                    pr=_metric(row["pr_auc"]),
                    ap=_metric(row["average_precision"]),
                    sensitivity=_metric(row["sensitivity"]),
                    specificity=_metric(row["specificity"]),
                )
            )
        lines.append("")

    if report["pairwise_deltas"]:
        lines.extend(
            [
                "## Masked minus unmasked",
                "",
                "| Comparison | N | ROC-AUC unmasked | ROC-AUC masked | Delta | PR-AUC unmasked | PR-AUC masked | Delta |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in report["pairwise_deltas"]:
            safe_unit = str(row["comparison_unit"]).replace("|", "\\|")
            lines.append(
                "| {unit} | {n} | {roc_u} | {roc_m} | {roc_d} | {pr_u} | {pr_m} | {pr_d} |".format(
                    unit=safe_unit,
                    n=row["n_patients"],
                    roc_u=_metric(row["roc_auc_unmasked"]),
                    roc_m=_metric(row["roc_auc_masked"]),
                    roc_d=_metric(row["roc_auc_masked_minus_unmasked"]),
                    pr_u=_metric(row["pr_auc_unmasked"]),
                    pr_m=_metric(row["pr_auc_masked"]),
                    pr_d=_metric(row["pr_auc_masked_minus_unmasked"]),
                )
            )
        lines.append("")

    lines.extend(
        [
            "The CSV and JSON artifacts contain the complete selected metrics,",
            "per-run inventories, contract fingerprints, and all cohort-level rows.",
            "",
        ]
    )
    return "\n".join(lines)


def summarize_inference_runs(
    *,
    runs_root: str | Path,
    output_dir: str | Path,
    evaluation_mode: str = "dataset_calibrated",
    expected_runs: int | None = None,
    cohort_overrides_csv: str | Path | None = None,
    bootstrap_samples: int = 2000,
    seed: int = 42,
    threshold: float = 0.5,
    target_fpr: float = 0.01,
) -> dict[str, Any]:
    """Recompute all reports and write one cross-run, completeness-gated summary."""
    evaluation_contract = _evaluation_contract(evaluation_mode)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Final summary destination already exists: {destination}")
    runs = discover_inference_runs(runs_root, expected_runs=expected_runs)

    cohort_overrides = None
    cohort_overrides_source = None
    if cohort_overrides_csv is not None:
        override_path = Path(cohort_overrides_csv)
        if not override_path.is_file():
            raise FileNotFoundError(f"Cohort override CSV not found: {override_path}")
        cohort_overrides = pd.read_csv(
            override_path,
            dtype={"m_id": "string"},
            low_memory=False,
        )
        cohort_overrides_source = override_path.name

    selected_rows: list[dict[str, Any]] = []
    metric_tables: list[pd.DataFrame] = []
    map_order = {name: index for index, name in enumerate(ABLATION_MAP_ORDER)}
    for run in runs:
        predictions = pd.read_csv(
            run["predictions"],
            dtype={"m_id": "string"},
            low_memory=False,
        )
        expected_prediction_rows = run["coverage"].get("predicted_rows")
        if expected_prediction_rows is None or int(expected_prediction_rows) != len(predictions):
            raise ValueError(f"Inference run {run['run_id']!r} prediction row count does not match coverage.json.")
        run_cohort_overrides = None
        run_cohort_overrides_source = None
        if run["profile"] == "internal":
            configured = bool(run["coverage"].get("report_configuration", {}).get("cohort_overrides_configured", False))
            provided = cohort_overrides is not None
            if configured != provided:
                raise ValueError(
                    f"Inference run {run['run_id']!r} internal cohort-override "
                    "configuration does not match the final summary job."
                )
            run_cohort_overrides = cohort_overrides
            run_cohort_overrides_source = cohort_overrides_source
        artifacts = build_performance_report(
            predictions,
            profile=run["profile"],
            image_policy=run["image_policy"],
            threshold=threshold,
            target_fpr=target_fpr,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
            cohort_overrides=run_cohort_overrides,
            cohort_overrides_source=run_cohort_overrides_source,
        )
        selected, _ = _select_evaluation_result(
            artifacts.report,
            evaluation_mode=evaluation_mode,
            run_id=run["run_id"],
        )
        inventory = artifacts.report["prediction_inventory"]
        run_name = Path(run["run_id"]).name
        selected.update(
            {
                "run_id": run["run_id"],
                "evaluation_mode": evaluation_mode,
                "result_key": evaluation_contract["result_key"],
                "calibration_policy": evaluation_contract["calibration_policy"],
                "patient_label_fingerprint": _patient_fingerprint(
                    artifacts.patient_predictions,
                    selected,
                ),
                "modalities": ";".join(inventory["modalities"]),
                "datasets": ";".join(inventory["datasets"]),
                "n_scan_rows": inventory["n_scan_rows"],
                "primary_cohort": artifacts.report["primary"]["cohort"],
                "primary_ensemble": artifacts.report["primary"]["ensemble"],
                "primary_calibration": artifacts.report["primary"]["calibration"],
                "primary_roc_auc": artifacts.report["primary"]["roc_auc"],
                "primary_pr_auc": artifacts.report["primary"]["pr_auc"],
                "_map_order": map_order.get(run_name, len(map_order)),
            }
        )
        selected_rows.append(selected)
        metrics = artifacts.summary.copy()
        metrics.insert(0, "run_id", run["run_id"])
        metric_tables.append(metrics)

    selected_frame = pd.DataFrame(selected_rows).sort_values(
        ["profile", "_map_order", "run_id"],
        kind="stable",
    )
    if evaluation_mode == "dataset_calibrated":
        contract_columns = [
            "profile",
            "image_policy",
            "cohort",
            "ensemble",
            "calibration",
        ]
    else:
        contract_columns = ["cohort", "ensemble", "calibration"]
    contract_rows: list[dict[str, Any]] = []
    for keys, group in selected_frame.groupby(contract_columns, dropna=False):
        if group["patient_label_fingerprint"].nunique() != 1:
            run_ids = ", ".join(group["run_id"].astype(str))
            raise ValueError(
                f"Selected runs do not use the exact same patients and labels for contract {keys}: {run_ids}"
            )
        contract_rows.append(
            {
                **dict(zip(contract_columns, keys)),
                "n_runs": int(len(group)),
                "n_patients": int(group["n_patients"].iloc[0]),
                "n_positive": int(group["n_positive"].iloc[0]),
                "n_negative": int(group["n_negative"].iloc[0]),
                "patient_label_fingerprint": group["patient_label_fingerprint"].iloc[0],
            }
        )

    selected_frame = selected_frame.drop(columns="_map_order")
    all_metrics = pd.concat(metric_tables, ignore_index=True)
    pairwise_deltas = _build_masking_pairwise_deltas(selected_frame) if evaluation_mode == "masking_raw" else None
    artifact_names = [
        "ablation_performance_summary.csv",
        "ablation_performance_metrics.csv",
        "ablation_performance_report.json",
        "ablation_performance_report.md",
        "_SUCCESS",
    ]
    if pairwise_deltas is not None:
        artifact_names.insert(1, "masking_pairwise_deltas.csv")

    report = {
        "schema_version": 2,
        "evaluation_mode": evaluation_mode,
        "result_key": evaluation_contract["result_key"],
        "calibration_policy": evaluation_contract["calibration_policy"],
        "n_runs": int(len(selected_frame)),
        "expected_runs": expected_runs,
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": seed,
            "confidence_level": 0.95,
        },
        "completeness_gate": "passed",
        "patient_label_contract_gate": "passed",
        "contracts": contract_rows,
        "runs": _json_records(selected_frame),
        "pairwise_delta_definition": ("masked minus unmasked" if pairwise_deltas is not None else None),
        "pairwise_deltas": (_json_records(pairwise_deltas) if pairwise_deltas is not None else []),
        "artifacts": artifact_names,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    try:
        selected_frame.to_csv(
            staging / "ablation_performance_summary.csv",
            index=False,
        )
        if pairwise_deltas is not None:
            pairwise_deltas.to_csv(
                staging / "masking_pairwise_deltas.csv",
                index=False,
            )
        all_metrics.to_csv(
            staging / "ablation_performance_metrics.csv",
            index=False,
        )
        with (staging / "ablation_performance_report.json").open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        (staging / "ablation_performance_report.md").write_text(
            _render_markdown(report),
            encoding="utf-8",
        )
        (staging / "_SUCCESS").write_text("complete\n", encoding="utf-8")
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def main(args: argparse.Namespace) -> None:
    report = summarize_inference_runs(
        runs_root=args.runs_root,
        output_dir=args.output_dir,
        evaluation_mode=args.evaluation_mode,
        expected_runs=args.expected_runs,
        cohort_overrides_csv=args.cohort_overrides,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        threshold=args.threshold,
        target_fpr=args.target_fpr,
    )
    print(
        json.dumps(
            {
                "n_runs": report["n_runs"],
                "evaluation_mode": report["evaluation_mode"],
                "result_key": report["result_key"],
                "completeness_gate": report["completeness_gate"],
                "patient_label_contract_gate": report["patient_label_contract_gate"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main(parse_args())
