"""Deterministic prediction coverage checks and patient-level aggregation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from utils.analysis import (
    avg_logits_ensemble,
    grouped_avg_prob_ensemble,
    grouped_avg_prob_ensemble_smri,
)

HIERARCHY = {
    "FLAIR": ["3DFLAIR_NCE", "3DFLAIR_CE", "2DFLAIR_NCE", "2DFLAIR_CE"],
    "T1": ["3DT1_NCE", "2DT1_NCE"],
    "T1CE": ["3DT1_CE", "2DT1_CE"],
    "b0": ["b0"],
    "DTI": ["fa_dti", "md_dti", "ad_dti", "rd_dti"],
    "SMI": ["f_smi", "p2_smi", "DePerp_smi", "DePar_smi", "Da_smi"],
    "DKI": ["ak_wdki", "mk_wdki", "rk_wdki"],
}

SECOND_LEVEL_HIERARCHY = {
    "sMRI": ["FLAIR", "T1", "T1CE", "b0"],
    "dMRI": ["DTI", "DKI", "SMI"],
}


def _finite_mean(values: Sequence[float]) -> float:
    """Return a mean over finite values, or NaN when none are available."""
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else np.nan


def binary_metrics(y_true: Sequence[int], y_prob: Sequence[float]) -> dict[str, float | int]:
    """Compute deterministic binary metrics without failing on a one-class subset."""
    labels = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(y_prob, dtype=float)
    valid = np.isfinite(labels) & np.isfinite(probabilities)
    labels = labels[valid]
    probabilities = probabilities[valid]

    if len(labels) == 0:
        return {
            "accuracy": np.nan,
            "auc": np.nan,
            "pr_auc": np.nan,
            "average_precision": np.nan,
            "count": 0,
        }

    metrics: dict[str, float | int] = {
        "accuracy": float(accuracy_score(labels, probabilities >= 0.5)),
        "auc": np.nan,
        "pr_auc": np.nan,
        "average_precision": np.nan,
        "count": int(len(labels)),
    }
    if len(np.unique(labels)) >= 2:
        metrics["auc"] = float(roc_auc_score(labels, probabilities))
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        metrics["pr_auc"] = float(auc(recall, precision))
        metrics["average_precision"] = float(average_precision_score(labels, probabilities))
    return metrics


def validate_prediction_coverage(
    predictions: pd.DataFrame,
    *,
    expected_row_ids: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Validate one prediction per source row and consistent patient labels."""
    required = {"row_id", "m_id", "modality", "ms", "ms_prob"}
    missing_columns = sorted(required - set(predictions.columns))
    if missing_columns:
        raise ValueError(f"Prediction frame is missing columns: {missing_columns}")

    frame = predictions.copy()
    if frame["row_id"].isna().any():
        raise ValueError("Prediction frame contains missing row_id values.")
    frame["row_id"] = frame["row_id"].astype(int)
    frame["m_id"] = frame["m_id"].astype(str)

    duplicate_count = int(frame["row_id"].duplicated(keep=False).sum())
    if duplicate_count:
        raise ValueError(f"Prediction frame contains {duplicate_count} duplicate row_id rows.")

    if expected_row_ids is not None:
        expected = {int(value) for value in expected_row_ids}
        actual = set(frame["row_id"].tolist())
        missing = expected - actual
        unexpected = actual - expected
        if missing or unexpected:
            raise ValueError(
                "Prediction coverage mismatch: "
                f"expected={len(expected)}, actual={len(actual)}, "
                f"missing={len(missing)}, unexpected={len(unexpected)}."
            )

    invalid_labels = ~frame["ms"].isin([0, 1])
    if invalid_labels.any():
        raise ValueError(f"Prediction frame contains {int(invalid_labels.sum())} non-binary labels.")

    inconsistent = frame.groupby("m_id", sort=False)["ms"].nunique(dropna=False) > 1
    if inconsistent.any():
        raise ValueError(
            f"Prediction frame contains {int(inconsistent.sum())} patients with conflicting labels."
        )

    probabilities = pd.to_numeric(frame["ms_prob"], errors="coerce")
    probability_array = probabilities.to_numpy(dtype=float)
    if not np.isfinite(probability_array).all():
        raise ValueError("Prediction frame contains non-finite probabilities.")
    if ((probability_array < 0.0) | (probability_array > 1.0)).any():
        raise ValueError("Prediction frame contains probabilities outside [0, 1].")
    frame["ms_prob"] = probabilities.astype(float)

    return frame.sort_values("row_id", kind="stable").reset_index(drop=True)


def aggregate_patient_modality(predictions: pd.DataFrame) -> pd.DataFrame:
    """Average repeated scans before any cross-modality ensemble is computed."""
    frame = validate_prediction_coverage(predictions)
    aggregations: dict[str, tuple[str, str]] = {
        "ms": ("ms", "first"),
        "ms_prob": ("ms_prob", "mean"),
        "n_scans": ("row_id", "size"),
    }
    if "ms_logits" in frame.columns:
        logits = pd.to_numeric(frame["ms_logits"], errors="coerce")
        if not np.isfinite(logits.to_numpy(dtype=float)).all():
            raise ValueError("Prediction frame contains non-scalar or non-finite logits.")
        frame["ms_logits"] = logits.astype(float)
        aggregations["ms_logits"] = ("ms_logits", "mean")

    result = (
        frame.groupby(["m_id", "modality"], as_index=False, sort=False)
        .agg(**aggregations)
        .sort_values(["m_id", "modality"], kind="stable")
        .reset_index(drop=True)
    )
    result["label"] = result["ms"]
    return result


def build_inference_outputs(
    predictions: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, float | int]]]:
    """Build scan-, modality-, and patient-level outputs under explicit contracts."""
    scan_level = validate_prediction_coverage(predictions)
    patient_modality = aggregate_patient_modality(scan_level)

    flat_patient, flat_metrics_raw = avg_logits_ensemble(patient_modality, verbose=False)
    smri_patient, smri_metrics_raw = grouped_avg_prob_ensemble_smri(
        patient_modality,
        mode="ms_logits",
        verbose=False,
    )
    multimodal_patient = grouped_avg_prob_ensemble(
        patient_modality,
        print_result=False,
        return_metrics=False,
    )

    metrics = {
        "scan_level": binary_metrics(scan_level["ms"], scan_level["ms_prob"]),
        "flat_logit_patient": {
            "accuracy": flat_metrics_raw["accuracy"],
            "auc": flat_metrics_raw["roc_auc"],
            "pr_auc": flat_metrics_raw["pr_auc"],
            "average_precision": flat_metrics_raw["average_precision"],
            "count": flat_metrics_raw["n_samples"],
        },
        "smri_patient": {
            "accuracy": smri_metrics_raw["accuracy"],
            "auc": smri_metrics_raw["roc_auc"],
            "pr_auc": smri_metrics_raw["pr_auc"],
            "average_precision": smri_metrics_raw["average_precision"],
            "count": smri_metrics_raw["n_samples"],
        },
        "multimodal_two_level_patient": binary_metrics(
            multimodal_patient["ms"], multimodal_patient["ms_prob"]
        ),
    }
    outputs = {
        "scan": scan_level,
        "patient_modality": patient_modality,
        "patient_flat_logit": flat_patient,
        "patient_smri": smri_patient,
        "patient_multimodal": multimodal_patient,
    }
    return outputs, metrics


def summarize_validation_predictions(
    predictions: pd.DataFrame,
    *,
    requested_modalities: Sequence[str],
    auc_metric: str,
    expected_row_ids: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Create rank-independent validation metrics and checkpoint-selection score."""
    frame = validate_prediction_coverage(predictions, expected_row_ids=expected_row_ids)
    results: dict[str, Any] = {}

    for modality in requested_modalities:
        subset = frame[frame["modality"] == modality]
        if subset.empty:
            continue
        metrics = binary_metrics(subset["ms"], subset["ms_prob"])
        results[modality] = {
            "accuracy": metrics["accuracy"],
            "auc": metrics["auc"],
            "count": metrics["count"],
        }

    pooled = binary_metrics(frame["ms"], frame["ms_prob"])
    pooled_result = {
        "accuracy": pooled["accuracy"],
        "auc": pooled["auc"],
        "count": pooled["count"],
    }
    results["total"] = dict(pooled_result)
    results["micro_avg"] = dict(pooled_result)

    modality_results = [
        result
        for name, result in results.items()
        if name in requested_modalities and int(result["count"]) > 0
    ]
    macro_accuracy = _finite_mean(
        [float(result["accuracy"]) for result in modality_results]
    )
    macro_auc = _finite_mean([float(result["auc"]) for result in modality_results])
    results["macro_avg"] = {"accuracy": macro_accuracy, "auc": macro_auc}

    ensemble_df = grouped_avg_prob_ensemble(frame, print_result=False, return_metrics=False)
    ensemble_metrics = binary_metrics(ensemble_df["ms"], ensemble_df["ms_prob"])
    results["ensemble"] = {
        "accuracy": ensemble_metrics["accuracy"],
        "auc": ensemble_metrics["auc"],
        "count": ensemble_metrics["count"],
    }

    hierarchical_aucs: dict[str, float] = {}
    for group, modalities in HIERARCHY.items():
        values = [
            float(results[modality]["auc"])
            for modality in modalities
            if modality in results and np.isfinite(results[modality]["auc"])
        ]
        if values:
            hierarchical_aucs[group] = float(np.mean(values))

    for group, subgroups in SECOND_LEVEL_HIERARCHY.items():
        values = [hierarchical_aucs[name] for name in subgroups if name in hierarchical_aucs]
        if values:
            hierarchical_aucs[group] = float(np.mean(values))

    final_values = [
        hierarchical_aucs[name]
        for name in ("sMRI", "dMRI")
        if name in hierarchical_aucs
    ]
    hierarchical_avg_auc = float(np.mean(final_values)) if final_values else np.nan
    results["hierarchical_aucs"] = hierarchical_aucs
    results["hierarchical_avg_auc"] = hierarchical_avg_auc

    metric_values = {
        "micro": pooled_result["auc"],
        "macro": macro_auc,
        "hierarchical": hierarchical_avg_auc,
        "ensemble": ensemble_metrics["auc"],
    }
    if auc_metric not in metric_values:
        raise ValueError(f"Unknown auc_metric: {auc_metric}")
    selected = float(metric_values[auc_metric])
    results["best_metric"] = selected if np.isfinite(selected) else 0.5
    return results


def json_safe(value: Any) -> Any:
    """Convert NumPy scalars and non-finite metrics into strict JSON values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    return value
