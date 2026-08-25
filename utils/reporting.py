"""Notebook-compatible, dataset-aware inference performance reports.

The reporting contract mirrors the patient-level aggregation used by
analysis_final_1016_new.ipynb without depending on private metadata files:

1. keep scan-level predictions;
2. map structural modalities to FLAIR, T1-CE, and T1-NCE;
3. average logits directly within each patient/group;
4. average the available group logits; and
5. apply sigmoid once at patient level.

The module also makes the legacy Public External cohort definitions explicit so
masked and unmasked runs can be audited and compared on the same named subset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from utils.analysis import sigmoid
from utils.evaluation import validate_prediction_coverage

REPORT_PROFILES = (
    "generic",
    "internal",
    "krakow",
    "public_external_unmasked",
    "public_external_masked",
)

NOTEBOOK_FLAIR_MODALITIES = (
    "3DFLAIR_NCE",
    "3DFLAIR_CE",
    "2DFLAIR_NCE",
    "2DFLAIR_CE",
)
NOTEBOOK_T1_NCE_MODALITIES = ("3DT1_NCE", "2DT1_NCE")
NOTEBOOK_T1_CE_MODALITIES = ("3DT1_CE", "2DT1_CE")
NOTEBOOK_PRIMARY_SMRI_MODALITIES = (
    *NOTEBOOK_FLAIR_MODALITIES,
    "3DT1_NCE",
    *NOTEBOOK_T1_CE_MODALITIES,
)
NOTEBOOK_KRAKOW_SMRI_MODALITIES = (
    *NOTEBOOK_FLAIR_MODALITIES,
    *NOTEBOOK_T1_NCE_MODALITIES,
    *NOTEBOOK_T1_CE_MODALITIES,
)

# Public External overall cohort used in analysis_final_1016_new.ipynb.
PUBLIC_EXTERNAL_DATASETS = (
    "MSSEG2",
    "WMH",
    "ISLES-2022",
    "open_ms_cross_sectional",
    "MPI-Leipzig",
    "MSSEG-2016",
    "open_ms_longitudinal",
    "UCSF-PDGM",
    "MrBrainS18",
    "OpenNeuro-epilepsy",
    "PediMS",
    "QSM",
    "BraTS_Met",
    "MS-ISBI",
    "PediDemi",
)

# The legacy masked prediction run omitted datasets without a masked release.
PUBLIC_EXTERNAL_MASKED_DATASETS = (
    "WMH",
    "ISLES-2022",
    "open_ms_cross_sectional",
    "MSSEG-2016",
    "MPI-Leipzig",
    "UCSF-PDGM",
    "MrBrainS18",
    "OpenNeuro-epilepsy",
    "PediDemi",
    "QSM",
    "PediMS",
    "BraTS_Met",
)

# Broad WML subgroup used by prepare_df in the notebook.
PUBLIC_EXTERNAL_WML_DATASETS = (
    "MSSEG2",
    "WMH",
    "ISLES-2022",
    "open_ms_cross_sectional",
    "MSSEG-2016",
    "open_ms_longitudinal",
    "MrBrainS18",
    "PediMS",
    "QSM",
    "MS-ISBI",
    "PediDemi",
)

# Exact seven-dataset subset used for the before/after lesion-masking figure.
PUBLIC_EXTERNAL_MASKING_COMPARABLE_DATASETS = (
    "WMH",
    "open_ms_cross_sectional",
    "MSSEG-2016",
    "MrBrainS18",
    "PediMS",
    "PediDemi",
    "ISLES-2022",
)

# CIS cases explicitly excluded by the legacy Public External preparation code.
PUBLIC_EXTERNAL_EXCLUDED_CASES = (
    "open_ms_cross_sectional_patient02",
    "open_ms_cross_sectional_patient03",
    "open_ms_cross_sectional_patient29",
    "open_ms_longitudinal_patient01",
    "open_ms_longitudinal_patient04",
    "open_ms_longitudinal_patient09",
    "open_ms_longitudinal_patient10",
    "open_ms_longitudinal_patient12",
)

NOTEBOOK_TEMPERATURES = {
    "3dflair": 1.73,
    "2dflair": 1.82,
    "3dt1_ce": 1.81,
    "3dt1_nce": 1.29,
    "2dt1_ce": 1.87,
    "2dt1_nce": 1.10,
}

REPORT_METADATA_COLUMNS = (
    "patient_id",
    "dataset",
    "diagnosis",
    "final_diagnosis",
    "Sex",
    "Age",
    "wm_lesion",
    "subtype",
    "masked_image_available",
    "used_masked_image",
    "used_preprocessing_fallback",
    "image_source",
)


@dataclass(frozen=True)
class ProfileSpec:
    primary_datasets: tuple[str, ...] | None
    wml_datasets: tuple[str, ...] | None
    masking_datasets: tuple[str, ...] | None
    smri_modalities: tuple[str, ...] | None
    require_3d_flair: bool
    primary_cohort: str
    primary_ensemble: str
    primary_calibration: str


PROFILE_SPECS = {
    "generic": ProfileSpec(None, None, None, None, False, "manifest_all", "sMRI", "raw"),
    "internal": ProfileSpec(
        None,
        None,
        None,
        NOTEBOOK_PRIMARY_SMRI_MODALITIES,
        False,
        "notebook_primary",
        "sMRI",
        "notebook_temperature",
    ),
    "krakow": ProfileSpec(
        None,
        None,
        None,
        NOTEBOOK_KRAKOW_SMRI_MODALITIES,
        True,
        "notebook_primary",
        "sMRI",
        "notebook_temperature",
    ),
    "public_external_unmasked": ProfileSpec(
        PUBLIC_EXTERNAL_DATASETS,
        PUBLIC_EXTERNAL_WML_DATASETS,
        PUBLIC_EXTERNAL_MASKING_COMPARABLE_DATASETS,
        NOTEBOOK_PRIMARY_SMRI_MODALITIES,
        False,
        "notebook_primary",
        "sMRI",
        "notebook_temperature",
    ),
    "public_external_masked": ProfileSpec(
        PUBLIC_EXTERNAL_MASKED_DATASETS,
        PUBLIC_EXTERNAL_WML_DATASETS,
        PUBLIC_EXTERNAL_MASKING_COMPARABLE_DATASETS,
        NOTEBOOK_PRIMARY_SMRI_MODALITIES,
        False,
        "masking_comparable",
        "FLAIR",
        "raw",
    ),
}

PROFILE_IMAGE_POLICIES = {
    "generic": None,
    "internal": "preprocessing",
    "krakow": "preprocessing",
    "public_external_unmasked": "preprocessing",
    "public_external_masked": "masked_image_path_then_preprocessing",
}


def validate_report_image_policy(profile: str, image_policy: str) -> None:
    """Fail when a named profile is paired with the wrong image source."""
    if profile not in PROFILE_SPECS:
        raise ValueError(f"Unknown report profile: {profile}")
    expected = PROFILE_IMAGE_POLICIES[profile]
    if expected is not None and image_policy != expected:
        raise ValueError(f"Report profile {profile!r} requires image_policy {expected!r}; got {image_policy!r}.")


@dataclass
class PerformanceArtifacts:
    report: dict[str, Any]
    summary: pd.DataFrame
    patient_predictions: pd.DataFrame


def _strict_json(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, dict):
        return {str(key): _strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json(item) for item in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _notebook_temperature_key(modality: str) -> str:
    modality_lower = str(modality).lower()
    if "3dflair" in modality_lower:
        return "3dflair"
    if "2dflair" in modality_lower:
        return "2dflair"
    return modality_lower


def _smri_group(modality: str) -> str | None:
    modality_lower = str(modality).lower()
    if "flair" in modality_lower:
        return "flair"
    if "t1_ce" in modality_lower:
        return "t1_ce"
    if "t1_nce" in modality_lower:
        return "t1_nce"
    return None


def _boolean_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    numeric = pd.to_numeric(values, errors="coerce")
    text = values.astype("string").str.strip().str.lower()
    return numeric.fillna(0).ne(0) | text.isin({"true", "yes", "y"})


def attach_manifest_metadata(
    predictions: pd.DataFrame,
    manifest_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Attach non-path cohort metadata to predictions by stable row_id."""
    validated = validate_prediction_coverage(predictions)
    if "row_id" not in manifest_rows:
        raise ValueError("Manifest rows must contain row_id for metadata attachment.")
    if manifest_rows["row_id"].duplicated().any():
        raise ValueError("Manifest rows contain duplicate row_id values.")

    columns = [column for column in REPORT_METADATA_COLUMNS if column in manifest_rows]
    if not columns:
        return validated

    metadata = manifest_rows[["row_id", *columns]].copy()
    overlapping = [column for column in columns if column in validated]
    if overlapping:
        validated = validated.drop(columns=overlapping)
    merged = validated.merge(metadata, on="row_id", how="left", validate="one_to_one")
    return validate_prediction_coverage(merged)


def _apply_cohort_overrides(
    frame: pd.DataFrame,
    cohort_overrides: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    stats = {
        "rows_excluded": 0,
        "patients_excluded": 0,
        "labels_overridden_patients": 0,
    }
    if cohort_overrides is None or cohort_overrides.empty:
        return frame, stats
    if "m_id" not in cohort_overrides:
        raise ValueError("Cohort override CSV must contain m_id.")

    overrides = cohort_overrides.copy()
    overrides["m_id"] = overrides["m_id"].astype("string").str.strip()
    if overrides["m_id"].isna().any() or overrides["m_id"].eq("").any():
        raise ValueError("Cohort override CSV contains an empty m_id.")
    if overrides["m_id"].duplicated().any():
        raise ValueError("Cohort override CSV contains duplicate m_id values.")
    if not {"include", "label_override"} & set(overrides):
        raise ValueError("Cohort override CSV must contain include and/or label_override.")

    result = frame.copy()
    if "include" in overrides:
        include = pd.to_numeric(overrides["include"], errors="coerce")
        invalid = include.notna() & ~include.isin([0, 1])
        if invalid.any():
            raise ValueError("Cohort override include values must be 0 or 1.")
        excluded_ids = set(overrides.loc[include.eq(0), "m_id"].astype(str))
        excluded = result["m_id"].astype(str).isin(excluded_ids)
        stats["rows_excluded"] = int(excluded.sum())
        stats["patients_excluded"] = int(result.loc[excluded, "m_id"].astype(str).nunique())
        result = result.loc[~excluded].copy()

    if "label_override" in overrides:
        labels = pd.to_numeric(overrides["label_override"], errors="coerce")
        invalid = labels.notna() & ~labels.isin([0, 1])
        if invalid.any():
            raise ValueError("Cohort label_override values must be 0 or 1.")
        label_map = dict(
            zip(
                overrides.loc[labels.notna(), "m_id"].astype(str),
                labels.loc[labels.notna()].astype(int),
            )
        )
        mapped = result["m_id"].astype(str).map(label_map)
        overridden = mapped.notna()
        result.loc[overridden, "ms"] = mapped.loc[overridden].astype(int)
        result.loc[overridden, "label"] = mapped.loc[overridden].astype(int)
        stats["labels_overridden_patients"] = int(result.loc[overridden, "m_id"].astype(str).nunique())
    return result, stats


def normalize_report_predictions(
    predictions: pd.DataFrame,
    *,
    profile: str,
    cohort_overrides: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Normalize a current or historical prediction CSV for reporting."""
    if profile not in PROFILE_SPECS:
        raise ValueError(f"Unknown report profile: {profile}")

    frame = predictions.copy()
    if "row_id" not in frame:
        frame["row_id"] = np.arange(len(frame), dtype=np.int64)
    if "ms" not in frame and "label" in frame:
        frame["ms"] = frame["label"]
    if "label" not in frame and "ms" in frame:
        frame["label"] = frame["ms"]
    if "ms_prob" not in frame and "ms_logits" in frame:
        logits = pd.to_numeric(frame["ms_logits"], errors="coerce").to_numpy(dtype=float)
        frame["ms_prob"] = sigmoid(logits)

    frame, override_stats = _apply_cohort_overrides(frame, cohort_overrides)
    frame = validate_prediction_coverage(frame)
    if "ms_logits" not in frame:
        eps = np.finfo(float).eps
        probabilities = frame["ms_prob"].clip(eps, 1.0 - eps)
        frame["ms_logits"] = np.log(probabilities / (1.0 - probabilities))
    frame["ms_logits"] = pd.to_numeric(frame["ms_logits"], errors="coerce")
    if not np.isfinite(frame["ms_logits"].to_numpy(dtype=float)).all():
        raise ValueError("Prediction frame contains non-finite ms_logits.")

    frame["source_m_id"] = frame["m_id"].astype(str)
    frame["report_m_id"] = frame["source_m_id"]
    if profile.startswith("public_external") and {"dataset", "patient_id"} <= set(frame):
        pediatric = frame["dataset"].eq("PediMS") & frame["patient_id"].notna()
        normalized_ids = frame.loc[pediatric, "patient_id"].astype(str).str.strip()
        frame.loc[pediatric, "report_m_id"] = normalized_ids

    inconsistent = frame.groupby("report_m_id", sort=False)["ms"].nunique(dropna=False) > 1
    if inconsistent.any():
        raise ValueError(
            f"Report identifier normalization produced {int(inconsistent.sum())} patients with conflicting labels."
        )

    mask_provenance_available = "used_masked_image" in frame
    if "dataset" in frame:
        frame["dataset"] = frame["dataset"].astype("string").fillna("Unknown")
    else:
        frame["dataset"] = "Unknown"

    if "masked_image_available" not in frame:
        frame["masked_image_available"] = _boolean_column(frame, "mask_path")
    else:
        frame["masked_image_available"] = _boolean_column(frame, "masked_image_available")
    frame["used_masked_image"] = _boolean_column(frame, "used_masked_image")
    frame["used_preprocessing_fallback"] = _boolean_column(frame, "used_preprocessing_fallback")
    frame.attrs["cohort_overrides"] = override_stats
    frame.attrs["mask_provenance_available"] = mask_provenance_available
    return frame


def notebook_patient_predictions(
    predictions: pd.DataFrame,
    *,
    ensemble: str,
    calibration: str,
    threshold: float,
    allowed_modalities: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Apply the notebook's direct scan-row grouped-logit aggregation."""
    if ensemble not in {"FLAIR", "sMRI"}:
        raise ValueError("ensemble must be FLAIR or sMRI.")
    if calibration not in {"raw", "notebook_temperature"}:
        raise ValueError("calibration must be raw or notebook_temperature.")

    frame = predictions.copy()
    if allowed_modalities is not None:
        frame = frame[frame["modality"].isin(allowed_modalities)].copy()
    frame["modality_group"] = frame["modality"].map(_smri_group)
    frame = frame[frame["modality_group"].notna()].copy()
    if ensemble == "FLAIR":
        frame = frame[frame["modality_group"].eq("flair")].copy()
    if frame.empty:
        return pd.DataFrame()

    frame["report_score"] = frame["ms_logits"].astype(float)
    if calibration == "notebook_temperature":
        temperature_keys = frame["modality"].map(_notebook_temperature_key)
        temperatures = temperature_keys.map(NOTEBOOK_TEMPERATURES).fillna(1.0)
        frame["report_score"] = frame["report_score"] / temperatures

    grouped_scores = pd.pivot_table(
        frame,
        values="report_score",
        index="report_m_id",
        columns="modality_group",
        aggfunc="mean",
    )
    grouped_scores.columns = [f"{column}_score" for column in grouped_scores.columns]
    for column in ("flair_score", "t1_ce_score", "t1_nce_score"):
        if column not in grouped_scores:
            grouped_scores[column] = np.nan

    grouped = frame.groupby("report_m_id", sort=False)
    labels = grouped["ms"].first().rename("ms")
    counts = grouped.agg(
        n_rows=("row_id", "size"),
        n_modalities=("modality", "nunique"),
        n_source_m_ids=("source_m_id", "nunique"),
        masked_available_rows=("masked_image_available", "sum"),
        used_masked_rows=("used_masked_image", "sum"),
        preprocessing_fallback_rows=("used_preprocessing_fallback", "sum"),
    )
    metadata_columns = [
        column
        for column in (
            "dataset",
            "patient_id",
            "diagnosis",
            "final_diagnosis",
            "Sex",
            "Age",
            "wm_lesion",
            "subtype",
        )
        if column in frame
    ]
    metadata = grouped[metadata_columns].first() if metadata_columns else pd.DataFrame(index=labels.index)

    patient = pd.concat([labels, grouped_scores, counts, metadata], axis=1)
    target_columns = ["flair_score"] if ensemble == "FLAIR" else ["flair_score", "t1_ce_score", "t1_nce_score"]
    patient["mean_score"] = patient[target_columns].mean(axis=1, skipna=True)
    patient["ms_prob"] = sigmoid(patient["mean_score"].to_numpy(dtype=float))
    patient["ensemble_pred"] = (patient["ms_prob"] >= threshold).astype(int)
    patient["has_flair"] = patient["flair_score"].notna()
    patient["has_t1_ce"] = patient["t1_ce_score"].notna()
    patient["has_t1_nce"] = patient["t1_nce_score"].notna()
    patient["complete_smri_groups"] = patient[["has_flair", "has_t1_ce", "has_t1_nce"]].sum(axis=1) == 3
    patient["ensemble"] = ensemble
    patient["calibration"] = calibration
    patient.index.name = "m_id"
    return patient.reset_index().sort_values("m_id", kind="stable").reset_index(drop=True)


def _profile_base_ids(
    frame: pd.DataFrame,
    spec: ProfileSpec,
    *,
    profile: str,
) -> set[str]:
    selected = frame.copy()
    if spec.primary_datasets is not None:
        selected = selected[selected["dataset"].isin(spec.primary_datasets)]
    if selected.empty:
        return set()
    if profile.startswith("public_external"):
        selected = selected[~selected["report_m_id"].isin(PUBLIC_EXTERNAL_EXCLUDED_CASES)]
    if spec.require_3d_flair:
        flair_ids = selected.loc[selected["modality"].eq("3DFLAIR_NCE"), "report_m_id"]
        selected = selected[selected["report_m_id"].isin(flair_ids)]
    return set(selected["report_m_id"].astype(str))


def _cohort_masks(
    patient: pd.DataFrame,
    *,
    frame: pd.DataFrame,
    profile: str,
) -> dict[str, pd.Series]:
    spec = PROFILE_SPECS[profile]
    base_ids = _profile_base_ids(frame, spec, profile=profile)
    masks: dict[str, pd.Series] = {
        "manifest_all": pd.Series(True, index=patient.index, dtype=bool),
        "notebook_primary": patient["m_id"].isin(base_ids),
    }

    if spec.wml_datasets is not None:
        masks["notebook_wml_subgroup"] = masks["notebook_primary"] & patient["dataset"].isin(spec.wml_datasets)
    elif "wm_lesion" in patient:
        masks["notebook_wml_subgroup"] = masks["notebook_primary"] & pd.to_numeric(
            patient["wm_lesion"], errors="coerce"
        ).eq(1)
    elif profile == "krakow" and "diagnosis" in patient:
        masks["notebook_wml_subgroup"] = masks["notebook_primary"] & patient["diagnosis"].isin(["MS", "WML"])

    if spec.masking_datasets is not None:
        masks["masking_comparable"] = masks["notebook_primary"] & patient["dataset"].isin(spec.masking_datasets)

    if not patient["dataset"].eq("Unknown").all():
        for dataset in sorted(patient["dataset"].dropna().astype(str).unique()):
            masks[f"dataset:{dataset}"] = masks["notebook_primary"] & patient["dataset"].astype(str).eq(dataset)
    return masks


def _bootstrap_intervals(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    keys = ("roc_auc", "pr_auc", "average_precision", "sensitivity", "specificity")
    empty = {f"{key}_ci_{bound}": None for key in keys for bound in ("low", "high")}
    empty["bootstrap_valid_samples"] = 0
    if samples <= 0 or len(labels) == 0 or len(np.unique(labels)) < 2:
        return empty

    rng = np.random.RandomState(seed)
    values: dict[str, list[float]] = {key: [] for key in keys}
    size = len(labels)
    for _ in range(samples):
        indices = rng.randint(0, size, size)
        sampled_labels = labels[indices]
        if len(np.unique(sampled_labels)) < 2:
            continue
        sampled_probabilities = probabilities[indices]
        predictions = sampled_probabilities >= threshold
        positive = sampled_labels == 1
        negative = ~positive
        precision, recall, _ = precision_recall_curve(sampled_labels, sampled_probabilities)
        values["roc_auc"].append(float(roc_auc_score(sampled_labels, sampled_probabilities)))
        values["pr_auc"].append(float(auc(recall, precision)))
        values["average_precision"].append(float(average_precision_score(sampled_labels, sampled_probabilities)))
        values["sensitivity"].append(float(predictions[positive].mean()))
        values["specificity"].append(float((~predictions[negative]).mean()))

    valid_samples = len(values["roc_auc"])
    result = dict(empty)
    result["bootstrap_valid_samples"] = valid_samples
    if valid_samples:
        for key, metric_values in values.items():
            low, high = np.percentile(metric_values, [2.5, 97.5])
            result[f"{key}_ci_low"] = float(low)
            result[f"{key}_ci_high"] = float(high)
    return result


def patient_metrics(
    patient: pd.DataFrame,
    *,
    threshold: float,
    target_fpr: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Compute report metrics with deterministic notebook-style bootstrap CIs."""
    labels = patient["ms"].to_numpy(dtype=int)
    probabilities = patient["ms_prob"].to_numpy(dtype=float)
    count = len(patient)
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    predictions = probabilities >= threshold
    tp = int((predictions & (labels == 1)).sum())
    tn = int((~predictions & (labels == 0)).sum())
    fp = int((predictions & (labels == 0)).sum())
    fn = int((~predictions & (labels == 1)).sum())

    metrics: dict[str, Any] = {
        "n_patients": int(count),
        "n_positive": positives,
        "n_negative": negatives,
        "n_rows": int(patient["n_rows"].sum()) if count else 0,
        "n_used_masked_rows": int(patient["used_masked_rows"].sum()) if count else 0,
        "n_preprocessing_fallback_rows": (int(patient["preprocessing_fallback_rows"].sum()) if count else 0),
        "threshold": float(threshold),
        "accuracy": float((predictions == labels).mean()) if count else None,
        "sensitivity": float(tp / (tp + fn)) if positives else None,
        "specificity": float(tn / (tn + fp)) if negatives else None,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "roc_auc": None,
        "pr_auc": None,
        "average_precision": None,
        "target_fpr": float(target_fpr),
        "partial_auc": None,
        "threshold_at_target_fpr": None,
        "sensitivity_at_target_fpr": None,
        "specificity_at_target_fpr": None,
    }
    metrics.update(
        _bootstrap_intervals(
            labels,
            probabilities,
            threshold=threshold,
            samples=bootstrap_samples,
            seed=seed,
        )
    )

    if count and len(np.unique(labels)) >= 2:
        metrics["roc_auc"] = float(roc_auc_score(labels, probabilities))
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        metrics["pr_auc"] = float(auc(recall, precision))
        metrics["average_precision"] = float(average_precision_score(labels, probabilities))
        metrics["partial_auc"] = float(roc_auc_score(labels, probabilities, max_fpr=target_fpr))
        false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, probabilities)
        eligible = np.flatnonzero(false_positive_rate <= target_fpr)
        if eligible.size:
            best = eligible[np.argmax(true_positive_rate[eligible])]
            operating_threshold = float(thresholds[best])
            operating_predictions = probabilities >= operating_threshold
            operating_fp = int((operating_predictions & (labels == 0)).sum())
            operating_tn = int((~operating_predictions & (labels == 0)).sum())
            metrics["threshold_at_target_fpr"] = operating_threshold
            metrics["sensitivity_at_target_fpr"] = float(true_positive_rate[best])
            metrics["specificity_at_target_fpr"] = float(operating_tn / (operating_tn + operating_fp))
    return metrics


def _dataset_inventory(
    frame: pd.DataFrame,
    *,
    spec: ProfileSpec,
    profile: str,
) -> pd.DataFrame:
    rows = frame.groupby("dataset", as_index=False, sort=True).agg(
        n_rows=("row_id", "size"),
        n_patients=("report_m_id", "nunique"),
        n_modalities=("modality", "nunique"),
        masked_available_rows=("masked_image_available", "sum"),
        used_masked_rows=("used_masked_image", "sum"),
        preprocessing_fallback_rows=("used_preprocessing_fallback", "sum"),
    )
    patient_labels = frame.drop_duplicates(["dataset", "report_m_id"])
    label_counts = patient_labels.groupby("dataset", as_index=False).agg(
        n_positive=("ms", lambda values: int((values == 1).sum())),
        n_negative=("ms", lambda values: int((values == 0).sum())),
    )
    rows = rows.merge(label_counts, on="dataset", how="left", validate="one_to_one")

    base_ids = _profile_base_ids(frame, spec, profile=profile)
    notebook_rows = frame[frame["report_m_id"].astype(str).isin(base_ids)]
    notebook_counts = notebook_rows.groupby("dataset", as_index=False, sort=True).agg(
        n_notebook_primary_rows=("row_id", "size"),
        n_notebook_primary_patients=("report_m_id", "nunique"),
    )
    rows = rows.merge(
        notebook_counts,
        on="dataset",
        how="left",
        validate="one_to_one",
    )

    masking_rows = notebook_rows.iloc[0:0]
    if spec.masking_datasets is not None:
        masking_rows = notebook_rows[notebook_rows["dataset"].isin(spec.masking_datasets)]
    masking_counts = masking_rows.groupby("dataset", as_index=False, sort=True).agg(
        n_masking_comparable_rows=("row_id", "size"),
        n_masking_comparable_patients=("report_m_id", "nunique"),
    )
    rows = rows.merge(
        masking_counts,
        on="dataset",
        how="left",
        validate="one_to_one",
    )
    count_columns = (
        "n_notebook_primary_rows",
        "n_notebook_primary_patients",
        "n_masking_comparable_rows",
        "n_masking_comparable_patients",
    )
    rows[list(count_columns)] = rows[list(count_columns)].fillna(0).astype(int)
    rows["n_outside_notebook_primary_patients"] = rows["n_patients"] - rows["n_notebook_primary_patients"]
    rows["included_in_notebook_primary"] = (
        True if spec.primary_datasets is None else rows["dataset"].isin(spec.primary_datasets)
    )
    rows["included_in_masking_comparable"] = (
        False if spec.masking_datasets is None else rows["dataset"].isin(spec.masking_datasets)
    )
    return rows


def _select_summary_result(
    summary: pd.DataFrame,
    *,
    cohort: str,
    ensemble: str,
    calibration: str,
    result_name: str,
) -> dict[str, Any]:
    selected = summary["cohort"].eq(cohort) & summary["ensemble"].eq(ensemble) & summary["calibration"].eq(calibration)
    if int(selected.sum()) != 1:
        raise RuntimeError(
            "Could not resolve exactly one "
            f"{result_name} performance row: cohort={cohort}, "
            f"ensemble={ensemble}, calibration={calibration}."
        )
    return summary.loc[selected].iloc[0].to_dict()


def _prediction_inventory(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize prediction inputs without persisting image or checkpoint paths."""
    modality_counts = (
        frame.groupby("modality", sort=True)
        .agg(
            n_rows=("row_id", "size"),
            n_source_m_ids=("source_m_id", "nunique"),
            n_report_m_ids=("report_m_id", "nunique"),
        )
        .reset_index()
    )
    return {
        "n_scan_rows": int(len(frame)),
        "n_source_m_ids": int(frame["source_m_id"].nunique()),
        "n_report_m_ids": int(frame["report_m_id"].nunique()),
        "modalities": sorted(frame["modality"].astype(str).unique().tolist()),
        "datasets": sorted(frame["dataset"].astype(str).unique().tolist()),
        "modality_counts": modality_counts.to_dict(orient="records"),
    }


def build_performance_report(
    predictions: pd.DataFrame,
    *,
    profile: str,
    image_policy: str,
    threshold: float = 0.5,
    target_fpr: float = 0.01,
    bootstrap_samples: int = 2000,
    seed: int = 42,
    cohort_overrides: pd.DataFrame | None = None,
    cohort_overrides_source: str | None = None,
) -> PerformanceArtifacts:
    """Build long-form patient predictions, summary metrics, and report metadata."""
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be non-negative.")
    if not 0.0 < target_fpr <= 1.0:
        raise ValueError("target_fpr must be in (0, 1].")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1].")

    validate_report_image_policy(profile, image_policy)

    frame = normalize_report_predictions(
        predictions,
        profile=profile,
        cohort_overrides=cohort_overrides,
    )
    spec = PROFILE_SPECS[profile]
    inventory = _dataset_inventory(frame, spec=spec, profile=profile)
    prediction_inventory = _prediction_inventory(frame)
    patient_tables: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for calibration in ("raw", "notebook_temperature"):
        for ensemble in ("FLAIR", "sMRI"):
            patient = notebook_patient_predictions(
                frame,
                ensemble=ensemble,
                calibration=calibration,
                threshold=threshold,
                allowed_modalities=spec.smri_modalities,
            )
            if patient.empty:
                continue
            masks = _cohort_masks(patient, frame=frame, profile=profile)
            for cohort, mask in masks.items():
                patient[f"in_{cohort.replace(':', '_')}"] = mask.to_numpy(dtype=bool)
                subset = patient.loc[mask].copy()
                # The notebook only reports CIs for aggregate cohorts, not each source dataset.
                samples = 0 if cohort.startswith("dataset:") else bootstrap_samples
                metrics = patient_metrics(
                    subset,
                    threshold=threshold,
                    target_fpr=target_fpr,
                    bootstrap_samples=samples,
                    seed=seed,
                )
                summary_rows.append(
                    {
                        "profile": profile,
                        "image_policy": image_policy,
                        "cohort": cohort,
                        "dataset": cohort.partition(":")[2] or None,
                        "ensemble": ensemble,
                        "calibration": calibration,
                        **metrics,
                    }
                )
            patient["profile"] = profile
            patient["image_policy"] = image_policy
            patient_tables.append(patient)

    if not patient_tables:
        raise ValueError("No notebook-compatible structural MRI predictions were available for performance reporting.")
    summary = pd.DataFrame(summary_rows)
    patients = pd.concat(patient_tables, ignore_index=True)
    primary = _select_summary_result(
        summary,
        cohort=spec.primary_cohort,
        ensemble=spec.primary_ensemble,
        calibration=spec.primary_calibration,
        result_name="primary",
    )

    ablation_results = {
        "flair_raw": _select_summary_result(
            summary,
            cohort="notebook_primary",
            ensemble="FLAIR",
            calibration="raw",
            result_name="raw FLAIR ablation",
        ),
        "flair_notebook_temperature": _select_summary_result(
            summary,
            cohort="notebook_primary",
            ensemble="FLAIR",
            calibration="notebook_temperature",
            result_name="temperature-scaled FLAIR ablation",
        ),
        "smri_raw": _select_summary_result(
            summary,
            cohort="notebook_primary",
            ensemble="sMRI",
            calibration="raw",
            result_name="raw sMRI ablation",
        ),
        "smri_notebook_temperature": _select_summary_result(
            summary,
            cohort="notebook_primary",
            ensemble="sMRI",
            calibration="notebook_temperature",
            result_name="temperature-scaled sMRI ablation",
        ),
    }

    masking_comparison = None
    if "masking_comparable" in set(summary["cohort"]):
        masking_comparison = _select_summary_result(
            summary,
            cohort="masking_comparable",
            ensemble="FLAIR",
            calibration="raw",
            result_name="masking comparison",
        )

    mask_provenance_available = bool(frame.attrs["mask_provenance_available"])
    if mask_provenance_available and profile == "public_external_unmasked":
        if primary["n_used_masked_rows"] != 0:
            raise ValueError("The Public External unmasked report contains rows marked as using lesion-masked images.")
    if mask_provenance_available and profile == "public_external_masked":
        if primary["n_rows"] == 0 or primary["n_used_masked_rows"] != primary["n_rows"]:
            raise ValueError(
                "The Public External masked primary cohort must use an explicit "
                "masked image for every contributing FLAIR row; fallback rows "
                "are not valid for the seven-dataset comparison."
            )

    report = {
        "schema_version": 1,
        "reference_contract": "ms-diagnosis/analysis/data_analysis/analysis_final_1016_new.ipynb",
        "profile": profile,
        "image_policy": image_policy,
        "aggregation_contract": {
            "unit": "patient",
            "score": "ms_logits",
            "scan_aggregation": "direct mean within patient and FLAIR/T1-CE/T1-NCE group",
            "group_aggregation": "equal mean across available groups",
            "probability_conversion": "sigmoid once after group-logit averaging",
            "threshold": threshold,
        },
        "temperature_scaling": {
            "reported_modes": ["raw", "notebook_temperature"],
            "temperatures": NOTEBOOK_TEMPERATURES,
            "biases": "all zero",
        },
        "ablation_contract": {
            "inference_inputs": "structural MRI only",
            "recommended_selector": "smri_notebook_temperature",
            "dataset_evaluation_selector": "smri_notebook_temperature",
            "masking_evaluation_selector": "masking_comparison",
            "available_selectors": list(ablation_results),
            "comparison_rule": (
                "dataset evaluation uses fixed notebook-temperature sMRI on "
                "notebook_primary; lesion masking uses raw FLAIR on the exact "
                "masking_comparable cohort"
            ),
        },
        "cohort_overrides": {
            "source": cohort_overrides_source,
            **frame.attrs["cohort_overrides"],
        },
        "cohort_definitions": {
            "public_external_primary": list(spec.primary_datasets or []),
            "public_external_wml_subgroup": list(spec.wml_datasets or []),
            "public_external_masking_comparable": list(spec.masking_datasets or []),
            "excluded_cases": (list(PUBLIC_EXTERNAL_EXCLUDED_CASES) if profile.startswith("public_external") else []),
            "krakow_requires_3dflair_nce": spec.require_3d_flair,
            "structural_modalities": list(spec.smri_modalities or []),
            "pedims_identifier": ("patient_id" if profile.startswith("public_external") else "m_id"),
        },
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": seed,
            "confidence_level": 0.95,
            "dataset_rows_have_ci": False,
        },
        "primary": primary,
        "ablation_results": ablation_results,
        "masking_comparison": masking_comparison,
        "mask_provenance": {
            "available": mask_provenance_available,
            "primary_rows": primary["n_rows"],
            "primary_used_masked_rows": primary["n_used_masked_rows"],
            "primary_preprocessing_fallback_rows": (primary["n_preprocessing_fallback_rows"]),
        },
        "prediction_inventory": prediction_inventory,
        "dataset_inventory": inventory.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
    }
    return PerformanceArtifacts(
        report=_strict_json(report),
        summary=summary,
        patient_predictions=patients,
    )


def _format_metric(value: Any, low: Any = None, high: Any = None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    formatted = f"{float(value):.4f}"
    if low is not None and high is not None and not pd.isna(low) and not pd.isna(high):
        formatted += f" [{float(low):.4f}, {float(high):.4f}]"
    return formatted


def render_performance_markdown(report: dict[str, Any], summary: pd.DataFrame) -> str:
    """Render a compact human-readable report alongside machine-readable files."""
    primary = report["primary"]
    ablation_flair = report["ablation_results"]["flair_raw"]
    ablation_smri = report["ablation_results"]["smri_notebook_temperature"]
    aggregate = summary[~summary["cohort"].str.startswith("dataset:")].copy()
    lines = [
        "# DeepMS inference performance report",
        "",
        f"- Profile: {report['profile']}",
        f"- Image policy: {report['image_policy']}",
        f"- Reference: {report['reference_contract']}",
        (
            "- Primary result: "
            f"{primary['cohort']} / {primary['ensemble']} / {primary['calibration']}"
            f" (n={primary['n_patients']})"
        ),
        (
            "- Primary ROC-AUC: "
            + _format_metric(
                primary["roc_auc"],
                primary["roc_auc_ci_low"],
                primary["roc_auc_ci_high"],
            )
        ),
        (
            "- Primary PR-AUC: "
            + _format_metric(
                primary["pr_auc"],
                primary["pr_auc_ci_low"],
                primary["pr_auc_ci_high"],
            )
        ),
        (
            "- Ablation FLAIR result: notebook_primary / FLAIR / raw "
            f"(n={ablation_flair['n_patients']}, "
            f"ROC-AUC={_format_metric(ablation_flair['roc_auc'])})"
        ),
        (
            "- Dataset evaluation sMRI result: notebook_primary / sMRI / notebook_temperature "
            f"(n={ablation_smri['n_patients']}, "
            f"ROC-AUC={_format_metric(ablation_smri['roc_auc'])})"
        ),
        "",
        "## Aggregate cohorts",
        "",
        "| Cohort | Ensemble | Calibration | N | Positive | ROC-AUC | PR-AUC | Sensitivity | Specificity |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate.to_dict(orient="records"):
        lines.append(
            "| {cohort} | {ensemble} | {calibration} | {n_patients} | "
            "{n_positive} | {roc} | {pr} | {sensitivity} | {specificity} |".format(
                cohort=row["cohort"],
                ensemble=row["ensemble"],
                calibration=row["calibration"],
                n_patients=row["n_patients"],
                n_positive=row["n_positive"],
                roc=_format_metric(row["roc_auc"], row["roc_auc_ci_low"], row["roc_auc_ci_high"]),
                pr=_format_metric(row["pr_auc"], row["pr_auc_ci_low"], row["pr_auc_ci_high"]),
                sensitivity=_format_metric(row["sensitivity"]),
                specificity=_format_metric(row["specificity"]),
            )
        )
    lines.extend(
        [
            "",
            "The CSV and JSON files contain the full per-dataset breakdown, operating-point",
            "metrics, partial AUC at the configured target FPR, and exact cohort definitions.",
            "",
        ]
    )
    return "\n".join(lines)


def save_performance_report(
    predictions: pd.DataFrame,
    *,
    output_dir: str | Path,
    profile: str,
    image_policy: str,
    threshold: float = 0.5,
    target_fpr: float = 0.01,
    bootstrap_samples: int = 2000,
    seed: int = 42,
    cohort_overrides_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Build and persist all notebook-compatible report artifacts."""
    cohort_overrides = None
    cohort_overrides_source = None
    if cohort_overrides_csv is not None:
        override_path = Path(cohort_overrides_csv)
        if not override_path.is_file():
            raise FileNotFoundError(f"Cohort override CSV not found: {override_path}")
        cohort_overrides = pd.read_csv(override_path, dtype={"m_id": "string"}, low_memory=False)
        cohort_overrides_source = override_path.name
    artifacts = build_performance_report(
        predictions,
        profile=profile,
        image_policy=image_policy,
        threshold=threshold,
        target_fpr=target_fpr,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        cohort_overrides=cohort_overrides,
        cohort_overrides_source=cohort_overrides_source,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    artifacts.summary.to_csv(destination / "performance_summary.csv", index=False)
    artifacts.patient_predictions.to_csv(destination / "prediction_patient_report.csv", index=False)
    with (destination / "performance_report.json").open("w", encoding="utf-8") as handle:
        json.dump(artifacts.report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    markdown = render_performance_markdown(artifacts.report, artifacts.summary)
    (destination / "performance_report.md").write_text(markdown, encoding="utf-8")
    return artifacts.report
