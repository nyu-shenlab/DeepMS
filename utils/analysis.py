from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    auc,
)


DEFAULT_ID_COL = "m_id"
DEFAULT_LABEL_COL = "ms"
DEFAULT_PROB_COL = "ms_prob"
DEFAULT_LOGIT_COL = "ms_logits"
DEFAULT_MODALITY_COL = "modality"


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def _validate_columns(df: pd.DataFrame, required_cols: Sequence[str]) -> None:
    """Raise ValueError if required columns are missing."""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _safe_binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute binary classification metrics safely.

    Returns
    -------
    metrics : dict
        Dictionary containing:
        - accuracy
        - roc_auc
        - pr_auc
        - average_precision
        - n_samples
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    valid_mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[valid_mask]
    y_score = y_score[valid_mask]

    if len(y_true) == 0:
        return {
            "accuracy": np.nan,
            "roc_auc": np.nan,
            "pr_auc": np.nan,
            "average_precision": np.nan,
            "n_samples": 0,
        }

    y_pred = (y_score >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "average_precision": np.nan,
        "n_samples": int(len(y_true)),
    }

    # ROC-AUC / PR-AUC require both classes to be present
    if len(np.unique(y_true)) >= 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        metrics["pr_auc"] = float(auc(recall, precision))
        metrics["average_precision"] = float(average_precision_score(y_true, y_score))

    return metrics


def _print_metrics(metrics: Dict[str, float], prefix: str = "[Ensemble]") -> None:
    """Pretty-print metrics."""
    print(f"{prefix} n_samples:          {metrics['n_samples']}")
    print(f"{prefix} Accuracy:           {metrics['accuracy']:.4f}" if not np.isnan(metrics["accuracy"]) else f"{prefix} Accuracy:           undefined")
    print(f"{prefix} ROC-AUC:            {metrics['roc_auc']:.4f}" if not np.isnan(metrics["roc_auc"]) else f"{prefix} ROC-AUC:            undefined")
    print(f"{prefix} PR-AUC:             {metrics['pr_auc']:.4f}" if not np.isnan(metrics["pr_auc"]) else f"{prefix} PR-AUC:             undefined")
    print(f"{prefix} Average Precision:  {metrics['average_precision']:.4f}" if not np.isnan(metrics["average_precision"]) else f"{prefix} Average Precision:  undefined")


def aggregate_ensemble(
    all_results_df: pd.DataFrame,
    score_col: str,
    score_type: str,
    id_col: str = DEFAULT_ID_COL,
    label_col: str = DEFAULT_LABEL_COL,
    threshold: float = 0.5,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Aggregate predictions at the subject level by averaging scores within each ID.

    Parameters
    ----------
    all_results_df : pd.DataFrame
        Input dataframe containing per-sample predictions.
    score_col : str
        Column containing prediction scores (`ms_prob` or `ms_logits`).
    score_type : str
        One of {"prob", "logit"}.
    id_col : str, default="m_id"
        Subject/group identifier.
    label_col : str, default="ms"
        Ground-truth label column.
    threshold : float, default=0.5
        Threshold for converting probabilities into binary predictions.
    verbose : bool, default=True
        Whether to print metrics.

    Returns
    -------
    ensemble_df : pd.DataFrame
        Aggregated dataframe with subject-level scores/predictions.
    metrics : dict
        Dictionary of evaluation metrics.
    """
    if score_type not in {"prob", "logit"}:
        raise ValueError("score_type must be either 'prob' or 'logit'.")

    _validate_columns(all_results_df, [id_col, score_col, label_col])

    if all_results_df.empty:
        empty_df = pd.DataFrame(columns=[id_col, score_col, label_col, DEFAULT_PROB_COL, "ensemble_pred"])
        metrics = {
            "accuracy": np.nan,
            "roc_auc": np.nan,
            "pr_auc": np.nan,
            "average_precision": np.nan,
            "n_samples": 0,
        }
        return empty_df, metrics

    ensemble_df = (
        all_results_df
        .groupby(id_col, as_index=False)
        .agg({
            score_col: "mean",
            label_col: "first",
        })
        .copy()
    )

    if score_type == "prob":
        ensemble_df[DEFAULT_PROB_COL] = ensemble_df[score_col].astype(float)
    else:
        ensemble_df[DEFAULT_PROB_COL] = sigmoid(ensemble_df[score_col].to_numpy())

    ensemble_df["ensemble_pred"] = (ensemble_df[DEFAULT_PROB_COL] >= threshold).astype(int)

    metrics = _safe_binary_metrics(
        y_true=ensemble_df[label_col].to_numpy(),
        y_score=ensemble_df[DEFAULT_PROB_COL].to_numpy(),
        threshold=threshold,
    )

    if verbose:
        prefix = "[Avg Probability Ensemble]" if score_type == "prob" else "[Avg Logits Ensemble]"
        _print_metrics(metrics, prefix=prefix)

    return ensemble_df, metrics


def avg_prob_ensemble(
    all_results_df: pd.DataFrame,
    id_col: str = DEFAULT_ID_COL,
    label_col: str = DEFAULT_LABEL_COL,
    prob_col: str = DEFAULT_PROB_COL,
    threshold: float = 0.5,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Average probability ensemble."""
    return aggregate_ensemble(
        all_results_df=all_results_df,
        score_col=prob_col,
        score_type="prob",
        id_col=id_col,
        label_col=label_col,
        threshold=threshold,
        verbose=verbose,
    )


def avg_logits_ensemble(
    all_results_df: pd.DataFrame,
    id_col: str = DEFAULT_ID_COL,
    label_col: str = DEFAULT_LABEL_COL,
    logit_col: str = DEFAULT_LOGIT_COL,
    threshold: float = 0.5,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Average logit ensemble."""
    return aggregate_ensemble(
        all_results_df=all_results_df,
        score_col=logit_col,
        score_type="logit",
        id_col=id_col,
        label_col=label_col,
        threshold=threshold,
        verbose=verbose,
    )


def _map_smri_modality(modality: str) -> Optional[str]:
    """
    Map raw modality string to one of the target sMRI modality groups.

    Returns
    -------
    modality_type : str or None
        One of {"flair", "t1_ce", "t1_nce"} or None if unmatched.
    """
    modality_lower = str(modality).lower()

    if "flair" in modality_lower:
        return "flair"
    if "t1_ce" in modality_lower:
        return "t1_ce"
    if "t1_nce" in modality_lower:
        return "t1_nce"
    return None


def grouped_avg_prob_ensemble_smri(
    all_results_df: pd.DataFrame,
    mode: str = "ms_logits",
    id_col: str = DEFAULT_ID_COL,
    label_col: str = DEFAULT_LABEL_COL,
    modality_col: str = DEFAULT_MODALITY_COL,
    prob_col: str = DEFAULT_PROB_COL,
    logit_col: str = DEFAULT_LOGIT_COL,
    target_modalities: Optional[Sequence[str]] = None,
    threshold: float = 0.5,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Perform grouped sMRI ensemble across flair / t1_ce / t1_nce.

    The function first groups predictions by (subject, modality_type), averages
    scores within each modality, then averages across available target modalities.

    Parameters
    ----------
    all_results_df : pd.DataFrame
        Input dataframe with at least:
        - m_id
        - modality
        - ms
        - ms_prob or ms_logits
    mode : str, default="ms_logits"
        One of {"ms_prob", "ms_logits"}.
    target_modalities : sequence of str, optional
        Target modalities to ensemble. Default is ["flair", "t1_ce", "t1_nce"].
    threshold : float, default=0.5
        Probability threshold for binary prediction.
    verbose : bool, default=True
        Whether to print metrics.

    Returns
    -------
    ensemble_df : pd.DataFrame
        Subject-level grouped ensemble results.
    metrics : dict
        Evaluation metrics.
    """
    if target_modalities is None:
        target_modalities = ["flair", "t1_ce", "t1_nce"]

    if mode not in {prob_col, logit_col}:
        raise ValueError(f"mode must be either '{prob_col}' or '{logit_col}'.")

    required_cols = [id_col, modality_col, label_col, mode]
    _validate_columns(all_results_df, required_cols)

    if all_results_df.empty:
        empty_df = pd.DataFrame()
        metrics = {
            "accuracy": np.nan,
            "roc_auc": np.nan,
            "pr_auc": np.nan,
            "average_precision": np.nan,
            "n_samples": 0,
        }
        return empty_df, metrics

    df = all_results_df.copy()
    df["modality_type"] = df[modality_col].apply(_map_smri_modality)
    df = df.dropna(subset=["modality_type"]).copy()

    if df.empty:
        if verbose:
            print("Warning: no valid sMRI modalities found for grouped ensemble.")
        empty_df = pd.DataFrame()
        metrics = {
            "accuracy": np.nan,
            "roc_auc": np.nan,
            "pr_auc": np.nan,
            "average_precision": np.nan,
            "n_samples": 0,
        }
        return empty_df, metrics

    grouped_scores = pd.pivot_table(
        df,
        values=mode,
        index=id_col,
        columns="modality_type",
        aggfunc="mean",
    )

    score_cols = [f"{mod}_score" for mod in grouped_scores.columns]
    grouped_scores.columns = score_cols

    true_labels = df.groupby(id_col)[label_col].first()
    ensemble_df = pd.concat([true_labels, grouped_scores], axis=1)

    expected_score_cols = [f"{mod}_score" for mod in target_modalities]
    for col in expected_score_cols:
        if col not in ensemble_df.columns:
            ensemble_df[col] = np.nan

    ensemble_df["mean_score"] = ensemble_df[expected_score_cols].mean(axis=1, skipna=True)

    if mode == prob_col:
        ensemble_df[DEFAULT_PROB_COL] = ensemble_df["mean_score"]
    else:
        ensemble_df[DEFAULT_PROB_COL] = sigmoid(ensemble_df["mean_score"].to_numpy())

    valid_df = ensemble_df.dropna(subset=[DEFAULT_PROB_COL]).copy()

    if valid_df.empty:
        ensemble_df["ensemble_pred"] = pd.NA
        ensemble_df = ensemble_df.reset_index()

        metrics = {
            "accuracy": np.nan,
            "roc_auc": np.nan,
            "pr_auc": np.nan,
            "average_precision": np.nan,
            "n_samples": 0,
        }

        if verbose:
            print("Warning: no valid samples available for grouped sMRI ensemble metrics.")

        return ensemble_df, metrics

    valid_df["ensemble_pred"] = (valid_df[DEFAULT_PROB_COL] >= threshold).astype(int)

    metrics = _safe_binary_metrics(
        y_true=valid_df[label_col].to_numpy(),
        y_score=valid_df[DEFAULT_PROB_COL].to_numpy(),
        threshold=threshold,
    )

    ensemble_df["ensemble_pred"] = pd.NA
    ensemble_df.loc[valid_df.index, "ensemble_pred"] = valid_df["ensemble_pred"]
    ensemble_df = ensemble_df.reset_index()

    if verbose:
        _print_metrics(metrics, prefix="[Grouped sMRI Ensemble]")

    return ensemble_df, metrics