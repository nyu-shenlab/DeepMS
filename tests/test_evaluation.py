import math

import numpy as np
import pandas as pd
import pytest

from utils.analysis import grouped_avg_prob_ensemble, sigmoid
from utils.evaluation import (
    aggregate_patient_modality,
    build_inference_outputs,
    summarize_validation_predictions,
    validate_prediction_coverage,
)


def logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def base_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"row_id": 0, "m_id": "p0", "modality": "3DFLAIR_NCE", "ms": 0, "ms_prob": 0.1},
            {"row_id": 1, "m_id": "p1", "modality": "3DFLAIR_NCE", "ms": 1, "ms_prob": 0.9},
            {"row_id": 2, "m_id": "p2", "modality": "fa_dti", "ms": 0, "ms_prob": 0.8},
            {"row_id": 3, "m_id": "p3", "modality": "fa_dti", "ms": 1, "ms_prob": 0.2},
        ]
    )


def test_prediction_coverage_sorts_and_requires_exactly_one_prediction_per_row() -> None:
    frame = base_predictions().iloc[::-1]
    validated = validate_prediction_coverage(frame, expected_row_ids=range(4))
    assert validated["row_id"].tolist() == [0, 1, 2, 3]

    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate row_id"):
        validate_prediction_coverage(duplicated)

    with pytest.raises(ValueError, match="coverage mismatch"):
        validate_prediction_coverage(frame, expected_row_ids=range(5))


def test_prediction_coverage_rejects_invalid_probabilities_and_labels() -> None:
    frame = base_predictions()
    invalid_probability = frame.copy()
    invalid_probability.loc[0, "ms_prob"] = 1.1
    with pytest.raises(ValueError, match="outside"):
        validate_prediction_coverage(invalid_probability)

    conflicting = frame.copy()
    conflicting.loc[1, "m_id"] = "p0"
    with pytest.raises(ValueError, match="conflicting labels"):
        validate_prediction_coverage(conflicting)


def test_repeated_scans_are_averaged_before_cross_modality_weighting() -> None:
    frame = pd.DataFrame(
        [
            {"row_id": 0, "m_id": "p", "modality": "3DFLAIR_NCE", "ms": 1, "ms_prob": 0.0},
            {"row_id": 1, "m_id": "p", "modality": "3DFLAIR_NCE", "ms": 1, "ms_prob": 1.0},
            {"row_id": 2, "m_id": "p", "modality": "3DFLAIR_NCE", "ms": 1, "ms_prob": 1.0},
            {"row_id": 3, "m_id": "p", "modality": "fa_dti", "ms": 1, "ms_prob": 1.0},
        ]
    )

    modality = aggregate_patient_modality(frame)
    flair = modality[modality["modality"] == "3DFLAIR_NCE"].iloc[0]
    assert flair["n_scans"] == 3
    assert flair["ms_prob"] == pytest.approx(2.0 / 3.0)

    patient = grouped_avg_prob_ensemble(frame, print_result=False)
    assert patient.loc[0, "ms_prob"] == pytest.approx(5.0 / 6.0)


def test_two_level_multimodal_output_is_not_flat_modality_averaging() -> None:
    rows = []
    structural_modalities = ["2DFLAIR_NCE", "2DT1_NCE", "3DT1_CE", "b0"]
    for patient, label, structural_probability, diffusion_probability in [
        ("p0", 0, 0.1, 0.9),
        ("p1", 1, 0.9, 0.1),
    ]:
        for modality in structural_modalities:
            rows.append(
                {
                    "row_id": len(rows),
                    "m_id": patient,
                    "modality": modality,
                    "ms": label,
                    "ms_prob": structural_probability,
                    "ms_logits": logit(structural_probability),
                }
            )
        rows.append(
            {
                "row_id": len(rows),
                "m_id": patient,
                "modality": "fa_dti",
                "ms": label,
                "ms_prob": diffusion_probability,
                "ms_logits": logit(diffusion_probability),
            }
        )

    outputs, metrics = build_inference_outputs(pd.DataFrame(rows))
    flat = outputs["patient_flat_logit"].set_index("m_id")
    multimodal = outputs["patient_multimodal"].set_index("m_id")

    assert flat.loc["p0", "ms_prob"] == pytest.approx(0.21109548)
    assert flat.loc["p1", "ms_prob"] == pytest.approx(0.78890452)
    assert multimodal.loc["p0", "ms_prob"] == pytest.approx(0.5)
    assert multimodal.loc["p1", "ms_prob"] == pytest.approx(0.5)
    assert metrics["flat_logit_patient"]["auc"] == pytest.approx(1.0)
    assert metrics["multimodal_two_level_patient"]["auc"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("metric_name", "expected"),
    [("micro", 0.75), ("macro", 0.5), ("hierarchical", 0.5), ("ensemble", 0.75)],
)
def test_validation_metric_selection_uses_global_predictions(
    metric_name: str,
    expected: float,
) -> None:
    results = summarize_validation_predictions(
        base_predictions(),
        requested_modalities=["3DFLAIR_NCE", "fa_dti"],
        auc_metric=metric_name,
        expected_row_ids=range(4),
    )

    assert results["micro_avg"]["count"] == 4
    assert results["micro_avg"]["auc"] == pytest.approx(0.75)
    assert results["macro_avg"]["auc"] == pytest.approx(0.5)
    assert results["hierarchical_avg_auc"] == pytest.approx(0.5)
    assert results["ensemble"]["auc"] == pytest.approx(0.75)
    assert results["best_metric"] == pytest.approx(expected)


def test_sigmoid_is_stable_for_extreme_logits() -> None:
    with np.errstate(over="raise"):
        result = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert result.tolist() == pytest.approx([0.0, 0.5, 1.0])
