import json
import math

import pandas as pd
import pytest

from utils.analysis import sigmoid
from utils.reporting import (
    PUBLIC_EXTERNAL_DATASETS,
    attach_manifest_metadata,
    build_performance_report,
    normalize_report_predictions,
    notebook_patient_predictions,
    save_performance_report,
)


def prediction_row(
    row_id: int,
    m_id: str,
    modality: str,
    label: int,
    score: float,
    *,
    dataset: str = "WMH",
    patient_id: str | None = None,
    used_masked_image: bool = False,
    used_preprocessing_fallback: bool = False,
) -> dict:
    return {
        "row_id": row_id,
        "m_id": m_id,
        "patient_id": patient_id or m_id,
        "dataset": dataset,
        "modality": modality,
        "ms": label,
        "label": label,
        "ms_logits": score,
        "ms_prob": float(sigmoid([score])[0]),
        "used_masked_image": used_masked_image,
        "used_preprocessing_fallback": used_preprocessing_fallback,
    }


def test_notebook_aggregation_uses_direct_scan_rows_and_temperature_scaling() -> None:
    rows = [
        prediction_row(0, "p0", "3DFLAIR_NCE", 0, 0.0),
        prediction_row(1, "p0", "3DFLAIR_NCE", 0, 0.0),
        prediction_row(2, "p0", "2DFLAIR_NCE", 0, 6.0),
        prediction_row(3, "p0", "3DT1_CE", 0, 4.0),
        prediction_row(4, "p1", "3DFLAIR_NCE", 1, 1.0),
        prediction_row(5, "p1", "3DT1_CE", 1, 2.0),
    ]
    normalized = normalize_report_predictions(pd.DataFrame(rows), profile="generic")

    raw = notebook_patient_predictions(
        normalized,
        ensemble="sMRI",
        calibration="raw",
        threshold=0.5,
    ).set_index("m_id")
    assert raw.loc["p0", "flair_score"] == pytest.approx(2.0)
    assert raw.loc["p0", "mean_score"] == pytest.approx(3.0)
    assert raw.loc["p0", "ms_prob"] == pytest.approx(1.0 / (1.0 + math.exp(-3.0)))

    calibrated = notebook_patient_predictions(
        normalized,
        ensemble="sMRI",
        calibration="notebook_temperature",
        threshold=0.5,
    ).set_index("m_id")
    expected_flair = ((0.0 / 1.73) + (0.0 / 1.73) + (6.0 / 1.82)) / 3.0
    expected_mean = (expected_flair + (4.0 / 1.81)) / 2.0
    assert calibrated.loc["p0", "flair_score"] == pytest.approx(expected_flair)
    assert calibrated.loc["p0", "mean_score"] == pytest.approx(expected_mean)


def public_external_predictions() -> pd.DataFrame:
    rows = [
        prediction_row(0, "wmh", "2DFLAIR_NCE", 1, 2.0, dataset="WMH"),
        prediction_row(1, "brats", "3DT1_CE", 0, -2.0, dataset="BraTS_Met"),
        prediction_row(2, "mslesseg", "2DFLAIR_NCE", 0, -1.0, dataset="MSLesSeg"),
        prediction_row(3, "msseg2", "3DFLAIR_NCE", 1, 1.0, dataset="MSSEG2"),
        prediction_row(
            4,
            "visit_1",
            "2DFLAIR_NCE",
            1,
            1.5,
            dataset="PediMS",
            patient_id="PediMS_P1",
        ),
        prediction_row(
            5,
            "visit_2",
            "3DT1_CE",
            1,
            0.5,
            dataset="PediMS",
            patient_id="PediMS_P1",
        ),
        prediction_row(
            6,
            "open_ms_cross_sectional_patient02",
            "3DFLAIR_NCE",
            0,
            -0.5,
            dataset="open_ms_cross_sectional",
        ),
    ]
    return pd.DataFrame(rows)


def test_public_external_report_encodes_notebook_cohorts_and_pedims_ids() -> None:
    artifacts = build_performance_report(
        public_external_predictions(),
        profile="public_external_unmasked",
        image_policy="preprocessing",
        bootstrap_samples=0,
    )

    primary = artifacts.report["primary"]
    masking = artifacts.report["masking_comparison"]
    assert primary["cohort"] == "notebook_primary"
    assert primary["calibration"] == "notebook_temperature"
    assert primary["n_patients"] == 4
    assert masking["n_patients"] == 2
    ablation = artifacts.report["ablation_results"]
    assert ablation["flair_raw"]["calibration"] == "raw"
    assert ablation["flair_notebook_temperature"]["calibration"] == "notebook_temperature"
    assert ablation["smri_raw"]["ensemble"] == "sMRI"
    contract = artifacts.report["ablation_contract"]
    assert contract["recommended_selector"] == "smri_notebook_temperature"
    assert contract["masking_evaluation_selector"] == "masking_comparison"
    assert "MSLesSeg" not in PUBLIC_EXTERNAL_DATASETS

    prediction_inventory = artifacts.report["prediction_inventory"]
    assert prediction_inventory["n_scan_rows"] == 7
    assert prediction_inventory["n_source_m_ids"] == 7
    assert prediction_inventory["n_report_m_ids"] == 6
    assert prediction_inventory["modalities"] == ["2DFLAIR_NCE", "3DFLAIR_NCE", "3DT1_CE"]
    assert prediction_inventory["datasets"] == sorted(
        ["BraTS_Met", "MSLesSeg", "MSSEG2", "PediMS", "WMH", "open_ms_cross_sectional"]
    )

    inventory = {row["dataset"]: row for row in artifacts.report["dataset_inventory"]}
    assert inventory["MSLesSeg"]["included_in_notebook_primary"] is False
    assert inventory["MSLesSeg"]["n_patients"] == 1
    assert inventory["MSLesSeg"]["n_notebook_primary_patients"] == 0
    assert inventory["open_ms_cross_sectional"]["included_in_notebook_primary"] is True
    assert inventory["open_ms_cross_sectional"]["n_notebook_primary_patients"] == 0
    excluded_dataset_rows = artifacts.summary[
        artifacts.summary["cohort"].isin(["dataset:MSLesSeg", "dataset:open_ms_cross_sectional"])
    ]
    assert excluded_dataset_rows["n_patients"].eq(0).all()

    patient_rows = artifacts.patient_predictions
    pediatric = patient_rows[patient_rows["m_id"].eq("PediMS_P1")]
    assert len(pediatric) == 4
    source_counts = pediatric.set_index(["ensemble", "calibration"])["n_source_m_ids"]
    assert source_counts.loc["FLAIR"].eq(1).all()
    assert source_counts.loc["sMRI"].eq(2).all()


def test_masked_profile_primary_is_raw_flair_on_comparable_datasets() -> None:
    predictions = public_external_predictions()
    predictions.loc[predictions["dataset"].isin(["WMH", "PediMS"]), "used_masked_image"] = True
    artifacts = build_performance_report(
        predictions,
        profile="public_external_masked",
        image_policy="masked_image_path_then_preprocessing",
        bootstrap_samples=0,
    )

    primary = artifacts.report["primary"]
    assert primary["cohort"] == "masking_comparable"
    assert primary["ensemble"] == "FLAIR"
    assert primary["calibration"] == "raw"
    assert primary["n_patients"] == 2
    assert primary["n_used_masked_rows"] == 2
    assert artifacts.report["mask_provenance"]["available"] is True
    assert artifacts.report["mask_provenance"]["primary_preprocessing_fallback_rows"] == 0


def test_private_cohort_overrides_filter_patients_and_correct_labels() -> None:
    predictions = pd.DataFrame(
        [
            prediction_row(0, "excluded", "3DFLAIR_NCE", 1, 2.0),
            prediction_row(1, "corrected", "3DFLAIR_NCE", 1, -2.0),
        ]
    )
    overrides = pd.DataFrame(
        [
            {"m_id": "excluded", "include": 0, "label_override": None},
            {"m_id": "corrected", "include": 1, "label_override": 0},
        ]
    )

    artifacts = build_performance_report(
        predictions,
        profile="internal",
        image_policy="preprocessing",
        bootstrap_samples=0,
        cohort_overrides=overrides,
        cohort_overrides_source="private.csv",
    )

    assert artifacts.report["primary"]["n_patients"] == 1
    assert artifacts.report["primary"]["n_negative"] == 1
    assert artifacts.report["cohort_overrides"] == {
        "source": "private.csv",
        "rows_excluded": 1,
        "patients_excluded": 1,
        "labels_overridden_patients": 1,
    }


def test_manifest_metadata_attachment_does_not_copy_image_paths() -> None:
    predictions = pd.DataFrame(
        [
            prediction_row(0, "p0", "3DFLAIR_NCE", 0, -1.0),
            prediction_row(1, "p1", "3DFLAIR_NCE", 1, 1.0),
        ]
    ).drop(columns=["dataset", "patient_id", "used_masked_image", "used_preprocessing_fallback"])
    manifest = pd.DataFrame(
        [
            {
                "row_id": 0,
                "dataset": "A",
                "patient_id": "p0",
                "preprocessing": "/private/a.nii.gz",
                "mask_path": "/private/mask-a.nii.gz",
                "used_masked_image": False,
            },
            {
                "row_id": 1,
                "dataset": "B",
                "patient_id": "p1",
                "preprocessing": "/private/b.nii.gz",
                "mask_path": "/private/mask-b.nii.gz",
                "used_masked_image": True,
            },
        ]
    )

    attached = attach_manifest_metadata(predictions, manifest)
    assert attached["dataset"].tolist() == ["A", "B"]
    assert attached["used_masked_image"].tolist() == [False, True]
    assert "preprocessing" not in attached
    assert "mask_path" not in attached


def test_performance_artifacts_are_written_as_strict_json(tmp_path) -> None:
    report = save_performance_report(
        public_external_predictions(),
        output_dir=tmp_path,
        profile="public_external_unmasked",
        image_policy="preprocessing",
        bootstrap_samples=0,
    )

    assert report["primary"]["n_patients"] == 4
    expected = {
        "performance_summary.csv",
        "prediction_patient_report.csv",
        "performance_report.json",
        "performance_report.md",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}
    parsed = json.loads((tmp_path / "performance_report.json").read_text())
    assert parsed["primary"]["roc_auc"] is not None


def test_profile_rejects_the_wrong_image_policy() -> None:
    with pytest.raises(ValueError, match="requires image_policy"):
        build_performance_report(
            public_external_predictions(),
            profile="public_external_unmasked",
            image_policy="masked_image_path_then_preprocessing",
            bootstrap_samples=0,
        )


def test_masked_primary_rejects_preprocessing_fallback() -> None:
    predictions = public_external_predictions()
    comparable = predictions["dataset"].isin(["WMH", "PediMS"])
    predictions.loc[comparable, "used_masked_image"] = True
    predictions.loc[predictions["dataset"].eq("WMH"), "used_masked_image"] = False
    predictions.loc[predictions["dataset"].eq("WMH"), "used_preprocessing_fallback"] = True

    with pytest.raises(ValueError, match="must use an explicit masked image"):
        build_performance_report(
            predictions,
            profile="public_external_masked",
            image_policy="masked_image_path_then_preprocessing",
            bootstrap_samples=0,
        )


def test_historical_masked_predictions_without_provenance_remain_reportable() -> None:
    predictions = public_external_predictions().drop(columns=["used_masked_image", "used_preprocessing_fallback"])
    artifacts = build_performance_report(
        predictions,
        profile="public_external_masked",
        image_policy="masked_image_path_then_preprocessing",
        bootstrap_samples=0,
    )

    assert artifacts.report["mask_provenance"]["available"] is False
    assert artifacts.report["primary"]["n_patients"] == 2


def test_generic_profile_does_not_apply_public_external_exclusions() -> None:
    predictions = pd.DataFrame(
        [
            prediction_row(
                0,
                "open_ms_cross_sectional_patient02",
                "3DFLAIR_NCE",
                0,
                -1.0,
            ),
            prediction_row(1, "positive", "3DFLAIR_NCE", 1, 1.0),
        ]
    )
    artifacts = build_performance_report(
        predictions,
        profile="generic",
        image_policy="unspecified",
        bootstrap_samples=0,
    )

    assert artifacts.report["primary"]["n_patients"] == 2


def test_bootstrap_intervals_are_deterministic() -> None:
    predictions = pd.DataFrame(
        [
            prediction_row(0, "n0", "3DFLAIR_NCE", 0, -2.0),
            prediction_row(1, "n1", "3DFLAIR_NCE", 0, -0.5),
            prediction_row(2, "p0", "3DFLAIR_NCE", 1, 0.5),
            prediction_row(3, "p1", "3DFLAIR_NCE", 1, 2.0),
        ]
    )
    first = build_performance_report(
        predictions,
        profile="generic",
        image_policy="unspecified",
        bootstrap_samples=50,
        seed=7,
    ).report["primary"]
    second = build_performance_report(
        predictions,
        profile="generic",
        image_policy="unspecified",
        bootstrap_samples=50,
        seed=7,
    ).report["primary"]

    assert first["bootstrap_valid_samples"] > 0
    assert first["roc_auc_ci_low"] is not None
    assert first["pr_auc_ci_high"] is not None
    assert first == second
