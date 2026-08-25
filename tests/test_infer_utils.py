from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from accelerate import Accelerator, DataLoaderConfiguration
from torch.utils.data import DataLoader

from infer import (
    infer_probability,
    positive_class_logit,
    prepare_test_dataframe,
    run_inference_for_modality,
)


def inference_args(csv_path, *, use_cis: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        test_patient_ids=str(csv_path),
        modalities=["3DFLAIR_NCE", "fa_dti"],
        use_cis=use_cis,
        use_preprocess=True,
        use_bet_only=False,
        use_mask_img=False,
    )


def write_manifest(path) -> None:
    pd.DataFrame(
        [
            {"m_id": "001", "modality": "3DFLAIR_NCE", "label": 0, "ms": 0, "preprocessing": "/a.nii.gz"},
            {"m_id": "002", "modality": "3DFLAIR_NCE", "label": 2, "ms": 2, "preprocessing": "/b.nii.gz"},
            {"m_id": "003", "modality": "3DFLAIR_NCE", "label": 1, "ms": 1, "preprocessing": None},
            {"m_id": "999", "modality": "not_requested", "label": 1, "ms": 1, "preprocessing": "/c.nii.gz"},
        ]
    ).to_csv(path, index=False)


def test_positive_class_logit_has_one_stable_binary_contract() -> None:
    logits = torch.tensor([-2.0, 2.0])
    assert positive_class_logit(logits, "bce_with_logits").tolist() == pytest.approx([-2.0, 2.0])

    probabilities = torch.tensor([[0.25], [0.75]])
    converted = positive_class_logit(probabilities, "bce")
    assert converted.tolist() == pytest.approx([-1.0986123, 1.0986123])

    two_class = torch.tensor([[1.0, 3.5], [2.0, -1.0]])
    assert positive_class_logit(two_class, "bce_with_logits").tolist() == pytest.approx([2.5, -3.0])

    with pytest.raises(ValueError, match="Expected score shape"):
        positive_class_logit(torch.zeros(2, 3), "bce_with_logits")


def test_infer_probability_handles_binary_and_two_class_outputs() -> None:
    assert infer_probability(torch.tensor([0.0]), "bce_with_logits").item() == pytest.approx(0.5)
    assert infer_probability(torch.tensor([0.25]), "bce").item() == pytest.approx(0.25)
    expected = torch.softmax(torch.tensor([[1.0, 2.0]]), dim=1)[0, 1].item()
    assert infer_probability(torch.tensor([[1.0, 2.0]]), "bce_with_logits").item() == pytest.approx(expected)


def test_manifest_preserves_string_ids_maps_cis_and_records_exclusions(tmp_path) -> None:
    csv_path = tmp_path / "manifest.csv"
    write_manifest(csv_path)

    frame = prepare_test_dataframe(inference_args(csv_path, use_cis=True))

    assert frame["m_id"].tolist() == ["001", "002"]
    assert frame["label"].tolist() == [0, 1]
    assert frame["row_id"].tolist() == [0, 1]
    assert frame["source_row"].tolist() == [0, 1]
    assert frame.attrs["input_coverage"] == {
        "input_rows": 4,
        "requested_modality_rows": 3,
        "excluded_non_binary_label_rows": 0,
        "excluded_missing_image_rows": 1,
        "selected_rows": 2,
        "selected_patients": 2,
        "label_counts_before_mapping": {"0": 1, "2": 1, "1": 1},
        "requested_modalities": ["3DFLAIR_NCE", "fa_dti"],
        "available_modalities": ["3DFLAIR_NCE"],
        "missing_modalities": ["fa_dti"],
        "cis_mapped_to_positive": True,
        "image_column": "preprocessing",
        "image_policy": "preprocessing",
        "masked_image_column_present": False,
        "explicit_masked_image_rows": 0,
        "preprocessing_fallback_rows": 0,
    }


def test_manifest_excludes_cis_without_explicit_mapping(tmp_path) -> None:
    csv_path = tmp_path / "manifest.csv"
    write_manifest(csv_path)

    frame = prepare_test_dataframe(inference_args(csv_path, use_cis=False))

    assert frame["m_id"].tolist() == ["001"]
    assert frame.attrs["input_coverage"]["excluded_non_binary_label_rows"] == 1
    assert frame.attrs["input_coverage"]["excluded_missing_image_rows"] == 1


def test_masked_image_policy_records_explicit_and_fallback_rows(tmp_path) -> None:
    csv_path = tmp_path / "masked_manifest.csv"
    pd.DataFrame(
        [
            {
                "m_id": "001",
                "modality": "3DFLAIR_NCE",
                "label": 1,
                "ms": 1,
                "preprocessing": "/preprocessed-a.nii.gz",
                "masked_image_path": "/masked-a.nii.gz",
            },
            {
                "m_id": "002",
                "modality": "3DFLAIR_NCE",
                "label": 0,
                "ms": 0,
                "preprocessing": "/preprocessed-b.nii.gz",
                "masked_image_path": None,
            },
            {
                "m_id": "003",
                "modality": "3DFLAIR_NCE",
                "label": 0,
                "ms": 0,
                "preprocessing": "/preprocessed-c.nii.gz",
                "masked_image_path": "   ",
            },
        ]
    ).to_csv(csv_path, index=False)

    args = inference_args(csv_path, use_cis=False)
    args.use_preprocess = False
    args.use_mask_img = True
    frame = prepare_test_dataframe(args)

    assert frame["image"].tolist() == [
        "/masked-a.nii.gz",
        "/preprocessed-b.nii.gz",
        "/preprocessed-c.nii.gz",
    ]
    assert frame["mask_path"].tolist() == [1, 0, 0]
    coverage = frame.attrs["input_coverage"]
    assert coverage["image_policy"] == "masked_image_path_then_preprocessing"
    assert coverage["masked_image_column_present"] is True
    assert coverage["explicit_masked_image_rows"] == 1
    assert coverage["preprocessing_fallback_rows"] == 2


def test_manifest_rejects_conflicting_label_columns_and_path_flags(tmp_path) -> None:
    csv_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [{"m_id": "001", "modality": "3DFLAIR_NCE", "label": 0, "ms": 1, "preprocessing": "/a.nii.gz"}]
    ).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="label and ms disagree"):
        prepare_test_dataframe(inference_args(csv_path))

    write_manifest(csv_path)
    args = inference_args(csv_path)
    args.use_bet_only = True
    with pytest.raises(ValueError, match="Choose only one"):
        prepare_test_dataframe(args)

    args = inference_args(csv_path)
    args.use_preprocess = False
    args.use_mask_img = True
    args.report_profile = "public_external_unmasked"
    with pytest.raises(ValueError, match="requires image_policy"):
        prepare_test_dataframe(args)


class DummyInferenceModel(torch.nn.Module):
    def forward(self, images: torch.Tensor, train: bool = False) -> dict:
        del train
        batch_size = images.shape[0]
        scores = images.mean(dim=(1, 2, 3, 4)).unsqueeze(1)
        probability_map = torch.full(
            (batch_size, 1, 2, 2, 2),
            10_000.0,
            dtype=torch.float16,
            device=images.device,
        )
        attention_map = torch.ones(
            (batch_size, 1, 2, 2, 2),
            dtype=torch.float16,
            device=images.device,
        )
        return {"score": scores, "prob": probability_map, "SA_map": attention_map}


def test_inference_loop_returns_exact_scalar_rows_and_uses_fp32_reductions() -> None:
    records = [
        {
            "image": torch.full((1, 2, 2, 2), -2.0),
            "label": torch.tensor(0.0),
            "row_id": 0,
            "source_row": 10,
            "m_id": "001",
            "structural_mri": 1,
            "SMI": 0,
        },
        {
            "image": torch.full((1, 2, 2, 2), 2.0),
            "label": torch.tensor(1.0),
            "row_id": 1,
            "source_row": 11,
            "m_id": "002",
            "structural_mri": 1,
            "SMI": 0,
        },
    ]
    accelerator = Accelerator(
        cpu=True,
        dataloader_config=DataLoaderConfiguration(even_batches=False),
    )
    dataloader = accelerator.prepare(DataLoader(records, batch_size=2, shuffle=False))
    args = SimpleNamespace(loss_type="bce_with_logits", visualization=False)

    result = run_inference_for_modality(
        model=DummyInferenceModel(),
        dataloader=dataloader,
        modality="3DFLAIR_NCE",
        args=args,
        accelerator=accelerator,
        expected_row_ids=[0, 1],
    )

    assert result is not None
    assert result["row_id"].tolist() == [0, 1]
    assert result["m_id"].tolist() == ["001", "002"]
    assert result["ms_logits"].tolist() == pytest.approx([-2.0, 2.0])
    assert result["ms_prob"].tolist() == pytest.approx(
        [torch.sigmoid(torch.tensor(-2.0)).item(), torch.sigmoid(torch.tensor(2.0)).item()]
    )
    assert result["weighted_prob_sum"].tolist() == pytest.approx([80_000.0, 80_000.0])
