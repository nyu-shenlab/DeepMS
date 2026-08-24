import pandas as pd
import pytest
import torch

from utils.dataset import SingleModalityDataset, collate_skip_none


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image": ["unused.nii.gz"],
            "modality": ["3DFLAIR_NCE"],
            "label": [1],
            "m_id": ["001"],
            "SMI": [0],
        }
    )


def test_dataset_fails_fast_when_transform_rejects_a_sample() -> None:
    dataset = SingleModalityDataset(sample_frame(), transform=lambda sample: None)
    with pytest.raises(RuntimeError, match="Image transform failed"):
        dataset[0]


def test_dataset_preserves_string_identifiers() -> None:
    def tensor_transform(sample):
        return {
            **sample,
            "image": torch.zeros(1, 2, 2, 2),
            "lesion_mask": torch.zeros(1, 2, 2, 2),
            "wm_mask": torch.ones(1, 2, 2, 2),
        }

    dataset = SingleModalityDataset(sample_frame(), transform=tensor_transform)
    assert dataset[0]["m_id"] == "001"
    assert dataset[0]["label"].dtype == torch.float32


def test_collate_raises_instead_of_silently_dropping_an_entire_batch() -> None:
    with pytest.raises(RuntimeError, match="All samples"):
        collate_skip_none([None, None])
