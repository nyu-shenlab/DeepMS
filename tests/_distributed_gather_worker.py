"""Two-process CPU smoke test for the real training validation function."""

import logging
from types import SimpleNamespace

import pandas as pd
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset

from train import validate_model


class ValidationDataset(Dataset):
    def __init__(self) -> None:
        self.data = pd.DataFrame(
            {
                "row_id": list(range(7)),
                "m_id": [f"patient-{index}" for index in range(7)],
                "modality": ["3DFLAIR_NCE"] * 7,
                "label": [index % 2 for index in range(7)],
            }
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        row = self.data.iloc[index]
        label = int(row["label"])
        evidence = 2.0 if label == 1 else -2.0
        return {
            "image": torch.tensor([evidence], dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32),
            "row_id": torch.tensor(int(row["row_id"]), dtype=torch.int64),
            "m_id": str(row["m_id"]),
        }


class ValidationModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, images: torch.Tensor, train: bool = False) -> dict:
        del train
        evidence = images.reshape(images.shape[0]) + self.anchor * 0.0
        scores = torch.stack((-evidence, evidence), dim=1)
        return {"score": scores, "prob": None, "SA_map": None}


def main() -> None:
    accelerator = Accelerator(cpu=True)
    model = accelerator.prepare(ValidationModel())
    loader = accelerator.prepare(
        DataLoader(
            ValidationDataset(),
            batch_size=2,
            shuffle=False,
            drop_last=False,
        )
    )
    args = SimpleNamespace(
        backbone="VoCo",
        loss_type="ce",
        val_modalities=["3DFLAIR_NCE"],
        auc_metric="micro",
    )
    logger = logging.getLogger("distributed-validation-smoke")

    results = validate_model(
        model=model,
        val_dataloaders={"3DFLAIR_NCE": loader},
        accelerator=accelerator,
        args=args,
        logger=logger,
    )

    if accelerator.is_main_process:
        assert results["3DFLAIR_NCE"]["count"] == 7, results
        assert results["micro_avg"]["count"] == 7, results
        assert results["micro_avg"]["auc"] == 1.0, results
        assert results["micro_avg"]["accuracy"] == 1.0, results


if __name__ == "__main__":
    main()
