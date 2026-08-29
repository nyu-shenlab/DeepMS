from types import SimpleNamespace

import pytest
from torch.utils.data import WeightedRandomSampler

from train import _worker_loader_kwargs, create_dataloaders


def _args(**overrides):
    values = {
        "batch_size": 4,
        "gradient_accumulation_steps": 1,
        "oversampling": False,
        "seed": 88,
        "num_workers": 2,
        "val_num_workers": 0,
        "val_batch_size": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_validation_workers_are_not_persistent() -> None:
    train_loader, val_loaders = create_dataloaders(
        train_ds=list(range(8)),
        val_datasets={"test": list(range(4))},
        sampling_weights=None,
        args=_args(),
        accelerator=SimpleNamespace(num_processes=2),
    )

    assert train_loader.num_workers == 2
    assert train_loader.persistent_workers is True
    assert train_loader.prefetch_factor == 2

    val_loader = val_loaders["test"]
    assert val_loader.num_workers == 0
    assert val_loader.persistent_workers is False
    assert val_loader.prefetch_factor is None


def test_nonzero_validation_workers_use_bounded_prefetch_without_persistence() -> None:
    _, val_loaders = create_dataloaders(
        train_ds=list(range(8)),
        val_datasets={"test": list(range(4))},
        sampling_weights=None,
        args=_args(val_num_workers=1),
        accelerator=SimpleNamespace(num_processes=2),
    )

    val_loader = val_loaders["test"]
    assert val_loader.num_workers == 1
    assert val_loader.persistent_workers is False
    assert val_loader.prefetch_factor == 1


def test_zero_training_workers_disable_persistence_and_prefetch() -> None:
    train_loader, _ = create_dataloaders(
        train_ds=list(range(8)),
        val_datasets={},
        sampling_weights=None,
        args=_args(num_workers=0),
        accelerator=SimpleNamespace(num_processes=2),
    )

    assert train_loader.num_workers == 0
    assert train_loader.persistent_workers is False
    assert train_loader.prefetch_factor is None


def test_weighted_sampler_keeps_the_full_global_draw_budget() -> None:
    train_loader, _ = create_dataloaders(
        train_ds=list(range(8)),
        val_datasets={},
        sampling_weights=[1.0] * 8,
        args=_args(oversampling=True),
        accelerator=SimpleNamespace(num_processes=2),
    )

    assert isinstance(train_loader.sampler, WeightedRandomSampler)
    assert len(train_loader.sampler) == 8


def test_worker_counts_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _worker_loader_kwargs(-1, persistent=False, prefetch_factor=1)
