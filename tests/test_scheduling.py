import math

import pytest
import torch

from utils.scheduling import (
    UpdateWarmupCosineScheduler,
    compute_optimizer_update_counts,
)


def test_total_steps_count_optimizer_updates_after_accumulation() -> None:
    updates_per_epoch, total_updates = compute_optimizer_update_counts(
        num_batches_per_process=10,
        gradient_accumulation_steps=4,
        num_epochs=3,
    )
    assert updates_per_epoch == 3
    assert total_updates == 9


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_batches_per_process": 0, "gradient_accumulation_steps": 1, "num_epochs": 1},
        {"num_batches_per_process": 1, "gradient_accumulation_steps": 0, "num_epochs": 1},
        {"num_batches_per_process": 1, "gradient_accumulation_steps": 1, "num_epochs": 0},
    ],
)
def test_total_update_count_rejects_non_positive_inputs(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        compute_optimizer_update_counts(**kwargs)


def make_optimizer(lr: float = 1.0) -> torch.optim.Optimizer:
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    return torch.optim.SGD([parameter], lr=lr)


def learning_rates_used(
    optimizer: torch.optim.Optimizer,
    scheduler: UpdateWarmupCosineScheduler,
    count: int,
) -> list[float]:
    values = []
    for _ in range(count):
        values.append(float(optimizer.param_groups[0]["lr"]))
        optimizer.step()
        scheduler.step()
    return values


def test_cosine_schedule_is_indexed_by_successful_updates() -> None:
    optimizer = make_optimizer()
    scheduler = UpdateWarmupCosineScheduler(
        optimizer,
        total_steps=5,
        min_lr=0.1,
    )

    values = learning_rates_used(optimizer, scheduler, 5)

    expected = [
        0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * index / 4))
        for index in range(5)
    ]
    assert values == pytest.approx(expected)
    assert scheduler.completed_steps == 5
    assert scheduler.get_last_lr() == pytest.approx([0.1])


def test_warmup_reaches_base_lr_before_cosine_decay() -> None:
    optimizer = make_optimizer()
    scheduler = UpdateWarmupCosineScheduler(
        optimizer,
        total_steps=6,
        warmup_steps=2,
        warmup_start_lr=0.1,
        min_lr=0.01,
    )

    values = learning_rates_used(optimizer, scheduler, 6)

    assert values[:3] == pytest.approx([0.1, 0.55, 1.0])
    assert values[-1] == pytest.approx(0.01)
    assert all(left >= right for left, right in zip(values[2:], values[3:]))


def test_one_point_cosine_phase_reaches_base_lr_after_warmup() -> None:
    optimizer = make_optimizer()
    scheduler = UpdateWarmupCosineScheduler(
        optimizer,
        total_steps=3,
        warmup_steps=2,
        warmup_start_lr=0.1,
        min_lr=0.01,
    )

    assert learning_rates_used(optimizer, scheduler, 3) == pytest.approx(
        [0.1, 0.55, 1.0]
    )


def test_single_update_uses_optimizer_base_lr() -> None:
    optimizer = make_optimizer(lr=0.3)
    scheduler = UpdateWarmupCosineScheduler(
        optimizer,
        total_steps=1,
        min_lr=0.01,
    )

    assert learning_rates_used(optimizer, scheduler, 1) == pytest.approx([0.3])
    assert scheduler.get_last_lr() == pytest.approx([0.3])


def test_scheduler_state_resumes_the_exact_next_update() -> None:
    first_optimizer = make_optimizer()
    first_scheduler = UpdateWarmupCosineScheduler(
        first_optimizer,
        total_steps=8,
        warmup_steps=2,
        warmup_start_lr=0.2,
        min_lr=0.05,
    )
    learning_rates_used(first_optimizer, first_scheduler, 3)

    second_optimizer = make_optimizer()
    second_scheduler = UpdateWarmupCosineScheduler(
        second_optimizer,
        total_steps=8,
        warmup_steps=2,
        warmup_start_lr=0.2,
        min_lr=0.05,
    )
    second_scheduler.load_state_dict(first_scheduler.state_dict())

    assert second_scheduler.completed_steps == 3
    assert second_scheduler.get_last_lr() == pytest.approx(first_scheduler.get_last_lr())
    assert learning_rates_used(second_optimizer, second_scheduler, 5) == pytest.approx(
        learning_rates_used(first_optimizer, first_scheduler, 5)
    )


def test_scheduler_rejects_incompatible_resume_configuration() -> None:
    optimizer = make_optimizer()
    scheduler = UpdateWarmupCosineScheduler(optimizer, total_steps=4)
    state = scheduler.state_dict()

    other_optimizer = make_optimizer()
    other_scheduler = UpdateWarmupCosineScheduler(other_optimizer, total_steps=5)
    with pytest.raises(ValueError, match="configuration mismatch"):
        other_scheduler.load_state_dict(state)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"total_steps": 0}, "total_steps"),
        ({"total_steps": 2, "warmup_steps": 2}, "warmup_steps"),
        ({"total_steps": 2, "min_lr": 2.0}, "min_lr"),
    ],
)
def test_scheduler_rejects_invalid_configuration(kwargs: dict, message: str) -> None:
    optimizer = make_optimizer()
    with pytest.raises(ValueError, match=message):
        UpdateWarmupCosineScheduler(optimizer, **kwargs)
