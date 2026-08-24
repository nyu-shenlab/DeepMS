"""Learning-rate schedules with explicit optimizer-update semantics."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def compute_optimizer_update_counts(
    *,
    num_batches_per_process: int,
    gradient_accumulation_steps: int,
    num_epochs: int,
) -> tuple[int, int]:
    """Return optimizer updates per epoch and across the full planned run.

    ``num_batches_per_process`` must be measured from the Accelerate-prepared
    training loader. A final partial accumulation still produces one optimizer
    update, so integer ceiling division is required.
    """
    if num_batches_per_process <= 0:
        raise ValueError("num_batches_per_process must be positive.")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive.")

    updates_per_epoch = (
        num_batches_per_process + gradient_accumulation_steps - 1
    ) // gradient_accumulation_steps
    return updates_per_epoch, updates_per_epoch * num_epochs


class UpdateWarmupCosineScheduler:
    """Warm up and cosine-decay learning rates by optimizer update.

    The optimizer learning rate is set for update index zero at construction.
    Call :meth:`step` exactly once *after* each successful ``optimizer.step``.
    The first successful update therefore uses ``base_lr`` when warmup is
    disabled, or ``warmup_start_lr`` when warmup is enabled. When the cosine
    phase contains at least two updates, its last planned update uses ``min_lr``.
    """

    state_version = 1

    def __init__(
        self,
        optimizer,
        *,
        total_steps: int,
        warmup_steps: int = 0,
        min_lr: float = 0.0,
        warmup_start_lr: float = 0.0,
    ) -> None:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive.")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative.")
        if warmup_steps >= total_steps:
            raise ValueError("warmup_steps must be smaller than total_steps.")
        if min_lr < 0 or warmup_start_lr < 0:
            raise ValueError("Learning rates must be non-negative.")

        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr = float(min_lr)
        self.warmup_start_lr = float(warmup_start_lr)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]

        for base_lr in self.base_lrs:
            if self.min_lr > base_lr:
                raise ValueError("min_lr cannot exceed an optimizer base learning rate.")
            if self.warmup_steps > 0 and self.warmup_start_lr > base_lr:
                raise ValueError("warmup_start_lr cannot exceed an optimizer base learning rate.")

        self.completed_steps = 0
        self._set_lrs(self.get_lr_for_step(0))

    def _lr_for_group(self, step_index: int, base_lr: float) -> float:
        step_index = min(max(int(step_index), 0), self.total_steps - 1)

        if self.total_steps == 1:
            return base_lr

        if self.warmup_steps > 0 and step_index < self.warmup_steps:
            progress = step_index / self.warmup_steps
            return self.warmup_start_lr + (base_lr - self.warmup_start_lr) * progress

        cosine_steps = self.total_steps - self.warmup_steps
        if cosine_steps <= 1:
            return base_lr

        cosine_index = step_index - self.warmup_steps
        progress = cosine_index / (cosine_steps - 1)
        multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (base_lr - self.min_lr) * multiplier

    def get_lr_for_step(self, step_index: int) -> list[float]:
        """Return learning rates used by a zero-based optimizer update."""
        return [self._lr_for_group(step_index, base_lr) for base_lr in self.base_lrs]

    def _set_lrs(self, learning_rates: list[float]) -> None:
        if len(learning_rates) != len(self.optimizer.param_groups):
            raise ValueError("Learning-rate count does not match optimizer parameter groups.")
        for param_group, learning_rate in zip(self.optimizer.param_groups, learning_rates):
            param_group["lr"] = float(learning_rate)

    def step(self) -> None:
        """Advance after one successful optimizer update."""
        if self.completed_steps < self.total_steps:
            self.completed_steps += 1
        next_step = min(self.completed_steps, self.total_steps - 1)
        self._set_lrs(self.get_lr_for_step(next_step))

    def set_completed_steps(self, completed_steps: int) -> None:
        """Restore an update position when converting a legacy checkpoint."""
        if not 0 <= completed_steps <= self.total_steps:
            raise ValueError(
                f"completed_steps must be in [0, {self.total_steps}], got {completed_steps}."
            )
        self.completed_steps = int(completed_steps)
        next_step = min(self.completed_steps, self.total_steps - 1)
        self._set_lrs(self.get_lr_for_step(next_step))

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        return {
            "state_version": self.state_version,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
            "warmup_start_lr": self.warmup_start_lr,
            "base_lrs": list(self.base_lrs),
            "completed_steps": self.completed_steps,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if state_dict.get("state_version") != self.state_version:
            raise ValueError("Checkpoint does not contain an update-based scheduler state.")

        expected = {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
            "warmup_start_lr": self.warmup_start_lr,
        }
        mismatches = {
            key: (expected_value, state_dict.get(key))
            for key, expected_value in expected.items()
            if state_dict.get(key) != expected_value
        }
        stored_base_lrs = [float(value) for value in state_dict.get("base_lrs", [])]
        if stored_base_lrs != self.base_lrs:
            mismatches["base_lrs"] = (self.base_lrs, stored_base_lrs)
        if mismatches:
            details = ", ".join(
                f"{key}: current={current!r}, checkpoint={stored!r}"
                for key, (current, stored) in mismatches.items()
            )
            raise ValueError(f"Scheduler configuration mismatch: {details}")

        self.set_completed_steps(int(state_dict["completed_steps"]))
