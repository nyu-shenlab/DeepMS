"""Pure, testable training-loop control helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EarlyStoppingUpdate:
    """State transition produced by one completed validation check."""

    improved: bool
    best_metric: float
    best_metric_epoch: int
    non_improve_validations: int
    should_stop: bool


def update_early_stopping(
    *,
    current_metric: float,
    best_metric: float,
    best_metric_epoch: int,
    non_improve_validations: int,
    epoch: int,
    patience: int,
) -> EarlyStoppingUpdate:
    """Advance strict-improvement early stopping by one validation check."""
    if patience <= 0:
        raise ValueError("Early-stopping patience must be a positive integer.")
    if non_improve_validations < 0:
        raise ValueError("non_improve_validations must be non-negative.")
    if not math.isfinite(current_metric) or not math.isfinite(best_metric):
        raise ValueError("Early-stopping metrics must be finite.")

    improved = current_metric > best_metric
    if improved:
        return EarlyStoppingUpdate(
            improved=True,
            best_metric=float(current_metric),
            best_metric_epoch=int(epoch),
            non_improve_validations=0,
            should_stop=False,
        )

    next_non_improve = non_improve_validations + 1
    return EarlyStoppingUpdate(
        improved=False,
        best_metric=float(best_metric),
        best_metric_epoch=int(best_metric_epoch),
        non_improve_validations=next_non_improve,
        should_stop=next_non_improve >= patience,
    )
