import pytest

from utils.training_control import update_early_stopping


def test_early_stopping_triggers_on_fifth_consecutive_non_improvement() -> None:
    best_metric = -1.0
    best_epoch = 0
    non_improve = 0
    updates = []

    for epoch, metric in enumerate([0.70, 0.69, 0.70, 0.68, 0.67, 0.66], start=1):
        update = update_early_stopping(
            current_metric=metric,
            best_metric=best_metric,
            best_metric_epoch=best_epoch,
            non_improve_validations=non_improve,
            epoch=epoch,
            patience=5,
        )
        updates.append(update)
        best_metric = update.best_metric
        best_epoch = update.best_metric_epoch
        non_improve = update.non_improve_validations

    assert [update.improved for update in updates] == [True, False, False, False, False, False]
    assert [update.non_improve_validations for update in updates] == [0, 1, 2, 3, 4, 5]
    assert not any(update.should_stop for update in updates[:-1])
    assert updates[-1].should_stop
    assert best_metric == pytest.approx(0.70)
    assert best_epoch == 1


def test_strict_improvement_resets_early_stopping_patience() -> None:
    update = update_early_stopping(
        current_metric=0.71,
        best_metric=0.70,
        best_metric_epoch=2,
        non_improve_validations=4,
        epoch=7,
        patience=5,
    )

    assert update.improved
    assert update.best_metric == pytest.approx(0.71)
    assert update.best_metric_epoch == 7
    assert update.non_improve_validations == 0
    assert not update.should_stop


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "current_metric": float("nan"),
            "best_metric": 0.7,
            "best_metric_epoch": 1,
            "non_improve_validations": 0,
            "epoch": 2,
            "patience": 5,
        },
        {
            "current_metric": 0.7,
            "best_metric": 0.6,
            "best_metric_epoch": 1,
            "non_improve_validations": 0,
            "epoch": 2,
            "patience": 0,
        },
    ],
)
def test_early_stopping_rejects_invalid_state(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        update_early_stopping(**kwargs)
