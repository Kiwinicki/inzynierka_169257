import torch
import pytest
from src.metrics import TieAwareAccuracy


@pytest.fixture
def metric():
    return TieAwareAccuracy()


def test_tie_aware_accuracy_tie_match_first(metric):
    target = torch.tensor([[0.4, 0.4, 0.2]])
    pred = torch.tensor([[0.5, 0.3, 0.2]])
    metric.update(pred, target)
    assert metric.compute().item() == 1.0


def test_tie_aware_accuracy_tie_match_second(metric):
    target = torch.tensor([[0.4, 0.4, 0.2]])
    pred = torch.tensor([[0.3, 0.5, 0.2]])
    metric.update(pred, target)
    assert metric.compute().item() == 1.0


def test_tie_aware_accuracy_tie_mismatch(metric):
    target = torch.tensor([[0.4, 0.4, 0.2]])
    pred = torch.tensor([[0.2, 0.2, 0.6]])
    metric.update(pred, target)
    assert metric.compute().item() == 0.0


def test_tie_aware_accuracy_no_tie(metric):
    target = torch.tensor([[0.8, 0.1, 0.1]])
    pred = torch.tensor([[0.9, 0.05, 0.05]])
    metric.update(pred, target)
    assert metric.compute().item() == 1.0


def test_tie_aware_accuracy_batch(metric):
    target1 = torch.tensor([[0.4, 0.4, 0.2]])
    pred1 = torch.tensor([[0.5, 0.3, 0.2]])

    target2 = torch.tensor([[0.4, 0.4, 0.2]])
    pred2 = torch.tensor([[0.3, 0.5, 0.2]])

    target3 = torch.tensor([[0.4, 0.4, 0.2]])
    pred3 = torch.tensor([[0.2, 0.2, 0.6]])

    target4 = torch.tensor([[0.8, 0.1, 0.1]])
    pred4 = torch.tensor([[0.9, 0.05, 0.05]])

    targets = torch.cat([target1, target2, target3, target4])
    preds = torch.cat([pred1, pred2, pred3, pred4])

    metric.update(preds, targets)
    assert metric.compute().item() == 0.75
