import torch
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
)


class PairwiseAccuracy(Metric):
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("correct_pairs", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "total_valid_pairs", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        preds: (B, Num_Classes) - Probabilities or Logits
        target: (B, Num_Classes) - Votes or Distribution
        """
        # Compare each with each other
        pred_diff = preds.unsqueeze(2) - preds.unsqueeze(1) # [B, C, 1] - [B, 1, C] -> [B, C, C]
        target_diff = target.unsqueeze(2) - target.unsqueeze(1) # [B, C, 1] - [B, 1, C] -> [B, C, C]

        # Don't count with ties in target
        valid_mask = target_diff > 0

        # Check if model also predicted i > j
        correct_predictions = (pred_diff > 0) & valid_mask

        self.correct_pairs += correct_predictions.sum()
        self.total_valid_pairs += valid_mask.sum()

    def compute(self):
        if self.total_valid_pairs == 0:
            return torch.tensor(0.0)
        return self.correct_pairs / self.total_valid_pairs


class Metrics:
    def __init__(self, num_classes, device):
        self.pairwise_acc = PairwiseAccuracy().to(device)
        self.collection = MetricCollection(
            {
                "acc": MulticlassAccuracy(num_classes=num_classes),
                "precision": MulticlassPrecision(
                    num_classes=num_classes, average="macro"
                ),
                "recall": MulticlassRecall(num_classes=num_classes, average="macro"),
                "f1": MulticlassF1Score(num_classes=num_classes, average="macro"),
                "conf_mat": MulticlassConfusionMatrix(num_classes=num_classes),
            }
        ).to(device)

    def update(self, preds, target):
        self.pairwise_acc.update(preds, target)

        # Convert to indices for binary metrics
        if target.dim() > 1 and target.is_floating_point():
            target_indices = torch.argmax(target, dim=1)
        else:
            target_indices = target

        self.collection.update(preds, target_indices)

    def compute(self):
        res = self.collection.compute()
        res["pairwise_acc"] = self.pairwise_acc.compute()
        return res

    def reset(self):
        self.collection.reset()
        self.pairwise_acc.reset()
