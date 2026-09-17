"""TorchMetrics evaluation over unique scans, including distributed sampler padding."""

import torch
from torch import Tensor
from torch.nn import functional as F
from torchmetrics import MeanMetric, Metric, MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from torchmetrics.functional.classification import (
    binary_auroc,
    binary_average_precision,
)
from torchmetrics.utilities.data import dim_zero_cat


class ScanClassificationMetrics(Metric):
    """Accumulate scans and compute classification metrics exactly once per identity.

    TorchMetrics synchronizes tensor states across ranks at compute time. The
    gathered predictions are deduplicated before calling the standard metrics;
    report metadata never participates in numerical evaluation. State memory is
    linear in the number of scans (three logits, one target and one ID per scan).
    """

    full_state_update = False
    is_differentiable = False
    higher_is_better = None

    def __init__(self, num_classes: int = 3, ignore_index: int = -100) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.add_state("logits", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.add_state("sample_ids", default=[], dist_reduce_fx="cat")

    def update(self, logits: Tensor, targets: Tensor, sample_ids: Tensor) -> None:
        """Accept raw N-by-C logits, integer class targets and global scan IDs."""
        self.logits.append(logits.detach().float())
        self.targets.append(targets.detach())
        self.sample_ids.append(sample_ids.detach())

    def compute(self) -> dict[str, Tensor]:
        logits = dim_zero_cat(self.logits)
        targets = dim_zero_cat(self.targets)
        ids = dim_zero_cat(self.sample_ids)
        order = ids.argsort(stable=True)
        first = torch.ones_like(order, dtype=torch.bool)
        first[1:] = ids[order][1:] != ids[order][:-1]
        keep = order[first]
        targets, logits = targets[keep], logits[keep]
        valid = targets != self.ignore_index
        context_count = (~valid).sum()
        logits, targets = logits[valid], targets[valid]
        probabilities = logits.softmax(-1)
        # Outer Metric has already synchronized and deduplicated the cohort.
        # Disable inner synchronization to avoid a second distributed reduction.
        metrics = MetricCollection(
            {
                "accuracy": MulticlassAccuracy(
                    average="micro", num_classes=self.num_classes, sync_on_compute=False
                ),
                "precision": MulticlassPrecision(
                    average=None, num_classes=self.num_classes, sync_on_compute=False
                ),
                "recall": MulticlassRecall(
                    average=None, num_classes=self.num_classes, sync_on_compute=False
                ),
                "f1": MulticlassF1Score(
                    average=None, num_classes=self.num_classes, sync_on_compute=False
                ),
                "confusion_matrix": MulticlassConfusionMatrix(
                    num_classes=self.num_classes, sync_on_compute=False
                ),
            },
            compute_groups=False,
        ).to(logits.device)
        metrics.update(probabilities, targets)
        result = metrics.compute()
        support = result["confusion_matrix"].sum(dim=1)
        loss = MeanMetric(sync_on_compute=False).to(logits.device)
        loss.update(F.cross_entropy(logits, targets, reduction="none"))
        auroc = logits.new_full((self.num_classes,), torch.nan)
        average_precision = auroc.clone()
        for index in range(self.num_classes):
            if 0 < support[index] < targets.numel():
                binary = (targets == index).long()
                auroc[index] = binary_auroc(probabilities[:, index], binary)
                average_precision[index] = binary_average_precision(
                    probabilities[:, index], binary
                )
        return result | {
            "support": support,
            "scans": support.sum(),
            "context_only_scans": context_count,
            "macro_f1": result["f1"].mean(),
            "balanced_accuracy": result["recall"][support > 0].mean(),
            "loss": loss.compute(),
            "auroc": auroc,
            "average_precision": average_precision,
        }
