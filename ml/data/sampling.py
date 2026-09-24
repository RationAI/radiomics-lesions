"""Group complete lesion timelines by their scan-level class composition."""

from collections.abc import Iterator

from torch import Tensor
from torch.utils.data import BatchSampler, Sampler


class StratifiedLesionBatchSampler(BatchSampler):
    """Greedily approximate the cohort's scan-class ratio without resampling.

    The underlying sampler determines the available indices and random tie order.
    This also lets Lightning inject its distributed sampler: grouping stays local
    to each rank, while the target proportions still use the full training split.
    Whole mixed-label timelines and integer batch sizes prevent exact quotas.
    """

    def __init__(
        self,
        sampler: Sampler[int],
        batch_size: int,
        drop_last: bool,
        class_counts: Tensor,
    ) -> None:
        super().__init__(sampler, batch_size, drop_last)
        self.class_counts = class_counts.detach().cpu().double()
        counts = self.class_counts.sum(dim=0)
        self.proportions = counts / counts.sum().clamp_min(1)

    def __iter__(self) -> Iterator[list[int]]:
        remaining = list(self.sampler)
        while remaining:
            if self.drop_last and len(remaining) < self.batch_size:
                return
            # Random anchors retain epoch-to-epoch variety; fill each batch with
            # the remaining timelines that best correct its class proportions.
            batch = [remaining.pop(0)]
            counts = self.class_counts[batch[0]].clone()
            while remaining and len(batch) < self.batch_size:
                candidates = counts + self.class_counts[remaining]
                ratios = candidates / candidates.sum(dim=1, keepdim=True).clamp_min(1)
                error = (ratios - self.proportions).square().sum(dim=1)
                index = remaining.pop(int(error.argmin()))
                batch.append(index)
                counts += self.class_counts[index]
            yield batch
