"""Write evaluation artifacts independently of Lightning optimization and metrics."""

from tempfile import TemporaryDirectory

import torch.distributed as dist
from lightning import Callback, LightningModule, Trainer

from ml.evaluation import format_report, prediction_rows, write_report


class EvaluationWriter(Callback):
    """Gather report metadata only; TorchMetrics owns all numerical computation."""

    def __init__(self) -> None:
        self.rows = []

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self.rows.clear()

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict,
        batch: dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not trainer.sanity_checking:
            self.rows.extend(prediction_rows(outputs, batch["metadata"]))

    def _write(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if trainer.sanity_checking:
            return
        metrics = getattr(pl_module, f"{stage}_metrics")
        # Callback epoch-end hooks precede module epoch-end hooks. compute() is
        # cached by TorchMetrics; the module subsequently logs and resets it.
        values = metrics.compute()
        rows = self.rows
        if dist.is_initialized():
            gathered = [[] for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered, rows)
            rows = [row for group in gathered for row in group]
        if trainer.is_global_zero:
            unique = {
                (r["patient_id"], r["lesion_id"], r["scan_number"]): r for r in rows
            }
            rows = [unique[key] for key in sorted(unique)]
            report = format_report(values, rows)
            with TemporaryDirectory() as directory:
                write_report(directory, stage, rows, report)
                trainer.logger.log_artifacts(directory, "evaluation")
        self.rows.clear()

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self._write(trainer, pl_module, "val")
