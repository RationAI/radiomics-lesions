from typing import Any

import lightning
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch import Tensor, nn
from torch.nn import functional as F

from ml.data.dataset import CLASS_NAMES
from ml.metrics import ScanClassificationMetrics
from ml.modeling.dinov3d import DinoV3D
from ml.modeling.temporal import PackedCausalTransformer, RadiationEncoder


class MetaArch(lightning.LightningModule):
    def __init__(
        self,
        dim: int = 256,
        heads: int = 8,
        layers: int = 3,
        dropout: float = 0.1,
        warmup_epochs: int = 30,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.warmup_epochs = warmup_epochs
        self.register_buffer("class_weights", torch.tensor([1.0, 2.0]))
        self.backbone = DinoV3D().eval()
        self.proj = nn.Sequential(nn.Linear(792, dim))
        self.radiation_encoder = RadiationEncoder(dim)
        self.modality_embedding = nn.Embedding(2, dim)
        self.temporal = PackedCausalTransformer(
            dim, heads, layers, dropout, num_classes=len(CLASS_NAMES)
        )

        self.val_metrics = ScanClassificationMetrics(num_classes=len(CLASS_NAMES))
        self.test_metrics = ScanClassificationMetrics(num_classes=len(CLASS_NAMES))

    def forward(self, batch: dict) -> Tensor:
        """Encode stacked N,C,Z,Y,X crops and return one class-logit row per MRI."""
        mri = self.proj(self.backbone(batch["images"]))
        dose = self.radiation_encoder(batch["doses"])
        mri = mri + self.modality_embedding.weight[0].to(mri.dtype)
        dose = dose.to(mri.dtype) + self.modality_embedding.weight[1].to(mri.dtype)

        tokens = mri.new_zeros((len(batch["lesion_ids"]), self.dim))
        tokens = tokens.index_copy(0, batch["scan_indices"], mri)
        tokens = tokens.index_copy(0, batch["dose_indices"], dose)

        logits = self.temporal(tokens, batch["lesion_ids"], batch["positions"])
        return logits[batch["scan_indices"]]

    def training_step(self, batch: dict) -> Tensor:
        logits = self(batch)
        loss = F.cross_entropy(logits, batch["labels"], weight=self.class_weights)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["labels"]),
        )
        return loss

    def validation_step(self, batch: dict) -> None:
        logits = self(batch)
        self.val_metrics.update(logits, batch["labels"], batch["sample_ids"])

    def test_step(self, batch: dict) -> None:
        logits = self(batch)
        self.test_metrics.update(logits, batch["labels"], batch["sample_ids"])

    def on_validation_epoch_end(self) -> None:
        values = self.val_metrics.compute()
        self.log_dict(
            {
                f"validation/{key}": values[key]
                for key in ("loss", "accuracy", "macro_f1", "balanced_accuracy")
            },
            prog_bar=True,
        )
        self.val_metrics.reset()

    def on_test_epoch_end(self) -> None:
        values = self.test_metrics.compute()
        self.log_dict(
            {
                f"test/{key}": values[key]
                for key in ("loss", "accuracy", "macro_f1", "balanced_accuracy")
            },
            prog_bar=True,
        )
        self.test_metrics.reset()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=1e-4 / self.trainer.accumulate_grad_batches,
            weight_decay=1e-4,
        )

        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=self.trainer.max_epochs,
            lr_min=1e-6 / self.trainer.accumulate_grad_batches,
            warmup_lr_init=1e-7 / self.trainer.accumulate_grad_batches,
            warmup_t=self.warmup_epochs,
        )
        return [optimizer], [scheduler]

    def lr_scheduler_step(
        self, scheduler: CosineLRScheduler, metric: Any | None
    ) -> None:
        scheduler.step(epoch=self.current_epoch)
