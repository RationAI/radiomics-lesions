"""Lightning training with frozen or fine-tuned MRI features and packed timelines."""

import math
from collections import defaultdict

import lightning as L
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ml.metrics import ScanClassificationMetrics
from ml.modeling.dinov3d import DinoV3D
from ml.modeling.temporal import PackedCausalTransformer, RadiationEncoder


class MetaArch(L.LightningModule):
    def __init__(
        self, dim: int = 256, heads: int = 8, layers: int = 3, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.backbone = DinoV3D().eval()
        self.mri_projection = nn.Sequential(nn.LayerNorm(792), nn.Linear(792, dim))
        self.radiation_encoder = RadiationEncoder(dim)
        self.modality_embedding = nn.Embedding(2, dim)
        self.temporal = PackedCausalTransformer(dim, heads, layers, dropout)

        self.val_metrics = ScanClassificationMetrics()
        self.test_metrics = ScanClassificationMetrics()

    def train(self, mode: bool = True) -> "MetaArch":
        super().train(mode)
        self.backbone.eval()
        return self

    def _encode(
        self, encoder: nn.Module, volumes: list[Tensor], frozen: bool = False
    ) -> Tensor:
        groups = defaultdict(list)
        for index, volume in enumerate(volumes):
            groups[tuple(volume.shape)].append(index)
        features = {}
        for indices in groups.values():
            for start in range(0, len(indices), self.hparams.encoder_batch_size):
                chunk = indices[start : start + self.hparams.encoder_batch_size]
                with torch.set_grad_enabled(torch.is_grad_enabled() and not frozen):
                    values = encoder(torch.stack([volumes[i] for i in chunk]))
                for index, value in zip(chunk, values, strict=True):
                    features[index] = value
        return torch.stack([features[i] for i in range(len(volumes))])

    def forward(self, batch: dict) -> Tensor:
        mri = self.mri_projection(
            self._encode(self.mri_encoder, batch["images"], self.hparams.freeze_mri)
        )
        dose = self._encode(self.radiation_encoder, batch["doses"])
        tokens = mri.new_zeros((len(batch["lesion_ids"]), self.hparams.dim))
        tokens = tokens.index_copy(
            0, batch["scan_indices"], mri + self.modality_embedding.weight[0]
        )
        tokens = tokens.index_copy(
            0, batch["dose_indices"], dose + self.modality_embedding.weight[1]
        )
        logits = self.temporal(tokens, batch["lesion_ids"], batch["positions"])
        return logits[batch["scan_indices"]]

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        logits = self(batch)
        valid = batch["labels"] >= 0
        if not valid.any():
            return logits.sum() * 0
        loss = F.binary_cross_entropy_with_logits(
            logits[valid].float(),
            batch["labels"][valid],
            weight=self.class_weights,
        )
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            batch_size=int(valid.sum()),
            sync_dist=True,
        )
        return loss

    def _evaluation_step(self, batch: dict, metrics: ScanClassificationMetrics) -> dict:
        logits = self(batch).float()
        metrics.update(logits, batch["labels"], batch["sample_ids"])
        return {
            "logits": logits.detach(),
            "targets": batch["labels"],
            "sample_ids": batch["sample_ids"],
        }

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        return self._evaluation_step(batch, self.val_metrics)

    def test_step(self, batch: dict, batch_idx: int) -> dict:
        return self._evaluation_step(batch, self.test_metrics)

    def _log_metrics(self, stage: str, metrics: ScanClassificationMetrics) -> None:
        values = metrics.compute()
        self.log_dict(
            {
                f"{stage}/{key}": values[key]
                for key in ("loss", "accuracy", "macro_f1", "balanced_accuracy")
            },
            prog_bar=True,
            sync_dist=False,  # TorchMetrics already synchronized the cohort.
        )
        metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self._log_metrics("val", self.val_metrics)

    def on_test_epoch_end(self) -> None:
        self._log_metrics("test", self.test_metrics)

    def configure_optimizers(self) -> dict:
        groups = defaultdict(list)
        for name, param in self.named_parameters():
            if param.requires_grad:
                backbone = name.startswith("mri_encoder.")
                decay = param.ndim > 1 and not any(k in name for k in ("token", "norm"))
                groups[(backbone, decay)].append(param)
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": parameters,
                    "lr": self.hparams.learning_rate
                    * (self.hparams.backbone_lr_multiplier if backbone else 1),
                    "weight_decay": self.hparams.weight_decay if decay else 0.0,
                }
                for (backbone, decay), parameters in groups.items()
            ],
            betas=(0.9, 0.999),
        )
        steps = int(self.trainer.estimated_stepping_batches)
        warmup = int(steps * self.hparams.warmup_fraction)

        def schedule(step: int) -> float:
            if step < warmup:
                return (step + 1) / max(warmup, 1)
            progress = min(1.0, (step - warmup) / max(1, steps - warmup - 1))
            return self.hparams.min_lr_ratio + (1 - self.hparams.min_lr_ratio) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, schedule),
                "interval": "step",
            },
        }
