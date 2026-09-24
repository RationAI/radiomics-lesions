import json
from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler

from ml.data.augmentation import LesionAugmentation
from ml.data.dataset import CLASS_NAMES, DEFAULT_LABEL_MAP, LesionDataset, pack_lesions
from ml.data.sampling import StratifiedLesionBatchSampler


def read_manifest(path: str | Path) -> tuple[dict, list[dict]]:
    """Flatten the manifest into lesions with stable IDs for each MRI scan."""
    manifest = json.loads(Path(path).read_text())
    records = []
    next_scan_id = 0
    for patient in manifest["patients"]:
        for lesion in patient["lesions"]:
            scans = sorted(lesion["scans"], key=lambda scan: scan["scan_number"])
            records.append(
                {
                    "patient_id": patient["patient_id"],
                    "lesion_id": lesion["lesion_id"],
                    "scans": scans,
                    "radiation": lesion["radiation"],
                    "scan_ids": list(range(next_scan_id, next_scan_id + len(scans))),
                }
            )
            next_scan_id += len(scans)
    return manifest, records


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        manifest: str | Path = "/mnt/projects/radiomics/crops_192/manifest.json",
        root: str | Path | None = None,
        batch_size: int = 2,
        num_workers: int = 4,
        augmentation: LesionAugmentation | None = None,
    ) -> None:
        super().__init__()
        self.manifest = manifest
        self.root = Path(root) if root is not None else None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augmentation = augmentation
        self.label_map = DEFAULT_LABEL_MAP
        self.splits = None

    def setup(self, stage: str | None = None) -> None:
        manifest, self.records = read_manifest(self.manifest)

        if self.splits is None:
            patients = sorted({r["patient_id"] for r in self.records})
            order = np.random.permutation(patients).tolist()
            n_train = int(0.8 * len(order))
            self.splits = {
                "train": order[:n_train],
                "val": order[n_train:],
            }
        root = self.root if self.root is not None else Path(manifest["dataset_root"])
        self.datasets = {
            split: LesionDataset(
                [r for r in self.records if r["patient_id"] in group],
                root,
                augmentation=self.augmentation if split == "train" else None,
                spacing_xyz=tuple(manifest["preprocessing"]["spacing_mm"]),
            )
            for split, group in self.splits.items()
        }

    def class_counts(self, split: str = "train") -> Tensor:
        labels = [
            self.label_map[s["label"]]
            for r in self.datasets[split].records
            for s in r["scans"]
            if self.label_map[s["label"]] >= 0
        ]
        return torch.bincount(
            torch.tensor(labels, dtype=torch.long), minlength=len(CLASS_NAMES)
        )

    def _dataloader(self, split: str) -> DataLoader:
        dataset = self.datasets[split]
        batching = {"batch_size": self.batch_size}
        if split == "train":
            class_counts = torch.zeros(
                (len(dataset), len(CLASS_NAMES)), dtype=torch.long
            )
            for index, record in enumerate(dataset.records):
                for scan in record["scans"]:
                    label = self.label_map[scan["label"]]
                    if label >= 0:
                        class_counts[index, label] += 1
            batching = {
                "batch_sampler": StratifiedLesionBatchSampler(
                    RandomSampler(dataset),
                    batch_size=self.batch_size,
                    drop_last=False,
                    class_counts=class_counts,
                )
            }
        return DataLoader(
            dataset,
            **batching,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=pack_lesions,
        )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self._dataloader("val")
