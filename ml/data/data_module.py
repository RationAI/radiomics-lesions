import json
from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ml.data.dataset import DEFAULT_LABEL_MAP, LesionDataset, pack_lesions


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
                    "radiation": patient["radiation"],
                    "scan_ids": list(range(next_scan_id, next_scan_id + len(scans))),
                }
            )
            next_scan_id += len(scans)
    return manifest, records


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        manifest: str | Path = "manifest.json",
        root: str | Path | None = None,
        batch_size: int = 2,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.manifest = manifest
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.label_map = DEFAULT_LABEL_MAP
        self.splits = None

    def setup(self, stage: str | None = None) -> None:
        manifest, self.records = read_manifest(self.manifest)

        if self.splits is None:
            patients = sorted({r["patient_id"] for r in self.records})
            order = np.random.permutation(patients).tolist()
            n_train, n_val = int(0.7 * len(order)), int(0.1 * len(order))
            self.splits = {
                "train": order[:n_train],
                "val": order[n_train : n_train + n_val],
                "test": order[n_train + n_val :],
            }
        root = Path(manifest["dataset_root"])
        self.datasets = {
            split: LesionDataset(
                [r for r in self.records if r["patient_id"] in group],
                root,
                manifest["registration_directory"],
                label_map=self.label_map,
                augment=split == "train",
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
        return torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=3)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=pack_lesions,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets["val"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=pack_lesions,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=pack_lesions,
        )
