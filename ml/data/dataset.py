"""Load preprocessed lesion crops and pack their chronological MRI/dose events."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ml.data.augmentation import LesionAugmentation


CLASS_NAMES = ("healthy", "PD")
LABEL_MAP = {
    "MTS": 0,
    "KAVITA": 0,
    "SD": 0,
    "PR": 0,
    "healthy": 0,
    "PD": 1,
    "PSP": 0,
}
DEFAULT_LABEL_MAP = LABEL_MAP


class LesionDataset(Dataset):
    """One item contains a lesion's saved MRI crops, dose and scan metadata."""

    def __init__(
        self,
        records: list[dict],
        root: str | Path,
        augmentation: LesionAugmentation | None = None,
        spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        self.records = records
        self.root = Path(root)
        self.augmentation = augmentation
        self.spacing_xyz = spacing_xyz

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        scans = record["scans"]
        # Radiation follows pretreatment scans and precedes the first follow-up.
        insertion = next(
            (
                i
                for i, scan in enumerate(scans)
                if scan["scan_type"] not in ("predop", "plan")
            ),
            len(scans),
        )
        images = [
            torch.from_numpy(np.load(self.root / scan["image"], allow_pickle=False))
            for scan in scans
        ]
        dose = torch.from_numpy(
            np.load(self.root / record["radiation"], allow_pickle=False)
        )
        if self.augmentation is not None:
            images, dose = self.augmentation(images, dose, self.spacing_xyz)
        return {
            "images": images,
            "dose": dose,
            "dose_insertion": insertion,
            "labels": torch.tensor([LABEL_MAP[scan["label"]] for scan in scans]),
            "sample_ids": torch.tensor(record["scan_ids"]),
            "metadata": [
                {
                    "patient_id": record["patient_id"],
                    "lesion_id": record["lesion_id"],
                    "scan_number": scan["scan_number"],
                    "scan_type": scan["scan_type"],
                    "label": scan["label"],
                    "image": scan["image"],
                }
                for scan in scans
            ],
        }


def pack_lesions(samples: list[dict]) -> dict:
    """Pack MRI/dose events into one sequence; labels index only the MRI tokens."""
    lengths = torch.tensor([len(sample["images"]) + 1 for sample in samples])
    scan_indices, dose_indices, offset = [], [], 0
    for sample in samples:
        insertion = sample["dose_insertion"]
        count = len(sample["images"])
        scan_indices.extend(offset + i + (i >= insertion) for i in range(count))
        dose_indices.append(offset + insertion)
        offset += count + 1
    return {
        "images": torch.stack([x for sample in samples for x in sample["images"]]),
        "doses": torch.stack([sample["dose"] for sample in samples]),
        "scan_indices": torch.tensor(scan_indices),
        "dose_indices": torch.tensor(dose_indices),
        "labels": torch.cat([s["labels"] for s in samples]),
        "sample_ids": torch.cat([s["sample_ids"] for s in samples]),
        "positions": torch.cat([torch.arange(n) for n in lengths.tolist()]),
        "lesion_ids": torch.repeat_interleave(torch.arange(len(samples)), lengths),
        "lengths": lengths,
        "metadata": [x for sample in samples for x in sample["metadata"]],
    }
