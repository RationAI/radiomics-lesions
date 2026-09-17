"""Lesion timelines and affine-aware, physically sized NIfTI crops."""

import itertools
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import affine_transform
from torch import Tensor
from torch.utils.data import Dataset


CLASS_NAMES = ("healthy", "PD", "PSP")
LABEL_MAP = {
    "MTS": 0,
    "KAVITA": 0,
    "SD": 0,
    "PR": 0,
    "healthy": 0,
    "PD": 1,
    "PSP": 2,
}
DEFAULT_LABEL_MAP = LABEL_MAP


def crop_geometry(
    image: nib.Nifti1Image, bbox: dict, target_spacing: float, crop_size: int
) -> Tensor:
    """Return a bbox-centered RAS crop shaped (1, crop_size, crop_size, crop_size).

    The bbox uses XYZ voxel coordinates with an exclusive upper bound.
    Crop size is in voxels; isotropic target spacing is in millimeters.
    """
    center = (np.asarray(bbox["min"]) + np.asarray(bbox["max_exclusive"]) - 1) / 2
    center = nib.affines.apply_affine(image.affine, center)
    affine = np.eye(4)
    affine[:3, :3] *= target_spacing
    affine[:3, 3] = center - (crop_size - 1) * target_spacing / 2
    return resample_crop(image, (crop_size,) * 3, affine)


def resample_crop(
    image: nib.Nifti1Image,
    shape: tuple[int, int, int],
    affine: np.ndarray,
) -> Tensor:
    """Interpolate the source ROI, preserving intensities and padding with zeros."""
    transform = np.linalg.solve(image.affine, affine)
    corners = np.array(list(itertools.product(*[(0, n - 1) for n in shape])))
    source = nib.affines.apply_affine(transform, corners)
    lo = np.maximum(np.floor(source.min(0)).astype(int) - 2, 0)
    hi = np.minimum(np.ceil(source.max(0)).astype(int) + 3, image.shape)
    data = np.asarray(
        image.dataobj[
            tuple(slice(int(a), int(b)) for a, b in zip(lo, hi, strict=True))
        ],
        dtype=np.float32,
    )
    output = affine_transform(
        data,
        transform[:3, :3],
        transform[:3, 3] - lo,
        output_shape=shape,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    # NIfTI XYZ -> tensor C,D(Z),H(Y),W(X).
    return torch.from_numpy(
        np.ascontiguousarray(output.transpose(2, 1, 0), dtype=np.float32)
    ).unsqueeze(0)


class LesionDataset(Dataset):
    """One item contains every MRI, one dose crop, labels and scan metadata."""

    def __init__(
        self,
        records: list[dict],
        root: str | Path,
        registration_directory: str,
        crop_size: int = 96,
        target_spacing: float = 1.0,
    ) -> None:
        self.records = records
        self.root = Path(root)
        self.registration_directory = registration_directory
        self.crop_size = crop_size
        self.target_spacing = target_spacing

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        root = self.root / record["patient_id"] / self.registration_directory
        surgery = any(scan["scan_type"] == "predop" for scan in record["scans"])

        dose = nib.load(root / record["radiation"])
        dose = crop_geometry(
            dose,
            record["scans"][surgery]["bbox_voxel"],
            self.target_spacing,
            self.crop_size,
        )

        images = []
        labels = []

        for scan in record["scans"]:
            image = nib.load(root / scan["image"])
            crop = crop_geometry(
                image, scan["bbox_voxel"], self.target_spacing, self.crop_size
            )
            images.append(crop)
            labels.append(LABEL_MAP[scan["label"]])

        return {
            "patient_id": record["patient_id"],
            "surgery": surgery,
            "lesion_id": record["lesion_id"],
            "images": images,
            "dose": dose,
            "labels": torch.tensor(labels),
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
        "images": [x for sample in samples for x in sample["images"]],
        "doses": [sample["dose"] for sample in samples],
        "scan_indices": torch.tensor(scan_indices),
        "dose_indices": torch.tensor(dose_indices),
        "labels": torch.cat([s["labels"] for s in samples]),
        "sample_ids": torch.cat([s["sample_ids"] for s in samples]),
        "positions": torch.cat([torch.arange(n) for n in lengths.tolist()]),
        "lesion_ids": torch.repeat_interleave(torch.arange(len(samples)), lengths),
        "lengths": lengths,
        "metadata": [x for sample in samples for x in sample["metadata"]],
    }
