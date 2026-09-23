"""Materialize lesion-centered 192³, 1 mm MRI and dose crops for fast loading."""

import itertools
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import ray
from scipy.ndimage import affine_transform
from tqdm import tqdm


MANIFEST_PATH = Path("manifest.json")
DATASET_ROOT: Path | None = None  # None uses dataset_root from the manifest.
OUTPUT_DIRECTORY = Path("/mnt/projects/radiomics/crops_192")
CROP_SIZE = 192
SPACING_MM = 1.0
CLIP_PERCENTILES = (0.5, 99.5)


def crop_affine(image: nib.Nifti1Image, bbox: dict) -> np.ndarray:
    """Center a RAS-aligned grid on an XYZ, max-exclusive source voxel bbox."""
    lower = np.asarray(bbox["min"], dtype=float)
    upper = np.asarray(bbox["max_exclusive"], dtype=float)
    if lower.shape != (3,) or upper.shape != (3,) or not np.all(upper > lower):
        raise ValueError(f"Invalid bounding box: {bbox}")
    center = nib.affines.apply_affine(image.affine, (lower + upper - 1) / 2)
    affine = np.diag([SPACING_MM, SPACING_MM, SPACING_MM, 1.0])
    affine[:3, 3] = center - (CROP_SIZE - 1) * SPACING_MM / 2
    return affine


def resample(
    image: nib.Nifti1Image, affine: np.ndarray, statistics: dict | None = None
) -> np.ndarray:
    """Read the source ROI, optionally normalize it, then interpolate the crop."""
    if len(image.shape) != 3:
        raise ValueError(f"Expected a 3D volume, got {image.shape}")
    units = image.header.get_xyzt_units()[0]
    if units not in ("unknown", "mm"):
        raise ValueError(f"Expected NIfTI spatial units in mm, got {units!r}")
    transform = np.linalg.solve(image.affine, affine)
    corners = np.array(list(itertools.product((0, CROP_SIZE - 1), repeat=3)))
    source = nib.affines.apply_affine(transform, corners)
    lower = np.maximum(np.floor(source.min(0)).astype(int) - 2, 0)
    upper = np.minimum(np.ceil(source.max(0)).astype(int) + 3, image.shape)
    if np.any(upper <= lower):
        return np.zeros((CROP_SIZE,) * 3, dtype=np.float32)
    region = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper, strict=True))
    data = np.asarray(image.dataobj[region], dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError("Source crop contains NaN or infinite intensities")
    if statistics is not None:
        data = zscore(data, statistics)
    return affine_transform(
        data,
        transform[:3, :3],
        transform[:3, 3] - lower,
        output_shape=(CROP_SIZE,) * 3,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )


def scan_statistics(image: nib.Nifti1Image) -> dict:
    """Compute clipping bounds and clipped whole-scan statistics, including zeros."""
    if len(image.shape) != 3:
        raise ValueError(f"Expected a 3D volume, got {image.shape}")
    # Own this buffer: percentile partitioning may reorder it, which is harmless
    # for statistics but must never modify the source image or its cached data.
    data = np.array(image.dataobj, dtype=np.float32, copy=True)
    if not np.isfinite(data).all():
        raise ValueError("MRI scan contains NaN or infinite intensities")
    lower, upper = np.percentile(data, CLIP_PERCENTILES, overwrite_input=True)
    np.clip(data, lower, upper, out=data)
    mean = float(data.mean(dtype=np.float64))
    # Accumulate variance in planes to avoid another full-volume float64 buffer.
    variance = (
        sum(np.square(plane.astype(np.float64) - mean).sum() for plane in data)
        / data.size
    )
    return {
        "clip_lower": float(lower),
        "clip_upper": float(upper),
        "mean": mean,
        "std": float(np.sqrt(variance)),
    }


def zscore(data: np.ndarray, statistics: dict) -> np.ndarray:
    """Clip and normalize source voxels using shared whole-scan statistics."""
    if statistics["std"] == 0:
        return np.zeros_like(data)
    clipped = np.clip(data, statistics["clip_lower"], statistics["clip_upper"])
    return (
        (clipped.astype(np.float64) - statistics["mean"]) / statistics["std"]
    ).astype(np.float32)


def save_crop(output: Path, relative: Path, data: np.ndarray) -> str:
    """Save contiguous float32 CZYX arrays, ready for torch.from_numpy."""
    path = output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.ascontiguousarray(data.transpose(2, 1, 0)[None], dtype=np.float32)
    np.save(path, array, allow_pickle=False)
    return relative.as_posix()


@ray.remote(num_cpus=1)
def preprocess_patient(
    patient: dict, root: Path, registration: str, output: Path
) -> dict:
    """Process each lesion independently, retaining source annotation metadata."""
    source_dir = root / patient["patient_id"] / registration
    result = {"patient_id": patient["patient_id"], "lesions": []}
    # Reuse file proxies for repeated scans without retaining whole volumes in RAM.
    images = {}
    normalization = {}

    def load(filename: str) -> nib.Nifti1Image:
        if filename not in images:
            images[filename] = nib.load(source_dir / filename)
        return images[filename]

    for lesion in patient["lesions"]:
        scans = sorted(lesion["scans"], key=lambda scan: scan["scan_number"])
        if not scans:
            raise ValueError(
                f"Empty lesion: {patient['patient_id']} / {lesion['lesion_id']}"
            )
        prefix = Path(patient["patient_id"]) / f"lesion_{lesion['lesion_id']}"
        prepared = {"lesion_id": lesion["lesion_id"], "scans": []}
        affines = []
        for index, scan in enumerate(scans):
            image = load(scan["image"])
            affine = crop_affine(image, scan["bbox_voxel"])
            if scan["image"] not in normalization:
                normalization[scan["image"]] = scan_statistics(image)
            statistics = normalization[scan["image"]]
            data = resample(image, affine, statistics)
            filename = save_crop(output, prefix / f"scan_{index:03d}.npy", data)
            prepared["scans"].append(
                {
                    **{
                        k: v
                        for k, v in scan.items()
                        if k not in ("image", "mask", "bbox_voxel")
                    },
                    "image": filename,
                    "affine_xyz": affine.tolist(),
                    "normalization": statistics,
                    "source": scan,
                }
            )
            affines.append(affine)
        # Prefer planning; otherwise last pretreatment, then first available MRI.
        anchor = next(
            (i for i, s in enumerate(scans) if s["scan_type"] == "plan"), None
        )
        if anchor is None:
            anchor = next(
                (
                    i
                    for i in reversed(range(len(scans)))
                    if scans[i]["scan_type"] == "predop"
                ),
                0,
            )
        dose = resample(load(patient["radiation"]), affines[anchor])
        prepared["radiation"] = save_crop(output, prefix / "dose.npy", dose)
        prepared["radiation_affine_xyz"] = affines[anchor].tolist()
        prepared["radiation_anchor_scan_number"] = scans[anchor]["scan_number"]
        prepared["source_radiation"] = patient["radiation"]
        result["lesions"].append(prepared)
    return result


def main(manifest_path: Path, output: Path, root: Path | None) -> Path:
    """Write a new dataset directory; publish its manifest only after all crops succeed."""
    manifest = json.loads(manifest_path.read_text())
    root = (root or Path(manifest["dataset_root"])).expanduser().resolve()
    output = output.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Input directory does not exist: {root}")
    patients = manifest["patients"]
    if len({p["patient_id"] for p in patients}) != len(patients):
        raise ValueError("Duplicate patient IDs")
    for patient in patients:
        if len({l["lesion_id"] for l in patient["lesions"]}) != len(patient["lesions"]):
            raise ValueError(f"Duplicate lesion IDs: {patient['patient_id']}")
        for part in [
            patient["patient_id"],
            *(str(l["lesion_id"]) for l in patient["lesions"]),
        ]:
            if part in ("", ".", "..") or "/" in part or "\\" in part:
                raise ValueError(f"Invalid patient or lesion ID: {part!r}")
    # Refuse existing destinations to prevent replacing raw scans or mixing runs.
    output.mkdir(parents=True, exist_ok=False)
    pending = {
        preprocess_patient.remote(
            patient, root, manifest["registration_directory"], output
        ): index
        for index, patient in enumerate(patients)
    }
    records = [None] * len(patients)
    with tqdm(total=len(patients), desc="Preprocessing", unit="patient") as progress:
        while pending:
            ready, _ = ray.wait(list(pending), num_returns=1)
            reference = ready[0]
            records[pending.pop(reference)] = ray.get(reference)
            progress.update(1)
    result = {
        "schema_version": "preprocessed-1.0",
        "dataset_root": str(output),
        "path_resolution": "dataset_root / image_or_radiation_path",
        "source_manifest": str(manifest_path.resolve()),
        "source_dataset_root": str(root),
        "source_registration_directory": manifest["registration_directory"],
        "preprocessing": {
            "shape_czyx": [1, CROP_SIZE, CROP_SIZE, CROP_SIZE],
            "spacing_mm": [SPACING_MM] * 3,
            "orientation": "RAS",
            "dtype": "float32",
            "interpolation": "linear",
            "padding_value": 0,
            "mri_normalization": "clipped whole-scan z-score over all voxels, population std",
            "clip_percentiles": list(CLIP_PERCENTILES),
            "normalization_order": "clip and normalize source voxels before resampling",
            "constant_scan": "zeros",
            "dose_normalization": "none; original units preserved",
            "affine_convention": "affine_xyz maps XYZ voxel indices to RAS millimeters",
        },
        "patients": records,
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return path


if __name__ == "__main__":
    main(MANIFEST_PATH, OUTPUT_DIRECTORY, DATASET_ROOT)
