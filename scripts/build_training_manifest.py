#!/usr/bin/env python3
"""Build a JSON training manifest from registered MRI, dose, and mask volumes."""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from tqdm import tqdm


if __package__:
    from scripts.analyze_patient_scans import (
        PatientFolder,
        Scan,
        Structure,
        analyze_patient,
        discover_patients,
        read_nifti_geometry,
    )
else:
    from analyze_patient_scans import (  # type: ignore[import-not-found]
        PatientFolder,
        Scan,
        Structure,
        analyze_patient,
        discover_patients,
        read_nifti_geometry,
    )


DATE_KEYS = (
    "AcquisitionDateTime",
    "AcquisitionDate",
    "SeriesDate",
    "StudyDate",
    "ContentDate",
)
ISO_DATE_RE = re.compile(r"(?<!\d)(20\d{2})[-./]?([01]\d)[-./]?([0-3]\d)(?!\d)")
EUROPEAN_DATE_RE = re.compile(r"(?<!\d)([0-3]\d)[-./]([01]\d)[-./](20\d{2})(?!\d)")


def relative_path(path: Path, dataset_root: Path) -> str:
    try:
        return path.resolve().relative_to(dataset_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def sidecar_path(path: Path) -> Path:
    if path.name.lower().endswith(".nii.gz"):
        return path.with_name(f"{path.name[:-7]}.json")
    return path.with_suffix(".json")


def normalize_date(value: object) -> str | None:
    text = str(value)
    match = ISO_DATE_RE.search(text)
    if match:
        year, month, day = match.groups()
        return f"{year}-{month}-{day}"
    match = EUROPEAN_DATE_RE.search(text)
    if match:
        day, month, year = match.groups()
        return f"{year}-{month}-{day}"
    return None


def scan_creation_date(path: Path) -> tuple[str | None, str | None]:
    """Return a date only from embedded NIfTI metadata or a JSON sidecar."""
    sidecar = sidecar_path(path)
    if sidecar.is_file():
        try:
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
        for key in DATE_KEYS:
            date = normalize_date(metadata.get(key))
            if date:
                return date, f"sidecar:{key}"

    image = nib.load(path, mmap=False)
    header_values: list[tuple[str, str]] = []
    for field in ("descrip", "aux_file", "db_name", "intent_name"):
        if field not in image.header:
            continue
        value = bytes(image.header[field]).rstrip(b"\x00").decode("utf-8", "ignore")
        header_values.append((field, value))
    for field, value in header_values:
        date = normalize_date(value)
        if date:
            return date, f"nifti_header:{field}"

    for index, extension in enumerate(image.header.extensions):
        try:
            content = extension.get_content()
        except (NotImplementedError, ValueError):
            continue
        text = (
            content.decode("utf-8", "ignore")
            if isinstance(content, bytes)
            else str(content)
        )
        date = normalize_date(text)
        if date:
            return date, f"nifti_extension:{index}"
    return None, None


def mask_bbox(path: Path) -> tuple[dict[str, Any] | None, int]:
    """Return an XYZ voxel bbox with an exclusive upper bound."""
    image = nib.load(path, mmap=False)
    data = np.asanyarray(image.dataobj)
    foreground = np.isfinite(data) & (data != 0)
    if foreground.ndim > 3:
        foreground = np.any(foreground, axis=tuple(range(3, foreground.ndim)))
    coordinates = np.nonzero(foreground)
    if not coordinates or coordinates[0].size == 0:
        return None, 0
    minimum = [int(axis.min()) for axis in coordinates[:3]]
    maximum_exclusive = [int(axis.max()) + 1 for axis in coordinates[:3]]
    return (
        {
            "min": minimum,
            "max_exclusive": maximum_exclusive,
            "size": [
                upper - lower
                for lower, upper in zip(minimum, maximum_exclusive, strict=True)
            ],
        },
        int(np.count_nonzero(foreground)),
    )


def volume_record(path: Path, dataset_root: Path, scan: Scan) -> dict[str, Any]:
    geometry = read_nifti_geometry(path)
    date, date_source = scan_creation_date(path)
    return {
        "scan_number": scan.scan_number,
        "scan_label": scan.scan_label,
        "image_type": scan.image_type,
        "image": relative_path(path, dataset_root),
        "dimensions_px": list(geometry.dimensions_px),
        "voxel_spacing_mm": (
            list(geometry.spacing_mm_per_px)
            if geometry.spacing_mm_per_px is not None
            else None
        ),
        "scan_creation_date": date,
        "scan_creation_date_source": date_source,
    }


def annotation_record(
    structure: Structure,
    mask_path: Path,
    image_record: dict[str, Any] | None,
    dataset_root: Path,
) -> dict[str, Any]:
    geometry = read_nifti_geometry(mask_path)
    bbox, foreground_voxel_count = mask_bbox(mask_path)
    response_label = (
        structure.response_label if structure.response_label != "NONE" else None
    )
    return {
        "scan_number": structure.scan_number,
        "scan_label": image_record["scan_label"] if image_record else None,
        "label": response_label or structure.structure_type,
        "structure_type": structure.structure_type,
        "response_label": response_label,
        "image": image_record["image"] if image_record else None,
        "mask": relative_path(mask_path, dataset_root),
        "mask_dimensions_px": list(geometry.dimensions_px),
        "mask_voxel_spacing_mm": (
            list(geometry.spacing_mm_per_px)
            if geometry.spacing_mm_per_px is not None
            else None
        ),
        "scan_creation_date": (
            image_record["scan_creation_date"] if image_record else None
        ),
        "scan_creation_date_source": (
            image_record["scan_creation_date_source"] if image_record else None
        ),
        "bbox_voxel": bbox,
        "bbox_convention": "XYZ, zero-based, max_exclusive",
        "foreground_voxel_count": foreground_voxel_count,
    }


def compact_annotation(annotation: dict[str, Any]) -> dict[str, Any]:
    bbox = annotation["bbox_voxel"]
    compact: dict[str, Any] = {
        "scan_number": annotation["scan_number"],
        "scan_type": annotation["scan_label"],
        "label": annotation["label"],
        "image": Path(annotation["image"]).name if annotation["image"] else None,
        "mask": Path(annotation["mask"]).name,
        "bbox_voxel": (
            {"min": bbox["min"], "max_exclusive": bbox["max_exclusive"]}
            if bbox is not None
            else None
        ),
        "foreground_voxel_count": annotation["foreground_voxel_count"],
    }
    if annotation["label"] != annotation["structure_type"]:
        compact["structure_type"] = annotation["structure_type"]
    if annotation["scan_creation_date"] is not None:
        compact["scan_creation_date"] = annotation["scan_creation_date"]
    return compact


def same_values(values: list[tuple[float, ...]], tolerance: float = 1e-5) -> bool:
    return not values or all(
        np.allclose(values[0], value, rtol=0.0, atol=tolerance) for value in values[1:]
    )


def build_patient_record(patient: PatientFolder, dataset_root: Path) -> dict[str, Any]:
    analysis = analyze_patient(
        patient.patient_id, patient.files, patient.source_directory
    )
    files_by_name = {path.name: path for path in patient.files}
    errors: list[str] = []
    warnings: list[str] = list(analysis["warnings"])
    scan_records: list[tuple[Scan, dict[str, Any]]] = []

    for scan in analysis["scans"]:
        path = files_by_name.get(scan.image_filename)
        if path is None:
            errors.append(f"Missing image for {scan.scan_label}")
            continue
        try:
            record = volume_record(path, dataset_root, scan)
        except (OSError, ValueError, nib.filebasedimages.ImageFileError) as error:
            errors.append(f"Could not read {path.name}: {error}")
            continue
        scan_records.append((scan, record))

    spacings = [
        tuple(record["voxel_spacing_mm"])
        for _, record in scan_records
        if record["voxel_spacing_mm"] is not None
    ]
    dimensions = [tuple(record["dimensions_px"]) for _, record in scan_records]
    spacing_consistent = (
        bool(scan_records)
        and len(spacings) == len(scan_records)
        and same_values(spacings)
    )
    dimensions_consistent = bool(scan_records) and len(set(dimensions)) <= 1
    reference_spacing = list(spacings[0]) if spacings else None
    isotropic_spacing = (
        float(reference_spacing[0])
        if reference_spacing
        and np.allclose(reference_spacing, reference_spacing[0], rtol=0.0, atol=1e-5)
        else None
    )
    if not scan_records:
        errors.append("No readable image volumes")
    if not spacing_consistent:
        errors.append("Voxel spacing is missing or differs across image volumes")
    if reference_spacing is not None and isotropic_spacing is None:
        errors.append("Voxel spacing is not isotropic within each spatial dimension")
    if not dimensions_consistent:
        errors.append("Voxel dimensions differ across image volumes")

    records_by_number = {
        scan.scan_number: record
        for scan, record in scan_records
        if scan.image_type == "MRI" and scan.scan_number is not None
    }
    radiation_records = [
        record for scan, record in scan_records if scan.image_type == "RTDOSE"
    ]
    if len(radiation_records) != 1:
        errors.append(
            f"Expected exactly one RTDOSE volume, found {len(radiation_records)}"
        )
    radiation = Path(radiation_records[0]["image"]).name if radiation_records else None

    lesions: dict[int, dict[str, Any]] = {}
    auxiliary_masks: list[dict[str, Any]] = []
    for structure in analysis["structures"]:
        mask_path = files_by_name.get(structure.filename)
        if mask_path is None:
            errors.append(f"Missing mask: {structure.filename}")
            continue
        try:
            annotation = annotation_record(
                structure,
                mask_path,
                records_by_number.get(structure.scan_number),
                dataset_root,
            )
        except (OSError, ValueError, nib.filebasedimages.ImageFileError) as error:
            errors.append(f"Could not read mask {mask_path.name}: {error}")
            continue
        if dimensions and tuple(annotation["mask_dimensions_px"]) != dimensions[0]:
            errors.append(
                f"Mask dimensions differ from image grid: {structure.filename}"
            )
        mask_spacing = annotation["mask_voxel_spacing_mm"]
        if reference_spacing is not None and (
            mask_spacing is None
            or not np.allclose(mask_spacing, reference_spacing, rtol=0.0, atol=1e-5)
        ):
            errors.append(
                f"Mask voxel spacing differs from image grid: {structure.filename}"
            )
        if structure.lesion_number is None:
            auxiliary_masks.append(annotation)
            continue
        lesion = lesions.setdefault(
            structure.lesion_number,
            {
                "lesion_id": structure.lesion_number,
                "scans": [],
            },
        )
        lesion["scans"].append(compact_annotation(annotation))

    for lesion in lesions.values():
        lesion["scans"].sort(key=lambda item: item["scan_number"])

    record: dict[str, Any] = {
        "patient_id": patient.patient_id,
        "voxel_spacing_mm": isotropic_spacing if spacing_consistent else None,
        "grid_dimensions_px": list(dimensions[0])
        if dimensions_consistent and dimensions
        else None,
        "radiation": radiation,
        "lesions": [lesions[key] for key in sorted(lesions)],
    }
    if auxiliary_masks:
        record["auxiliary_masks"] = [
            compact_annotation(annotation) for annotation in auxiliary_masks
        ]
    if errors:
        record["validation_errors"] = errors
    if warnings:
        record["warnings"] = warnings
    return record


def build_manifest(
    root: Path,
    registration_directory: str,
    *,
    show_progress: bool = False,
    workers: int = 1,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    patients = discover_patients(root, registration_directory)
    if workers == 1:
        records = [
            build_patient_record(patient, root)
            for patient in tqdm(
                patients,
                desc="Building training manifest",
                unit="patient",
                disable=not show_progress,
            )
        ]
    else:
        records_by_index: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(build_patient_record, patient, root): index
                for index, patient in enumerate(patients)
            }
            completed = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Building training manifest",
                unit="patient",
                disable=not show_progress,
            )
            for future in completed:
                records_by_index[futures[future]] = future.result()
        records = [records_by_index[index] for index in range(len(patients))]
    return {
        "schema_version": "2.0",
        "dataset_root": str(root.resolve()),
        "registration_directory": registration_directory,
        "path_resolution": (
            "dataset_root / patient_id / registration_directory / image_or_mask_filename"
        ),
        "bbox_convention": "XYZ voxel coordinates, zero-based, max_exclusive",
        "patients": records,
        "summary": {
            "patient_count": len(records),
            "valid_patient_count": sum(
                "validation_errors" not in record for record in records
            ),
            "invalid_patient_count": sum(
                "validation_errors" in record for record in records
            ),
            "lesion_count": sum(len(record["lesions"]) for record in records),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a patient/scan/lesion JSON manifest for model training."
    )
    parser.add_argument("root", type=Path, help="Directory containing patient folders")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training_manifest.json"),
        help="Output JSON path (default: ./training_manifest.json)",
    )
    parser.add_argument(
        "--registration-directory",
        default="_registered_using_binarized_MR",
        help=(
            "Registration subdirectory to use (default: _registered_using_binarized_MR)"
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a failure status if any patient fails validation",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the patient progress bar",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Patients to process concurrently (default: 2; use 1 for serial)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Input directory does not exist: {root}")
    manifest = build_manifest(
        root,
        args.registration_directory,
        show_progress=not args.no_progress,
        workers=args.workers,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"{json.dumps(manifest, indent=2, ensure_ascii=False)}\n", encoding="utf-8"
    )
    summary = manifest["summary"]
    print(
        f"Wrote {summary['patient_count']} patients and {summary['lesion_count']} lesions "
        f"to {output} ({summary['invalid_patient_count']} invalid patients)."
    )
    return 1 if args.strict and summary["invalid_patient_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
