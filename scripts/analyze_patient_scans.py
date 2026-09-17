#!/usr/bin/env python3
"""Summarize MRI and RTSTRUCT filenames stored in one folder per patient.

The parser follows the project naming glossary.  It reads filenames only; NIfTI
voxel data are never loaded.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING

import nibabel as nib


if TYPE_CHECKING:
    from collections.abc import Iterable


NIFTI_SUFFIXES = (".nii", ".nii.gz")
EXCLUDED_PATIENT_MARKERS = ("old_incomplete_annotations",)
SCAN_RE = re.compile(r"(?<![A-Z0-9])MR[ _-]*0*(\d+)(?!\d)", re.IGNORECASE)
STRUCTURE_RE = re.compile(
    r"(?<![A-Z0-9])(?:A?M|MTS)\s*0*(\d+)\s+"
    r"(?:(PD|PR|SD|PSP|N)\s+)?"
    r"(KAVITA|MTS)(?:\s+(PD|PR|SD|PSP|N))?\s*0*(\d+)(?!\d)",
    re.IGNORECASE,
)
FOLLOW_UP_RE = re.compile(r"(?<![A-Z0-9])FU\s*0*(\d*)(?![A-Z0-9])", re.IGNORECASE)
PREOP_RE = re.compile(r"(?<![A-Z0-9])PREDOP(?![A-Z0-9])", re.IGNORECASE)
PLAN_RE = re.compile(r"(?<![A-Z0-9])PLAN(?![A-Z0-9])", re.IGNORECASE)
RTSTRUCT_RE = re.compile(r"(?<![A-Z0-9])RTSTRUCT(?![A-Z0-9])", re.IGNORECASE)
RTSTRUCT_LABEL_RE = re.compile(
    r"(?<![A-Z0-9])RTSTRUCT\s+LABEL(?![A-Z0-9])", re.IGNORECASE
)
RTDOSE_RE = re.compile(r"(?<![A-Z0-9])RTDOSE(?![A-Z0-9])", re.IGNORECASE)


@dataclass(frozen=True)
class Structure:
    patient_id: str
    filename: str
    scan_number: int
    lesion_number: int | None
    structure_type: str
    response_label: str
    drawing_mri_number: int | None


@dataclass(frozen=True)
class Scan:
    patient_id: str
    image_type: str
    scan_number: int | None
    scan_label: str
    is_preop: bool
    is_plan: bool
    is_follow_up: bool
    follow_up_number: int | None
    file_count: int
    filenames: str
    image_filename: str
    dimension_x_px: int | None
    dimension_y_px: int | None
    dimension_z_px: int | None
    spacing_x: float | None
    spacing_y: float | None
    spacing_z: float | None
    spacing_unit: str
    spacing_x_mm_per_px: float | None
    spacing_y_mm_per_px: float | None
    spacing_z_mm_per_px: float | None
    field_of_view_x_mm: float | None
    field_of_view_y_mm: float | None
    field_of_view_z_mm: float | None
    geometry_status: str


@dataclass(frozen=True)
class PatientFolder:
    patient_id: str
    source_directory: str
    files: tuple[Path, ...]


@dataclass(frozen=True)
class ImageGeometry:
    dimensions_px: tuple[int, int, int]
    spacing: tuple[float, float, float]
    spatial_unit: str
    spacing_mm_per_px: tuple[float, float, float] | None


def is_nifti(path: Path) -> bool:
    return path.name.lower().endswith(NIFTI_SUFFIXES)


def normalized_name(path: Path) -> str:
    """Return a separator-normalized stem while keeping meaningful spaces."""
    name = path.name
    if name.lower().endswith(".nii.gz"):
        name = name[:-7]
    elif name.lower().endswith(".nii"):
        name = name[:-4]
    return re.sub(r"[_-]+", " ", name).strip()


def read_nifti_geometry(path: Path) -> ImageGeometry:
    """Read 3-D dimensions and voxel spacing via nibabel's lazy image proxy."""
    image = nib.load(path, mmap=False)
    dimensions_xyz = tuple(
        int(image.shape[index]) if index < len(image.shape) else 1 for index in range(3)
    )
    zooms = image.header.get_zooms()
    spacing_xyz = tuple(
        abs(float(zooms[index])) if index < len(zooms) else 1.0 for index in range(3)
    )
    if any(value <= 0 for value in dimensions_xyz):
        raise ValueError(f"invalid image dimensions: {dimensions_xyz}")
    if any(value <= 0 for value in spacing_xyz):
        raise ValueError(f"invalid voxel spacing: {spacing_xyz}")

    spatial_unit, _ = image.header.get_xyzt_units()
    units = {
        "unknown": ("unknown/px", None),
        "meter": ("m/px", 1000.0),
        "mm": ("mm/px", 1.0),
        "micron": ("um/px", 0.001),
    }
    output_unit, millimetre_factor = units.get(
        spatial_unit, (f"{spatial_unit or 'unknown'}/px", None)
    )
    spacing_mm = (
        tuple(value * millimetre_factor for value in spacing_xyz)
        if millimetre_factor is not None
        else None
    )
    return ImageGeometry(
        dimensions_px=dimensions_xyz,
        spacing=spacing_xyz,
        spatial_unit=output_unit,
        spacing_mm_per_px=spacing_mm,
    )


def rounded(value: float) -> float:
    return round(value, 6)


def nifti_files(directory: Path, *, recursive: bool = True) -> tuple[Path, ...]:
    candidates = directory.rglob("*") if recursive else directory.iterdir()
    return tuple(
        sorted(path for path in candidates if path.is_file() and is_nifti(path))
    )


def discover_patients(
    root: Path, registration_directory: str = "_registered_using_binarized_MR"
) -> list[PatientFolder]:
    """Find one selected registration directory inside each patient folder.

    The real dataset contains two parallel registration outputs with duplicate
    filenames. Only ``registration_directory`` is read, so those files are not
    double-counted. A root that is itself a patient folder or a legacy flat
    patient directory is also supported.
    """
    if any(marker in root.name.casefold() for marker in EXCLUDED_PATIENT_MARKERS):
        return []

    selected_root = root / registration_directory
    if selected_root.is_dir():
        files = nifti_files(selected_root)
        return (
            [PatientFolder(root.name, registration_directory, files)] if files else []
        )

    root_files = nifti_files(root, recursive=False)
    patients: list[PatientFolder] = []
    if root_files:
        patients.append(PatientFolder(root.name, ".", root_files))

    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        if any(
            marker in directory.name.casefold() for marker in EXCLUDED_PATIENT_MARKERS
        ):
            continue
        selected = directory / registration_directory
        if selected.is_dir():
            files = nifti_files(selected)
            source_directory = registration_directory
        else:
            # Backwards compatibility for patient folders containing files directly.
            files = nifti_files(directory, recursive=False)
            source_directory = "."
        if files:
            patients.append(PatientFolder(directory.name, source_directory, files))
    return patients


def scan_label(name: str) -> tuple[str, int | None]:
    if PREOP_RE.search(name):
        return "predop", None
    if PLAN_RE.search(name):
        return "plan", None
    match = FOLLOW_UP_RE.search(name)
    if match:
        number = int(match.group(1)) if match.group(1) else 1
        return ("FU" if number == 1 else f"FU{number}"), number
    return "unlabelled", None


def analyze_patient(
    patient_id: str, files: Iterable[Path], source_directory: str = "."
) -> dict[str, object]:
    scan_files: dict[int, list[tuple[Path, str]]] = defaultdict(list)
    dose_files: list[Path] = []
    structures: list[Structure] = []
    warnings: list[str] = []
    ignored_files: list[str] = []

    for path in files:
        name = normalized_name(path)
        scan_match = SCAN_RE.search(name)
        if not scan_match:
            if RTDOSE_RE.search(name):
                dose_files.append(path)
            else:
                ignored_files.append(path.name)
            continue

        scan_number = int(scan_match.group(1))
        scan_files[scan_number].append((path, name))

        if RTSTRUCT_RE.search(name):
            structure_match = STRUCTURE_RE.search(name)
            if not structure_match:
                if RTSTRUCT_LABEL_RE.search(name):
                    structures.append(
                        Structure(
                            patient_id=patient_id,
                            filename=path.name,
                            scan_number=scan_number,
                            lesion_number=None,
                            structure_type="LABEL",
                            response_label="NONE",
                            drawing_mri_number=None,
                        )
                    )
                    continue
                warnings.append(f"Could not parse RTSTRUCT filename: {path.name}")
                continue
            (
                lesion_number,
                response_before_type,
                structure_type,
                response_after_type,
                drawing_mri,
            ) = structure_match.groups()
            response = response_before_type or response_after_type
            structures.append(
                Structure(
                    patient_id=patient_id,
                    filename=path.name,
                    scan_number=scan_number,
                    lesion_number=int(lesion_number),
                    structure_type=structure_type.upper(),
                    response_label=(response or "none").upper(),
                    drawing_mri_number=int(drawing_mri),
                )
            )

    scans: list[Scan] = []
    for number, entries in sorted(scan_files.items()):
        labels = [scan_label(name) for _, name in entries]
        labelled = [
            (label, fu_number) for label, fu_number in labels if label != "unlabelled"
        ]
        distinct_labels = sorted({label for label, _ in labelled})
        if len(distinct_labels) > 1:
            warnings.append(
                f"MR {number:02d} has conflicting labels: {', '.join(distinct_labels)}"
            )
        label = labelled[0][0] if labelled else "unlabelled"
        image_paths = [path for path, name in entries if not RTSTRUCT_RE.search(name)]
        image_path = image_paths[0] if image_paths else None
        geometry: ImageGeometry | None = None
        if len(image_paths) > 1:
            warnings.append(
                f"MR {number:02d} has multiple primary image files; using {image_paths[0].name}"
            )
        if image_path is None:
            geometry_status = "missing_primary_image"
            warnings.append(f"MR {number:02d} has no primary image file")
        elif not image_path.is_file():
            geometry_status = "file_not_found"
        else:
            try:
                geometry = read_nifti_geometry(image_path)
                geometry_status = "ok"
            except (OSError, ValueError, nib.filebasedimages.ImageFileError) as error:
                geometry_status = f"unreadable_header: {error}"
                warnings.append(
                    f"Could not read geometry from {image_path.name}: {error}"
                )

        dimensions = geometry.dimensions_px if geometry else (None, None, None)
        native_spacing = geometry.spacing if geometry else (None, None, None)
        spacing_mm = (
            geometry.spacing_mm_per_px
            if geometry and geometry.spacing_mm_per_px
            else (None, None, None)
        )
        field_of_view_mm = tuple(
            rounded(dimension * voxel_spacing)
            if dimension is not None and voxel_spacing is not None
            else None
            for dimension, voxel_spacing in zip(dimensions, spacing_mm, strict=True)
        )
        scans.append(
            Scan(
                patient_id=patient_id,
                image_type="MRI",
                scan_number=number,
                scan_label=label,
                is_preop=any(item[0] == "predop" for item in labels),
                is_plan=any(item[0] == "plan" for item in labels),
                is_follow_up=any(item[0].startswith("FU") for item in labels),
                follow_up_number=next(
                    (item[1] for item in labels if item[0].startswith("FU")), None
                ),
                file_count=len(entries),
                filenames=" | ".join(path.name for path, _ in entries),
                image_filename=image_path.name if image_path else "",
                dimension_x_px=dimensions[0],
                dimension_y_px=dimensions[1],
                dimension_z_px=dimensions[2],
                spacing_x=rounded(native_spacing[0]) if native_spacing[0] else None,
                spacing_y=rounded(native_spacing[1]) if native_spacing[1] else None,
                spacing_z=rounded(native_spacing[2]) if native_spacing[2] else None,
                spacing_unit=geometry.spatial_unit if geometry else "",
                spacing_x_mm_per_px=rounded(spacing_mm[0]) if spacing_mm[0] else None,
                spacing_y_mm_per_px=rounded(spacing_mm[1]) if spacing_mm[1] else None,
                spacing_z_mm_per_px=rounded(spacing_mm[2]) if spacing_mm[2] else None,
                field_of_view_x_mm=field_of_view_mm[0],
                field_of_view_y_mm=field_of_view_mm[1],
                field_of_view_z_mm=field_of_view_mm[2],
                geometry_status=geometry_status,
            )
        )

    for dose_path in dose_files:
        geometry: ImageGeometry | None = None
        if not dose_path.is_file():
            geometry_status = "file_not_found"
        else:
            try:
                geometry = read_nifti_geometry(dose_path)
                geometry_status = "ok"
            except (OSError, ValueError, nib.filebasedimages.ImageFileError) as error:
                geometry_status = f"unreadable_header: {error}"
                warnings.append(
                    f"Could not read geometry from {dose_path.name}: {error}"
                )

        dimensions = geometry.dimensions_px if geometry else (None, None, None)
        native_spacing = geometry.spacing if geometry else (None, None, None)
        spacing_mm = (
            geometry.spacing_mm_per_px
            if geometry and geometry.spacing_mm_per_px
            else (None, None, None)
        )
        field_of_view_mm = tuple(
            rounded(dimension * voxel_spacing)
            if dimension is not None and voxel_spacing is not None
            else None
            for dimension, voxel_spacing in zip(dimensions, spacing_mm, strict=True)
        )
        scans.append(
            Scan(
                patient_id=patient_id,
                image_type="RTDOSE",
                scan_number=None,
                scan_label="RTDOSE",
                is_preop=False,
                is_plan=False,
                is_follow_up=False,
                follow_up_number=None,
                file_count=1,
                filenames=dose_path.name,
                image_filename=dose_path.name,
                dimension_x_px=dimensions[0],
                dimension_y_px=dimensions[1],
                dimension_z_px=dimensions[2],
                spacing_x=rounded(native_spacing[0]) if native_spacing[0] else None,
                spacing_y=rounded(native_spacing[1]) if native_spacing[1] else None,
                spacing_z=rounded(native_spacing[2]) if native_spacing[2] else None,
                spacing_unit=geometry.spatial_unit if geometry else "",
                spacing_x_mm_per_px=rounded(spacing_mm[0]) if spacing_mm[0] else None,
                spacing_y_mm_per_px=rounded(spacing_mm[1]) if spacing_mm[1] else None,
                spacing_z_mm_per_px=rounded(spacing_mm[2]) if spacing_mm[2] else None,
                field_of_view_x_mm=field_of_view_mm[0],
                field_of_view_y_mm=field_of_view_mm[1],
                field_of_view_z_mm=field_of_view_mm[2],
                geometry_status=geometry_status,
            )
        )

    valid_geometry_scans = [scan for scan in scans if scan.geometry_status == "ok"]
    dimension_signatures = {
        (scan.dimension_x_px, scan.dimension_y_px, scan.dimension_z_px)
        for scan in valid_geometry_scans
    }
    spacing_signatures = {
        (
            scan.spacing_x_mm_per_px,
            scan.spacing_y_mm_per_px,
            scan.spacing_z_mm_per_px,
            scan.spacing_unit if scan.spacing_x_mm_per_px is None else "mm/px",
        )
        for scan in valid_geometry_scans
    }
    if len(valid_geometry_scans) != len(scans):
        resolution_consistency = "unknown_missing_geometry"
        dimensions_consistent: bool | None = None
        spacing_consistent: bool | None = None
    elif len(scans) <= 1:
        resolution_consistency = "single_scan"
        dimensions_consistent = True
        spacing_consistent = True
    else:
        dimensions_consistent = len(dimension_signatures) == 1
        spacing_consistent = len(spacing_signatures) == 1
        resolution_consistency = (
            "same" if dimensions_consistent and spacing_consistent else "different"
        )
        if resolution_consistency == "different":
            variants = "; ".join(
                f"{f'MR {scan.scan_number:02d}' if scan.image_type == 'MRI' else 'RTDOSE'}="
                f"{scan.dimension_x_px}x{scan.dimension_y_px}x{scan.dimension_z_px} px, "
                f"{scan.spacing_x}x{scan.spacing_y}x{scan.spacing_z} {scan.spacing_unit}"
                for scan in valid_geometry_scans
            )
            warnings.append(
                f"Image resolution differs across MRI/RTDOSE volumes: {variants}"
            )

    lesion_ids = sorted(
        {
            structure.lesion_number
            for structure in structures
            if structure.lesion_number is not None
        }
    )
    response_occurrences = Counter(
        structure.response_label
        for structure in structures
        if structure.response_label != "NONE"
    )
    response_lesions = {
        label: len({s.lesion_number for s in structures if s.response_label == label})
        for label in ("PD", "N", "PSP")
    }
    undocumented_label_lesions = {
        label: len({s.lesion_number for s in structures if s.response_label == label})
        for label in ("PR", "SD")
    }
    patient_row = {
        "patient_id": patient_id,
        "source_directory": source_directory,
        # Per the glossary, predop is present only when an operation took place.
        "operation_done": any(scan.is_preop for scan in scans),
        "image_volume_count": len(scans),
        "mri_scan_count": sum(scan.image_type == "MRI" for scan in scans),
        "rtdose_count": sum(scan.image_type == "RTDOSE" for scan in scans),
        "preop_scan_count": sum(scan.is_preop for scan in scans),
        "plan_scan_count": sum(scan.is_plan for scan in scans),
        "post_radiation_scan_count": sum(scan.is_follow_up for scan in scans),
        "geometry_available_scan_count": sum(
            scan.geometry_status == "ok" for scan in scans
        ),
        "geometry_issue_scan_count": sum(
            scan.geometry_status != "ok" for scan in scans
        ),
        "scan_resolution_consistency": resolution_consistency,
        "scan_dimensions_consistent": dimensions_consistent,
        "scan_voxel_spacing_consistent": spacing_consistent,
        "follow_up_labels": ";".join(
            scan.scan_label for scan in scans if scan.is_follow_up
        ),
        "lesion_count": len(lesion_ids),
        "lesion_ids": ";".join(map(str, lesion_ids)),
        "rtstruct_count": len(structures),
        "label_structure_count": sum(s.structure_type == "LABEL" for s in structures),
        "cavity_structure_count": sum(s.structure_type == "KAVITA" for s in structures),
        "metastasis_structure_count": sum(
            s.structure_type == "MTS" for s in structures
        ),
        "pd_structure_count": response_occurrences["PD"],
        "necrosis_structure_count": response_occurrences["N"],
        "pseudoprogression_structure_count": response_occurrences["PSP"],
        "pd_lesion_count": response_lesions["PD"],
        "necrosis_lesion_count": response_lesions["N"],
        "pseudoprogression_lesion_count": response_lesions["PSP"],
        "undocumented_pr_structure_count": response_occurrences["PR"],
        "undocumented_sd_structure_count": response_occurrences["SD"],
        "undocumented_pr_lesion_count": undocumented_label_lesions["PR"],
        "undocumented_sd_lesion_count": undocumented_label_lesions["SD"],
        "warning_count": len(warnings),
    }
    return {
        "patient": patient_row,
        "scans": scans,
        "structures": structures,
        "warnings": warnings,
        "ignored_files": ignored_files,
    }


def describe(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"total": 0, "mean": 0.0, "median": 0.0, "minimum": 0, "maximum": 0}
    return {
        "total": sum(values),
        "mean": round(statistics.mean(values), 3),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def build_summary(results: list[dict[str, object]]) -> dict[str, object]:
    patients = [result["patient"] for result in results]
    assert all(isinstance(patient, dict) for patient in patients)
    rows: list[dict[str, object]] = patients  # type: ignore[assignment]
    scans: list[Scan] = [
        scan
        for result in results
        for scan in result["scans"]  # type: ignore[union-attr]
    ]
    resolution_counts = Counter(
        f"{scan.dimension_x_px}x{scan.dimension_y_px}x{scan.dimension_z_px}"
        for scan in scans
        if scan.geometry_status == "ok"
    )
    spacing_counts = Counter(
        f"{scan.spacing_x_mm_per_px:g}x{scan.spacing_y_mm_per_px:g}x"
        f"{scan.spacing_z_mm_per_px:g} mm/px"
        for scan in scans
        if scan.geometry_status == "ok" and scan.spacing_x_mm_per_px is not None
    )
    return {
        "patient_count": len(rows),
        "image_volume_count": sum(int(row["image_volume_count"]) for row in rows),
        "mri_scan_count": sum(int(row["mri_scan_count"]) for row in rows),
        "rtdose_volume_count": sum(int(row["rtdose_count"]) for row in rows),
        "operated_patient_count": sum(bool(row["operation_done"]) for row in rows),
        "non_operated_or_not_recorded_patient_count": sum(
            not bool(row["operation_done"]) for row in rows
        ),
        "patients_with_follow_up_count": sum(
            int(row["post_radiation_scan_count"]) > 0 for row in rows
        ),
        "lesions_per_patient": describe([int(row["lesion_count"]) for row in rows]),
        "mri_scans_per_patient": describe([int(row["mri_scan_count"]) for row in rows]),
        "post_radiation_scans_per_patient": describe(
            [int(row["post_radiation_scan_count"]) for row in rows]
        ),
        "response_structure_totals": {
            "PD": sum(int(row["pd_structure_count"]) for row in rows),
            "N": sum(int(row["necrosis_structure_count"]) for row in rows),
            "PSP": sum(int(row["pseudoprogression_structure_count"]) for row in rows),
        },
        "undocumented_annotation_label_totals": {
            "PR": sum(int(row["undocumented_pr_structure_count"]) for row in rows),
            "SD": sum(int(row["undocumented_sd_structure_count"]) for row in rows),
        },
        "scan_geometry": {
            "available_count": sum(scan.geometry_status == "ok" for scan in scans),
            "issue_count": sum(scan.geometry_status != "ok" for scan in scans),
            "resolution_px_counts": dict(sorted(resolution_counts.items())),
            "voxel_spacing_mm_per_px_counts": dict(sorted(spacing_counts.items())),
            "patients_with_same_resolution_count": sum(
                row["scan_resolution_consistency"] == "same" for row in rows
            ),
            "patients_with_different_resolution_count": sum(
                row["scan_resolution_consistency"] == "different" for row in rows
            ),
            "single_scan_patient_count": sum(
                row["scan_resolution_consistency"] == "single_scan" for row in rows
            ),
            "patients_with_unverified_resolution_count": sum(
                row["scan_resolution_consistency"] == "unknown_missing_geometry"
                for row in rows
            ),
        },
        "warning_count": sum(int(row["warning_count"]) for row in rows),
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate patient, scan, lesion, and response statistics from NIfTI filenames."
    )
    parser.add_argument(
        "root", type=Path, help="Directory containing one subfolder per patient"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scan_statistics"),
        help="Output directory (default: ./scan_statistics)",
    )
    parser.add_argument(
        "--registration-directory",
        default="_registered_using_binarized_MR",
        help=(
            "Registration subdirectory to analyze inside each patient folder "
            "(default: _registered_using_binarized_MR)"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(
            f"Input directory does not exist or is not a directory: {root}"
        )

    discovered = discover_patients(root, args.registration_directory)
    if not discovered:
        raise SystemExit(
            f"No .nii or .nii.gz files found in '{args.registration_directory}' "
            f"patient directories below: {root}"
        )

    results = [
        analyze_patient(patient.patient_id, patient.files, patient.source_directory)
        for patient in discovered
    ]
    patient_rows = [result["patient"] for result in results]
    scan_rows = [asdict(scan) for result in results for scan in result["scans"]]
    structure_rows = [
        asdict(item) for result in results for item in result["structures"]
    ]
    warnings = {
        str(result["patient"]["patient_id"]): result["warnings"]
        for result in results
        if result["warnings"]
    }
    ignored = {
        str(result["patient"]["patient_id"]): result["ignored_files"]
        for result in results
        if result["ignored_files"]
    }
    summary = build_summary(results)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "patients.csv", patient_rows, list(patient_rows[0]))
    write_csv(
        output_dir / "scans.csv",
        scan_rows,
        [field.name for field in fields(Scan)],
    )
    write_csv(
        output_dir / "structures.csv",
        structure_rows,
        list(asdict(Structure("", "", 0, 0, "", "", 0))),
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "input_root": str(root),
                "summary": summary,
                "warnings": warnings,
                "ignored_files": ignored,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nDetailed results written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
