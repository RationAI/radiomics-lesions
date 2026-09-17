"""Formatting and serialization only; numerical evaluation lives in ml.metrics."""

import csv
import json
import math
from pathlib import Path

from torch import Tensor
from torch.nn import functional as F

from ml.data.dataset import CLASS_NAMES


def prediction_rows(output: dict, metadata: list[dict]) -> list[dict]:
    logits, targets = output["logits"].float().cpu(), output["targets"].cpu()
    probabilities = logits.softmax(-1)
    losses = F.cross_entropy(logits, targets, ignore_index=-100, reduction="none")
    return [
        {
            **meta,
            "target": int(target),
            "prediction": CLASS_NAMES[int(probability.argmax())],
            "loss": float(loss),
            "p_healthy": float(probability[0]),
            "p_PD": float(probability[1]),
            "p_PSP": float(probability[2]),
        }
        for meta, target, probability, loss in zip(
            metadata, targets, probabilities, losses, strict=True
        )
    ]


def optional_float(value: Tensor) -> float | None:
    scalar = float(value)
    return scalar if math.isfinite(scalar) else None


def format_report(values: dict[str, Tensor], rows: list[dict]) -> dict:
    labeled = [row for row in rows if row["target"] >= 0]
    per_class = {
        name: {
            "support": int(values["support"][index]),
            "precision": float(values["precision"][index]),
            "recall": float(values["recall"][index]),
            "f1": float(values["f1"][index]),
            "auroc": optional_float(values["auroc"][index]),
            "average_precision": optional_float(values["average_precision"][index]),
        }
        for index, name in enumerate(CLASS_NAMES)
    }
    return {
        "scans": int(values["scans"]),
        "context_only_scans": int(values["context_only_scans"]),
        "patients": len({r["patient_id"] for r in labeled}),
        "lesions": len({(r["patient_id"], r["lesion_id"]) for r in labeled}),
        "accuracy": float(values["accuracy"]),
        "macro_f1": float(values["macro_f1"]),
        "balanced_accuracy": float(values["balanced_accuracy"]),
        "loss": float(values["loss"]),
        "confusion_matrix": values["confusion_matrix"].tolist(),
        "class_order": list(CLASS_NAMES),
        "per_class": per_class,
    }


def write_report(
    directory: str | Path, stage: str, rows: list[dict], report: dict
) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stage}_metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    with (directory / f"{stage}_predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
