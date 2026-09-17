# Longitudinal lesion classification

PyTorch Lightning pipeline for one prediction per MRI scan: **healthy, PD, PSP**.
Each dataset item is a complete lesion timeline from the root `manifest.json`.
MRI scans use the pretrained **huggingbrain/Dinov3d-Neuro** backbone; radiation
uses a trainable residual 3D CNN. A custom transformer packs all lesions in a
batch into one sequence with a block-diagonal causal attention mask.

## Run

```bash
uv sync

# Train, select the best validation checkpoint, then evaluate the test cohort.
uv run python -m ml \
  data.root=/mnt/projects/radiomics/raw_data/preprocessed \
  data.cache_dir=/path/to/crop-cache \
  trainer.accelerator=cuda trainer.precision=bf16-mixed \
  metadata.run_name=experiment-01

# Train without running test evaluation.
uv run python -m ml mode=fit checkpoint=null

# Resume, then test the best checkpoint.
uv run python -m ml checkpoint.fit=/path/to/last.ckpt

# Standalone evaluation with the same model and preprocessing configuration.
uv run python -m ml mode=test checkpoint=/path/to/checkpoint.ckpt
uv run python -m ml mode=validate checkpoint=/path/to/checkpoint.ckpt

# Fine-tune the MRI encoder with a lower learning rate.
uv run python -m ml model.freeze_mri=false \
  model.backbone_lr_multiplier=0.05 model.encoder_batch_size=1 \
  trainer.accelerator=cuda trainer.precision=bf16-mixed
```

All settings are in [`configs/default.yaml`](configs/default.yaml). The entry point follows
[RationAI/lsp-detr's bump branch](https://github.com/RationAI/lsp-detr/tree/bump):
`@autolog` creates the MLKit logger, Hydra instantiates the data/model/trainer,
and each configured mode calls the corresponding Trainer method. The default
`mode: [fit, test]` uses `checkpoint.fit: null` and `checkpoint.test: best`.
For one mode, supply a single checkpoint path or `null` instead of a mapping.

`data.root` defaults to the manifest's `dataset_root`. Paths resolve as
`root / patient_id / registration_directory / filename`. Run from the repository
root. Use `trainer.accelerator=cpu data.num_workers=0` on a CPU and `32-true`
precision where BF16 is unavailable. Offline training accepts
`model.encoder_checkpoint=/path/to/teacher_checkpoint.pth`. Fresh training loads
pretrained MRI weights in `on_fit_start`; checkpoint restoration is handled by
Lightning and does not download the MRI weights again.

The logger uses `MLFLOW_TRACKING_URI` when set and otherwise a local
`sqlite:///mlflow.db` store. Override `logger.tracking_uri` or
`metadata.experiment_name` as needed. MLKit logs configs, console output, metrics
and checkpoints automatically. Callbacks are declared under `trainer.callbacks`.

Resume and evaluation use the supplied experiment configuration; there is no
custom checkpoint inspection, configuration merging or override filtering.
For a run with nondefault settings, reuse its experiment configuration or the
logged `configs/config-resolved.yaml` artifact, for example:

```bash
uv run python -m ml --config-path /path/to/downloaded/configs \
  --config-name config-resolved mode=test checkpoint=/path/to/checkpoint.ckpt
```

Lightning restores the saved patient split. Reuse the same manifest, class map,
model architecture and physical preprocessing when resuming or evaluating.

## Data and chronology

Patient IDs are shuffled at runtime using `seed` (default 42). The first
`floor(0.7 * N)` patients train, the next `floor(0.1 * N)` validate, and the
remainder test. All lesions and scans from one patient stay together. Thus the
70/10/20 proportions apply to **patients**, with integer rounding; lesion and
scan counts vary. This prevents shared anatomy and dose volumes leaking across
sets. Splits are not repeatedly sampled until validation/test scores look good.

The current manifest has 92 patients, 172 lesions, and 584 MRI scans. Seed 42
produces 64/9/19 patients and 110/25/37 lesions. Dataset length is the number of
lesions in that split; DataLoader length is the usual number of lesion batches.
No scans are dropped, no timelines are truncated, and the final short batch is kept.

The user-specified label mapping is:

| Manifest label | Class ID | Output class |
| --- | --- | --- |
| MTS, KAVITA, SD, PR, healthy | 0 | healthy |
| PD | 1 | PD |
| PSP | 2 | PSP |

Here `healthy` is the requested combined class name. Custom mappings may use
`-100` for context-only scans; those remain visible in the timeline but are excluded
from loss/metrics.

MRI scans are sorted by `scan_number`. Every lesion contributes one radiation token:

```text
[predop, if available] -> plan -> radiation -> FU -> FU2 -> ...
```

Only MRI tokens receive labels and classification loss. A token attends to itself
and earlier tokens of the **same lesion**. In particular, plan/predop predictions
cannot see radiation, future MRIs, or any other lesion. Position indices restart
for every lesion; they represent ordered events, not elapsed days (the manifest
does not reliably contain acquisition dates). Learned modality embeddings identify
MRI and radiation tokens.

Seventeen lesions in the supplied manifest have no planning MRI annotation. They
are retained: radiation is inserted before the first follow-up, or after available
pretreatment scans if there are no follow-ups. Its crop uses the last available
pretreatment MRI bbox; when none exists it uses the first available MRI bbox.
A planning bbox is always preferred when present. This fallback never uses a later
follow-up to define the crop of an earlier prediction.

## Physical preprocessing

- Use the NIfTI affine as the source of physical geometry. Manifest bboxes are
  XYZ voxels with an exclusive upper bound.
- Convert the current scan's bbox voxel edges into physical coordinates, add
  `margin_mm` on each side, and define a RAS-aligned **1 mm isotropic** output grid.
- Minimum crop side is `crop_size` mm (96 by default). Expand to contain the bbox
  plus context and round each side up to a multiple of the MRI encoder patch size,
  16. Variable-size crops are encoded in groups of matching shapes. There is no
  geometric stretching to force a large lesion into a fixed tensor.
- MRI normalization uses whole-volume nonzero mean/std, computed in slabs and
  retained in a bounded per-worker cache. Padding uses normalized minimum intensity.
- Radiation is resampled directly from its own affine onto the planning/anchor
  crop grid. Dose magnitudes are preserved using a **fixed** `dose_scale`, default
  70 in the NIfTI's units. Set this appropriately if volumes use cGy or another
  unit; dose is never standardized independently per patient.
- NIfTI geometry is interpreted in millimeters, consistent with this manifest.
- Training applies shared random spatial flips to the lesion's MRI and radiation
  crops. Validation/test have no random augmentation. No future-scan union bbox,
  label mask, or label text is supplied as an input feature.

`data.cache_dir` enables an atomic, disk-backed deterministic crop cache, keyed
by source path/size/modification time, geometry, normalization version and dose
scale. Augmentation happens after loading cached crops. Prefer uncompressed `.nii`
for efficient memory-mapped reads. Keep the cache on fast local/shared storage;
changing source files without changing their size or timestamp requires clearing
that cache. Cache contents and prediction CSVs contain dataset-derived information.

## Training and outputs

Defaults freeze the 93.7M-parameter MRI backbone and train its projection, radiation
CNN, modality embeddings and a pre-normalized causal transformer. Fine-tuning is
supported with a separate backbone learning rate and activation checkpointing.
Encoder microbatches limit transient volume-encoder memory; they do not truncate
the temporal sequence. Fine-tuning still retains/recomputes activations across the
whole lesion batch, so reduce lesion batch size for long timelines.

Optimization uses AdamW, zero weight decay on biases/normalizations/tokens, linear
warmup followed by cosine decay, gradient clipping and gradient accumulation.
Inverse-frequency class weights are calculated **only from training scans**.
Validation/test loss is unweighted cross-entropy; model selection and early stopping
use macro-F1 over the three fixed classes. `mode=fit checkpoint=null` defers test access until a separate evaluation.

The MLflow run contains:

- `configs/`: Hydra configuration, source config and resolved config, saved by
  MLKit's `autolog` decorator.
- `console.log`, training/validation/test metrics and learning rates.
- `checkpoints/`: best and last Lightning checkpoints, uploaded by MLKit's logger.
  Model/optimizer/scheduler state and the data module's patient split follow the
  standard Lightning checkpoint lifecycle.
- `evaluation/val_metrics.json` / `evaluation/test_metrics.json`: accuracy,
  macro-F1, balanced accuracy, unweighted loss, confusion matrix, class support,
  precision, recall, F1, one-vs-rest AUROC and average precision.
- `evaluation/val_predictions.csv` / `evaluation/test_predictions.csv`: patient,
  lesion, scan identity, original label, target ID, prediction and probabilities.

Validation reports are overwritten each epoch; test reports use the selected
best model. Undefined AUROC/AP (missing positive or negative examples) are JSON
`null`. Macro-F1 includes all three classes with zero for absent classes; balanced
accuracy averages recall over supported classes. Evaluation uses a custom
`torchmetrics.Metric` with registered tensor states and a `MetricCollection` of
standard classification metrics. TorchMetrics synchronizes logits, targets and
stable scan IDs across ranks; duplicate sampler padding is removed before
computation. Report metadata is gathered separately by an export callback. Lightning
DDP can be configured through `+trainer.strategy=ddp trainer.devices=2`; CPU and
multi-worker execution were exercised locally, but CUDA/DDP were not available
for verification. Deterministic execution is requested, but bitwise equivalence
across hardware, precision modes, or interrupted runs is not promised.

## Encoder provenance and verification limits

The [model card](https://huggingface.co/huggingbrain/Dinov3d-Neuro) supplies raw
teacher checkpoints, not a Transformers `AutoModel`. This pipeline pins revision
`8f1355d6ba4159c1d6fdc2e1452212c0ab029481` and
`eval/training_137499/teacher_checkpoint.pth`. It strictly loads **every backbone
tensor**, dropping only the self-supervised DINO/iBOT projection heads. There is
no random-weight fallback. Weights are CC-BY-NC-SA-4.0, separately from this
repository's license.

The linked `nrdg/dinov3d` / `asagilmore/dinov3d` repositories returned 404 during
implementation. The native backbone here implements the published 792-wide,
12-layer, 12-head, single-channel 3D ViT with three-axis axial RoPE, using the
published tensor schema and DINOv3 attention formulation. **Strict tensor loading
and successful inference do not establish numerical equivalence with the original
unavailable 3D implementation.** That parity check, including orientation and RoPE
conventions, remains necessary when its source/reference outputs become available.

Runtime verification used `uv`, temporary synthetic NIfTI files with non-isotropic
and flipped affines, and the actual downloaded pretrained weights. Training,
validation, best-checkpoint testing, standalone evaluation, resumed training,
multiple loader workers and deterministic crop caching were exercised. The
TorchMetrics refactor was additionally exercised with two local Gloo processes,
including duplicate scan IDs, ignored labels, missing classes and metric reset. No test
suite or synthetic dataset is added to the repository. Real-image registration,
actual dose units, CUDA behavior and predictive performance remain unverified
because the clinical volumes are not available locally.

Manifest-building utilities are in `scripts/`. The training entry point uses
MLKit autolog and Trainer. The local MLflow backend works without a remote service.


## Code organization and checks

The training structure follows the separation used in
[RationAI/lsp-detr](https://github.com/RationAI/lsp-detr/tree/bump): metric objects
with `update` / `compute` / `reset`, and small Lightning hooks. Parameter and return
annotations document inputs and outputs; batches and configuration use ordinary
dictionaries and Lightning hyperparameters. The pipeline assumes valid inputs
and uses no custom validation or exception handling.

- `ml/data/`: manifest loading, physical crops, packed batches and patient splits.
- `ml/metrics/scan_classification.py`: TorchMetrics state, distributed scan
  deduplication, built-in classification metrics and mean cross-entropy.
- `ml/callbacks/evaluation_writer.py`: prediction/report export. Numerical metrics
  are computed independently of CSV records and Python metadata.
- `ml/evaluation.py`: report formatting and serialization.
- `ml/meta_arch.py`: forward/training orchestration, metric lifecycle and optimizer.

TorchMetrics state is separate for validation and test, reset after every epoch,
and excluded from saved model weights. Previously generated model checkpoints
remain compatible when instantiated with their original configuration. Report
fields and class semantics are unchanged; reports are stored as MLflow artifacts.

```bash
uv run ruff check ml
uv run ruff format --check ml
```
