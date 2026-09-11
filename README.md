# River Garbage Detection — Official Paper Companion

Reference models, network-configuration files, training logs, released
checkpoints and the full experimental protocol for the paper on
**TEA8-based oriented perception of river garbage**
*(working title — replace with the camera-ready title)*.

This repository is the **artifact / supplementary material** that accompanies
the paper. Every table and figure in the paper can be regenerated from the
files released here:

- network-architecture YAMLs for every variant and ablation
- per-epoch training / validation metrics (`results.csv`) for every run
- complete training and validation logs
- released `best.pt` checkpoints
- learning curves, PR / P / R / F1 curves, confusion matrices, label statistics
- the **experimental protocol** ([`PROTOCOL.md`](PROTOCOL.md)) and the
  **dataset protocol** ([`DATASETS.md`](DATASETS.md))

> Dataset images are **not** redistributed in this repository. See
> [`DATASETS.md`](DATASETS.md) for provenance and the expected directory layout.

---

## Contents

- [1. Repository layout](#1-repository-layout)
- [2. Datasets](#2-datasets)
- [3. Models and configuration files](#3-models-and-configuration-files)
- [4. Results](#4-results)
  - [4.1 River detection — FloW-Img](#41-river-detection--flow-img)
  - [4.2 Multi-class detection — HELVLAJI](#42-multi-class-detection--helvlaji)
  - [4.3 River segmentation — fenGeHe](#43-river-segmentation--fengehe)
- [5. Methods: TEA8, GLE and WTG](#5-methods-tea8-gle-and-wtg)
- [6. Training logs](#6-training-logs)
- [7. Reproducing the paper](#7-reproducing-the-paper)
- [8. Environment](#8-environment)
- [9. Citation](#9-citation)
- [10. License](#10-license)

---

## 1. Repository layout

```
.
├── README.md                     # this file
├── PROTOCOL.md                   # experimental protocol (reproducibility)
├── DATASETS.md                   # dataset cards and data.yaml specification
├── CITATION.cff                  # citation metadata
├── LICENSE                       # AGPL-3.0 (GNU Affero GPL v3) license text
├── datasets/                     # ready-to-use data.yaml templates
│   ├── flow-img-yolo.yaml
│   ├── helvlaji.yaml
│   └── fenGeHe.yaml
│
├── yolo11s.yaml                  # YOLO11s detection baseline   (FloW-Img, nc=1)
├── yolo11s-flow-glare.yaml       # YOLO11s + GLE                (FloW-Img, nc=1)
├── yolo11s-tea8.yaml             # YOLO11s + TEA8 (backbone + P4 + P3)
├── yolo11s-tea8-backbone.yaml    # YOLO11s + TEA8 (backbone only)    -- ablation
├── yolo11s-tea8-neck.yaml        # YOLO11s + TEA8 (neck P4 + P3)     -- ablation
├── yolo11s-helvlaji.yaml         # YOLO11s detection baseline   (HELVLAJI, nc=9)
├── yolo11s-helvlaji-glare.yaml   # YOLO11s + GLE                (HELVLAJI)
├── yolo11s-helvlaji-tea8-glare.yaml  # YOLO11s + GLE + TEA8     (HELVLAJI)
├── seg-yolo11s-baseline.yaml     # YOLO11s-seg baseline         (fenGeHe, nc=1)
├── seg-yolo11s-wtg.yaml          # YOLO11s-seg + WTG            (fenGeHe)
│
├── r2/                           # reference-comparison configs
│   ├── yolov8s-flow.yaml
│   ├── yolov8s-helv.yaml
│   └── yolov8s-seg-fengehe.yaml
│
├── <experiment>/                 # one directory per trained run (see §4)
│   ├── args.yaml                 # exact Ultralytics training arguments
│   ├── results.csv               # per-epoch train / val metrics
│   ├── results.png, P_curve.png, R_curve.png, PR_curve.png, F1_curve.png
│   ├── confusion_matrix.png, confusion_matrix_normalized.png
│   ├── train_batch*.jpg, val_batch*_labels.jpg, val_batch*_pred.jpg
│   ├── labels.jpg, labels_correlogram.jpg
│   └── weights/best.pt           # released checkpoint
│
└── logs/                         # console logs mirrored from each run
```

Every `<experiment>/` directory is self-contained: the YAML graph it was built
from, its exact training arguments, its metrics, its figures and its weights.

---

## 2. Datasets

| Dataset | Task | Classes | Train / Val / Test | Used by |
|---------|------|---------|--------------------|---------|
| **FloW-Img** | detection of floating river garbage | 1 (`bottle`) | 1400 / 400 / 200 | `yolo11s*.yaml` |
| **HELVLAJI** | multi-class garbage detection | 9 | as released | `yolo11s-helvlaji*.yaml` |
| **fenGeHe** | river-surface instance segmentation | 1 | 682 val images | `seg-yolo11s*.yaml` |

Full dataset cards, provenance, the required directory layout and ready-to-use
`data.yaml` templates are in [`DATASETS.md`](DATASETS.md) and
[`datasets/`](datasets/).

---

## 3. Models and configuration files

All `*.yaml` files are Ultralytics network-architecture definitions; nodes are
`[from, repeats, module, args]`. The model scale is selected by the `s` token in
the file name (`yolo11s*.yaml`, `seg-yolo11s*.yaml`, `yolov8s*.yaml`).

### 3.1 Proposed modules

| Module | Meaning | Where it is inserted | YAML keyword |
|--------|---------|----------------------|--------------|
| **TEA8** | 8-direction Sobel target-edge attention | backbone (P5) + neck (P4, P3) | `TEA8` |
| **GLE** | glare-enhancement augmentation (`p = 0.25`) | training-time image augmentation only (no graph change) | — |
| **WTG** | water texture gate | segmentation neck (P4, P3) | `WTG` |

### 3.2 Detection configurations — FloW-Img (`nc: 1`)

| Config file | Description | TEA8 placement |
|-------------|-------------|----------------|
| `yolo11s.yaml` | YOLO11s baseline | — |
| `yolo11s-flow-glare.yaml` | YOLO11s + GLE | — |
| `yolo11s-tea8.yaml` | YOLO11s + GLE + TEA8 | backbone + P4 + P3 (3 modules) |
| `yolo11s-tea8-backbone.yaml` | ablation | backbone only (1 module) |
| `yolo11s-tea8-neck.yaml` | ablation | neck P4 + P3 (2 modules) |

### 3.3 Detection configurations — HELVLAJI (`nc: 9`)

| Config file | Description |
|-------------|-------------|
| `yolo11s-helvlaji.yaml` | YOLO11s baseline |
| `yolo11s-helvlaji-glare.yaml` | YOLO11s + GLE |
| `yolo11s-helvlaji-tea8-glare.yaml` | YOLO11s + GLE + TEA8 (backbone + P4 + P3) |

### 3.4 Segmentation configurations — fenGeHe (`nc: 1`)

| Config file | Description |
|-------------|-------------|
| `seg-yolo11s-baseline.yaml` | YOLO11s-seg baseline |
| `seg-yolo11s-wtg.yaml` | YOLO11s-seg + WTG (P4, P3) |

### 3.5 Reference baselines (`r2/`)

| Config file | Description |
|-------------|-------------|
| `r2/yolov8s-flow.yaml` | YOLOv8s detection baseline (FloW-Img) |
| `r2/yolov8s-helv.yaml` | YOLOv8s detection baseline (HELVLAJI) |
| `r2/yolov8s-seg-fengehe.yaml` | YOLOv8s-seg baseline (fenGeHe) |

### 3.6 Model complexity

| Model | Layers | Parameters | GFLOPs |
|-------|-------:|-----------:|-------:|
| YOLO11s (detection baseline) | 319 | 9,428,179 | 21.7 |
| YOLO11s + TEA8 (raw / fused) | 331 / 250 | 9,428,209 / 9,413,217 | 21.5 / 21.3 |
| YOLO11s-seg (baseline, fused) | 265 | 10,067,203 | 32.8 |

---

## 4. Results

Every number below is the **final validation value** reported by Ultralytics for
the corresponding run. `Epochs` is the last epoch present in
`<experiment>/results.csv`; the full per-epoch trace, the exact
`args.yaml`, the console log and the checkpoint are in the run directory.

### 4.1 River detection — FloW-Img (`nc = 1`, 400 val images / 1107 instances)

| Model | Run directory | Config | Epochs | P | R | mAP@50 | mAP@50-95 |
|-------|---------------|--------|-------:|---:|---:|---:|---:|
| YOLO11s (baseline) | `yolo11s-baseline-flow/` | `yolo11s.yaml` | 500 | 0.885 | 0.810 | 0.879 | **0.482** |
| YOLO11s + GLE | `yolo11s-gle-flow/` | `yolo11s-flow-glare.yaml` | 500 | 0.897 | 0.799 | 0.880 | 0.476 |
| YOLO11s + TEA8 | `tea8-all-r2/` | `yolo11s-tea8.yaml` | 500 | 0.892 | 0.806 | **0.888** | 0.481 |
| YOLO11s + TEA8 (backbone only) | `tea8-backbone/` | `yolo11s-tea8-backbone.yaml` | 500 | 0.867 | 0.818 | 0.878 | 0.476 |
| YOLO11s + TEA8 (neck only) | `tea8-neck/` | `yolo11s-tea8-neck.yaml` | 500 | **0.907** | 0.795 | 0.884 | 0.479 |
| YOLO11s + GLE + TEA8 | `tea8-glare-flow-s2/` | `yolo11s-tea8.yaml` | 500 | 0.873 | **0.822** | 0.885 | 0.477 |
| YOLOv8s (baseline) | `yolov8s-flow/` | `r2/yolov8s-flow.yaml` | 300 | 0.892 | 0.793 | 0.873 | 0.464 |

Bold = best value in each column. `tea8-all-r2` and `tea8-glare-flow-s2` share
the same graph (`yolo11s-tea8.yaml`); the only difference is the GLE glare
augmentation (`p = 0.25`) enabled in `tea8-glare-flow-s2`.

### 4.2 Multi-class detection — HELVLAJI (`nc = 9`)

| Model | Run directory | Config | Epochs | P | R | mAP@50 | mAP@50-95 |
|-------|---------------|--------|-------:|---:|---:|---:|---:|
| YOLO11s (baseline) | `yolo11s-baseline-helv/` | `yolo11s-helvlaji.yaml` | 300 | 0.631 | 0.534 | 0.596 | 0.307 |
| YOLO11s + GLE | `yolo11s-gle-helv/` | `yolo11s-helvlaji-glare.yaml` | 300 | **0.677** | 0.554 | 0.602 | **0.330** |
| YOLO11s + GLE + TEA8 | `tea8-glare-helv/` | `yolo11s-helvlaji-tea8-glare.yaml` | 293 | 0.664 | 0.560 | 0.582 | 0.302 |
| YOLOv8s (baseline) | `yolov8s-helv/` | `r2/yolov8s-helv.yaml` | 258 | 0.607 | **0.601** | **0.637** | 0.300 |

Bold = best value in each column. This set is the hard, 9-class,
strong-glare robustness benchmark; the GLE augmentation gives the largest
precision / mAP@50-95 gains over the YOLO11s baseline.

---

### 4.3 River segmentation — fenGeHe (`nc = 1`, 682 val images / 735 instances)

| Model | Run directory | Config | Epochs | Box P | Box R | Box mAP@50 | Box mAP@50-95 | Mask P | Mask R | Mask mAP@50 | Mask mAP@50-95 |
|-------|---------------|--------|-------:|------:|------:|-----------:|--------------:|-------:|-------:|------------:|---------------:|
| YOLO11s-seg (baseline) | `yolo11s-seg-baseline/` | `seg-yolo11s-baseline.yaml` | 88 | **0.991** | 0.942 | 0.975 | 0.950 | **0.990** | 0.940 | 0.974 | 0.954 |
| YOLO11s-seg + WTG | `yolo11s-seg-wtg/` | `seg-yolo11s-wtg.yaml` | 300 | 0.988 | **0.955** | **0.977** | **0.959** | 0.987 | **0.954** | **0.976** | **0.958** |
| YOLOv8s-seg (baseline) | `yolov8s-seg-fengehe/` | `r2/yolov8s-seg-fengehe.yaml` | 243 | 0.990 | 0.954 | 0.972 | 0.957 | 0.988 | 0.952 | 0.970 | 0.955 |

Bold = best value in each column.

> **Note on the baseline row.** The original `yolo11s-seg-baseline` training run
> (Jul 2025, 88 epochs) was interrupted and its log was not archived.
> The values shown are from the reproducible official re-validation of the
> released checkpoint (`md5 826c255d095479ecf396c33c09a66318`), recorded in
> [`logs/yolo11s-seg-baseline.log`](logs/yolo11s-seg-baseline.log). The WTG and
> YOLOv8s-seg rows come from their own `results.csv`.

### 4.4 Figure index

Each run directory contains the standard Ultralytics plots, which are the
source of the paper's figures:

| File | Contents |
|------|----------|
| `results.png` | training / validation loss and metric curves per epoch |
| `P_curve.png`, `R_curve.png`, `F1_curve.png`, `PR_curve.png` | per-class curves |
| `confusion_matrix.png`, `confusion_matrix_normalized.png` | confusion matrices |
| `labels.jpg`, `labels_correlogram.jpg` | label distribution statistics |
| `train_batch*.jpg` | augmented training batches |
| `val_batch*_labels.jpg`, `val_batch*_pred.jpg` | ground truth vs. prediction |

---

## 5. Methods: TEA8, GLE and WTG

- **TEA8** — an 8-direction Sobel target-edge attention block. The node is
  appended to the YAML graph as `[-1, 1, TEA8, [C]]` (where `C` is the channel
  width of the level) after `C2PSA` in the backbone and after the `C3k2` fusion
  blocks at P4 and P3 in the neck. See §3.2 for the placement ablations.
- **GLE** — a glare-enhancement augmentation injected during training with
  probability `p = 0.25` (`GLE_Reduced`). It leaves the network graph unchanged,
  so inference cost is identical to the baseline.
- **WTG** — a water-texture gate, inserted at P4 and P3 of the segmentation
  head in place of TEA8 (`seg-yolo11s-wtg.yaml`).

The implementation of the three modules lives in the Ultralytics fork used for
training; the YAML files in this repository are the exact architecture
definitions that were loaded for each run (see `model:` in each `args.yaml`).

---

## 6. Training logs

The full console log of every run is mirrored in [`logs/`](logs/):

| Log file | Experiment |
|----------|-----------|
| `logs/yolo11s-baseline-flow.log` | YOLO11s baseline (FloW-Img) |
| `logs/yolo11s-gle-flow.log` | YOLO11s + GLE (FloW-Img) |
| `logs/tea8-all-r2.log` | YOLO11s + TEA8 (FloW-Img) |
| `logs/tea8-backbone.log` | TEA8 backbone-only ablation |
| `logs/tea8-neck.log` | TEA8 neck-only ablation |
| `logs/tea8-glare-flow-s2.log` | YOLO11s + GLE + TEA8 (FloW-Img) |
| `logs/yolo11s-baseline-helv.log` | YOLO11s baseline (HELVLAJI) |
| `logs/yolo11s-gle-helv.log` | YOLO11s + GLE (HELVLAJI) |
| `logs/tea8-glare-helv.log` | YOLO11s + GLE + TEA8 (HELVLAJI) |
| `logs/yolo11s-seg-baseline.log` | YOLO11s-seg baseline official re-validation |
| `logs/yolo11s-seg-wtg.log` | YOLO11s-seg + WTG |
| `logs/yolov8s-flow.log`, `logs/yolov8s-helv.log`, `logs/yolov8s-seg-fengehe.log` | YOLOv8s references |

Each log contains the module-initialisation trace, the model summary
(layers / parameters / GFLOPs), the training loop and the final validation
table, so the reported metrics can be checked line by line.

---

## 7. Reproducing the paper

```bash
# 0. environment (see §8)
export PYTHONPATH=/path/to/my1Yolo11          # Ultralytics 8.3.13 fork

# 1. point the data template at your local dataset copy
$EDITOR datasets/flow-img-yolo.yaml

# 2. train a configuration (example: the full model on FloW-Img)
python train_ablation.py \
    --cfg    yolo11s-tea8.yaml \
    --data   datasets/flow-img-yolo.yaml \
    --task   detect \
    --device 0 \
    --project runs \
    --name    tea8-all \
    --epochs  500 \
    --batch   32

# 3. or validate a released checkpoint directly
yolo val model=tea8-all-r2/weights/best.pt \
     data=datasets/flow-img-yolo.yaml imgsz=640
```

To reproduce a specific row of §4, use the `Config` in that row together with
the dataset from §2 and the hyper-parameters recorded in
`<experiment>/args.yaml`. A **step-by-step protocol, including the ablation
design and the reproducibility checklist, is in
[`PROTOCOL.md`](PROTOCOL.md)**.

---

## 8. Environment

| Component | Value |
|-----------|-------|
| OS | Linux (Ubuntu) |
| GPU | NVIDIA L20, 48 GB |
| CUDA | 13.0 (`torch 2.13.0+cu130`) |
| Python | 3.10.6 (detection) / 3.13.11 (segmentation) |
| Framework | Ultralytics 8.3.13 (fork `my1Yolo11`) |
| Input size | 640 x 640 |
| Optimizer | SGD (`lr0 = 0.01`, `momentum = 0.937`, `weight_decay = 5e-4`) |

---

## 9. Citation

If you use the code, models, configuration files or protocol in this
repository, please cite the paper. Citation metadata is provided in
[`CITATION.cff`](CITATION.cff):

```bibtex
@article{river_garbage_tea8,
  title   = {TEA8-based oriented perception of river garbage},
  author  = {TODO},
  journal = {TODO},
  year    = {2026},
  note    = {Code and models: https://github.com/forkyguo/river-garbage}
}
```

---

## 10. License

[![License: AGPL v3](https://www.gnu.org/graphics/agplv3-155x51.png)](LICENSE)

This project is released under the **GNU Affero General Public License v3.0
(AGPL-3.0)**. The full license text is bundled in this repository as
[`LICENSE`](LICENSE) and is also auto-detected by GitHub (see the license
badge in the repository sidebar).

The training framework derives from
[Ultralytics](https://github.com/ultralytics/ultralytics), which is distributed
under AGPL-3.0, so the same license applies to this derivative work.

> Use of the datasets is governed by the license of each original dataset —
> see [`DATASETS.md`](DATASETS.md).

