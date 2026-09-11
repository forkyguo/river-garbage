# Dataset Protocol

This document defines the datasets used in the paper, their provenance, the
directory layout expected by the training scripts, and ready-to-use `data.yaml`
files. Images are **not** redistributed with this repository.

---

## 1. FloW-Img — river garbage detection (primary detection set)

| Property | Value |
|----------|-------|
| Task | object detection |
| Classes | 1 — `bottle` |
| Images | 2000 |
| Split | train 1400 · val 400 · test 200 |
| Ground truth | 2003 YOLO-format `.txt` annotations |
| Validation set used in the paper | 400 images / 1107 instances |
| Label format | YOLO (`.txt`), class-0 `bottle` |

`data.yaml` (also shipped as [`datasets/flow-img-yolo.yaml`](datasets/flow-img-yolo.yaml)):

```yaml
path: /path/to/flow-img-yolo
train: images/train
val: images/val
test: images/test
nc: 1
names:
  0: bottle
```

---

## 2. HELVLAJI — multi-class river garbage detection (robustness set)

| Property | Value |
|----------|-------|
| Task | object detection |
| Classes | 9 |
| Purpose | verify the method under strong sun glare and multiple waste types |
| Label format | YOLO (`.txt`) |

`data.yaml` template — fill in the real directory and class names
(also shipped as [`datasets/helvlaji.yaml`](datasets/helvlaji.yaml)):

```yaml
path: /path/to/helvlaji
train: images/train
val: images/val
test: images/test
nc: 9
names:
  0: class_0
  # ... replace with the real class names 1..8
  8: class_8
```

---

## 3. fenGeHe — river-surface segmentation

| Property | Value |
|----------|-------|
| Task | instance segmentation |
| Classes | 1 |
| Layout | Roboflow YOLOv8 segmentation export |
| Validation set used in the paper | 682 images / 735 instances |
| Validation directory | `valid/` (Roboflow convention), not `val/` |

`data.yaml` template (also shipped as
[`datasets/fenGeHe.yaml`](datasets/fenGeHe.yaml)):

```yaml
path: /path/to/fenGeHe.yolov8
train: train/images
val: valid/images
test: test/images
nc: 1
names:
  0: river    # replace with the class name used by the original release
```

---

## 4. Expected directory layout

Detection datasets (FloW-Img, HELVLAJI):

```
<dataset_root>/
├── data.yaml
├── images/
│   ├── train/  *.jpg
│   ├── val/    *.jpg
│   └── test/   *.jpg
└── labels/
    ├── train/  *.txt
    ├── val/    *.txt
    └── test/   *.txt
```

Segmentation dataset (fenGeHe, Roboflow export):

```
<dataset_root>/
├── data.yaml
├── train/{images,labels}/
├── valid/{images,labels}/
└── test/{images,labels}/
```

---

## 5. Using the templates

```bash
# 1. edit the `path:` entry of the template for your machine
$EDITOR datasets/flow-img-yolo.yaml

# 2. train / validate with the template
python train_ablation.py --cfg yolo11s-tea8.yaml \
        --data datasets/flow-img-yolo.yaml --task detect ...
```

---

## 6. Frozen splits

Every model, every ablation and every baseline in the paper uses the **same**
split files. Do not regenerate splits — doing so invalidates the comparison.

---

## 7. Validation-set sizes used for the reported tables

| Dataset | Images | Instances (boxes) |
|---------|-------:|------------------:|
| FloW-Img (val) | 400 | 1107 |
| fenGeHe (valid) | 682 | 735 |

These counts appear verbatim in `logs/*.log` and can be used to check that a
re-run used the same split.

---

## 8. Provenance and citation

Please cite the original source of each dataset together with the paper that
accompanies this repository:

| Dataset | Source |
|---------|--------|
| FloW-Img | TODO — add the dataset paper / DOI / URL |
| HELVLAJI | TODO — add the dataset paper / DOI / URL |
| fenGeHe | TODO — add the dataset paper / DOI / URL |

> Redistribution of the images is governed by the licence of each original
> dataset, not by the AGPL-3.0 licence of this repository's code.


