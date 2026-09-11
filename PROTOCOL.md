# Experimental Protocol

**Artifact:** <https://github.com/forkyguo/river-garbage>
**Paper:** *TEA8-based oriented perception of river garbage* (working title —
replace with the camera-ready title and DOI).

This protocol defines **exactly** how the numbers reported in the paper were
produced. It is intended for reviewers and for anyone re-running the
experiments. The machine-readable evidence lives in this repository:

| Evidence | Location |
|----------|----------|
| exact training arguments | `<experiment>/args.yaml` |
| per-epoch train/val metrics | `<experiment>/results.csv` |
| console log (incl. final validation table) | `logs/<experiment>.log` |
| trained checkpoint | `<experiment>/weights/best.pt` |
| learning curves and confusion matrices | `<experiment>/*_curve.png`, `<experiment>/confusion_matrix*.png` |
| dataset specification | [`DATASETS.md`](DATASETS.md), `datasets/*.yaml` |

---

## 1. Scope

The paper makes three claims, each backed by a dedicated experiment family:

1. **TEA8** (8-direction Sobel target-edge attention) improves oriented
   river-garbage detection on FloW-Img, and its placement matters
   (backbone / neck ablations).
2. **GLE** (glare enhancement) augmentation improves robustness under strong
   sun glare (FloW-Img and HELVLAJI).
3. **WTG** (water texture gate) improves river-surface segmentation (fenGeHe).

## 2. Protocol summary

| Item | Setting |
|------|---------|
| Tasks | object detection (FloW-Img, HELVLAJI); instance segmentation (fenGeHe) |
| Primary backbone | YOLO11s (`scale: s`) |
| Reference baseline | YOLOv8s |
| Proposed modules | TEA8, GLE, WTG |
| Input resolution | 640 x 640 (`imgsz: 640`) |
| Augmentation | see §4 |
| Optimizer | SGD, `lr0 = 0.01`, `lrf = 0.01`, `momentum = 0.937`, `weight_decay = 5e-4` |
| LR schedule | linear warm-up, 3 epochs (`warmup_momentum = 0.8`, `warmup_bias_lr = 0.1`), then linear decay |
| Batch size | 32 (`16` for the fenGeHe baseline re-validation) |
| Epochs | 500 (FloW-Img) · 300 (HELVLAJI, fenGeHe) |
| Early stopping | `patience: 100` |
| Mixed precision | disabled (`amp: false`) |
| Validation during training | `val: false`; the final `best.pt` is re-validated once |
| Hardware | NVIDIA L20 48 GB (single GPU per run) |
| Software | Ultralytics 8.3.13 (project fork `my1Yolo11`), PyTorch 2.13.0+cu130, Python 3.10.6 / 3.13.11 |
| Seed | `0`, single repetition per configuration |

## 3. Datasets and splits

| Dataset | Task | Classes | Split (train / val / test) |
|---------|------|--------:|----------------------------|
| FloW-Img | detection of floating garbage | 1 (`bottle`) | 1400 / 400 / 200 |
| HELVLAJI | multi-class detection | 9 | as released by the dataset authors |
| fenGeHe | river-surface segmentation | 1 | 682 validation images |

- Labels are in the standard YOLO text format; annotation counts, directory
  layout and ready-to-use `data.yaml` files are given in
  [`DATASETS.md`](DATASETS.md).
- **No test-set labels were used at any point during model selection.**
  Test splits are held out for the final paper evaluation only.
- Splits are fixed and identical for every model and every ablation.

## 4. Input pipeline and augmentation

All images are letter-boxed to 640 x 640. The Ultralytics augmentations below
are shared by every run (values taken from `<experiment>/args.yaml`):

| Augmentation | Value |
|--------------|------:|
| `hsv_h` / `hsv_s` / `hsv_v` | 0.015 / 0.7 / 0.4 |
| `degrees` | 0.0 |
| `translate` | 0.1 |
| `scale` | 0.5 |
| `shear` / `perspective` | 0.0 / 0.0 |
| `flipud` / `fliplr` | 0.0 / 0.5 |
| `mosaic` | 1.0 (disabled for the last `close_mosaic = 10` epochs) |
| `mixup` / `copy_paste` | 0.0 / 0.0 |
| `erasing` | 0.4 |
| `auto_augment` | randaugment |

**GLE** is the only dataset-level change: a glare-enhancement augmentation
(`GLE_Reduced`) is injected with probability **p = 0.25**. It affects training
images only and adds no inference-time cost — the network graph is unchanged.
Runs without GLE use the identical pipeline with the glare transform disabled.

## 5. Model configurations

See [README §3](README.md#3-models-and-configuration-files) for the full
configuration matrix. Key points:

- Architectural changes are **only** in the YAML graph files; every other
  setting is held constant.
- TEA8 is a lightweight edge-attention block inserted after `C3k2`/`C2PSA` at
  the stated pyramid levels.
- WTG replaces TEA8 in the segmentation head at P4 and P3.

## 6. Training protocol

```bash
export PYTHONPATH=/path/to/my1Yolo11            # Ultralytics 8.3.13 fork
python train_ablation.py \
    --cfg     yolo11s-tea8.yaml \
    --data    datasets/flow-img-yolo.yaml \
    --task    detect \
    --device  0 \
    --project runs \
    --name    tea8-all \
    --epochs  500 \
    --batch   32
```

- One GPU per run; runs are independent and were executed in parallel on
  separate devices (see `device` in each `args.yaml`).
- `pretrained: true` — the s-scale COCO checkpoint initialises the network.
- `optimizer: auto` resolves to the SGD settings listed in §2.
- Model selection uses the checkpoint that maximises the validation fitness
  (`0.1 * mAP@50 + 0.9 * mAP@50-95`); it is stored as `weights/best.pt`.

## 7. Evaluation protocol

- Framework: `model.val()` of Ultralytics 8.3.13 with `imgsz = 640`,
  `conf` unset (default 0.001), `iou = 0.7`, `max_det = 300`.
- Metrics: precision, recall, mAP@50, mAP@50-95 for boxes; mask metrics are
  added for segmentation.
- The released `best.pt` of every run is re-validated once so that the reported
  table can be reproduced from the released checkpoint alone
  (see `logs/yolo11s-seg-baseline.log` for an example record).
- Reported numbers are taken verbatim from `results.csv` (final epoch) or from
  the re-validation table in `logs/`.

## 8. Ablation design

| Ablation | Configurations compared | Controlled variable |
|----------|-------------------------|---------------------|
| TEA8 placement | `yolo11s-tea8.yaml`, `yolo11s-tea8-backbone.yaml`, `yolo11s-tea8-neck.yaml` | number / location of TEA8 blocks |
| GLE | `yolo11s.yaml` vs `yolo11s-flow-glare.yaml`; `yolo11s-helvlaji.yaml` vs `yolo11s-helvlaji-glare.yaml` | glare probability (0 → 0.25) |
| GLE + TEA8 | `yolo11s-tea8.yaml`, `yolo11s-helvlaji-tea8-glare.yaml` | combined effect |
| Architecture | `yolo11s*` vs `r2/yolov8s*` | YOLO11s vs YOLOv8s |
| WTG | `seg-yolo11s-baseline.yaml` vs `seg-yolo11s-wtg.yaml` | presence of WTG |

Each ablation varies exactly one factor; dataset, splits, augmentation,
schedule and evaluation are held fixed.

## 9. Environment

| Component | Value |
|-----------|-------|
| OS | Linux (Ubuntu) |
| GPU | NVIDIA L20, 48 GB |
| CUDA driver / runtime | 13.0 (`torch 2.13.0+cu130`) |
| Python | 3.10.6 (detection) / 3.13.11 (segmentation) |
| Framework | Ultralytics 8.3.13 (fork `my1Yolo11`) |
| DataLoader workers | 8 |

## 10. Reproducibility checklist

- [x] Datasets documented with exact splits ([`DATASETS.md`](DATASETS.md)).
- [x] Ready-to-use `data.yaml` templates (`datasets/`).
- [x] Network architecture YAMLs for every reported model.
- [x] Exact training arguments (`args.yaml`) for every run.
- [x] Per-epoch metrics (`results.csv`) for every run.
- [x] Console logs including the final validation table (`logs/`).
- [x] Released `best.pt` checkpoints for every run.
- [x] Fixed seed (`0`) and identical augmentation across runs.
- [ ] Multi-seed error bars — reported numbers are single-seed.
- [ ] One-command end-to-end script — training is launched per run (see §6).

## 11. Deviations and limitations

1. **Single seed.** Each configuration was trained once (seed 0); no variance
   is reported. Differences below run-to-run noise should not be treated as
   significant.
2. **fenGeHe baseline.** The original baseline training log (Jul 2025,
   88 epochs) was not archived. The released `best.pt`
   (md5 `826c255d095479ecf396c33c09a66318`) is therefore accompanied by a
   reproducible *validation* record (`logs/yolo11s-seg-baseline.log`) instead
   of a training log.
3. **Interrupted runs.** A few runs terminate before the configured epoch
   budget (e.g. `tea8-glare-helv` at epoch 293, `yolov8s-helv` at 258,
   `yolov8s-seg-fengehe` at 243). The final surviving epoch is reported and the
   full trace is in `results.csv`.
4. **`amp: false`.** Automatic mixed precision was disabled for maximum
   numerical reproducibility; this does not affect the accuracy conclusions.
5. **GPU device IDs** in `args.yaml` refer to the authors' server and are
   irrelevant to reproduction.

## 12. Ethics, licensing and data availability

- **Data availability.** Dataset images are not redistributed. FloW-Img,
  HELVLAJI and fenGeHe must be obtained from their original sources; see
  [`DATASETS.md`](DATASETS.md) for the citation of each dataset.
- **Ethics.** The datasets contain only environmental imagery; no personal or
  sensitive data is processed.
- **License.** The training framework derives from Ultralytics and is therefore
  released under **AGPL-3.0**. Check the licence of each dataset before
  redistribution or commercial use.

