# 模型及日志
---


## 1. 模型的目录

| 子目录 | 说明 |
|--------|------|
| `yolo11s-baseline-flow` | YOLO11s 基线（FloW-Img） |
| `tea8-all-r2` | YOLO11s + TEA8（三处部署） |
| `yolo11s-gle-flow` | YOLO11s + GLE（p=0.25） |
| `tea8-glare-flow-s2` | YOLO11s + GLE + TEA8（p=0.25） |
| `tea8-backbone` | TEA8 仅部署于 backbone |
| `tea8-neck` | TEA8 仅部署于 Neck P3+P4 |
| `yolo11s-baseline-helv` | YOLO11s 基线（HELVLAJI） |
| `yolo11s-gle-helv` | YOLO11s + GLE（p=0.25） |
| `tea8-glare-helv` | YOLO11s + GLE + TEA8（p=0.25） |
| `yolo11s-seg-baseline` | YOLO11s-seg 基线（fenGeHe） |
| `yolo11s-seg-wtg` | YOLO11s-seg + WTG |
| `yolov8s-flow` | YOLOv8s 对比模型（FloW-Img） |
| `yolov8s-helv` | YOLOv8s 对比模型（HELVLAJI） |
| `yolov8s-seg-fengehe` | YOLOv8s-seg 对比模型（fenGeHe） |



## 2. 训练日志

`logs/` 

## 3. 模型的网络结构配置文件
yaml文件都是模型的网络结构配置文件