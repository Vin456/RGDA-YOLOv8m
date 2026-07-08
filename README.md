# Pothole Detection: Cross-Architecture Benchmark

Fine-tuning and evaluation of object detectors for pothole detection on a
deduplicated multi-source dataset, under a single evaluation protocol.

## Notebooks

| Notebook | Configurations |
|---|---|
| `pothole_base_model.ipynb` | YOLOv8m, YOLOv10m, YOLOv11m, Faster R-CNN, SSD-VGG16 |
| `pothole_hybrid.ipynb` | YOLOv8m+CBAM, YOLOv8m+CoordAttn, YOLO+Faster R-CNN ensemble (WBF) |
| `rtdetr.ipynb` | RT-DETR |

## Setup

```bash
pip install ultralytics torch torchvision kagglehub opencv-python \
            pandas numpy matplotlib seaborn tqdm pyyaml
```

Trained on a single NVIDIA GeForce RTX 3090 (24 GB). CPU inference works but
the throughput figures above will not reproduce.

## Licence

Released for academic use. The source datasets retain their original
licences on Kaggle.
