# Pothole Detection: Cross-Architecture Benchmark

Fine-tuning and evaluation of object detectors for pothole detection on a
deduplicated multi-source dataset, under a single evaluation protocol.

## Notebooks

| Notebook | Configurations |
|---|---|
| `pothole_base_model.ipynb` | YOLOv8m, YOLOv10m, YOLOv11m, Faster R-CNN, SSD-VGG16 |
| `pothole_hybrid.ipynb` | YOLOv8m+CBAM, YOLOv8m+CoordAttn, YOLO+Faster R-CNN ensemble (WBF) |
| `rtdetr.ipynb` | RT-DETR |

## Dataset

Three public Kaggle repositories are merged and deduplicated at runtime via
`kagglehub`. No image data is stored in this repository.

| Source | Retained | Boxes |
|---|---|---|
| `chitholian/annotated-potholes-dataset` | 665 | 1,740 |
| `andrewmvd/pothole-detection` | 9 | 31 |
| `ashishkumarak/training-setzip` | 252 | 583 |
| **Total** | **926** | **2,354** |

Deduplication removes 1,078 of 2,004 images -- over half the corpus, because
two of the three repositories redistribute overlapping data. Byte-level MD5
is unreliable here: redistributed images are re-encoded and so differ at the
byte level while being visually identical. Matching is therefore done on
64x64 greyscale downsamples, treating a pair as duplicate when the mean
absolute difference falls below 1.0 on a 0-255 scale.

Without this step near-duplicates appear in both partitions and every
reported figure is inflated.

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
