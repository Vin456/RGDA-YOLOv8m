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

## Results

Validation partition of 186 images and 474 annotated instances. Precision,
recall and F1 are at a fixed confidence of 0.25; F1* is the maximum over a
confidence sweep. Throughput is measured on an NVIDIA GeForce RTX 3090 at
FP32 with batch size 1.

| Configuration | mAP@0.5 | Precision | Recall | F1 | F1* | FPS |
|---|---|---|---|---|---|---|
| YOLO + Faster R-CNN ensemble (WBF) | **0.8272** | **0.8564** | 0.6793 | 0.7576 | 0.7774 | 23.9 |
| RT-DETR | 0.8174 | 0.5674 | **0.8523** | 0.6813 | **0.7919** | 61.1 |
| YOLOv8m | 0.8100 | 0.7783 | 0.7553 | **0.7666** | 0.7755 | 152.5 |
| Faster R-CNN | 0.7976 | 0.7389 | 0.7342 | 0.7365 | 0.7399 | 28.3 |
| YOLOv11m | 0.7899 | 0.7419 | 0.7278 | 0.7348 | 0.7446 | 161.0 |
| YOLOv8m + CoordAttn | 0.7879 | 0.7543 | 0.7384 | 0.7463 | 0.7495 | 144.4 |
| YOLOv8m + CBAM | 0.7782 | 0.7400 | 0.7447 | 0.7424 | 0.7541 | 142.3 |
| YOLOv10m | 0.7598 | 0.7014 | 0.7236 | 0.7124 | 0.7235 | 173.9 |
| SSD-VGG16 | 0.6165 | 0.6936 | 0.5253 | 0.5978 | 0.6169 | **174.7** |

No configuration leads on all three headline measures. The ensemble leads
mAP@0.5, RT-DETR leads F1* and YOLOv8m leads F1 at the fixed threshold.

RT-DETR is the clearest illustration of why two operating points are
reported: its F1 rises by 11.1 percentage points when swept from 0.25 to its
own optimum of 0.65, with no change to the model itself.

Each configuration was fine-tuned in a separate experiment, so differences
between nominally identical backbones across notebooks reflect run-to-run
variation.

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
