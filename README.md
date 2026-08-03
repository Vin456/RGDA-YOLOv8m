# Pothole Detection: Cross-Architecture Benchmark

Fine-tuning and evaluation of object detectors for pothole detection on a
deduplicated multi-source dataset, under a single evaluation protocol.

## Notebooks

| Notebook | Configurations |
|---|---|
| `pothole_base_model.ipynb` | YOLOv8m, YOLOv10m, YOLOv11m, Faster R-CNN, SSD-VGG16 |
| `pothole_hybrid.ipynb` | YOLOv8m+CBAM, YOLOv8m+CoordAttn, YOLO+Faster R-CNN ensemble (WBF) |
| `rtdetr.ipynb` | RT-DETR |
| `YOLOv8_CBAM_ECA.ipynb` | **RGDA-YOLOv8m** (proposed), the 2x2 ECA/CBAM ablation, and the same block on YOLOv11m |

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

## RGDA-YOLOv8m

`YOLOv8_CBAM_ECA.ipynb` adds the proposed configuration: Efficient Channel
Attention followed by CBAM at the three PAN-FPN neck blocks that feed the
detection head, with channel widths 192, 384 and 576.

The refined tensor is not substituted for the original. It is mixed back
through a learnable scalar gate

```
out = x + tanh(alpha) * (A(x) - x)          alpha initialised to 0
```

so at initialisation `tanh(0) = 0` and the modified network computes exactly
the pretrained detector. The attention pathway is admitted only to the extent
that training opens the gate. Writing it as an interpolation rather than
`x + tanh(alpha)*A(x)` means `tanh(alpha) = 1` is full attention rather than
double the feature magnitude.

Modules are attached by PyTorch forward hooks rather than by editing the
Ultralytics model definition, so the modified and unmodified runs differ in
exactly one respect. Two consequences follow. Hooks must be registered from an
`on_train_start` callback, because `YOLO.train()` builds its own `nn.Module`
and discards anything hooked beforehand -- and does so silently, reproducing
the baseline exactly. And the attention parameters live outside the detector's
`state_dict`, so they are saved separately; a checkpoint loaded without them
is an unmodified YOLOv8m that runs without error.

| Injection point | C | ECA | Channel gate | Spatial gate | Total |
|---|---|---|---|---|---|
| Shallow (P3 fusion) | 192 | 5 | 4,608 | 98 | 4,712 |
| Middle (P4 fusion) | 384 | 5 | 18,432 | 98 | 18,536 |
| Deep (P5 fusion) | 576 | 5 | 41,472 | 98 | 41,576 |
| **Total** | | **15** | **64,512** | **294** | **64,824** |

64,824 parameters is 0.25% over the 25.9M base model. The CBAM channel gate is
99.5% of it, since its cost grows as C^2/r while the rest is constant or
logarithmic in C.

### Multi-seed result

Trained at seeds 42, 123 and 456 under a two-phase schedule: 10 epochs with the
first ten layers frozen at lr 1e-3, then 40 epochs unfrozen at lr 2e-4.

| Configuration | mAP@0.5 | F1 | FPS |
|---|---|---|---|
| YOLOv8m (no attention) | 0.8059 +- 0.0107 | 0.7601 +- 0.0111 | 152.2 |
| **RGDA-YOLOv8m** | **0.8141 +- 0.0142** | **0.7708 +- 0.0121** | 139.9 |

0.82 and 1.07 percentage points, for 0.25% more parameters and 8.1% less
throughput. Both precision and recall rise, so the F1 gain is not a trade.

The mean improvement sits inside the seed-to-seed standard deviation, so it is
not established by the means alone. What supports it is the pairing: the
combined configuration beats the identically trained baseline at **every** seed,
by 0.62, 1.30 and 0.52 points of mAP@0.5. Three paired runs still cannot bound
the size of the effect.

### Ablation

All four cells share the split, the injection points, the gate and the
schedule; only the transformation the gate interpolates towards changes.

| Configuration | mAP@0.5 | F1 |
|---|---|---|
| No attention | 0.8059 +- 0.0107 | 0.7601 +- 0.0111 |
| ECA only | 0.7986 +- 0.0073 | 0.7533 +- 0.0068 |
| CBAM only | 0.8012 +- 0.0034 | 0.7544 +- 0.0079 |
| **ECA + CBAM** | **0.8141 +- 0.0142** | **0.7708 +- 0.0121** |

Neither module helps alone. ECA costs 0.73 points of mAP@0.5 and CBAM 0.47, so
an additive account predicts about -1.20 for the pair; the measured value is
+0.82. Whatever the block does, it is not the sum of its parts. Note that CBAM
carries a channel branch of its own, so the ECA factor measures an *additional*
parameter-light channel interaction placed before CBAM's bottlenecked gate --
not channel attention versus spatial attention.

### The same block on YOLOv11m

YOLOv11m places a C2PSA partial-self-attention module after the backbone's SPPF
stage, so features reach its neck already reweighted once. The injector selects
neck blocks positionally by class name, so it moved between hosts unchanged.

| Host | No attention | + ECA+CBAM | Effect |
|---|---|---|---|
| YOLOv8m | 0.8059 +- 0.0107 | 0.8141 +- 0.0142 | **+0.0082** |
| YOLOv11m | 0.7899 | 0.7823 +- 0.0048 | **-0.0076** |

The effect reverses in sign, in the same direction at all three seeds (0.7777,
0.7873, 0.7818, all under the 0.7899 baseline).

Read this with its caveat. The YOLOv11m baseline is a **single run** carried
over from the benchmark, not a three-seed mean, so the comparison is unpaired
and no dispersion estimate exists for it. The 0.76-point degradation is smaller
than the 2.3-point single-run-to-three-seed gap seen elsewhere in these
experiments. The direction is consistent; the magnitude is not established.
Training a three-seed YOLOv11m baseline is the single change that would settle
it.

Taken with that qualification, the suggestion is that the value of an added
attention module depends on what the host already provides -- a property of the
pairing rather than of the module.

## Setup

```bash
pip install ultralytics torch torchvision kagglehub opencv-python \
            pandas numpy matplotlib seaborn tqdm pyyaml
```

Trained on a single NVIDIA GeForce RTX 3090 (24 GB). CPU inference works but
the throughput figures above will not reproduce.

## Reproducing

Run the notebooks in this order -- the baseline notebook writes the weights
that the hybrid notebook's ensemble depends on:

1. `pothole_base_model.ipynb`
2. `pothole_hybrid.ipynb`
3. `rtdetr.ipynb`
4. `YOLOv8_CBAM_ECA.ipynb`

Each notebook is self-contained: it downloads the three source datasets,
deduplicates them, builds the same seed-42 split and trains from
COCO-pretrained weights. Set `SKIP_TRAINING = True` to reuse saved weights.

**Run every notebook top to bottom from a fresh kernel.** The throughput
benchmark writes its measurements back into the in-memory result
dictionaries, so the results table and export cells must run after it. A
partial or out-of-order run will export the in-loop timings rather than the
dedicated-pass figures reported above.

## Evaluation protocol

All configurations are scored through one evaluator so the comparison
measures detectors rather than evaluation code.

- **AP** -- COCO-style 101-point interpolation at IoU 0.5. Scoring identical
  predictions under the PASCAL VOC 2007 11-point rule shifts mAP@0.5 by 2.7
  percentage points, more than the differences between several of the
  configurations above, so the two are never mixed.
- **Matching** -- greedy and score-ordered, one detection per ground-truth
  box. A detection whose best match is already claimed falls through to the
  best still-unmatched box rather than being counted a false positive
  outright, which is the COCO and VOC behaviour.
- **Operating points** -- P/R/F1 at a fixed confidence of 0.25, and again at
  the confidence maximising F1. A single fixed threshold is not neutral
  across detector families.
- **Throughput** -- images divided by total wall-clock over a dedicated pass,
  with images pre-decoded, a warm-up discarded and the device synchronised.
  Not the mean of per-image reciprocals, which is biased upward.

## Licence

Released for academic use. The source datasets retain their original
licences on Kaggle.
