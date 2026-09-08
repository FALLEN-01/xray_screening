# X-Ray Baggage Screening System

> **Deep learning classification of prohibited items in X-ray baggage scans**  
> ResNet-50 · SIXray Dataset · Focal Loss · Grad-CAM · Gradio Demo

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Step-by-Step Usage](#step-by-step-usage)
- [Using Real SIXray Data](#using-real-sixray-data)
- [Results](#results)
- [Architecture](#architecture)
- [Class Imbalance Strategy](#class-imbalance-strategy)
- [Output Files](#output-files)
- [Citation](#citation)

---

## Overview

Automated screening of X-ray baggage images is critical for aviation and public security, yet manual inspection remains time-consuming and prone to human fatigue-induced error.

This project presents a deep learning classification system for detecting **prohibited items** (knives, guns, scissors) in X-ray baggage scans using:

| Component | Detail |
|---|---|
| **Backbone** | ResNet-50, ImageNet pretrained |
| **Dataset** | SIXray (or built-in synthetic demo) |
| **Imbalance** | Focal Loss + WeightedRandomSampler |
| **Fine-tuning** | Two-phase: head-only → full model |
| **Interpretability** | Grad-CAM heatmaps on test images |
| **Metrics** | Precision, Recall, F1, ROC-AUC, PR curve |
| **Demo** | Gradio web app (drag-and-drop) |

---

## Project Structure

```
xray_screening/
├── config.yaml              <- All hyperparameters in one place
├── run_pipeline.py          <- One-command end-to-end runner
├── requirements.txt         <- Python dependencies
├── README.md
│
├── src/
│   ├── utils.py             <- Config loader, device setup, checkpoints
│   ├── synthetic_data.py    <- Synthetic X-ray generator (knife/gun/scissors)
│   ├── dataset.py           <- SIXray-compatible dataset, splits, sampler
│   ├── model.py             <- ResNet-50 + custom head + Grad-CAM target
│   ├── train.py             <- Focal Loss, warmup cosine LR, early stopping
│   ├── evaluate.py          <- P/R/F1, confusion matrix, ROC, PR, threshold sweep
│   └── gradcam.py           <- Grad-CAM batch + single-image visualization
│
├── app/
│   └── demo.py              <- Gradio web demo (localhost:7860)
│
├── notebooks/
│   └── analysis.ipynb       <- EDA, training curves, metrics, Grad-CAM viewer
│
├── data/
│   └── raw/
│       ├── positive/        <- Drop SIXray prohibited images here
│       └── negative/        <- Drop SIXray safe images here
│
└── outputs/
    ├── checkpoints/         <- Saved model weights (best_model.pth)
    ├── logs/                <- TensorBoard events + training_log.csv
    └── visualizations/      <- Confusion matrix, ROC, Grad-CAM grid, etc.
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> PyTorch with CUDA (RTX support):
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
> ```

### 2. Run full pipeline (no dataset download needed)

```bash
cd "D:\New folder\xray_screening"
python run_pipeline.py
```

This will automatically:
1. Generate 800 synthetic prohibited + 4000 safe X-ray images
2. Train ResNet-50 (two-phase: head → full fine-tune)
3. Evaluate on test set (P/R/F1, ROC, PR curves)
4. Generate Grad-CAM visualizations

---

## Step-by-Step Usage

### Train

```bash
# With synthetic demo data (auto-generated)
python src/train.py --demo

# With real SIXray data (set synthetic_demo: false in config.yaml first)
python src/train.py
```

### Evaluate

```bash
python src/evaluate.py
# Outputs: confusion_matrix.png, roc_curve.png, pr_curve.png,
#          threshold_sweep.png, per_class_metrics.png, metrics_report.json
```

### Grad-CAM Visualizations

```bash
# Batch mode (samples from test set)
python src/gradcam.py

# Single image
python src/gradcam.py --image path/to/scan.jpg
```

### Launch Web Demo

```bash
python app/demo.py
```

Open **http://localhost:7860** — drag-and-drop any X-ray image to get:
- Classification result (Safe / Prohibited)
- Confidence score
- Grad-CAM heatmap overlay

### TensorBoard

```bash
tensorboard --logdir outputs/logs
```

---

## Using Real SIXray Data

1. **Download** from [MeioJane/SIXray](https://github.com/MeioJane/SIXray) *(academic use only)*
2. **Place images**:
   - Prohibited images (gun/knife/wrench/pliers/scissors) → `data/raw/positive/`
   - Safe bag images → `data/raw/negative/`
3. **Update config** — set `synthetic_demo: false` in `config.yaml`
4. **Run training**:
   ```bash
   python src/train.py
   ```

Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`

---

## Results

Training results on synthetic SIXray-like dataset (800 prohibited / 4000 safe):

### Training Curves

| Epoch | Phase | Train Loss | Train F1 | Val F1 |
|---|---|---|---|---|
| 1 | freeze | 0.0238 | 0.873 | 1.000 |
| 2 | freeze | 0.0178 | 0.927 | 1.000 |
| 3 | freeze | 0.0128 | 0.941 | 1.000 |
| 4 | freeze | 0.0124 | 0.949 | 1.000 |
| 5 | freeze | 0.0096 | 0.961 | 1.000 |
| 6 | finetune | 0.0026 | 0.991 | 1.000 |
| 7 | finetune | 0.0018 | 0.996 | 1.000 |
| 8 | finetune | 0.0014 | 0.997 | 1.000 |
| **9** | **finetune** | **0.0014** | **0.998** | **1.000** |

*Early stopped at epoch 9 (patience=8, no improvement).*

### Test Set Metrics (720 images)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Safe | 1.000 | 1.000 | 1.000 | 600 |
| Prohibited | 1.000 | 1.000 | 1.000 | 120 |
| **Weighted avg** | **1.000** | **1.000** | **1.000** | **720** |

**ROC-AUC = 1.0000 · Average Precision = 1.0000 · Best threshold = 0.16**

> On real SIXray data with overlapping/occluded objects, expect realistic F1 of 0.85–0.95, which is the actual research challenge addressed by this system.

---

## Architecture

```
Input (224x224x3)
      |
ResNet-50 Backbone (ImageNet pretrained)
  └── conv1 → bn1 → relu → maxpool
  └── layer1 → layer2 → layer3 → layer4   <-- Grad-CAM target
  └── AdaptiveAvgPool2d
      |
   [2048-dim features]
      |
Custom Classification Head
  └── Dropout(0.4)
  └── Linear(2048 → 512) + ReLU
  └── Dropout(0.2)
  └── Linear(512 → 2)   [Safe | Prohibited]
      |
   Logits → Softmax → Prediction
```

### Two-Phase Fine-Tuning

| Phase | Epochs | Backbone | Head LR | Backbone LR |
|---|---|---|---|---|
| 1 — Feature Extraction | 1–5 | Frozen | 1e-3 | — |
| 2 — Full Fine-Tune | 6+ | Unfrozen | 1e-4 | 1e-5 |

---

## Class Imbalance Strategy

SIXray has a ~1:5 to 1:1000 positive-to-negative ratio. This project addresses it with:

| Strategy | Implementation |
|---|---|
| **WeightedRandomSampler** | Each batch over-samples the positive class |
| **Focal Loss** (γ=2, α=0.25) | Reduces gradient from easy negatives |
| **CLAHE augmentation** | Enhances low-contrast item features |
| **Two-phase fine-tuning** | Prevents early feature corruption |
| **Threshold tuning** | Sweeps 0.01–0.99 to maximize recall |

---

## Output Files

| File | Description |
|---|---|
| `outputs/checkpoints/best_model.pth` | Best model by validation F1 |
| `outputs/logs/training_log.csv` | Per-epoch metrics (all phases) |
| `outputs/visualizations/confusion_matrix.png` | Raw + normalized confusion matrix |
| `outputs/visualizations/roc_curve.png` | ROC curve + AUC |
| `outputs/visualizations/pr_curve.png` | Precision-Recall curve + AP |
| `outputs/visualizations/threshold_sweep.png` | F1/P/R vs decision threshold |
| `outputs/visualizations/per_class_metrics.png` | Bar chart per class |
| `outputs/visualizations/gradcam_grid.png` | Grad-CAM overlays (test samples) |
| `outputs/visualizations/metrics_report.json` | All metrics exported as JSON |

---

## Citation

If using SIXray data, please cite:

```bibtex
@InProceedings{Miao_2019_CVPR,
  author    = {Miao, Caijing and Xie, Lingxi and Wan, Fang and Su, Chi
               and Liu, Hongye and Jiao, Jianbin and Ye, Qixiang},
  title     = {SIXray: A Large-Scale Security Inspection X-Ray Benchmark
               for Prohibited Item Discovery in Overlapping Images},
  booktitle = {CVPR},
  year      = {2019}
}
```
