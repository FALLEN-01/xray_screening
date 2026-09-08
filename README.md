# 🔍 X-Ray Baggage Screening System

A deep learning classification system for detecting **prohibited items** (knives, guns, scissors) in X-ray baggage scans, built as a lab/minor project using the **SIXray dataset** and **ResNet-50**.

---

## 📌 Features

| Feature | Detail |
|---|---|
| **Model** | ResNet-50, ImageNet pretrained, fine-tuned |
| **Dataset** | SIXray (or synthetic demo mode) |
| **Imbalance** | Focal Loss + WeightedRandomSampler |
| **Interpretability** | Grad-CAM heatmaps |
| **Metrics** | Precision, Recall, F1, ROC-AUC, PR curve |
| **Demo** | Gradio web app (drag-and-drop) |

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run full pipeline (synthetic demo data — no download needed)
```bash
python run_pipeline.py
```

### 3. Train only
```bash
python src/train.py --demo       # generates synthetic data + trains
```

### 4. Evaluate
```bash
python src/evaluate.py
```

### 5. Grad-CAM visualizations
```bash
python src/gradcam.py
python src/gradcam.py --image path/to/scan.jpg   # single image
```

### 6. Launch web demo
```bash
python app/demo.py
```
Open `http://localhost:7860`

---

## 📁 Project Structure

```
xray_screening/
├── config.yaml              ← All hyperparameters
├── run_pipeline.py          ← One-command end-to-end runner
├── requirements.txt
├── src/
│   ├── dataset.py           ← SIXray-compatible dataset, splits, sampler
│   ├── model.py             ← ResNet-50 with custom head + Grad-CAM hook
│   ├── train.py             ← Focal Loss, two-phase fine-tuning, TensorBoard
│   ├── evaluate.py          ← Full metrics, plots, JSON report
│   ├── gradcam.py           ← Grad-CAM batch + single-image visualization
│   ├── synthetic_data.py    ← Synthetic X-ray generator (demo mode)
│   └── utils.py             ← Config, logging, checkpoints
├── app/
│   └── demo.py              ← Gradio web demo
├── data/
│   └── raw/
│       ├── positive/        ← Drop SIXray prohibited images here
│       └── negative/        ← Drop SIXray safe images here
└── outputs/
    ├── checkpoints/         ← Saved model weights
    ├── logs/                ← TensorBoard + CSV logs
    └── visualizations/      ← Confusion matrix, ROC, Grad-CAM grid
```

---

## 📊 Using Real SIXray Data

1. Download from [MeioJane/SIXray](https://github.com/MeioJane/SIXray) (academic use only)
2. Place positive images (gun/knife/wrench/pliers/scissors) → `data/raw/positive/`
3. Place negative images → `data/raw/negative/`
4. Set `synthetic_demo: false` in `config.yaml`
5. Run: `python src/train.py`

---

## 🔬 Class Imbalance Strategy

SIXray has ~1:5 to 1:1000 positive-to-negative ratio. We address this with:

1. **Focal Loss** (γ=2): reduces gradient from easy negatives
2. **WeightedRandomSampler**: each batch has balanced class representation
3. **CLAHE augmentation**: enhances low-contrast prohibited item features
4. **Threshold tuning**: sweeps decision threshold to maximize recall ≥ 0.95

---

## 📈 Outputs

After running the pipeline:

| File | Contents |
|---|---|
| `outputs/checkpoints/best_model.pth` | Best model by val F1 |
| `outputs/logs/training_log.csv` | Per-epoch metrics |
| `outputs/visualizations/confusion_matrix.png` | Raw + normalized CM |
| `outputs/visualizations/roc_curve.png` | ROC + AUC |
| `outputs/visualizations/pr_curve.png` | Precision-Recall curve |
| `outputs/visualizations/threshold_sweep.png` | F1/P/R vs threshold |
| `outputs/visualizations/per_class_metrics.png` | Bar chart per class |
| `outputs/visualizations/gradcam_grid.png` | Grad-CAM overlays |
| `outputs/visualizations/metrics_report.json` | All metrics as JSON |

---

## 🖥️ TensorBoard

```bash
tensorboard --logdir outputs/logs
```

---

## 📝 Citation

If using SIXray data:
> Miao, C., Xie, L., Wan, F., Su, C., Liu, H., Jiao, J., & Ye, Q. (2019). SIXray: A Large-scale Security Inspection X-Ray Benchmark for Prohibited Item Discovery in Overlapping Images. CVPR 2019.
