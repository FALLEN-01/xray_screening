# 🔍 X-Ray Prohibited-Item Object Detector

> **Status: ✅ Complete**

An end-to-end deep learning pipeline for detecting prohibited items in X-ray baggage scans. Built on [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) and trained on the **SPXray dataset**, the system detects 12 classes of dangerous/prohibited items with bounding boxes, confidence scores, and an interactive Gradio demo.

---

## 🎯 Features

- **12-class YOLO object detector** trained on real X-ray scan data (SPXray)
- **Full training pipeline** — data loading → augmentation → training → evaluation → visualization
- **Grad-CAM visualizations** to interpret model attention
- **Interactive Gradio demo** — upload an X-ray image and get instant predictions
- **Checkpointing** — resume training from best weights at any time
- **TensorBoard logging** for training metrics

---

## 🗂️ Project Structure

```text
xray_screening/
├── src/
│   ├── dataset.py          # Dataset loading & YOLO-format handling
│   ├── model.py            # Model definitions (ResNet-50, YOLO wrapper)
│   ├── train.py            # Training loop & checkpoint management
│   ├── evaluate.py         # mAP50 / mAP50-95 evaluation
│   ├── gradcam.py          # Grad-CAM saliency visualizations
│   ├── synthetic_data.py   # Synthetic augmentation utilities
│   └── utils.py            # Shared helpers
├── app/                    # Gradio demo application
├── notebooks/              # Analysis & visualization notebooks
├── outputs/
│   ├── checkpoints/        # Saved model weights (best.pt)
│   ├── logs/               # Training logs
│   └── visualizations/     # Annotated predictions & plots
├── runs/                   # YOLO training run artefacts
├── data/                   # Dataset (YOLO layout — see below)
├── config.yaml             # Training hyperparameters
├── run_pipeline.py         # Main entry point
├── requirements.txt
└── yolov8n.pt              # Pretrained YOLOv8n backbone
```

---

## 📦 Dataset Layout

The project expects data in standard YOLO format under `data/`:

```text
data/
├── data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

### 12 Prohibited-Item Classes

| # | Class | # | Class |
|---|-------|---|-------|
| 0 | Baton | 6 | Gun |
| 1 | Plier | 7 | Bullet |
| 2 | Hammer | 8 | Sprayer |
| 3 | Powerbank | 9 | HandCuffs |
| 4 | Scissors | 10 | Knife |
| 5 | Wrench | 11 | Lighter |

---

## ⚙️ Installation

```bash
git clone https://github.com/FALLEN-01/xray_screening.git
cd xray_screening
pip install -r requirements.txt
```

> For **Google Colab**, enable a GPU runtime before running training.

---

## 🚀 Usage

### Full Pipeline (Train → Evaluate → Visualize)

```bash
python run_pipeline.py
```

This will:
1. Train the YOLOv8 detector on the SPXray dataset
2. Evaluate on the test split — reports **mAP50** and **mAP50-95**
3. Write annotated predictions with class labels and bounding boxes to `outputs/visualizations/`

### Skip Training (Use Existing Checkpoint)

```bash
python run_pipeline.py --skip-train
```

The best checkpoint is saved at:
```text
outputs/checkpoints/detector/weights/best.pt
```

### Interactive Demo Only

```bash
python run_pipeline.py --demo-only
```

Launches a **Gradio web interface** — upload any X-ray scan image to see:
- Detected prohibited item classes
- Per-class confidence scores
- Bounding box overlays

---

## 📊 Results

Training was conducted across multiple detector iterations (`detector-1` through `detector-4`), with checkpoints and evaluation outputs saved per run under `outputs/checkpoints/`.

Metrics reported on the SPXray test split:
- **mAP50** — mean Average Precision at IoU threshold 0.50
- **mAP50-95** — mean Average Precision averaged over IoU thresholds 0.50–0.95

---

## 🛠️ Configuration

All training hyperparameters are controlled via [`config.yaml`](config.yaml):

```yaml
# Key parameters (see config.yaml for full list)
epochs: ...
batch_size: ...
img_size: ...
model: yolov8n.pt
```

---

## 🧠 Model Architecture

- **Backbone**: YOLOv8n (pretrained on COCO)
- **Head**: YOLO detection head fine-tuned for 12 SPXray classes
- **Loss**: YOLO composite loss (box + cls + dfl)
- **Augmentation**: Albumentations pipeline + synthetic data generation
- **Explainability**: Grad-CAM saliency maps via `pytorch-grad-cam`

---

## 📋 Requirements

| Package | Version |
|---------|---------|
| torch | ≥ 2.0.0 |
| torchvision | ≥ 0.15.0 |
| ultralytics | ≥ 8.2.0 |
| gradio | ≥ 4.7.1 |
| albumentations | ≥ 1.3.1 |
| grad-cam | ≥ 1.4.8 |
| opencv-python | ≥ 4.8.1 |
| tensorboard | ≥ 2.14.0 |

See [`requirements.txt`](requirements.txt) for the full pinned list.

---

## 📄 License

This project is for academic and research purposes.

---

*Built with ❤️ using YOLOv8, PyTorch, and Gradio.*
