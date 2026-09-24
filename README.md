# X-Ray Prohibited-Item Object Detector

This project trains a 12-class YOLO object detector on the SPXray dataset.
Each image can contain multiple prohibited items, and each item is labeled
with a bounding box.

## Dataset layout

The repository expects the YOLO layout already present under `data/`:

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

The twelve classes are Baton, Plier, Hammer, Powerbank, Scissors, Wrench,
Gun, Bullet, Sprayer, HandCuffs, Knife, and Lighter.

## Install

```bash
pip install -r requirements.txt
```

For Google Colab, enable a GPU before training.

## Train, evaluate, and visualize

```bash
python run_pipeline.py
```

The pipeline:

1. Trains a pretrained YOLO detector.
2. Evaluates it on the test split and reports mAP50 and mAP50-95.
3. Writes annotated test predictions with class labels and bounding boxes.

To reuse an existing checkpoint:

```bash
python run_pipeline.py --skip-train
```

The detector checkpoint is saved at:

```text
outputs/checkpoints/detector/weights/best.pt
```

Annotated predictions and validation plots are written under
`outputs/visualizations/`.

## Interactive demo

After training:

```bash
python run_pipeline.py --demo-only
```

Upload an X-ray image to see detected classes, confidence scores, and
bounding boxes.
