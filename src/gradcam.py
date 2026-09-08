"""
gradcam.py -- Grad-CAM visualizations for the ResNet-50 X-ray classifier.
Generates side-by-side: Original | Heatmap | Overlay
Saves PNG grid to outputs/visualizations/

Usage:
    python src/gradcam.py
    python src/gradcam.py --config config.yaml --checkpoint outputs/checkpoints/best_model.pth
    python src/gradcam.py --image path/to/scan.jpg    # single image mode
"""
import os
import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, get_device, load_checkpoint
from src.model import build_model
from src.dataset import build_splits, XRayDataset, get_transforms

CLASS_NAMES = ["Safe", "Prohibited"]
LABEL_COLOR = {"Safe": "#4CAF50", "Prohibited": "#F44336"}


# --- Single image Grad-CAM ---------------------------------------------------

def gradcam_single(model, image_path: str, device,
                   image_size: int = 224, alpha: float = 0.5):
    """
    Returns (original_rgb, heatmap, overlay, pred_class, confidence).
    """
    # Load & preprocess
    img_pil = Image.open(image_path).convert("RGB")
    img_rgb = np.array(img_pil.resize((image_size, image_size),
                                       Image.BILINEAR)) / 255.0

    transform = get_transforms("test", image_size)
    tensor = transform(image=(img_rgb * 255).astype(np.uint8))["image"]
    input_tensor = tensor.unsqueeze(0).to(device)

    # Grad-CAM
    target_layers = model.get_gradcam_target_layers()
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=None)          # targets=None -> predicted class
        grayscale_cam = grayscale_cam[0]           # (H, W)

    # Overlay
    overlay = show_cam_on_image(
        img_rgb.astype(np.float32), grayscale_cam,
        use_rgb=True, colormap=cv2.COLORMAP_JET, image_weight=alpha)

    # Prediction
    with torch.no_grad():
        logits = model(input_tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        pred   = probs.argmax().item()
        conf   = probs[pred].item()

    return (img_rgb, grayscale_cam, overlay,
            CLASS_NAMES[pred], conf)


# --- Batch visualization -----------------------------------------------------

def visualize_batch(model, loader, device, save_dir: str,
                    n_samples: int = 16,
                    image_size: int = 224, alpha: float = 0.5):
    """
    Randomly samples n_samples from loader, runs Grad-CAM,
    saves a grid PNG for both correctly and incorrectly classified images.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    samples = []  # (image_tensor, label, path)
    for batch in loader:
        imgs, labels, paths = batch
        for i in range(len(imgs)):
            samples.append((imgs[i], labels[i].item(), paths[i]))
        if len(samples) >= n_samples * 3:
            break

    import random
    random.shuffle(samples)
    selected = samples[:n_samples]

    results = []
    for tensor, true_label, path in selected:
        img_pil = Image.open(path).convert("RGB")
        img_rgb = np.array(img_pil.resize((image_size, image_size))) / 255.0
        input_t = tensor.unsqueeze(0).to(device)

        target_layers = model.get_gradcam_target_layers()
        with GradCAM(model=model, target_layers=target_layers) as cam:
            gc = cam(input_tensor=input_t, targets=None)[0]

        overlay = show_cam_on_image(
            img_rgb.astype(np.float32), gc, use_rgb=True,
            colormap=cv2.COLORMAP_JET, image_weight=alpha)

        with torch.no_grad():
            logits = model(input_t)
            probs  = torch.softmax(logits, dim=1)[0]
            pred   = probs.argmax().item()
            conf   = probs[pred].item()

        results.append({
            "original": img_rgb,
            "heatmap":  gc,
            "overlay":  overlay,
            "true":     CLASS_NAMES[true_label],
            "pred":     CLASS_NAMES[pred],
            "conf":     conf,
            "correct":  (pred == true_label),
        })

    _save_grid(results, save_dir, n_samples, image_size)


def _save_grid(results, save_dir, n_samples, image_size):
    cols = 3   # original | heatmap | overlay
    rows = min(n_samples, len(results))
    fig_h = rows * 2.5
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, fig_h))
    if rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle("Grad-CAM Visualizations -- X-Ray Baggage Screening",
                 fontsize=13, fontweight="bold", y=1.01)

    col_titles = ["Original X-Ray", "Grad-CAM Heatmap", "Overlay"]
    for ax, t in zip(axes[0], col_titles):
        ax.set_title(t, fontsize=10, fontweight="bold")

    for r, res in enumerate(results[:rows]):
        # Original
        axes[r, 0].imshow(res["original"])
        # Heatmap
        axes[r, 1].imshow(res["heatmap"], cmap="jet", vmin=0, vmax=1)
        # Overlay
        axes[r, 2].imshow(res["overlay"])

        for c in range(cols):
            axes[r, c].axis("off")

        # Label annotation on overlay
        color = "#4CAF50" if res["correct"] else "#F44336"
        label = (f"GT:{res['true']}  Pred:{res['pred']}\n"
                 f"Conf:{res['conf']:.1%}")
        axes[r, 2].text(
            0.02, 0.98, label,
            transform=axes[r, 2].transAxes,
            fontsize=7, va="top", color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc=color, alpha=0.85))

    plt.tight_layout()
    path = os.path.join(save_dir, "gradcam_grid.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[GradCAM] Grid saved -> {path}")

    # Separate correct vs. incorrect
    correct = [r for r in results if r["correct"]]
    wrong   = [r for r in results if not r["correct"]]
    print(f"[GradCAM] {len(correct)} correct, {len(wrong)} misclassified "
          f"out of {len(results)} samples.")


# --- Main --------------------------------------------------------------------

def run_gradcam(cfg: dict, ckpt_path: str, single_image: str = None):
    device  = get_device(cfg)
    viz_dir = cfg["outputs"]["viz_dir"]
    os.makedirs(viz_dir, exist_ok=True)

    model = build_model(cfg).to(device)
    load_checkpoint(ckpt_path, model, device=device)

    if single_image:
        orig, heatmap, overlay, pred_class, conf = gradcam_single(
            model, single_image, device,
            image_size=cfg["dataset"]["image_size"],
            alpha=cfg["gradcam"]["alpha"],
        )
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, img, title in zip(axes,
                                   [orig, heatmap, overlay],
                                   ["Original", "Heatmap", "Overlay"]):
            ax.imshow(img, cmap="jet" if title == "Heatmap" else None)
            ax.set_title(title); ax.axis("off")
        color = LABEL_COLOR.get(pred_class, "grey")
        fig.suptitle(f"Prediction: {pred_class}  ({conf:.1%})",
                     fontsize=13, fontweight="bold", color=color)
        out = os.path.join(viz_dir, "gradcam_single.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[GradCAM] {pred_class} ({conf:.1%})  ->  {out}")
        return

    # Batch mode from test set
    csv_paths = build_splits(
        raw_root      = cfg["dataset"]["root"],
        processed_dir = cfg["dataset"]["processed"],
        train_ratio   = cfg["dataset"]["train_ratio"],
        val_ratio     = cfg["dataset"]["val_ratio"],
    )
    from src.dataset import get_dataloaders
    loaders = get_dataloaders(
        csv_paths={"test": csv_paths["test"]},
        batch_size=cfg["training"]["batch_size"],
        image_size=cfg["dataset"]["image_size"],
        use_weighted_sampler=False,
    )
    visualize_batch(
        model, loaders["test"], device,
        save_dir   = viz_dir,
        n_samples  = cfg["gradcam"]["num_visualize"],
        image_size = cfg["dataset"]["image_size"],
        alpha      = cfg["gradcam"]["alpha"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--image",      default=None, help="Single image path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt = args.checkpoint or cfg["outputs"]["best_model"]
    run_gradcam(cfg, ckpt, single_image=args.image)
