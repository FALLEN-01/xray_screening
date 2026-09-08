"""
evaluate.py -- Full test-set evaluation:
  * Per-class precision, recall, F1
  * Weighted + macro averages
  * Confusion matrix (raw + normalized)
  * ROC curve + AUC
  * Precision-Recall curve + Average Precision
  * Threshold sweep to maximize F1
  * Saves all plots to outputs/visualizations/
  * Exports JSON metrics report

Usage:
    python src/evaluate.py
    python src/evaluate.py --config config.yaml --checkpoint outputs/checkpoints/best_model.pth
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    f1_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, get_device, load_checkpoint
from src.model import build_model
from src.dataset import build_splits, get_dataloaders, XRayDataset, get_transforms


CLASS_NAMES = ["Safe", "Prohibited"]
PALETTE     = ["#4CAF50", "#F44336"]      # green=safe, red=prohibited


# --- Inference pass -----------------------------------------------------------

def get_predictions(model, loader, device):
    model.eval()
    all_labels, all_probs, all_preds = [], [], []
    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="Evaluating", ncols=80):
            images = images.to(device)
            logits = model(images)
            probs  = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)
            all_preds.extend(preds)
    return (np.array(all_labels),
            np.array(all_probs),
            np.array(all_preds))


# --- Plot helpers -------------------------------------------------------------

def plot_confusion_matrix(cm, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Confusion Matrix", fontsize=14, fontweight="bold")

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
    axes[0].set_title("Raw Counts")
    axes[0].set_ylabel("True Label"); axes[0].set_xlabel("Predicted Label")

    # Normalized
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1])
    axes[1].set_title("Normalized (Row %)")
    axes[1].set_ylabel("True Label"); axes[1].set_xlabel("Predicted Label")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_roc(labels, probs, save_path):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc     = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#E91E63", lw=2,
            label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve -- Prohibited Item Detection")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")
    return roc_auc


def plot_pr_curve(labels, probs, save_path):
    prec, rec, thresholds = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec, prec, color="#2196F3", lw=2,
            label=f"PR Curve (AP = {ap:.4f})")
    ax.axhline(labels.mean(), ls="--", color="grey", lw=1,
               label="Baseline (prevalence)")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve -- Prohibited Item Detection")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")
    return ap, prec, rec, thresholds


def plot_threshold_sweep(labels, probs, save_path):
    thresholds = np.linspace(0.01, 0.99, 200)
    f1s, precs, recs = [], [], []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1s.append(f1_score(labels, preds, pos_label=1, zero_division=0))
        precs.append(labels[preds == 1].mean() if preds.sum() > 0 else 0.0)
        recs.append((preds[labels == 1] == 1).mean() if (labels == 1).sum() > 0 else 0.0)

    best_idx = np.argmax(f1s)
    best_t   = thresholds[best_idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, f1s,   color="#9C27B0", lw=2, label="F1")
    ax.plot(thresholds, precs, color="#2196F3", lw=1.5, ls="--", label="Precision")
    ax.plot(thresholds, recs,  color="#FF5722", lw=1.5, ls="--", label="Recall")
    ax.axvline(best_t, color="black", ls=":", lw=1.5,
               label=f"Best threshold = {best_t:.2f}")
    ax.set_xlabel("Decision Threshold"); ax.set_ylabel("Score")
    ax.set_title("Threshold Sweep -- F1 / Precision / Recall")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")
    return best_t, f1s[best_idx]


def plot_per_class_metrics(report_dict, save_path):
    classes = [c for c in report_dict if c in CLASS_NAMES]
    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, cls in enumerate(classes):
        vals = [report_dict[cls][m] for m in metrics]
        ax.bar(x + i * width - width / 2, vals, width,
               label=cls, color=PALETTE[i], alpha=0.85)

    ax.set_xticks(x); ax.set_xticklabels(["Precision", "Recall", "F1"])
    ax.set_ylim(0, 1.1); ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# --- Main --------------------------------------------------------------------

def evaluate(cfg: dict, ckpt_path: str):
    device = get_device(cfg)
    viz_dir = cfg["outputs"]["viz_dir"]
    os.makedirs(viz_dir, exist_ok=True)

    # -- Load model -----------------------------------------------------------
    model = build_model(cfg).to(device)
    load_checkpoint(ckpt_path, model, device=device)

    # -- Test data ------------------------------------------------------------
    csv_paths = build_splits(
        raw_root      = cfg["dataset"]["root"],
        processed_dir = cfg["dataset"]["processed"],
        train_ratio   = cfg["dataset"]["train_ratio"],
        val_ratio     = cfg["dataset"]["val_ratio"],
    )
    loaders = get_dataloaders(
        csv_paths    = {"test": csv_paths["test"]},
        batch_size   = cfg["training"]["batch_size"],
        image_size   = cfg["dataset"]["image_size"],
        use_weighted_sampler=False,
    )
    test_loader = loaders["test"]

    # -- Predictions ----------------------------------------------------------
    labels, probs, preds = get_predictions(model, test_loader, device)

    # -- Classification report -------------------------------------------------
    report_str  = classification_report(labels, preds,
                                        target_names=CLASS_NAMES,
                                        digits=4)
    report_dict = classification_report(labels, preds,
                                        target_names=CLASS_NAMES,
                                        output_dict=True)
    print("\n" + "=" * 55)
    print("  CLASSIFICATION REPORT")
    print("=" * 55)
    print(report_str)

    # -- Plots ----------------------------------------------------------------
    cm      = confusion_matrix(labels, preds)
    roc_auc = plot_roc(labels, probs,
                       os.path.join(viz_dir, "roc_curve.png"))
    ap, _, _, _ = plot_pr_curve(labels, probs,
                                os.path.join(viz_dir, "pr_curve.png"))
    best_t, best_f1 = plot_threshold_sweep(
        labels, probs, os.path.join(viz_dir, "threshold_sweep.png"))

    plot_confusion_matrix(cm, os.path.join(viz_dir, "confusion_matrix.png"))
    plot_per_class_metrics(report_dict,
                           os.path.join(viz_dir, "per_class_metrics.png"))

    # -- JSON report ----------------------------------------------------------
    metrics = {
        "roc_auc":           round(float(roc_auc), 4),
        "average_precision": round(float(ap), 4),
        "best_threshold":    round(float(best_t), 4),
        "best_f1_at_thresh": round(float(best_f1), 4),
        "per_class":         report_dict,
    }
    report_path = os.path.join(viz_dir, "metrics_report.json")
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved -> {report_path}")
    print(f"ROC-AUC = {roc_auc:.4f}  |  AP = {ap:.4f}  "
          f"|  Best threshold = {best_t:.2f}  (F1={best_f1:.4f})")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt = args.checkpoint or cfg["outputs"]["best_model"]
    evaluate(cfg, ckpt)
