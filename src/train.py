"""
train.py -- Full training loop with:
  * Focal Loss (handles class imbalance at loss level)
  * WeightedRandomSampler (handles imbalance at sampling level)
  * Two-phase fine-tuning (Phase 1: frozen backbone -> Phase 2: full)
  * Cosine annealing LR scheduler with linear warmup
  * Early stopping on val F1 (prohibited-item class)
  * TensorBoard + CSV logging
  * Best-model checkpointing

Usage:
    python src/train.py                        # uses config.yaml
    python src/train.py --config config.yaml
    python src/train.py --demo                 # generate synthetic data first
"""
import os
import sys
import csv
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

# Add parent dir so imports work from any cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (load_config, get_device, set_seed,
                       setup_logger, save_checkpoint)
from src.model import build_model
from src.dataset import build_splits, get_dataloaders


# --- Focal Loss --------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss for binary/multi-class imbalanced datasets.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 num_classes: int = 2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)            # (B,)
        pt = torch.exp(-ce_loss)                      # probability of true class
        focal_term = (1 - pt) ** self.gamma
        alpha_t = torch.where(targets == 1,
                              torch.tensor(self.alpha, device=logits.device),
                              torch.tensor(1 - self.alpha, device=logits.device))
        loss = alpha_t * focal_term * ce_loss
        return loss.mean()


# --- LR warmup scheduler -----------------------------------------------------

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr

    def step(self, epoch):
        import math
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


# --- One-epoch helpers -------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, device, is_train: bool):
    model.train() if is_train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    ctx = torch.enable_grad if is_train else torch.no_grad
    with ctx():
        for batch in tqdm(loader, desc="train" if is_train else "eval ",
                          leave=False, ncols=80):
            images, labels, _ = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    n = len(all_labels)
    avg_loss = total_loss / n
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / n
    # Metrics with zero_division=0 to avoid warnings on early epochs
    prec = precision_score(all_labels, all_preds,
                           pos_label=1, zero_division=0)
    rec  = recall_score(all_labels, all_preds,
                        pos_label=1, zero_division=0)
    f1   = f1_score(all_labels, all_preds,
                    pos_label=1, zero_division=0)
    return avg_loss, acc, prec, rec, f1


# --- Main training function --------------------------------------------------

def train(cfg: dict):
    set_seed(42)
    device = get_device(cfg)
    log_dir  = cfg["outputs"]["log_dir"]
    ckpt_dir = cfg["outputs"]["checkpoint_dir"]
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    logger  = setup_logger("train", log_dir)
    writer  = SummaryWriter(log_dir=log_dir)

    # -- Data -----------------------------------------------------------------
    csv_paths = build_splits(
        raw_root      = cfg["dataset"]["root"],
        processed_dir = cfg["dataset"]["processed"],
        train_ratio   = cfg["dataset"]["train_ratio"],
        val_ratio     = cfg["dataset"]["val_ratio"],
    )
    loaders = get_dataloaders(
        csv_paths            = csv_paths,
        batch_size           = cfg["training"]["batch_size"],
        image_size           = cfg["dataset"]["image_size"],
        use_weighted_sampler = cfg["training"]["use_weighted_sampler"],
    )

    # -- Model ----------------------------------------------------------------
    model = build_model(cfg).to(device)

    # -- Loss -----------------------------------------------------------------
    criterion = FocalLoss(
        alpha      = cfg["training"]["focal_loss_alpha"],
        gamma      = cfg["training"]["focal_loss_gamma"],
        num_classes= cfg["model"]["num_classes"],
    )

    # -- Optimizer & Scheduler ------------------------------------------------
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = cfg["training"]["lr"],
        weight_decay = cfg["training"]["weight_decay"],
    )
    scheduler = WarmupCosineScheduler(
        optimizer     = optimizer,
        warmup_epochs = cfg["training"]["warmup_epochs"],
        total_epochs  = cfg["training"]["epochs"],
        base_lr       = cfg["training"]["lr"],
    )

    # -- Phase 1: freeze backbone ----------------------------------------------
    model.freeze_backbone()

    # -- CSV log ---------------------------------------------------------------
    csv_log_path = os.path.join(log_dir, "training_log.csv")
    csv_fields   = ["epoch", "phase", "lr",
                    "train_loss", "train_acc", "train_prec",
                    "train_rec",  "train_f1",
                    "val_loss",   "val_acc",   "val_prec",
                    "val_rec",    "val_f1"]
    with open(csv_log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    best_val_f1 = 0.0
    patience_counter = 0
    freeze_epochs    = cfg["model"]["freeze_epochs"]
    total_epochs     = cfg["training"]["epochs"]
    patience         = cfg["training"]["early_stop_patience"]

    logger.info("=" * 60)
    logger.info("  X-Ray Baggage Screening -- Training Start")
    logger.info("=" * 60)

    for epoch in range(total_epochs):

        # Switch to Phase 2 after freeze_epochs
        if epoch == freeze_epochs:
            model.unfreeze_backbone()
            # Lower LR for full fine-tuning
            for pg in optimizer.param_groups:
                pg["lr"] = cfg["training"]["lr"] * 0.1
            optimizer.add_param_group({
                "params": model.features.parameters(),
                "lr": cfg["training"]["lr"] * 0.01,
            })

        lr = scheduler.step(epoch)
        phase = "freeze" if epoch < freeze_epochs else "finetune"

        # Train
        t0 = time.time()
        tr_loss, tr_acc, tr_prec, tr_rec, tr_f1 = run_epoch(
            model, loaders["train"], criterion, optimizer, device, is_train=True)

        # Validate
        vl_loss, vl_acc, vl_prec, vl_rec, vl_f1 = run_epoch(
            model, loaders["val"], criterion, optimizer, device, is_train=False)

        elapsed = time.time() - t0

        # Log
        logger.info(
            f"Epoch {epoch+1:03d}/{total_epochs} [{phase:8s}] "
            f"lr={lr:.2e}  "
            f"Train  loss={tr_loss:.4f} acc={tr_acc:.3f} "
            f"P={tr_prec:.3f} R={tr_rec:.3f} F1={tr_f1:.3f}  |  "
            f"Val    loss={vl_loss:.4f} acc={vl_acc:.3f} "
            f"P={vl_prec:.3f} R={vl_rec:.3f} F1={vl_f1:.3f}  "
            f"({elapsed:.1f}s)"
        )

        # TensorBoard
        writer.add_scalars("Loss",      {"train": tr_loss, "val": vl_loss}, epoch)
        writer.add_scalars("Accuracy",  {"train": tr_acc,  "val": vl_acc},  epoch)
        writer.add_scalars("F1",        {"train": tr_f1,   "val": vl_f1},   epoch)
        writer.add_scalars("Precision", {"train": tr_prec, "val": vl_prec}, epoch)
        writer.add_scalars("Recall",    {"train": tr_rec,  "val": vl_rec},  epoch)
        writer.add_scalar("LR", lr, epoch)

        # CSV
        with open(csv_log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writerow({
                "epoch": epoch + 1, "phase": phase, "lr": lr,
                "train_loss": tr_loss, "train_acc": tr_acc,
                "train_prec": tr_prec, "train_rec": tr_rec, "train_f1": tr_f1,
                "val_loss": vl_loss,   "val_acc": vl_acc,
                "val_prec": vl_prec,   "val_rec": vl_rec,   "val_f1": vl_f1,
            })

        # Save best model (primary metric: val recall on prohibited class)
        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            patience_counter = 0
            save_checkpoint({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_f1": best_val_f1,
                "val_recall": vl_rec,
            }, cfg["outputs"]["best_model"])
            logger.info(f"  [BEST] Model saved  (val_f1={best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1} "
                            f"(no improvement for {patience} epochs).")
                break

    writer.close()
    logger.info(f"Training complete. Best val F1 = {best_val_f1:.4f}")
    logger.info(f"Best model -> {cfg['outputs']['best_model']}")


# --- Entry point -------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--demo", action="store_true",
                        help="Generate synthetic data before training")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.demo or cfg["dataset"]["synthetic_demo"]:
        from src.synthetic_data import generate_synthetic_dataset
        print("[Demo] Generating synthetic SIXray-like dataset...")
        generate_synthetic_dataset(
            root       = cfg["dataset"]["root"],
            n_positive = cfg["dataset"]["synthetic_pos"],
            n_negative = cfg["dataset"]["synthetic_neg"],
            image_size = cfg["dataset"]["image_size"],
        )

    train(cfg)
