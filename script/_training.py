"""Shared training utilities for segmentation models.

Provides common training loop, checkpoint management, and optimizer/scheduler builders.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from core.config import Config, TrainConfig
from data.MedicalPatchDataset import MedicalPatchDataset
from script.eval_utils import align_target_shape, combined_loss_with_parts, dice_loss


def build_optimizer(model: nn.Module, lr: float, weight_decay: float, optimizer: str = "adamw") -> torch.optim.Optimizer:
    """Build optimizer from configuration."""
    optimizer = optimizer.lower()
    if optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {optimizer}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler: str,
    total_epochs: int,
    warmup_epochs: int = 0,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Build learning rate scheduler."""
    scheduler = scheduler.lower()
    if scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6
        )
    elif scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=total_epochs // 3, gamma=0.1)
    elif scheduler == "none":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    raise ValueError(f"Unsupported scheduler: {scheduler}")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metric: float,
    checkpoint_dir: Path,
    is_best: bool = False,
    filename: str = "checkpoint.pth",
) -> None:
    """Save model checkpoint with optional best model."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "metric": metric},
        checkpoint_dir / filename,
    )
    logging.info("Saved checkpoint: %s", checkpoint_dir / filename)
    if is_best:
        torch.save(
            {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "metric": metric},
            checkpoint_dir / "best.pth",
        )
        logging.info("Saved best model (metric=%.4f)", metric)


def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> tuple[int, float]:
    """Load checkpoint and return epoch and metric."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint.get("metric", 0.0)


def _default_loss_kwargs() -> dict[str, float | str]:
    cfg = TrainConfig()
    return cfg.segmentation_loss_kwargs()


def build_eval_report_dir(base_dir: Path, run_started_at: datetime | None = None) -> Path:
    timestamp = run_started_at or datetime.now()
    report_dir = base_dir / "eval_reports" / timestamp.strftime("%Y-%m-%d") / f"run_{timestamp.strftime('%H-%M-%S')}"
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir


def save_eval_report(
    report_dir: Path,
    epoch: int,
    report: dict[str, Any],
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"eval_epoch_{epoch + 1:04d}.json"
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    report_path.write_text(payload, encoding="utf-8")
    (report_dir / "latest.json").write_text(payload, encoding="utf-8")
    logging.info("Saved eval report to %s", report_path)
    return report_path


def _update_confusion_counts(
    prob: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    pred = (prob >= threshold)
    target_bool = target >= 0.5

    tp = float((pred & target_bool).sum().item())
    tn = float((~pred & ~target_bool).sum().item())
    fp = float((pred & ~target_bool).sum().item())
    fn = float((~pred & target_bool).sum().item())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _metrics_from_confusion_counts(tp: float, tn: float, fp: float, fn: float) -> dict[str, float]:
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "iou": iou,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _update_auc_histograms(
    prob: torch.Tensor,
    target: torch.Tensor,
    pos_hist: np.ndarray,
    neg_hist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    prob_np = prob.detach().float().cpu().clamp(0.0, 1.0).view(-1).numpy()
    target_np = (target.detach().float().cpu().view(-1).numpy() >= 0.5)
    if prob_np.size == 0:
        return pos_hist, neg_hist

    num_bins = int(pos_hist.shape[0])
    bin_indices = np.minimum((prob_np * num_bins).astype(np.int64), num_bins - 1)
    pos_bins = bin_indices[target_np]
    neg_bins = bin_indices[~target_np]
    if pos_bins.size > 0:
        np.add.at(pos_hist, pos_bins, 1.0)
    if neg_bins.size > 0:
        np.add.at(neg_hist, neg_bins, 1.0)
    return pos_hist, neg_hist


def _update_threshold_confusions(
    prob: torch.Tensor,
    target: torch.Tensor,
    thresholds: list[float],
    confusion_by_threshold: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    target_bool = target >= 0.5
    for threshold in thresholds:
        pred = prob >= threshold
        key = f"{threshold:.2f}"
        confusion = confusion_by_threshold[key]
        confusion["tp"] += float((pred & target_bool).sum().item())
        confusion["tn"] += float((~pred & ~target_bool).sum().item())
        confusion["fp"] += float((pred & ~target_bool).sum().item())
        confusion["fn"] += float((~pred & target_bool).sum().item())
    return confusion_by_threshold


def _histogram_auc(pos_hist: np.ndarray, neg_hist: np.ndarray) -> float:
    pos_total = float(pos_hist.sum())
    neg_total = float(neg_hist.sum())
    if pos_total <= 0.0 or neg_total <= 0.0:
        return 0.5

    auc_numerator = 0.0
    neg_seen_lower = 0.0
    for bin_idx in range(len(pos_hist)):
        pos_count = float(pos_hist[bin_idx])
        neg_count = float(neg_hist[bin_idx])
        if pos_count > 0.0:
            auc_numerator += pos_count * neg_seen_lower
            auc_numerator += 0.5 * pos_count * neg_count
        neg_seen_lower += neg_count
    return auc_numerator / (pos_total * neg_total)


def train_one_epoch(
    model: nn.Module,
    dataset: MedicalPatchDataset,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    batch_size: int,
    log_interval: int = 10,
    loss_kwargs: Optional[dict[str, float | str]] = None,
) -> float:
    """Train for one epoch on case-based dataset."""
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    lk = loss_kwargs if loss_kwargs is not None else _default_loss_kwargs()

    progress = tqdm(range(len(dataset)), desc=f"Epoch {epoch + 1}")
    for batch_idx in progress:
        images, labels = dataset[batch_idx]
        if labels is None:
            continue

        for patch_start in range(0, int(images.shape[0]), batch_size):
            patch_end = min(patch_start + batch_size, int(images.shape[0]))
            batch_images = images[patch_start:patch_end].to(device)
            batch_labels = labels[patch_start:patch_end].float().to(device)

            optimizer.zero_grad()
            outputs = model(batch_images)
            batch_labels = align_target_shape(outputs, batch_labels)
            loss, bce_l, t_l = combined_loss_with_parts(outputs, batch_labels, **lk)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            progress.set_postfix({"loss": f"{epoch_loss / max(num_batches, 1):.4f}, bce: {bce_l:.4f}, tver: {t_l:.4f}"})

            if num_batches % log_interval == 0:
                writer.add_scalar("Loss/train_batch", loss.item(), epoch * len(dataset) + num_batches)

    return epoch_loss / max(num_batches, 1)


def validate(
    model: nn.Module,
    dataset: MedicalPatchDataset,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    batch_size: int,
    loss_kwargs: Optional[dict[str, float | str]] = None,
    cfg: Config | None = None,
    report_dir: Path | None = None,
    is_best_so_far: bool = False,
) -> dict[str, Any]:
    """Run validation and return aggregate metrics, optionally persisting a JSON report."""
    model.eval()
    total_dice = 0.0
    total_loss = 0.0
    total_primary = 0.0
    total_aux = 0.0
    num_batches = 0
    num_cases = 0
    num_patches_total = 0
    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    threshold = 0.5
    threshold_sweep = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    confusion_by_threshold = {
        f"{thr:.2f}": {"tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0} for thr in threshold_sweep
    }
    auc_bins = 1024
    pos_hist = np.zeros(auc_bins, dtype=np.float64)
    neg_hist = np.zeros(auc_bins, dtype=np.float64)
    positive_voxel_count = 0.0
    total_voxel_count = 0.0
    lk = loss_kwargs if loss_kwargs is not None else _default_loss_kwargs()
    smooth = float(lk.get("smooth", 1e-6))

    with torch.no_grad():
        for sample_idx in range(len(dataset)):
            images, labels = dataset.get_patches(sample_idx, sampling_mode="sequential")
            if labels is None:
                continue

            num_cases += 1
            num_patches_total += int(images.shape[0])

            for patch_start in range(0, int(images.shape[0]), batch_size):
                patch_end = min(patch_start + batch_size, int(images.shape[0]))
                batch_images = images[patch_start:patch_end].to(device)
                batch_labels = labels[patch_start:patch_end].float().to(device)

                outputs = model(batch_images)
                batch_labels = align_target_shape(outputs, batch_labels)
                loss, primary_loss, aux_loss = combined_loss_with_parts(outputs, batch_labels, **lk)
                dice = 1.0 - dice_loss(outputs, batch_labels, smooth=smooth)

                total_loss += loss.item()
                total_primary += float(primary_loss.item())
                total_aux += float(aux_loss.item())
                total_dice += dice.item()
                num_batches += 1

                detached_outputs = outputs.detach().float()
                detached_labels = batch_labels.detach().float()
                counts = _update_confusion_counts(detached_outputs, detached_labels, threshold=threshold)
                tp += counts["tp"]
                tn += counts["tn"]
                fp += counts["fp"]
                fn += counts["fn"]
                confusion_by_threshold = _update_threshold_confusions(
                    detached_outputs,
                    detached_labels,
                    threshold_sweep,
                    confusion_by_threshold,
                )
                pos_hist, neg_hist = _update_auc_histograms(detached_outputs, detached_labels, pos_hist, neg_hist)
                positive_voxel_count += float((detached_labels >= 0.5).sum().item())
                total_voxel_count += float(detached_labels.numel())

    avg_dice = total_dice / max(num_batches, 1)
    avg_loss = total_loss / max(num_batches, 1)
    avg_primary = total_primary / max(num_batches, 1)
    avg_aux = total_aux / max(num_batches, 1)
    voxel_auc = _histogram_auc(pos_hist, neg_hist)
    voxel_metrics = _metrics_from_confusion_counts(tp, tn, fp, fn)
    threshold_metrics: dict[str, dict[str, float]] = {}
    for threshold_key, confusion in confusion_by_threshold.items():
        metrics = _metrics_from_confusion_counts(
            confusion["tp"],
            confusion["tn"],
            confusion["fp"],
            confusion["fn"],
        )
        total_at_threshold = confusion["tp"] + confusion["tn"] + confusion["fp"] + confusion["fn"]
        metrics["threshold"] = float(threshold_key)
        metrics["pred_positive_ratio"] = (
            (confusion["tp"] + confusion["fp"]) / total_at_threshold if total_at_threshold > 0.0 else 0.0
        )
        threshold_metrics[threshold_key] = metrics
    best_threshold_key = max(threshold_metrics, key=lambda key: threshold_metrics[key]["f1"]) if threshold_metrics else "0.50"
    best_threshold_metrics = threshold_metrics.get(best_threshold_key, voxel_metrics)
    pred_positive_ratio = (tp + fp) / total_voxel_count if total_voxel_count > 0.0 else 0.0
    gt_positive_ratio = positive_voxel_count / total_voxel_count if total_voxel_count > 0.0 else 0.0
    writer.add_scalar("Loss/val", avg_loss, epoch)
    writer.add_scalar("Dice/val", avg_dice, epoch)
    writer.add_scalar("AUC/val_voxel", voxel_auc, epoch)
    writer.add_scalar("F1/val_voxel", voxel_metrics["f1"], epoch)
    writer.add_scalar("F1/val_voxel_best_threshold", best_threshold_metrics["f1"], epoch)
    logging.info(
        "Validation - Loss: %.4f, Dice: %.4f, AUC: %.4f, F1@0.50: %.4f, BestF1: %.4f@thr=%.2f, Recall@best: %.4f, PredPos@best: %.6f",
        avg_loss,
        avg_dice,
        voxel_auc,
        voxel_metrics["f1"],
        best_threshold_metrics["f1"],
        best_threshold_metrics["threshold"],
        best_threshold_metrics["recall"],
        best_threshold_metrics["pred_positive_ratio"],
    )

    report: dict[str, Any] = {
        "meta": {
            "task": "segmentation_eval",
            "model_name": model.__class__.__name__,
            "report_version": "1.0",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "epoch": epoch + 1,
            "is_best_so_far": is_best_so_far,
        },
        "metrics": {
            "loss": {
                "mean_total": avg_loss,
                "mean_primary": avg_primary,
                "mean_aux": avg_aux,
            },
            "segmentation": {
                "mean_dice": avg_dice,
                "voxel_auc": voxel_auc,
                "voxel_auc_method": "histogram",
                "voxel_auc_bins": auc_bins,
                "voxel_f1": voxel_metrics["f1"],
                "voxel_iou": voxel_metrics["iou"],
                "voxel_accuracy": voxel_metrics["accuracy"],
                "voxel_precision": voxel_metrics["precision"],
                "voxel_recall": voxel_metrics["recall"],
                "voxel_specificity": voxel_metrics["specificity"],
                "threshold": threshold,
                "pred_positive_ratio": pred_positive_ratio,
                "gt_positive_ratio": gt_positive_ratio,
                "best_f1_threshold": best_threshold_metrics["threshold"],
                "best_f1": best_threshold_metrics["f1"],
                "best_f1_recall": best_threshold_metrics["recall"],
                "best_f1_precision": best_threshold_metrics["precision"],
                "best_f1_pred_positive_ratio": best_threshold_metrics["pred_positive_ratio"],
            },
            "confusion_matrix": {
                "tp": voxel_metrics["tp"],
                "tn": voxel_metrics["tn"],
                "fp": voxel_metrics["fp"],
                "fn": voxel_metrics["fn"],
            },
            "threshold_sweep": threshold_metrics,
            "sampling": {
                "num_cases": num_cases,
                "num_batches": num_batches,
                "num_patches_total": num_patches_total,
            },
        },
    }

    if cfg is not None:
        report["config_snapshot"] = {
            "device": cfg.device,
            "seed": cfg.seed,
            "eval_every_n_epochs": cfg.eval_every_n_epochs,
            "train": {
                "batch_size": cfg.train.batch_size,
                "epochs": cfg.train.epochs,
                "learning_rate": cfg.train.learning_rate,
                "weight_decay": cfg.train.weight_decay,
                "optimizer": cfg.train.optimizer,
                "scheduler": cfg.train.scheduler,
                "warmup_epochs": cfg.train.warmup_epochs,
                "loss": cfg.train.segmentation_loss_kwargs(),
            },
            "inference": {
                "patch_size": list(cfg.inference.patch_size),
                "effective_size": list(cfg.inference.effective_size),
                "batch_size": cfg.inference.batch_size,
                "use_amp": cfg.inference.use_amp,
            },
            "data": {
                "patch_sampling_mode": dataset.patch_sampling_mode,
                "patches_per_volume": dataset.patches_per_volume,
                "target_spacing": list(dataset.target_spacing),
                "origin_suffix": dataset.cfg.origin_suffix,
                "label_suffixes": dataset.cfg.label_suffixes,
                "preprocess": dataset.cfg.preprocess,
            },
        }

    report["dataset_summary"] = {
        "num_eval_cases": num_cases,
        "patch_sampling_mode": dataset.patch_sampling_mode,
        "patches_per_volume": dataset.patches_per_volume,
        "target_spacing": list(dataset.target_spacing),
        "origin_suffix": dataset.cfg.origin_suffix,
        "label_suffixes": dataset.cfg.label_suffixes,
        "preprocess": dataset.cfg.preprocess,
        "eval_case_names": [Path(case.image_path).name for case in dataset.cases],
    }

    if report_dir is not None:
        save_eval_report(report_dir=report_dir, epoch=epoch, report=report)

    return report
