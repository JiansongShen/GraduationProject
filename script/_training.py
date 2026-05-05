"""Shared training utilities for segmentation models.

Provides common training loop, checkpoint management, and optimizer/scheduler builders.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

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
            loss, _, _ = combined_loss_with_parts(outputs, batch_labels, **lk)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            progress.set_postfix({"loss": f"{epoch_loss / max(num_batches, 1):.4f}"})

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

    avg_dice = total_dice / max(num_batches, 1)
    avg_loss = total_loss / max(num_batches, 1)
    avg_primary = total_primary / max(num_batches, 1)
    avg_aux = total_aux / max(num_batches, 1)
    writer.add_scalar("Loss/val", avg_loss, epoch)
    writer.add_scalar("Dice/val", avg_dice, epoch)
    logging.info("Validation - Loss: %.4f, Dice: %.4f", avg_loss, avg_dice)

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
            },
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
                "label_suffix": dataset.cfg.label_suffix,
            },
        }

    report["dataset_summary"] = {
        "num_eval_cases": num_cases,
        "patch_sampling_mode": dataset.patch_sampling_mode,
        "patches_per_volume": dataset.patches_per_volume,
        "target_spacing": list(dataset.target_spacing),
        "origin_suffix": dataset.cfg.origin_suffix,
        "label_suffix": dataset.cfg.label_suffix,
        "eval_case_names": [Path(case.image_path).name for case in dataset.cases],
    }

    if report_dir is not None:
        save_eval_report(report_dir=report_dir, epoch=epoch, report=report)

    return report
