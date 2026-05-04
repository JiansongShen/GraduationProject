"""Shared training utilities for segmentation models.

Provides common training loop, checkpoint management, and optimizer/scheduler builders.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.MedicalPatchDataset import MedicalPatchDataset
from script.eval_utils import align_target_shape, combined_loss_with_parts


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


def train_one_epoch(
    model: nn.Module,
    dataset: MedicalPatchDataset,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    batch_size: int,
    log_interval: int = 10,
) -> float:
    """Train for one epoch on case-based dataset."""
    model.train()
    epoch_loss = 0.0
    num_batches = 0

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
            loss, _, _ = combined_loss_with_parts(outputs, batch_labels)
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
) -> float:
    """Run validation and return average Dice score."""
    model.eval()
    total_dice = 0.0
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for sample_idx in range(len(dataset)):
            images, labels = dataset.get_patches(sample_idx, sampling_mode="sequential")
            if labels is None:
                continue

            for patch_start in range(0, int(images.shape[0]), batch_size):
                patch_end = min(patch_start + batch_size, int(images.shape[0]))
                batch_images = images[patch_start:patch_end].to(device)
                batch_labels = labels[patch_start:patch_end].float().to(device)

                outputs = model(batch_images)
                batch_labels = align_target_shape(outputs, batch_labels)
                loss = (combined_loss_with_parts(outputs, batch_labels)[0])
                dice = 1.0 - combined_loss_with_parts(outputs, batch_labels)[2]

                total_loss += loss.item()
                total_dice += dice.item()
                num_batches += 1

    avg_dice = total_dice / max(num_batches, 1)
    avg_loss = total_loss / max(num_batches, 1)
    writer.add_scalar("Loss/val", avg_loss, epoch)
    writer.add_scalar("Dice/val", avg_dice, epoch)
    logging.info("Validation - Loss: %.4f, Dice: %.4f", avg_loss, avg_dice)
    return avg_dice
