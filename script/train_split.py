#!/usr/bin/env python3
"""Training script for 3D medical image segmentation with configurable models."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, StepLR
from torch.utils.tensorboard import SummaryWriter
import SimpleITK as sitk

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.config import Config
from core.config_loader import load_config
from core.global_setting import SystemSetting
from data.MedicalPatchDataset import MedicalPatchDataset
from model.aneurysm.model.AttentionUnet import AttentionUnet
from script.eval_split import evaluate


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train 3D medical image segmentation model"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file (e.g., config/host.yaml)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device from config (cuda/cpu)"
    )
    return parser.parse_args()


def build_model(cfg: Config, device: torch.device) -> AttentionUnet | Callable[[Any], Any] | Any:
    """Build model based on configuration."""
    # Currently using AttentionUnet - can be extended to support multiple models
    model = AttentionUnet(
        in_ch=cfg.model.in_channels,
        out_ch=cfg.model.out_channels,
        depth=cfg.model.depth,
        norm_type=cfg.model.norm_type,
        activation=cfg.model.activation,
        dropout=cfg.model.dropout
    )
    
    model = model.to(device)
    
    # Optional: compile model for faster training (PyTorch 2.0+)
    if cfg.compile.enabled and hasattr(torch, 'compile'):
        logging.info(f"Compiling model with mode: {cfg.compile.mode}")
        model = torch.compile(model, mode=cfg.compile.mode, fullgraph=cfg.compile.fullgraph)
    
    return model


def build_optimizer(model: torch.nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """Build optimizer based on configuration."""
    if cfg.train.optimizer.lower() == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay
        )
    elif cfg.train.optimizer.lower() == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay
        )
    elif cfg.train.optimizer.lower() == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
            momentum=0.9
        )
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.train.optimizer}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Config, total_epochs: int) -> CosineAnnealingLR | StepLR | LambdaLR:
    """Build learning rate scheduler based on configuration."""
    if cfg.train.scheduler.lower() == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - cfg.train.warmup_epochs,
            eta_min=1e-6
        )
    elif cfg.train.scheduler.lower() == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=total_epochs // 3,
            gamma=0.1
        )
    elif cfg.train.scheduler.lower() == "none":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    else:
        raise ValueError(f"Unsupported scheduler: {cfg.train.scheduler}")


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Compute Dice loss for segmentation."""
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    dice_coeff = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    return 1.0 - dice_coeff


def combined_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Combined BCE + Dice loss."""
    bce = torch.nn.functional.binary_cross_entropy(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice


def align_target_shape(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match target tensor rank/shape to prediction tensor for segmentation losses."""
    if target.ndim == pred.ndim - 1:
        target = target.unsqueeze(1)

    if target.shape != pred.shape:
        raise ValueError(
            f"Prediction/target shape mismatch after alignment: pred={tuple(pred.shape)}, target={tuple(target.shape)}"
        )

    return target


def combine_to_nifti(patch_list: list[torch.Tensor], patch_shape: tuple[int, int, int], src_shape: tuple[int, int, int]) -> sitk.Image:
    """Stitch sequentially traversed prediction patches back into a volume.

    The patch order must match `MedicalPatchDataset.__getitem__`: z -> y -> x,
    non-overlapping full patches only. Because `patches_per_volume` is a cap, the
    stitched result may cover only the first part of the source volume; uncovered
    voxels remain zero.
    """
    if not patch_list:
        raise ValueError("Cannot stitch prediction result: patch_list is empty.")

    patch_d, patch_h, patch_w = patch_shape
    src_d, src_h, src_w = src_shape
    combined = torch.zeros(src_shape, dtype=torch.float32)
    patch_idx = 0

    for z in range(0, src_d - patch_d + 1, patch_d):
        for y in range(0, src_h - patch_h + 1, patch_h):
            for x in range(0, src_w - patch_w + 1, patch_w):
                if patch_idx >= len(patch_list):
                    return sitk.GetImageFromArray(combined.numpy())

                patch = patch_list[patch_idx].detach().cpu()
                if patch.ndim == 5:
                    patch = patch[0, 0]
                elif patch.ndim == 4:
                    patch = patch[0]
                if tuple(patch.shape) != patch_shape:
                    raise ValueError(f"Invalid prediction patch shape: {tuple(patch.shape)}, expected {patch_shape}")

                combined[z:z + patch_d, y:y + patch_h, x:x + patch_w] = patch.float()
                patch_idx += 1

    return sitk.GetImageFromArray(combined.numpy())


def train_one_epoch(
    model: torch.nn.Module,
    dataset: MedicalPatchDataset,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    prediction_dir: Path,
    store_single_res: bool = True,
) -> float:
    """Train for one epoch by loading one case at a time."""
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    total_volumes = len(dataset)
    logging.info("Epoch %s training started: total_volumes=%s", epoch + 1, total_volumes)

    for batch_idx in range(total_volumes):
        logging.info("Epoch %s volume %s/%s loading", epoch + 1, batch_idx + 1, total_volumes)
        _, label_src = dataset.get_src_item(batch_idx)
        images, labels = dataset[batch_idx]
        patch_list: list[torch.Tensor] = []

        if labels is None:
            logging.warning("Epoch %s volume %s/%s has no labels. Skipping.", epoch + 1, batch_idx + 1, total_volumes)
            continue

        patch_count = int(images.shape[0])
        logging.info(
            "Epoch %s volume %s/%s loaded: patches=%s image_shape=%s label_shape=%s",
            epoch + 1,
            batch_idx + 1,
            total_volumes,
            patch_count,
            tuple(images.shape),
            tuple(labels.shape),
        )

        for patch_idx in range(patch_count):
            logging.info(
                "Epoch %s volume %s/%s patch %s/%s training started",
                epoch + 1,
                batch_idx + 1,
                total_volumes,
                patch_idx + 1,
                patch_count,
            )

            patch_images = images[patch_idx:patch_idx + 1].to(device)
            patch_labels = labels[patch_idx:patch_idx + 1].float().to(device)

            optimizer.zero_grad()
            outputs = model(patch_images)
            patch_labels = align_target_shape(outputs, patch_labels)
            loss = combined_loss(outputs, patch_labels)
            loss.backward()
            optimizer.step()

            if store_single_res and batch_idx == 0:
                patch_list.append(outputs.detach().cpu())

            epoch_loss += loss.item()
            num_batches += 1
            avg_loss = epoch_loss / num_batches
            logging.info(
                "Epoch %s volume %s/%s patch %s/%s done: loss=%.6f avg_loss=%.6f global_patch_step=%s",
                epoch + 1,
                batch_idx + 1,
                total_volumes,
                patch_idx + 1,
                patch_count,
                loss.item(),
                avg_loss,
                num_batches,
            )

            if num_batches % 10 == 0:
                writer.add_scalar("Loss/train_batch", loss.item(), epoch * len(dataset) + num_batches)

        if store_single_res and batch_idx == 0:
            src_shape_tuple = tuple(int(dim) for dim in label_src.shape)
            if len(src_shape_tuple) != 3:
                raise ValueError(f"Invalid source shape: {src_shape_tuple}")
            prediction_dir.mkdir(parents=True, exist_ok=True)
            nifti = combine_to_nifti(patch_list, dataset.patch_size, src_shape_tuple)
            prediction_path = prediction_dir / f"epoch_{epoch + 1:04d}_case_{batch_idx:04d}_prediction.nii.gz"
            sitk.WriteImage(nifti, str(prediction_path))
            logging.info("Saved sample prediction to %s", prediction_path)

    return epoch_loss / max(num_batches, 1)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    dice_score: float,
    checkpoint_dir: Path,
    is_best: bool = False
) -> None:
    """Save model checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "dice_score": dice_score,
    }
    
    # Save latest checkpoint
    latest_path = checkpoint_dir / "latest.pth"
    torch.save(checkpoint, latest_path)
    logging.info(f"Saved latest checkpoint to {latest_path}")
    
    # Save best checkpoint
    if is_best:
        best_path = checkpoint_dir / "best.pth"
        torch.save(checkpoint, best_path)
        logging.info(f"Saved best checkpoint to {best_path} (Dice: {dice_score:.4f})")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> tuple[int, float]:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    logging.info(f"Loaded checkpoint from {checkpoint_path} (epoch: {checkpoint['epoch']})")
    return checkpoint["epoch"], checkpoint.get("dice_score", 0.0)


def main() -> None:
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    cfg = load_config(config_path)
    
    # Override device if specified
    if args.device:
        cfg.device = args.device
    
    # Initialize system settings (random seeds, etc.)
    SystemSetting.set_seed(cfg.seed)
    
    # Setup logging
    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "training.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info(f"Configuration loaded from: {config_path}")
    logging.info(f"Model: {cfg.model.name}")
    logging.info(f"Device: {cfg.device}")
    
    # Setup device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Build model
    logging.info("Building model...")
    model = build_model(cfg, device)
    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Build optimizer and scheduler
    optimizer = build_optimizer(model, cfg)
    
    # Setup datasets and dataloaders
    logging.info("Setting up datasets...")
    patch_size = tuple(cfg.train.patch_size)
    
    train_dataset = MedicalPatchDataset(
        cfg=cfg.data,
        patch_size=patch_size,
        seed=cfg.seed
    )
    
    # For now, use same dataset structure for validation
    # In practice, you'd want separate validation directories
    val_dataset = MedicalPatchDataset(
        cfg=cfg.data,
        patch_size=patch_size,
        seed=cfg.seed + 1
    )
    
    logging.info(f"Training samples: {len(train_dataset)}")
    logging.info(f"Validation samples: {len(val_dataset)}")
    
    # Setup TensorBoard
    writer = SummaryWriter(log_dir=log_dir / "tensorboard")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_dice = 0.0
    
    if args.resume:
        start_epoch, best_dice = load_checkpoint(args.resume, model, optimizer)
        logging.info(f"Resuming training from epoch {start_epoch}")
    
    # Training loop
    logging.info("Starting training...")
    best_dice = max(best_dice, 0.0)
    
    checkpoint_dir = Path(cfg.checkpoint.save_dir)
    prediction_dir = checkpoint_dir / "predictions"

    for epoch in range(start_epoch, cfg.train.epochs):
        # Train
        train_loss = train_one_epoch(
            model,
            train_dataset,
            optimizer,
            device,
            epoch,
            writer,
            prediction_dir=prediction_dir,
            store_single_res=True,
        )
        writer.add_scalar("Loss/train_epoch", train_loss, epoch)
        
        # Evaluate
        if (epoch + 1) % cfg.eval_interval == 0:
            val_dice = evaluate(model, val_dataset, device, epoch, writer)
            
            # Check if best model
            is_best = val_dice > best_dice
            if is_best:
                best_dice = val_dice
            
            # Save checkpoint
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                val_dice,
                checkpoint_dir,
                is_best
            )
        
        # Step scheduler (if using warmup, handle it here)
        if epoch >= cfg.train.warmup_epochs:
            # Scheduler stepping logic would go here
            pass
        
        # Periodic checkpoint saving
        if (epoch + 1) % cfg.checkpoint.save_interval == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                best_dice,
                checkpoint_dir,
                is_best=False
            )
    
    # Final evaluation
    logging.info(f"Training completed! Best Dice: {best_dice:.4f}")
    writer.close()


if __name__ == "__main__":
    main()
