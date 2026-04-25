#!/usr/bin/env python3
"""Training script for 3D medical image segmentation with configurable models."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Callable

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.config import Config
from core.config_loader import load_config
from core.global_setting import SystemSetting
from data.Dataset import MedicalPatchDataset
from model.aneurysm.model.AttentionUnet import AttentionUnet


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


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter
) -> float:
    """Train for one epoch."""
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} Training")
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        if labels is None:
            continue
            
        images = images.to(device)
        labels = labels.float().to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        labels = align_target_shape(outputs, labels)
        loss = combined_loss(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        epoch_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        avg_loss = epoch_loss / num_batches
        progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
        
        # Log to TensorBoard every 10 batches
        if batch_idx % 10 == 0:
            writer.add_scalar("Loss/train_batch", loss.item(), epoch * len(dataloader) + batch_idx)
    
    return epoch_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter
) -> float:
    """Evaluate model on validation set."""
    model.eval()
    epoch_loss = 0.0
    epoch_dice = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} Validation")
    
    for images, labels in progress_bar:
        if labels is None:
            continue
            
        images = images.to(device)
        labels = labels.float().to(device)
        
        # Forward pass
        outputs = model(images)
        labels = align_target_shape(outputs, labels)
        loss = combined_loss(outputs, labels)
        
        # Compute Dice score
        pred_binary = (outputs > 0.5).float()
        dice = 1.0 - dice_loss(pred_binary, labels)
        
        # Update metrics
        epoch_loss += loss.item()
        epoch_dice += dice.item()
        num_batches += 1
        
        # Update progress bar
        avg_loss = epoch_loss / num_batches
        avg_dice = epoch_dice / num_batches
        progress_bar.set_postfix({"loss": f"{avg_loss:.4f}", "dice": f"{avg_dice:.4f}"})
    
    avg_loss = epoch_loss / max(num_batches, 1)
    avg_dice = epoch_dice / max(num_batches, 1)
    
    # Log to TensorBoard
    writer.add_scalar("Loss/val", avg_loss, epoch)
    writer.add_scalar("Dice/val", avg_dice, epoch)
    
    logging.info(f"Validation - Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}")
    
    return avg_dice


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
        patches_per_volume=cfg.train.max_patches_per_volume or 1,
        seed=cfg.seed
    )
    
    # For now, use same dataset structure for validation
    # In practice, you'd want separate validation directories
    val_dataset = MedicalPatchDataset(
        cfg=cfg.data,
        patch_size=patch_size,
        patches_per_volume=max(1, cfg.train.max_patches_per_volume // 4 if cfg.train.max_patches_per_volume else 1),
        seed=cfg.seed + 1
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None,
        pin_memory=True
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
    
    for epoch in range(start_epoch, cfg.train.epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, writer)
        writer.add_scalar("Loss/train_epoch", train_loss, epoch)
        
        # Evaluate
        if (epoch + 1) % cfg.eval_interval == 0:
            val_dice = evaluate(model, val_loader, device, epoch, writer)
            
            # Check if best model
            is_best = val_dice > best_dice
            if is_best:
                best_dice = val_dice
            
            # Save checkpoint
            checkpoint_dir = Path(cfg.checkpoint.save_dir)
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
            checkpoint_dir = Path(cfg.checkpoint.save_dir)
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
