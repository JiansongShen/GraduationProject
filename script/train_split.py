#!/usr/bin/env python3
"""Training script for 3D medical image segmentation with configurable models."""

import argparse
import dataclasses
import logging
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.config import Config
from core.config_loader import load_config
from core.global_setting import SystemSetting
from data.MedicalPatchDataset import MedicalPatchDataset
from model.aneurysm.model.AttentionUnet import AttentionUnet
from script.eval_split import evaluate, dice_loss, combined_loss


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


def build_model(cfg: Config, device: torch.device) -> AttentionUnet:
    """Build model based on configuration."""
    # Currently using AttentionUnet - can be extended to support multiple models
    logging.info(f"build a model: depth: {cfg.model.depth}")
    model = AttentionUnet(
        in_ch=cfg.model.in_channels,
        out_ch=cfg.model.out_channels,
        depth=cfg.model.depth,
        base_filter=cfg.model.base_filters, 
        norm_type=cfg.model.norm_type,
        activation=cfg.model.activation,
        dropout=cfg.model.dropout
    )
    
    model = model.to(device)
    
    # Optional: compile model for faster training (PyTorch 2.0+)
    # if cfg.compile.enabled and hasattr(torch, 'compile'):
    #     logging.info(f"Compiling model with mode: {cfg.compile.mode}")
    #     model = torch.compile(model, mode=cfg.compile.mode, fullgraph=cfg.compile.fullgraph)
    #
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
    dataset: MedicalPatchDataset,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    cfg: Config,
) -> float:
    """Train for one epoch by loading one case at a time."""
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    total_volumes = len(dataset)
    logging.debug("Epoch %s training started: total_volumes=%s", epoch + 1, total_volumes)

    for batch_idx in range(total_volumes):
        logging.debug("Epoch %s volume %s/%s loading", epoch + 1, batch_idx + 1, total_volumes)
        
        # if !dataset.ensure_can_load_case(batch_idx):
        #     continue

        images, labels = dataset[batch_idx]

        if labels is None:
            logging.warning("Epoch %s volume %s/%s has no labels. Skipping.", epoch + 1, batch_idx + 1, total_volumes)
            continue

        patch_count = int(images.shape[0])
        logging.debug(
            "Epoch %s volume %s/%s loaded: patches=%s image_shape=%s label_shape=%s",
            epoch + 1,
            batch_idx + 1,
            total_volumes,
            patch_count,
            tuple(images.shape),
            tuple(labels.shape),
        )

        # Process patches in batches
        batch_size = cfg.train.batch_size
        for patch_start_idx in range(0, patch_count, batch_size):
            patch_end_idx = min(patch_start_idx + batch_size, patch_count)
            
            batch_images = images[patch_start_idx:patch_end_idx].to(device)
            batch_labels = labels[patch_start_idx:patch_end_idx].float().to(device)

            optimizer.zero_grad()
            outputs = model(batch_images)
            batch_labels = align_target_shape(outputs, batch_labels)
            # Using dice_loss as combined_loss was not defined/imported
            loss = combined_loss(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            avg_loss = epoch_loss / num_batches
            logging.debug(
                "Epoch %s volume %s/%s patch batch %s-%s done: loss=%.6f avg_loss=%.6f global_patch_step=%s",
                epoch + 1,
                batch_idx + 1,
                total_volumes,
                patch_start_idx + 1,
                patch_end_idx,
                loss.item(),
                avg_loss,
                num_batches,
            )

            if num_batches % 10 == 0:
                writer.add_scalar("Loss/train_batch", loss.item(), epoch * len(dataset) + num_batches)

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
    model: torch.nn.Module = build_model(cfg, device)
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
    
    eval_data_cfg = cfg.data
    if cfg.data.eval_dirs:
        eval_data_cfg = dataclasses.replace(cfg.data, train_dirs=list(cfg.data.eval_dirs))
        logging.info("Using dedicated eval dirs: %s", cfg.data.eval_dirs)
    else:
        logging.warning(
            "No eval_dirs configured under data.eval_dirs; validation will reuse train_dirs."
        )

    eval_data_cfg.patch_sampling_mode = "sequential"
    eval_data_cfg.max_load = 10000
    eval_data_cfg.patches_per_volume = 64

    val_dataset = MedicalPatchDataset(
        cfg=eval_data_cfg,
        patch_size=patch_size,
        seed=cfg.seed + 1
    )
    
    logging.info(f"Training samples: {len(train_dataset)}")
    logging.info(f"Validation samples: {len(val_dataset)}")
    
    # Setup TensorBoard
    writer = SummaryWriter(log_dir=(log_dir / "tensorboard").__str__())
    
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

    # progress bar
    progress_bar = tqdm(range(start_epoch, cfg.train.epochs), desc="Training", initial=start_epoch, total=cfg.train.epochs)
    for epoch in progress_bar:
        # Train
        train_loss = train_one_epoch(
            model,
            train_dataset,
            optimizer,
            device,
            epoch,
            writer,
            cfg=cfg,
        )
        writer.add_scalar("Loss/train_epoch", train_loss, epoch)
        
        # Evaluate
        if (epoch + 1) % cfg.eval_interval == 0:
            val_dice = evaluate(
                model,
                val_dataset,
                device,
                epoch,
                writer,
                prediction_dir=prediction_dir / "eval",
                cfg=cfg,  # Pass the config
                store_single_res=True,
            )
            
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
