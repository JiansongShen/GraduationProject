#!/usr/bin/env python3
"""Training script for 3D medical image segmentation."""

import argparse
import dataclasses
import logging
import sys
from datetime import datetime
from logging import DEBUG
from pathlib import Path

import SimpleITK as sitk
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.config import Config
from core.config_loader import load_config
from core.global_setting import SystemSetting
from data.MedicalPatchDataset import MedicalPatchDataset
from model.aneurysm.model.AttentionUnet import AttentionUnet
from script._training import (
    build_eval_report_dir,
    build_optimizer,
    build_scheduler,
    load_checkpoint,
    save_checkpoint,
    save_eval_report,
    train_one_epoch,
    validate,
)
from script.common import setup_basic_logging
from script.eval_split import _save_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 3D medical image segmentation model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training")
    parser.add_argument("--device", type=str, default=None, help="Override device (cuda/cpu)")
    return parser.parse_args()


@torch.no_grad()
def _save_epoch_start_first_case_nifti(
    model: nn.Module,
    val_dataset: MedicalPatchDataset,
    device: torch.device,
    epoch: int,
    out_dir: Path,
    batch_size: int,
) -> None:
    """每个 epoch 训练前：对验证集第 0 例做顺序 patch 推理并写入 NIfTI（概率 + 二值）。"""
    if len(val_dataset) == 0:
        logging.warning("epoch NIfTI snapshot skipped: empty validation dataset")
        return
    sample_idx = 0
    images, labels = val_dataset.get_patches(sample_idx, sampling_mode="sequential")
    if labels is None:
        logging.warning("epoch NIfTI snapshot skipped: case 0 has no label patches")
        return
    if int(images.shape[0]) == 0:
        logging.warning("epoch NIfTI snapshot skipped: case 0 has zero sequential patches")
        return

    was_training = model.training
    model.eval()
    try:
        original_image = sitk.ReadImage(val_dataset.cases[sample_idx].image_path)
        image_tensor, _ = val_dataset.get_src_item(sample_idx)
        patch_list: list[torch.Tensor] = []
        for patch_start in range(0, int(images.shape[0]), batch_size):
            patch_end = min(patch_start + batch_size, int(images.shape[0]))
            batch_images = images[patch_start:patch_end].to(device)
            outputs = model(batch_images)
            patch_list.extend([p.detach().cpu() for p in outputs[:, 0]])
        _save_predictions(
            out_dir,
            epoch,
            val_dataset,
            sample_idx,
            patch_list,
            image_tensor.shape,
            original_image,
        )
    finally:
        model.train(was_training)


def build_model(cfg: Config, device: torch.device) -> AttentionUnet:
    model = AttentionUnet(
        in_ch=cfg.model.in_channels,
        out_ch=cfg.model.out_channels,
        depth=cfg.model.depth,
        base_filter=cfg.model.base_filters,
        norm_type=cfg.model.norm_type,
        activation=cfg.model.activation,
        dropout=cfg.model.dropout,
    )
    return model.to(device)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=DEBUG)

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = load_config(config_path)
    if args.device:
        cfg.device = args.device

    SystemSetting.set_seed(cfg.seed)

    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_basic_logging(log_dir / "training.log")

    logging.info("Config: %s, Device: %s", config_path, cfg.device)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Model
    model = build_model(cfg, device)
    logging.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    # Optimizer and scheduler
    optimizer = build_optimizer(
        model,
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        optimizer=cfg.train.optimizer,
    )
    scheduler = build_scheduler(
        optimizer, cfg.train.scheduler, cfg.train.epochs, cfg.train.warmup_epochs
    )

    # Datasets
    patch_size = tuple(cfg.inference.patch_size)
    train_dataset = MedicalPatchDataset(cfg=cfg.data, patch_size=patch_size, seed=cfg.seed)

    eval_data_cfg = cfg.data
    if cfg.data.eval_dirs:
        eval_data_cfg = dataclasses.replace(cfg.data, train_dirs=list(cfg.data.eval_dirs))
    eval_data_cfg.patch_sampling_mode = "sequential"
    eval_data_cfg.max_load = cfg.data.eval_max_load if cfg.data.eval_max_load is not None else 10000
    eval_data_cfg.patches_per_volume = 100000
    val_dataset = MedicalPatchDataset(cfg=eval_data_cfg, patch_size=patch_size, seed=cfg.seed + 1)

    logging.info(
        "Training samples: %d, Validation samples: %d (eval_max_load=%s)",
        len(train_dataset),
        len(val_dataset),
        cfg.data.eval_max_load,
    )

    writer = SummaryWriter(log_dir=str(log_dir / "tensorboard"))
    checkpoint_dir = Path(cfg.checkpoint.save_dir)
    epoch_nifti_dir = log_dir / "epoch_predictions_nifti"
    eval_report_dir = build_eval_report_dir(log_dir, run_started_at=datetime.now())
    logging.info("Eval reports will be saved to %s", eval_report_dir)

    start_epoch, best_dice = 0, 0.0
    if args.resume:
        start_epoch, best_dice = load_checkpoint(args.resume, model, optimizer)
        logging.info("Resuming from epoch %d", start_epoch)

    logging.info(
        "Starting training for %d epochs... eval_every_n_epochs=%d save_interval=%d",
        cfg.train.epochs,
        cfg.eval_every_n_epochs,
        cfg.checkpoint.save_interval,
    )
    for epoch in range(start_epoch, cfg.train.epochs):
        _save_epoch_start_first_case_nifti(
            model,
            val_dataset,
            device,
            epoch,
            epoch_nifti_dir,
            cfg.train.batch_size,
        )
        train_loss = train_one_epoch(
            model,
            train_dataset,
            optimizer,
            device,
            epoch,
            writer,
            batch_size=cfg.train.batch_size,
            log_interval=10,
            loss_kwargs=cfg.train.segmentation_loss_kwargs(),
        )
        writer.add_scalar("Loss/train_epoch", train_loss, epoch)
        writer.add_scalar("LR/train", optimizer.param_groups[0]["lr"], epoch)

        if cfg.train.scheduler.lower() != "none" and epoch >= cfg.train.warmup_epochs:
            scheduler.step()

        if (epoch + 1) % cfg.eval_every_n_epochs == 0:
            logging.info(
                "Running validation/report at epoch %d because eval_every_n_epochs=%d",
                epoch + 1,
                cfg.eval_every_n_epochs,
            )
            eval_report = validate(
                model,
                val_dataset,
                device,
                epoch,
                writer,
                batch_size=cfg.train.batch_size,
                loss_kwargs=cfg.train.segmentation_loss_kwargs(),
                cfg=cfg,
            )
            val_dice = float(eval_report["metrics"]["segmentation"]["mean_dice"])
            is_best = val_dice > best_dice
            if is_best:
                best_dice = val_dice
                eval_report["meta"]["is_best_so_far"] = True
            save_eval_report(report_dir=eval_report_dir, epoch=epoch, report=eval_report)
            save_checkpoint(model, optimizer, epoch + 1, val_dice, checkpoint_dir, is_best)

        if (epoch + 1) % cfg.checkpoint.save_interval == 0:
            save_checkpoint(model, optimizer, epoch + 1, best_dice, checkpoint_dir, is_best=False)

    logging.info("Training completed! Best Dice: %.4f", best_dice)
    writer.close()


if __name__ == "__main__":
    main()
