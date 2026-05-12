#!/usr/bin/env python3
"""Run segmentation ablation experiments for U-Net, ASPP, CoordAttention, and both.

This script reuses one base YAML config for data/training/eval settings, then
creates four isolated experiment folders with different model module switches.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

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
from script.train_split import build_model


ABLATIONS: dict[str, dict[str, bool]] = {
    "unet": {"use_aspp": False, "use_coord_attention": False},
    "aspp": {"use_aspp": True, "use_coord_attention": False},
    "coord_attention": {"use_aspp": False, "use_coord_attention": True},
    "aspp_coord_attention": {"use_aspp": True, "use_coord_attention": True},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 3D segmentation ablation training")
    parser.add_argument("--config", type=str, required=True, help="Base YAML config shared by all ablations")
    parser.add_argument("--output-root", type=str, default="ablation_runs", help="Root directory for all ablation outputs")
    parser.add_argument("--device", type=str, default=None, help="Override device (cuda/cpu)")
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=sorted(ABLATIONS),
        default=list(ABLATIONS),
        help="Subset of ablation experiments to run",
    )
    parser.add_argument("--resume", action="store_true", help="Resume each experiment from its latest.pth if present")
    return parser.parse_args()


@torch.no_grad()
def save_epoch_start_snapshot(
    model: nn.Module,
    val_dataset: MedicalPatchDataset,
    device: torch.device,
    epoch: int,
    out_dir: Path,
    batch_size: int,
) -> None:
    if len(val_dataset) == 0:
        return
    sample_idx = 0
    images, labels = val_dataset.get_patches(sample_idx, sampling_mode="sequential")
    if labels is None or int(images.shape[0]) == 0:
        return
    was_training = model.training
    model.eval()
    try:
        original_image = sitk.ReadImage(val_dataset.cases[sample_idx].image_path)
        image_tensor, _ = val_dataset.get_src_item(sample_idx)
        patch_list: list[torch.Tensor] = []
        for patch_start in range(0, int(images.shape[0]), batch_size):
            batch_images = images[patch_start : patch_start + batch_size].to(device)
            outputs = model(batch_images)
            patch_list.extend([p.detach().cpu() for p in outputs[:, 0]])
        _save_predictions(out_dir, epoch, val_dataset, sample_idx, patch_list, image_tensor.shape, original_image)
    finally:
        model.train(was_training)


def prepare_eval_dataset(cfg: Config) -> MedicalPatchDataset:
    eval_data_cfg = cfg.data
    if cfg.data.eval_dirs:
        eval_data_cfg = dataclasses.replace(cfg.data, train_dirs=list(cfg.data.eval_dirs))
    eval_data_cfg.patch_sampling_mode = "sequential"
    eval_data_cfg.max_load = cfg.data.eval_max_load if cfg.data.eval_max_load is not None else 10000
    eval_data_cfg.patches_per_volume = 100000
    return MedicalPatchDataset(cfg=eval_data_cfg, patch_size=tuple(cfg.inference.patch_size), seed=cfg.seed + 1)


def combine_eval_reports(epoch: int, sequential_report: dict[str, Any], foreground_report: dict[str, Any]) -> dict[str, Any]:
    seq_metric = sequential_report["metrics"]["segmentation"]
    fg_metric = foreground_report["metrics"]["segmentation"]
    return {
        "meta": {
            "task": "segmentation_dual_sampling_eval",
            "report_version": "1.0",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "epoch": epoch + 1,
        },
        "summary": {
            "sequential_mean_dice": seq_metric["mean_dice"],
            "sequential_best_f1": seq_metric["best_f1"],
            "foreground_only_mean_dice": fg_metric["mean_dice"],
            "foreground_only_best_f1": fg_metric["best_f1"],
        },
        "sequential": sequential_report,
        "foreground_only": foreground_report,
    }


def run_one_experiment(base_cfg: Config, name: str, switches: dict[str, bool], output_root: Path, device_override: str | None, resume: bool) -> None:
    cfg = copy.deepcopy(base_cfg)
    cfg.model.use_aspp = switches["use_aspp"]
    cfg.model.use_coord_attention = switches["use_coord_attention"]

    exp_dir = output_root / name
    log_dir = exp_dir / "logs"
    checkpoint_dir = exp_dir / "checkpoints"
    cfg.log_dir = str(log_dir)
    cfg.checkpoint.save_dir = str(checkpoint_dir)
    if device_override:
        cfg.device = device_override

    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    setup_basic_logging(log_dir / "training.log")
    logging.info("Starting ablation=%s use_aspp=%s use_coord_attention=%s", name, cfg.model.use_aspp, cfg.model.use_coord_attention)

    SystemSetting.set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device)
    optimizer = build_optimizer(model, cfg.train.learning_rate, cfg.train.weight_decay, cfg.train.optimizer)
    scheduler = build_scheduler(optimizer, cfg.train.scheduler, cfg.train.epochs, cfg.train.warmup_epochs)

    train_dataset = MedicalPatchDataset(cfg=cfg.data, patch_size=tuple(cfg.inference.patch_size), seed=cfg.seed)
    val_dataset = prepare_eval_dataset(cfg)
    writer = SummaryWriter(log_dir=str(log_dir / "tensorboard"))
    eval_report_dir = build_eval_report_dir(log_dir, run_started_at=datetime.now())
    epoch_nifti_dir = log_dir / "epoch_predictions_nifti"

    start_epoch = 0
    best_score = 0.0
    latest_checkpoint = checkpoint_dir / "latest.pth"
    if resume and latest_checkpoint.is_file():
        start_epoch, best_score = load_checkpoint(str(latest_checkpoint), model, optimizer, scheduler)
        logging.info("Resumed %s from epoch=%d best_score=%.4f", name, start_epoch, best_score)

    for epoch in range(start_epoch, cfg.train.epochs):
        save_epoch_start_snapshot(model, val_dataset, device, epoch, epoch_nifti_dir, cfg.train.batch_size)
        train_loss = train_one_epoch(
            model,
            train_dataset,
            optimizer,
            device,
            epoch,
            writer,
            batch_size=cfg.train.batch_size,
            loss_kwargs=cfg.train.segmentation_loss_kwargs(),
        )
        writer.add_scalar("Loss/train_epoch", train_loss, epoch)
        writer.add_scalar("LR/train", optimizer.param_groups[0]["lr"], epoch)
        if cfg.train.scheduler.lower() != "none" and epoch >= cfg.train.warmup_epochs:
            scheduler.step()

        if (epoch + 1) % cfg.eval_every_n_epochs == 0:
            sequential_report = validate(
                model,
                val_dataset,
                device,
                epoch,
                writer,
                batch_size=cfg.train.batch_size,
                loss_kwargs=cfg.train.segmentation_loss_kwargs(),
                cfg=cfg,
                sampling_mode="sequential",
                writer_prefix=f"{name}/sequential_val",
            )
            foreground_report = validate(
                model,
                val_dataset,
                device,
                epoch,
                writer,
                batch_size=cfg.train.batch_size,
                loss_kwargs=cfg.train.segmentation_loss_kwargs(),
                cfg=cfg,
                sampling_mode="foreground_only",
                writer_prefix=f"{name}/foreground_only_val",
            )
            combined_report = combine_eval_reports(epoch, sequential_report, foreground_report)
            score = float(sequential_report["metrics"]["segmentation"]["mean_dice"])
            is_best = score > best_score
            if is_best:
                best_score = score
                sequential_report["meta"]["is_best_so_far"] = True
                foreground_report["meta"]["is_best_so_far"] = True
            save_eval_report(eval_report_dir / "sequential", epoch, sequential_report)
            save_eval_report(eval_report_dir / "foreground_only", epoch, foreground_report)
            save_eval_report(eval_report_dir / "combined", epoch, combined_report)
            save_checkpoint(model, optimizer, epoch + 1, best_score, checkpoint_dir, is_best, scheduler=scheduler)

        if (epoch + 1) % cfg.checkpoint.save_interval == 0:
            save_checkpoint(model, optimizer, epoch + 1, best_score, checkpoint_dir, is_best=False, scheduler=scheduler)

    writer.close()
    logging.info("Finished ablation=%s best sequential Dice=%.4f", name, best_score)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = load_config(args.config)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for experiment in args.experiments:
        run_one_experiment(cfg, experiment, ABLATIONS[experiment], output_root, args.device, args.resume)


if __name__ == "__main__":
    main()
