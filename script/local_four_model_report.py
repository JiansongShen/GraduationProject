#!/usr/bin/env python3
"""Train four local 3D segmentation variants and generate a complete experiment report.

The project is for intracranial aneurysm / CTA-style 3D medical image segmentation.
This runner keeps the existing dataset, model, loss, evaluation and checkpoint
utilities, then executes four architecture variants:

1. U-Net baseline
2. U-Net + ASPP
3. U-Net + Coordinate Attention
4. U-Net + ASPP + Coordinate Attention

Outputs per model include checkpoints, TensorBoard logs, epoch JSON reports and
NIfTI snapshots. A final Markdown + JSON summary compares all four models.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import SimpleITK as sitk
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import Config  # noqa: E402
from core.config_loader import load_config  # noqa: E402
from core.global_setting import SystemSetting  # noqa: E402
from data.MedicalPatchDataset import MedicalPatchDataset  # noqa: E402
from script._training import (  # noqa: E402
    build_eval_report_dir,
    build_optimizer,
    build_scheduler,
    load_checkpoint,
    save_checkpoint,
    save_eval_report,
    train_one_epoch,
    validate,
)
from script.common import setup_basic_logging  # noqa: E402
from script.eval_split import _save_predictions  # noqa: E402
from script.train_split import build_model  # noqa: E402


EXPERIMENTS: dict[str, dict[str, Any]] = {
    "01_unet_baseline": {
        "display_name": "3D U-Net baseline",
        "description": "基础 3D U-Net，作为医学图像分割基线模型。",
        "switches": {"use_aspp": False, "use_coord_attention": False},
    },
    "02_unet_aspp": {
        "display_name": "3D U-Net + ASPP",
        "description": "在瓶颈层加入 ASPP，多尺度空洞卷积增强不同尺度动脉瘤区域表达。",
        "switches": {"use_aspp": True, "use_coord_attention": False},
    },
    "03_unet_coord_attention": {
        "display_name": "3D U-Net + CoordAttention",
        "description": "在瓶颈层加入三维坐标注意力，强化空间方向敏感特征。",
        "switches": {"use_aspp": False, "use_coord_attention": True},
    },
    "04_unet_aspp_coord_attention": {
        "display_name": "3D U-Net + ASPP + CoordAttention",
        "description": "组合 ASPP 与三维坐标注意力，是本项目的增强分割模型。",
        "switches": {"use_aspp": True, "use_coord_attention": True},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run four local segmentation trainings and generate reports")
    parser.add_argument("--config", type=str, default="config/local_four_models.yaml", help="Base YAML config")
    parser.add_argument("--output-root", type=str, default="outputs/local_four_models", help="Output root")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cuda or cpu")
    parser.add_argument("--resume", action="store_true", help="Resume each model from latest.pth when available")
    parser.add_argument("--skip-snapshots", action="store_true", help="Skip per-epoch first-case NIfTI snapshots")
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=sorted(EXPERIMENTS),
        default=list(EXPERIMENTS),
        help="Subset of the four model experiments to run",
    )
    return parser.parse_args()


def _json_ready(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_ready(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def _metric(report: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if not report:
        return default
    try:
        return float(report["metrics"]["segmentation"][key])
    except (KeyError, TypeError, ValueError):
        return default


def prepare_eval_dataset(cfg: Config) -> MedicalPatchDataset:
    eval_data_cfg = cfg.data
    if cfg.data.eval_dirs:
        eval_data_cfg = dataclasses.replace(cfg.data, train_dirs=list(cfg.data.eval_dirs))
    eval_data_cfg.patch_sampling_mode = "sequential"
    eval_data_cfg.max_load = cfg.data.eval_max_load if cfg.data.eval_max_load is not None else cfg.data.max_load
    eval_data_cfg.patches_per_volume = 100000
    return MedicalPatchDataset(cfg=eval_data_cfg, patch_size=tuple(cfg.inference.patch_size), seed=cfg.seed + 1)


@torch.no_grad()
def save_first_case_snapshot(
    model: nn.Module,
    val_dataset: MedicalPatchDataset,
    device: torch.device,
    epoch: int,
    out_dir: Path,
    batch_size: int,
) -> None:
    if len(val_dataset) == 0:
        return
    images, labels = val_dataset.get_patches(0, sampling_mode="sequential")
    if labels is None or int(images.shape[0]) == 0:
        return

    was_training = model.training
    model.eval()
    try:
        original_image = sitk.ReadImage(val_dataset.cases[0].image_path)
        image_tensor, _ = val_dataset.get_src_item(0)
        patch_list: list[torch.Tensor] = []
        for patch_start in range(0, int(images.shape[0]), batch_size):
            batch_images = images[patch_start : patch_start + batch_size].to(device)
            outputs = model(batch_images)
            patch_list.extend([patch.detach().cpu() for patch in outputs[:, 0]])
        _save_predictions(out_dir, epoch, val_dataset, 0, patch_list, image_tensor.shape, original_image)
    finally:
        model.train(was_training)


def combine_reports(
    experiment_key: str,
    experiment_meta: dict[str, Any],
    epoch: int,
    train_loss: float,
    best_score: float,
    sequential_report: dict[str, Any],
    foreground_report: dict[str, Any],
) -> dict[str, Any]:
    seq = sequential_report["metrics"]["segmentation"]
    fg = foreground_report["metrics"]["segmentation"]
    return {
        "meta": {
            "task": "four_model_segmentation_training_report",
            "report_version": "1.0",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "experiment_key": experiment_key,
            "display_name": experiment_meta["display_name"],
            "description": experiment_meta["description"],
            "epoch": epoch + 1,
        },
        "summary": {
            "train_loss": train_loss,
            "best_sequential_mean_dice_so_far": best_score,
            "sequential_mean_dice": seq["mean_dice"],
            "sequential_voxel_auc": seq["voxel_auc"],
            "sequential_best_f1": seq["best_f1"],
            "sequential_best_f1_threshold": seq["best_f1_threshold"],
            "foreground_only_mean_dice": fg["mean_dice"],
            "foreground_only_best_f1": fg["best_f1"],
        },
        "sequential": sequential_report,
        "foreground_only": foreground_report,
    }


def run_experiment(
    base_cfg: Config,
    experiment_key: str,
    output_root: Path,
    device_override: str | None,
    resume: bool,
    skip_snapshots: bool,
) -> dict[str, Any]:
    meta = EXPERIMENTS[experiment_key]
    cfg = copy.deepcopy(base_cfg)
    cfg.model.use_aspp = bool(meta["switches"]["use_aspp"])
    cfg.model.use_coord_attention = bool(meta["switches"]["use_coord_attention"])
    if device_override:
        cfg.device = device_override

    exp_dir = output_root / experiment_key
    log_dir = exp_dir / "logs"
    checkpoint_dir = exp_dir / "checkpoints"
    reports_dir = exp_dir / "reports"
    cfg.log_dir = str(log_dir)
    cfg.checkpoint.save_dir = str(checkpoint_dir)
    for directory in (log_dir, checkpoint_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    setup_basic_logging(log_dir / "training.log")
    logging.info("Experiment %s: %s", experiment_key, meta["display_name"])
    logging.info("Switches: %s", meta["switches"])

    SystemSetting.set_seed(cfg.seed)
    device = torch.device(cfg.device if cfg.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device)
    parameter_count = sum(p.numel() for p in model.parameters())
    optimizer = build_optimizer(model, cfg.train.learning_rate, cfg.train.weight_decay, cfg.train.optimizer)
    scheduler = build_scheduler(optimizer, cfg.train.scheduler, cfg.train.epochs, cfg.train.warmup_epochs)

    train_dataset = MedicalPatchDataset(cfg=cfg.data, patch_size=tuple(cfg.inference.patch_size), seed=cfg.seed)
    val_dataset = prepare_eval_dataset(cfg)
    writer = SummaryWriter(log_dir=str(log_dir / "tensorboard"))
    eval_report_dir = build_eval_report_dir(reports_dir, run_started_at=datetime.now())

    start_epoch = 0
    best_score = 0.0
    latest_checkpoint = checkpoint_dir / "latest.pth"
    if resume and latest_checkpoint.is_file():
        start_epoch, best_score = load_checkpoint(str(latest_checkpoint), model, optimizer, scheduler)
        logging.info("Resumed from %s at epoch=%d best_score=%.6f", latest_checkpoint, start_epoch, best_score)

    latest_combined: dict[str, Any] | None = None
    last_train_loss = 0.0
    for epoch in range(start_epoch, cfg.train.epochs):
        if not skip_snapshots:
            save_first_case_snapshot(model, val_dataset, device, epoch, log_dir / "epoch_predictions_nifti", cfg.train.batch_size)

        last_train_loss = train_one_epoch(
            model,
            train_dataset,
            optimizer,
            device,
            epoch,
            writer,
            batch_size=cfg.train.batch_size,
            loss_kwargs=cfg.train.segmentation_loss_kwargs(),
        )
        writer.add_scalar("Loss/train_epoch", last_train_loss, epoch)
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
                writer_prefix=f"{experiment_key}/sequential",
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
                writer_prefix=f"{experiment_key}/foreground_only",
            )
            score = _metric(sequential_report, "mean_dice")
            is_best = score > best_score
            if is_best:
                best_score = score
                sequential_report["meta"]["is_best_so_far"] = True
                foreground_report["meta"]["is_best_so_far"] = True
            latest_combined = combine_reports(
                experiment_key,
                meta,
                epoch,
                last_train_loss,
                best_score,
                sequential_report,
                foreground_report,
            )
            save_eval_report(eval_report_dir / "sequential", epoch, sequential_report)
            save_eval_report(eval_report_dir / "foreground_only", epoch, foreground_report)
            save_eval_report(eval_report_dir / "combined", epoch, latest_combined)
            save_checkpoint(model, optimizer, epoch + 1, best_score, checkpoint_dir, is_best, scheduler=scheduler)

        if (epoch + 1) % cfg.checkpoint.save_interval == 0:
            save_checkpoint(model, optimizer, epoch + 1, best_score, checkpoint_dir, is_best=False, scheduler=scheduler)

    writer.close()

    if latest_combined is None:
        save_checkpoint(model, optimizer, cfg.train.epochs, best_score, checkpoint_dir, is_best=True, scheduler=scheduler)

    summary = {
        "experiment_key": experiment_key,
        "display_name": meta["display_name"],
        "description": meta["description"],
        "use_aspp": cfg.model.use_aspp,
        "use_coord_attention": cfg.model.use_coord_attention,
        "parameter_count": parameter_count,
        "train_cases": len(train_dataset),
        "eval_cases": len(val_dataset),
        "epochs": cfg.train.epochs,
        "last_train_loss": last_train_loss,
        "best_sequential_mean_dice": best_score,
        "latest_sequential_mean_dice": _metric(latest_combined.get("sequential") if latest_combined else None, "mean_dice"),
        "latest_sequential_voxel_auc": _metric(latest_combined.get("sequential") if latest_combined else None, "voxel_auc"),
        "latest_sequential_best_f1": _metric(latest_combined.get("sequential") if latest_combined else None, "best_f1"),
        "latest_foreground_only_mean_dice": _metric(latest_combined.get("foreground_only") if latest_combined else None, "mean_dice"),
        "checkpoint_latest": str(checkpoint_dir / "latest.pth"),
        "checkpoint_best": str(checkpoint_dir / "best.pth"),
        "log_dir": str(log_dir),
        "report_dir": str(eval_report_dir),
    }
    (reports_dir / "model_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Finished %s, best sequential Dice %.6f", experiment_key, best_score)
    return summary


def write_final_reports(output_root: Path, config_path: Path, cfg: Config, summaries: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    sorted_summaries = sorted(summaries, key=lambda item: item["best_sequential_mean_dice"], reverse=True)
    payload = {
        "meta": {
            "task": "local_four_model_segmentation_experiment",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "config_path": str(config_path),
            "output_root": str(output_root),
        },
        "config_snapshot": _json_ready(cfg),
        "ranking_metric": "best_sequential_mean_dice",
        "summaries": summaries,
        "ranking": sorted_summaries,
    }
    (output_root / "complete_training_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 本地四模型医学图像分割训练完整报告",
        "",
        f"- 生成时间：{payload['meta']['generated_at']}",
        f"- 配置文件：`{config_path}`",
        f"- 输出目录：`{output_root}`",
        f"- 排名指标：`{payload['ranking_metric']}`",
        "",
        "## 项目理解",
        "",
        "本项目面向 3D CTA/医学影像动脉瘤分割，数据加载器按 NIfTI 影像与标签对读取病例，重采样到训练空间后进行 patch 采样。模型主干为 3D U-Net，并支持在瓶颈层启用 ASPP 多尺度上下文模块与 3D 坐标注意力模块。训练使用 BCE 与 Tversky/Focal 类分割损失组合，验证阶段输出 Dice、AUC、F1、IoU、Precision、Recall、Specificity 与阈值扫描结果。",
        "",
        "## 四个模型设置",
        "",
        "| 模型 | ASPP | CoordAttention | 参数量 | 最佳 Dice | 最新 Dice | 最新 AUC | 最新 Best-F1 | checkpoint |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in summaries:
        lines.append(
            "| {display_name} | {use_aspp} | {use_coord_attention} | {parameter_count} | {best:.6f} | {dice:.6f} | {auc:.6f} | {f1:.6f} | `{ckpt}` |".format(
                display_name=item["display_name"],
                use_aspp="是" if item["use_aspp"] else "否",
                use_coord_attention="是" if item["use_coord_attention"] else "否",
                parameter_count=item["parameter_count"],
                best=item["best_sequential_mean_dice"],
                dice=item["latest_sequential_mean_dice"],
                auc=item["latest_sequential_voxel_auc"],
                f1=item["latest_sequential_best_f1"],
                ckpt=item["checkpoint_best"],
            )
        )

    if sorted_summaries:
        best = sorted_summaries[0]
        lines.extend(
            [
                "",
                "## 最优模型",
                "",
                f"当前以最佳 sequential mean Dice 排名，最优模型为 **{best['display_name']}**，最佳 Dice 为 `{best['best_sequential_mean_dice']:.6f}`。",
                f"最优权重路径：`{best['checkpoint_best']}`",
            ]
        )

    lines.extend(
        [
            "",
            "## 输出说明",
            "",
            "每个模型目录下包含：",
            "",
            "- `checkpoints/latest.pth`：最新 checkpoint，可继续训练。",
            "- `checkpoints/best.pth`：按验证 Dice 保存的最佳模型。",
            "- `logs/training.log`：训练日志。",
            "- `logs/tensorboard/`：TensorBoard 曲线。",
            "- `reports/eval_reports/.../combined/latest.json`：该模型最新完整评估报告。",
            "- `reports/model_summary.json`：该模型摘要。",
            "",
            "总报告文件：",
            "",
            "- `complete_training_report.md`：人类可读完整报告。",
            "- `complete_training_report.json`：机器可读完整报告，可用于后续论文表格或可视化。",
        ]
    )
    (output_root / "complete_training_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = load_config(config_path)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_root / "used_config.yaml")

    summaries: list[dict[str, Any]] = []
    for experiment_key in args.experiments:
        summaries.append(
            run_experiment(
                cfg,
                experiment_key,
                output_root,
                args.device,
                args.resume,
                args.skip_snapshots,
            )
        )

    write_final_reports(output_root, config_path, cfg, summaries)
    print(f"Complete reports saved to: {output_root / 'complete_training_report.md'}")
    print(f"Machine-readable report saved to: {output_root / 'complete_training_report.json'}")


if __name__ == "__main__":
    main()
