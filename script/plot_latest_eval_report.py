#!/usr/bin/env python3
"""Locate the latest segmentation eval report and plot metric curves."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot curves from the latest eval report run")
    parser.add_argument(
        "--reports-root",
        type=str,
        default="logs/eval_reports",
        help="Root directory containing dated eval report runs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output PNG path. Defaults to <latest_run>/eval_metrics_summary.png",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after saving",
    )
    return parser.parse_args()


def _latest_run_dir(reports_root: Path) -> Path:
    if not reports_root.exists():
        raise FileNotFoundError(f"Reports root does not exist: {reports_root}")

    run_dirs = [
        path
        for path in reports_root.glob("*/*")
        if path.is_dir() and path.name.startswith("run_")
    ]
    if not run_dirs:
        raise FileNotFoundError(f"No eval report run directories found under: {reports_root}")
    return max(run_dirs, key=lambda path: path.stat().st_mtime)


def _load_reports(run_dir: Path) -> list[dict[str, Any]]:
    report_files = sorted(run_dir.glob("eval_epoch_*.json"))
    if not report_files:
        raise FileNotFoundError(f"No eval report files found in: {run_dir}")

    reports: list[dict[str, Any]] = []
    for report_file in report_files:
        with report_file.open("r", encoding="utf-8") as fp:
            reports.append(json.load(fp))
    return reports


def _extract_curve(reports: list[dict[str, Any]], path: tuple[str, ...]) -> list[float]:
    values: list[float] = []
    for report in reports:
        current: Any = report
        for key in path:
            current = current[key]
        values.append(float(current))
    return values


def _extract_best_threshold_f1(reports: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for report in reports:
        sweep = report["metrics"].get("threshold_sweep", {})
        if not sweep:
            values.append(0.0)
            continue
        values.append(max(float(item.get("f1", 0.0)) for item in sweep.values()))
    return values


def plot_reports(run_dir: Path, reports: list[dict[str, Any]], output_path: Path, show: bool) -> None:
    epochs = [int(report["meta"]["epoch"]) for report in reports]
    train_like_loss = _extract_curve(reports, ("metrics", "loss", "mean_total"))
    mean_dice = _extract_curve(reports, ("metrics", "segmentation", "mean_dice"))
    voxel_auc = _extract_curve(reports, ("metrics", "segmentation", "voxel_auc"))
    voxel_f1 = _extract_curve(reports, ("metrics", "segmentation", "voxel_f1"))
    best_threshold_f1 = _extract_best_threshold_f1(reports)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(f"Segmentation Eval Summary\n{run_dir}", fontsize=14)

    axes[0, 0].plot(epochs, train_like_loss, marker="o")
    axes[0, 0].set_title("Validation Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, mean_dice, marker="o", label="Mean Dice")
    axes[0, 1].plot(epochs, voxel_f1, marker="s", label="Voxel F1@0.50")
    axes[0, 1].plot(epochs, best_threshold_f1, marker="^", label="Best F1 in Sweep")
    axes[0, 1].set_title("Dice / F1")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, voxel_auc, marker="o", color="tab:green")
    axes[1, 0].set_title("Voxel AUC")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("AUC")
    axes[1, 0].grid(True, alpha=0.3)

    pred_pos_ratio = _extract_curve(reports, ("metrics", "segmentation", "pred_positive_ratio"))
    gt_pos_ratio = _extract_curve(reports, ("metrics", "segmentation", "gt_positive_ratio"))
    axes[1, 1].plot(epochs, pred_pos_ratio, marker="o", label="Pred positive ratio")
    axes[1, 1].plot(epochs, gt_pos_ratio, marker="s", label="GT positive ratio")
    axes[1, 1].set_title("Foreground Ratio")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Ratio")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logging.info("Saved plot to %s", output_path)

    if show:
        image = plt.imread(output_path)
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        plt.axis("off")
        plt.show()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

    reports_root = Path(args.reports_root)
    latest_run = _latest_run_dir(reports_root)
    reports = _load_reports(latest_run)
    output_path = Path(args.output) if args.output else latest_run / "eval_metrics_summary.png"

    logging.info("Using latest eval report run: %s", latest_run)
    logging.info("Loaded %d report files", len(reports))
    plot_reports(latest_run, reports, output_path, args.show)


if __name__ == "__main__":
    main()
