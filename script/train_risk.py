#!/usr/bin/env python3
"""Train risk prediction model with cross-attention on tabular data."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.config import Config
from core.config_loader import load_config
from core.global_setting import SystemSetting
from data.risk_tabular import RiskTabularDataset, build_risk_data_bundle
from model.risk.RiskCrossAttentionModel import RiskCrossAttentionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train cross-attention risk prediction model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--device", type=str, default=None, help="Override device from config (cuda/cpu)")
    return parser.parse_args()


def _binary_metrics(prob: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> tuple[float, float]:
    pred = (prob >= threshold).float()
    correct = (pred == target).float().mean().item()

    tp = ((pred == 1) & (target == 1)).sum().item()
    fp = ((pred == 1) & (target == 0)).sum().item()
    fn = ((pred == 0) & (target == 1)).sum().item()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    return correct, f1


def _safe_auc(prob: torch.Tensor, target: torch.Tensor) -> float:
    y_true = target.detach().cpu().numpy()
    y_prob = prob.detach().cpu().numpy()
    if np.unique(y_true).shape[0] < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_prob))


def _find_best_threshold(
    prob: torch.Tensor,
    target: torch.Tensor,
    steps: int,
) -> tuple[float, float, float]:
    best_threshold = 0.5
    best_acc = 0.0
    best_f1 = -1.0
    num_steps = max(2, steps)
    for threshold in np.linspace(0.05, 0.95, num_steps):
        acc, f1 = _binary_metrics(prob, target, float(threshold))
        if f1 > best_f1:
            best_f1 = f1
            best_acc = acc
            best_threshold = float(threshold)
    return best_threshold, best_acc, best_f1


def _weighted_bce_loss(
    pred_prob: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float,
) -> torch.Tensor:
    pred_prob = pred_prob.clamp(min=1e-6, max=1.0 - 1e-6)
    weighted_log_likelihood = pos_weight * target * torch.log(pred_prob) + (1.0 - target) * torch.log(1.0 - pred_prob)
    return -weighted_log_likelihood.mean()


def _run_epoch(
    model: RiskCrossAttentionModel,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
    pos_weight: float,
    threshold: float = 0.5,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, float, float, torch.Tensor, torch.Tensor]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_f1 = 0.0
    num_batches = 0
    all_probs: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    desc = "Train" if is_train else "Eval"
    progress = tqdm(dataloader, desc=desc)

    for x_num, x_cat, y in progress:
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        y = y.to(device)

        if is_train:
            assert optimizer is not None
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            pred_prob = model(x_num, x_cat)
            loss = _weighted_bce_loss(pred_prob, y, pos_weight=pos_weight)
            if is_train:
                loss.backward()
                optimizer.step()

        detached_prob = pred_prob.detach().cpu()
        detached_target = y.detach().cpu()
        all_probs.append(detached_prob)
        all_targets.append(detached_target)

        acc, f1 = _binary_metrics(detached_prob, detached_target, threshold=threshold)
        total_loss += loss.item()
        total_acc += acc
        total_f1 += f1
        num_batches += 1
        progress.set_postfix({"loss": f"{total_loss / num_batches:.4f}", "acc": f"{total_acc / num_batches:.4f}"})

    denom = max(1, num_batches)
    probs = torch.cat(all_probs, dim=0) if all_probs else torch.zeros(0)
    targets = torch.cat(all_targets, dim=0) if all_targets else torch.zeros(0)
    auc = _safe_auc(probs, targets) if probs.numel() > 0 else 0.5
    return total_loss / denom, total_acc / denom, total_f1 / denom, auc, probs, targets


def main() -> None:
    args = parse_args()
    cfg: Config = load_config(args.config)
    risk_cfg = cfg.risk

    if not risk_cfg.enabled:
        raise ValueError("Risk training is disabled. Set `risk.enabled: true` in config.")

    if args.device:
        cfg.device = args.device

    SystemSetting.set_seed(cfg.seed)
    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_dir / "risk_training.log"), logging.StreamHandler(sys.stdout)],
    )

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logging.info("Building risk data bundle...")
    bundle = build_risk_data_bundle(risk_cfg)
    logging.info(
        "Risk dataset sizes | train=%d val=%d test=%d | numeric=%d categorical=%d",
        len(bundle.y_train),
        len(bundle.y_val),
        len(bundle.y_test),
        len(bundle.numeric_columns),
        len(bundle.categorical_columns),
    )
    selected_numeric_columns = [c for c in bundle.numeric_columns if c != "__dummy_numeric__"]
    selected_categorical_columns = list(bundle.categorical_columns)
    selected_total_columns = len(selected_numeric_columns) + len(selected_categorical_columns)
    logging.info(
        "Selected training columns | total=%d clinic+categorical=%d numeric/radiomics=%d",
        selected_total_columns,
        len(selected_categorical_columns),
        len(selected_numeric_columns),
    )
    logging.info(
        "Selected column preview | categorical=%s | numeric=%s",
        selected_categorical_columns[:20],
        selected_numeric_columns[:20],
    )

    train_set = RiskTabularDataset(bundle.x_num_train, bundle.x_cat_train, bundle.y_train)
    val_set = RiskTabularDataset(bundle.x_num_val, bundle.x_cat_val, bundle.y_val)
    test_set = RiskTabularDataset(bundle.x_num_test, bundle.x_cat_test, bundle.y_test)

    train_loader = DataLoader(train_set, batch_size=risk_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=risk_cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=risk_cfg.batch_size, shuffle=False)

    model = RiskCrossAttentionModel(
        num_numeric_features=bundle.x_num_train.shape[1],
        categorical_cardinalities=bundle.categorical_cardinalities,
        hidden_dim=risk_cfg.hidden_dim,
        num_heads=risk_cfg.num_heads,
        num_layers=risk_cfg.num_layers,
        dropout=risk_cfg.dropout,
    ).to(device)
    logging.info("Risk model params: %d", RiskCrossAttentionModel.parameter_count(model))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=risk_cfg.learning_rate,
        weight_decay=risk_cfg.weight_decay,
    )

    pos_count = float((bundle.y_train > 0.5).sum())
    neg_count = float((bundle.y_train <= 0.5).sum())
    if risk_cfg.use_pos_weight and pos_count > 0.0:
        pos_weight = max(1.0, neg_count / pos_count)
    else:
        pos_weight = 1.0
    logging.info("Risk training pos_weight=%.4f (pos=%.0f neg=%.0f)", pos_weight, pos_count, neg_count)

    best_val_f1 = -1.0
    best_val_auc = 0.5
    best_threshold = 0.5
    current_threshold = 0.5
    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    early_stop_counter = 0
    for epoch in range(risk_cfg.epochs):
        train_loss, train_acc, train_f1, train_auc, _, _ = _run_epoch(
            model,
            train_loader,
            device,
            pos_weight=pos_weight,
            threshold=current_threshold,
            optimizer=optimizer,
        )
        val_loss, _, _, val_auc, val_probs, val_targets = _run_epoch(
            model,
            val_loader,
            device,
            pos_weight=pos_weight,
            threshold=current_threshold,
            optimizer=None,
        )
        current_threshold, val_acc, val_f1 = _find_best_threshold(
            val_probs,
            val_targets,
            steps=risk_cfg.threshold_search_steps,
        )
        logging.info(
            "Epoch %d/%d | train loss=%.4f acc=%.4f f1=%.4f auc=%.4f | val loss=%.4f acc=%.4f f1=%.4f auc=%.4f thr=%.3f",
            epoch + 1,
            risk_cfg.epochs,
            train_loss,
            train_acc,
            train_f1,
            train_auc,
            val_loss,
            val_acc,
            val_f1,
            val_auc,
            current_threshold,
        )
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_auc = val_auc
            best_threshold = current_threshold
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if val_loss + risk_cfg.early_stopping_min_delta < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= risk_cfg.early_stopping_patience:
                logging.info(
                    "Early stopping triggered at epoch %d (patience=%d, min_delta=%.6f).",
                    epoch + 1,
                    risk_cfg.early_stopping_patience,
                    risk_cfg.early_stopping_min_delta,
                )
                break

    if best_state is None:
        raise RuntimeError("No best checkpoint produced during risk training.")
    model.load_state_dict(best_state)
    test_loss, test_acc, test_f1, test_auc, _, _ = _run_epoch(
        model,
        test_loader,
        device,
        pos_weight=pos_weight,
        threshold=best_threshold,
        optimizer=None,
    )
    logging.info(
        "Final test | loss=%.4f acc=%.4f f1=%.4f auc=%.4f thr=%.3f",
        test_loss,
        test_acc,
        test_f1,
        test_auc,
        best_threshold,
    )

    save_dir = Path(risk_cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / "risk_cross_attention.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "hidden_dim": risk_cfg.hidden_dim,
                "num_heads": risk_cfg.num_heads,
                "num_layers": risk_cfg.num_layers,
                "dropout": risk_cfg.dropout,
                "numeric_columns": bundle.numeric_columns,
                "categorical_columns": bundle.categorical_columns,
                "categorical_cardinalities": bundle.categorical_cardinalities,
            },
        },
        model_path,
    )

    report_path = save_dir / "risk_report.json"
    report_path.write_text(
        json.dumps(
            {
                "best_val_f1": best_val_f1,
                "best_val_auc": best_val_auc,
                "best_threshold": best_threshold,
                "pos_weight": pos_weight,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "test_f1": test_f1,
                "test_auc": test_auc,
                "selected_numeric_columns": bundle.numeric_columns,
                "selected_categorical_columns": bundle.categorical_columns,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logging.info("Saved risk model to %s", model_path)
    logging.info("Saved risk report to %s", report_path)


if __name__ == "__main__":
    main()
