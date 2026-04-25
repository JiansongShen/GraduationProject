#!/usr/bin/env python3
"""Train risk prediction model with cross-attention on tabular data."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
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


def _binary_metrics(prob: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    pred = (prob >= 0.5).float()
    correct = (pred == target).float().mean().item()

    tp = ((pred == 1) & (target == 1)).sum().item()
    fp = ((pred == 1) & (target == 0)).sum().item()
    fn = ((pred == 0) & (target == 1)).sum().item()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    return correct, f1


def _run_epoch(
    model: RiskCrossAttentionModel,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_f1 = 0.0
    num_batches = 0

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
            loss = criterion(pred_prob, y)
            if is_train:
                loss.backward()
                optimizer.step()

        acc, f1 = _binary_metrics(pred_prob.detach(), y.detach())
        total_loss += loss.item()
        total_acc += acc
        total_f1 += f1
        num_batches += 1
        progress.set_postfix({"loss": f"{total_loss / num_batches:.4f}", "acc": f"{total_acc / num_batches:.4f}"})

    denom = max(1, num_batches)
    return total_loss / denom, total_acc / denom, total_f1 / denom


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

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=risk_cfg.learning_rate,
        weight_decay=risk_cfg.weight_decay,
    )

    best_val_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(risk_cfg.epochs):
        train_loss, train_acc, train_f1 = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc, val_f1 = _run_epoch(model, val_loader, criterion, device, optimizer=None)
        logging.info(
            "Epoch %d/%d | train loss=%.4f acc=%.4f f1=%.4f | val loss=%.4f acc=%.4f f1=%.4f",
            epoch + 1,
            risk_cfg.epochs,
            train_loss,
            train_acc,
            train_f1,
            val_loss,
            val_acc,
            val_f1,
        )
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No best checkpoint produced during risk training.")
    model.load_state_dict(best_state)
    test_loss, test_acc, test_f1 = _run_epoch(model, test_loader, criterion, device, optimizer=None)
    logging.info("Final test | loss=%.4f acc=%.4f f1=%.4f", test_loss, test_acc, test_f1)

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
                "test_loss": test_loss,
                "test_acc": test_acc,
                "test_f1": test_f1,
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
