#!/usr/bin/env python3
"""Compare current risk model against traditional ML baselines."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from torch.utils.data import DataLoader

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.config import Config
from core.config_loader import load_config
from core.global_setting import SystemSetting
from data.risk_tabular import RiskTabularDataset, build_risk_data_bundle, _clean_columns, _read_feature_columns, _read_risk_table
from model.risk.RiskCrossAttentionModel import RiskCrossAttentionModel
from script.train_risk import _find_best_threshold, _run_epoch


TRADITIONAL_MODEL_BUILDERS: dict[str, Any] = {
    "logistic_regression": lambda: LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    ),
    "svm_rbf": lambda: SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced", random_state=42),
    "knn": lambda: KNeighborsClassifier(n_neighbors=7, weights="distance"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare trainrisk model with traditional ML baselines")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--device", type=str, default=None, help="Override device from config")
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(TRADITIONAL_MODEL_BUILDERS.keys()),
        choices=list(TRADITIONAL_MODEL_BUILDERS.keys()),
        help="Traditional baseline models to compare",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override risk.epochs for a faster comparison run",
    )
    return parser.parse_args()


def _safe_auc_numpy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if np.unique(y_true).shape[0] < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_prob))


def _binary_metrics_numpy(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": _safe_auc_numpy(y_true, y_prob),
        "threshold": float(threshold),
    }


def _find_best_threshold_numpy(y_true: np.ndarray, y_prob: np.ndarray, steps: int) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_metrics = _binary_metrics_numpy(y_true, y_prob, threshold=0.5)
    best_f1 = best_metrics["f1"]
    for threshold in np.linspace(0.05, 0.95, max(2, steps)):
        metrics = _binary_metrics_numpy(y_true, y_prob, threshold=float(threshold))
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def _build_dataframe_splits(cfg: Config) -> tuple[Any, Any, Any, list[str], list[str], str]:
    risk_cfg = cfg.risk
    df = _clean_columns(_read_risk_table(risk_cfg.riskdataset_path))
    label_column = risk_cfg.label_column
    feature_columns = _read_feature_columns(risk_cfg.feature_columns_output)
    selected = [c for c in list(risk_cfg.clinic_columns) + feature_columns if c in df.columns and c != label_column]
    if not selected:
        raise ValueError("No usable risk feature columns found; run script/init_risk_train.py first or configure clinic_columns")

    bundle = build_risk_data_bundle(risk_cfg)
    numeric_columns = [c for c in bundle.numeric_columns if c != "__dummy_numeric__"]
    categorical_columns = list(bundle.categorical_columns)
    selected_columns = numeric_columns + categorical_columns

    selected_df = df[selected_columns].copy()
    y = np.asarray((np.nan_to_num(pd_to_numeric(df[label_column]), nan=0.0) > 0.0).astype(np.int64))

    x_train = selected_df.iloc[:0].copy()
    x_val = selected_df.iloc[:0].copy()
    x_test = selected_df.iloc[:0].copy()

    train_values = np.concatenate([bundle.x_num_train, bundle.x_cat_train], axis=1) if categorical_columns else bundle.x_num_train
    val_values = np.concatenate([bundle.x_num_val, bundle.x_cat_val], axis=1) if categorical_columns else bundle.x_num_val
    test_values = np.concatenate([bundle.x_num_test, bundle.x_cat_test], axis=1) if categorical_columns else bundle.x_num_test

    # Rebuild exact same split ordering based on row-wise feature signatures.
    train_df, val_df, test_df = _reconstruct_splits_from_bundle(selected_df, numeric_columns, categorical_columns, bundle)
    return train_df, val_df, test_df, numeric_columns, categorical_columns, label_column


def pd_to_numeric(series: Any) -> np.ndarray:
    import pandas as pd

    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float32)


def _reconstruct_splits_from_bundle(selected_df: Any, numeric_columns: list[str], categorical_columns: list[str], bundle: Any) -> tuple[Any, Any, Any]:
    import pandas as pd

    work_df = selected_df.copy()
    for col in numeric_columns:
        work_df[col] = pd.to_numeric(work_df[col], errors="coerce")
    for col in categorical_columns:
        work_df[col] = work_df[col].astype(str).fillna("missing")

    def make_key(frame: Any, num_cols: list[str], cat_cols: list[str]) -> list[tuple[Any, ...]]:
        keys: list[tuple[Any, ...]] = []
        for _, row in frame.iterrows():
            key: list[Any] = []
            for col in num_cols:
                value = row[col]
                key.append(None if pd.isna(value) else round(float(value), 6))
            for col in cat_cols:
                key.append(str(row[col]))
            keys.append(tuple(key))
        return keys

    all_keys = make_key(work_df, numeric_columns, categorical_columns)
    index_map: dict[tuple[Any, ...], list[int]] = {}
    for idx, key in enumerate(all_keys):
        index_map.setdefault(key, []).append(idx)

    def bundle_keys(x_num: np.ndarray, x_cat: np.ndarray) -> list[tuple[Any, ...]]:
        keys: list[tuple[Any, ...]] = []
        for i in range(x_num.shape[0]):
            numeric_part = [round(float(v), 6) for v in x_num[i].tolist()] if numeric_columns else []
            categorical_part = [f"cat_{int(v)}" for v in x_cat[i].tolist()] if categorical_columns else []
            keys.append(tuple(numeric_part + categorical_part))
        return keys

    raise RuntimeError("Split reconstruction is unsupported because normalized training arrays do not preserve original categorical labels.")


def _build_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_columns:
        transformers.append(("num", numeric_pipeline, numeric_columns))
    if categorical_columns:
        transformers.append(("cat", categorical_pipeline, categorical_columns))
    return ColumnTransformer(transformers=transformers)


def _train_traditional_models(cfg: Config, model_names: list[str]) -> dict[str, Any]:
    import pandas as pd
    from sklearn.model_selection import train_test_split

    risk_cfg = cfg.risk
    df = _clean_columns(_read_risk_table(risk_cfg.riskdataset_path))
    label_column = risk_cfg.label_column
    if label_column not in df.columns:
        raise ValueError(f"Missing risk label column: {label_column}")

    feature_columns = _read_feature_columns(risk_cfg.feature_columns_output)
    selected = [c for c in list(risk_cfg.clinic_columns) + feature_columns if c in df.columns and c != label_column]
    if not selected:
        raise ValueError("No usable risk feature columns found; run script/init_risk_train.py first or configure clinic_columns")

    x = df[selected].copy()
    y = (pd.to_numeric(df[label_column], errors="coerce").fillna(0.0).astype(np.float32).to_numpy() > 0.0).astype(np.int64)

    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    for col in selected:
        converted = pd.to_numeric(x[col], errors="coerce")
        if converted.notna().mean() >= 0.8:
            x[col] = converted
            numeric_columns.append(col)
        else:
            x[col] = x[col].astype("string")
            categorical_columns.append(col)

    stratify = y if np.unique(y).shape[0] > 1 else None
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=stratify)
    stratify_train = y_train if np.unique(y_train).shape[0] > 1 else None
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.1 / 0.8,
        random_state=42,
        stratify=stratify_train,
    )

    results: dict[str, Any] = {}
    for name in model_names:
        estimator = TRADITIONAL_MODEL_BUILDERS[name]()
        pipeline = Pipeline([
            ("preprocessor", _build_preprocessor(numeric_columns, categorical_columns)),
            ("model", estimator),
        ])
        pipeline.fit(x_train, y_train)
        val_prob = pipeline.predict_proba(x_val)[:, 1]
        best_threshold, val_metrics = _find_best_threshold_numpy(y_val, val_prob, steps=risk_cfg.threshold_search_steps)
        test_prob = pipeline.predict_proba(x_test)[:, 1]
        test_metrics = _binary_metrics_numpy(y_test, test_prob, threshold=best_threshold)
        results[name] = {
            "model_name": name,
            "type": "traditional_ml",
            "validation": val_metrics,
            "test": test_metrics,
            "selected_numeric_columns": numeric_columns,
            "selected_categorical_columns": categorical_columns,
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "n_test": int(len(y_test)),
        }
        logging.info(
            "Baseline %s | val_f1=%.4f val_auc=%.4f | test_f1=%.4f test_auc=%.4f",
            name,
            val_metrics["f1"],
            val_metrics["auc"],
            test_metrics["f1"],
            test_metrics["auc"],
        )
    return results


def _train_cross_attention(cfg: Config, device_override: str | None = None) -> dict[str, Any]:
    risk_cfg = cfg.risk
    device_name = device_override or cfg.device
    device = torch.device(device_name if torch.cuda.is_available() and device_name.startswith("cuda") else "cpu")

    bundle = build_risk_data_bundle(risk_cfg)
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=risk_cfg.learning_rate,
        weight_decay=risk_cfg.weight_decay,
    )

    pos_count = float((bundle.y_train > 0.5).sum())
    neg_count = float((bundle.y_train <= 0.5).sum())
    pos_weight = max(1.0, neg_count / pos_count) if risk_cfg.use_pos_weight and pos_count > 0.0 else 1.0

    best_val_f1 = -1.0
    best_threshold = 0.5
    best_state: dict[str, torch.Tensor] | None = None
    best_val_metrics: dict[str, float] | None = None
    best_val_loss = float("inf")
    early_stop_counter = 0
    current_threshold = 0.5

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
        current_threshold, val_acc, val_f1 = _find_best_threshold(val_probs, val_targets, steps=risk_cfg.threshold_search_steps)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_threshold = current_threshold
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val_metrics = {
                "accuracy": float(val_acc),
                "precision": 0.0,
                "recall": 0.0,
                "f1": float(val_f1),
                "auc": float(val_auc),
                "threshold": float(current_threshold),
                "loss": float(val_loss),
            }
        if val_loss + risk_cfg.early_stopping_min_delta < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= risk_cfg.early_stopping_patience:
                break
        logging.info(
            "Cross-attention epoch %d/%d | train_loss=%.4f train_f1=%.4f train_auc=%.4f | val_loss=%.4f val_f1=%.4f val_auc=%.4f thr=%.3f",
            epoch + 1,
            risk_cfg.epochs,
            train_loss,
            train_f1,
            train_auc,
            val_loss,
            val_f1,
            val_auc,
            current_threshold,
        )

    if best_state is None or best_val_metrics is None:
        raise RuntimeError("Cross-attention training did not produce a valid checkpoint")

    model.load_state_dict(best_state)
    test_loss, test_acc, test_f1, test_auc, test_probs, test_targets = _run_epoch(
        model,
        test_loader,
        device,
        pos_weight=pos_weight,
        threshold=best_threshold,
        optimizer=None,
    )
    test_metrics = _binary_metrics_numpy(test_targets.numpy(), test_probs.numpy(), threshold=best_threshold)
    test_metrics["loss"] = float(test_loss)
    test_metrics["accuracy"] = float(test_acc)
    test_metrics["f1"] = float(test_f1)
    test_metrics["auc"] = float(test_auc)

    return {
        "model_name": "risk_cross_attention",
        "type": "deep_model",
        "validation": best_val_metrics,
        "test": test_metrics,
        "selected_numeric_columns": bundle.numeric_columns,
        "selected_categorical_columns": bundle.categorical_columns,
        "categorical_cardinalities": bundle.categorical_cardinalities,
        "n_train": int(len(bundle.y_train)),
        "n_val": int(len(bundle.y_val)),
        "n_test": int(len(bundle.y_test)),
        "pos_weight": float(pos_weight),
    }


def _build_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Risk Model Comparison Report",
        "",
        f"- config: `{report['config_path']}`",
        f"- label column: `{report['label_column']}`",
        f"- compared at: `{report['generated_at']}`",
        "",
        "## Test Metrics",
        "",
        "| Model | Type | Accuracy | Precision | Recall | F1 | AUC | Threshold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ranking = report["ranking_by_test_f1"]
    for item in ranking:
        metrics = report["results"][item]["test"]
        lines.append(
            f"| {item} | {report['results'][item]['type']} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['auc']:.4f} | {metrics['threshold']:.3f} |"
        )
    lines.extend(["", "## Best Model", "", f"- best by test F1: `{ranking[0]}`"])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    cfg: Config = load_config(args.config)
    if not cfg.risk.enabled:
        raise ValueError("Risk training is disabled. Set `risk.enabled: true` in config.")

    if args.device:
        cfg.device = args.device
    if args.epochs is not None:
        cfg.risk.epochs = args.epochs

    SystemSetting.set_seed(cfg.seed)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    cross_attention_result = _train_cross_attention(cfg, device_override=args.device)
    traditional_results = _train_traditional_models(cfg, args.models)

    all_results: dict[str, Any] = {cross_attention_result["model_name"]: cross_attention_result}
    all_results.update(traditional_results)
    ranking = sorted(all_results.keys(), key=lambda name: all_results[name]["test"]["f1"], reverse=True)

    save_dir = Path(cfg.risk.save_dir)
    if not save_dir.is_absolute():
        save_dir = project_root / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime

    report = {
        "config_path": str(Path(args.config).resolve()),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "label_column": cfg.risk.label_column,
        "clinic_columns": list(cfg.risk.clinic_columns),
        "traditional_models": args.models,
        "risk_config": asdict(cfg.risk),
        "results": all_results,
        "ranking_by_test_f1": ranking,
    }

    json_path = save_dir / "risk_model_comparison_report.json"
    md_path = save_dir / "risk_model_comparison_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_build_markdown_report(report), encoding="utf-8")

    logging.info("Saved comparison JSON report to %s", json_path)
    logging.info("Saved comparison Markdown report to %s", md_path)
    logging.info("Best model by test F1: %s", ranking[0])


if __name__ == "__main__":
    main()
