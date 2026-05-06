#!/usr/bin/env python3
"""Initialize risk-training feature column intersection.

Flow:
1. Read saved origin/label pairs from risk.saved_data_dir
2. Extract pyradiomics-parsable feature names from saved samples
3. Read riskdataset table columns
4. Compute intersection and save to configured output file
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.config import Config
from core.config_loader import load_config
from data.risk_tabular import extract_pyradiomics_features_from_predict_res_and_src


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize risk feature column intersection")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    return parser.parse_args()


def _resolve_saved_data_dir(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = project_root / path
    if not path.exists():
        raise FileNotFoundError(f"saved_data_dir not found: {path}")
    return path


def _resolve_table_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = project_root / path
    if not path.exists():
        raise FileNotFoundError(f"riskdataset_path not found: {path}")
    return path


def _read_table_columns(table_path: Path) -> list[str]:
    import pandas as pd

    suffix = table_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(table_path, nrows=0)
    elif suffix == ".csv":
        df = pd.read_csv(table_path, nrows=0)
    else:
        raise ValueError(f"Unsupported riskdataset table format: {table_path}")
    return [str(col) for col in df.columns]


def _collect_feature_names(saved_data_dir: Path) -> list[str]:
    origin_paths = sorted(saved_data_dir.glob("*_origin.nii.gz"))
    label_paths = sorted(saved_data_dir.glob("*_label.nii.gz"))
    if not origin_paths or not label_paths:
        raise ValueError(f"No saved origin/label nii.gz pairs found in {saved_data_dir}")

    label_by_stem = {p.name.replace("_label.nii.gz", ""): p for p in label_paths}
    feature_names: set[str] = set()
    matched = 0
    for origin_path in origin_paths:
        stem = origin_path.name.replace("_origin.nii.gz", "")
        label_path = label_by_stem.get(stem)
        if label_path is None:
            continue
        matched += 1
        features = extract_pyradiomics_features_from_predict_res_and_src(
            predict_res_path=str(label_path),
            src_path=str(origin_path),
        )
        numeric_feature_names = [
            key
            for key, value in features.items()
            if isinstance(value, (int, float)) and not str(key).startswith("diagnostics_")
        ]
        feature_names.update(numeric_feature_names)

    if matched == 0:
        raise ValueError(f"No matching *_origin.nii.gz / *_label.nii.gz pairs found in {saved_data_dir}")
    return sorted(feature_names)


def _save_output(output_path: Path, payload: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".txt":
        output_path.write_text("\n".join(payload["intersection_columns"]), encoding="utf-8")
    else:
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg: Config = load_config(args.config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    saved_data_dir = _resolve_saved_data_dir(cfg.risk.saved_data_dir)
    riskdataset_path = _resolve_table_path(cfg.risk.riskdataset_path)
    output_path = Path(cfg.risk.feature_columns_output)
    if not output_path.is_absolute():
        output_path = project_root / output_path

    radiomics_feature_names = _collect_feature_names(saved_data_dir)
    risk_columns = _read_table_columns(riskdataset_path)
    protected = set(cfg.risk.clinic_columns + [cfg.risk.label_column])
    risk_feature_columns = sorted(
        col for col in (set(risk_columns) - protected) if not str(col).startswith("diagnostics_")
    )
    intersection = sorted(set(radiomics_feature_names).intersection(risk_feature_columns))
    missing_from_radiomics = sorted(set(risk_feature_columns) - set(radiomics_feature_names))

    payload = {
        "saved_data_dir": str(saved_data_dir),
        "riskdataset_path": str(riskdataset_path),
        "radiomics_feature_count": len(radiomics_feature_names),
        "riskdataset_column_count": len(risk_columns),
        "risk_feature_column_count": len(risk_feature_columns),
        "intersection_count": len(intersection),
        "clinic_columns": list(cfg.risk.clinic_columns),
        "intersection_columns": intersection,
        "missing_from_radiomics": missing_from_radiomics,
    }
    _save_output(output_path, payload)
    logging.info("Saved %d intersected feature columns to %s", len(intersection), output_path)


if __name__ == "__main__":
    main()
