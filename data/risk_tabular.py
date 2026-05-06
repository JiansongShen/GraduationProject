from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from core.config import RiskConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RiskDataBundle:
    x_num_train: np.ndarray
    x_cat_train: np.ndarray
    y_train: np.ndarray
    x_num_val: np.ndarray
    x_cat_val: np.ndarray
    y_val: np.ndarray
    x_num_test: np.ndarray
    x_cat_test: np.ndarray
    y_test: np.ndarray
    numeric_columns: list[str]
    categorical_columns: list[str]
    categorical_cardinalities: list[int]


class RiskTabularDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, x_num: np.ndarray, x_cat: np.ndarray, y: np.ndarray) -> None:
        self.x_num = torch.from_numpy(x_num.astype(np.float32))
        self.x_cat = torch.from_numpy(x_cat.astype(np.int64))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x_num[index], self.x_cat[index], self.y[index]


def extract_pyradiomics_features_from_predict_res_and_src(predict_res_path: str, src_path: str) -> dict[str, float]:
    """Extract the same broad pyradiomics feature namespace used by init/test scripts."""
    try:
        import SimpleITK as sitk
        from radiomics import featureextractor
    except Exception as exc:
        logger.warning("Pyradiomics dependency unavailable: %s", exc)
        return {}

    extractor = featureextractor.RadiomicsFeatureExtractor(
        binWidth=25.0,
        resampledPixelSpacing=None,
        interpolator=sitk.sitkBSpline,
    )
    extractor.enableImageTypes(
        Original={},
        Wavelet={},
        LoG={"sigma": [1.0, 3.0, 5.0]},
        Exponential={},
        Gradient={},
        LBP2D={},
        LBP3D={},
    )
    extractor.enableAllFeatures()
    result = extractor.execute(sitk.ReadImage(src_path), sitk.ReadImage(predict_res_path))
    features: dict[str, float] = {}
    for key, value in result.items():
        key_str = str(key)
        if key_str.startswith("diagnostics_"):
            features[key_str] = 0.0
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric_value):
            features[key_str] = numeric_value
    return features


def _read_risk_table(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"riskdataset_path not found: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported risk dataset format: {path}")


def _read_feature_columns(path_str: str) -> list[str]:
    path = Path(path_str)
    if not path.exists():
        return []
    if path.suffix.lower() == ".txt":
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(v) for v in payload]
    return [str(v) for v in payload.get("intersection_columns", [])]


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.dropna(axis=1, how="all").drop_duplicates().copy()
    cleaned.columns = [str(c).strip() for c in cleaned.columns]
    return cleaned


def _split_numeric_categorical(df: pd.DataFrame, columns: list[str]) -> tuple[list[str], list[str]]:
    numeric, categorical = [], []
    for col in columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().mean() >= 0.8:
            df[col] = converted
            numeric.append(col)
        else:
            categorical.append(col)
    return numeric, categorical


def build_risk_data_bundle(cfg: RiskConfig) -> RiskDataBundle:
    """Use only clinic columns + init_risk_train intersection columns."""
    df = _clean_columns(_read_risk_table(cfg.riskdataset_path))
    label_column = cfg.label_column
    if label_column not in df.columns:
        raise ValueError(f"Missing risk label column: {label_column}")

    feature_columns = _read_feature_columns(cfg.feature_columns_output)
    feature_columns = [c for c in feature_columns if not str(c).startswith("diagnostics_")]
    clinic_columns = [c for c in list(cfg.clinic_columns) if not str(c).startswith("diagnostics_")]
    selected = [c for c in clinic_columns + feature_columns if c in df.columns and c != label_column]
    if not selected:
        raise ValueError("No usable risk feature columns found; run script/init_risk_train.py first or configure clinic_columns")

    y = pd.to_numeric(df[label_column], errors="coerce").fillna(0.0).astype(np.float32).to_numpy()
    y = (y > 0.0).astype(np.float32)
    numeric_columns, categorical_columns = _split_numeric_categorical(df, selected)

    numeric_df = df[numeric_columns].copy() if numeric_columns else pd.DataFrame(index=df.index)
    for col in numeric_columns:
        median = float(numeric_df[col].median()) if numeric_df[col].notna().any() else 0.0
        numeric_df[col] = numeric_df[col].fillna(median)
    x_num = numeric_df.to_numpy(dtype=np.float32) if numeric_columns else np.zeros((len(df), 1), dtype=np.float32)
    if numeric_columns:
        mean = x_num.mean(axis=0, keepdims=True)
        std = x_num.std(axis=0, keepdims=True)
        std[std < 1e-8] = 1.0
        x_num = (x_num - mean) / std

    cat_arrays: list[np.ndarray] = []
    categorical_cardinalities: list[int] = []
    for col in categorical_columns:
        values = df[col].astype(str).fillna("missing")
        categories = sorted(values.unique().tolist())
        mapping = {value: idx for idx, value in enumerate(categories)}
        cat_arrays.append(values.map(mapping).fillna(0).astype(np.int64).to_numpy())
        categorical_cardinalities.append(len(mapping))
    x_cat = np.stack(cat_arrays, axis=1) if cat_arrays else np.zeros((len(df), 0), dtype=np.int64)

    stratify = y if (y == 0).sum() > 0 and (y == 1).sum() > 0 else None
    x_num_train, x_num_test, x_cat_train, x_cat_test, y_train, y_test = train_test_split(
        x_num, x_cat, y, test_size=0.2, random_state=42, stratify=stratify
    )
    val_ratio = 0.1 / 0.8
    stratify_train = y_train if (y_train == 0).sum() > 0 and (y_train == 1).sum() > 0 else None
    x_num_train, x_num_val, x_cat_train, x_cat_val, y_train, y_val = train_test_split(
        x_num_train,
        x_cat_train,
        y_train,
        test_size=val_ratio,
        random_state=42,
        stratify=stratify_train,
    )

    return RiskDataBundle(
        x_num_train=x_num_train,
        x_cat_train=x_cat_train,
        y_train=y_train,
        x_num_val=x_num_val,
        x_cat_val=x_cat_val,
        y_val=y_val,
        x_num_test=x_num_test,
        x_cat_test=x_cat_test,
        y_test=y_test,
        numeric_columns=numeric_columns if numeric_columns else ["__dummy_numeric__"],
        categorical_columns=categorical_columns,
        categorical_cardinalities=categorical_cardinalities,
    )
