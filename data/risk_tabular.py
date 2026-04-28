from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from core.config import RiskConfig

logger = logging.getLogger(__name__)

DEFAULT_CLINIC_COLUMNS = [
    "性别",
    "年龄",
    "高血压",
    "心脏病",
    "糖尿病",
    "脑血管硬化",
    "饮酒",
    "抽烟",
    "出血史",
]


@dataclass(frozen=True)
class RiskDataBundle:
    """Preprocessed tabular bundle for training and evaluation."""

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
    """Torch dataset wrapper for tabular risk features."""

    def __init__(self, x_num: np.ndarray, x_cat: np.ndarray, y: np.ndarray) -> None:
        self.x_num = torch.from_numpy(x_num.astype(np.float32))
        self.x_cat = torch.from_numpy(x_cat.astype(np.int64))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x_num[index], self.x_cat[index], self.y[index]


def _generate_dummy_dataframe(cfg: RiskConfig) -> pd.DataFrame:
    """Generate reproducible metadata dummy data."""
    rng = np.random.default_rng(cfg.random_state)
    num_rows = max(32, int(cfg.dummy_num_rows))

    age = rng.integers(20, 90, size=num_rows)
    sex = rng.choice(["M", "F"], size=num_rows, p=[0.48, 0.52])
    lesion_site = rng.choice(["ICA", "MCA", "ACom", "PCom", "BA"], size=num_rows)
    diameter_mm = rng.uniform(1.0, 18.0, size=num_rows)
    irregular = rng.choice([0, 1], size=num_rows, p=[0.65, 0.35])
    smoking = rng.choice([0, 1], size=num_rows, p=[0.72, 0.28])

    logit = -5.2 + 0.06 * age + 0.22 * diameter_mm + 0.85 * irregular + 0.45 * smoking
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(0.0, 1.0, size=num_rows) < prob).astype(np.int64)

    return pd.DataFrame(
        {
            cfg.id_column: [f"dummy_{i:05d}" for i in range(num_rows)],
            "年龄": age,
            "性别": sex,
            "部位": lesion_site,
            "直径_mm": diameter_mm,
            "是否不规则": irregular,
            "是否吸烟": smoking,
            cfg.label_column: y,
        }
    )


def _load_dataframe(cfg: RiskConfig) -> pd.DataFrame:
    if cfg.use_dummy_metadata:
        return _generate_dummy_dataframe(cfg)

    excel_path = cfg.excel_path
    return pd.read_excel(excel_path)


def extract_pyradiomics_features_from_predict_res_and_src(
    predict_res_path: str,
    src_path: str,
) -> dict[str, float]:
    """Extract numeric pyradiomics features from prediction mask and source volume."""
    try:
        import SimpleITK as sitk
        from radiomics import featureextractor
    except Exception as exc:  # pragma: no cover - runtime dependency
        logger.warning("Pyradiomics dependency unavailable: %s", exc)
        return {}

    image = sitk.ReadImage(src_path)
    mask = sitk.ReadImage(predict_res_path)
    extractor = featureextractor.RadiomicsFeatureExtractor(
        binWidth=25.0,
        resampledPixelSpacing=None,
        interpolator=sitk.sitkBSpline,
        verbose=False,
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
    result = extractor.execute(image, mask)
    numeric_features: dict[str, float] = {}
    for key, value in result.items():
        if key.startswith("diagnostics_"):
            continue
        if isinstance(value, (int, float, np.number)):
            numeric_features[key] = float(value)
    return numeric_features


def _resolve_saved_data_paths(cfg: RiskConfig) -> tuple[Path | None, Path | None]:
    if cfg.predict_res_path and cfg.src_path:
        predict_res_path = Path(cfg.predict_res_path)
        src_path = Path(cfg.src_path)
        return predict_res_path, src_path

    saved_data_dir = Path(cfg.saved_data_dir)
    if not saved_data_dir.is_absolute():
        saved_data_dir = Path(__file__).resolve().parent.parent / saved_data_dir
    if not saved_data_dir.exists():
        return None, None

    src_candidates = sorted(saved_data_dir.glob("*_origin.nii.gz"))
    predict_res_candidates = sorted(saved_data_dir.glob("*_label.nii.gz"))
    if not src_candidates or not predict_res_candidates:
        return None, None
    return predict_res_candidates[0], src_candidates[0]


def _select_overlap_and_clinic_columns(raw_df: pd.DataFrame, cfg: RiskConfig) -> pd.DataFrame:
    """Keep only overlap(pyradiomics, train columns) + configured clinic columns."""
    predict_res_path, src_path = _resolve_saved_data_paths(cfg)
    overlap_columns: list[str] = []
    if predict_res_path and src_path:
        try:
            pyradiomics_features = extract_pyradiomics_features_from_predict_res_and_src(
                predict_res_path=str(predict_res_path),
                src_path=str(src_path),
            )
            overlap_columns = [col for col in raw_df.columns if col in pyradiomics_features]
            logger.info(
                "Selected %d overlapped pyradiomics columns from %s and %s.",
                len(overlap_columns),
                predict_res_path,
                src_path,
            )
        except Exception as exc:
            logger.warning("Failed to compute overlap pyradiomics columns: %s", exc)
    else:
        logger.warning("No predict_res/src files found, skipped pyradiomics overlap selection.")

    clinic_columns = cfg.clinic_columns or DEFAULT_CLINIC_COLUMNS
    clinic_present = [col for col in clinic_columns if col in raw_df.columns]
    selected = overlap_columns + [col for col in clinic_present if col not in overlap_columns]
    required_columns = [cfg.id_column, cfg.label_column]
    selected += [col for col in required_columns if col in raw_df.columns and col not in selected]

    if not selected:
        logger.warning("No overlap/clinic columns selected, falling back to original dataframe.")
        return raw_df
    return raw_df[selected].copy()


def _clean_dataframe(df: pd.DataFrame, cfg: RiskConfig) -> pd.DataFrame:
    """Basic robust cleaning for tabular medical data."""
    if not cfg.use_cleaned_data:
        return df.copy()

    cleaned = df.copy()
    cleaned.columns = [str(c).strip() for c in cleaned.columns]

    # Drop fully empty columns and duplicated rows.
    cleaned = cleaned.dropna(axis=1, how="all").drop_duplicates()

    # Convert common numeric-like strings to numeric.
    for col in cleaned.columns:
        if cleaned[col].dtype == object:
            as_numeric = pd.to_numeric(cleaned[col], errors="coerce")
            # Keep numeric conversion only when most rows are parseable.
            if as_numeric.notna().mean() >= 0.8:
                cleaned[col] = as_numeric

    return cleaned


def _validate_columns(df: pd.DataFrame, cfg: RiskConfig) -> None:
    required_columns = [cfg.label_column, cfg.id_column]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column `{column}` in risk dataset.")


def _select_feature_columns(df: pd.DataFrame, cfg: RiskConfig) -> list[str]:
    if cfg.feature_columns:
        missing = [c for c in cfg.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Configured feature columns not found: {missing}")
        return cfg.feature_columns

    ignore_columns = {cfg.label_column, cfg.id_column}
    return [col for col in df.columns if col not in ignore_columns]


def _split_numeric_and_categorical(df: pd.DataFrame, columns: list[str]) -> tuple[list[str], list[str]]:
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_columns.append(col)
        else:
            categorical_columns.append(col)
    return numeric_columns, categorical_columns


def _select_top_numeric_features(
    x_num: pd.DataFrame,
    y: np.ndarray,
    cfg: RiskConfig,
) -> list[str]:
    if x_num.shape[1] <= cfg.auto_max_features:
        return list(x_num.columns)
    if cfg.feature_select_method != "mutual_info":
        raise ValueError(f"Unsupported feature selection method: {cfg.feature_select_method}")

    scores = mutual_info_classif(x_num.fillna(x_num.median()), y, discrete_features=False, random_state=cfg.random_state)
    ranked = sorted(zip(x_num.columns, scores), key=lambda item: item[1], reverse=True)
    return [name for name, _ in ranked[: cfg.auto_max_features]]

def build_risk_data_bundle(cfg: RiskConfig) -> RiskDataBundle:
    """Load, preprocess, split and select columns for risk task."""
    raw_df = _load_dataframe(cfg)
    if cfg.use_pyradiomics_overlap_only:
        raw_df = _select_overlap_and_clinic_columns(raw_df, cfg)
    df = _clean_dataframe(raw_df, cfg)
    _validate_columns(df, cfg)

    y = pd.to_numeric(df[cfg.label_column], errors="coerce").fillna(0.0).astype(np.float32).to_numpy()
    y = (y > 0.0).astype(np.float32)

    all_feature_columns = _select_feature_columns(df, cfg)
    numeric_columns, categorical_columns = _split_numeric_and_categorical(df, all_feature_columns)

    numeric_df = df[numeric_columns].copy() if numeric_columns else pd.DataFrame(index=df.index)
    categorical_df = df[categorical_columns].copy() if categorical_columns else pd.DataFrame(index=df.index)

    if numeric_columns:
        selected_numeric = _select_top_numeric_features(numeric_df, y.astype(np.int64), cfg)
        numeric_df = numeric_df[selected_numeric]
        numeric_columns = selected_numeric

    for col in numeric_columns:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")
        median_value = float(numeric_df[col].median()) if numeric_df[col].notna().any() else 0.0
        numeric_df[col] = numeric_df[col].fillna(median_value)

    for col in categorical_columns:
        categorical_df[col] = categorical_df[col].astype(str).fillna("missing")

    x_num = numeric_df.to_numpy(dtype=np.float32) if numeric_columns else np.zeros((len(df), 1), dtype=np.float32)
    if numeric_columns:
        mean = x_num.mean(axis=0, keepdims=True)
        std = x_num.std(axis=0, keepdims=True)
        std[std < 1e-8] = 1.0
        x_num = (x_num - mean) / std

    cat_mappings: list[dict[str, int]] = []
    cat_arrays: list[np.ndarray] = []
    categorical_cardinalities: list[int] = []
    for col in categorical_columns:
        value_counts = categorical_df[col].value_counts(dropna=False)
        rare_values = set(value_counts[value_counts < cfg.categorical_min_frequency].index.tolist())
        values = categorical_df[col].apply(lambda v: "__RARE__" if v in rare_values else v)
        categories = sorted(values.unique().tolist())
        mapping = {value: idx for idx, value in enumerate(categories)}
        encoded = values.map(mapping).fillna(0).astype(np.int64).to_numpy()
        cat_mappings.append(mapping)
        cat_arrays.append(encoded)
        categorical_cardinalities.append(len(mapping))

    x_cat = np.stack(cat_arrays, axis=1) if cat_arrays else np.zeros((len(df), 0), dtype=np.int64)

    stratify_target = y if (y == 0).sum() > 0 and (y == 1).sum() > 0 else None
    x_num_train, x_num_test, x_cat_train, x_cat_test, y_train, y_test = train_test_split(
        x_num,
        x_cat,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=stratify_target,
    )

    val_ratio = cfg.val_size / (1.0 - cfg.test_size)
    stratify_train = y_train if (y_train == 0).sum() > 0 and (y_train == 1).sum() > 0 else None
    x_num_train, x_num_val, x_cat_train, x_cat_val, y_train, y_val = train_test_split(
        x_num_train,
        x_cat_train,
        y_train,
        test_size=val_ratio,
        random_state=cfg.random_state,
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
