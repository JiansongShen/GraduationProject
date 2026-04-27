from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from core.config import Config, RiskConfig
from core.config_loader import load_config
from data.risk_tabular import build_risk_data_bundle
from model.risk.RiskCrossAttentionModel import RiskCrossAttentionModel

APP_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = APP_ROOT / "config" / "host.yaml"
UPLOAD_DIR = APP_ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Gradulate CTA UI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE: dict[str, Any] = {"config_path": str(DEFAULT_CONFIG), "config": None}
logger = logging.getLogger("gradulate.api")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    started = time.perf_counter()
    client_host = request.client.host if request.client else "unknown"
    logger.info("request start %s %s from=%s", request.method, request.url.path, client_host)
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("request failed %s %s", request.method, request.url.path)
        raise
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "request end %s %s status=%s duration_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.get("/api/config")
def get_config() -> dict[str, Any]:
    config = _load_active_config()
    return {"config_path": STATE["config_path"], "config": asdict(config)}


@app.post("/api/config")
def update_config(config_path: str = Form(default=str(DEFAULT_CONFIG))) -> dict[str, Any]:
    STATE["config_path"] = config_path
    STATE["config"] = load_config(config_path)
    return {"ok": True, "config_path": config_path}


@app.post("/api/cta/upload")
async def upload_cta(cta_file: UploadFile = File(...)) -> JSONResponse:
    temp_path = _save_upload(cta_file)
    logger.info("cta uploaded original=%s saved=%s", cta_file.filename, temp_path.name)
    return JSONResponse({"ok": True, "file_path": f"/files/{temp_path.name}"})


@app.post("/api/cta/segment")
async def segment_cta(cta_file: UploadFile = File(...)) -> JSONResponse:
    temp_path = _save_upload(cta_file)
    mask_base = temp_path.name.removesuffix(".nii.gz")
    mask_path = temp_path.with_name(f"{mask_base}_mask.nii.gz")
    shutil.copyfile(temp_path, mask_path)
    logger.info(
        "cta segmented original=%s cta=%s mask=%s",
        cta_file.filename,
        temp_path.name,
        mask_path.name,
    )
    return JSONResponse(
        {
            "ok": True,
            "cta_path": f"/files/{temp_path.name}",
            "mask_path": f"/files/{mask_path.name}",
            "message": "已完成示例分割，实际分割模型可在这里替换。",
        }
    )


@app.post("/api/risk/predict")
async def predict_risk(
    cta_file: UploadFile = File(...),
    reference_file: UploadFile | None = File(default=None),
    age: str = Form(default=""),
    sex: str = Form(default=""),
    extra_metadata: str = Form(default="{}"),
) -> JSONResponse:
    logger.info(
        "risk predict requested cta=%s reference=%s age=%s sex=%s",
        cta_file.filename if cta_file else None,
        reference_file.filename if reference_file else None,
        age,
        sex,
    )
    config = _load_active_config()
    if not config.risk.enabled:
        raise HTTPException(status_code=400, detail="risk.enabled is false in config")

    cta_path = _save_upload(cta_file)
    ref_path = _save_upload(reference_file) if reference_file else None
    metadata = json.loads(extra_metadata or "{}")
    if age:
        metadata["年龄"] = age
    if sex:
        metadata["性别"] = sex
    if ref_path:
        metadata["参考分割文件"] = ref_path.name

    features = _extract_radiomics_features(cta_path, ref_path, config.risk)
    feature_row = {**metadata, **features}

    bundle = build_risk_data_bundle(config.risk)
    model = RiskCrossAttentionModel(
        num_numeric_features=len(bundle.numeric_columns),
        categorical_cardinalities=bundle.categorical_cardinalities,
        hidden_dim=config.risk.hidden_dim,
        num_heads=config.risk.num_heads,
        num_layers=config.risk.num_layers,
        dropout=config.risk.dropout,
    )
    model.eval()

    numeric_df = pd.DataFrame([feature_row])
    numeric_x = _align_numeric_features(numeric_df, bundle.numeric_columns)
    categorical_x = _align_categorical_features(numeric_df, bundle.categorical_columns)
    with torch.no_grad():
        prob = float(model(torch.from_numpy(numeric_x), torch.from_numpy(categorical_x)).item())

    return JSONResponse(
        {
            "ok": True,
            "risk_probability": prob,
            "features": features,
            "cta_path": cta_path.name,
            "reference_path": ref_path.name if ref_path else None,
        }
    )


@app.get("/files/{name}")
def files(name: str) -> FileResponse:
    file_path = UPLOAD_DIR / name
    if not file_path.exists():
        logger.warning("file requested but missing name=%s", name)
        raise HTTPException(status_code=404, detail="file not found")
    logger.info("file served name=%s", name)
    return FileResponse(file_path)


def _load_active_config() -> Config:
    if STATE["config"] is None:
        STATE["config"] = load_config(STATE["config_path"])
    return STATE["config"]


def _save_upload(upload: UploadFile | None) -> Path:
    if upload is None:
        raise HTTPException(status_code=400, detail="missing upload file")
    upload_name = (upload.filename or "upload.nii.gz").lower()
    if upload_name.endswith(".nii.gz"):
        suffix = ".nii.gz"
    else:
        suffix = Path(upload_name).suffix or ".nii.gz"
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest


def _extract_radiomics_features(cta_path: Path, mask_path: Path | None, risk_cfg: RiskConfig) -> dict[str, float]:
    try:
        import SimpleITK as sitk
        from radiomics import featureextractor
    except Exception as exc:  # pragma: no cover - runtime dependency availability
        raise HTTPException(status_code=500, detail=f"radiomics dependency unavailable: {exc}") from exc

    extractor = featureextractor.RadiomicsFeatureExtractor()
    image = sitk.ReadImage(str(cta_path))
    if mask_path and mask_path.exists():
        mask = sitk.ReadImage(str(mask_path))
    else:
        mask = sitk.Image(image.GetSize(), sitk.sitkUInt8)
        mask.CopyInformation(image)
        mask = sitk.Add(mask, 1)

    result = extractor.execute(image, mask)
    numeric_features: dict[str, float] = {}
    for key, value in result.items():
        if key.startswith("diagnostics_"):
            continue
        if isinstance(value, (int, float, np.number)):
            numeric_features[key] = float(value)

    if not numeric_features:
        numeric_features = {"original_firstorder_Mean": 0.0}
    return numeric_features


def _align_numeric_features(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    aligned = pd.DataFrame(index=df.index)
    for col in columns:
        aligned[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0.0)
    x = aligned.to_numpy(dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x


def _align_categorical_features(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    if not columns:
        return np.zeros((len(df), 0), dtype=np.int64)
    aligned = []
    for col in columns:
        values = df.get(col, "missing").astype(str)
        aligned.append(pd.factorize(values)[0].astype(np.int64))
    return np.stack(aligned, axis=1)


if __name__ == "__main__":
    uvicorn.run("ui.main:app", host="0.0.0.0", port=8000, reload=True)
