from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

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
from ui.cta_segmentation import prepare_uploaded_volume_for_dataset, run_patch_based_segmentation
from data.morphology_features import extract_morphology_features
from model.risk.RiskCrossAttentionModel import RiskCrossAttentionModel

APP_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = APP_ROOT / "config" / "host.yaml"
UPLOAD_DIR = APP_ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Gradulate FastAPI UI")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to the YAML config file used by the UI",
    )
    return parser.parse_args()

app = FastAPI(title="Gradulate CTA UI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # type: ignore[list-item]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ARGS = parse_args()
STATE: dict[str, Any] = {"config_path": str(Path(ARGS.config).expanduser()), "config": None}
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
async def segment_cta(
    cta_file: UploadFile = File(...),
    config_path: str = Form(default=""),
    checkpoint_path: str = Form(
        default="",
        description="Optional .pth on the server; if omitted, uses checkpoint.save_dir/latest.pth from the YAML.",
    ),
) -> JSONResponse:
    """Run 3D segmentation: load YAML config and weights, resample, tile, predict, stitch, restore spacing."""
    resolved_config_path = Path(config_path.strip() or STATE["config_path"]).expanduser()
    resolved_config_path = (
        resolved_config_path.resolve()
        if resolved_config_path.is_absolute()
        else (APP_ROOT / resolved_config_path).resolve()
    )
    if not resolved_config_path.is_file():
        raise HTTPException(status_code=400, detail=f"config file not found: {resolved_config_path}")

    segmentation_configuration = load_config(resolved_config_path)

    checkpoint_path_stripped = checkpoint_path.strip()
    if checkpoint_path_stripped:
        resolved_checkpoint_path = Path(checkpoint_path_stripped).expanduser()
        resolved_checkpoint_path = (
            resolved_checkpoint_path.resolve()
            if resolved_checkpoint_path.is_absolute()
            else (APP_ROOT / resolved_checkpoint_path).resolve()
        )
    else:
        resolved_checkpoint_path = (
            APP_ROOT / Path(segmentation_configuration.checkpoint.save_dir) / "latest.pth"
        ).resolve()

    if not resolved_checkpoint_path.is_file():
        raise HTTPException(
            status_code=400,
            detail=(
                f"Checkpoint not found: {resolved_checkpoint_path}. "
                "Send form field checkpoint_path, or train once so latest.pth exists under checkpoint.save_dir."
            ),
        )
    uploaded_cta_path = _save_upload(cta_file)
    upload_identifier = uploaded_cta_path.name.removesuffix(".nii.gz").removesuffix(".nii")
    # One directory per request so ``MedicalPatchDataset`` never mixes this scan with older uploads.
    inference_workspace = UPLOAD_DIR / "segmentation_runs" / upload_identifier
    inference_workspace.mkdir(parents=True, exist_ok=True)

    dataset_ready_volume_path = prepare_uploaded_volume_for_dataset(
        uploaded_file=uploaded_cta_path,
        destination_directory=inference_workspace,
        file_patterns=list(segmentation_configuration.data.file_patterns),
        upload_identifier=upload_identifier,
    )

    # Flat names under ``uploads/`` so ``GET /files/{name}`` can serve them without nested paths.
    probability_output_path = UPLOAD_DIR / f"{upload_identifier}_segmentation_probability.nii.gz"
    binary_mask_output_path = UPLOAD_DIR / f"{upload_identifier}_segmentation_mask.nii.gz"

    device = torch.device(
        segmentation_configuration.device if torch.cuda.is_available() else "cpu"
    )

    await asyncio.to_thread(
        run_patch_based_segmentation,
        configuration=segmentation_configuration,
        checkpoint_file=resolved_checkpoint_path,
        input_volume_path=dataset_ready_volume_path,
        probability_output_path=probability_output_path,
        binary_mask_output_path=binary_mask_output_path,
        device=device,
    )

    logger.info(
        "cta segmentation finished original=%s config=%s checkpoint=%s mask=%s",
        cta_file.filename,
        resolved_config_path.name,
        resolved_checkpoint_path.name,
        binary_mask_output_path.name,
    )
    return JSONResponse(
        {
            "ok": True,
            "config_path": str(resolved_config_path),
            "checkpoint_path": str(resolved_checkpoint_path),
            "cta_path": f"/files/{uploaded_cta_path.name}",
            "mask_path": f"/files/{binary_mask_output_path.name}",
            "probability_path": f"/files/{probability_output_path.name}",
            "message": "Resampled, patch-inferred, stitched, and geometry restored to match the uploaded volume.",
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
    
    # Build metadata from frontend inputs
    metadata = json.loads(extra_metadata or "{}")
    if age:
        metadata["年龄"] = age
    if sex:
        metadata["性别"] = sex
    if ref_path:
        metadata["参考分割文件"] = ref_path.name

    # Extract morphology features from segmentation mask
    morphology_features = {}
    if ref_path:
        try:
            morphology_features = extract_morphology_features(
                image_path=str(cta_path),
                mask_path=str(ref_path),
            )
            logger.info("Extracted %d morphology features", len(morphology_features))
        except Exception as e:
            logger.warning("Failed to extract morphology features: %s", str(e))
            morphology_features = {}
    
    # Combine all features: metadata + morphology + radiomics
    radiomics_features = _extract_radiomics_features(cta_path, ref_path, config.risk)
    feature_row = {**metadata, **morphology_features, **radiomics_features}

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
            "features": {
                "morphology": morphology_features,
                "radiomics": radiomics_features,
                "metadata": metadata,
            },
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
        shutil.copyfileobj(upload.file, cast("Any", f))
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
