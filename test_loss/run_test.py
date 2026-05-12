#!/usr/bin/env python3
"""项目最小可运行实现。

功能：
1. 直接在脚本里写死数据与训练参数，不依赖配置文件。
2. 自动查找 image/label 配对并加载 NIfTI。
3. 将图像/标签重采样到统一 spacing。
4. 训练时仅采样前景 patch。
5. 使用最小 3D UNet 完成分割训练。
6. eval 时对整例顺序滑窗推理并计算 Dice / IoU。

默认文件命名约定：
- 图像：*_origin.nii.gz
- 标签：*_label.nii.gz
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot
from tensorboard.plugins.hparams import metrics
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from LossMax import LossMax
from test_loss.MultiScaleWeightNet import MultiScaleWeightNet

# 根据 config/host.yaml 写死后的最小配置。
TRAIN_DIRS = [
    "/home/napbad/project/graduate_proj/core_model/dataset/SingleCrop",
]
EVAL_DIRS = [
    "/home/napbad/project/graduate_proj/core_model/dataset/Single",
]

IMAGE_SUFFIXES = ["_origin.nii.gz", "_brainpre.nii.gz"]
LABEL_SUFFIXES = ["_label.nii.gz", "_ias.nii.gz"]
TARGET_SPACING = (1.0, 1.0, 1.0)
PATCH_SIZE = (16, 16, 16)  # (D, H, W)，与 train.patch_size / overlap_inference.patch_size 对齐
PATCHES_PER_CASE = 4
MAX_LOAD = 40
EVAL_MAX_LOAD = 1
BATCH_SIZE = 1
EPOCHS = 50
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-5
BASE_CHANNELS = 64
NUM_WORKERS = 2
PREFETCH_FACTOR = 2
SEED = 42
THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = Path("checkpoints/minimal_unet_host")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class CaseRecord:
    image_path: Path
    label_path: Path


def find_case_records(roots: Iterable[str], max_cases: int | None = None) -> list[CaseRecord]:
    cases: list[CaseRecord] = []
    seen: set[tuple[str, str]] = set()
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            logging.warning("Skip missing directory: %s", root_path)
            continue
        for image_suffix in IMAGE_SUFFIXES:
            for image_path in sorted(root_path.rglob(f"*{image_suffix}")):
                image_prefix = str(image_path)[: -len(image_suffix)]
                label_path = None
                for label_suffix in LABEL_SUFFIXES:
                    candidate = Path(image_prefix + label_suffix)
                    if candidate.exists():
                        label_path = candidate
                        break
                if label_path is None:
                    logging.warning("Missing label for image: %s", image_path)
                    continue
                key = (str(image_path), str(label_path))
                if key in seen:
                    continue
                seen.add(key)
                cases.append(CaseRecord(image_path=image_path, label_path=label_path))
                if max_cases is not None and len(cases) >= max_cases:
                    return cases
    return cases


class ResampleHelper:
    def __init__(self, target_spacing: tuple[float, float, float]):
        self.target_spacing = tuple(float(v) for v in target_spacing)

    @staticmethod
    def _compute_size(
            size: tuple[int, int, int],
            in_spacing: tuple[float, float, float],
            out_spacing: tuple[float, float, float],
    ) -> tuple[int, int, int]:
        return tuple(max(1, int(round(size[i] * in_spacing[i] / out_spacing[i]))) for i in range(3))

    def resample(self, image: sitk.Image, *, is_label: bool) -> sitk.Image:
        out_size = self._compute_size(image.GetSize(), image.GetSpacing(), self.target_spacing)
        return sitk.Resample(
            image,
            out_size,
            sitk.Transform(),
            sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear,
            image.GetOrigin(),
            self.target_spacing,
            image.GetDirection(),
            0.0,
            sitk.sitkUInt8 if is_label else sitk.sitkFloat32,
        )


def zscore_normalize(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    mask = np.isfinite(image)
    if not mask.any():
        return np.zeros_like(image, dtype=np.float32)
    values = image[mask]
    mean = float(values.mean())
    std = float(values.std())
    if std < 1e-6:
        std = 1.0
    out = (image - mean) / std
    out[~mask] = 0.0
    return out.astype(np.float32)


def load_case_arrays(case: CaseRecord, resampler: ResampleHelper) -> tuple[np.ndarray, np.ndarray]:
    image_itk = sitk.ReadImage(str(case.image_path))
    label_itk = sitk.ReadImage(str(case.label_path))
    image_np = sitk.GetArrayFromImage(resampler.resample(image_itk, is_label=False)).astype(np.float32)
    label_np = sitk.GetArrayFromImage(resampler.resample(label_itk, is_label=True)).astype(np.uint8)
    image_np = zscore_normalize(image_np)
    label_np = (label_np > 0).astype(np.uint8)
    return image_np, label_np


def compute_valid_start(center: tuple[int, int, int], shape: tuple[int, int, int], patch_size: tuple[int, int, int]) -> \
tuple[int, int, int]:
    starts = []
    for c, dim, size in zip(center, shape, patch_size):
        low = max(0, c - size // 2)
        high = max(0, dim - size)
        starts.append(int(np.clip(low, 0, high)))
    return tuple(starts)


def crop_with_padding(array: np.ndarray, start: tuple[int, int, int], patch_size: tuple[int, int, int]) -> np.ndarray:
    z, y, x = start
    pd, ph, pw = patch_size
    patch = array[z:z + pd, y:y + ph, x:x + pw]
    pad_d = max(0, pd - patch.shape[0])
    pad_h = max(0, ph - patch.shape[1])
    pad_w = max(0, pw - patch.shape[2])
    if pad_d or pad_h or pad_w:
        patch = np.pad(patch, ((0, pad_d), (0, pad_h), (0, pad_w)), mode="constant")
    return patch


class ForegroundPatchDataset(Dataset):
    """每次从某个病例中随机取一个前景 patch。"""

    def __init__(
            self,
            roots: Iterable[str],
            patch_size: tuple[int, int, int],
            target_spacing: tuple[float, float, float],
            patches_per_case: int,
            seed: int,
    ):
        self.cases = find_case_records(roots, max_cases=MAX_LOAD)
        if not self.cases:
            raise ValueError("No training cases found.")
        self.patch_size = tuple(int(v) for v in patch_size)
        self.patches_per_case = int(patches_per_case)
        self.resampler = ResampleHelper(target_spacing)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.cases) * self.patches_per_case

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        case = self.cases[index % len(self.cases)]
        image, label = load_case_arrays(case, self.resampler)
        fg_coords = np.argwhere(label > 0)
        if fg_coords.size == 0:
            raise ValueError(f"Case has no foreground voxels: {case.label_path}")

        center = tuple(int(v) for v in fg_coords[self.rng.randrange(len(fg_coords))])
        start = compute_valid_start(center, image.shape, self.patch_size)
        image_patch = crop_with_padding(image, start, self.patch_size)
        label_patch = crop_with_padding(label, start, self.patch_size)

        image_tensor = torch.from_numpy(image_patch[None, ...].astype(np.float32))
        label_tensor = torch.from_numpy(label_patch[None, ...].astype(np.float32))
        return image_tensor, label_tensor


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class MinimalUNet3D(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = BASE_CHANNELS):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.enc2 = DownBlock(base_channels, base_channels * 2)
        self.enc3 = DownBlock(base_channels * 2, base_channels * 4)
        self.enc4 = DownBlock(base_channels * 4, base_channels * 8)
        self.up3 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up1 = UpBlock(base_channels * 2, base_channels, base_channels)
        self.head = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        x = self.enc4(s3)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = self.head(x)
        return x


def tversky_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.3, beta: float = 0.7) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    smooth = 1e-6

    # Flatten tensors
    probs_flat = probs.view(-1)
    target_flat = target.view(-1)

    # Calculate true positives, false negatives, and false positives
    tp = (probs_flat * target_flat).sum()
    fn = ((1 - probs_flat) * target_flat).sum()
    fp = (probs_flat * (1 - target_flat)).sum()

    # Calculate Tversky index
    tversky_index = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)

    return 1.0 - tversky_index


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    tversky_ls = tversky_loss(logits, target)
    return bce + tversky_ls


def segmentation_loss_with_partial_loss(logits: torch.Tensor, target: torch.Tensor) -> tuple[
    torch.Tensor, torch.Tensor]:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    tversky_ls = tversky_loss(logits, target)
    return bce, tversky_ls


@torch.no_grad()
def dice_score_from_binary(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    inter = float((pred * target).sum())
    return (2.0 * inter + smooth) / (float(pred.sum() + target.sum()) + smooth)


@torch.no_grad()
def iou_score_from_binary(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Calculate Intersection over Union (IoU)."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = float(np.logical_and(pred, target).sum())
    union = float(np.logical_or(pred, target).sum())
    return (inter + smooth) / (union + smooth)


@torch.no_grad()
def volume_overlap_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Calculate volume-based overlap metrics."""
    pred_vol = float(pred.sum())
    target_vol = float(target.sum())
    
    # Volume difference ratio
    if target_vol > 0:
        vol_diff_ratio = abs(pred_vol - target_vol) / target_vol
    else:
        vol_diff_ratio = float('inf') if pred_vol > 0 else 0.0
    
    # Overlap ratio
    if pred_vol > 0 or target_vol > 0:
        overlap_ratio = float(np.logical_and(pred.astype(bool), target.astype(bool)).sum()) / max(pred_vol + target_vol, 1e-6)
    else:
        overlap_ratio = 0.0
    
    return {
        "pred_volume": pred_vol,
        "target_volume": target_vol,
        "volume_diff_ratio": vol_diff_ratio,
        "overlap_ratio": overlap_ratio
    }


@torch.no_grad()
def sensitivity_from_binary(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Calculate sensitivity (recall) - ability to detect positive cases."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = float(np.logical_and(pred, target).sum())
    fn = float(np.logical_and(~pred, target).sum())
    return (tp + smooth) / (tp + fn + smooth)


@torch.no_grad()
def specificity_from_binary(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Calculate specificity - ability to correctly identify negative cases."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    tn = float(np.logical_and(~pred, ~target).sum())
    fp = float(np.logical_and(pred, ~target).sum())
    return (tn + smooth) / (tn + fp + smooth)


@torch.no_grad()
def precision_from_binary(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Calculate precision - proportion of positive predictions that are correct."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = float(np.logical_and(pred, target).sum())
    fp = float(np.logical_and(pred, ~target).sum())
    return (tp + smooth) / (tp + fp + smooth)


@torch.no_grad()
def f1_score_from_binary(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Calculate F1 score - harmonic mean of precision and recall."""
    prec = precision_from_binary(pred, target, smooth)
    rec = sensitivity_from_binary(pred, target, smooth)
    if prec + rec == 0:
        return 0.0
    return 2.0 * (prec * rec) / (prec + rec)


@torch.no_grad()
def false_positive_rate(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Calculate false positive rate."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    fp = float(np.logical_and(pred, ~target).sum())
    tn = float(np.logical_and(~pred, ~target).sum())
    return (fp + smooth) / (fp + tn + smooth)


@torch.no_grad()
def false_negative_rate(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Calculate false negative rate."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    fn = float(np.logical_and(~pred, target).sum())
    tp = float(np.logical_and(pred, target).sum())
    return (fn + smooth) / (fn + tp + smooth)


@torch.no_grad()
def sliding_window_predict(
        model: nn.Module,
        image: np.ndarray,
        patch_size: tuple[int, int, int],
        device: torch.device,
) -> np.ndarray:
    model.eval()
    pd, ph, pw = patch_size
    d, h, w = image.shape
    stride = patch_size

    prob_sum = np.zeros((d, h, w), dtype=np.float32)
    count_map = np.zeros((d, h, w), dtype=np.float32)

    for z in range(0, d, stride[0]):
        for y in range(0, h, stride[1]):
            for x in range(0, w, stride[2]):
                patch = crop_with_padding(image, (z, y, x), patch_size)
                tensor = torch.from_numpy(patch[None, None, ...].astype(np.float32)).to(device)
                logits = model(tensor)
                prob = torch.sigmoid(logits).squeeze().detach().cpu().numpy()

                z2 = min(z + pd, d)
                y2 = min(y + ph, h)
                x2 = min(x + pw, w)
                dz, dy, dx = z2 - z, y2 - y, x2 - x
                prob_sum[z:z2, y:y2, x:x2] += prob[:dz, :dy, :dx]
                count_map[z:z2, y:y2, x:x2] += 1.0

    return prob_sum / np.maximum(count_map, 1e-6)


def only_foreground_part_predict(model: nn.Module, image: np.ndarray, label: np.ndarray, patch_size: tuple[int, int, int], device: torch.device) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    fg_coords = np.argwhere(label > 0)
    if fg_coords.size == 0:
        raise ValueError(f"Case has no foreground voxels")

    center = tuple(int(v) for v in fg_coords[random.randrange(len(fg_coords))])
    start = compute_valid_start(center, image.shape, patch_size)
    image_patch = crop_with_padding(image, start, patch_size)
    label_patch = crop_with_padding(label, start, patch_size)

    image_tensor = torch.from_numpy(image_patch[None, None, ...].astype(np.float32)).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        prob = torch.sigmoid(logits).squeeze().detach().cpu().numpy()
    return prob, label_patch, start


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total_steps = 0

    lm = LossMax()
    lm.to_device(device)

    progress_bar = tqdm(loader, desc="Training", leave=False)
    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)

        src_bce_loss, src_tversky_loss = segmentation_loss_with_partial_loss(logits, labels)
        src_loss = src_bce_loss + src_tversky_loss
        # if src_loss.item() < 0.1:
        # lm.find_max_conv_block(logits, labels, 10, loss_fn=segmentation_loss)
        # logits = lm.apply_pred(logits)

        loss = segmentation_loss(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(src_loss.item())
        total_steps += 1

        avg_loss = total_loss / total_steps
        progress_bar.set_postfix({"loss": f"{avg_loss:.4f}", "bce_loss": f"{src_bce_loss.item():.4f}",
                                  "tversky_loss": f"{src_tversky_loss.item():.4f}"})

    return total_loss / max(total_steps, 1)

#
# def train_one_epoch(model, loader, optimizer, weight_net, optimizer_w, device):
#     model.train()
#     weight_net.train()
#     total_loss = 0.0
#
#     progress_bar = tqdm(loader, desc="Training", leave=False)
#     for images, labels in progress_bar:
#         images, labels = images.to(device), labels.to(device)
#
#         # --- 第一步：更新权重网络 (D-Step) ---
#         # 目标：寻找当前 U-Net 分得不好的地方，最大化这些地方的权重
#         optimizer_w.zero_grad()
#         with torch.no_grad():
#             logits = model(images)
#
#         # 计算当前的像素级 BCE
#         pixel_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
#
#         # 获取权重图并计算“对抗损失”
#         weights = weight_net(logits, labels)
#         # 我们想让 Loss 变大，所以给 Loss 加负号来做梯度下降
#         loss_w = -(weights * pixel_loss.detach()).mean()
#
#         loss_w.backward()
#         optimizer_w.step()
#
#         # --- 第二步：更新分割网络 (G-Step) ---
#         # 目标：在权重核指出的难点上，最小化加权后的损失
#         optimizer.zero_grad()
#         logits = model(images)
#
#         # 重新获取更新后的权重（注意此时不更新 weight_net 的参数）
#         with torch.no_grad():
#             weights = weight_net(logits, labels)
#
#         # 计算加权后的 BCE + 原始 Tversky
#         weighted_bce = (F.binary_cross_entropy_with_logits(logits, labels, reduction='none') * weights).mean()
#         _, src_tversky = segmentation_loss_with_partial_loss(logits, labels)  # 假设 Tversky 保持不变
#
#         total_g_loss = weighted_bce + src_tversky
#         total_g_loss.backward()
#         optimizer.step()
#
#         total_loss += total_g_loss.item()
#         progress_bar.set_postfix({
#             "g_loss": f"{total_g_loss.item():.4f}",
#             "w_max": f"{weights.max().item():.2f}",  # 看看最高被加权了多少倍
#             "w_std": f"{weights.std().item():.4f}"  # 看看权重分布的差异大不大
#         })
#         # progress_bar.set_postfix({"g_loss": f"{total_g_loss.item():.4f}", "w_mean": f"{weights.mean():.2f}"})
#
#     return total_loss / len(loader)

@torch.no_grad()
def evaluate(model: nn.Module, eval_cases: list[CaseRecord], device: torch.device) -> dict[str, float]:
    if not eval_cases:
        raise ValueError("No evaluation cases found.")

    resampler = ResampleHelper(TARGET_SPACING)
    dice_list: list[float] = []
    iou_list: list[float] = []
    sensitivity_list: list[float] = []
    specificity_list: list[float] = []
    precision_list: list[float] = []
    f1_list: list[float] = []
    fpr_list: list[float] = []
    fnr_list: list[float] = []
    pred_volumes: list[float] = []
    target_volumes: list[float] = []
    volume_diff_ratios: list[float] = []
    
    for case in eval_cases:
        image, label = load_case_arrays(case, resampler)
        pred_patch, label_patch, start = only_foreground_part_predict(model, image, label, PATCH_SIZE, device)
        pred = (pred_patch >= THRESHOLD).astype(np.uint8)
        
        # Calculate all metrics
        dice_list.append(dice_score_from_binary(pred, label_patch))
        iou_list.append(iou_score_from_binary(pred, label_patch))
        sensitivity_list.append(sensitivity_from_binary(pred, label_patch))
        specificity_list.append(specificity_from_binary(pred, label_patch))
        precision_list.append(precision_from_binary(pred, label_patch))
        f1_list.append(f1_score_from_binary(pred, label_patch))
        fpr_list.append(false_positive_rate(pred, label_patch))
        fnr_list.append(false_negative_rate(pred, label_patch))
        
        # Volume metrics
        vol_metrics = volume_overlap_metrics(pred, label_patch)
        pred_volumes.append(vol_metrics["pred_volume"])
        target_volumes.append(vol_metrics["target_volume"])
        volume_diff_ratios.append(vol_metrics["volume_diff_ratio"])

    # Calculate statistics
    def safe_mean(lst):
        return float(np.mean(lst)) if lst else 0.0
    
    def safe_std(lst):
        return float(np.std(lst)) if len(lst) > 1 else 0.0

    return {
        "dice": safe_mean(dice_list),
        "dice_std": safe_std(dice_list),
        "iou": safe_mean(iou_list),
        "iou_std": safe_std(iou_list),
        "sensitivity": safe_mean(sensitivity_list),
        "sensitivity_std": safe_std(sensitivity_list),
        "specificity": safe_mean(specificity_list),
        "specificity_std": safe_std(specificity_list),
        "precision": safe_mean(precision_list),
        "precision_std": safe_std(precision_list),
        "f1": safe_mean(f1_list),
        "f1_std": safe_std(f1_list),
        "fpr": safe_mean(fpr_list),
        "fpr_std": safe_std(fpr_list),
        "fnr": safe_mean(fnr_list),
        "fnr_std": safe_std(fnr_list),
        "cases": float(len(eval_cases)),
        "avg_pred_volume": safe_mean(pred_volumes),
        "avg_target_volume": safe_mean(target_volumes),
        "avg_volume_diff_ratio": safe_mean(volume_diff_ratios),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    set_seed(SEED)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    weight_net = MultiScaleWeightNet().to(DEVICE)
    optimizer_w = torch.optim.Adam(weight_net.parameters(), lr=1e-3)

    train_dataset = ForegroundPatchDataset(
        roots=TRAIN_DIRS,
        patch_size=PATCH_SIZE,
        target_spacing=TARGET_SPACING,
        patches_per_case=PATCHES_PER_CASE,
        seed=SEED,
    )
    eval_cases = find_case_records(EVAL_DIRS, max_cases=EVAL_MAX_LOAD)
    if not eval_cases:
        raise ValueError("No evaluation cases found.")

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "shuffle": True,
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
    }
    if NUM_WORKERS > 0:
        loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR

    train_loader = DataLoader(
        train_dataset,
        **loader_kwargs,
    )

    device = torch.device(DEVICE)
    model = MinimalUNet3D().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    loss_history = []
    metrics_history = []

    best_dice = -1.0
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer
                                     # , weight_net, optimizer_w
                                     , device)

        # Evaluate with progress bar
        eval_metrics = evaluate(model, eval_cases, device)

        loss_history.append(train_loss)
        metrics_history.append(eval_metrics)

        logging.info(
            "Epoch [%d/%d] train_loss=%.6f | "
            "Eval - Dice: %.4f±%.4f, IoU: %.4f±%.4f, F1: %.4f±%.4f | "
            "Sens: %.4f±%.4f, Spec: %.4f±%.4f, Prec: %.4f±%.4f | "
            "FPR: %.4f±%.4f, FNR: %.4f±%.4f | "
            "Vol Diff Ratio: %.4f",
            epoch + 1,
            EPOCHS,
            train_loss,
            eval_metrics["dice"],
            eval_metrics["dice_std"],
            eval_metrics["iou"],
            eval_metrics["iou_std"],
            eval_metrics["f1"],
            eval_metrics["f1_std"],
            eval_metrics["sensitivity"],
            eval_metrics["sensitivity_std"],
            eval_metrics["specificity"],
            eval_metrics["specificity_std"],
            eval_metrics["precision"],
            eval_metrics["precision_std"],
            eval_metrics["fpr"],
            eval_metrics["fpr_std"],
            eval_metrics["fnr"],
            eval_metrics["fnr_std"],
            eval_metrics["avg_volume_diff_ratio"],
        )

        latest_ckpt = SAVE_DIR / "latest.pt"
        torch.save({"model": model.state_dict(), "epoch": epoch + 1, "metrics": eval_metrics}, latest_ckpt)

        if eval_metrics["dice"] > best_dice:
            best_dice = eval_metrics["dice"]
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "metrics": eval_metrics}, SAVE_DIR / "best.pt")

    # draw and save picture for each metric
    import matplotlib.pyplot as plt

    # Plot training loss
    plt.figure()
    plt.plot(loss_history, label="train_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.savefig(SAVE_DIR / "loss.png")
    plt.close()

    # Extract metrics from history
    dice_history = [m["dice"] for m in metrics_history]
    iou_history = [m["iou"] for m in metrics_history]
    cases_history = [m["cases"] for m in metrics_history]

    # Plot eval dice
    plt.figure()
    plt.plot(dice_history, label="eval_dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title("Evaluation Dice")
    plt.legend()
    plt.savefig(SAVE_DIR / "eval_dice.png")
    plt.close()

    # Plot eval IoU
    plt.figure()
    plt.plot(iou_history, label="eval_iou")
    plt.xlabel("Epoch")
    plt.ylabel("IoU Score")
    plt.title("Evaluation IoU")
    plt.legend()
    plt.savefig(SAVE_DIR / "eval_iou.png")
    plt.close()

    logging.info("Training finished. Best eval dice = %.6f", best_dice)


if __name__ == "__main__":
    main()
