# 本地四模型医学图像分割训练完整报告

- 生成时间：2026-05-12T18:25:24
- 配置文件：`config/local_four_models.yaml`
- 输出目录：`outputs/local_four_models`
- 排名指标：`best_sequential_mean_dice`

## 项目理解

本项目面向 3D CTA/医学影像动脉瘤分割，数据加载器按 NIfTI 影像与标签对读取病例，重采样到训练空间后进行 patch 采样。模型主干为 3D U-Net，并支持在瓶颈层启用 ASPP 多尺度上下文模块与 3D 坐标注意力模块。训练使用 BCE 与 Tversky/Focal 类分割损失组合，验证阶段输出 Dice、AUC、F1、IoU、Precision、Recall、Specificity 与阈值扫描结果。

## 四个模型设置

| 模型 | ASPP | CoordAttention | 参数量 | 最佳 Dice | 最新 Dice | 最新 AUC | 最新 Best-F1 | checkpoint |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 3D U-Net baseline | 否 | 否 | 1366279 | 0.000002 | 0.000001 | 0.758432 | 0.000002 | `outputs/local_four_models/01_unet_baseline/checkpoints/best.pth` |
| 3D U-Net + ASPP | 是 | 否 | 1825543 | 0.000001 | 0.000001 | 0.695315 | 0.000008 | `outputs/local_four_models/02_unet_aspp/checkpoints/best.pth` |
| 3D U-Net + CoordAttention | 否 | 是 | 1370775 | 0.000001 | 0.000001 | 0.471012 | 0.000001 | `outputs/local_four_models/03_unet_coord_attention/checkpoints/best.pth` |
| 3D U-Net + ASPP + CoordAttention | 是 | 是 | 1830039 | 0.000001 | 0.000001 | 0.467133 | 0.000001 | `outputs/local_four_models/04_unet_aspp_coord_attention/checkpoints/best.pth` |

## 最优模型

当前以最佳 sequential mean Dice 排名，最优模型为 **3D U-Net baseline**，最佳 Dice 为 `0.000002`。
最优权重路径：`outputs/local_four_models/01_unet_baseline/checkpoints/best.pth`

## 输出说明

每个模型目录下包含：

- `checkpoints/latest.pth`：最新 checkpoint，可继续训练。
- `checkpoints/best.pth`：按验证 Dice 保存的最佳模型。
- `logs/training.log`：训练日志。
- `logs/tensorboard/`：TensorBoard 曲线。
- `reports/eval_reports/.../combined/latest.json`：该模型最新完整评估报告。
- `reports/model_summary.json`：该模型摘要。

总报告文件：

- `complete_training_report.md`：人类可读完整报告。
- `complete_training_report.json`：机器可读完整报告，可用于后续论文表格或可视化。
