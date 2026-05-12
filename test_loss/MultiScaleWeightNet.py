import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleWeightNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用空洞卷积捕捉不同尺度的形态学特征
        # 1. 局部细节 (3x3x3)
        self.branch1 = nn.Conv3d(2, 4, kernel_size=3, padding=1, dilation=1)
        # 2. 中等形态 (通过 dilation=3 模拟 7x7x7 的感受野)
        self.branch2 = nn.Conv3d(2, 4, kernel_size=3, padding=3, dilation=3)
        # 3. 全局拓扑 (通过 dilation=6 模拟 13x13x13 的感受野)
        self.branch3 = nn.Conv3d(2, 4, kernel_size=3, padding=6, dilation=6)

        # 融合层：输出单通道权重图
        self.fusion = nn.Sequential(
            nn.Conv3d(12, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, logits, labels):
        # 拼接预测和标签作为输入 [B, 2, D, H, W]
        # 使用 detach 确保权重网络的更新不会直接通过这里回传给 U-Net
        x = torch.cat([logits.detach(), labels], dim=1)

        feat1 = F.leaky_relu(self.branch1(x), 0.2)
        feat2 = F.leaky_relu(self.branch2(x), 0.2)
        feat3 = F.leaky_relu(self.branch3(x), 0.2)

        combined = torch.cat([feat1, feat2, feat3], dim=1)
        weights = self.fusion(combined)

        # 归一化：保持权重图的均值为 1，防止总 Loss 的量级剧烈波动
        weights = weights / (weights.mean() + 1e-7)
        return weights