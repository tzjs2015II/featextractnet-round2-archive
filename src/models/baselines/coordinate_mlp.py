"""CoordMLP —— 纯坐标逐点 MLP 监督基线。

用途
----
作为"无结构先验"监督基线：只用单点坐标回归该点的物理量，不含任何邻域 /
全局上下文，用于对照评估 FeatExtractNet 中邻域与全局聚合结构的贡献。

接口
----
forward(x: [B, Cin, N]) -> [B, Cout, N]
    热场 Cin=2（x,y 归一化坐标）、Cout=4（u,v,p,T）；
    声场网格化为点云时 Cout=1（p_total）。
    与 FeatExtractNet 的输入约定一致：坐标在通道维、点在最后一维。

实现要点
--------
- 用 nn.Conv1d(kernel_size=1) 堆叠：逐点共享的全连接 MLP（通道维），
  与 FeatExtractNet 的逐点卷积形态一致，便于参数量直接对照。
- 激活 ReLU，最后一层线性（回归头）。
- 参数量（忽略 bias）：Cin*h + (num_layers-2)*h^2 + h*Cout，
  其中 num_layers 为总线性层数（>=2，含输入映射与输出映射），
  隐藏层数为 num_layers-2。默认 h=512、num_layers=6 -> ~1.05M，
  与 FeatExtractNet(~1.15M) 同量级（可在 90%~110% 内用 h/层数微调）。
- 逐点模型在 B*N 个点上并行，显存与点数成正比，适合大 N 声场对照。
- 权重按 kaiming 初始化（Linear/Conv 统一 fan_in+ReLU 增益）；模块内不设
  随机种子，由外层训练脚本统一 set_seed。
"""

import torch
import torch.nn as nn


class CoordMLP(nn.Module):
    """Pure per-point coordinate MLP baseline (Conv1d k=1 equivalent)."""

    def __init__(self, input_channels=2, output_channels=4,
                 hidden_dim=512, num_layers=6):
        super().__init__()
        if num_layers < 2:
            raise ValueError(
                "num_layers must be >= 2 (input mapping + output mapping).")

        layers = []
        # 输入映射: [B, Cin, N] -> [B, h, N]
        layers.append(nn.Conv1d(input_channels, hidden_dim, 1))
        layers.append(nn.ReLU(inplace=True))
        # 隐藏层: num_layers - 2 个 h->h 块
        for _ in range(num_layers - 2):
            layers.append(nn.Conv1d(hidden_dim, hidden_dim, 1))
            layers.append(nn.ReLU(inplace=True))
        # 输出映射: [B, h, N] -> [B, Cout, N]（最后一层线性）
        layers.append(nn.Conv1d(hidden_dim, output_channels, 1))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def forward(self, x):
        # x: [B, Cin, N] -> [B, Cout, N]
        return self.net(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
