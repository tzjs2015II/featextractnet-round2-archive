"""PointNetBaseline —— 标准 PointNet 监督版（逐点回归）。

用途
----
PointNet 式"逐点共享 MLP + 全局 max-pool 回灌"的监督回归基线，对应
FeatExtractNet 的经典全局池化路线的对照实现。

接口
----
forward(x: [B, Cin, N]) -> [B, Cout, N]，坐标在通道维、点在最后一维。
与 FeatExtractNet 输入约定一致（热场 Cin=2、Cout=4）。

结构（width_scale 同时缩放全部宽度）
----
1. 逐点共享 MLP(64, 128, 256)：
       conv1:  Cin -> 64s，ReLU
       conv2:  64s -> 128s，ReLU
       conv3:  128s -> 256s，ReLU（全局池化输入）
2. 全局 max-pool 于 N 维 -> [B, 256s, 1]，expand 回 N。
3. 逐点特征与全局特征拼接：局部逐点特征取 conv1 输出（64s）——
   这是 PIPN 原文 PointNet 的做法（其 local 取前两个 64 卷积后、global 取
   1024），此处仅把全局宽度由 1024 降到 256s 以控制容量。
   拼接维度 = 64s + 256s = 320s。
4. 共享分类/回归 MLP（与 PIPN 头同构）：320s -> 512s -> 256s -> 128s
   -> Cout，ReLU 激活，输出层线性，可选 tanh 收尾（构造参数 tanh_output）。

实现要点
--------
- 全部为 nn.Conv1d(kernel_size=1) 逐点卷积，无 BN/Dropout（与 PIPN 原码、
  FeatExtractNet 一致的轻量风格）。
- 输出对每个点不同：concat 的是逐点局部特征 + 全局回灌，属逐点回归头。
- kaiming 初始化；模块内不设随机种子（由外层负责）。
"""

import torch
import torch.nn as nn


class PointNetBaseline(nn.Module):
    """Supervised PointNet for per-point regression."""

    def __init__(self, input_channels=2, output_channels=4,
                 width_scale=1.0, tanh_output=False):
        super().__init__()
        s = lambda v: int(v * width_scale)
        self.s64, self.s128, self.s256 = s(64), s(128), s(256)
        self.tanh_output = tanh_output

        # 逐点共享 MLP(64,128,256)
        self.conv1 = nn.Conv1d(input_channels, self.s64, 1)
        self.conv2 = nn.Conv1d(self.s64, self.s128, 1)
        self.conv3 = nn.Conv1d(self.s128, self.s256, 1)

        # 共享分类/回归 MLP：输入 = local(64s) + global(256s) = 320s
        head_in = self.s64 + self.s256
        self.head1 = nn.Conv1d(head_in, s(512), 1)
        self.head2 = nn.Conv1d(s(512), s(256), 1)
        self.head3 = nn.Conv1d(s(256), s(128), 1)
        self.head_out = nn.Conv1d(s(128), output_channels, 1)

        self._init_weights()

    def forward(self, x):
        # x: [B, Cin, N]
        B, _, N = x.shape

        # 逐点 MLP：h1 保留为"早期逐点局部特征"
        h1 = torch.relu(self.conv1(x))      # [B, 64s, N]  局部特征
        h2 = torch.relu(self.conv2(h1))     # [B, 128s, N]
        h3 = torch.relu(self.conv3(h2))     # [B, 256s, N]

        # 全局 max-pool -> [B, 256s, 1] -> expand 回 N
        global_feat = h3.max(dim=-1, keepdim=True).values  # [B, 256s, 1]
        global_feat = global_feat.expand(-1, -1, N)        # [B, 256s, N]

        # 逐点特征(local) 与全局特征拼接 -> 分类/回归 MLP
        f = torch.cat([h1, global_feat], dim=1)            # [B, 320s, N]
        f = torch.relu(self.head1(f))                      # [B, 512s, N]
        f = torch.relu(self.head2(f))                      # [B, 256s, N]
        f = torch.relu(self.head3(f))                      # [B, 128s, N]
        out = self.head_out(f)                             # [B, Cout, N]

        if self.tanh_output:
            out = torch.tanh(out)
        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                # 主路径为 ReLU；输出层本身线性，用 ReLU 增益近似即可
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
