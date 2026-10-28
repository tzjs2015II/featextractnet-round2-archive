"""PipnStyleBaseline —— Physics-Informed PointNet 风格网络（忠实移植）。

用途
----
将参考仓库 PhysicsInformedPointNet-main/models.py 中的 NormalizedGeoPointNet
原样移植为可 import 的监督回归基线。该网络是 FeatExtractNet 的直接前身
（低/中/高三级逐点特征 + 全局统计 + 交叉层门控融合），作为同族对照基线。

接口
----
forward(x: [B, Cin, N]) -> [B, Cout, N]，坐标在通道维、点在最后一维。
原模型本身即为 batch 版 [B,C,N] 前向、无 physics 输入（physics 损失在训练
外层实现），故无需裁剪输入，仅作类名与文档适配。

移植忠实性说明
--------------
- 模块结构、激活（tanh/relu/sigmoid 门控）、前向数据流与原仓库 models.py
  逐层一致；宽度由 scaling 同时缩放（s64/s128/s256/s512）。
- 原码 forward 构造了双支路：支路 A = final_conv(tanh)->output_conv 直接出
  output；支路 B = conv7/8/9/10 对同输入 final_feat1 做 tanh 链得到 output1
  (=conv10(x3)+output)。**原码最终 return output** —— 支路 B 在原仓库中属
  未参与前向的遗留权重。本移植保持相同语义（return output），参数量与原始
  模型一致（可复现原模型）；注释保留支路 B 说明，便于后续按需启用。

原模型参数量(scale=1, Cout=4)约 1.15M，与 FeatExtractNet(~1.15M) 同量级，
可作容量受控的 PIPN 风格对照。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PipnStyleBaseline(nn.Module):
    """Faithful port of NormalizedGeoPointNet (PIPN repo models.py)."""

    def __init__(self, input_channels=2, output_channels=4, scaling=1.0):
        super().__init__()
        s = lambda x: int(x * scaling)
        self.s64 = s(64)
        self.s128 = s(128)
        self.s256 = s(256)
        self.s512 = s(512)

        # ---- 1) 低/中/高三级逐点特征 ----
        self.conv1 = nn.Conv1d(input_channels, self.s64, 1)   # 低层
        self.conv2 = nn.Conv1d(self.s64, self.s128, 1)        # 中层
        self.conv3 = nn.Conv1d(self.s128, self.s256, 1)       # 高层

        # ---- 2) 全局统计（max/avg/var 拼接）压缩 ----
        self.global_conv = nn.Conv1d(3 * self.s256, self.s256, 1)

        # ---- 3) 交叉层门控融合 ----
        self.gate1 = nn.Conv1d(self.s64 + self.s256, self.s64, 1)   # 低层门
        self.gate2 = nn.Conv1d(self.s128 + self.s256, self.s128, 1) # 中层门
        self.gate3 = nn.Conv1d(self.s256 + self.s256, self.s256, 1) # 高层门
        self.compress1 = nn.Conv1d(self.s256, self.s64, 1)
        self.compress2 = nn.Conv1d(self.s256, self.s128, 1)
        self.compress3 = nn.Conv1d(self.s256, self.s256, 1)

        # ---- 4) 拼接融合与输出（原码支路 A）----
        fused_dim = self.s64 + self.s128 + self.s256 + self.s256
        self.final_conv = nn.Conv1d(fused_dim, self.s512, 1)
        self.output_conv = nn.Conv1d(self.s512, output_channels, 1)

        # ---- 原码支路 B（conv7..10；原前向未返回，遗留权重，忠实保留）----
        self.conv7 = nn.Conv1d(fused_dim, self.s256, 1)
        self.conv8 = nn.Conv1d(self.s256, self.s128, 1)
        self.conv9 = nn.Conv1d(self.s128, self.s128, 1)
        self.conv10 = nn.Conv1d(self.s128, output_channels, 1)

        self._init_weights()

    def forward(self, x):
        # x: [B, Cin, N]
        B, _, N = x.shape

        # 1) 低/中/高逐点特征
        x1 = F.relu(self.conv1(x))                            # [B,s64,N]
        x2 = F.relu(self.conv2(x1))                           # [B,s128,N]
        x3 = F.relu(self.conv3(x2))                           # [B,s256,N]

        # 2) 全局统计: max + avg + var -> 压缩
        g_max = F.max_pool1d(x3, kernel_size=N)               # [B,s256,1]
        g_avg = F.avg_pool1d(x3, kernel_size=N)               # [B,s256,1]
        g_var = torch.var(x3, dim=2, keepdim=True)            # [B,s256,1]
        global_feat = torch.cat([g_max, g_avg, g_var], dim=1) # [B,3*s256,1]
        global_feat = F.relu(self.global_conv(global_feat))   # [B,s256,1]
        global_exp = global_feat.expand(-1, -1, N)            # 回灌 N

        # 3) 门控融合（每级 = gate*本级 + (1-gate)*压缩后的全局）
        cat1 = torch.cat([x1, global_exp], dim=1)
        g1 = torch.sigmoid(self.gate1(cat1))                  # [B,s64,N]
        x1 = g1 * x1 + (1 - g1) * self.compress1(global_exp)

        cat2 = torch.cat([x2, global_exp], dim=1)
        g2 = torch.sigmoid(self.gate2(cat2))
        x2 = g2 * x2 + (1 - g2) * self.compress2(global_exp)

        cat3 = torch.cat([x3, global_exp], dim=1)
        g3 = torch.sigmoid(self.gate3(cat3))
        x3 = g3 * x3 + (1 - g3) * self.compress3(global_exp)

        # 4) 支路 A（原码返回该支路）
        fused = torch.cat([x1, x2, x3, global_exp], dim=1)    # [B,fused_dim,N]
        out = self.output_conv(F.tanh(self.final_conv(fused)))  # [B,Cout,N]

        # 支路 B（忠实保留原码的未返回结构）
        xb1 = F.tanh(self.conv7(fused))
        xb2 = F.tanh(self.conv8(xb1))
        xb3 = F.tanh(self.conv9(xb2))
        # output1 = self.conv10(xb3) + out   # 原码未返回
        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                # 原结构以 relu/tanh/sigmoid 混合激活，统一用 tanh 增益近似，
                # 仅影响初始尺度，不影响结构忠实性
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
