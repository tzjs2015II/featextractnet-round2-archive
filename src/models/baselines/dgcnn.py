"""DGCNN —— 动态图卷积网络（Dynamic Graph CNN, EdgeConv）监督回归基线。

用途
----
以"逐层动态重建 kNN 图 + 边卷积（EdgeConv）"做局部特征学习的监督基线，
对照 FeatExtractNet 的全局池化/门控融合与固定邻域（PointNet++）路线。

接口
----
forward(x: [B, Cin, N]) -> [B, Cout, N]，坐标在通道维、点在最后一维
（Cin=2 为 (x,y) 坐标）。

结构
----
- 两层 EdgeConv（默认 k=16），每层在**当前点特征空间**上重建 kNN 图
  （第一层以原始坐标建图，第二层以第一层输出特征建图 —— 即动态图）。
  边特征 = [邻点特征 f_j ; f_j - f_i]（2*C 维），经共享 MLP
  (2*C -> width -> width, ReLU) 后按邻域 max-pool 得逐点新特征。
- 两层逐点输出按通道拼接：[B, 2*width, N]。
- 全局 max-pool -> [B, 2*width, 1] 回灌拼接逐点特征
  （融合头输入 = 4*width = 256），经共享 MLP(256 -> 128 -> Cout)
  得逐点预测，输出层线性。

实现要点
--------
- 邻域聚合为真实 EdgeConv（非逐点）：每组边共享同一 MLP 权重。
- 距离矩阵用平方距离 + topk 求 kNN；共享 MLP 以 Conv1d(k=1) 在通道维实现，
  邻点作为独立 token（reshape -> conv -> view），避免 [B,N,C] permute。
- 采样与图结构皆确定（无随机采样），模块内不设随机种子（外层负责）。
- kaiming 初始化。
"""

import torch
import torch.nn as nn


def _sq_dist_same(a):
    """a:[B,N,D] 自身两两平方距离 -> [B,N,N]（matmul 实现）。"""
    aa = (a * a).sum(-1, keepdim=True)                     # [B,N,1]
    inner = torch.matmul(a, a.transpose(-1, -2))           # [B,N,N]
    return (aa + aa.transpose(-1, -2) - 2.0 * inner).clamp_min(0.0)


def _gather(feats, idx):
    """按点维(第二维)聚集特征: feats:[B,C,N], idx:[B,...,K] -> [B,C,...,K]。

    idx 展平为 [B,L] 后与 feats 同为 3D 做 gather，再还原其余维度
    （gather 要求 index 与 input 维数一致）。
    """
    B, C, N = feats.shape
    idx = idx.long()
    s = idx.shape[1:]
    idxf = idx.reshape(B, -1)
    idx_c = idxf.unsqueeze(1).expand(B, C, -1)
    out = torch.gather(feats, dim=2, index=idx_c)
    return out.reshape(B, C, *s)


class DGCNN(nn.Module):
    """Dynamic Graph CNN (EdgeConv) baseline."""

    def __init__(self, input_channels=2, output_channels=4,
                 k=16, width=64, n_blocks=2, tanh_output=False):
        super().__init__()
        self.k = k
        self.width = width
        self.tanh_output = tanh_output

        # 每层一个 EdgeConv 共享 MLP：(2*C_in -> width -> width)
        self.edge_mlps = nn.ModuleList()
        cin = input_channels
        for _ in range(n_blocks):
            self.edge_mlps.append(
                nn.Sequential(
                    nn.Conv1d(2 * cin, width, 1), nn.ReLU(inplace=True),
                    nn.Conv1d(width, width, 1), nn.ReLU(inplace=True)))
            cin = width

        # 融合头：逐点(2*width) + 全局回灌(2*width) = 4*width -> 128 -> Cout
        self.head1 = nn.Conv1d(4 * width, 128, 1)
        self.head_out = nn.Conv1d(128, output_channels, 1)

        self._init_weights()

    def _edge_conv_block(self, mlp, feats):
        """单层 EdgeConv：以 feats 为坐标空间动态建 kNN 图并聚合。

        feats:[B,C,N] -> [B,width,N]。
        """
        B, C, N = feats.shape
        # 1) 在特征空间重建 kNN 图
        pts = feats.transpose(1, 2)                          # [B,N,C] 作距离空间
        d = _sq_dist_same(pts)                               # [B,N,N]
        kk = min(self.k, N)
        _, knn = d.topk(kk, dim=-1, largest=False)           # [B,N,kk]（含自身）

        # 2) 边特征: 邻点 f_j 与差分 f_j - f_i
        f_j = _gather(feats, knn)                            # [B,C,N,kk]
        f_i = feats.unsqueeze(-1).expand_as(f_j)             # [B,C,N,kk]
        edge = torch.cat([f_j, f_j - f_i], dim=1)            # [B,2C,N,kk]

        # 3) 共享边 MLP + 邻域 max-pool
        B, Ce, _, kk = edge.shape
        h = mlp(edge.reshape(B, Ce, N * kk))               # [B,width,N*kk]
        h = h.view(B, self.width, N, kk)
        return h.max(dim=-1).values                          # [B,width,N]

    def forward(self, x):
        # x: [B, Cin, N]
        B, _, N = x.shape

        feats = x
        block_outs = []
        for mlp in self.edge_mlps:
            feats = self._edge_conv_block(mlp, feats)        # [B,width,N]
            block_outs.append(feats)

        per = torch.cat(block_outs, dim=1)                   # [B, 2*width, N]
        g = per.max(dim=-1, keepdim=True).values             # [B, 2*width, 1]
        g = g.expand(-1, -1, N)                              # 全局回灌
        f = torch.cat([per, g], dim=1)                       # [B, 4*width, N]

        h = torch.relu(self.head1(f))                        # [B,128,N]
        out = self.head_out(h)                               # [B,Cout,N]

        if self.tanh_output:
            out = torch.tanh(out)
        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
