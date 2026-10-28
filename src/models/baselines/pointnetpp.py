"""PointNetPlusPlusBaseline —— 简化但真实的 PointNet++（SSG, kNN 分组）。

用途
----
局部邻域分层聚合（set abstraction）的监督回归基线：验证"多尺度局部邻域 +
子采样"相对 FeatExtractNet 全局池化/门控融合路线的差异。

接口
----
forward(x: [B, Cin, N]) -> [B, Cout, N]，坐标在通道维、点在最后一维。
（与 FeatExtractNet 的点云接口一致；Cin=2 为 (x,y) 坐标。）

结构（SSG, 最远点采样 + kNN 邻域聚合）
----
- SA1：对 N 个原始点最远点采样到 N1（默认 512），对每个中心做 K=16
  kNN 分组；组内每个邻点特征 = [绝对坐标(2), 相对坐标(2)] 共 4 通道，
  经共享 MLP(64,64,128)（输出宽度即 (64,64,128)，逐组共享）后按组
  max-pool -> 局部特征 [B, 128, N1]。
- SA2：对 N1 个子点再采样到 N2（默认 128），kNN 分组；组内邻点特征 =
  [上层局部特征(C1=128), 相对坐标(2)]，经共享 MLP(128,256,512)
  （输出宽度即 (128,256,512)，输入维度自动为 C1+2）后 max-pool
  -> [B, 512, N2]。
- Feature propagation（最近邻插值 + skip 拼接）：
    * N2 -> N1：将 N2 中心特征按最近邻赋给每个 N1 子点，与 SA1 局部特征
      拼接（640 维）经 MLP(640->256->128) -> [B, 128, N1]；
    * N1 -> N：将上一步特征按最近邻赋给每个原始点，与逐点基座特征
      conv0(坐标 -> 64 维) 拼接（192 维）经 MLP(192->128) 出 Cout，
      输出层线性。
  最近邻插值为"零阶"插值，属简化（PointNet++ 原文用距离加权线性插值），
  但聚合/分层/回传链路真实，非逐点模型。

实现要点
--------
- 采样用最远点采样（FPS，确定性：首点取离原点最远点），构造参数
  n_points=(N1,N2)、k 可调；kNN 用 torch.cdist + topk（见 _sq_dist，
  以矩阵乘法实现平方距离以加速 CPU/GPU）。
- 局部共享 MLP 用 Conv1d(kernel_size=1) 在通道维实现（组内各邻点视为
  独立 token），避免 permute 到 [B,N,C]。
- kaiming 初始化；模块内不设随机种子。
"""

import torch
import torch.nn as nn


def _sq_dist(a, b):
    """两组点间平方距离: a:[B,M,D], b:[B,N,D] -> [B,M,N]（matmul 实现）。"""
    aa = (a * a).sum(-1, keepdim=True)              # [B,M,1]
    bb = (b * b).sum(-1, keepdim=True).transpose(-1, -2)  # [B,1,N]
    inner = torch.matmul(a, b.transpose(-1, -2))    # [B,M,N]
    return (aa + bb - 2.0 * inner).clamp_min(0.0)


def _gather(feats, idx):
    """按点维(第二维)聚集特征: feats:[B,C,N], idx:[B, ..., K] -> [B,C,...,K]。

    将 idx 展平为 [B, L] 后与 feats 同为 3D 做 gather，再还原索引的其余维度，
    以规避 gather 要求 index 与 input 维数一致的限制。
    """
    B, C, N = feats.shape
    idx = idx.long()
    s = idx.shape[1:]                                        # (…, K)
    idxf = idx.reshape(B, -1)                                # [B, L]
    idx_c = idxf.unsqueeze(1).expand(B, C, -1)               # [B, C, L]
    out = torch.gather(feats, dim=2, index=idx_c)            # [B, C, L]
    return out.reshape(B, C, *s)


def _farthest_point_sample(coords, n_sample):
    """最远点采样: coords:[B,N,D] -> [B,n_sample]（索引）。

    确定性实现：首点取"离原点最远"的点，随后每次取距已选集合最远的点。
    """
    B, N, D = coords.shape
    device = coords.device
    centroids = torch.zeros(B, n_sample, dtype=torch.long, device=device)
    dist = torch.full((B, N), float('inf'), device=device)
    far = (coords * coords).sum(-1).argmax(-1)          # 离原点最远作首点
    for i in range(n_sample):
        centroids[:, i] = far
        c = torch.gather(
            coords, 1, far.view(B, 1, 1).expand(B, 1, D)).squeeze(1)  # [B,D]
        d = ((coords - c.unsqueeze(1)) ** 2).sum(-1)    # [B,N]
        dist = torch.minimum(dist, d)
        far = dist.argmax(-1)
    return centroids


class PointNetPlusPlusBaseline(nn.Module):
    """Simplified-but-real PointNet++ SSG baseline (kNN grouping)."""

    def __init__(self, input_channels=2, output_channels=4,
                 n_points=(512, 128), k=16, tanh_output=False):
        super().__init__()
        self.n1, self.n2 = n_points
        self.k = k
        self.tanh_output = tanh_output

        # 逐点基座特征（回传时与高层特征拼接的 per-point 信息源）
        self.conv0 = nn.Conv1d(input_channels, 64, 1)

        # SA1 组内共享 MLP：输入 4 = 绝对坐标(2)+相对坐标(2) -> (64,64,128)
        self.sa1 = self._make_mlp(4, [64, 64, 128])
        # SA2 组内共享 MLP：输入 C1+2 = 128+2 -> (128,256,512)
        self.sa2 = self._make_mlp(128 + 2, [128, 256, 512])

        # 特征回传 MLP1：N2 特征(512)+SA1 局部特征(128) = 640 -> 256 -> 128
        self.fp1 = self._make_mlp(512 + 128, [256, 128])
        # 解码头：回传特征(128)+逐点基座(64) = 192 -> 128 -> Cout
        self.head = self._make_mlp(128 + 64, [128, output_channels], last_linear=True)

        self._init_weights()

    @staticmethod
    def _make_mlp(in_ch, widths, last_linear=False):
        """按宽度列表构造 Conv1d(k=1) 共享 MLP，中间 ReLU。"""
        seq, cin = [], in_ch
        for i, w in enumerate(widths):
            seq.append(nn.Conv1d(cin, w, 1))
            if not (last_linear and i == len(widths) - 1):
                seq.append(nn.ReLU(inplace=True))
            cin = w
        return nn.Sequential(*seq)

    def _set_abstraction(self, src_coords, src_feats, n_out, mlp):
        """一层 set abstraction：采样 -> kNN 分组 -> 局部共享 MLP -> max-pool。

        src_coords:[B,N,D], src_feats:[B,C,N]（可为坐标本身）,
        n_out 目标子点数; mlp 输入通道数须与分组特征通道数一致。
        返回中心特征 [B,C_out,n_out]、中心坐标 [B,n_out,D]。
        """
        B, N, D = src_coords.shape
        idx = _farthest_point_sample(src_coords, n_out)       # [B,n_out]
        # 中心坐标：与 src_coords 布局一致，返回 [B, n_out, D]
        center_coords = _gather(
            src_coords.transpose(1, 2), idx).transpose(1, 2)  # [B,n_out,D]

        d = _sq_dist(center_coords, src_coords)              # [B,n_out,N]
        kk = min(self.k, N)
        _, knn = d.topk(kk, dim=-1, largest=False)            # [B,n_out,k]

        # 邻点特征/坐标聚集
        if src_feats is not None:
            nf = _gather(src_feats, knn)                      # [B,C,n_out,k]
        else:
            nf = None
        nc = _gather(src_coords.transpose(1, 2), knn)         # [B,D,n_out,k]
        # 中心坐标转 D-first 以便逐通道广播: [B,D,n_out,1]
        rel = nc - center_coords.transpose(1, 2).unsqueeze(-1)  # 相对坐标
        # 组内特征通道拼接（共享 MLP 的输入维度由调用方保证一致）
        if nf is None:
            grouped = torch.cat([nc, rel], dim=1)             # [B,2D,n_out,k]
        else:
            grouped = torch.cat([nf, rel], dim=1)             # [B,C+D,n_out,k]

        B, Cin, n_grp, kk = grouped.shape   # n_grp 个子集点, 每组 kk 个邻点
        flat = grouped.reshape(B, Cin, n_grp * kk)         # 邻点作独立 token
        h = mlp(flat)                                      # [B,C',n_grp*kk]
        h = h.view(B, -1, n_grp, kk)                       # 还原 (n_grp, kk)
        out = h.max(dim=-1).values                         # [B,C',n_grp]
        return out, center_coords

    def _nearest_interp(self, src_feats, dst_coords, src_coords):
        """最近邻插值: src_feats:[B,C,M], dst_coords:[B,N,D],
        src_coords:[B,M,D] -> [B,C,N]。"""
        d = _sq_dist(dst_coords, src_coords)                  # [B,N,M]
        idx = d.argmin(dim=-1)                                # [B,N]
        return _gather(src_feats, idx)

    def forward(self, x):
        # x: [B, Cin, N]
        B, Cin, N = x.shape
        pts = x.transpose(1, 2).contiguous()                  # [B,N,Cin]
        coords = pts                                          # 层级 0 坐标

        # ---- SA1: N -> N1，输入特征为坐标本身 ----
        f1, c1 = self._set_abstraction(
            coords, x, self.n1, self.sa1)                     # f1:[B,128,N1]
        # ---- SA2: N1 -> N2 ----
        f2, c2 = self._set_abstraction(
            c1, f1, self.n2, self.sa2)                        # f2:[B,512,N2]

        # ---- 特征回传 N2 -> N1（最近邻 + skip 拼接 SA1 局部特征）----
        z1 = self._nearest_interp(f2, c1, c2)                 # [B,512,N1]
        z1 = torch.cat([z1, f1], dim=1)                       # [B,640,N1]
        z1 = self.fp1(z1)                                     # [B,128,N1]

        # ---- 回传 N1 -> N（最近邻），拼接逐点基座特征后解码 ----
        base = torch.relu(self.conv0(x))                      # [B,64,N]
        zp = self._nearest_interp(z1, pts, c1)                # [B,128,N]
        f = torch.cat([zp, base], dim=1)                      # [B,192,N]
        out = self.head(f)                                    # [B,Cout,N]

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
