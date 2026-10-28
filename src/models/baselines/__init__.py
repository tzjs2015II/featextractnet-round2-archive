"""监督基线模型包（src/models/baselines）。

统一接口约定（与 FeatExtractNet 一致）：
  - 点云类: forward(x: [B, Cin, N]) -> [B, Cout, N]，坐标在通道维、点在最后维
    （热场 Cin=2 / Cout=4；声场网格化点云 Cout=1）。
  - 网格类 FNO2d: forward_grid(grid: [B, Cin, H, W]) -> [B, Cout, H, W]。

导出类：
  CoordMLP                纯坐标逐点 MLP（无结构先验基线）
  PointNetBaseline        标准 PointNet 监督版（全局 max-pool 回灌）
  PointNetPlusPlusBaseline  PointNet++ SSG（kNN 分组，简化但真实）
  DGCNN                   动态图卷积 EdgeConv（逐层重建 kNN 图）
  PipnStyleBaseline       PIPN 仓库 NormalizedGeoPointNet 忠实移植
  FNO2d                   规则网格 2D 谱方法（自实现 FFT 谱卷积）
"""

from .coordinate_mlp import CoordMLP
from .pointnet import PointNetBaseline
from .pointnetpp import PointNetPlusPlusBaseline
from .dgcnn import DGCNN
from .pipn_style import PipnStyleBaseline
from .fno import FNO2d

__all__ = [
    'CoordMLP',
    'PointNetBaseline',
    'PointNetPlusPlusBaseline',
    'DGCNN',
    'PipnStyleBaseline',
    'FNO2d',
]
