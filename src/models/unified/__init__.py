"""统合网络包（src/models/unified）。

设计文档：返稿意见/返稿修订工作区/docs/U1_统合架构设计.md。
核心类 UnifiedNet：共享语义空间 + 动态路由（热/声两任务统合，作者 D1 决策）；
辅助类 BranchP / BranchF / Router / TaskHead / ResidualBlock 亦作导出，
供按组件统计参数量与消融复用。

接口约定（与 FeatExtractNet 一致）：forward(x: [B, Cin, N], cond=None,
task=None, hard=False) -> (out: [B, Cout, N], diag: dict)。task 只决定取
哪个任务头，不影响路由输入（路由输入禁止任务名/数据集 id，见 unified_net 模块
docstring）。
"""

from .unified_net import (
    UnifiedNet,
    BranchP,
    BranchF,
    Router,
    TaskHead,
    ResidualBlock,
)

__all__ = [
    'UnifiedNet',
    'BranchP',
    'BranchF',
    'Router',
    'TaskHead',
    'ResidualBlock',
]
