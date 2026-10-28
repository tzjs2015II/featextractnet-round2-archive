"""src.core —— 统一实验 harness 基础设施包。

模块
----
thermal_io    热场 IO、train-only 归一化、几何级 batch 采样、逐几何整场评估
              （物理单位 RE/MAE/maxErr + 边界点 RE），几何级留出协议配套。
acoustic_io   声场 IO、工况清单、pooled train 输出统计、整场评估与
              网格/点云互转（FNO 用 [B,Cin,H,W] 组装）。
runner        监督训练通用循环：几何级取数、val 选优 checkpoint、seed 落盘、
              路由 diag 透传、跨 seed 聚合（mean±std + LaTeX 行）。
registry      results/registry.json 追加/更新（按 id 读取-合并-写回）。

接口约定（与 src/utils/metrics.py、data_loader.py 一致）
--------------------------------------------------------
- 评估主口径 = 物理单位 RE：model 输出在归一化域，先在 eval 层反归一化到
  物理单位再算 RE=||ŷ−y||₂/||y||₂；绝不在归一化域计算主指标。
- 热场归一化统计只用 train 几何（fit_stats 传入 train geom 索引）。
- val 只用于模型选择（checkpoint 选优），test 仅在冻结最优权重后评估一次。
- 坐标/通道布局沿用 [B, C, N]（点云）与 [B, C, H, W]（FNO 网格）。
"""

from . import thermal_io, acoustic_io, runner, registry
from .thermal_io import (  # noqa: F401
    load_thermal,
    fit_stats,
    normalize,
    denormalize,
    make_geometric_batches,
    make_thermal_train_provider,
    eval_geometry_wise,
)
from .acoustic_io import (  # noqa: F401
    load_case,
    make_case_list,
    fit_stats_cases,
    eval_case_physical,
    eval_case_model,
    scatter_to_grid,
    grid_to_points,
    grid_to_tensor,
    summarize_case_rows,
)
from .registry import (  # noqa: F401
    append_registry,
    new_experiment_entry,
    DEFAULT_REGISTRY_PATH,
)
from .runner import train_supervised, aggregate_seeds, latex_row  # noqa: F401

__all__ = [
    'thermal_io',
    'acoustic_io',
    'runner',
    'registry',
    'load_thermal',
    'fit_stats',
    'normalize',
    'denormalize',
    'make_geometric_batches',
    'make_thermal_train_provider',
    'eval_geometry_wise',
    'load_case',
    'make_case_list',
    'fit_stats_cases',
    'eval_case_physical',
    'eval_case_model',
    'scatter_to_grid',
    'grid_to_points',
    'grid_to_tensor',
    'summarize_case_rows',
    'append_registry',
    'new_experiment_entry',
    'train_supervised',
    'aggregate_seeds',
    'latex_row',
]
