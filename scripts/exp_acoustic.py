"""exp_acoustic.py —— 声学整场留出对比实验（A2）单 seed 训练 + 聚合登记 / eval-only。

用途
----
实施 Track A2 声学对比实验（实验计划.md §7 Track A）：5 个模型在整场留出
协议下独立训练（每个 (model, seed) 一个进程），跨 seed（0,1,2）聚合 mean±std
并登记 results/registry.json（experiment_id = A2_<model>）。

== 声学数据协议（A2，本文件即权威，docstring 与 scripts/smoke_acoustic_cpu.py 同步）==
* case 全集（来自 configs/acoustic_cases.json，硬编码 role 划分）：
    train = {(200,0.3), (200,0.5), (200,0.7), (300,0.5), (400,0.3), (400,0.5)}
    val   = {(400,0.7)}（仅模型选择，不参与训练/统计）
    test  = {(100,0.3), (100,0.5), (100,0.7), (300,0.3), (300,0.7),
             (500,0.2), (500,0.3), (500,0.5), (500,0.7), 400_0.5_srcR}
  其中 400_0.5_srcR 为 400 Hz 源位置在右侧的变体（source_side=right），用于
  检验对源几何的泛化；test 的 f 全为 train 未见过的 100/500 Hz（低频与高频
  两个方向的外推），(f, Ma) 对与 train 无交集。100_0.2/200_0.2/300_0.2/400_0.2
  四个 Ma=0.2 工况在本协议未使用（A4 频率/Ma 扫描预留）。
  数据文件 data/acoustic/case_{case_id}.npz：481x481 规则网格，坐标 [-15,15] m
  （Opt-1：dx=1/16 m、空气域 30 m），键 p_total/x/y/meta（meta 含 f_hz、Ma、
  source_side、dt_s 等，见 src/core/acoustic_io）。
* 归一化（train-only）：p_total pooled z-score + 坐标仿射 [-1,1]，统计只用
  6 个 train case（acoustic_io.fit_stats_cases）；cond 定常量归一化
  cond = [f_hz/500, Ma]（分母取 Opt-1 的频率上限 500 Hz，随 metrics 记录）。
  模型输入/输出均为归一化域；物理单位 RE/MAE/maxErr 由 acoustic_io 评测层
  反归一化后计算（eval_case_model 内部完成），主指标不在归一化域算。
* 训练（点模型 unif_acous/fourier_nc/fourier_c/mlp_c）：
    默认 --point-mode=fullgrid：每 epoch 遍历 6 个 train case，每个 case 整场
    231361 点作为 1 个 step（steps_per_epoch=6；仅 --point-mode=subset 才退回
    "每步随机采 8192 点的 32 步"模式，该模式实测会在高频波场上梯度坍缩）。
    epochs=1000（与已报道的 A2 协议一致），lr=1e-3 Adam；损失 = 归一化域逐点
    MSE。val_selector = val case (400,0.7) 全网格物理单位 RE（越小越好），每
    val_every=50 epoch 评估一次；最优 checkpoint 冻结后 test 一次。
  FNO（fno）：
    输入用 acoustic_io.grid_to_tensor 组装 [1, Cin=4, H, W]（2 通道仿射坐标
    [-1,1] + 2 通道 cond 常数网格）；每 epoch 6 个 train case 各一个整场 step
    （steps_per_epoch=6），epochs=1000，val/test 同整场（不用点级 chunk）。
* 评测：test 各 case 逐 case 全网格物理单位 RE/MAE/maxErr（点模型经
  eval_case_model、chunk 65536；FNO 整场一次），逐 case 与跨 case 均值都落盘；
  跨 seed(0,1,2) 取 mean±std，写 summary.json 并登记 registry。
* 计时：每 epoch 平均耗时（s_per_epoch）与总 wall_time 写入 metrics protocol。

模型清单（--model）与封装见 build_model()；cond 常量与 case 集合随每个 seed 的
metrics_seed{n}.json 的 protocol 快照记录（归一化统计、case 划分、cond_norm）。

== Track A3 声学一致消融（本脚本的 ab_* token；设计见 A 设计文档 §6）==
与 A2 的 `fourier_c` **同架构、同协议、同 seeds**，只改单一组件，故对照即 A2 的
`fourier_c` 行（levels=16、block_kind='full'）：
  ab_nofourier  fourier_levels=0（去 Fourier 编码，输入维 68->4；参数量 813,569->797,185）
  ab_L8 / ab_L32  fourier_levels=8 / 32（带宽 vs 容量；须并报参数量）
  ab_nores      block_kind='nores'（去残差 skip；参数量与对照相同）
  ab_nonorm     block_kind='nonorm'（去 LayerNorm；参数量略减）
条件化 on/off **不重复跑**——A2 的 `fourier_nc`（无条件）与 `fourier_c`（有 cond）
已是该对照。消融行的 registry id 记为 A3_<model>（与 A2_ 区分）。
编排见 sbatch/acoustic_wp5_ablation.sbatch；对照表落 §4.3。

输入输出
--------
  in  : --data-cfg configs/acoustic_cases.json（default）
  out : --out-dir = 模型级目录（default <ws>/results/acoustic_protocolA/<model>）
        seed{n}/checkpoint_best.pth + seed{n}/metrics_seed{n}.json
        <out-dir>/summary.json（3 个 seed 齐时由聚合写）
        results/registry.json 按 A2_<model>（消融 token ab_* 记为 A3_<model>）
        upsert（fcntl 锁防并发丢写）
  --eval-only <ckpt>：加载给定 checkpoint 对 test case 重评一次（供 A4 频率
        扫描/补测试复用），写 <out-dir>/eval_only_seed{n}.json。

运行方式
--------
  cd 返稿意见/返稿修订工作区
  # 单 seed 训练（GPU；正式运行由 sbatch/acoustic_protocolA.sbatch 编排）
  srun --partition=compute --gres=gpu:1 python3 scripts/exp_acoustic.py \
      --model unif_acous --seed 0 --out-dir results/acoustic_protocolA/unif_acous
  # 仅 eval（加载 checkpoint 补测试，不训练）
  python3 scripts/exp_acoustic.py --model fourier_c \
      --eval-only results/acoustic_protocolA/fourier_c/seed0/checkpoint_best.pth \
      --out-dir results/acoustic_protocolA/fourier_c
  # 跨 seed 聚合 + registry 登记（CPU 可跑，3 个 seed metrics 齐时写）
  python3 scripts/exp_acoustic.py --model unif_acous --aggregate-only \
      --out-dir results/acoustic_protocolA/unif_acous

约束：登录节点不得跑训练；脚本只新增结果文件，不改 src/ 现有文件。
"""

import argparse
import fcntl
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WS not in sys.path:
    sys.path.insert(0, WS)

from src.core.acoustic_io import (  # noqa: E402
    DEFAULT_FIELD,
    coord_affine,
    eval_case_model,
    eval_case_physical,
    fit_stats_cases,
    grid_to_points,
    grid_to_tensor,
    load_case,
    make_case_list,
    summarize_case_rows,
    unzscore_p,
    zscore_p,
)
from src.core import registry, runner  # noqa: E402
from src.models.baselines import FNO2d  # noqa: E402
# imported from its module (not via the package __init__) so that the neuralop
# import cost/noise is paid only by the runs that actually build a UNO row
from src.models.baselines.uno import UNO2d  # noqa: E402
from src.models.featextractnet import FourierMLP  # noqa: E402
from src.models.unified import UnifiedNet  # noqa: E402
from src.models.unified.cond_mlps import (  # noqa: E402
    CondCoordMLP,
    CondFourierMLP,
)
from src.utils.data_loader import set_seed  # noqa: E402
from src.utils.device import get_device  # noqa: E402

# ---------------------------------------------------------------------------
# 协议常量（A2 权威口径，写入各 metrics protocol 快照）
# ---------------------------------------------------------------------------
SPLIT_VERSION = 'acoustic_protocolA_v1'
FIELD = DEFAULT_FIELD  # 'p_total'
# cond = [f_hz/COND_F_DENOM, Ma]，防泄漏的定常量（随 metrics 记录）。
# Opt-1 网格把频率上限从 800 Hz 改为 500 Hz，故分母同步改为 500，
# 与该设计的记录口径一致（docs/A_声学数据与任务设计.md §5.1）。
COND_F_DENOM = 500.0
SEEDS = (0, 1, 2)

# case role 划分（Opt-1 重生成后的 21 工况；必须与
# configs/acoustic_cases.json 的 roles 段、scripts/generate_acoustic_cases.py
# 的 ROLES 常量一致。旧的 800_* 工况已不存在——若这里残留旧 id，
# load_case_groups 的数量校验会直接报错而非静默跑偏）。
CASE_ROLES = {
    # train: 6 例，f 覆盖 200–400 Hz（由旧的 3 例提升到 6 例）
    '200_0.3': 'train', '200_0.5': 'train', '200_0.7': 'train',
    '300_0.5': 'train', '400_0.3': 'train', '400_0.5': 'train',
    '400_0.7': 'val',
    # test: 10 例，整场完全留出（含 100 Hz 低频外推与 500 Hz 高频外推）
    '100_0.3': 'test', '100_0.5': 'test', '100_0.7': 'test',
    '300_0.3': 'test', '300_0.7': 'test',
    '500_0.2': 'test', '500_0.3': 'test', '500_0.5': 'test',
    '500_0.7': 'test', '400_0.5_srcR': 'test',
}
N_CASE_BY_ROLE = {'train': 6, 'val': 1, 'test': 10}

MODEL_TOKENS = ('unif_acous', 'fourier_nc', 'fourier_c', 'mlp_c', 'fno', 'uno',
                # B R2 / A#7 的容量匹配对照：与 fourier_c 参数量对齐（差 0.12%）
                # 的条件化坐标 MLP（hidden 512 -> 450）
                'mlp_c_matched',
                # Track A3 声学一致消融：与 fourier_c 同架构，只改单一组件
                'ab_nofourier', 'ab_L8', 'ab_L32', 'ab_nores', 'ab_nonorm')
# 网格类模型：走 grid_to_tensor 的 [1, 4, H, W] 输入与整场评估路径（B R1 的算子类对标）
GRID_MODEL_TOKENS = frozenset({'fno', 'uno'})
POINT_STEPS_PER_EPOCH = 32
# 网格模型（FNO/UNO）每 epoch 的整场 step 数 = train case 数（make_fno_provider
# 遍历全部 train buffers）。**必须等于 len(train)**：Opt-1 协议把 train 从 3 例提到
# 6 例后，此常量若仍写 3，就会把实跑的 6 记成 3（"记录≠实跑"，同审计 P0-1/P0-4 一类）。
FNO_STEPS_PER_EPOCH = 6
# Epoch budget. Both defaults are set to the value actually used by the reported
# A2 protocol (1000, passed by sbatch/acoustic_protocolA.sbatch) so that running
# without --epochs reproduces the published protocol instead of silently using a
# different budget. Previously the defaults (1500/600) disagreed with the runs
# (1000) — see docs/审计_协议合规核对.md P0-3.
POINT_EPOCHS_DEFAULT = 1000
FNO_EPOCHS_DEFAULT = 1000
POINTS_PER_STEP = 8192
EVAL_CHUNK_DEFAULT = 65536
REG_METRICS = ('re', 'mae', 'max_err')
# R2-IMPL-04B: Fourier layout / strict-fp32 switches. Defaults are legacy so
# that omitting the flags reproduces the published A2/A3 behaviour exactly.
FOURIER_LAYOUT_DEFAULT = 'legacy'
IMPLEMENTATION_VERSION = 'R2-IMPL-04B'
# primary test subset when pointwise correct-source layout is used: the 9
# left-side cases; 400_0.5_srcR (right-side) is kept as a diagnostic row only.
PRIMARY_TEST_CASE_IDS = ('100_0.3', '100_0.5', '100_0.7', '300_0.3',
                         '300_0.7', '500_0.2', '500_0.3', '500_0.5',
                         '500_0.7')
DIAGNOSTIC_TEST_CASE_IDS = ('400_0.5_srcR',)
HISTORICAL_N_CASE = 10


def default_epochs(token):
    return FNO_EPOCHS_DEFAULT if token in GRID_MODEL_TOKENS else POINT_EPOCHS_DEFAULT


def default_steps(token):
    return FNO_STEPS_PER_EPOCH if token in GRID_MODEL_TOKENS else POINT_STEPS_PER_EPOCH


# ---------------------------------------------------------------------------
# 数据装载与 case 划分
# ---------------------------------------------------------------------------
def cond_vector(entry):
    """case 条目的归一化条件标量 cond=[f_hz/COND_F_DENOM, Ma]（常量归一化，防泄漏）。"""
    f = float(entry['f'])
    ma = float(entry['Ma'])
    return [f / COND_F_DENOM, ma]


def _coord_stats(p_stats):
    """从 fit_stats_cases 返回 dict 中提取坐标仿射域（grid_to_tensor 复用）。"""
    return {k: p_stats[k] for k in ('x_min', 'x_max', 'y_min', 'y_max')}


def load_case_groups(data_cfg):
    """读 acoustic_cases.json，按 CASE_ROLES 返回 {'train':[], 'val':[], 'test':[]}。

    其余 case 记 'unassigned' 并在返回 doc 中列出（协议未使用）。返回
    (groups, unused_case_ids)；train/val/test 内部按 case_id 排序保证确定性。
    """
    entries = make_case_list(data_cfg, roles=CASE_ROLES)
    groups = {'train': [], 'val': [], 'test': []}
    unused = []
    for e in entries:
        r = e['role']
        if r in groups:
            if e['f'] is None or e['Ma'] is None:
                raise ValueError(f'case {e["case_id"]}: missing f/Ma params')
            groups[r].append(e)
        else:
            unused.append(e['case_id'])
    for r in groups:
        groups[r].sort(key=lambda e: e['case_id'])
    for r, expect in N_CASE_BY_ROLE.items():
        if len(groups[r]) != expect:
            raise ValueError(
                f'role {r}: expected {expect} cases, got {len(groups[r])}: '
                f'{[e["case_id"] for e in groups[r]]}')
    return groups, sorted(unused)


def _case_snapshot(entries):
    """case 条目 -> JSON 原生 list（protocol 快照用；file 为绝对路径）。"""
    return [{'case_id': e['case_id'], 'f': e['f'], 'Ma': e['Ma'],
             'source_side': e['source_side'], 'file': e['file']}
            for e in entries]


def _protocol_case_groups(groups):
    return {r: _case_snapshot(groups[r]) for r in ('train', 'val', 'test')}


# ---------------------------------------------------------------------------
# 模型与包装（point: forward(x) 用 set_cond 注入；grid: forward(x) -> forward_grid）
# ---------------------------------------------------------------------------
class AcousticHead(nn.Module):
    """点级模型包装：把 (x, cond) 模型包装成 runner 所需 forward(x)。

    kind:
      'unif'   UnifiedNet 声学输出：forward(x, cond, task='acoustic', hard=False)
      'uncond' 无条件模型（FourierMLP）：仅吃 x
      'cond'   条件模型（CondFourierMLP / CondCoordMLP）：forward(x, cond)
    cond 通过 set_cond 预置（训练 provider 每 step 与评测逐 case 设置）；
    forward 内自动把 cond 搬到 x 的设备。
    """

    def __init__(self, core, kind):
        super().__init__()
        self.core = core
        self.kind = kind
        self.grid_based = False
        self._cond = None

    def set_cond(self, cond):
        """设置当前样本条件 cond [1, Nc]（None 表示无 cond）。"""
        self._cond = cond

    def forward(self, x):
        kind = self.kind
        c = self._cond
        if c is not None and c.device != x.device:
            c = c.to(x.device)
        if kind == 'unif':
            out, _diag = self.core(x, cond=c, task='acoustic', hard=False)
            return out
        if kind == 'uncond':
            return self.core(x)
        # kind == 'cond'
        return self.core(x, c)


class GridModel(nn.Module):
    """网格模型包装（FNO2d）：runner/eval 统一走 forward(x_t) -> forward_grid。"""

    def __init__(self, core):
        super().__init__()
        self.core = core
        self.grid_based = True

    def forward(self, x):
        return self.core.forward_grid(x)


def build_model(token, fourier_layout=FOURIER_LAYOUT_DEFAULT):
    """按 token 构造 (模型包装, 描述 label)。模型输入输出均为归一化域。

    fourier_layout: 'legacy'（默认，与已报道协议一致）或 'pointwise'。仅
    UnifiedNet / FourierMLP / CondFourierMLP 三个核心类接受该参数；其余 token
    （CondCoordMLP/FNO/UNO/ab_* 中 CondCoordMLP 通路）不传。

    unif_acous  UnifiedNet(cond_dim=2, branch='unified') 声学任务（软路由全程）
    fourier_nc  FourierMLP(levels=16, blocks=6, hidden=256) 无条件化（纯对照）
    fourier_c   CondFourierMLP（同规模 + cond concat）
    mlp_c       CondCoordMLP(hidden=512, layers=6) ~1.05M
    mlp_c_matched 同上的容量匹配版（hidden=450 -> 814,501 参数），与
                fourier_c（813,569）差 +0.12%，用于 B R2/A#7 的"同等容量"对照
    fno         FNO2d(in_channels=4, modes=16, width=16, n_layers=4) <=1.6M
    """
    if token == 'unif_acous':
        core = UnifiedNet(input_channels=2, cond_dim=2, branch_mode='unified',
                          fourier_layout=fourier_layout)
        return (AcousticHead(core, 'unif'),
                'UnifiedNet(cond_dim=2, branch=unified, task=acoustic, '
                f'soft route, fourier_layout={fourier_layout})')
    if token == 'fourier_nc':
        core = FourierMLP(input_channels=2, output_channels=1,
                          hidden_dim=256, num_blocks=6, fourier_levels=16,
                          fourier_layout=fourier_layout)
        return (AcousticHead(core, 'uncond'),
                'FourierMLP(levels=16, blocks=6, hidden=256), unconditional, '
                f'fourier_layout={fourier_layout}')
    if token == 'fourier_c':
        core = CondFourierMLP(input_channels=2, output_channels=1, cond_dim=2,
                              hidden_dim=256, num_blocks=6, fourier_levels=16,
                              fourier_layout=fourier_layout)
        return (AcousticHead(core, 'cond'),
                'CondFourierMLP(levels=16, blocks=6, hidden=256, cond concat, '
                f'fourier_layout={fourier_layout})')
    if token == 'mlp_c':
        core = CondCoordMLP(input_channels=2, output_channels=1, cond_dim=2,
                            hidden_dim=512, num_layers=6)
        return (AcousticHead(core, 'cond'),
                'CondCoordMLP(hidden=512, layers=6, cond concat) ~1.05M')
    if token == 'mlp_c_matched':
        # B R2 / A#7：与 fourier_c 容量对齐的坐标 MLP。hidden=450 给出 814,501 参数，
        # 相对 fourier_c（813,569）差 +0.12%；其余（cond concat、层数、优化预算、
        # 评测）与 mlp_c 完全相同，故两者之差可归因于"宽度/容量"而非协议。
        core = CondCoordMLP(input_channels=2, output_channels=1, cond_dim=2,
                            hidden_dim=450, num_layers=6)
        return (AcousticHead(core, 'cond'),
                'CondCoordMLP(hidden=450, layers=6, cond concat) ~0.81M '
                '(capacity-matched to fourier_c)')
    if token == 'fno':
        core = FNO2d(in_channels=4, out_channels=1,
                     modes=16, width=16, n_layers=4)
        return (GridModel(core),
                'FNO2d(modes=16, width=16, n_layers=4, in=2coord+2cond)')
    if token == 'uno':
        # 官方 neuraloperator 的 UNO（B R1 点名的算子类对标；配置取自该包自带测试，
        # 见 src/models/baselines/uno.py 与 docs/B_扩展对标设计.md §6）
        core = UNO2d(in_channels=4, out_channels=1)
        return (GridModel(core),
                'UNO2d(official neuralop.models.UNO, in=2coord+2cond, out=1)')

    # ---- Track A3 声学一致消融（与 fourier_c 同架构，只改单一组件）--------
    # 设计见 docs/A_声学数据与任务设计.md §6；对照 = fourier_c（levels=16,
    # block_kind='full'，即 A2 主对比里的同一模型）。
    if token == 'ab_nofourier':
        core = CondFourierMLP(input_channels=2, output_channels=1, cond_dim=2,
                              hidden_dim=256, num_blocks=6, fourier_levels=0,
                              fourier_layout=fourier_layout)
        return (AcousticHead(core, 'cond'),
                'CondFourierMLP(levels=0 [no Fourier], blocks=6, hidden=256) '
                '== fourier_c minus Fourier encoding')
    if token == 'ab_L8':
        core = CondFourierMLP(input_channels=2, output_channels=1, cond_dim=2,
                              hidden_dim=256, num_blocks=6, fourier_levels=8,
                              fourier_layout=fourier_layout)
        return (AcousticHead(core, 'cond'),
                'CondFourierMLP(levels=8, blocks=6, hidden=256) '
                '== fourier_c with reduced Fourier bandwidth')
    if token == 'ab_L32':
        core = CondFourierMLP(input_channels=2, output_channels=1, cond_dim=2,
                              hidden_dim=256, num_blocks=6, fourier_levels=32,
                              fourier_layout=fourier_layout)
        return (AcousticHead(core, 'cond'),
                'CondFourierMLP(levels=32, blocks=6, hidden=256) '
                '== fourier_c with doubled Fourier bandwidth')
    if token == 'ab_nores':
        core = CondFourierMLP(input_channels=2, output_channels=1, cond_dim=2,
                              hidden_dim=256, num_blocks=6, fourier_levels=16,
                              block_kind='nores', fourier_layout=fourier_layout)
        return (AcousticHead(core, 'cond'),
                'CondFourierMLP(levels=16, blocks=6, hidden=256, no residual) '
                '== fourier_c minus residual skip')
    if token == 'ab_nonorm':
        core = CondFourierMLP(input_channels=2, output_channels=1, cond_dim=2,
                              hidden_dim=256, num_blocks=6, fourier_levels=16,
                              block_kind='nonorm', fourier_layout=fourier_layout)
        return (AcousticHead(core, 'cond'),
                'CondFourierMLP(levels=16, blocks=6, hidden=256, no LayerNorm) '
                '== fourier_c minus LayerNorm')
    raise ValueError(f'unknown model token: {token}')


# ---------------------------------------------------------------------------
# 训练数据 provider
# ---------------------------------------------------------------------------
def build_train_point_buffers(train_entries, p_stats):
    """预载 6 个 train case 的归一化网格点缓冲（xn 坐标 / zn 目标 / cond）。

    每个 buffer：{'entry', 'cond': [Nc] float32,
                  'xn': [N,2] float32 仿射坐标, 'zn': [N] float32 z-score p}
    grid_to_points 的 ravel 顺序与 p2d.ravel() 一致，xn 行与 zn 逐点对应。
    """
    bufs = []
    for e in train_entries:
        x2d, y2d, p2d, _ = load_case(e['file'])
        pts = grid_to_points(x2d, y2d)                      # [N, 2]
        xn = coord_affine(pts, p_stats['x_min'], p_stats['x_max'],
                          p_stats['y_min'], p_stats['y_max'])  # [N, 2] float32
        zn = zscore_p(p2d.ravel(), p_stats['p_mean'],
                      p_stats['p_std']).astype(np.float32)     # [N]
        bufs.append({'entry': e,
                     'cond': np.asarray(cond_vector(e), dtype=np.float32),
                     'xn': xn, 'zn': zn})
    return bufs


def make_point_provider(model, buffers, steps_per_epoch, n_points, seed):
    """每 step 从 train case 池随机取 1 case、采 n_points 点；返回 (x,y) CPU 张量。

    用独立 torch.Generator(seed, epoch) 抽样，不消耗全局 RNG（训练确定性不依赖
    外层随机序列消费顺序）。step 前调用 model.set_cond(该 case cond)。
    """

    def data_provider(epoch):
        gen = torch.Generator().manual_seed(int(seed) * 100003 + int(epoch))
        nb = len(buffers)
        for _ in range(steps_per_epoch):
            ci = int(torch.randint(nb, (1,), generator=gen).item())
            buf = buffers[ci]
            N = buf['xn'].shape[0]
            n = min(n_points, N)
            idx = torch.randperm(N, generator=gen)[:n].numpy()
            model.set_cond(torch.from_numpy(buf['cond']).unsqueeze(0))  # [1,Nc]
            x_t = torch.from_numpy(buf['xn'][idx].T).contiguous().unsqueeze(0)  # [1,2,n]
            y_t = torch.from_numpy(buf['zn'][idx]).reshape(1, 1, n).contiguous()
            yield x_t, y_t

    return data_provider


def build_fno_grid_buffers(train_entries, p_stats):
    """预组装 6 个 train case 的整网格张量 [1,4,H,W] 与归一化目标 [1,1,H,W]。"""
    cs = _coord_stats(p_stats)
    bufs = []
    for e in train_entries:
        x2d, y2d, p2d, _ = load_case(e['file'])
        x_t, y_norm_t, _ = grid_to_tensor(
            x2d, y2d, p2d=p2d, cond=cond_vector(e), p_stats=p_stats,
            coord_stats=cs)
        bufs.append({'entry': e, 'x': x_t, 'y': y_norm_t})
    return bufs


def make_fullgrid_point_provider(model, buffers):
    """整场训练：每 epoch 把每个 train case 的整场点作为 1 个 step。

    与子集模式（make_point_provider）不同，这里每步使用完整 231k 点网格，
    避免小随机子集在高频波场上的梯度方差坍缩（实测 8192 点子集训练坍缩到
    常数输出、loss≈1.0；整场/≥65k 子集可正常学习）。
    step 前调用 model.set_cond(该 case cond)，语义与子集模式一致。
    """

    def data_provider(_epoch):
        for b in buffers:
            model.set_cond(torch.from_numpy(b['cond']).unsqueeze(0))
            x_t = torch.from_numpy(b['xn'].T).contiguous().unsqueeze(0)
            y_t = torch.from_numpy(b['zn']).reshape(1, 1, -1).contiguous()
            yield x_t, y_t

    return data_provider


def make_fno_provider(buffers):
    """FNO/UNO：每 epoch 每个 train case 各一个整场 step
    （steps_per_epoch = len(train) = FNO_STEPS_PER_EPOCH = 6）。"""

    def data_provider(_epoch):
        for b in buffers:
            yield b['x'], b['y']

    return data_provider


def make_data_provider(model, token, groups, p_stats, args):
    """按 token/point_mode 构造 data_provider（含预载缓冲）。"""
    if token in GRID_MODEL_TOKENS:
        bufs = build_fno_grid_buffers(groups['train'], p_stats)
        return make_fno_provider(bufs)
    bufs = build_train_point_buffers(groups['train'], p_stats)
    if args.point_mode == 'fullgrid':
        return make_fullgrid_point_provider(model, bufs)
    steps = args.steps_per_epoch if args.steps_per_epoch else default_steps(token)
    return make_point_provider(model, bufs, steps, args.points_per_step, args.seed)


# ---------------------------------------------------------------------------
# 评测（val 选优 / test 一次）：与 acoustic_io.eval_case_model 同口径
# ---------------------------------------------------------------------------
def _eval_fno_grid(model, entry, p_stats, device):
    """FNO 整场评估：grid_to_tensor -> forward_grid -> 反 z-score -> 物理单位指标。"""
    x2d, y2d, p2d, _ = load_case(entry['file'])
    x_t, _, _ = grid_to_tensor(x2d, y2d, p2d=None, cond=cond_vector(entry),
                               p_stats=None, coord_stats=_coord_stats(p_stats))
    with torch.no_grad():
        pz = model(x_t.to(device))                        # [1,1,H,W] z 域
    p_phys = unzscore_p(pz.detach().cpu().numpy()[0, 0],
                        p_stats['p_mean'], p_stats['p_std'])
    m = eval_case_physical(p_phys, p2d)
    cid = entry['case_id']
    row = {'case_id': cid, 're': m['re'], 'mae': m['mae'],
           'max_err': m['max_err'], 'f': entry['f'], 'Ma': entry['Ma'],
           'source_side': entry['source_side']}
    return {'agg': {'re': {FIELD: m['re']}, 'mae': {FIELD: m['mae']},
                    'max_err': {FIELD: m['max_err']}},
            'rows': [row],
            'pred_grid': p_phys.astype(np.float32),
            'truth_grid': p2d}


def _eval_case_metrics(model, entry, p_stats, device, chunk_points):
    """单 case 整场评测（逐 case 全网格物理单位 RE/MAE/maxErr）。

    点模型经 acoustic_io.eval_case_model（chunk 65536，内部反归一化）；
    FNO 整场一次（grid_based）。返回结构与 eval_case_model 一致
    {'agg', 'rows', 'pred_grid', 'truth_grid'}；rows[0] 附 f/Ma/source_side。
    """
    if model.grid_based:
        return _eval_fno_grid(model, entry, p_stats, device)
    cond_t = torch.tensor(cond_vector(entry), dtype=torch.float32,
                          device=device).unsqueeze(0)      # [1, Nc]
    model.set_cond(cond_t)
    ev = eval_case_model(lambda x: model(x), entry['file'], p_stats,
                         device, chunk_points=chunk_points,
                         case_id=entry['case_id'])
    row = ev['rows'][0]
    row.update({'f': entry['f'], 'Ma': entry['Ma'],
                'source_side': entry['source_side']})
    ev['rows'] = [row]
    return ev


def build_val_selector(model, groups, p_stats, device, chunk_points):
    """val_selector：val case (400,0.3) 全网格物理单位 RE（越小越好）。"""
    val_entry = groups['val'][0]

    def selector(_model):
        ev = _eval_case_metrics(_model, val_entry, p_stats, device, chunk_points)
        return float(ev['agg']['re'][FIELD])

    return selector


def build_test_eval(model, groups, p_stats, device, chunk_points,
                    fixed_source_primary=False):
    """test_eval_fn：test case 逐 case 整场评测（冻结最优权重后调一次）。

    fixed_source_primary=False（默认，legacy）：保留全部 10 行，agg 覆盖全部
    10 case（旧行为不变）。

    fixed_source_primary=True（仅 pointwise 正确源布局运行）：仍保留全部 10
    行（400_0.5_srcR 作为 diagnostic），但 agg 只覆盖 9 个 left-side 主 case。
    行本身保持原始 role/ID 顺序，不重排、不改写任何 case_id。
    """
    test_entries = groups['test']

    def test_fn(_model):
        rows = []
        for e in test_entries:
            ev = _eval_case_metrics(_model, e, p_stats, device, chunk_points)
            rows.append(ev['rows'][0])
        n_case = len(rows)
        if not fixed_source_primary:
            return {'n_case': n_case, 'field': FIELD,
                    'agg': summarize_case_rows(rows, field_name=FIELD),
                    'rows': rows}
        primary_ids = set(PRIMARY_TEST_CASE_IDS)
        diagnostic_ids = set(DIAGNOSTIC_TEST_CASE_IDS)
        row_ids = [r['case_id'] for r in rows]
        if len(set(row_ids)) != len(row_ids):
            raise AssertionError(
                f'fixed_source_primary: 行 case_id 必须互不相同: {row_ids}')
        if set(row_ids) != (primary_ids | diagnostic_ids):
            raise AssertionError(
                'fixed_source_primary: 行 case_id 必须恰为 primary9∪diagnostic1: '
                f'rows={sorted(set(row_ids))} expected='
                f'{sorted(primary_ids | diagnostic_ids)}')
        primary_rows = []
        diagnostic_rows = []
        for e, row in zip(test_entries, rows):
            cid = row['case_id']
            side = row.get('source_side', e.get('source_side'))
            if cid in diagnostic_ids:
                if side is not None and side == 'left':
                    raise AssertionError(
                        f'diagnostic case {cid} 的 source_side 不得为 left')
                diagnostic_rows.append(row)
            elif cid in primary_ids:
                if side is not None and side == 'right':
                    raise AssertionError(
                        f'primary case {cid} 的 source_side 不得为 right')
                primary_rows.append(row)
            else:
                raise AssertionError(
                    'fixed_source_primary: 未知 case_id %r（不得回退归类）' % cid)
        if len(primary_rows) != 9:
            raise AssertionError(
                f'fixed_source_primary: primary 必须 9 例，实际 {len(primary_rows)}')
        if len(diagnostic_rows) != 1:
            raise AssertionError(
                f'fixed_source_primary: diagnostic 必须 1 例，实际 '
                f'{len(diagnostic_rows)}')
        return {'n_case': len(primary_rows), 'field': FIELD,
                'agg': summarize_case_rows(primary_rows, field_name=FIELD),
                'agg_all_cases': summarize_case_rows(rows, field_name=FIELD),
                'rows': rows,
                'primary_case_ids': [r['case_id'] for r in primary_rows],
                'n_primary_case': len(primary_rows),
                'diagnostic_rows': diagnostic_rows,
                'diagnostic_case_ids': [r['case_id'] for r in diagnostic_rows],
                'n_diagnostic_case': len(diagnostic_rows),
                'historical_n_case': HISTORICAL_N_CASE}

    return test_fn


# ---------------------------------------------------------------------------
# 单 seed 训练
# ---------------------------------------------------------------------------
def _default_outdir(model, fourier_layout):
    """默认模型级输出目录。legacy 走旧路径；pointwise 落到独立的
    results/round2_acoustic_pointwise/<model>，绝不与旧协议结果混放。"""
    if fourier_layout == 'pointwise':
        return os.path.join(WS, 'results', 'round2_acoustic_pointwise', model)
    return os.path.join(WS, 'results', 'acoustic_protocolA', model)


def _validate_custom_outdir(out_dir):
    """自定义 --out-dir 必须带 round2/r2 路径成分，且不得位于旧协议目录内。

    这是防止误把 R2 pointwise 结果写进 A2/A3 旧结果目录（会污染历史材料）的
    硬闸门；不通过直接报错而非静默修正。"""
    norm = os.path.realpath(os.path.abspath(out_dir)).replace('\\', '/')
    comps = [c for c in norm.split('/') if c]
    low = [c.lower() for c in comps]
    has_r2 = any(('round2' in c) or ('r2' in c) for c in low)
    if not has_r2:
        raise ValueError(
            '--out-dir 需含 round2 或 r2 路径成分（当前: %s）' % out_dir)
    forbidden = ('results/acoustic_protocolA', 'results/acoustic_ablation')
    for fb in forbidden:
        if fb.lower() in norm.lower():
            raise ValueError(
                '--out-dir 不得位于旧结果目录 %s 内（当前: %s）'
                % (fb, out_dir))


def resolve_paths(args):
    data_cfg = args.data_cfg or os.path.join(WS, 'configs', 'acoustic_cases.json')
    if not os.path.exists(data_cfg):
        raise FileNotFoundError(f'acoustic_cases.json 不存在: {data_cfg}')
    # --out-dir 语义 = 模型级目录：seed 子目录 = <out-dir>/seed{n}，
    # summary.json 与 registry 均以 <out-dir> 为锚（与 exp_thermal.py 一致）。
    fourier_layout = getattr(args, 'fourier_layout', FOURIER_LAYOUT_DEFAULT)
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = _default_outdir(args.model, fourier_layout)
    else:
        out_dir = os.path.abspath(out_dir)
        # legacy 用户显式指定目录时保持旧宽松行为；pointwise 才强制闸门。
        if fourier_layout == 'pointwise':
            _validate_custom_outdir(out_dir)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    return data_cfg, out_dir


def _protocol_extra(args, groups, p_stats, model_label, points_per_step,
                    steps_per_epoch):
    """protocol 快照：case 集合 + cond 常量 + 归一化统计（随 metrics_seed 落盘）。"""
    return {
        'split_version': SPLIT_VERSION,
        'experiment': 'A2_acoustic_full-field-leave-out',
        'dataset_config': os.path.abspath(args.data_cfg),
        'case_set': _protocol_case_groups(groups),
        'n_cases': dict(N_CASE_BY_ROLE),
        'cond_norm': {'spec': f'cond=[f_hz/{COND_F_DENOM:g}, Ma]',
                      'f_denom': COND_F_DENOM,
                      'ma_scale': 1.0},
        'norm': 'p_total pooled z-score + coord affine [-1,1], fit on '
                f'{len(groups["train"])} train cases only (fit_stats_cases)',
        'p_stats': {'p_mean': float(p_stats['p_mean']),
                    'p_std': float(p_stats['p_std']),
                    'n_train_points': int(p_stats['n_points']),
                    'x_min': p_stats['x_min'], 'x_max': p_stats['x_max'],
                    'y_min': p_stats['y_min'], 'y_max': p_stats['y_max']},
        'points_per_step': int(points_per_step),
        'steps_per_epoch': int(steps_per_epoch),
        'eval_chunk': int(args.eval_chunk),
        'model_label': model_label,
        'fourier_layout': getattr(args, 'fourier_layout', FOURIER_LAYOUT_DEFAULT),
        'implementation_version': IMPLEMENTATION_VERSION,
        # 记录实际生效的 bool（CPU 上也照实记录），不随 cuda 可用性置 None。
        'tf32_matmul_allow': bool(torch.backends.cuda.matmul.allow_tf32),
        'tf32_cudnn_allow': bool(torch.backends.cudnn.allow_tf32),
    }


def train_one(args, groups, p_stats, device):
    token = args.model
    epochs = args.epochs if args.epochs else default_epochs(token)
    model_root = args.out_dir
    seed_dir = os.path.join(model_root, f'seed{args.seed}')
    # 防静默续跑：同 seed 目录已存在 checkpoint_best/metrics 时直接报错，
    # 交由人工清理或换 seed 目录，绝不覆盖/续训旧产物。
    for guard in ('checkpoint_best.pth', f'metrics_seed{int(args.seed)}.json'):
        gp = os.path.join(seed_dir, guard)
        if os.path.exists(gp):
            raise FileExistsError(
                f'拒绝训练：{gp} 已存在（同 seed 产物禁止静默覆盖）。'
                '请改用新的 --out-dir 或先人工归档旧结果。')
    os.makedirs(seed_dir, exist_ok=True)

    set_seed(args.seed)
    model, model_label = build_model(token, fourier_layout=args.fourier_layout)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[model] token={token} | {model_label} | params={n_params:,}',
          flush=True)

    if token in GRID_MODEL_TOKENS:
        if args.steps_per_epoch not in (None, FNO_STEPS_PER_EPOCH):
            raise SystemExit(
                f'{token} 的 steps_per_epoch 固定为 {FNO_STEPS_PER_EPOCH}，'
                f'不得覆盖（当前 {args.steps_per_epoch}）')
        steps = FNO_STEPS_PER_EPOCH
    elif args.point_mode == 'fullgrid':
        steps = len(groups['train'])  # 每 epoch = 每个 train case 一次整场
    else:
        steps = args.steps_per_epoch if args.steps_per_epoch else POINT_STEPS_PER_EPOCH
    provider = make_data_provider(model, token, groups, p_stats, args)
    extra = _protocol_extra(args, groups, p_stats, model_label,
                            args.points_per_step, steps)
    extra['fixed_source_primary'] = bool(
        args.fixed_source_primary or args.fourier_layout == 'pointwise')
    if args.point_mode == 'fullgrid':
        extra['point_mode'] = 'fullgrid'
        extra['points_per_step'] = None

    t_start = time.monotonic()
    summary = runner.train_supervised(
        model, task='acoustic', data_provider=provider,
        epochs=epochs, lr=args.lr, seed=args.seed, device=device,
        val_eval_every=args.val_every,
        val_selector=build_val_selector(model, groups, p_stats, device,
                                        args.eval_chunk),
        test_eval_fn=build_test_eval(
            model, groups, p_stats, device, args.eval_chunk,
            fixed_source_primary=(args.fourier_layout == 'pointwise'
                                     or args.fixed_source_primary)),
        out_dir=seed_dir, patience=args.patience,
        protocol_extra=extra)
    wall_s = time.monotonic() - t_start
    n_ep = int(summary['n_epochs'])
    s_per_epoch = wall_s / n_ep if n_ep else float('nan')

    mpath = os.path.join(seed_dir, f'metrics_seed{int(args.seed)}.json')
    with open(mpath, 'r', encoding='utf-8') as fh:
        metrics = json.load(fh)
    # --- 协议快照 / 实跑一致性自检（P0-4）-------------------------------
    # 记录值必须等于实际生效值，否则协议就是"意图"而非"事实"。历史上出现过
    # epochs 记录值(1000)与默认值(1500)、以及声明步数与实跑步数不符的问题，
    # 因此把该检查固化成硬断言。
    for key, recorded, actual in (
            ('epochs', metrics['protocol'].get('epochs'), int(epochs)),
            ('lr', metrics['protocol'].get('lr'), float(args.lr)),
            ('steps_per_epoch', metrics['protocol'].get('steps_per_epoch'),
             int(steps)),
            ('seed', metrics['protocol'].get('seed'), int(args.seed))):
        if recorded is None or abs(float(recorded) - float(actual)) > 1e-12:
            raise AssertionError(
                f'protocol/actual mismatch for {key}: recorded={recorded!r} '
                f'actual={actual!r} (model={token}, seed={args.seed})')
    metrics['protocol']['s_per_epoch'] = round(s_per_epoch, 6)
    metrics['protocol']['wall_time_s'] = round(wall_s, 3)
    with open(mpath, 'w', encoding='utf-8') as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)
        fh.write('\n')

    agg = metrics['test']['agg']
    print(f'[train] seed={args.seed} done | epochs={n_ep} | '
          f'best_epoch={summary["best_epoch"]} | '
          f'best_val(RE)={summary["best_val_score"]:.6f} | '
          f's_per_epoch={s_per_epoch:.3f}s | test agg RE='
          f'{agg["re"][FIELD]:.4f} (mae={agg["mae"][FIELD]:.3e}, '
          f'max={agg["max_err"][FIELD]:.3e})', flush=True)
    for row in metrics['test']['rows']:
        print(f'  [test-case] {row["case_id"]:<14} '
              f'RE={row["re"]:.4f} MAE={row["mae"]:.3e} '
              f'maxErr={row["max_err"]:.3e}', flush=True)
    return model_root, n_params


# ---------------------------------------------------------------------------
# eval-only（加载 checkpoint 补测，A4 频率扫描/补测试复用）
# ---------------------------------------------------------------------------
def run_eval_only(args, groups, p_stats, device):
    token = args.model
    ckpt = torch.load(args.eval_only, map_location=device)
    seed = int(ckpt.get('seed', 0))
    # pointwise 重评必须绑定同 checkpoint 的 sidecar 协议快照；缺失/不匹配
    # 直接拒绝，不得把旧 legacy checkpoint 重新解释为新 pointwise 模型。
    if getattr(args, 'fourier_layout', FOURIER_LAYOUT_DEFAULT) == 'pointwise':
        sidecar = os.path.join(os.path.dirname(os.path.abspath(args.eval_only)),
                               f'metrics_seed{seed}.json')
        if not os.path.exists(sidecar):
            raise SystemExit(
                f'[eval-only] pointwise 重评需要 checkpoint 旁的协议快照 '
                f'{sidecar}（缺失，拒绝重解释旧 checkpoint）。')
        with open(sidecar, 'r', encoding='utf-8') as fh:
            sidecar_doc = json.load(fh)
        actual_layout = (sidecar_doc.get('protocol') or {}).get('fourier_layout')
        if actual_layout != 'pointwise':
            raise SystemExit(
                f'[eval-only] sidecar {sidecar} 的 protocol.fourier_layout='
                f'{actual_layout!r}，与请求 pointwise 不符（拒绝重解释）。')
    model, model_label = build_model(token, fourier_layout=args.fourier_layout)
    model = model.to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'[eval-only] token={token} ckpt={args.eval_only} seed={seed} '
          f'({model_label})', flush=True)

    ev = build_test_eval(
        model, groups, p_stats, device, args.eval_chunk,
        fixed_source_primary=(args.fourier_layout == 'pointwise'))(model)
    agg = ev['agg']
    print(f'[eval-only] test agg RE={agg["re"][FIELD]:.4f} '
          f'(mae={agg["mae"][FIELD]:.3e}, max={agg["max_err"][FIELD]:.3e})',
          flush=True)
    for row in ev['rows']:
        print(f'  [test-case] {row["case_id"]:<14} '
              f'RE={row["re"]:.4f} MAE={row["mae"]:.3e} '
              f'maxErr={row["max_err"]:.3e}', flush=True)

    doc = {
        'seed': seed,
        'task': 'acoustic',
        'mode': 'eval_only',
        'checkpoint': os.path.abspath(args.eval_only),
        'model': model.__class__.__name__,
        'model_label': model_label,
        'model_params': int(sum(p.numel() for p in model.parameters())),
        'implementation_version': IMPLEMENTATION_VERSION,
        'protocol': _protocol_extra(args, groups, p_stats, model_label,
                                    args.points_per_step,
                                    FNO_STEPS_PER_EPOCH if token in
                                    GRID_MODEL_TOKENS else
                                    (args.steps_per_epoch or
                                     POINT_STEPS_PER_EPOCH)),
        'test': ev,
    }
    out_path = os.path.join(args.out_dir, f'eval_only_seed{seed}.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
        fh.write('\n')
    print(f'[eval-only] written -> {out_path}', flush=True)


# ---------------------------------------------------------------------------
# 跨 seed 聚合 + registry 登记
# ---------------------------------------------------------------------------
def _load_all_seed_metrics(model_root):
    """读 <model_root>/seed{s}/metrics_seed{s}.json，返回按 seed 升序 [(s, doc)]。"""
    out = []
    for s in SEEDS:
        fp = os.path.join(model_root, f'seed{s}', f'metrics_seed{s}.json')
        if os.path.exists(fp):
            with open(fp, 'r', encoding='utf-8') as fh:
                out.append((s, json.load(fh)))
    return out


def _mean_std(vals):
    arr = np.asarray([float(v) for v in vals], dtype=np.float64)
    if arr.size == 0:
        return None, None
    return (float(arr.mean()),
            float(arr.std(ddof=1)) if arr.size > 1 else 0.0)


def _agg_case_metric_across_seeds(docs, metric, case_id):
    vals = []
    for _, d in docs:
        for row in d['test']['rows']:
            if row['case_id'] == case_id:
                vals.append(float(row[metric]))
                break
    m, s = _mean_std(vals)
    return {'mean': m, 'std': s, 'n': len(vals)}


def _locked_append_registry(entry, registry_path):
    lock_path = registry_path + '.lock'
    with open(lock_path, 'w', encoding='utf-8') as lfh:
        fcntl.flock(lfh, fcntl.LOCK_EX)
        try:
            return registry.append_registry(entry, registry_path=registry_path)
        finally:
            fcntl.flock(lfh, fcntl.LOCK_UN)


def aggregate_and_register(args):
    """跨 seed(0,1,2) 聚合（mean±std）、写 summary.json；3 seed 齐时登记 registry。

    registry experiment_id = A2_<model>；Track A3 的消融 token（`ab_*`）记为
    A3_<model>，避免与 A2 主对比混在同一 id 空间。fcntl 锁防多任务并发丢写
    （与 exp_thermal.py 同款）。seed 未齐时打印提示并跳过登记。
    """
    token = args.model
    exp_prefix = 'A3' if token.startswith('ab_') else 'A2'
    model_root = args.out_dir
    docs = _load_all_seed_metrics(model_root)
    if not docs:
        raise FileNotFoundError(f'{model_root} 下无 metrics_seed*.json')
    n_seeds = len(docs)
    case_ids = [r['case_id'] for r in docs[0][1]['test']['rows']]
    first = docs[0][1]['protocol']
    pointwise = (first.get('fourier_layout') == 'pointwise')
    fixed_primary = bool(pointwise or first.get('fixed_source_primary'))
    # 请求的协议必须与实际首 doc 协议一致；在任何写盘之前硬校验，
    # 绝不把 request=pointwise / actual=legacy 的产物当 legacy 静默聚合。
    requested_layout = getattr(args, 'fourier_layout', FOURIER_LAYOUT_DEFAULT)
    if requested_layout == 'pointwise' and not pointwise:
        raise SystemExit(
            '[aggregate] 请求 --fourier-layout pointwise，但实际首 doc '
            f'protocol.fourier_layout={first.get("fourier_layout")!r}（拒绝回退为 legacy）。')
    if pointwise:
        _verify_pointwise_docs(docs, first)
        exp_id = f'{exp_prefix}_{token}_R2_pointwise'
    else:
        exp_id = f'{exp_prefix}_{token}'

    # primary case id 集合（fixed_source_primary 为 9 例；旧 legacy 为全部 10 例）
    if fixed_primary:
        primary_ids = list(docs[0][1]['test'].get('primary_case_ids')
                           or PRIMARY_TEST_CASE_IDS)
    else:
        primary_ids = case_ids

    # 跨 case 均值指标（每 doc 的 test.agg.<metric>.<field>）跨 seed mean±std
    over_cases = {}
    for metric in REG_METRICS:
        vals = [d['test']['agg'][metric][FIELD] for _, d in docs]
        m, s = _mean_std(vals)
        over_cases[metric] = {'mean': m, 'std': s, 'n': n_seeds,
                              'field': FIELD, 'values': vals}
    # 逐 case 跨 seed mean±std
    per_case = {}
    for cid in case_ids:
        per_case[cid] = {m: _agg_case_metric_across_seeds(docs, m, cid)
                         for m in REG_METRICS}
    # val best epoch/score 跨 seed（仅供参考）
    val_best = [float(d['val_best']['score']) for _, d in docs
                if d.get('val_best') and d['val_best'].get('score') is not None]
    val_best_m, val_best_s = _mean_std(val_best)

    print(f'\n=== aggregate {exp_prefix}_{token} ({model_root}) | seeds='
          f'{[s for s, _ in docs]} | n={n_seeds} ===', flush=True)
    for metric in REG_METRICS:
        print(f'  test agg over cases {metric:<7} mean±std across seeds: '
              f'{over_cases[metric]["mean"]:.4f} ± {over_cases[metric]["std"]:.4f}'
              if over_cases[metric]['mean'] is not None
              else f'  test agg {metric}: --')
    for cid in case_ids:
        pc = per_case[cid]['re']
        if pc['mean'] is not None:
            print(f'  case {cid:<14} RE={pc["mean"]:.4f} ± {pc["std"]:.4f}')

    sp = [d['protocol'].get('s_per_epoch') for _, d in docs
          if d['protocol'].get('s_per_epoch') is not None]
    summary = {
        'experiment_id': exp_id,
        'model': token,
        'fourier_layout': first.get('fourier_layout',
                                    FOURIER_LAYOUT_DEFAULT),
        'implementation_version': IMPLEMENTATION_VERSION,
        'tf32_matmul_allow': first.get('tf32_matmul_allow'),
        'tf32_cudnn_allow': first.get('tf32_cudnn_allow'),
        'primary_case_ids': primary_ids if pointwise else None,
        'model_label': first.get('model_label'),
        'model_params': first.get('model_params'),
        'split_version': first.get('split_version'),
        'case_set': first.get('case_set'),
        'cond_norm': first.get('cond_norm'),
        'n_seeds': n_seeds,
        'seeds': [s for s, _ in docs],
        'epochs': first.get('epochs'),
        'lr': first.get('lr'),
        'val_eval_every': first.get('val_eval_every'),
        'n_cases': first.get('n_cases'),
        'agg_over_cases': over_cases,
        'per_case': per_case,
        'val_best_re_mean': val_best_m,
        'val_best_re_std': val_best_s,
        's_per_epoch_mean': (float(np.mean(sp)) if sp else None),
        'seed_files': [os.path.join(model_root, f'seed{s}',
                                    f'metrics_seed{s}.json') for s, _ in docs],
    }
    with open(os.path.join(model_root, 'summary.json'), 'w',
              encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
        fh.write('\n')

    if n_seeds < len(SEEDS):
        print(f'[skip registry] 只有 {n_seeds}/{len(SEEDS)} 个 seed 的 metrics，'
              '不做 registry 登记（非完整 3-seed 运行）。')
        return

    if not pointwise:
        pass
    elif not (args.fourier_layout == 'pointwise'
              and args.strict_fp32):
        raise SystemExit(
            '[aggregate] pointwise 聚合需显式 --fourier-layout pointwise '
            '--strict-fp32（清晰性要求）。')
    proto_extra = {}
    if pointwise:
        proto_extra = {
            'fourier_layout': 'pointwise',
            'tf32_disabled': True,
            'primary_test_case_ids': primary_ids,
            'historical_n_case': HISTORICAL_N_CASE,
            'n_primary_cases': len(primary_ids),
            'n_diagnostic_cases': len(case_ids) - len(primary_ids),
        }
    elif fixed_primary:
        proto_extra = {
            'fixed_source_primary': True,
            'primary_test_case_ids': primary_ids,
            'historical_n_case': HISTORICAL_N_CASE,
            'n_primary_cases': len(primary_ids),
            'n_diagnostic_cases': len(case_ids) - len(primary_ids),
        }
    entry = registry.new_experiment_entry(
        experiment_id=exp_id,
        protocol={
            'split_version': SPLIT_VERSION,
            **proto_extra,
            'split': 'acoustic full-field leave-out (Opt-1, 21 cases): '
                     'train6/val1/test10 (+400_0.5_srcR variant in test); '
                     'test f in {100,300,500} Hz, unseen (f,Ma) pairs only',
            'cond_norm': first.get('cond_norm'),
            'norm': first.get('norm'),
            'n_train_cases': len(first['case_set']['train']),
            'n_val_cases': len(first['case_set']['val']),
            'n_test_cases': len(first['case_set']['test']),
            'test_case_ids': case_ids,
            'primary_test_case_ids': primary_ids if fixed_primary else None,
            'points_per_step': first.get('points_per_step'),
            'steps_per_epoch': first.get('steps_per_epoch'),
            'epochs': first.get('epochs'),
            'lr': first.get('lr'),
            'val_eval_every': first.get('val_eval_every'),
            'seeds': list(SEEDS),
            's_per_epoch_mean': summary['s_per_epoch_mean'],
            'hardware': first.get('hardware'),
        },
        seed=list(SEEDS),
        epochs=first.get('epochs'),
        lr=first.get('lr'),
        model=f'{token} ({first.get("model_label")})',
        model_params=first.get('model_params'),
        hardware=first.get('hardware'),
        torch_version=first.get('hardware', {}).get('pytorch_version'),
        tool='exp_acoustic.py',
        metrics={
            'file': os.path.join(model_root, 'summary.json'),
            'n_seeds': n_seeds,
            'test_agg_over_cases': {m: {'mean': over_cases[m]['mean'],
                                        'std': over_cases[m]['std']}
                                    for m in REG_METRICS},
            'test_agg_scope': ('primary9' if fixed_primary else 'all'),
            'n_primary_cases': len(primary_ids),
            'test_per_case_re_mean_std': {
                cid: {'mean': per_case[cid]['re']['mean'],
                      'std': per_case[cid]['re']['std']}
                for cid in case_ids},
        },
    )
    registry_path = getattr(args, 'registry_path', None)
    if not registry_path:
        registry_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(registry.__file__)))), 'results', 'registry.json')
    n_reg = _locked_append_registry(entry, registry_path=registry_path)
    print(f'[registry] upsert {exp_id} -> {registry_path} '
          f'(total {n_reg} entries)', flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', choices=MODEL_TOKENS,
                    help='模型 token（train/eval/aggregate 模式均需要）')
    ap.add_argument('--seeds', default=None,
                    help='explicit seed set, e.g. 0,1,2,3,4; default historical SEEDS=(0,1,2)')
    ap.add_argument('--registry-path', default=None,
                    help='optional isolated registry JSON; default canonical results/registry.json')
    ap.add_argument('--seed', type=int, default=None,
                    help='训练随机种子，需在 {0,1,2}（--aggregate-only 忽略）')
    ap.add_argument('--epochs', type=int, default=None,
                    help=f'epoch 数（默认 {POINT_EPOCHS_DEFAULT}，与已报道的 A2 '
                         '协议一致）')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--val-every', type=int, default=50,
                    help='每隔多少 epoch 在 val case 上评估一次')
    ap.add_argument('--out-dir', default=None,
                    help='模型级输出目录（default <ws>/results/acoustic_protocolA/'
                         '<model>；seed 子目录=seed{n}）')
    ap.add_argument('--device', default='auto', help='auto/cuda/cpu')
    ap.add_argument('--data-cfg', default=None,
                    help='configs/acoustic_cases.json 路径（默认取工作区）')
    ap.add_argument('--points-per-step', type=int, default=POINTS_PER_STEP,
                    help='点模型每 step 采样点数')
    ap.add_argument('--steps-per-epoch', type=int, default=None,
                    help='点模型 steps/epoch（--point-mode=subset 默认 32；fno=6 不可覆盖）')
    ap.add_argument('--point-mode', choices=('fullgrid', 'subset'),
                    default='fullgrid',
                    help="点模型训练数据模式：fullgrid=每 epoch 每个 train case 整场一次"
                         "（推荐，等价原论文整场训练，避免小子集梯度坍缩）；"
                         "subset=每步随机采 --points-per-step 点")
    ap.add_argument('--eval-chunk', type=int, default=EVAL_CHUNK_DEFAULT,
                    help='点模型整场评估单次前向点数上限')
    ap.add_argument('--patience', type=int, default=None,
                    help='val 早停（None=训满 --epochs）')
    ap.add_argument('--aggregate-only', action='store_true',
                    help='只做跨 seed 聚合 + registry 登记，不训练')
    ap.add_argument('--eval-only', default=None, metavar='CKPT',
                    help='加载给定 checkpoint 对 test case 重评一次（不训练）')
    ap.add_argument('--fourier-layout', dest='fourier_layout',
                    choices=('legacy', 'pointwise'),
                    default=FOURIER_LAYOUT_DEFAULT,
                    help='Fourier 编码布局：legacy（默认，已报道协议）或 '
                         'pointwise（需 --strict-fp32；unif_acous 亦需 '
                         '--eval-chunk 0）')
    ap.add_argument('--strict-fp32', dest='strict_fp32', action='store_true',
                    default=False,
                    help='关闭 TF32（matmul+cudnn）并要求 strict fp32；pointwise '
                         '必须开启')
    ap.add_argument('--fixed-source-primary', dest='fixed_source_primary',
                    action='store_true', default=False,
                    help='仅 FNO/UNO legacy 新协议：test 聚合仅 9 个 left-side '
                         '主 case，400_0.5_srcR 作为 diagnostic 单列')

    return ap.parse_args()


def _verify_pointwise_docs(docs, first):
    """pointwise 聚合前一致性校验：逐 doc（含首 doc）layout/tf32、协议同质性、
    行 caseID 恰为 10 例并集且 primary 恰为 9 例已知集。"""
    primary_ids = set(PRIMARY_TEST_CASE_IDS)
    all_ids = primary_ids | set(DIAGNOSTIC_TEST_CASE_IDS)
    for seed, d in docs:
        proto = d['protocol']
        if proto.get('fourier_layout') != 'pointwise':
            raise AssertionError(
                f'pointwise 聚合：seed{seed} protocol.fourier_layout 非 pointwise')
        if proto.get('tf32_matmul_allow') is not False \
                or proto.get('tf32_cudnn_allow') is not False:
            raise AssertionError(
                f'pointwise 聚合：seed{seed} tf32 matmul/cudnn 必须均为 False')
        ids = [r['case_id'] for r in d['test']['rows']]
        if len(set(ids)) != len(ids) or set(ids) != all_ids:
            raise AssertionError(
                f'pointwise 聚合：seed{seed} 行 caseID 必须唯一且恰为 10 例并集')
        doc_primary = set(d['test'].get('primary_case_ids') or ())
        if doc_primary != primary_ids:
            raise AssertionError(
                f'pointwise 聚合：seed{seed} primary_case_ids 必须恰为 9 例已知集')
    def _same(select):
        ref = select(first)
        for _, d in docs[1:]:
            if select(d['protocol']) != ref:
                return False
        return True
    for select in (
            lambda p: p.get('epochs'),
            lambda p: p.get('lr'),
            lambda p: p.get('point_mode'),
            lambda p: p.get('steps_per_epoch'),
            lambda p: p.get('eval_chunk'),
            lambda p: p.get('val_eval_every'),
            lambda p: p.get('p_stats')):
        if not _same(select):
            raise AssertionError(
                'pointwise 聚合：跨 seed 协议不一致（epochs/lr/point_mode/'
                'steps_per_epoch/eval_chunk/val_eval_every/p_stats 必须相同）。')


def _apply_strict_fp32(args):
    """显式 --strict-fp32 时关闭 TF32 matmul/cudnn（训练前）；
    pointwise 另强制 strict fp32 与整场前向要求。legacy 显式 strict 时
    仅关 TF32 后直接返回（不改变旧协议其余行为）。"""
    if args.strict_fp32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if args.fourier_layout != 'pointwise':
        return
    if not args.strict_fp32:
        raise SystemExit(
            '--fourier-layout pointwise 必须配合 --strict-fp32。')
    if args.model == 'unif_acous' and int(args.eval_chunk) != 0:
        raise SystemExit(
            'unif_acous pointwise 仅支持完整整场前向：需 --eval-chunk 0。')


def main():
    global SEEDS
    args = parse_args()
    if args.seeds:
        vals = tuple(sorted(int(x) for x in str(args.seeds).split(',') if x.strip()))
        if not vals or any(v < 0 or v > 4 for v in vals):
            raise SystemExit('--seeds must be a non-empty subset of 0..4')
        SEEDS = vals
    if args.model is None:
        raise SystemExit('--model 必填: ' + '|'.join(MODEL_TOKENS))

    # 训练前（含 eval-only 的前向）先落实 strict fp32 钳制与前置校验
    _apply_strict_fp32(args)

    if args.aggregate_only:
        _data_cfg, _out_dir = resolve_paths(args)
        args.data_cfg, args.out_dir = _data_cfg, _out_dir
        aggregate_and_register(args)
        return

    if args.eval_only is None and args.seed not in SEEDS:
        raise SystemExit(f'--seed 需在 {SEEDS} 中（当前 {args.seed}）')

    data_cfg, out_dir = resolve_paths(args)
    args.data_cfg, args.out_dir = data_cfg, out_dir
    device, _backend = get_device(args.device)
    print(f'[env] workspace={WS}\ndevice={device} torch={torch.__version__} '
          f'cuda={torch.cuda.is_available()} data_cfg={data_cfg}\n'
          f'out-dir={out_dir}', flush=True)
    if device.type == 'cpu' and args.eval_only is None:
        print('[warn] CPU 训练（仅 dry-run 小规模用；正式 A2 需 GPU）',
              flush=True)

    groups, unused = load_case_groups(args.data_cfg)
    print('[data] case roles: '
          + ' '.join(f'{r}={[e["case_id"] for e in groups[r]]}'
                     for r in ('train', 'val', 'test'))
          + f' | unused={unused}', flush=True)
    p_stats = fit_stats_cases(groups['train'])
    print(f'[stats] fit on train cases only: p_mean={float(p_stats["p_mean"]):.6f} '
          f'p_std={float(p_stats["p_std"]):.6f} '
          f'coord=[{p_stats["x_min"]},{p_stats["x_max"]}]^2 '
          f'n_points={int(p_stats["n_points"])}', flush=True)

    if args.eval_only:
        run_eval_only(args, groups, p_stats, device)
        return

    train_one(args, groups, p_stats, device)
    # 若该模型 3 个 seed 全部齐备则自动聚合登记（否则等最后完成的 seed 进程触发）
    aggregate_and_register(args)


if __name__ == '__main__':
    main()
