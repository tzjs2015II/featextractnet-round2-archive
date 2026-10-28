"""exp_thermal.py —— 热场核心实验（几何级 Protocol-A）单 seed 训练 + 跨 seed 聚合登记。

用途
----
实施 T1 热场主表 / T2 同协议基线的监督训练（实验计划.md §5 Track T，几何级
Protocol-A，splits_thermal.json）。一个进程训练一个 (model, seed)，或仅做跨
seed 聚合登记（--aggregate-only）：

  * Protocol-A：135 几何 -> train 100 / val 15 / test 20（configs/splits_thermal.json）
  * 归一化：仅用 train 几何拟合 z-score 统计（fit_stats(X, Y, geom_ids=train)），
    model 输入/输出都在归一化域；评估先反归一化回物理单位再算
    RE = ||y_hat - y||_2 / ||y||_2（per-channel，逐 test 几何，跨几何取 mean±std）
  * 模型选择只用 val 几何：val_selector = 15 个 val 几何 4 通道物理单位 RE 的均值
    （越小越好），test 20 几何只在冻结最优 checkpoint 后评估一次
  * 种子：seed in {0,1,2,3,4}，各独立训练；全部 5 个 seed 的 metrics 落盘后
    自动做跨 seed 聚合（mean±std）并登记 results/registry.json
    （experiment_id = T_T1_<model token>，协议快照齐全）

模型（统一包装为 forward(x)->[B,4,N]）
--------------------------------------
  unif_therm  UnifiedNet(cond_dim=0, branch_mode='unified', compute_both_heads=False)，
              forward(x, task='thermal')[0]，软路由全程（不切 hard）
  feat_extract FeatExtractNet(input_channels=2, output_channels=4, scaling=1.0)
  mlp         CoordMLP 配到 ~1.0M（hidden_dim=512, num_layers=6，见其 docstring 公式）
  pointnet / pointnetpp / dgcnn / pipn  默认构造参数（打印参数量）

输入输出
--------
  in  : --data HeatTransferData.npy、--splits configs/splits_thermal.json
  out : <out-dir>/<model>/seed{n}/checkpoint_best.pth + metrics_seed{n}.json
        （s_per_epoch/wall_time 训练后补写入 metrics protocol）
        <out-dir>/<model>/summary.json（全 seed 聚合，仅全部 seed 齐时写）
        results/registry.json 按 T_T1_<model> upsert（fcntl 文件锁防并发丢写）

运行方式
--------
  cd 返稿意见/返稿修订工作区
  # 单 seed 训练（GPU）
  srun --partition=compute --gres=gpu:1 python3 scripts/exp_thermal.py \
      --model unif_therm --seed 0 --epochs 500 --lr 0.001 --val-every 25 \
      --batch-geoms 4 --out-dir results/thermal_protocolA
  # 跨 seed 聚合 + 登记（CPU 可跑，仅当 seed0..4 metrics 齐备时写 registry）
  python3 scripts/exp_thermal.py --model unif_therm --aggregate-only \
      --out-dir results/thermal_protocolA

约束：登录节点不得跑训练；脚本不改 src/ 现有文件。
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

from src.core import (  # noqa: E402
    append_registry,
    new_experiment_entry,
    registry,
    runner,
    thermal_io,
)
from src.models.baselines import (  # noqa: E402
    CoordMLP,
    DGCNN,
    PipnStyleBaseline,
    PointNetBaseline,
    PointNetPlusPlusBaseline,
)
# WP7 扩展对标：从各自模块直接导入，避免 package __init__ 为所有实验引入
# torch_geometric 的导入开销（见 docs/B_扩展对标设计.md §6）
from src.models.baselines.siren import SIRENNet  # noqa: E402
from src.models.baselines.point_transformer import PointTransformerBaseline  # noqa: E402
from src.models.baselines.meshgraphnet_style import (  # noqa: E402
    MeshGraphNetStyleBaseline,
)
from src.models.baselines.transolver_official.adapter import (  # noqa: E402
    TransolverAdapter,
)
from src.models.featextractnet import FeatExtractNet  # noqa: E402
from src.models.unified import UnifiedNet  # noqa: E402
from src.utils.data_loader import set_seed  # noqa: E402
from src.utils.device import get_device  # noqa: E402

SPLIT_VERSION = 'splits_thermal_v1_protocolA'
FIELD_NAMES = ('u', 'v', 'p', 'T')
SEEDS = (0, 1, 2, 3, 4)
MODEL_CHOICES = ('unif_therm', 'feat_extract', 'mlp', 'pointnet',
                 'pointnetpp', 'dgcnn', 'pipn', 'siren', 'pointtransformer',
                 'meshgraphnet', 'transolver')
REGISTRY_KEYS = ('re', 'mae', 'max_err', 're_bd', 'mae_bd', 'max_bd')

# R2 池化上下文（R2-IMPL-01B）：池化上下文实现版本号写进 protocol 快照
POOL_CONTEXT_CHOICES = ('legacy_index_bins', 'symmetric')
IMPLEMENTATION_VERSION = 'R2-IMPL-01B'
LEGACY_OUT_NAME = 'thermal_protocolA'
SYMMETRIC_OUT_NAME = 'round2_thermal_symmetric'
# 校正池化仅对 FeatExtractNet（feat_extract）有实现；其余模型拒绝 symmetric
POOL_CONTEXT_SUPPORTED = ('feat_extract',)
# 聚合时必须跨 seed 一致的协议字段（数值/结构不一致即拒绝聚合）
AGG_CONSISTENCY_KEYS = ('epochs', 'lr', 'n_pts', 'batch_geoms', 'val_eval_every',
                        'cuda_matmul_allow_tf32', 'cudnn_allow_tf32')


class UnifiedThermalHead(nn.Module):
    """把 UnifiedNet 包装成 train_supervised 所需 forward(x)->tensor。

    热场任务：cond=None、task='thermal'，取 forward(x)[0] 输出 [B,4,N]；
    hard=False 全程软路由（UnifiedNet 未实现 set_route_hard，runner 不会切硬路由）。
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        out, _diag = self.net(x, cond=None, task='thermal', hard=False)
        return out


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', choices=MODEL_CHOICES,
                    help='模型 token（train/aggregate 模式均需要）')
    ap.add_argument('--seed', type=int, default=None,
                    help='训练随机种子（--aggregate-only 时忽略）')
    ap.add_argument('--epochs', type=int, default=500)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--val-every', type=int, default=25,
                    help='每隔多少 epoch 在 val 上评估一次（None=只在末 epoch）')
    ap.add_argument('--batch-geoms', type=int, default=4,
                    help='每个 batch 的几何数（几何级采样单元）')
    ap.add_argument('--n-pts', type=int, default=None,
                    help='每几何随机下采样点数；None=全量 5000 点')
    ap.add_argument('--data', default=None, help='HeatTransferData.npy 路径')
    ap.add_argument('--splits', default=None,
                    help='configs/splits_thermal.json 路径')
    ap.add_argument('--out-dir', default=None,
                    help='输出根目录（默认 <ws>/results/thermal_protocolA；'
                         '模型目录 = <out-dir>/<model>，seed 目录 = seed{n}）')
    ap.add_argument('--device', default='auto',
                    help='auto/cuda/cpu（仅训练用；自动选择）')
    ap.add_argument('--eval-chunk', type=int, default=None,
                    help='评估时单次前向最大点数（None=整场 5000 一次）')
    ap.add_argument('--patience', type=int, default=None,
                    help='val 早停（None=训满 --epochs）')
    ap.add_argument('--aggregate-only', action='store_true',
                    help='只做跨 seed 聚合 + registry 登记，不训练')
    ap.add_argument('--strict-fp32', action='store_true', default=False,
                    help='显式关闭 TF32（cuda.matmul/cudnn allow_tf32=False），'
                         'R2 symmetric 新跑分强制要求；legacy 默认不变')
    ap.add_argument('--pool-context', choices=POOL_CONTEXT_CHOICES,
                    default='legacy_index_bins',
                    help='feat_extract 的池化上下文：legacy_index_bins=历史 K=min(32,N) '
                         '索引分箱（默认，历史结果口径）；symmetric=全支持域对称 K=1 '
                         '（仅 feat_extract，需配 --out-dir 含 round2/r2 且 --eval-chunk None）')
    ap.add_argument('--registry-path', default=None,
                    help='optional isolated registry JSON path; default canonical results/registry.json')
    return ap.parse_args()


def _norm_out_path(path):
    """out-dir 归一化（abspath + realpath + normcase）用于比较是否指向旧目录。"""
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def check_pool_context_args(args):
    """在训练/聚合前校验 --pool-context 相关约束（任何训练/聚合前调用）。"""
    if getattr(args, 'model', None) == 'transolver' and not bool(
            getattr(args, 'strict_fp32', False)):
        raise SystemExit(
            '--model transolver 要求显式 --strict-fp32（全 FP32：关闭 '
            'cuda.matmul/cudnn TF32），禁止默认 TF32 混入正式结果。')
    sym = args.pool_context == 'symmetric'
    if sym and not bool(getattr(args, 'strict_fp32', False)):
        raise SystemExit(
            '--pool-context symmetric 要求显式 --strict-fp32（全 FP32：关闭 '
            'cuda.matmul/cudnn TF32），防止默认 TF32 混入 R2 新结果。')
    if sym and args.model not in POOL_CONTEXT_SUPPORTED:
        raise SystemExit(
            f'--pool-context symmetric 仅支持模型 {POOL_CONTEXT_SUPPORTED} '
            f'（当前 --model={args.model}）；其余模型只有 legacy_index_bins 实现。')
    if not sym:
        return
    if args.eval_chunk is not None:
        raise SystemExit(
            '--pool-context symmetric 要求 --eval-chunk None（整场全支持上下文），'
            f'当前 --eval-chunk={args.eval_chunk}')
    od = _norm_out_path(args.out_dir)
    base = os.path.basename(od)
    if 'round2' not in base and 'r2' not in base:
        raise SystemExit(
            f'--pool-context symmetric 的 --out-dir 目录名须含 round2 或 r2 '
            f'（当前 {args.out_dir}）')
    old_roots = {
        _norm_out_path(os.path.join(WS, 'results', LEGACY_OUT_NAME)),
        _norm_out_path(os.path.join(WS, 'results', 'thermal_protocolA')),
    }
    if od in old_roots:
        raise SystemExit(
            f'--pool-context symmetric 不得使用历史默认根 {LEGACY_OUT_NAME} '
            f'（当前 {args.out_dir}）')
    for old in old_roots:
        if od == old or od.startswith(old + os.sep):
            raise SystemExit(
                f'--pool-context symmetric 不得写入历史 legacy 结果目录 {old} '
                f'（当前 {args.out_dir}）')
    # symmetric 不读取历史 legacy 结果作为输入（已有 legacy metrics 目录下拒绝）
    model_root = os.path.join(args.out_dir, args.model)
    if os.path.isdir(model_root):
        for s in SEEDS:
            sd = os.path.join(model_root, f'seed{s}')
            legacy_metrics = os.path.join(sd, f'metrics_seed{s}.json')
            if os.path.exists(legacy_metrics):
                with open(legacy_metrics, 'r', encoding='utf-8') as fh:
                    doc = json.load(fh)
                ctx = doc.get('protocol', {}).get('pool_context',
                                                  'legacy_index_bins')
                if ctx == 'legacy_index_bins':
                    raise SystemExit(
                        f'--pool-context symmetric 的输出目录下存在 legacy 结果 '
                        f'{sd}（pool_context=legacy_index_bins），拒绝混入/覆盖；'
                        '请改用 r2/round2 专用目录')
    if args.aggregate_only:
        return


def resolve_paths(args):
    ws = WS
    data = args.data
    if data is None:
        repo_root = os.path.dirname(os.path.dirname(ws))
        for cand in (os.path.join(ws, 'data', 'thermal', 'HeatTransferData.npy'),
                     os.path.join(repo_root, 'HeatTransferData.npy')):
            if os.path.exists(cand):
                data = cand
                break
    if data is None or not os.path.exists(data):
        raise FileNotFoundError('HeatTransferData.npy 未找到（--data 指定）')
    splits = args.splits or os.path.join(ws, 'configs', 'splits_thermal.json')
    if not os.path.exists(splits):
        raise FileNotFoundError(f'splits 不存在: {splits}')
    if args.out_dir:
        out_dir = args.out_dir
    elif getattr(args, 'pool_context', 'legacy_index_bins') == 'symmetric':
        out_dir = os.path.join(ws, 'results', SYMMETRIC_OUT_NAME)
    else:
        out_dir = os.path.join(ws, 'results', LEGACY_OUT_NAME)
    os.makedirs(out_dir, exist_ok=True)
    return data, splits, out_dir


def build_model(token, pool_context='legacy_index_bins'):
    """按 token 构造模型，返回 (model, 描述 label)。forward(x)->[B,4,N]。

    pool_context 仅对 feat_extract 生效（作为 FeatExtractNet 的 pool_mode）；
    其余模型只支持 legacy_index_bins，symmetric 在此拒绝。
    """
    if pool_context not in POOL_CONTEXT_CHOICES:
        raise ValueError(f'unknown pool_context: {pool_context!r}')
    if pool_context == 'symmetric' and token not in POOL_CONTEXT_SUPPORTED:
        raise ValueError(f'pool_context=symmetric 仅支持 {POOL_CONTEXT_SUPPORTED} '
                         f'(token={token!r})')
    if token == 'unif_therm':
        net = UnifiedNet(cond_dim=0, branch_mode='unified',
                         compute_both_heads=False)
        label = ('UnifiedNet(branch=unified, soft, task=thermal, '
                 'compute_both_heads=False)')
        return UnifiedThermalHead(net), label
    if token == 'feat_extract':
        return (FeatExtractNet(input_channels=2, output_channels=4,
                               scaling=1.0, pool_mode=pool_context),
                f'FeatExtractNet(scale=1.0, pool_mode={pool_context})')
    if token == 'mlp':
        # CoordMLP 参数量 ~1.0M：见 coordinate_mlp.py docstring 公式
        # Cin*h + (L-2)*h^2 + h*Cout（h=512、L=6 -> ~1.05M，FeatExtractNet 的 ~91%）
        return (CoordMLP(input_channels=2, output_channels=4,
                         hidden_dim=512, num_layers=6),
                'CoordMLP(hidden=512, layers=6) ~1.05M')
    if token == 'pointnet':
        return (PointNetBaseline(input_channels=2, output_channels=4),
                'PointNetBaseline(default)')
    if token == 'pointnetpp':
        return (PointNetPlusPlusBaseline(input_channels=2, output_channels=4),
                'PointNetPlusPlusBaseline(default n_points=(512,128), k=16)')
    if token == 'dgcnn':
        return (DGCNN(input_channels=2, output_channels=4),
                'DGCNN(default k=16, width=64, n_blocks=2)')
    if token == 'pipn':
        return (PipnStyleBaseline(input_channels=2, output_channels=4,
                                  scaling=1.0),
                'PipnStyleBaseline(scale=1.0)')
    if token == 'siren':
        # 参数量 1,054,212，与 CoordMLP(512,6) 完全相同 -> 同容量、只差
        # 激活与初始化的受控对照（docs/B_扩展对标设计.md §6）
        return (SIRENNet(input_channels=2, output_channels=4,
                         hidden_dim=512, num_layers=6, omega_0=30.0),
                'SIRENNet(hidden=512, layers=6, omega_0=30) 1,054,212 params '
                '(= CoordMLP capacity)')
    if token == 'pointtransformer':
        # 官方 torch_geometric.nn.PointTransformerConv 堆叠；容量对齐到 928,644
        # 参数（相对 1.05M 行 −11.9%），结论须带容量限定
        return (PointTransformerBaseline(input_channels=2, output_channels=4,
                                         hidden_dim=320, num_layers=3, k=16),
                'PointTransformerBaseline(official PointTransformerConv, '
                'hidden=320, layers=3, k=16) 928,644 params')
    if token == 'meshgraphnet':
        # torch_geometric 无官方 MeshGraphNet 类，本行是自行组装的
        # encode-process-decode GNN，命名与结论均须带 "style (assembled)" 限定
        return (MeshGraphNetStyleBaseline(input_channels=2, output_channels=4,
                                          hidden_dim=128, num_layers=8, k=16),
                'MeshGraphNet-style encode-process-decode GNN (assembled, '
                'hidden=128, layers=8, k=16) 976,516 params')
    if token == 'transolver':
        # Official thuml/Transolver irregular-mesh model, isolated adapter.
        # Coordinates only; no truth/geometry ID; four outputs.
        model = TransolverAdapter()
        return model, model.model_label()
    raise ValueError(f'unknown model token: {token}')


# ---------------------------------------------------------------------------
# val_selector / test_eval_fn 闭包
# ---------------------------------------------------------------------------
def _eval_mean_re(model, geom_ids, X, Y, stats, device, chunk_points):
    """15 个 val 几何 4 通道物理单位 RE 的平均（model 选择标量，越小越好）。"""
    ev = thermal_io.eval_geometry_wise(
        lambda x: model(x), geom_ids, X, Y, stats, device,
        chunk_points=chunk_points)
    return float(np.mean(list(ev['agg']['re'].values())))


def _eval_test(model, geom_ids, X, Y, stats, device, chunk_points):
    """test 20 几何整场评估 -> dict（含 agg/rows/边界 MAE），冻结最优权重后调一次。"""
    ev = thermal_io.eval_geometry_wise(
        lambda x: model(x), geom_ids, X, Y, stats, device,
        chunk_points=chunk_points)
    return {'n_geom': ev['n_geom'], 'agg': ev['agg'], 'rows': ev['rows']}


# ---------------------------------------------------------------------------
# 单 seed 训练
# ---------------------------------------------------------------------------
def _guard_symmetric_seed_dir(args, seed_dir):
    """symmetric 训练前拒绝已存在的 seed checkpoint/metrics（禁止静默 resume/覆盖）。
    必须在 os.makedirs 之前调用。"""
    if args.pool_context != 'symmetric':
        return
    for name in (f'metrics_seed{int(args.seed)}.json', 'checkpoint_best.pth',
                 'checkpoint_last.pth'):
        fp = os.path.join(seed_dir, name)
        if os.path.exists(fp):
            raise SystemExit(
                f'[refuse] symmetric 训练目标已存在 {fp}；不静默 resume/覆盖，'
                '请换 --out-dir/--seed 或先人工确认旧产物')


def train_one(args, X, Y, stats, train_ids, val_ids, test_ids, device):
    token = args.model
    pool_context = args.pool_context
    model_root = os.path.join(args.out_dir, token)
    seed_dir = os.path.join(model_root, f'seed{args.seed}')
    _guard_symmetric_seed_dir(args, seed_dir)
    os.makedirs(seed_dir, exist_ok=True)

    set_seed(args.seed)
    model, model_label = build_model(token, pool_context=pool_context)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[model] token={token} | {model_label} | params={n_params:,}',
          flush=True)

    provider = thermal_io.make_thermal_train_provider(
        X, Y, stats, train_ids, batch_size=args.batch_geoms,
        n_pts=args.n_pts, device=device, seed=args.seed)

    protocol_extra = {
        'split_version': SPLIT_VERSION,
        'dataset': os.path.abspath(args.data),
        'splits_config': os.path.abspath(args.splits),
        'split_seed': 123,
        'norm': 'train-only z-score (fit on 100 Protocol-A train geoms)',
        'n_train_geom': len(train_ids),
        'n_val_geom': len(val_ids),
        'n_test_geom': len(test_ids),
        'batch_geoms': args.batch_geoms,
        'n_pts': args.n_pts,
        'model_label': model_label,
        'pool_context': pool_context,
        'implementation_version': IMPLEMENTATION_VERSION,
        'cuda_matmul_allow_tf32': bool(torch.backends.cuda.matmul.allow_tf32),
        'cudnn_allow_tf32': bool(torch.backends.cudnn.allow_tf32),
        # 归一化统计（train 几何拟合）随 metrics 存档，序列化为原生 list
        'x_mean': np.asarray(stats['x_mean']).tolist(),
        'x_std': np.asarray(stats['x_std']).tolist(),
        'y_mean': np.asarray(stats['y_mean']).tolist(),
        'y_std': np.asarray(stats['y_std']).tolist(),
    }

    t_start = time.monotonic()
    summary = runner.train_supervised(
        model, task='thermal', data_provider=provider,
        epochs=args.epochs, lr=args.lr, seed=args.seed, device=device,
        val_eval_every=args.val_every,
        val_selector=lambda m: _eval_mean_re(
            m, val_ids, X, Y, stats, device, args.eval_chunk),
        test_eval_fn=lambda m: _eval_test(
            m, test_ids, X, Y, stats, device, args.eval_chunk),
        out_dir=seed_dir, patience=args.patience,
        protocol_extra=protocol_extra)
    wall_s = time.monotonic() - t_start
    n_ep = int(summary['n_epochs'])
    s_per_epoch = wall_s / n_ep if n_ep else float('nan')

    # 补写计时到 metrics protocol（runner 写入后由本脚本追加）
    mpath = os.path.join(seed_dir, f'metrics_seed{int(args.seed)}.json')
    with open(mpath, 'r', encoding='utf-8') as fh:
        metrics = json.load(fh)
    metrics['protocol']['s_per_epoch'] = round(s_per_epoch, 6)
    metrics['protocol']['wall_time_s'] = round(wall_s, 3)
    with open(mpath, 'w', encoding='utf-8') as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)
        fh.write('\n')

    agg = metrics['test']['agg']['re']
    print(f'[train] seed={args.seed} done | epochs={n_ep} | '
          f'best_epoch={summary["best_epoch"]} | '
          f'best_val={summary["best_val_score"]:.6f} | '
          f's_per_epoch={s_per_epoch:.3f}s | test RE(u/v/p/T)='
          + ' '.join(f'{f}:{agg[f]:.4f}' for f in FIELD_NAMES), flush=True)
    return model_root, n_params


# ---------------------------------------------------------------------------
# 跨 seed 聚合 + registry 登记
# ---------------------------------------------------------------------------
def _load_all_seed_metrics(model_root):
    """读 model_root/seed{n}/metrics_seed{n}.json，返回按 seed 升序的 [(seed, doc)]。"""
    out = []
    for s in SEEDS:
        fp = os.path.join(model_root, f'seed{s}', f'metrics_seed{s}.json')
        if os.path.exists(fp):
            with open(fp, 'r', encoding='utf-8') as fh:
                out.append((s, json.load(fh)))
    return out


def _validate_metrics_context(docs, pool_context):
    """聚合前校验：每个 metrics 的 pool_context 必须等于请求值（缺失视为 legacy），
    且 epochs/lr/n_pts/batch_geoms/val_eval_every 跨 seed 一致；不一致即拒绝。"""
    for s, doc in docs:
        prot = doc.get('protocol') or {}
        ctx = prot.get('pool_context', 'legacy_index_bins')
        if ctx != pool_context:
            raise SystemExit(
                f'[refuse aggregate] seed{s} metrics 的 pool_context={ctx!r} '
                f'与请求 {pool_context!r} 不一致（{ctx} 结果不得混入 {pool_context} 聚合）')
    ref = docs[0][1].get('protocol') or {}
    for key in AGG_CONSISTENCY_KEYS:
        ref_val = ref.get(key, 'MISSING')
        for s, doc in docs[1:]:
            cur = doc.get('protocol') or {}
            cur_val = cur.get(key, 'MISSING')
            if cur_val != ref_val:
                raise SystemExit(
                    f'[refuse aggregate] 协议字段 {key} 跨 seed 不一致：'
                    f'seed{docs[0][0]}={ref_val!r} vs seed{s}={cur_val!r}')


def _aggregate_field_values(docs, dotted_key):
    """跨 seed 对 test.agg.<dotted_key> 的 {field: value} 做 mean±std（None 剔除）。"""
    vals_per_field = {f: [] for f in FIELD_NAMES}
    for _, doc in docs:
        cur = doc['test']['agg']
        for part in dotted_key.split('.'):
            cur = cur[part]
        for f in FIELD_NAMES:
            v = cur.get(f)
            if v is not None and np.isfinite(v):
                vals_per_field[f].append(float(v))
    mean, std, n_per = {}, {}, {}
    for f in FIELD_NAMES:
        arr = np.asarray(vals_per_field[f], dtype=np.float64)
        if arr.size:
            mean[f] = float(arr.mean())
            std[f] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        else:
            mean[f], std[f] = None, None
        n_per[f] = int(arr.size)
    return {'mean': mean, 'std': std, 'n_seeds': len(docs), 'n_per_field': n_per,
            'values': vals_per_field}


def _locked_append_registry(entry, registry_path):
    """fcntl 文件锁包 append_registry：并发（多模型任务同时收尾）下防丢写。"""
    lock_path = registry_path + '.lock'
    with open(lock_path, 'w', encoding='utf-8') as lfh:
        fcntl.flock(lfh, fcntl.LOCK_EX)
        try:
            return append_registry(entry, registry_path=registry_path)
        finally:
            fcntl.flock(lfh, fcntl.LOCK_UN)


def aggregate_and_register(args):
    """跨 seed 聚合（mean±std）、打印、写 summary.json 并登记 registry。

    仅当 model_root 下 seed0..4 的 metrics 全部齐备时写 registry；否则打印
    提示并跳过登记（例如干跑 epochs 少、seed 数不足的场景）。
    """
    token = args.model
    pool_context = args.pool_context
    model_root = os.path.join(args.out_dir, token)
    docs = _load_all_seed_metrics(model_root)
    if not docs:
        raise FileNotFoundError(f'{model_root} 下无 metrics_seed*.json')
    _validate_metrics_context(docs, pool_context)

    agg_all = {k: _aggregate_field_values(docs, k) for k in REGISTRY_KEYS}
    n_seeds = len(docs)
    print(f'\n=== aggregate model={token} ({model_root}) | '
          f'seeds={[s for s, _ in docs]} | n={n_seeds} ===', flush=True)

    re_a = agg_all['re']
    print('test.agg.re per field (mean ± std across seeds):')
    for f in FIELD_NAMES:
        if re_a['mean'][f] is None:
            print(f'  RE_{f}: --')
        else:
            print(f'  RE_{f}: {re_a["mean"][f]:.4f} ± {re_a["std"][f]:.4f}')
    latex_entry = {'n_seeds': n_seeds, 'mean': re_a['mean'], 'std': re_a['std'],
                   'values': re_a['values']}
    print('  latex:', runner.latex_row(token, latex_entry, fields=FIELD_NAMES))
    for f in FIELD_NAMES:
        if re_a['mean'][f] is None:
            continue
        mae = agg_all['mae']['mean'][f]
        mx = agg_all['max_err']['mean'][f]
        bd_mae = agg_all['mae_bd']['mean'][f]
        print(f'  {f}: RE={re_a["mean"][f]:.4f}±{re_a["std"][f]:.4f} | '
              f'MAE={mae:.3e} | max={mx:.3e} | boundary MAE={bd_mae:.3e}')

    # summary.json（聚合结果 + 各 seed 文件引用）
    first = docs[0][1]['protocol']
    experiment_id = (f'T_T1_{token}' if pool_context == 'legacy_index_bins'
                     else f'T_T1_{token}_R2_symmetric')
    summary = {
        'experiment_id': experiment_id,
        'model': token,
        'pool_context': pool_context,
        'implementation_version': IMPLEMENTATION_VERSION,
        'cuda_matmul_allow_tf32': first.get('cuda_matmul_allow_tf32'),
        'cudnn_allow_tf32': first.get('cudnn_allow_tf32'),
        'model_label': first.get('model_label'),
        'model_params': first.get('model_params'),
        'split_version': first.get('split_version'),
        'n_seeds': n_seeds,
        'seeds': [s for s, _ in docs],
        'epochs': first.get('epochs'),
        'lr': first.get('lr'),
        'agg': {k: {'mean': v['mean'], 'std': v['std'],
                    'n_per_field': v['n_per_field']}
                for k, v in agg_all.items()},
        'seed_files': [os.path.join(model_root, f'seed{s}',
                                    f'metrics_seed{s}.json') for s, _ in docs],
    }
    sp_sum = [d['protocol'].get('s_per_epoch') for _, d in docs
              if d['protocol'].get('s_per_epoch') is not None]
    if sp_sum:
        summary['s_per_epoch_mean'] = float(np.mean(sp_sum))
    with open(os.path.join(model_root, 'summary.json'), 'w',
              encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
        fh.write('\n')

    # 登记门槛按**协议实际要求**而非主表的 5 seeds：
    #   * 主表（T_T1_feat_extract 等）按协议为 5 seeds；
    #   * 基线/消融/扩展对标按协议为 ≥3 seeds（AGENTS §二.3、docs/B_扩展对标设计.md §2）。
    # 旧实现写成 `n_seeds < len(SEEDS)`（即 <5 就跳过），导致 3-seed 的扩展对标（SIREN/
    # PointTransformer/MeshGraphNet-style）根本进不了 registry，违反 AGENTS §3.4
    # "每个实验登记入 registry"；同时 entry 里硬写 `seed=list(SEEDS)` 会把 3 seeds 记成 5，
    # 属"协议记录 ≠ 实跑"（同 P0-1/P0-4 一类问题）。
    MIN_SEEDS_REGISTRY = 3
    actual_seeds = [s for s, _ in docs]
    if n_seeds < MIN_SEEDS_REGISTRY:
        print(f'[skip registry] 只有 {n_seeds} 个 seed 的 metrics（<{MIN_SEEDS_REGISTRY}），'
              '不做 registry 登记。')
        return
    if n_seeds < len(SEEDS):
        print(f'[registry] 本次为 {n_seeds} seeds（主表口径 {len(SEEDS)}）；按协议基线/消融'
              f' ≥{MIN_SEEDS_REGISTRY} seeds 即可登记，entry 记录**实际** seeds={actual_seeds}。')

    # registry 登记（协议快照齐全；锁防多任务并发丢写）
    protocol = {
        'split_version': SPLIT_VERSION,
        'split': 'Protocol-A train100/val15/test20',
        'split_seed': 123,
        'norm': first.get('norm'),
        'n_train_geom': first.get('n_train_geom'),
        'n_val_geom': first.get('n_val_geom'),
        'n_test_geom': first.get('n_test_geom'),
        'epochs': first.get('epochs'),
        'lr': first.get('lr'),
        'val_eval_every': first.get('val_eval_every'),
        'batch_geoms': first.get('batch_geoms'),
        'n_pts': first.get('n_pts'),
        'route_mode': first.get('route_mode'),
        'route_hard_switched': first.get('route_hard_switched'),
        'hard_after_requested': first.get('hard_after_requested'),
        # 记录**实际**跑到的 seeds，而不是主表口径的 seeds（P2-4 类口径问题）
        'seeds': actual_seeds,
        'seeds_requested': list(SEEDS),
        'n_seeds': n_seeds,
        'pool_context': pool_context,
        'implementation_version': IMPLEMENTATION_VERSION,
        'cuda_matmul_allow_tf32': first.get('cuda_matmul_allow_tf32'),
        'cudnn_allow_tf32': first.get('cudnn_allow_tf32'),
        's_per_epoch_mean': summary.get('s_per_epoch_mean'),
        'hardware': first.get('hardware'),
    }
    entry = new_experiment_entry(
        experiment_id=experiment_id,
        protocol=protocol,
        seed=actual_seeds,
        epochs=first.get('epochs'),
        lr=first.get('lr'),
        model=f'{token} ({first.get("model_label")})',
        model_params=first.get('model_params'),
        hardware=first.get('hardware'),
        torch_version=first.get('hardware', {}).get('pytorch_version'),
        tool='exp_thermal.py',
        metrics={
            'file': os.path.join(model_root, 'summary.json'),
            'n_seeds': n_seeds,
            'test_agg_re_mean': re_a['mean'],
            'test_agg_re_std': re_a['std'],
            'test_agg_mae_mean': agg_all['mae']['mean'],
            'test_agg_max_err_mean': agg_all['max_err']['mean'],
            'test_agg_mae_bd_mean': agg_all['mae_bd']['mean'],
        },
    )
    # registry 路径：默认 canonical；正式剩余补齐任务显式写隔离 snapshot。
    registry_path = getattr(args, 'registry_path', None)
    if not registry_path:
        registry_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(registry.__file__)))), 'results', 'registry.json')
    os.makedirs(os.path.dirname(os.path.abspath(registry_path)), exist_ok=True)
    n_reg = _locked_append_registry(entry, registry_path=registry_path)
    print(f'[registry] upsert {experiment_id} -> {registry_path} '
          f'(total {n_reg} entries)', flush=True)


def main():
    args = parse_args()
    if args.strict_fp32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    data, splits, out_dir = resolve_paths(args)
    args.data, args.splits, args.out_dir = data, splits, out_dir
    if args.model is None:
        raise SystemExit('--model 必填: ' + '|'.join(MODEL_CHOICES))

    check_pool_context_args(args)

    if args.aggregate_only:
        aggregate_and_register(args)
        return

    if args.seed not in SEEDS:
        raise SystemExit(f'--seed 需在 {SEEDS} 中（当前 {args.seed}）')
    if args.epochs < 1:
        raise SystemExit('--epochs >= 1')

    device, _backend = get_device(args.device)
    print(f'[env] workspace={WS}\ndevice={device} torch={torch.__version__} '
          f'cuda={torch.cuda.is_available()} data={data}', flush=True)
    if device.type == 'cpu':
        print('[warn] CPU 训练（仅 dry-run 小规模用；正式 Protocol-A 需 GPU）',
              flush=True)

    with open(splits, 'r', encoding='utf-8') as fh:
        splits_cfg = json.load(fh)
    pa = splits_cfg['protocol_A']
    train_ids, val_ids, test_ids = pa['train'], pa['val'], pa['test']
    assert [len(train_ids), len(val_ids), len(test_ids)] == [100, 15, 20]
    assert splits_cfg['split_seed'] == 123 and splits_cfg['n_geometries'] == 135

    X, Y = thermal_io.load_thermal(data)
    stats = thermal_io.fit_stats(X, Y, geom_ids=train_ids)
    print(f'[data] X={tuple(X.shape)} Y={tuple(Y.shape)} | '
          f'train/val/test = {len(train_ids)}/{len(val_ids)}/{len(test_ids)} | '
          'norm fit on train-only geoms', flush=True)

    train_one(args, X, Y, stats, train_ids, val_ids, test_ids, device)
    # 若该模型 5 个 seed 全部齐备则自动聚合登记（否则跳过，等最后完成的 seed 进程触发）
    aggregate_and_register(args)


if __name__ == '__main__':
    main()
