"""exp_ablation_thermal.py —— 热场组件消融（真消融表 T3）Protocol-A 训练。

用途
----
在几何级 Protocol-A（splits_thermal.json，train 100 / val 15 / test 20）上训练
FeatExtractNet 的组件消融变体（src/models/featextractnet_variants.py），对应
实验计划.md §5 Track T3（回复口径 A#3 / B R3：全部消融为独立真实训练，
无 projected/♯ 推算行）。训练/评估协议与 scripts/exp_thermal.py 完全一致，
仅模型集合、seed 数与 registry id 不同：

  * 模型（--variant 构造 FeatExtractNetVariants，forward(x)->[B,4,N]）
      fen_full        tri 池化 + CGF + head shortcut（= FeatExtractNet 结构，
                      参数量对照 1,149,512）
      fen_nocgf       去掉 CGF（门控 + compress），局部特征与 global 直接 concat
      fen_poolmax     三统计池化 -> 仅 max（去掉 avg/var 统计通道）
      fen_noshortcut  去掉 head residual shortcut
      fen_single128   单尺度 128 三层均匀 + CGF（历史 CGF† 421,960 参数移植）
  * 协议：train-only z-score 归一化（fit_stats 仅用 100 个 train 几何）；
    val 15 几何 4 通道物理单位 RE 均值选优（越小越好），test 20 几何只在
    冻结最优 checkpoint 后评估一次；物理单位 RE/MAE/max + 边界 MAE 口径
    （thermal_io.eval_geometry_wise）。
  * 种子：seed in {0,1,2}（消融 >=3 seeds，AGENTS §二.3），各独立训练；
    3 个 seed 的 metrics 齐备后自动聚合 mean±std 并登记
    results/registry.json（experiment_id = T3_fen_<variant>）。

输入输出
--------
  in  : --data HeatTransferData.npy、--splits configs/splits_thermal.json
  out : <out-dir>/<variant>/seed{n}/checkpoint_best.pth + metrics_seed{n}.json
        <out-dir>/<variant>/summary.json（3 seed 齐时写，含聚合）
        results/registry.json 按 T3_fen_<variant> upsert（fcntl 文件锁）

运行方式
--------
  cd 返稿意见/返稿修订工作区
  # 单 seed 训练（GPU）
  srun --partition=compute --gres=gpu:1 python3 scripts/exp_ablation_thermal.py \
      --variant fen_nocgf --seed 0 --epochs 500 --lr 0.001 --val-every 25 \
      --batch-geoms 4 --out-dir results/ablation_thermal_protocolA
  # 跨 seed 聚合 + 登记（CPU 可跑，3 个 seed 齐备时写 registry）
  python3 scripts/exp_ablation_thermal.py --variant fen_nocgf --aggregate-only \
      --out-dir results/ablation_thermal_protocolA

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
from src.models.featextractnet import FeatExtractNet  # noqa: E402
from src.models.featextractnet_variants import FeatExtractNetVariants  # noqa: E402
from src.utils.data_loader import set_seed  # noqa: E402
from src.utils.device import get_device  # noqa: E402

SPLIT_VERSION = 'splits_thermal_v1_protocolA'
FIELD_NAMES = ('u', 'v', 'p', 'T')
SEEDS = (0, 1, 2)  # 消融 >=3 seeds（AGENTS §二.3）
VARIANT_CHOICES = ('fen_full', 'fen_nocgf', 'fen_poolmax',
                   'fen_noshortcut', 'fen_single128',
                   'fen_nocgf_pm', 'fen_cgfonly_pm')
REGISTRY_KEYS = ('re', 'mae', 'max_err', 're_bd', 'mae_bd', 'max_bd')

# R2 池化上下文（R2-IMPL-01B）：corrected symmetric 支持全部 7 个变体
POOL_CONTEXT_CHOICES = ('legacy_index_bins', 'symmetric')
IMPLEMENTATION_VERSION = 'R2-IMPL-01B'
LEGACY_OUT_NAME = 'ablation_thermal_protocolA'
SYMMETRIC_OUT_NAME = 'round2_ablation_symmetric'
# R2 初始化对齐（R2-IMPL-01E）：main_default 复刻 FeatExtractNet 主模型默认初始化
INIT_MODE_CHOICES = ('legacy_kaiming', 'main_default')
MAININIT_OUT_NAME = 'round2_ablation_maininit'
AGG_CONSISTENCY_KEYS = ('epochs', 'lr', 'n_pts', 'batch_geoms', 'val_eval_every',
                        'cuda_matmul_allow_tf32', 'cudnn_allow_tf32',
                        'init_mode')

# 参数匹配臂的 scaling 由离线搜索确定（目标 = Full 的 1,149,512 参数），
# 详见 docs/T3b_参数匹配消融设计.md。搜索判据：在 scaling 网格上取
# |params - 1,149,512| / 1,149,512 最小者。
PM_FULL_PARAMS = 1149512
PM_SCALING = {'fen_nocgf_pm': 1.175,     # -> 1,144,209 参数（rel 0.46%）
              'fen_cgfonly_pm': 1.660}   # -> 1,152,970 参数（rel 0.30%）


def variant_config(token):
    """--variant token -> (FeatExtractNetVariants 构造 kwargs, label)。"""
    kw = {'input_channels': 2, 'output_channels': 4, 'scaling': 1.0}
    if token == 'fen_full':
        return kw, 'fen_full = FeatExtractNet 结构（tri+CGF+shortcut，参数量对照 1,149,512）'
    if token == 'fen_nocgf':
        kw.update(use_cgf=False)
        return kw, 'fen_nocgf = 去 CGF（门控+compress），局部特征直接 concat global'
    if token == 'fen_poolmax':
        kw.update(pool_mode='max')
        return kw, 'fen_poolmax = 三统计池化 -> 仅 max（去 avg/var 通道）'
    if token == 'fen_noshortcut':
        kw.update(use_shortcut=False)
        return kw, 'fen_noshortcut = 去 head residual shortcut'
    if token == 'fen_single128':
        kw.update(mode='single128')
        return kw, 'fen_single128 = CGF-only 单尺度 128x3（历史 CGF† 421,960 参数移植）'
    if token == 'fen_nocgf_pm':
        # MLFE-only 与 Full 参数匹配（回应 A#3 "control parameter counts"）
        kw.update(use_cgf=False, scaling=PM_SCALING['fen_nocgf_pm'])
        return kw, (f'fen_nocgf_pm = MLFE-only 参数匹配（scaling='
                    f'{PM_SCALING["fen_nocgf_pm"]:.3f}，1,144,209 参数 vs Full '
                    f'{PM_FULL_PARAMS:,}）')
    if token == 'fen_cgfonly_pm':
        # CGF-only 与 Full 参数匹配（同上）
        kw.update(mode='single128', scaling=PM_SCALING['fen_cgfonly_pm'])
        return kw, (f'fen_cgfonly_pm = CGF-only 参数匹配（scaling='
                    f'{PM_SCALING["fen_cgfonly_pm"]:.3f}，1,152,970 参数 vs Full '
                    f'{PM_FULL_PARAMS:,}）')
    raise ValueError(f'unknown variant token: {token}')


def build_variant(token, pool_context='legacy_index_bins',
                  init_mode='legacy_kaiming'):
    """按 token 构造模型，返回 (model, label)。forward(x)->[B,4,N]。

    pool_context 透传为 FeatExtractNetVariants(context_mode=...)；legacy_index_bins
    与 corrected symmetric 均适用于全部变体。
    init_mode 透传为 FeatExtractNetVariants(init_mode=...)；legacy_kaiming 保持
    历史 kaiming_normal 初始化与模块创建顺序，main_default 走 PyTorch 默认初始化。
    """
    if pool_context not in POOL_CONTEXT_CHOICES:
        raise ValueError(f'unknown pool_context: {pool_context!r}')
    if init_mode not in INIT_MODE_CHOICES:
        raise ValueError(f'unknown init_mode: {init_mode!r}')
    kw, label = variant_config(token)
    kw = dict(kw, context_mode=pool_context, init_mode=init_mode)
    tag = f' [pool_context={pool_context}]'
    if init_mode != 'legacy_kaiming':
        # 仅 main_default 追加标签，legacy 的 model_label 保持历史不变。
        tag += f' [init_mode={init_mode}]'
    return FeatExtractNetVariants(**kw), f'{label}{tag}'


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--variant', choices=VARIANT_CHOICES,
                    help='消融变体 token（train/aggregate 模式均需要）')
    ap.add_argument('--seeds', default=None,
                    help='explicit seed set, e.g. 0,1,2,3,4; default historical SEEDS=(0,1,2)')
    ap.add_argument('--registry-path', default=None,
                    help='optional isolated registry JSON; default canonical results/registry.json')
    ap.add_argument('--seed', type=int, default=None,
                    help='训练随机种子（--aggregate-only 时忽略；需在 {0,1,2}）')
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
                    help='输出根目录（默认 <ws>/results/ablation_thermal_protocolA；'
                         'symmetric 默认 <ws>/results/round2_ablation_symmetric；'
                         'main_default 默认 <ws>/results/round2_ablation_maininit；'
                         '模型目录 = <out-dir>/<variant>，seed 目录 = seed{n}）')
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
                    help='池化上下文：legacy_index_bins=历史索引分箱（默认）；symmetric=全支持域'
                         '对称 K=1（全部变体可用；需 --out-dir 含 round2/r2 且 --eval-chunk None）')
    ap.add_argument('--init-mode', choices=INIT_MODE_CHOICES,
                    default='legacy_kaiming',
                    help='初始化协议：legacy_kaiming=历史 kaiming_normal 显式初始化'
                         '（默认，历史结果保持不变）；main_default=PyTorch 默认初始化，'
                         '与真实 FeatExtractNet 主模型同 seed 逐位一致（需 --pool-context '
                         'symmetric + --strict-fp32，且 --out-dir 目录名含 maininit）')
    return ap.parse_args()


def _norm_out_path(path):
    """out-dir 归一化（abspath + realpath + normcase）用于比较是否指向旧目录。"""
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def check_pool_context_args(args):
    """在训练/聚合前校验 --pool-context 相关约束。"""
    sym = args.pool_context == 'symmetric'
    if not sym:
        return
    if not bool(getattr(args, 'strict_fp32', False)):
        raise SystemExit(
            '--pool-context symmetric 要求显式 --strict-fp32（全 FP32：关闭 '
            'cuda.matmul/cudnn TF32），防止默认 TF32 混入 R2 新结果。')
    if args.eval_chunk is not None:
        raise SystemExit(
            '--pool-context symmetric 要求 --eval-chunk None（整场全支持上下文），'
            f'当前 --eval-chunk={args.eval_chunk}')
    od = _norm_out_path(args.out_dir)
    base = os.path.basename(od)
    main_default = getattr(args, 'init_mode', 'legacy_kaiming') == 'main_default'
    if ('round2' not in base and 'r2' not in base
            and not (main_default and 'maininit' in base)):
        raise SystemExit(
            f'--pool-context symmetric 的 --out-dir 目录名须含 round2 或 r2'
            f'（--init-mode main_default 时须含 maininit）（当前 {args.out_dir}）')
    old_roots = {
        _norm_out_path(os.path.join(WS, 'results', LEGACY_OUT_NAME)),
        _norm_out_path(os.path.join(WS, 'results', 'ablation_thermal_protocolA')),
    }
    for old in old_roots:
        if od == old or od.startswith(old + os.sep):
            raise SystemExit(
                f'--pool-context symmetric 不得使用或嵌套在历史默认根 '
                f'{LEGACY_OUT_NAME} 内（当前 {args.out_dir}）')
    if main_default:
        prev_sym = _norm_out_path(
            os.path.join(WS, 'results', SYMMETRIC_OUT_NAME))
        if od == prev_sym or od.startswith(prev_sym + os.sep):
            raise SystemExit(
                f'--init-mode main_default 不得使用或嵌套在旧 symmetric'
                f'（Kaiming）根 {SYMMETRIC_OUT_NAME} 内（当前 {args.out_dir}）')
    # symmetric 不读取历史 legacy 结果作为输入（已有 legacy metrics 目录下拒绝）
    model_root = os.path.join(args.out_dir, args.variant)
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


def check_init_mode_args(args):
    """在训练/聚合前校验 --init-mode 约束（main_default 对齐真实主模型初值）。"""
    if args.init_mode == 'legacy_kaiming':
        return
    if args.pool_context != 'symmetric':
        raise SystemExit(
            '--init-mode main_default 要求 --pool-context symmetric'
            f'（当前 {args.pool_context}）')
    if not bool(getattr(args, 'strict_fp32', False)):
        raise SystemExit(
            '--init-mode main_default 要求显式 --strict-fp32（全 FP32：关闭 '
            'cuda.matmul/cudnn TF32），防止默认 TF32 混入 maininit 结果。')
    if args.out_dir is not None:
        base = os.path.basename(_norm_out_path(args.out_dir))
        if 'maininit' not in base:
            raise SystemExit(
                f'--init-mode main_default 的 --out-dir 目录名须含 maininit'
                f'（用于与先前 symmetric Kaiming 结果分离；当前 {args.out_dir}）')


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
        if getattr(args, 'init_mode', 'legacy_kaiming') == 'main_default':
            out_dir = os.path.join(ws, 'results', MAININIT_OUT_NAME)
        else:
            out_dir = os.path.join(ws, 'results', SYMMETRIC_OUT_NAME)
    else:
        out_dir = os.path.join(ws, 'results', LEGACY_OUT_NAME)
    os.makedirs(out_dir, exist_ok=True)
    return data, splits, out_dir


# ---------------------------------------------------------------------------
# val_selector / test_eval_fn 闭包（与 exp_thermal 完全一致的口径）
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
    token = args.variant
    pool_context = args.pool_context
    model_root = os.path.join(args.out_dir, token)
    seed_dir = os.path.join(model_root, f'seed{args.seed}')
    _guard_symmetric_seed_dir(args, seed_dir)
    os.makedirs(seed_dir, exist_ok=True)

    set_seed(args.seed)
    model, model_label = build_variant(
        token, pool_context=pool_context, init_mode=args.init_mode)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[model] variant={token} | {model_label} | params={n_params:,}',
          flush=True)

    provider = thermal_io.make_thermal_train_provider(
        X, Y, stats, train_ids, batch_size=args.batch_geoms,
        n_pts=args.n_pts, device=device, seed=args.seed)

    protocol_extra = {
        'split_version': SPLIT_VERSION,
        'experiment_group': 'T3_ablation',
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
        'init_mode': args.init_mode,
        'implementation_version': IMPLEMENTATION_VERSION,
        'cuda_matmul_allow_tf32': bool(torch.backends.cuda.matmul.allow_tf32),
        'cudnn_allow_tf32': bool(torch.backends.cudnn.allow_tf32),
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


def _consistency_value(protocol, key):
    """一致性比较取值：init_mode 缺失按 legacy_kaiming 归一（旧记录兼容）；
    其余字段缺失即 'MISSING'。"""
    if key == 'init_mode':
        return protocol.get('init_mode', 'legacy_kaiming')
    return protocol.get(key, 'MISSING')


def _validate_metrics_context(docs, pool_context,
                              init_mode='legacy_kaiming'):
    """聚合前校验：每个 metrics 的 pool_context 与 init_mode 必须等于请求值
    （缺失分别视为 legacy_index_bins / legacy_kaiming，旧记录保持兼容），
    且 AGG_CONSISTENCY_KEYS 跨 seed 一致；不一致即拒绝（在任何聚合写盘之前调用）。"""
    for s, doc in docs:
        prot = doc.get('protocol') or {}
        ctx = prot.get('pool_context', 'legacy_index_bins')
        if ctx != pool_context:
            raise SystemExit(
                f'[refuse aggregate] seed{s} metrics 的 pool_context={ctx!r} '
                f'与请求 {pool_context!r} 不一致（{ctx} 结果不得混入 {pool_context} 聚合）')
        im = prot.get('init_mode', 'legacy_kaiming')
        if im != init_mode:
            raise SystemExit(
                f'[refuse aggregate] seed{s} metrics 的 init_mode={im!r} '
                f'与请求 {init_mode!r} 不一致（初始化协议不同的结果不得混合聚合）')
    ref = docs[0][1].get('protocol') or {}
    for key in AGG_CONSISTENCY_KEYS:
        ref_val = _consistency_value(ref, key)
        for s, doc in docs[1:]:
            cur = doc.get('protocol') or {}
            cur_val = _consistency_value(cur, key)
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

    仅当 model_root 下 seed0..2 的 metrics 全部齐备时写 registry；否则打印
    提示并跳过登记（例如干跑 epochs 少、seed 数不足的场景）。
    """
    token = args.variant
    pool_context = args.pool_context
    init_mode = args.init_mode
    if pool_context == 'legacy_index_bins':
        experiment_id = f'T3_fen_{token}'
    elif init_mode == 'main_default':
        # 初始化对齐对照：独立 ID，不覆盖先前的 symmetric（Kaiming）结果
        experiment_id = f'T3_fen_{token}_R2_symmetric_maininit'
    else:
        experiment_id = f'T3_fen_{token}_R2_symmetric'
    model_root = os.path.join(args.out_dir, token)
    docs = _load_all_seed_metrics(model_root)
    if not docs:
        raise FileNotFoundError(f'{model_root} 下无 metrics_seed*.json')
    # 任何 summary/registry 写盘之前，init_mode/pool_context 必须与全部 seed 一致
    _validate_metrics_context(docs, pool_context, init_mode)

    agg_all = {k: _aggregate_field_values(docs, k) for k in REGISTRY_KEYS}
    n_seeds = len(docs)
    print(f'\n=== aggregate variant={token} ({model_root}) | '
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
    print('  latex:', runner.latex_row(experiment_id, latex_entry,
                                       fields=FIELD_NAMES))
    for f in FIELD_NAMES:
        if re_a['mean'][f] is None:
            continue
        mae = agg_all['mae']['mean'][f]
        mx = agg_all['max_err']['mean'][f]
        bd_mae = agg_all['mae_bd']['mean'][f]
        print(f'  {f}: RE={re_a["mean"][f]:.4f}±{re_a["std"][f]:.4f} | '
              f'MAE={mae:.3e} | max={mx:.3e} | boundary MAE={bd_mae:.3e}')

    first = docs[0][1]['protocol']
    summary = {
        'experiment_id': experiment_id,
        'variant': token,
        'pool_context': pool_context,
        'init_mode': init_mode,
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

    if n_seeds < len(SEEDS):
        print(f'[skip registry] 只有 {n_seeds}/{len(SEEDS)} 个 seed 的 metrics，'
              '不做 registry 登记（非完整 3-seed 消融运行）。')
        return

    protocol = {
        'split_version': SPLIT_VERSION,
        'experiment_group': 'T3_ablation',
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
        'seeds': list(SEEDS),
        'pool_context': pool_context,
        'init_mode': init_mode,
        'implementation_version': IMPLEMENTATION_VERSION,
        'cuda_matmul_allow_tf32': first.get('cuda_matmul_allow_tf32'),
        'cudnn_allow_tf32': first.get('cudnn_allow_tf32'),
        's_per_epoch_mean': summary.get('s_per_epoch_mean'),
        'hardware': first.get('hardware'),
    }
    entry = new_experiment_entry(
        experiment_id=experiment_id,
        protocol=protocol,
        seed=list(SEEDS),
        epochs=first.get('epochs'),
        lr=first.get('lr'),
        model=f'FeatExtractNetVariants({token}) — {first.get("model_label")}',
        model_params=first.get('model_params'),
        hardware=first.get('hardware'),
        torch_version=first.get('hardware', {}).get('pytorch_version'),
        tool='exp_ablation_thermal.py',
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
    # registry 路径锚定 src/core/registry.py -> <ws>/results/registry.json
    registry_path = getattr(args, 'registry_path', None)
    if not registry_path:
        registry_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(registry.__file__)))), 'results', 'registry.json')
    n_reg = _locked_append_registry(entry, registry_path=registry_path)
    print(f'[registry] upsert {experiment_id} -> {registry_path} '
          f'(total {n_reg} entries)', flush=True)


def main():
    global SEEDS
    args = parse_args()
    if args.seeds:
        vals = tuple(sorted(int(x) for x in str(args.seeds).split(',') if x.strip()))
        if not vals or any(v < 0 or v > 4 for v in vals):
            raise SystemExit('--seeds must be a non-empty subset of 0..4')
        SEEDS = vals
    if args.strict_fp32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    # main_default 约束先于 resolve_paths/makedirs 校验（避免创建错误目录）
    check_init_mode_args(args)
    data, splits, out_dir = resolve_paths(args)
    args.data, args.splits, args.out_dir = data, splits, out_dir
    if args.variant is None:
        raise SystemExit('--variant 必填: ' + '|'.join(VARIANT_CHOICES))

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
        print('[warn] CPU 训练（仅 dry-run 小规模用；正式消融需 GPU）',
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
    # 若该模型 3 个 seed 全部齐备则自动聚合登记（否则跳过）
    aggregate_and_register(args)


if __name__ == '__main__':
    main()
