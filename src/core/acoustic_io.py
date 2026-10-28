"""acoustic_io.py —— 声场数据 IO、统计与整场评估（src.core.acoustic_io）。

用途
----
为声场实验（实验计划.md §7 Track A）提供统一入口。声场数据为 FDTD
生成的规则网格工况，每个 case 一个 npz：

  data/acoustic/case_{f}_{Ma}.npz
    p_total / p_incident / p_scattered : [H, W] 物理单位声压（Pa）
    x / y                              : [W] / [H] 一维坐标（m），
                                        本数据集 H=W=481，坐标 [-30,30]，
                                        dx=dy=0.125（环境事实）
    meta                               : JSON 字符串（f_hz、Ma 等参数）

网格约定（已对本机数据核实）：p[i, j] 对应 (x=x[i], y=y[j])，即
np.meshgrid(x, y, indexing='ij') 的行列顺序（声场沿 x 传播，p_incident
沿行方向变化、沿列方向不变）。全网格点 N = H*W = 231k，评估时按
chunk_points（默认 65536）分块喂给模型，避免一次载入超大点集。

评估口径（呼应协议 §1）
------------------------
声场主指标同样是物理单位 RE：模型输出在归一化域（对 p_total 的 pooled
train-case z-score），本模块在整场 scatter 回网格后先反归一化到物理单位，
再算 RE/MAE/maxErr。坐标同域（[-30,30]）共享一套仿射映射，仅输出统计用
train case pooled 计算。
"""

import json
import os

import numpy as np
import torch

from ..utils.metrics import relative_l2_error

DEFAULT_FIELD = 'p_total'
DEFAULT_CHUNK = 65536  # 481*481=231k 点按 64k chunk，限制峰值显存


# ---------------------------------------------------------------------------
# 加载与网格互转
# ---------------------------------------------------------------------------
def parse_meta(meta):
    """把 npz 内 JSON 字符串 meta 解析为 dict（含 f_hz、Ma 等）。"""
    s = str(meta)
    try:
        out = json.loads(s)
        return out if isinstance(out, dict) else {'raw': s}
    except (ValueError, TypeError):
        return {'raw': s}


def load_case(npz_path):
    """Load a single acoustic case.

    Args:
        npz_path: data/acoustic/case_*.npz 路径。

    Returns:
        x2d: [H, W] float32 网格 x 坐标（meshgrid ij 顺序，行 = x）
        y2d: [H, W] float32 网格 y 坐标
        p2d: [H, W] float64 物理单位总声压 p_total
        meta: dict（f_hz、Ma 等参数）
    """
    with np.load(npz_path) as d:
        keys = set(d.keys())
        for need in ('p_total', 'x', 'y'):
            if need not in keys:
                raise KeyError(f'{npz_path} missing key {need}')
        p2d = d['p_total'].astype(np.float64)
        x1d = d['x'].astype(np.float64)
        y1d = d['y'].astype(np.float64)
        meta = parse_meta(d['meta'])
    if p2d.ndim != 2:
        raise ValueError(f'{npz_path}: p_total not 2D, got {p2d.shape}')
    x2d, y2d = np.meshgrid(x1d, y1d, indexing='ij')
    if p2d.shape != x2d.shape:
        raise ValueError(
            f'{npz_path}: p shape {p2d.shape} != grid shape {x2d.shape}')
    if not np.isfinite(p2d).all():
        raise ValueError(f'{npz_path}: p_total contains NaN/Inf')
    return x2d.astype(np.float32), y2d.astype(np.float32), p2d, meta


def grid_to_points(x2d, y2d):
    """2D 网格 -> 点列表 [H*W, 2]（meshgrid ij ravel 顺序，与 p2d.ravel 一致）。"""
    x2d = np.asarray(x2d)
    y2d = np.asarray(y2d)
    if x2d.shape != y2d.shape:
        raise ValueError('x2d/y2d shape mismatch')
    pts = np.stack([x2d.ravel(), y2d.ravel()], axis=-1)
    return pts.astype(np.float32)


def scatter_to_grid(pred_pointwise, x2d, y2d):
    """点预测还原 2D 场：pred_pointwise[N] -> [H, W]。

    该函数对点序不敏感：按坐标 searchsorted 回填到 (x2d[i,j], y2d[i,j])
    网格单元。x2d/y2d 必须来自同一 meshgrid（ij 顺序）。物理量或归一化量
    均可，逐元素放置。
    """
    pred = np.asarray(pred_pointwise).reshape(-1)
    x2d = np.asarray(x2d)
    y2d = np.asarray(y2d)
    H, W = x2d.shape
    if pred.size != H * W:
        raise ValueError(
            f'pred size {pred.size} != grid {H}x{W}')
    pts = grid_to_points(x2d, y2d)
    x1d = x2d[:, 0]
    y1d = y2d[0, :]
    # x1d/y1d 单调递增（meshgrid 坐标轴），searchsorted 映射回行列。
    i = np.searchsorted(x1d, pts[:, 0])
    j = np.searchsorted(y1d, pts[:, 1])
    if not (np.all(i < H) and np.all(j < W)):
        raise ValueError('point coordinates outside grid axes')
    grid = np.empty((H, W), dtype=pred.dtype)
    grid[i, j] = pred
    return grid


# ---------------------------------------------------------------------------
# 工况清单与统计
# ---------------------------------------------------------------------------
def make_case_list(config_path, roles=None):
    """读取 configs/acoustic_cases.json，返回工况条目列表。

    Args:
        config_path: acoustic_cases.json 路径（file 字段相对工作区根）。
        roles: 可选 dict {case_id: role}（如 {'400_0.5': 'train'}）；缺省
               条目 role 记 'unassigned'。

    Returns:
        list of dict：case_id / file（绝对路径）/ f / Ma / source_side /
        status / role。
    """
    with open(config_path, 'r', encoding='utf-8') as fh:
        doc = json.load(fh)
    ws = os.path.dirname(os.path.dirname(os.path.abspath(config_path)))
    roles = roles or {}
    out = []
    for cid, rec in doc.get('cases', {}).items():
        params = rec.get('params') or {}
        fpath = rec.get('file')
        if not fpath:
            raise ValueError(f'case {cid}: missing "file"')
        if not os.path.isabs(fpath):
            fpath = os.path.join(ws, fpath)
        out.append({
            'case_id': cid,
            'file': fpath,
            'f': params.get('f_hz'),
            'Ma': params.get('Ma'),
            'source_side': params.get('source_side'),
            'status': rec.get('status'),
            'role': roles.get(cid, 'unassigned'),
        })
    return out


def coord_affine(points, x_min, x_max, y_min, y_max):
    """把 [.., 2] 坐标（第 0 列 x、第 1 列 y）仿射映射到 [-1, 1]。"""
    points = np.asarray(points, dtype=np.float64)
    lo = np.array([x_min, y_min], dtype=np.float64)
    hi = np.array([x_max, y_max], dtype=np.float64)
    scale = (hi - lo) / 2.0
    if (scale <= 0).any():
        raise ValueError('coordinate ranges must be positive')
    out = (points - (hi + lo) / 2.0) / scale
    return out.astype(np.float32)


def fit_stats_cases(case_entries, eps=1e-8):
    """pooled train-case 统计：输出 p_total z-score + 坐标仿射域。

    只在传入的 case 条目（应为 train 整场留出组）上 pooled 计算输出
    mean/std；坐标全域共享（同一仿射映射，[-30,30]），取其 min/max。

    Args:
        case_entries: make_case_list 返回的子集（train case 条目列表）。
        eps: p_std 下限。

    Returns:
        dict:
            p_mean/p_std: float32（z-score 用，train pooled）
            x_min/x_max/y_min/y_max: float
            n_points / case_ids
    """
    if not case_entries:
        raise ValueError('case_entries is empty')
    n = 0
    s = 0.0
    sq = 0.0
    case_ids = []
    coord = None
    for c in case_entries:
        x2d, y2d, p2d, _ = load_case(c['file'])
        p = p2d.astype(np.float64).ravel()
        n += p.size
        s += float(p.sum())
        sq += float((p * p).sum())
        case_ids.append(c['case_id'])
        if coord is None:
            coord = {
                'x_min': float(x2d.min()), 'x_max': float(x2d.max()),
                'y_min': float(y2d.min()), 'y_max': float(y2d.max()),
            }
    mean = s / n
    var = max(sq / n - mean * mean, 0.0)
    std = max(np.sqrt(var), eps)
    return {
        'p_mean': np.float32(mean),
        'p_std': np.float32(std),
        'n_points': int(n),
        'case_ids': case_ids,
        'x_min': coord['x_min'], 'x_max': coord['x_max'],
        'y_min': coord['y_min'], 'y_max': coord['y_max'],
    }


def zscore_p(p, p_mean, p_std):
    """单通道声压 z-score（标量 mean/std）。"""
    return (np.asarray(p, dtype=np.float64) - float(p_mean)) / float(p_std)


def unzscore_p(z, p_mean, p_std):
    """反 z-score 回物理单位声压。"""
    return (np.asarray(z, dtype=np.float64) * float(p_std)) + float(p_mean)


# ---------------------------------------------------------------------------
# 整场评估
# ---------------------------------------------------------------------------
def eval_case_physical(pred_full_2d, p2d):
    """逐工况整场物理单位指标（pred/truth 均须为物理单位 2D 场）。

    RE = ||ŷ−y||₂ / ||y||₂、MAE = mean|ŷ−y|、maxErr = max|ŷ−y|。
    输入须为物理单位：归一化只在训练预处理，主指标在物理单位域计算。

    Returns:
        dict {'re': float, 'mae': float, 'max_err': float}
    """
    pred = np.asarray(pred_full_2d, dtype=np.float64).ravel()
    truth = np.asarray(p2d, dtype=np.float64).ravel()
    if pred.shape != truth.shape:
        raise ValueError('pred/truth shape mismatch')
    re = relative_l2_error(pred, truth)
    mae = float(np.abs(pred - truth).mean())
    mxe = float(np.abs(pred - truth).max())
    return {'re': re, 'mae': mae, 'max_err': mxe}


def eval_case_model(model_predict, case_file, p_stats, device,
                    chunk_points=DEFAULT_CHUNK, case_id=None,
                    field_name=DEFAULT_FIELD):
    """整场留出评估：点云预测 + chunk 分块 + scatter + 物理单位指标。

    Args:
        model_predict: callable(x_t [B,2,n]) -> pred [B,1,n]；输入为仿射
                       归一化坐标，输出为 z-score 归一化域（本函数反归一化）。
        case_file: case npz 绝对路径
        p_stats: fit_stats_cases 返回值（p_mean/p_std 与坐标域）
        device: torch 设备
        chunk_points: 单次前向点数上限（默认 65536，控制峰值显存）
        case_id: 记录用工况名；None 时取文件名
        field_name: 指标通道名（默认 'p_total'）

    Returns:
        dict:
            agg: {'re': {f: ..}, 'mae': {f: ..}, 'max_err': {f: ..}}
            rows: [{'case_id', 're', 'mae', 'max_err'}]
            pred_grid: [H, W] float32 物理单位预测场（画图用）
            truth_grid: [H, W] float64 物理单位真值
    """
    x2d, y2d, p2d, _ = load_case(case_file)
    pts = grid_to_points(x2d, y2d)                     # [N, 2]
    xn = coord_affine(pts, p_stats['x_min'], p_stats['x_max'],
                      p_stats['y_min'], p_stats['y_max'])   # [N, 2] float32
    N = xn.shape[0]
    z_chunks = []
    bounds = (list(range(0, N, chunk_points)) + [N]) if chunk_points else [0, N]
    for s, e in zip(bounds[:-1], bounds[1:]):
        xt = torch.tensor(xn[s:e][None], dtype=torch.float32,
                          device=device).permute(0, 2, 1)   # [1,2,n]
        with torch.no_grad():
            pz = model_predict(xt)                         # [1,1,n]
        z_chunks.append(pz.detach().to('cpu').numpy())
    z = np.concatenate(z_chunks, axis=2)[0, 0]              # [N]
    p_phys = unzscore_p(z, p_stats['p_mean'], p_stats['p_std'])
    pred_grid = scatter_to_grid(p_phys, x2d, y2d)
    m = eval_case_physical(pred_grid, p2d)
    cid = case_id if case_id is not None else os.path.basename(case_file)
    return {
        'agg': {'re': {field_name: m['re']},
                'mae': {field_name: m['mae']},
                'max_err': {field_name: m['max_err']}},
        'rows': [{'case_id': cid, 're': m['re'],
                  'mae': m['mae'], 'max_err': m['max_err']}],
        'pred_grid': pred_grid.astype(np.float32),
        'truth_grid': p2d,
    }


def summarize_case_rows(rows, field_name=DEFAULT_FIELD):
    """把多工况 eval_case_model 的 rows 汇总为 agg（跨工况 mean / std）。

    Args:
        rows: 各工况 dict，含 're'/'mae'/'max_err' 标量。
        field_name: 汇总通道名（声场默认单通道 'p_total'）。

    Returns:
        dict: {'re': {f: mean}, 'mae': {f: mean}, 'max_err': {f: mean},
               're_std': {f: std}, 'n_case': int}
    """
    if not rows:
        raise ValueError('rows is empty')
    agg = {}
    for grp in ('re', 'mae', 'max_err'):
        vals = [float(r[grp]) for r in rows]
        agg[grp] = {field_name: float(np.mean(vals))}
        agg[grp + '_std'] = {field_name: (float(np.std(vals, ddof=1))
                                                if len(vals) > 1 else 0.0)}
    agg['n_case'] = len(rows)
    return agg


# ---------------------------------------------------------------------------
# FNO 规则网格组装
# ---------------------------------------------------------------------------
def grid_to_tensor(x2d, y2d, p2d=None, cond=None, p_stats=None,
                   coord_stats=None):
    """把 2D 场与坐标网格组装为 [1, Cin, H, W] 张量（FNO 入口）。

    通道：前 2 通道 = 仿射归一化坐标 (x,y)（[-1,1]）；其后 K 通道 =
    cond（物理参数条件标量广播，如 [f_hz, Ma]），总 Cin = 2 + K。

    Args:
        x2d/y2d: [H, W] 网格坐标
        p2d: 可选物理单位 p_total [H, W]
        cond: 可选标量序列（每项广播一个常量通道）
        p_stats: fit_stats_cases 返回（z-score 目标用）；None 时不归一化目标
        coord_stats: 坐标仿射域 dict；None 时从本网格推导

    Returns:
        (x_t, y_norm_t, y_phys_t)：
            x_t       [1, Cin, H, W] float32
            y_norm_t  [1, 1, H, W] float32（p_stats 提供时；否则 None）
            y_phys_t  [1, 1, H, W] float32（p2d 提供时；否则 None）
    """
    x2d = np.asarray(x2d, dtype=np.float64)
    y2d = np.asarray(y2d, dtype=np.float64)
    if coord_stats is None:
        coord_stats = {'x_min': float(x2d.min()), 'x_max': float(x2d.max()),
                       'y_min': float(y2d.min()), 'y_max': float(y2d.max())}

    def _map_plane(plane, lo, hi):
        # 单个 2D 坐标平面按 [lo, hi] 仿射到 [-1,1]
        scale = (hi - lo) / 2.0
        if scale <= 0:
            raise ValueError('coordinate range must be positive')
        return ((plane - (hi + lo) / 2.0) / scale).astype(np.float32)

    xc = _map_plane(x2d, coord_stats['x_min'], coord_stats['x_max'])
    yc = _map_plane(y2d, coord_stats['y_min'], coord_stats['y_max'])
    chans = [xc, yc]
    if cond is not None:
        for v in cond:
            chans.append(np.full_like(xc, float(v)))
    grid_np = np.stack(chans, axis=0).astype(np.float32)[None]  # [1,Cin,H,W]
    x_t = torch.tensor(grid_np, dtype=torch.float32)

    y_phys_t = y_norm_t = None
    if p2d is not None:
        p = np.asarray(p2d, dtype=np.float64)
        y_phys_t = torch.tensor(p.astype(np.float32)[None, None])
        if p_stats is not None:
            y_norm_t = torch.tensor(
                zscore_p(p, p_stats['p_mean'], p_stats['p_std'])
                .astype(np.float32)[None, None])
    return x_t, y_norm_t, y_phys_t
