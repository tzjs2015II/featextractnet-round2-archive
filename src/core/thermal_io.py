"""thermal_io.py —— 热场数据 IO、归一化与几何级评估（src.core.thermal_io）。

用途
----
为"几何级留出协议"（Protocol-A/B，见 configs/splits_thermal.json 与
实验计划.md §1/§5 Track T）提供统一数据入口：

  * load_thermal              读取 HeatTransferData.npy -> X[G,N,2], Y[G,N,4]
  * fit_stats                 仅用给定几何（train-only）计算坐标/输出 mean±std
  * normalize / denormalize   对 [.., C] 数组按通道做 z-score / 反 z-score
  * make_geometric_batches    每 epoch 从 train 几何随机抽 batch（batch_size
                               个几何），每几何可随机下采样 n_pts（默认全 5000）
  * make_thermal_train_provider 包 normalize + to_torch 的 epoch provider
                               （供 src.core.runner.train_supervised 使用）
  * eval_geometry_wise        逐几何整场预测（可 chunk），反归一化到物理单位后
                               计算每通道 RE/MAE/maxErr，并额外返回边界点
                               （索引 < boundary_n=660）物理单位 RE

评估口径（呼应协议 §1，勿偏离）
--------------------------------
主指标一律在物理单位域计算：model 输出为归一化域 [B,Cout,N]，本模块内部
denorm 回物理单位后按 RE=||ŷ−y||₂/||y||₂ 逐几何逐通道计算；归一化只存在于
训练预处理管线。热场统计（坐标 2 通道 + 输出 4 通道）只用 train 几何。

边界点约定
----------
HeatTransferData.npy 每个几何前 660 个点为边界/近壁区（环境事实），
内部点索引为 660..4999。
"""

import numpy as np
import torch

from ..utils.data_loader import to_torch
from ..utils.metrics import relative_l2_error

DEFAULT_FIELD_NAMES = ('u', 'v', 'p', 'T')
DEFAULT_BOUNDARY_N = 660  # 每几何前 660 点为边界/近壁点


def load_thermal(path):
    """Load thermal field data -> (X, Y)。

    Args:
        path: HeatTransferData.npy 路径（shape (135,5000,6)，列序
              [x,y,u,v,p,T]，坐标约 [-1,1]）。

    Returns:
        X: [G, N, 2] float32 坐标
        Y: [G, N, 4] float32 物理量（u,v,p,T）
    """
    data = np.load(path)
    if data.ndim != 3 or data.shape[-1] != 6:
        raise ValueError(
            f"expected [G,N,6] data, got shape {data.shape}")
    X = data[:, :, :2].astype(np.float32)
    Y = data[:, :, 2:6].astype(np.float32)
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError("data contains NaN/Inf")
    return X, Y


def fit_stats(X, Y, geom_ids=None, eps=1e-8):
    """Compute per-channel mean/std over a geometry index list (train-only)。

    Args:
        X: [G, N, 2] 坐标（物理单位）
        Y: [G, N, 4] 输出（物理单位）
        geom_ids: 参与统计的几何索引列表（协议：只用 train 几何）。
                  默认 None = 全部几何。
        eps: std 下限，防止零方差通道除零。

    Returns:
        dict:
            x_mean/x_std: [2] float32 坐标通道统计
            y_mean/y_std: [4] float32 输出通道统计
    """
    if geom_ids is None:
        geom_ids = np.arange(X.shape[0])
    geom_ids = np.asarray(geom_ids)
    if geom_ids.size == 0:
        raise ValueError("geom_ids is empty; nothing to fit stats on")

    x_all = X[geom_ids].reshape(-1, X.shape[-1]).astype(np.float64)
    y_all = Y[geom_ids].reshape(-1, Y.shape[-1]).astype(np.float64)

    def _stat(arr):
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        std = np.maximum(std, eps)
        return mean.astype(np.float32), std.astype(np.float32)

    x_mean, x_std = _stat(x_all)
    y_mean, y_std = _stat(y_all)
    return {
        'x_mean': x_mean, 'x_std': x_std,
        'y_mean': y_mean, 'y_std': y_std,
        'n_train_geom': int(geom_ids.size),
    }


def normalize(arr, mean, std, eps=1e-8):
    """z-score：对 [.., C] 数组按通道广播 (arr - mean) / max(std, eps)。

    mean/std 形如 (C,)，沿数组最后一维对齐（支持 [N,C]、[B,N,C] 等）。
    """
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    std = np.asarray(std, dtype=np.float64).reshape(-1)
    std = np.maximum(std, eps)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.shape[-1] != mean.shape[0]:
        raise ValueError(
            f"last dim {arr.shape[-1]} != stat dim {mean.shape[0]}")
    out = (arr - mean) / std
    return out.astype(np.float32)


def denormalize(z, mean, std):
    """反 z-score：对 [.., C] 数组按通道广播 z * std + mean（回物理单位）。"""
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    std = np.asarray(std, dtype=np.float64).reshape(-1)
    z = np.asarray(z, dtype=np.float64)
    if z.shape[-1] != mean.shape[0]:
        raise ValueError(
            f"last dim {z.shape[-1]} != stat dim {mean.shape[0]}")
    return (z * std + mean).astype(np.float32)


def make_geometric_batches(X, Y, geom_ids, batch_size, n_pts=None, rng=None):
    """每 epoch 的几何级 batch 生成器（yield numpy 物理单位数组）。

    协议要点：以"几何"为采样单元（训练几何内点级切分只存在于历史旧协议，
    不作为主基准）；每 epoch 打乱几何顺序后按 batch_size 切块；每个几何可
    随机下采样固定 n_pts 个点（默认 None = 使用全部 N 点，保证 batch 内各
    几何点数一致以便张量化）。

    Args:
        X: [G, N, 2]
        Y: [G, N, 4]
        geom_ids: 本次参与采样的几何索引（如 Protocol-A train 100 个）
        batch_size: 每 batch 的几何数
        n_pts: 每几何点数上限；None 或 >= N 时取全量
        rng: np.random.RandomState 实例；None 时用 np.random

    Yields:
        (xb, yb, g_ids): xb [B, n_pts, 2]、yb [B, n_pts, 4] float32，
        g_ids 为对应几何索引 int 列表。
    """
    geom_ids = np.asarray(geom_ids)
    if geom_ids.size == 0:
        raise ValueError("geom_ids is empty")
    N_full = X.shape[1]
    if rng is None:
        rng = np.random

    order = rng.permutation(geom_ids.size)
    for start in range(0, order.size, batch_size):
        g = geom_ids[order[start:start + batch_size]]
        sub = np.empty((g.size, N_full, X.shape[-1]), dtype=np.float32)
        sub_y = np.empty((g.size, N_full, Y.shape[-1]), dtype=np.float32)
        sub[:] = X[g]
        sub_y[:] = Y[g]
        if n_pts is not None and 0 < n_pts < N_full:
            idx = np.stack([
                rng.permutation(N_full)[:n_pts] for _ in range(g.size)
            ])
            xb = np.take_along_axis(sub, idx[:, :, None], axis=1)
            yb = np.take_along_axis(sub_y, idx[:, :, None], axis=1)
        else:
            xb = sub
            yb = sub_y
        yield xb, yb, [int(v) for v in g]


def make_thermal_train_provider(X, Y, stats, geom_ids, batch_size,
                                n_pts=None, device='cpu', seed=0):
    """构造 train_supervised 用的 data_provider（每 epoch 产出归一化 torch batch）。

    Args:
        X: [G, N, 2]、Y: [G, N, 4]（物理单位）
        stats: fit_stats 返回值（train-only 归一化统计）
        geom_ids: train 几何索引
        batch_size / n_pts: 见 make_geometric_batches
        device: torch 设备字符串
        seed: 内部 RandomState 种子（与训练 seed 一致可复现 epoch 内打乱）

    Returns:
        provider(epoch) -> iterator of (x_t [B,Cin,N], y_t [B,Cout,N])。
        x_t/y_t 为已按 stats 归一化到归一化域的 float32 torch tensor。
    """
    rng = np.random.RandomState(seed)

    def provider(epoch):
        # epoch 仅用于记录；打乱状态由 rng 沿调用序列推进（seed 固定即确定）。
        for xb, yb, _ in make_geometric_batches(
                X, Y, geom_ids, batch_size, n_pts=n_pts, rng=rng):
            xn = normalize(xb, stats['x_mean'], stats['x_std'])
            yn = normalize(yb, stats['y_mean'], stats['y_std'])
            x_t, y_t = to_torch(xn, yn, device=device)
            yield x_t, y_t

    return provider


def _predict_geometry(model_predict, x_phys, stats, device, chunk_points):
    """单几何整场预测：归一化坐标 -> model_predict -> 反归一化回物理单位。

    Args:
        model_predict: callable(x_t [B,2,n]) -> pred [B,Cout,n]（归一化域）
        x_phys: [1, N, 2] float32 物理坐标
        stats: fit_stats 返回值
        chunk_points: 每次喂给模型的最大点数（None = 一次全量），
                      用于 CUDA 显存受限的大 N 场景。
    Returns:
        pred_phys: [N, Cout] float32 物理单位预测
    """
    N = x_phys.shape[1]
    Cout = stats['y_mean'].shape[0]
    bounds = [0]
    if chunk_points is None or chunk_points >= N:
        bounds = [0, N]
    else:
        bounds = list(range(0, N, chunk_points)) + [N]
    outs = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        xchunk = x_phys[:, s:e, :]
        xn = normalize(xchunk, stats['x_mean'], stats['x_std'])  # [1,n,2]
        xt = torch.tensor(xn, dtype=torch.float32, device=device).permute(0, 2, 1)
        with torch.no_grad():
            pn = model_predict(xt)  # [1, Cout, n]
        pn = pn.detach().to('cpu').numpy()  # noqa: no-op if already cpu
        outs.append(pn)
    pred_norm = np.concatenate(outs, axis=2)  # [1, Cout, N]
    pred_norm_t = np.transpose(pred_norm, (0, 2, 1))  # [1, N, Cout]
    pred_phys = denormalize(
        pred_norm_t[0], stats['y_mean'], stats['y_std'])  # [N, Cout]
    return pred_phys


def eval_geometry_wise(model_predict, geom_ids, X, Y, stats, device,
                       field_names=DEFAULT_FIELD_NAMES,
                       boundary_n=DEFAULT_BOUNDARY_N, chunk_points=None,
                       point_index=None):
    """逐几何整场评估，返回物理单位 RE/MAE/maxErr 与逐几何表。

    口径：对每个几何独立预测整场（N=5000 点，可 chunk），把归一化域输出
    denorm 到物理单位后计算：
      RE     = ||ŷ−y||₂ / ||y||₂            （主指标，物理单位）
      MAE    = mean(|ŷ−y|)
      maxErr = max(|ŷ−y|)
      *_bd   = 仅边界/近壁点（索引 < boundary_n）上的指标
    边界 RE（re_<f>_bd）退化为 None 的情形：本数据每几何前 boundary_n
    个点是固壁无滑移节点（u=v=0，逐几何实测），其子集真值 L2 范数恰为 0，
    相对误差 0/0 无定义。对此类通道 re_<f>_bd=None（JSON null）如实标记，
    并同时给出边界 MAE/maxErr（mae_<f>_bd / max_<f>_bd，恒有限）作为近壁
    误差的可用口径。**绝不在归一化域计算主指标**。

    Args:
        model_predict: callable(x_t [B,2,n]) -> pred [B,Cout,n]，
                      输入/输出均为归一化域（本函数负责 denorm）。
        geom_ids: 待评估几何索引列表
        X: [G, N, 2]、Y: [G, N, 4]（物理单位）
        stats: fit_stats 返回值（train-only 统计，与训练一致）
        device: torch 设备
        field_names: 输出通道名，默认 ('u','v','p','T')
        boundary_n: 边界/近壁点数（默认 660）
        chunk_points: 单次前向最大点数（显存受限场景），None = 全量一次
        point_index: 可选 {geom_id: 索引数组}。给出时只在该几何的这部分点上
                     评估（用于点密度鲁棒性实验，见
                     docs/T5_点密度鲁棒性设计.md）。**此时边界指标不可用**：
                     边界点由"前 boundary_n 个索引"定义，子采样后该索引约定
                     失效，故 *_bd 列一律置 None（JSON null），而不是给出一个
                     语义不明的小子集误差。

    Returns:
        dict:
            n_geom
            rows: 逐几何行（flat 列名 re_<f>, mae_<f>, max_<f>,
                  re_<f>_bd, mae_<f>_bd, max_<f>_bd）
            boundary_metrics_available: bool（子采样评估时为 False）
            agg: 跨几何汇总
                  {'re': {f: mean}, 'mae': {f: mean}, 'max_err': {f: mean},
                   're_bd': {f: mean|None}, 'mae_bd': {f: mean|None},
                   'max_bd': {f: mean|None}, 're_std': {f: std},
                   're_bd_std': {f: std|None}, ...}
    """
    if boundary_n is None or boundary_n > X.shape[1]:
        boundary_n = X.shape[1]
    geom_ids = np.asarray(geom_ids)
    Cout = Y.shape[2]
    if len(field_names) != Cout:
        raise ValueError(
            f"field_names {len(field_names)} != output channels {Cout}")

    re_mat, mae_mat, max_mat = [], [], []
    rebd_mat, maebd_mat, maxbd_mat = [], [], []
    rows = []
    for g in geom_ids:
        x_full = X[g:g + 1].astype(np.float32)      # [1, N, 2]
        y_full = Y[g].astype(np.float64)            # [N, Cout] 物理真值
        idx = None if point_index is None else point_index.get(int(g))
        if idx is None:
            x_phys, y_phys, bd_n = x_full, y_full, boundary_n
        else:
            idx = np.asarray(idx, dtype=np.int64)
            x_phys, y_phys, bd_n = x_full[:, idx, :], y_full[idx], 0
        pred_phys = _predict_geometry(
            model_predict, x_phys, stats, device, chunk_points)  # [N,Cout]

        row = {'geom_id': int(g), 'n_pts': int(x_phys.shape[1]),
               'evaluated_on_subset': bool(idx is not None)}
        re_list, mae_list, max_list = [], [], []
        rebd_list, maebd_list, maxbd_list = [], [], []
        for c, f in enumerate(field_names):
            p = pred_phys[:, c].astype(np.float64)
            t = y_phys[:, c]
            re = relative_l2_error(p, t)
            mae = float(np.abs(p - t).mean())
            mxe = float(np.abs(p - t).max())
            if bd_n > 0:
                tb = t[:bd_n]
                pb = p[:bd_n]
                # 边界 RE：子集真值 L2 范数 ~0（无滑移壁 u=v=0）时无定义 -> None
                rebd = None
                t_norm = float(np.linalg.norm(tb))
                if t_norm > 1e-12:
                    rebd = relative_l2_error(pb, tb)
                maebd = float(np.abs(pb - tb).mean())
                maxbd = float(np.abs(pb - tb).max())
            else:
                # 子采样评估：边界索引约定失效，不予报告（见 docstring）
                rebd = maebd = maxbd = None
            row[f're_{f}'] = re
            row[f'mae_{f}'] = mae
            row[f'max_{f}'] = mxe
            row[f're_{f}_bd'] = rebd
            row[f'mae_{f}_bd'] = maebd
            row[f'max_{f}_bd'] = maxbd
            re_list.append(re)
            mae_list.append(mae)
            max_list.append(mxe)
            rebd_list.append(rebd)
            maebd_list.append(maebd)
            maxbd_list.append(maxbd)
        rows.append(row)
        re_mat.append(re_list)
        mae_mat.append(mae_list)
        max_mat.append(max_list)
        rebd_mat.append(rebd_list)
        maebd_mat.append(maebd_list)
        maxbd_mat.append(maxbd_list)

    re_arr = np.asarray(re_mat, dtype=np.float64)     # [n_geom, Cout]
    mae_arr = np.asarray(mae_mat, dtype=np.float64)
    max_arr = np.asarray(max_mat, dtype=np.float64)
    # 边界列可能是 None（子采样评估 / 无滑移壁通道），故用 object 数组
    maebd_arr = np.asarray(maebd_mat, dtype=object)
    maxbd_arr = np.asarray(maxbd_mat, dtype=object)

    def _grp(arr):
        return {f: float(arr[:, c].mean()) for c, f in enumerate(field_names)}

    def _grp_bd(cols):
        """按 field 对带 None 的边界 RE 列做均值；全部 None 时该 field 记 None。"""
        out = {}
        for c, f in enumerate(field_names):
            vals = [v for v in cols[:, c] if v is not None]
            out[f] = float(np.mean(vals)) if vals else None
        return out

    rebd_cols = np.asarray(rebd_mat, dtype=object)   # [n_geom, Cout]，含 None

    def _std_bd(cols):
        out = {}
        for c, f in enumerate(field_names):
            vals = [v for v in cols[:, c] if v is not None]
            out[f] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
        return out

    agg = {
        're': _grp(re_arr),
        'mae': _grp(mae_arr),
        'max_err': _grp(max_arr),
        're_bd': _grp_bd(rebd_cols),
        'mae_bd': _grp_bd(maebd_arr),
        'max_bd': _grp_bd(maxbd_arr),
        're_std': {f: (float(re_arr[:, c].std(ddof=1))
                           if re_arr.shape[0] > 1 else 0.0)
                   for c, f in enumerate(field_names)},
        're_bd_std': _std_bd(rebd_cols),
    }
    return {'n_geom': int(geom_ids.size), 'rows': rows, 'agg': agg,
            'boundary_metrics_available': bool(boundary_n > 0
                                               and point_index is None)}
