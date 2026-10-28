"""runner.py —— 监督训练通用循环 + 跨 seed 聚合（src.core.runner）。

用途
----
为热场/声场监督任务提供统一训练器（实验计划.md §5 Track T、§7 Track A、
§9 E3 流水线）：

  * train_supervised: 几何级/整场留出 data_provider 驱动的监督训练；
    记录 train/val 曲线与逐 val 点评估；val 只用于 checkpoint 选优
    （val_selector 标量越小越好），冻结最优权重后再调 test_eval_fn 在
    test 集评估一次；落盘 checkpoint_best.pth 与 metrics_seed{n}.json
    （含协议快照、per-geom/per-case 指标、路由 diag）。
  * aggregate_seeds: 读取 exp_dir 下多个 metrics_seed*.json，对指定
    key 路径做跨 seed mean±std 汇总，并生成 LaTeX 行。

协议要点
--------
- 模型选择禁止接触 test：val_selector 返回标量在 val 几何/整场留出工况上
  计算，越小越好；训练结束后回载最优 checkpoint 才评估 test。
- 每个 seed 的落盘文件自带协议快照（split 版本、seed、epochs、lr、硬件、
  torch 版本、参数量），供 results/registry.json 登记复用（见 core.registry）。
- data_provider(epoch) 返回每 epoch 的迭代器，元素为
  (x_t [B,Cin,N]（或 [B,Cin,H,W]）, y_t [B,Cout,N])——均已归一化到
  训练域；loss 用 MSE。评估主指标（物理单位 RE 等）由调用方经
  test_eval_fn / val_selector 闭包完成（见 core.thermal_io / acoustic_io）。
"""

import glob
import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F

from ..utils.data_loader import set_seed
from ..utils.device import device_hardware_summary

SEED_FILE_RE = re.compile(r'^metrics_seed(\d+)\.json$')


# ---------------------------------------------------------------------------
# JSON/native 序列化工具
# ---------------------------------------------------------------------------
def _native(o):
    """把 numpy/torch 标量与数组递归转成 JSON 原生类型。"""
    if isinstance(o, dict):
        return {str(k): _native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_native(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        o = o.detach().cpu()
        return o.item() if o.ndim == 0 else o.tolist()
    return o


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def train_supervised(model, task, data_provider, epochs, lr, seed, device,
                     val_eval_every, val_selector, out_dir,
                     extra_loss_fn=None, router_reg=None, hard_after=0.1,
                     test_eval_fn=None, patience=None, save_router=True,
                     protocol_extra=None):
    """通用监督训练（单 seed）。

    Args:
        model: torch.nn.Module；forward(x_t) -> pred。若实现可选的
               set_route_hard(flag) / last_diag，供动态路由阶段使用。
        task: 'thermal' | 'acoustic'（仅记录进协议快照）。
        data_provider: callable(epoch) -> iterator of (x_t, y_t)，x_t/y_t
                       为已归一化、位于 device 的 torch tensor
                       （[B,Cin,N] 点云 / [B,Cin,H,W] 网格）。
        epochs: 训练 epoch 数。
        lr: Adam 学习率（恒定）。
        seed: 随机种子（set_seed 在本函数内调用，保证单 seed 可复现）。
        device: torch.device。
        val_eval_every: 每隔多少 epoch 评估一次 val（None = 只在末 epoch）。
        val_selector: callable(model) -> float，越小越好，仅用 val 集；
                      None 时不选优、以末 epoch 权重做最终评估。
        out_dir: results/<exp> 输出目录（含 checkpoint 与 metrics_seed{n}）。
        extra_loss_fn: callable(model, x_t, y_t, epoch) -> scalar tensor
                       或 None；叠加到监督 MSE（如 physics-informed 损失）。
        router_reg: callable(model) -> scalar tensor 或 None；路由正则项。
        hard_after: 前 hard_after 比例 epoch 为软路由阶段；此后若模型提供
                     set_route_hard(flag) 则切换为 hard 路由。
        test_eval_fn: callable(model) -> dict（任意 JSON 可序列化，建议含
                       'agg'/'rows'），在最优 checkpoint 上执行一次。
        patience: 连续多少次 val 评估无改善则早停；None = 训满 epochs。
        save_router: 若 model 暴露 last_diag（dict/tensor），且本参数为
                      True，则把最优 checkpoint 的路由 diag 写入 metrics。
        protocol_extra: dict，合并进 metrics 的 protocol 快照
                       （如 split_version、归一化统计、dataset 版本）。

    Returns:
        summary dict: {'out_dir', 'best_epoch', 'best_val_score',
                       'n_epochs', 'test': test_eval_fn 的返回}
    """
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    if epochs < 1:
        raise ValueError('epochs must be >= 1')

    set_seed(seed)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss_hist = []
    val_epochs, val_scores, val_points = [], [], []
    best_score = float('inf')
    best_epoch = 0
    best_path = os.path.join(out_dir, 'checkpoint_best.pth')
    stale_val_count = 0
    hard_switched = False
    first_hard_frac = float(hard_after)

    def _eval_val(ep):
        nonlocal best_score, best_epoch, stale_val_count
        model.eval()
        with torch.no_grad():
            score = float(val_selector(model))
        model.train()
        val_epochs.append(ep)
        val_scores.append(score)
        is_best = score < best_score
        if is_best:
            best_score = score
            best_epoch = ep
            torch.save({
                'epoch': ep, 'val_score': score,
                'model_state_dict': model.state_dict(),
                'task': task, 'seed': seed,
            }, best_path)
            stale_val_count = 0
        else:
            stale_val_count += 1
        val_points.append({'epoch': ep, 'score': score, 'best': is_best})
        return is_best

    for epoch in range(1, epochs + 1):
        model.train()
        frac = epoch / epochs
        if hasattr(model, 'cur_frac'):
            model.cur_frac = frac
        if (not hard_switched and frac >= first_hard_frac
                and hasattr(model, 'set_route_hard')):
            model.set_route_hard(True)
            hard_switched = True

        epoch_losses = []
        for x_t, y_t in data_provider(epoch):
            x_t = x_t.to(device)
            y_t = y_t.to(device)
            optimizer.zero_grad()
            pred = model(x_t)
            loss = F.mse_loss(pred, y_t)
            if extra_loss_fn is not None:
                extra = extra_loss_fn(model, x_t, y_t, epoch)
                if extra is not None:
                    loss = loss + extra
            if router_reg is not None:
                rl = router_reg(model)
                if rl is not None:
                    loss = loss + rl
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f'[seed {seed}] non-finite loss at epoch {epoch}: {loss.item()}')
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))
        train_loss_hist.append(
            float(np.mean(epoch_losses)) if epoch_losses else float('nan'))
        print(f'[train] seed={seed} epoch={epoch}/{epochs} '
              f'loss={train_loss_hist[-1]:.6f}', flush=True)

        do_val = (val_selector is not None and (
            (val_eval_every is not None and epoch % val_eval_every == 0)
            or epoch == epochs))
        if do_val:
            _eval_val(epoch)
        if (patience is not None and do_val and stale_val_count >= patience):
            print(f'[train] early stop at epoch {epoch} '
                  f'(no improvement for {patience} evals)', flush=True)
            break

    # 回载最优 checkpoint（若从未在 val 上更优且未保存，则保存末 epoch）
    if val_selector is not None and os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        best_epoch = int(ckpt['epoch'])
        best_score = float(ckpt['val_score'])
    else:
        best_epoch = len(train_loss_hist)
        best_score = None
        torch.save({'epoch': best_epoch, 'model_state_dict': model.state_dict(),
                    'task': task, 'seed': seed}, best_path)

    # test：仅一次，冻结最优权重
    test = None
    if test_eval_fn is not None:
        model.eval()
        with torch.no_grad():
            test = _native(test_eval_fn(model))
        print(f'[eval] seed={seed} best_epoch={best_epoch} '
              f'best_val={best_score}', flush=True)

    # 路由 diag（最优权重下）
    router_diag = None
    if save_router:
        diag = getattr(model, 'last_diag', None)
        if diag is not None:
            router_diag = _native(diag)

    protocol = {
        'task': task,
        'seed': int(seed),
        'epochs': int(epochs),
        'lr': float(lr),
        'val_eval_every': val_eval_every,
        'patience': patience,
        # 协议快照只记录**实际生效**的路由行为。既有的 'hard_after' 只是请求值，
        # 而 UnifiedNet 未实现 set_route_hard，硬的切换从未发生——若照抄请求值会
        # 让协议与实跑不符（见 docs/审计_协议合规核对.md P0-2）。
        'route_mode': ('hard after epoch fraction %.3f' % first_hard_frac
                       if hard_switched else 'soft only (no hard switch; the '
                       'model does not implement set_route_hard)'),
        'route_hard_switched': bool(hard_switched),
        'hard_after_requested': float(hard_after),
        'model': model.__class__.__name__,
        'model_params': int(n_params),
        'optimizer': 'Adam',
        'hardware': device_hardware_summary(),
    }
    if protocol_extra:
        protocol.update(protocol_extra)

    metrics = {
        'seed': int(seed),
        'task': task,
        'protocol': protocol,
        'curves': {'epochs': list(range(1, len(train_loss_hist) + 1)),
                   'train_loss': train_loss_hist,
                   'val_epochs': val_epochs,
                   'val_scores': val_scores},
        'val_points': val_points,
        'val_best': (None if best_score is None else
                     {'epoch': best_epoch, 'score': best_score}),
        'test': test,
        'router_diag': router_diag,
        'checkpoint_best': os.path.basename(best_path),
    }
    mpath = os.path.join(out_dir, f'metrics_seed{int(seed)}.json')
    with open(mpath, 'w', encoding='utf-8') as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)
        fh.write('\n')

    return {'out_dir': out_dir, 'best_epoch': best_epoch,
            'best_val_score': best_score, 'n_epochs': len(train_loss_hist),
            'test': test}


# ---------------------------------------------------------------------------
# 跨 seed 聚合
# ---------------------------------------------------------------------------
def _resolve(root, dotted):
    cur = root
    for part in dotted.split('.'):
        cur = cur[part]
    return cur


def _load_seed_files(exp_dir):
    """读取 exp_dir 下 metrics_seed*.json，按 seed 升序返回 [(seed, doc)]。"""
    found = []
    for fp in glob.glob(os.path.join(exp_dir, 'metrics_seed*.json')):
        m = SEED_FILE_RE.match(os.path.basename(fp))
        if m:
            with open(fp, 'r', encoding='utf-8') as fh:
                found.append((int(m.group(1)), json.load(fh)))
    found.sort(key=lambda t: t[0])
    return found


def aggregate_seeds(exp_dir, keys):
    """跨 seed 汇总多个 key 路径的指标。

    Args:
        exp_dir: 含 metrics_seed*.json 的目录。
        keys: list of dotted 路径，如 ['test.agg.re', 'test.agg.re_bd']。
              每个路径须解析为数字或 {field: 数字} 的 dict。

    Returns:
        dict {key: {n_seeds, mean, std, values}}：
            - 叶子为 dict 时 mean/std 也是 {field: ...} dict
            - 叶子为标量时 mean/std 为标量
        latex 行由 latex_row() 从其中一条生成。
    """
    seeds = _load_seed_files(exp_dir)
    if not seeds:
        raise FileNotFoundError(f'no metrics_seed*.json under {exp_dir}')
    gathered = {k: [] for k in keys}
    for _, doc in seeds:
        for k in keys:
            gathered[k].append(_resolve(doc, k))

    out = {}
    for k in keys:
        vals = gathered[k]
        if not vals:
            raise ValueError(f'key {k}: no values')
        n = len(vals)
        first = vals[0]
        if isinstance(first, dict):
            fields = list(first.keys())
            means = {}
            stds = {}
            per_field_n = {}
            for f in fields:
                arr = np.asarray(
                    [v[f] for v in vals if v[f] is not None],
                    dtype=np.float64)
                if arr.size:
                    means[f] = float(arr.mean())
                    stds[f] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
                else:
                    means[f] = None
                    stds[f] = None
                per_field_n[f] = int(arr.size)
            out[k] = {'n_seeds': n, 'mean': means, 'std': stds,
                      'values': vals, 'n_per_field': per_field_n}
        elif isinstance(first, (int, float, np.generic)):
            arr = np.asarray([float(v) for v in vals], dtype=np.float64)
            out[k] = {'n_seeds': n, 'mean': float(arr.mean()),
                      'std': (float(arr.std(ddof=1)) if arr.size > 1 else 0.0),
                      'values': vals}
        else:
            raise TypeError(f'key {k}: cannot aggregate type {type(first)}')
    return out


def latex_row(label, agg_entry, fields=None, precision=4):
    """由 aggregate_seeds 的某一条生成 LaTeX 表格行。

    Args:
        label: 行标签（模型/条件名）。
        agg_entry: aggregate_seeds()[key]，含 'mean'/'std'。
        fields: 需要输出的 field 顺序；None = 全部（按 agg 内顺序）。
        precision: 小数位。

    Returns:
        str：'label & m1$\\pm$s1 & m2$\\pm$s2 ...'
    """
    mean = agg_entry['mean']
    std = agg_entry['std']
    if isinstance(mean, dict):
        if fields is None:
            fields = list(mean.keys())
        cells = []
        for f in fields:
            if mean[f] is None:
                cells.append('--')
            else:
                cells.append(f'{mean[f]:.{precision}f}$\\pm${std[f]:.{precision}f}')
    else:
        if mean is None:
            cells = ['--']
        else:
            cells = [f'{mean:.{precision}f}$\\pm${std:.{precision}f}']
    return f'{label} & ' + ' & '.join(cells)
