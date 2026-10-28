"""exp_pinn_thermal.py —— P1 热场 NS-Boussinesq PINN（Protocol-A 子集先行）。

用途
----
严格按 docs/P1_热场NS_PINN设计.md 实现"物理域写残差"的 NS-Boussinesq PINN
（实验计划.md §6 Track P1；锚定 main.tex §5.2 的 PDE 公式），在本机核实的事实
与协议基础上跑 Protocol-A 子集：

  * 几何子集：Protocol-A train 100 取**前 --sub-train 个**（默认 10，
    id 0,1,3,4,5,6,7,8,9,10）训练；val 15 取**前 --sub-val 个**（默认 3，
    id 2,36,39）选优；**不接触 test 20**。
  * 主干：逐点 CoordMLP(hidden=512, layers=6) ~1.05M（容量对齐 FeatExtractNet，
    无全局池化 -> autograd.grad 只含逐点导数，残差语义干净；设计文档 §6）。
  * 归一化：坐标与输出 z-score 均只用子集 train 几何拟合（fit_stats，
    train-only）；网络输入/输出在归一化域。
  * 残差物理域写法 + 链式法则（设计 §4.3）：在损失内重建
    y_phys = y_mean + y_std*y_net，坐标导数为 autograd 对归一化坐标求导后再
    乘 1/s_x（一阶）、1/s_x^2（二阶）。绝不在 z 变量上直接照抄 NS 方程
    （exp35 失败主因 H1，见设计 §4.2）。
  * 白化（设计 §4.4）：L_pde = mean( (R_cont/w_cont)^2 + (R_momx/w_momx)^2
    + (R_momy/w_momy)^2 + (R_ener/w_ener)^2 )；w_* 冻结、不参与优化。默认给
    估计常数（--whiten-rms），或由纯监督 checkpoint 实测标定（--whiten-ckpt，
    读 CoordMLP 存档在子集 train 内部点上计算物理单位残差 RMS）。
  * BC（设计 §5.1）：每几何前 660 固壁点 u=v=0（数据精确为 0）+ T 用数据真值
    （**禁止 T=0 假设**——壁面 T 非零，见设计 §3）；BC 目标取归一化域数据目标
    （u/v 物理为 0 故归一化目标即 0 的 z 表示）。
  * collocation：每几何内部点(索引 660..4999)随机子采样 --n-colloc（默认 2000），
    全点 provider 保留原始点序，索引切片可靠（设计 §9 末尾注意）。
  * 损失与调度（设计 §5.1/§5.2）：L = L_obs + λ*(L_bc + L_pde)；
    前 --warmup（默认 100）epoch λ=0（纯监督暖机），此后 λ 线性 ramp
    0→1 over --ramp-epochs（默认 100）epoch。
  * 监控（设计 §5.3/§10）：每 --monitor-every（默认 25）epoch 记录四项残差
    物理单位 RMS 与逐项梯度范数（obs/bc/pde_cont/momx/momy/energy 各自对参数
    的 ||∇L_i||），并算 max_i||∇L_i||/||∇L_obs||；任何一步 loss/残差/梯度
    非有限即抛 FloatingPointError 终止（fail-fast）。
  * 模型选择：val 子集 3 几何 4 通道物理单位 RE 均值（越小越好）选优，冻结最优
    checkpoint 后在同一 val 子集上评估一次（runner 语义；test 不接触）。
  * 种子：seed in {0,1,2}，独立训练；3 seed metrics 齐备后自动聚合 mean±std
    并登记 results/registry.json（experiment_id = P1_pinn_10geom）。

输入输出
--------
  in  : --data HeatTransferData.npy、--splits configs/splits_thermal.json
  out : <out-dir>/seed{n}/checkpoint_best.pth + metrics_seed{n}.json
        （默认 <ws>/results/pinn_thermal/subset10/seed<n>，即调用方写
         --out-dir results/pinn_thermal/subset10 时最终路径）
        <out-dir>/summary.json（3 seed 齐时写聚合）
        results/registry.json 按 P1_pinn_10geom upsert（fcntl 文件锁）

运行方式
--------
  cd 返稿意见/返稿修订工作区
  # 单 seed 训练（GPU）
  srun --partition=compute --gres=gpu:1 python3 scripts/exp_pinn_thermal.py \
      --seed 0 --epochs 600 --warmup 100 \
      --out-dir results/pinn_thermal/subset10
  # 跨 seed 聚合 + 登记（CPU 可跑，3 seed 齐备时写 registry）
  python3 scripts/exp_pinn_thermal.py --aggregate-only \
      --out-dir results/pinn_thermal/subset10

约束：登录节点不得跑训练；脚本不改 src/ 现有文件；PDE 二阶图只在 collocation
点集上开（<=2000-4000 点/几何）。CPU 冒烟见 scripts/smoke_ablation_pinn_cpu.py。
"""

import argparse
import fcntl
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WS not in sys.path:
    sys.path.insert(0, WS)

from src.core import (  # noqa: E402
    append_registry,
    new_experiment_entry,
    registry,
    thermal_io,
)
from src.models.baselines import CoordMLP  # noqa: E402
from src.utils.data_loader import set_seed  # noqa: E402
from src.utils.device import device_hardware_summary, get_device  # noqa: E402

SPLIT_VERSION = 'splits_thermal_v1_protocolA'
EXPERIMENT_ID = 'P1_pinn_10geom'
FIELD_NAMES = ('u', 'v', 'p', 'T')
SEEDS = (0, 1, 2)
BOUNDARY_N = 660          # 每几何前 660 固壁点（thermal_io 约定一致）
POINTS_PER_GEOM = 5000
N_INTERIOR = POINTS_PER_GEOM - BOUNDARY_N  # 4340
REGISTRY_KEYS = ('re', 'mae', 'max_err', 're_bd', 'mae_bd', 'max_bd')

# 无量纲 Boussinesq 系数（main.tex §5.2 口径：rho=1, mu=0.01, kappa=1, cp=1,
# g_beta=1, T_ref=0；固定并在协议快照记录，不随几何变化）
PHYS_DEFAULT = {'rho': 1.0, 'mu': 0.01, 'kappa': 1.0, 'cp': 1.0,
                'g_beta': 1.0, 't_ref': 0.0}
# 白化分母默认估计值（物理单位量级，见设计 §4.4：cont~1e-1、压力梯度/浮力/
# 扩散~1e0）。正式表前用纯监督 checkpoint（--whiten-ckpt）实测标定覆盖。
WHITEN_DEFAULT = {'cont': 0.2, 'momx': 1.0, 'momy': 1.0, 'energy': 1.0}
WHITEN_KEYS = tuple(WHITEN_DEFAULT.keys())
GRAD_TERMS = ('obs', 'bc', 'pde_cont', 'pde_momx', 'pde_momy', 'pde_energy')


# ---------------------------------------------------------------------------
# 调度与 CLI
# ---------------------------------------------------------------------------
def ramp_weight(epoch, warmup, ramp_epochs):
    """PDE/BC 权重调度：epoch<=warmup 时 0；之后在 ramp_epochs 内线性 0->1。"""
    if epoch <= warmup:
        return 0.0
    k = epoch - warmup
    if k >= ramp_epochs:
        return 1.0
    return float(k) / max(1.0, float(ramp_epochs))


def parse_whiten_rms(text):
    """'cont=0.2,momx=1.0,...' -> dict；缺省键取 WHITEN_DEFAULT。"""
    out = dict(WHITEN_DEFAULT)
    for tok in text.split(','):
        tok = tok.strip()
        if not tok:
            continue
        k, _, v = tok.partition('=')
        k = k.strip()
        if k not in WHITEN_KEYS:
            raise ValueError(f'unknown whiten key {k!r} (keys={WHITEN_KEYS})')
        out[k] = float(v)
    return out


def parse_physics(text):
    out = dict(PHYS_DEFAULT)
    for tok in text.split(','):
        tok = tok.strip()
        if not tok:
            continue
        k, _, v = tok.partition('=')
        k = k.strip()
        if k not in out:
            raise ValueError(f'unknown physics key {k!r} (keys={sorted(out)})')
        out[k] = float(v)
    return out


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed', type=int, default=None,
                    help='训练随机种子（--aggregate-only 忽略；需在 {0,1,2}）')
    ap.add_argument('--epochs', type=int, default=600)
    ap.add_argument('--warmup', type=int, default=100,
                    help='纯监督暖机 epoch 数（λ=0）')
    ap.add_argument('--ramp-epochs', type=int, default=100,
                    help='warmup 后 λ 线性 ramp 0->1 的 epoch 数')
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--val-every', type=int, default=25,
                    help='每隔多少 epoch 在 val 子集上评估（None=只末 epoch）')
    ap.add_argument('--monitor-every', type=int, default=25,
                    help='每隔多少 epoch 记录残差 RMS 与逐项梯度范数')
    ap.add_argument('--batch-geoms', type=int, default=4,
                    help='每个 batch 的几何数（子集内几何级采样）')
    ap.add_argument('--n-colloc', type=int, default=2000,
                    help='每几何内部点(660..4999)随机子采样 collocation 点数')
    ap.add_argument('--sub-train', type=int, default=10,
                    help='Protocol-A train 取前 N 个几何做训练（<=100）')
    ap.add_argument('--sub-val', type=int, default=3,
                    help='Protocol-A val 取前 N 个几何做选优（<=15，不碰 test）')
    ap.add_argument('--hidden', type=int, default=512,
                    help='CoordMLP 隐藏宽（默认 512）')
    ap.add_argument('--num-layers', type=int, default=6,
                    help='CoordMLP 总线性层数（默认 6 -> ~1.05M）')
    ap.add_argument('--clip', type=float, default=None,
                    help='保险用梯度裁剪范数上限（None=不裁剪，正常路径不触发）')
    ap.add_argument('--whiten-rms', type=str, default=None,
                    help="白化分母 'cont=..,momx=..,momy=..,energy=..'"
                         '（默认常数估计，见模块 docstring）')
    ap.add_argument('--whiten-ckpt', default=None,
                    help='纯监督 CoordMLP checkpoint 路径；给定时在子集 train '
                         '内部点实测标定白化 RMS（覆盖 --whiten-rms）')
    ap.add_argument('--physics', type=str, default=None,
                    help="物理系数 'rho=..,mu=..,kappa=..,cp=..,g_beta=..,t_ref=..'"
                         '（默认 1,0.01,1,1,1,0，main.tex §5.2 口径）')
    ap.add_argument('--data', default=None, help='HeatTransferData.npy 路径')
    ap.add_argument('--splits', default=None,
                    help='configs/splits_thermal.json 路径')
    ap.add_argument('--out-dir', default=None,
                    help='输出根目录（默认 <ws>/results/pinn_thermal/subset10；'
                         '每 seed 实际输出 = <out-dir>/seed{n}/）')
    ap.add_argument('--device', default='auto', help='auto/cuda/cpu')
    ap.add_argument('--patience', type=int, default=None,
                    help='val 早停（None=训满 --epochs）')
    ap.add_argument('--aggregate-only', action='store_true',
                    help='只做跨 seed 聚合 + registry 登记，不训练')
    return ap.parse_args()


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
    out_dir = args.out_dir or os.path.join(ws, 'results', 'pinn_thermal',
                                           'subset10')
    os.makedirs(out_dir, exist_ok=True)
    return data, splits, out_dir


# ---------------------------------------------------------------------------
# 数据准备（Protocol-A 子集 + train-only 统计）
# ---------------------------------------------------------------------------
def prepare(args):
    """加载数据与 Protocol-A 子集，返回 (X, Y, stats, tr_ids, va_ids, device)。

    tr_ids/va_ids 为 Protocol-A train/val 列表的前 --sub-train/--sub-val 个
    （保持 splits_thermal.json 内顺序）；stats 只用 tr_ids 拟合（train-only）。
    """
    data, splits, out_dir = resolve_paths(args)
    args.data, args.splits, args.out_dir = data, splits, out_dir

    if args.seed is None:
        raise SystemExit('--seed 必填（--aggregate-only 除外）')
    if args.seed not in SEEDS:
        raise SystemExit(f'--seed 需在 {SEEDS} 中（当前 {args.seed}）')
    if args.sub_train < 1 or args.sub_train > 100:
        raise SystemExit('--sub-train 需在 1..100')
    if args.sub_val < 1 or args.sub_val > 15:
        raise SystemExit('--sub-val 需在 1..15')

    with open(splits, 'r', encoding='utf-8') as fh:
        splits_cfg = json.load(fh)
    pa = splits_cfg['protocol_A']
    train_all, val_all = pa['train'], pa['val']
    assert [len(train_all), len(val_all), len(pa['test'])] == [100, 15, 20]
    assert splits_cfg['split_seed'] == 123 and splits_cfg['n_geometries'] == 135
    tr_ids = list(train_all[:args.sub_train])
    va_ids = list(val_all[:args.sub_val])

    X, Y = thermal_io.load_thermal(data)
    stats = thermal_io.fit_stats(X, Y, geom_ids=tr_ids)
    device, _backend = get_device(args.device)
    return X, Y, stats, tr_ids, va_ids, device


# ---------------------------------------------------------------------------
# 物理残差（物理域 + 链式法则缩放；设计 §4.3）
# ---------------------------------------------------------------------------
def _physical_residuals(y_phys, xc, sx, phys):
    """在物理域写 NS-Boussinesq 四项残差（链式法则由 autograd + sx 完成）。

    Args:
        y_phys: [B,4,Nc] 已反归一化的物理量 (u,v,p,T)。
        xc: [B,2,Nc] **归一化**坐标（requires_grad=True leaf）。
        sx: [2] tensor，坐标各通道 std（物理导数 = 归一化导数 / sx；二阶 /sx^2）。
        phys: dict(rho, mu, kappa, cp, g_beta, t_ref)。

    Returns:
        dict: {'cont','momx','momy','energy'} 各为 [B,1,Nc] 物理单位残差 tensor。
    """
    rho = phys['rho']
    mu = phys['mu']
    kappa = phys['kappa']
    cp = phys['cp']
    g_beta = phys['g_beta']
    t_ref = phys['t_ref']
    sx2 = sx.view(1, 2, 1)

    def _first(f):
        """物理域一阶梯度：autograd 对归一化坐标求导后逐通道除以 sx。"""
        g = torch.autograd.grad(
            f, xc, grad_outputs=torch.ones_like(f),
            create_graph=True, retain_graph=True)[0]
        return g / sx2  # [B,2,Nc]

    def _second(frow_phys):
        """对已含 1/sx 的一阶物理导数再求物理域梯度（完成二阶链式 /sx^2）。"""
        g = torch.autograd.grad(
            frow_phys, xc, grad_outputs=torch.ones_like(frow_phys),
            create_graph=True, retain_graph=True)[0]
        return g / sx2  # [B,2,Nc]

    u = y_phys[:, 0:1]
    v = y_phys[:, 1:2]
    p = y_phys[:, 2:3]
    T = y_phys[:, 3:4]

    du = _first(u)
    u_x, u_y = du[:, 0:1], du[:, 1:2]
    dv = _first(v)
    v_x, v_y = dv[:, 0:1], dv[:, 1:2]
    dp = _first(p)
    p_x, p_y = dp[:, 0:1], dp[:, 1:2]
    dT = _first(T)
    T_x, T_y = dT[:, 0:1], dT[:, 1:2]

    u_xx = _second(u_x)[:, 0:1]
    u_yy = _second(u_y)[:, 1:2]
    v_xx = _second(v_x)[:, 0:1]
    v_yy = _second(v_y)[:, 1:2]
    T_xx = _second(T_x)[:, 0:1]
    T_yy = _second(T_y)[:, 1:2]

    return {
        'cont': u_x + v_y,
        'momx': (rho * (u * u_x + v * u_y) + p_x
                 - mu * (u_xx + u_yy)),
        'momy': (rho * (u * v_x + v * v_y) + p_y
                 - mu * (v_xx + v_yy)
                 - rho * g_beta * (T - t_ref)),
        'energy': (rho * cp * (u * T_x + v * T_y)
                   - kappa * (T_xx + T_yy)),
    }


def compute_loss_terms(model, xb, yb, y_mean_t, y_std_t, sx, whiten,
                       n_colloc, phys):
    """一个 (geometry-batch) 步的全部损失项（obs/bc/pde 各分项 + 残差 RMS）。

    Args:
        model: CoordMLP forward(x:[B,2,N])->[B,4,N]（归一化域输出）。
        xb/yb: [B,2/4,N] 归一化坐标/真值（全点 5000，点序=原始序）。
        y_mean_t/y_std_t: [4] tensor（输出通道 z-score 统计，用于物理重建）。
        sx: [2] tensor（坐标通道 std）。
        whiten: dict 四项白化分母（cont/momx/momy/energy）。
        n_colloc: 每几何内部 collocation 采样点数。
        phys: dict 物理系数。

    Returns:
        dict: obs/bc/pde_cont/pde_momx/pde_momy/pde_energy（scalar tensor），
              rms（四项物理单位残差 RMS float）。
    """
    B, Cin, N = xb.shape
    # 监督 + BC（一次整场前向；CoordMLP 逐点，列序对应原几何点序）
    pred = model(xb)  # [B,4,N]
    loss_obs = F.mse_loss(pred, yb)

    pb = pred[:, :, :BOUNDARY_N]
    yb_b = yb[:, :, :BOUNDARY_N]
    pb_uvt = torch.cat([pb[:, 0:1], pb[:, 1:2], pb[:, 3:4]], dim=1)
    yb_uvt = torch.cat([yb_b[:, 0:1], yb_b[:, 1:2], yb_b[:, 3:4]], dim=1)
    loss_bc = F.mse_loss(pb_uvt, yb_uvt)  # u,v 归一化目标~0；T 数据真值

    # collocation：内部点(660..4999)随机子采样（全点 provider 保留点序，索引可靠）
    n_int = N - BOUNDARY_N
    n_c = min(int(n_colloc), n_int)
    idx_int = BOUNDARY_N + torch.randperm(n_int, device=xb.device)[:n_c]
    xc = xb[:, :, idx_int].contiguous()
    xc.requires_grad_(True)

    yc_net = model(xc)  # [B,4,n_c] 归一化域
    y_phys = (y_mean_t.view(1, 4, 1)
              + y_std_t.view(1, 4, 1) * yc_net)  # 物理重建
    R = _physical_residuals(y_phys, xc, sx, phys)

    rms = {}
    pde_terms = {}
    for key, w in whiten.items():
        r_i = R[key]
        rms[key] = float(torch.sqrt((r_i ** 2).mean()).detach().cpu())
        pde_terms[key] = ((r_i / w) ** 2).mean()
    loss_pde_cont = pde_terms['cont']
    loss_pde_momx = pde_terms['momx']
    loss_pde_momy = pde_terms['momy']
    loss_pde_energy = pde_terms['energy']

    return {
        'obs': loss_obs,
        'bc': loss_bc,
        'pde_cont': loss_pde_cont,
        'pde_momx': loss_pde_momx,
        'pde_momy': loss_pde_momy,
        'pde_energy': loss_pde_energy,
        'rms': rms,
    }


def _pde_total(terms):
    return (terms['pde_cont'] + terms['pde_momx']
            + terms['pde_momy'] + terms['pde_energy'])


def _param_grad_norm(model, term):
    """loss 分项 term 对全部可训练参数梯度的 L2 范数 ||∇_θ L_i||₂。

    部分分项（如连续性残差仅含 u/v 通道）对某些参数（如 p/T 输出 bias）无
    依赖，autograd 返回 None —— allow_unused=True 并跳过这些项。
    """
    params = [p for p in model.parameters() if p.requires_grad]
    gs = torch.autograd.grad(term, params, retain_graph=True,
                             allow_unused=True)
    total = sum((g.reshape(-1) ** 2).sum() for g in gs if g is not None)
    return float(torch.sqrt(total).detach().cpu())


def compute_monitor(model, xb, yb, y_mean_t, y_std_t, sx, whiten,
                    n_colloc, phys):
    """监控步：重算各损失分项 + 四项残差 RMS + 逐项梯度范数（fail-fast 用）。"""
    terms = compute_loss_terms(model, xb, yb, y_mean_t, y_std_t, sx, whiten,
                               n_colloc, phys)
    grad_norm = {k: _param_grad_norm(model, terms[k]) for k in GRAD_TERMS}
    return {'terms': terms, 'grad_norm': grad_norm}


# ---------------------------------------------------------------------------
# 白化标定（纯监督 checkpoint 实测；设计 §4.4.1）
# ---------------------------------------------------------------------------
def calibrate_whiten_rms(ckpt_path, X, Y, stats, geom_ids, device,
                         n_colloc, phys, seed=0):
    """用收敛纯监督 CoordMLP 在 geom_ids（子集 train）内部点算物理单位残差 RMS。

    返回 whiten dict（四项 RMS），作为白化分母冻结；n_colloc 每几何内部随机
    采样点数。模型须与 --hidden/--num-layers 一致（默认 512/6）。
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'model_state_dict' in ckpt:
        ckpt = ckpt['model_state_dict']
    model = CoordMLP(input_channels=2, output_channels=4,
                     hidden_dim=512, num_layers=6).to(device)
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing or unexpected:
        raise ValueError(f'whiten-ckpt 权重不匹配: missing={missing} '
                         f'unexpected={unexpected}')
    model.eval()

    sx = torch.tensor(stats['x_std'], dtype=torch.float32, device=device)
    y_mean_t = torch.tensor(stats['y_mean'], dtype=torch.float32, device=device)
    y_std_t = torch.tensor(stats['y_std'], dtype=torch.float32, device=device)
    torch.manual_seed(seed)

    sums = {k: 0.0 for k in WHITEN_KEYS}
    counts = 0.0
    for g in geom_ids:
        x_phys = X[g:g + 1].astype(np.float32)
        xn = thermal_io.normalize(x_phys, stats['x_mean'], stats['x_std'])
        xt = torch.tensor(xn, dtype=torch.float32, device=device).permute(0, 2, 1)
        n_int = xt.shape[-1] - BOUNDARY_N
        n_c = min(int(n_colloc), n_int)
        idx_int = BOUNDARY_N + torch.randperm(n_int, device=device)[:n_c]
        xc = xt[:, :, idx_int].contiguous()
        xc.requires_grad_(True)
        with torch.enable_grad():
            y_phys = (y_mean_t.view(1, 4, 1)
                      + y_std_t.view(1, 4, 1) * model(xc))
            R = _physical_residuals(y_phys, xc, sx, phys)
            n_pts = R['cont'].numel()
            for k in WHITEN_KEYS:
                sums[k] += float((R[k] ** 2).sum().detach().cpu())
            counts += float(n_pts)
    return {k: float(np.sqrt(sums[k] / counts)) for k in WHITEN_KEYS}


# ---------------------------------------------------------------------------
# 单 seed 训练
# ---------------------------------------------------------------------------
def run_train_pinn(args, X, Y, stats, tr_ids, va_ids, device):
    """P1 单 seed 训练：obs 暖机 -> λ ramp 叠加 BC/PDE -> val 选优 -> 冻结评估。

    返回 summary dict（含 metrics 与 seed_dir 路径）。本函数是脚本核心，也供
    CPU 冒烟脚本直接调用（见 smoke_ablation_pinn_cpu.py）。
    """
    set_seed(args.seed)
    model = CoordMLP(input_channels=2, output_channels=4,
                     hidden_dim=args.hidden, num_layers=args.num_layers)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[model] CoordMLP(h={args.hidden}, layers={args.num_layers}) '
          f'| params={n_params:,} | device={device}', flush=True)

    y_mean_t = torch.tensor(stats['y_mean'], dtype=torch.float32, device=device)
    y_std_t = torch.tensor(stats['y_std'], dtype=torch.float32, device=device)
    sx = torch.tensor(stats['x_std'], dtype=torch.float32, device=device)
    whiten = dict(args.whiten)
    phys = dict(args.physics)

    # 归一化域子集张量（几何级 batch 采样用；点序保持原始序）
    Xn = thermal_io.normalize(X[tr_ids], stats['x_mean'], stats['x_std'])
    Yn = thermal_io.normalize(Y[tr_ids], stats['y_mean'], stats['y_std'])
    Xt = torch.tensor(Xn, dtype=torch.float32, device=device).permute(0, 2, 1)
    Yt = torch.tensor(Yn, dtype=torch.float32, device=device).permute(0, 2, 1)
    n_train = len(tr_ids)
    n_val = len(va_ids)
    print(f'[data] subset train={n_train} geoms (ids={tr_ids})\n'
          f'       subset val={n_val} geoms (ids={va_ids}) | '
          'test 20 不接触 | norm fit on train-subset only', flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.RandomState(args.seed)  # 几何级打乱（与训练 seed 一致）

    seed_dir = os.path.join(args.out_dir, f'seed{args.seed}')
    os.makedirs(seed_dir, exist_ok=True)
    best_path = os.path.join(seed_dir, 'checkpoint_best.pth')

    # 曲线 / 监控存储
    curve_epochs = []
    curve_obs, curve_bc, curve_pde, curve_total, curve_lam = [], [], [], [], []
    mon_epochs = []
    mon_rms = {k: [] for k in WHITEN_KEYS}
    mon_grad = {k: [] for k in GRAD_TERMS}
    mon_ratio = []
    val_epochs = []
    val_score_hist = []
    val_agg_re = {f: [] for f in FIELD_NAMES}
    val_rows = []  # 每 val epoch 的逐几何 rows（per-geom RE 曲线数据）

    best_val = float('inf')
    best_epoch = 0
    stale = 0
    last_xb = last_yb = None

    def _eval_val(ep):
        nonlocal best_val, best_epoch, stale
        model.eval()
        ev = thermal_io.eval_geometry_wise(
            lambda x: model(x), va_ids, X, Y, stats, device,
            chunk_points=args.eval_chunk if hasattr(args, 'eval_chunk') else None)
        model.train()
        score = float(np.mean(list(ev['agg']['re'].values())))
        val_epochs.append(ep)
        val_score_hist.append(score)
        for f in FIELD_NAMES:
            val_agg_re[f].append(ev['agg']['re'][f])
        val_rows.append({'epoch': ep, 'per_geom': ev['rows']})
        if score < best_val:
            best_val = score
            best_epoch = ep
            torch.save({'epoch': ep, 'val_score': score, 'seed': args.seed,
                        'model_state_dict': model.state_dict()}, best_path)
            stale = 0
        else:
            stale += 1
        return score

    t_start = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        lam = ramp_weight(epoch, args.warmup, args.ramp_epochs)
        order = rng.permutation(n_train)
        ep_obs, ep_bc, ep_pde, ep_total = [], [], [], []
        for st in range(0, n_train, args.batch_geoms):
            ids = order[st:st + args.batch_geoms]
            xb = Xt[ids]
            yb = Yt[ids]
            terms = compute_loss_terms(
                model, xb, yb, y_mean_t, y_std_t, sx, whiten,
                args.n_colloc, phys)
            pde = _pde_total(terms)
            total = terms['obs'] + lam * (terms['bc'] + pde)
            if not torch.isfinite(total):
                raise FloatingPointError(
                    f'[seed {args.seed}] non-finite total loss at epoch '
                    f'{epoch}: {total.item()}')
            optimizer.zero_grad()
            total.backward()
            if args.clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            ep_obs.append(float(terms['obs'].detach()))
            ep_bc.append(float(terms['bc'].detach()))
            ep_pde.append(float(pde.detach()))
            ep_total.append(float(total.detach()))
            last_xb, last_yb = xb, yb
        curve_epochs.append(epoch)
        curve_obs.append(float(np.mean(ep_obs)))
        curve_bc.append(float(np.mean(ep_bc)))
        curve_pde.append(float(np.mean(ep_pde)))
        curve_total.append(float(np.mean(ep_total)))
        curve_lam.append(lam)

        # 监控：残差 RMS + 逐项梯度范数（每 monitor_every 或末 epoch）
        do_mon = (epoch % args.monitor_every == 0) or epoch == args.epochs
        if do_mon and last_xb is not None:
            mon = compute_monitor(model, last_xb, last_yb, y_mean_t, y_std_t,
                                  sx, whiten, args.n_colloc, phys)
            # fail-fast：任何损失分项/残差 RMS/梯度范数非有限即终止
            for k in GRAD_TERMS:
                if not torch.isfinite(mon['terms'][k]).all():
                    raise FloatingPointError(
                        f'[seed {args.seed}] non-finite loss term {k} '
                        f'at epoch {epoch}')
            for k in WHITEN_KEYS:
                if not np.isfinite(mon['terms']['rms'][k]):
                    raise FloatingPointError(
                        f'[seed {args.seed}] non-finite residual RMS {k} '
                        f'at epoch {epoch}')
            for k in GRAD_TERMS:
                if not np.isfinite(mon['grad_norm'][k]):
                    raise FloatingPointError(
                        f'[seed {args.seed}] non-finite grad norm {k} '
                        f'at epoch {epoch}')
            mon_epochs.append(epoch)
            for k in WHITEN_KEYS:
                mon_rms[k].append(mon['terms']['rms'][k])
            for k in GRAD_TERMS:
                mon_grad[k].append(mon['grad_norm'][k])
            r_obs = mon['grad_norm']['obs']
            mon_ratio.append(max(mon['grad_norm'][k] / r_obs
                                 for k in ('pde_cont', 'pde_momx',
                                           'pde_momy', 'pde_energy'))
                             if r_obs > 0 else float('inf'))
            print(f'[mon] seed={args.seed} ep={epoch} '
                  f'rms=' + ' '.join(f'{k}:{mon["terms"]["rms"][k]:.3e}'
                                     for k in WHITEN_KEYS)
                  + f' | grad-ratio={mon_ratio[-1]:.2f} | lam={lam:.2f}',
                  flush=True)

        # val 选优（不与 test 接触）
        do_val = (args.val_every is not None and (
            epoch % args.val_every == 0 or epoch == args.epochs))
        if do_val:
            vs = _eval_val(epoch)
            print(f'[val] seed={args.seed} ep={epoch} val_mean_RE={vs:.6f} '
                  f'best={best_val:.6f}@{best_epoch}', flush=True)
        if (args.patience is not None and do_val and stale >= args.patience):
            print(f'[train] early stop at epoch {epoch} '
                  f'(no improvement for {args.patience} evals)', flush=True)
            break
    wall_s = time.monotonic() - t_start
    n_ep_done = len(curve_epochs)

    # 回载最优 checkpoint 后在 val 子集冻结评估一次（runner 语义；test 不接触）
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        best_epoch = int(ckpt['epoch'])
        best_val = float(ckpt['val_score'])
    model.eval()
    ev_final = thermal_io.eval_geometry_wise(
        lambda x: model(x), va_ids, X, Y, stats, device,
        chunk_points=getattr(args, 'eval_chunk', None))
    test_metrics = {'n_geom': ev_final['n_geom'], 'agg': ev_final['agg'],
                    'rows': ev_final['rows'],
                    'evaluation_set': ('Protocol-A val subset '
                                       f'(first {n_val} val geoms); '
                                       'test 20 不接触')}

    # 协议快照 + metrics 落盘
    protocol = {
        'task': 'thermal_pinn_ns_boussinesq',
        'experiment_id': EXPERIMENT_ID,
        'seed': int(args.seed),
        'split_version': SPLIT_VERSION,
        'split': (f'Protocol-A subset train{len(tr_ids)}/val{len(va_ids)} '
                  '(前 N 几何；test 20 不接触)'),
        'split_seed': 123,
        'subset': {'n_train_geom': n_train, 'n_val_geom': n_val,
                   'train_ids': tr_ids, 'val_ids': va_ids},
        'norm': f'train-subset-only z-score (fit on {len(tr_ids)} geoms)',
        'backbone': 'CoordMLP(hidden=%d, layers=%d)' % (args.hidden,
                                                        args.num_layers),
        'model_params': int(n_params),
        'physics': phys,
        'whiten_rms': whiten,
        'whiten_source': ('ckpt-calibrated' if args.whiten_ckpt
                          else 'const-estimate'),
        'whiten_ckpt': os.path.abspath(args.whiten_ckpt) if args.whiten_ckpt
                       else None,
        'epochs': int(args.epochs),
        'n_epochs_done': n_ep_done,
        'warmup': int(args.warmup),
        'ramp_epochs': int(args.ramp_epochs),
        'lr': float(args.lr),
        'val_every': args.val_every,
        'monitor_every': int(args.monitor_every),
        'batch_geoms': int(args.batch_geoms),
        'n_colloc': int(args.n_colloc),
        'patience': args.patience,
        'clip': args.clip,
        't_ref': phys['t_ref'],
        'optimizer': 'Adam',
        'collocation': 'interior indices 660..4999 random subsample',
        'bc': 'first 660 wall pts: u=v=0, T=data truth (no T=0 assumption)',
        's_per_epoch': round(wall_s / n_ep_done, 6) if n_ep_done else None,
        'wall_time_s': round(wall_s, 3),
        'hardware': device_hardware_summary(),
        'x_mean': np.asarray(stats['x_mean']).tolist(),
        'x_std': np.asarray(stats['x_std']).tolist(),
        'y_mean': np.asarray(stats['y_mean']).tolist(),
        'y_std': np.asarray(stats['y_std']).tolist(),
        'data': os.path.abspath(args.data),
        'splits_config': os.path.abspath(args.splits),
    }

    metrics = {
        'seed': int(args.seed),
        'task': 'thermal_pinn_ns_boussinesq',
        'protocol': protocol,
        'curves': {
            'epochs': curve_epochs,
            'obs_loss': curve_obs,
            'bc_loss': curve_bc,
            'pde_loss': curve_pde,
            'total_loss': curve_total,
            'lambda': curve_lam,
        },
        'monitor': {
            'epochs': mon_epochs,
            'rms': mon_rms,             # 四项物理单位残差 RMS 曲线
            'grad_norm': mon_grad,      # obs/bc/pde_* 逐项 ||∇L_i|| 曲线
            'ratio_max_over_obs': mon_ratio,
        },
        'val': {
            'epochs': val_epochs,
            'score_mean_re': val_score_hist,
            'agg_re': val_agg_re,
            'per_geom_rows': val_rows,  # per-geom RE(u/v/p/T) 曲线数据
        },
        'test': test_metrics,
        'best': {'epoch': best_epoch, 'val_mean_re': best_val},
        'checkpoint_best': os.path.basename(best_path),
    }
    mpath = os.path.join(seed_dir, f'metrics_seed{int(args.seed)}.json')
    with open(mpath, 'w', encoding='utf-8') as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)
        fh.write('\n')

    re_a = test_metrics['agg']['re']
    print(f'[train] seed={args.seed} done | epochs={n_ep_done} | '
          f'best_epoch={best_epoch} | best_val={best_val:.6f} | '
          f'wall={wall_s:.1f}s | val-subset RE(u/v/p/T)='
          + ' '.join(f'{f}:{re_a[f]:.4f}' for f in FIELD_NAMES), flush=True)
    return {'seed_dir': seed_dir, 'metrics_path': mpath, 'metrics': metrics,
            'best_epoch': best_epoch, 'best_val_score': best_val}


# ---------------------------------------------------------------------------
# 跨 seed 聚合 + registry 登记
# ---------------------------------------------------------------------------
def _load_all_seed_metrics(out_root):
    out = []
    for s in SEEDS:
        fp = os.path.join(out_root, f'seed{s}', f'metrics_seed{s}.json')
        if os.path.exists(fp):
            with open(fp, 'r', encoding='utf-8') as fh:
                out.append((s, json.load(fh)))
    return out


def _aggregate_field_values(docs, dotted_key):
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
    lock_path = registry_path + '.lock'
    with open(lock_path, 'w', encoding='utf-8') as lfh:
        fcntl.flock(lfh, fcntl.LOCK_EX)
        try:
            return append_registry(entry, registry_path=registry_path)
        finally:
            fcntl.flock(lfh, fcntl.LOCK_UN)


def aggregate_and_register(args):
    """跨 seed 聚合（mean±std）、写 summary.json 并登记 registry（id=P1_pinn_10geom）。

    仅当 out_root/seed{0,1,2} 的 metrics 全部齐备时写 registry；否则打印提示并
    跳过（例如干跑 epochs 少 / seed 数不足的场景）。
    """
    out_root = args.out_dir
    docs = _load_all_seed_metrics(out_root)
    if not docs:
        raise FileNotFoundError(f'{out_root} 下无 seed*/metrics_seed*.json')

    agg_all = {k: _aggregate_field_values(docs, k) for k in REGISTRY_KEYS}
    n_seeds = len(docs)
    print(f'\n=== aggregate {EXPERIMENT_ID} ({out_root}) | '
          f'seeds={[s for s, _ in docs]} | n={n_seeds} ===', flush=True)
    re_a = agg_all['re']
    for f in FIELD_NAMES:
        if re_a['mean'][f] is None:
            print(f'  RE_{f}: --')
        else:
            print(f'  RE_{f}: {re_a["mean"][f]:.4f} ± {re_a["std"][f]:.4f}')

    first = docs[0][1]['protocol']
    summary = {
        'experiment_id': EXPERIMENT_ID,
        'n_seeds': n_seeds,
        'seeds': [s for s, _ in docs],
        'backbone': first.get('backbone'),
        'model_params': first.get('model_params'),
        'physics': first.get('physics'),
        'whiten_rms': first.get('whiten_rms'),
        'whiten_source': first.get('whiten_source'),
        'split_version': first.get('split_version'),
        'split': first.get('split'),
        'epochs': first.get('epochs'),
        'warmup': first.get('warmup'),
        'ramp_epochs': first.get('ramp_epochs'),
        'lr': first.get('lr'),
        'agg': {k: {'mean': v['mean'], 'std': v['std'],
                    'n_per_field': v['n_per_field']}
                for k, v in agg_all.items()},
        'seed_files': [os.path.join(out_root, f'seed{s}',
                                    f'metrics_seed{s}.json') for s, _ in docs],
    }
    with open(os.path.join(out_root, 'summary.json'), 'w',
              encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
        fh.write('\n')

    if n_seeds < len(SEEDS):
        print(f'[skip registry] 只有 {n_seeds}/{len(SEEDS)} 个 seed 的 metrics，'
              '不做 registry 登记（非完整 P1 3-seed 运行）。')
        return

    protocol = {
        'experiment_id': EXPERIMENT_ID,
        'split_version': SPLIT_VERSION,
        'split': first.get('split'),
        'subset': first.get('subset'),
        'split_seed': 123,
        'norm': first.get('norm'),
        'backbone': first.get('backbone'),
        'physics': first.get('physics'),
        'whiten_rms': first.get('whiten_rms'),
        'whiten_source': first.get('whiten_source'),
        'epochs': first.get('epochs'),
        'warmup': first.get('warmup'),
        'ramp_epochs': first.get('ramp_epochs'),
        'lr': first.get('lr'),
        'val_every': first.get('val_every'),
        'monitor_every': first.get('monitor_every'),
        'batch_geoms': first.get('batch_geoms'),
        'n_colloc': first.get('n_colloc'),
        'seeds': list(SEEDS),
        's_per_epoch_mean': float(np.mean([
            d['protocol']['s_per_epoch'] for _, d in docs
            if d['protocol'].get('s_per_epoch') is not None])) or None,
        'hardware': first.get('hardware'),
    }
    entry = new_experiment_entry(
        experiment_id=EXPERIMENT_ID,
        protocol=protocol,
        seed=list(SEEDS),
        epochs=first.get('epochs'),
        lr=first.get('lr'),
        model=(f'CoordMLP(h{args.__dict__.get("hidden", 512)}...) + '
               'NS-Boussinesq PDE soft (物理域残差+白化)'),
        model_params=first.get('model_params'),
        hardware=first.get('hardware'),
        torch_version=first.get('hardware', {}).get('pytorch_version'),
        tool='exp_pinn_thermal.py',
        metrics={
            'file': os.path.join(out_root, 'summary.json'),
            'n_seeds': n_seeds,
            'test_agg_re_mean': re_a['mean'],
            'test_agg_re_std': re_a['std'],
            'test_agg_mae_mean': agg_all['mae']['mean'],
            'test_agg_max_err_mean': agg_all['max_err']['mean'],
            'test_agg_mae_bd_mean': agg_all['mae_bd']['mean'],
        },
    )
    registry_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(registry.__file__)))), 'results', 'registry.json')
    n_reg = _locked_append_registry(entry, registry_path=registry_path)
    print(f'[registry] upsert {EXPERIMENT_ID} -> {registry_path} '
          f'(total {n_reg} entries)', flush=True)


def main():
    args = parse_args()
    args.whiten = (parse_whiten_rms(args.whiten_rms)
                   if args.whiten_rms else dict(WHITEN_DEFAULT))
    args.physics = (parse_physics(args.physics)
                    if args.physics else dict(PHYS_DEFAULT))
    args.eval_chunk = None  # 5000 点整场一次即可

    if args.aggregate_only:
        data, splits, out_dir = resolve_paths(args)
        args.data, args.splits, args.out_dir = data, splits, out_dir
        aggregate_and_register(args)
        return

    X, Y, stats, tr_ids, va_ids, device = prepare(args)
    print(f'[env] workspace={WS}\ndevice={device} torch={torch.__version__} '
          f'cuda={torch.cuda.is_available()} data={args.data}', flush=True)
    if device.type == 'cpu':
        print('[warn] CPU 训练（仅 dry-run 小规模用；正式 P1 需 GPU）', flush=True)

    # 白化标定（可选：纯监督 checkpoint 实测；否则用给定常数）
    if args.whiten_ckpt:
        if not os.path.exists(args.whiten_ckpt):
            raise FileNotFoundError(f'--whiten-ckpt 不存在: {args.whiten_ckpt}')
        args.whiten = calibrate_whiten_rms(
            args.whiten_ckpt, X, Y, stats, tr_ids, device,
            args.n_colloc, args.physics, seed=args.seed)
        print('[whiten] 由纯监督 checkpoint 实测标定: ' +
              ', '.join(f'{k}={v:.3e}' for k, v in args.whiten.items()),
              flush=True)
    else:
        print('[whiten] 使用估计常数: ' +
              ', '.join(f'{k}={v:.3e}' for k, v in args.whiten.items()),
              flush=True)

    run_train_pinn(args, X, Y, stats, tr_ids, va_ids, device)
    aggregate_and_register(args)


if __name__ == '__main__':
    main()
