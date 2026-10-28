"""exp_r2_pinn_thermal.py —— R2-P1-MATCH-01：正确系数、边界分离的热场 NS-Boussinesq 三组对照。

用途
----
按《返稿修订 · 实验计划.md》§R2-P1-MATCH-01 实现最小正确补证（旧 P1 保留为系数失配诊断，
不得作为正确物理约束的效果结论）：

  * 主干 CoordMLP(hidden=512, layers=6)（逐点、无全局池化；与历史 P1 同容量）。
  * Protocol-A 前 10 个训练几何 / 前 3 个验证几何；原 test 20 只在训练冻结后计分一次。
  * 三个独立目标（objective）：
      supervised : L = L_obs
      bc         : L = L_obs + lambda(e) * L_bc
      bc_pde     : L = L_obs + lambda(e) * (L_bc + sum_j mean((R_j / w_j)^2))
    每组同 seed 的初始化、训练几何顺序与内部 collocation 采样流一致（三组都调用同一个
    真实 ``compute_loss_terms``，只是进入总损失的项不同）；因此墙钟时间不能当作训练成本比较。
  * lambda 调度：前 100 epoch 为 0（纯监督暖机），随后 100 epoch 线性升至 1，之后为 1。
  * collocation：每步对内部索引(660..4999)抽取**一个共享随机子集**，并把同一组索引应用到
    batch 内所有几何（这是真实 ``compute_loss_terms`` 的行为；不是每几何独立索引）。
    三组共享同一采样流；共享索引不改变三组对照身份。
  * 白化 w=(cont=0.2, momx=1, momy=1, energy=1) 为**预先固定的缩放常数**，明确不是实测
    RMS 白化，也不由数据标定。
  * 物理系数固定 rho=cp=g_beta=1、t_ref=0、mu=kappa=2*sqrt(2)*10^-2.5
    （=0.00894427190999916）；参考 L=2、DeltaT=1 时 nu=alpha=mu，
    Ra=g_beta*DeltaT*L^3/(nu*alpha)=1e5、Pr=nu/alpha=1（推导写入 snapshot）。
  * 归一化：仅用 10 个训练几何拟合 z-score（train-only），随后在物理（存储）域写残差；
    不加载任何旧 checkpoint、旧归一化统计或旧选中权重。
  * 精度：严格 FP32（关闭 TF32）；任何 loss 分项/残差 RMS/梯度范数/指标非有限立即
    FloatingPointError 终止，不回退、不重试。
  * 选择：仅 val 3 几何选 epoch；冻结 val 最佳权重后对 test 20 评估一次。不按 test 调参、
    不按 test 改路、不做 pressure gauge 平移。

真实函数复用
------------
残差、loss 项、监控与 lambda 调度直接复用旧的**真实实现**
``scripts/exp_pinn_thermal.py``（``_physical_residuals`` / ``compute_loss_terms`` /
``compute_monitor`` / ``_pde_total`` / ``ramp_weight``）；计分复用
``src/core/thermal_io.py`` 的 ``eval_geometry_wise``（物理单位 RE/MAE/max/边界 MAE）；
跨 seed 汇总复用其 ``_load_all_seed_metrics`` / ``_aggregate_field_values``；registry
写入复用其带文件锁的 ``_locked_append_registry``。本脚本不另造物理或评测体系。

输出（默认根 results/round2_pinn_thermal_matched）
-------------------------------------------------
  <root>/<objective>/seed<N>/protocol_snapshot.json   （训练开始前写）
  <root>/<objective>/seed<N>/checkpoint_best.pth      （val 最佳权重）
  <root>/<objective>/seed<N>/metrics_seed<N>.json     （曲线/监控/val/test）
  <root>/<objective>/seed<N>/report_seed<N>.md        （单 seed Markdown 报告）
  <root>/<objective>/summary.json                     （3 seed 齐时 mean±sample std）
  <root>/<objective>/report_<objective>.md            （3 seed 汇总报告）
  results/registry.json 独立条目 P1_R2_MATCH01_<objective>（3 seed 齐时登记）

用法
----
  python scripts/exp_r2_pinn_thermal.py --objective supervised --seed 0 --device cuda
  python scripts/exp_r2_pinn_thermal.py --objective bc         --seed 1 --device cuda
  python scripts/exp_r2_pinn_thermal.py --objective bc_pde     --seed 2 --device cuda
  # 9 次真实独立训练（3 objective x seed0..2）由 Codex 在 CUDA 上另行运行；本脚本不提交作业。

约束：不修改旧生产脚本/模型/配置/结果；不导入旧 checkpoint/norm；POSIX 运行（旧模块使用
fcntl，正式训练在 Linux 服务器执行）；本地 CPU 小点验证见 scripts/probe_r2_pinn_thermal.py。
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WS not in sys.path:
    sys.path.insert(0, WS)

from src.core import append_registry, new_experiment_entry, registry, thermal_io  # noqa: E402
from src.models.baselines import CoordMLP  # noqa: E402
from src.utils.data_loader import set_seed  # noqa: E402
from src.utils.device import device_hardware_summary, get_device  # noqa: E402

# 复用旧 P1 的真实物理残差 / loss / 监控 / 调度 / 汇总实现（不复制、不改写）
from scripts.exp_pinn_thermal import (  # noqa: E402
    BOUNDARY_N,
    FIELD_NAMES,
    GRAD_TERMS,
    SPLIT_VERSION,
    _aggregate_field_values,
    _locked_append_registry,
    _load_all_seed_metrics,
    _pde_total,
    _physical_residuals,
    compute_loss_terms,
    compute_monitor,
    ramp_weight,
)

EXPERIMENT_STEM = "P1_R2_MATCH01"
R2_TASK = "R2-P1-MATCH-01"
OBJECTIVES = ("supervised", "bc", "bc_pde")
SEEDS = (0, 1, 2)
SUB_TRAIN = 10
SUB_VAL = 3
N_GEOMETRIES = 135
POINTS_PER_GEOM = 5000
REGISTRY_KEYS = ("re", "mae", "max_err", "re_bd", "mae_bd", "max_bd")

# 正式协议（固定；调试参数可覆盖，但 snapshot 记录实际值与偏差）
FORMAL = {
    "epochs": 600,
    "lr": 0.001,
    "batch_geoms": 4,
    "val_every": 25,
    "monitor_every": 25,
    "n_colloc": 2000,
    "hidden": 512,
    "num_layers": 6,
    "warmup": 100,
    "ramp_epochs": 100,
}

# 正确物理系数（无量纲 Boussinesq；旧 P1 的 mu=0.01/kappa=1 是失配诊断，不再复用）
MU = KAPPA = 2.0 * math.sqrt(2.0) * 10.0 ** -2.5
PHYS = {"rho": 1.0, "mu": MU, "kappa": KAPPA, "cp": 1.0,
        "g_beta": 1.0, "t_ref": 0.0}
REF_L = 2.0       # 参考长度尺度（与 docs/P1_热场NS_PINN设计.md §4.2 一致）
REF_DT = 1.0      # 参考温差
# 固定缩放（不是实测 RMS 白化；与旧 WHITEN_DEFAULT 数值相同但语义冻结）
WHITEN = {"cont": 0.2, "momx": 1.0, "momy": 1.0, "energy": 1.0}
WHITEN_KEYS = ("cont", "momx", "momy", "energy")

# 每个 objective 实际进入总损失的项（其余项仍被计算，用于同流采样与监控）
OBJECTIVE_LOSS_TERMS = {
    "supervised": ("obs",),
    "bc": ("obs", "bc"),
    "bc_pde": ("obs", "bc", "pde_cont", "pde_momx", "pde_momy", "pde_energy"),
}

SOURCE_FILES = (
    "scripts/exp_r2_pinn_thermal.py",
    "scripts/exp_pinn_thermal.py",
    "src/core/thermal_io.py",
    "src/core/runner.py",
    "src/core/registry.py",
    "src/models/baselines/coordinate_mlp.py",
    "configs/splits_thermal.json",
)


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_state_dict(state):
    """确定性初始权重指纹（跨 objective 同 seed 可比）。"""
    h = hashlib.sha256()
    for key in sorted(state.keys()):
        arr = state[key].detach().cpu().numpy()
        h.update(key.encode("utf-8"))
        h.update(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
    return h.hexdigest()


def configure_precision():
    """严格 FP32：关闭 TF32；返回实际生效的精度标记。"""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "strict_fp32": True,
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


def derive_dimensionless(phys):
    """由系数与参考尺度推导 nu/alpha/Ra/Pr（写入 snapshot 供核对）。"""
    nu = phys["mu"] / phys["rho"]
    alpha = phys["kappa"] / (phys["rho"] * phys["cp"])
    ra = phys["g_beta"] * REF_DT * REF_L ** 3 / (nu * alpha)
    pr = nu / alpha
    return {"L": REF_L, "delta_T": REF_DT, "nu": nu, "alpha": alpha,
            "Ra": ra, "Pr": pr,
            "formula": ("nu=mu/rho; alpha=kappa/(rho*cp); "
                        "Ra=g_beta*delta_T*L^3/(nu*alpha); Pr=nu/alpha")}


def objective_total(terms, lam, objective):
    """三组总损失：三组计算同一批 terms，只有进入更新的项不同。"""
    if objective == "supervised":
        return terms["obs"]
    if objective == "bc":
        return terms["obs"] + lam * terms["bc"]
    if objective == "bc_pde":
        return terms["obs"] + lam * (terms["bc"] + _pde_total(terms))
    raise ValueError(f"unknown objective {objective!r}")


def experiment_id(objective):
    return f"{EXPERIMENT_STEM}_{objective}"


def resolve_paths(args):
    data = args.data
    if data is None:
        repo_root = os.path.dirname(os.path.dirname(WS))
        for cand in (os.path.join(WS, "data", "thermal", "HeatTransferData.npy"),
                     os.path.join(repo_root, "HeatTransferData.npy")):
            if os.path.exists(cand):
                data = cand
                break
    if data is None or not os.path.exists(data):
        raise FileNotFoundError("HeatTransferData.npy 未找到（--data 指定）")
    splits = os.path.join(WS, "configs", "splits_thermal.json")
    if not os.path.exists(splits):
        raise FileNotFoundError(f"splits 不存在: {splits}")
    out_root = (os.path.abspath(args.out_root) if args.out_root
                else os.path.join(WS, "results", "round2_pinn_thermal_matched"))
    return os.path.abspath(data), splits, out_root


def load_subsets(splits_path):
    with open(splits_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    pa = cfg["protocol_A"]
    train_all, val_all, test_all = pa["train"], pa["val"], pa["test"]
    if [len(train_all), len(val_all), len(test_all)] != [100, 15, 20]:
        raise SystemExit("[refuse] unexpected Protocol-A split sizes")
    if cfg.get("split_seed") != 123 or cfg.get("n_geometries") != N_GEOMETRIES:
        raise SystemExit("[refuse] unexpected split_seed/n_geometries")
    tr_ids = [int(i) for i in train_all[:SUB_TRAIN]]
    va_ids = [int(i) for i in val_all[:SUB_VAL]]
    te_ids = [int(i) for i in test_all]
    if len(set(tr_ids)) != SUB_TRAIN or len(set(va_ids)) != SUB_VAL:
        raise SystemExit("[refuse] duplicate ids in the subset")
    if set(tr_ids) & set(va_ids) or set(tr_ids) & set(te_ids) or set(va_ids) & set(te_ids):
        raise SystemExit("[refuse] train/val/test overlap")
    return cfg, tr_ids, va_ids, te_ids


def check_targets_clear(seed_dir, objective_dir, seed, objective):
    """关键文件存在即拒绝（不覆盖、不静默少跑）。"""
    planned = [
        os.path.join(seed_dir, "protocol_snapshot.json"),
        os.path.join(seed_dir, "checkpoint_best.pth"),
        os.path.join(seed_dir, f"metrics_seed{seed}.json"),
        os.path.join(seed_dir, f"report_seed{seed}.md"),
        os.path.join(objective_dir, "summary.json"),
        os.path.join(objective_dir, f"report_{objective}.md"),
    ]
    for path in planned:
        if os.path.exists(path):
            raise SystemExit(
                f"[refuse] target already exists: {path}; refusing to overwrite "
                "(pick a fresh --out-root or remove nothing by hand)")
    return True


def _write_json(path, doc):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


# ---------------------------------------------------------------------------
# snapshot（训练开始前写）
# ---------------------------------------------------------------------------
def build_snapshot(args, stats, tr_ids, va_ids, te_ids, n_params, device,
                   dev_type, precision, init_sha):
    deviations = {
        key: {"formal": FORMAL[key], "actual": getattr(args, key)}
        for key in FORMAL if getattr(args, key) != FORMAL[key]
    }
    return {
        "run_id": f"{experiment_id(args.objective)}_seed{args.seed}",
        "experiment_id": experiment_id(args.objective),
        "r2_task": R2_TASK,
        "objective": args.objective,
        "seed": int(args.seed),
        "task": "thermal_pinn_ns_boussinesq_matched_r2",
        "created_local": _now_local(),
        "intent": ("minimal matched correction of the historical P1 with the correct "
                   "physical coefficients; the old P1 remains a coefficient-mismatch "
                   "diagnostic and 'must not be worse' is NOT an acceptance condition"),
        "split_version": SPLIT_VERSION,
        "split_seed": 123,
        "n_geometries": N_GEOMETRIES,
        "subset": {
            "n_train_geom": len(tr_ids),
            "n_val_geom": len(va_ids),
            "n_test_geom": len(te_ids),
            "train_ids": tr_ids,
            "val_ids": va_ids,
            "test_ids": te_ids,
            "rule": ("Protocol-A train first 10 / val first 3; the original test 20 "
                     "is frozen and evaluated exactly once after training"),
        },
        "selection_policy": ("only the val-3 subset selects the epoch; test-20 is "
                             "never used for selection, tuning or routing; no early "
                             "stopping beyond the fixed epoch budget"),
        "norm": {
            "source": f"train-only z-score fit on the {len(tr_ids)} train-subset geoms",
            "x_mean": np.asarray(stats["x_mean"]).tolist(),
            "x_std": np.asarray(stats["x_std"]).tolist(),
            "y_mean": np.asarray(stats["y_mean"]).tolist(),
            "y_std": np.asarray(stats["y_std"]).tolist(),
        },
        "physics": dict(PHYS),
        "dimensionless": derive_dimensionless(PHYS),
        "physical_residuals": {
            "cont": "u_x + v_y",
            "momx": "rho*(u*u_x + v*u_y) + p_x - mu*(u_xx + u_yy)",
            "momy": ("rho*(u*v_x + v*v_y) + p_y - mu*(v_xx + v_yy) "
                     "- rho*g_beta*(T - t_ref)"),
            "energy": "rho*cp*(u*T_x + v*T_y) - kappa*(T_xx + T_yy)",
            "domain": ("physical/stored units; the network output is reconstructed "
                       "as y_phys = y_mean + y_std*y_net and autograd derivatives "
                       "w.r.t. normalized coordinates are rescaled by 1/sx (first) "
                       "and 1/sx^2 (second)"),
        },
        "whiten": dict(WHITEN),
        "whiten_note": ("w=(cont=0.2, momx=1, momy=1, energy=1) is a pre-fixed "
                        "scaling constant, NOT a measured RMS whitening; it is "
                        "frozen and never calibrated from data or checkpoints"),
        "lambda_schedule": {
            "warmup_epochs": int(args.warmup),
            "ramp_epochs": int(args.ramp_epochs),
            "formula": ("lambda(e)=0 for e<=warmup; (e-warmup)/ramp_epochs for "
                        "warmup<e<warmup+ramp_epochs; 1 afterwards"),
        },
        "loss_terms_actual": {
            "objective": args.objective,
            "terms_in_update": list(OBJECTIVE_LOSS_TERMS[args.objective]),
            "terms_computed": list(GRAD_TERMS),
            "note": ("all three objectives compute the same loss terms and only the "
                     "selected terms enter the update; wall-clock time is therefore "
                     "not a comparable training-cost measurement"),
        },
        "boundary": {
            "definition": f"first {BOUNDARY_N} points of each geometry are wall points",
            "bc_targets": ("u, v targets from the stored data (they are 0 by the "
                           "dataset) and T from the stored data; no pressure BC and "
                           "no hard-constraint envelope"),
            "collocation": (f"interior indices {BOUNDARY_N}..{POINTS_PER_GEOM - 1}; "
                            "compute_loss_terms draws one random permutation of the "
                            "interior indices per batch step and applies the same "
                            "n_colloc indices to every geometry in that batch "
                            "(shared indices, not per-geometry independent draws); "
                            "the input point order is not downsampled (full "
                            "5000-point provider)"),
        },
        "pressure": ("raw stored pressure is used as-is for observations/evaluation; "
                     "no pressure gauge shift or post-hoc translation is applied"),
        "model": {"name": "CoordMLP", "hidden": int(args.hidden),
                  "num_layers": int(args.num_layers), "params": int(n_params)},
        "training": {
            "epochs": int(args.epochs), "lr": float(args.lr),
            "batch_geoms": int(args.batch_geoms),
            "val_every": int(args.val_every),
            "monitor_every": int(args.monitor_every),
            "n_colloc": int(args.n_colloc),
            "warmup": int(args.warmup), "ramp_epochs": int(args.ramp_epochs),
            "optimizer": "Adam", "clip": args.clip, "patience": None,
        },
        "precision": dict(precision),
        "torch_version": torch.__version__,
        "device": str(device),
        "backend": dev_type,
        "hardware": device_hardware_summary(),
        "init_state_sha256": init_sha,
        "data": os.path.abspath(args.data),
        "data_sha256": sha256_file(os.path.abspath(args.data)),
        "splits_config": os.path.abspath(args.splits),
        "splits_sha256": sha256_file(os.path.abspath(args.splits)),
        "sources_sha256": {rel: sha256_file(os.path.join(WS, rel))
                           for rel in SOURCE_FILES},
        "not_ported": ("no checkpoint, selected weights, or normalization statistics "
                       "from any earlier run are loaded; only the real residual/loss "
                       "functions and core scoring are reused"),
        "formal_protocol": dict(FORMAL),
        "protocol_deviations": deviations,
        "is_formal_protocol": not deviations,
    }


def _now_local():
    from datetime import datetime
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# 单 seed 训练
# ---------------------------------------------------------------------------
def run_seed(args):
    if args.seed is None:
        raise SystemExit("--seed is required (0..2)")
    if args.seed not in SEEDS:
        raise SystemExit(f"--seed must be in {SEEDS}")
    if args.objective not in OBJECTIVES:
        raise SystemExit(f"--objective must be in {OBJECTIVES}")
    if args.aggregate_only:
        raise SystemExit("--aggregate-only must not be combined with training flags")

    data, splits, out_root = resolve_paths(args)
    args.data, args.splits = data, splits
    _cfg, tr_ids, va_ids, te_ids = load_subsets(splits)

    precision = configure_precision()
    device, dev_type = get_device(args.device)

    X, Y = thermal_io.load_thermal(data)
    stats = thermal_io.fit_stats(X, Y, geom_ids=tr_ids)  # train-only on the 10 geoms

    objective_dir = os.path.join(out_root, args.objective)
    seed_dir = os.path.join(objective_dir, f"seed{args.seed}")
    check_targets_clear(seed_dir, objective_dir, args.seed, args.objective)
    os.makedirs(seed_dir, exist_ok=True)

    set_seed(args.seed)  # same init for the three objectives at equal seed
    model = CoordMLP(input_channels=2, output_channels=4,
                     hidden_dim=args.hidden, num_layers=args.num_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    init_sha = sha256_state_dict(model.state_dict())
    print(f"[model] CoordMLP(h={args.hidden}, layers={args.num_layers}) "
          f"| params={n_params:,} | device={device} | init_sha={init_sha[:16]}",
          flush=True)
    print(f"[data] train={len(tr_ids)} geoms {tr_ids} | val={len(va_ids)} geoms "
          f"{va_ids} | test={len(te_ids)} geoms (frozen; scored after training) | "
          "norm fit on the train subset only", flush=True)

    snapshot = build_snapshot(args, stats, tr_ids, va_ids, te_ids, n_params,
                              device, dev_type, precision, init_sha)
    _write_json(os.path.join(seed_dir, "protocol_snapshot.json"), snapshot)
    if snapshot["protocol_deviations"]:
        print("[warn] non-formal protocol (debug overrides): "
              + json.dumps(snapshot["protocol_deviations"], ensure_ascii=False),
              flush=True)

    y_mean_t = torch.tensor(stats["y_mean"], dtype=torch.float32, device=device)
    y_std_t = torch.tensor(stats["y_std"], dtype=torch.float32, device=device)
    sx = torch.tensor(stats["x_std"], dtype=torch.float32, device=device)
    whiten = dict(WHITEN)
    phys = dict(PHYS)

    Xn = thermal_io.normalize(X[tr_ids], stats["x_mean"], stats["x_std"])
    Yn = thermal_io.normalize(Y[tr_ids], stats["y_mean"], stats["y_std"])
    Xt = torch.tensor(Xn, dtype=torch.float32, device=device).permute(0, 2, 1)
    Yt = torch.tensor(Yn, dtype=torch.float32, device=device).permute(0, 2, 1)
    n_train = len(tr_ids)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.RandomState(args.seed)  # geometry order (same across objectives)

    best_path = os.path.join(seed_dir, "checkpoint_best.pth")
    curves = {"epochs": [], "obs_loss": [], "bc_loss": [], "pde_loss": [],
              "total_loss": [], "lambda": []}
    monitor = {"epochs": [], "rms": {k: [] for k in WHITEN_KEYS},
               "grad_norm": {k: [] for k in GRAD_TERMS},
               "ratio_max_over_obs": []}
    val = {"epochs": [], "score_mean_re": [], "agg_re": {f: [] for f in FIELD_NAMES},
           "per_geom_rows": []}
    best_val = float("inf")
    best_epoch = 0
    last_xb = last_yb = None

    def eval_val(ep):
        nonlocal best_val, best_epoch
        model.eval()
        ev = thermal_io.eval_geometry_wise(
            lambda x: model(x), va_ids, X, Y, stats, device, chunk_points=None)
        model.train()
        score = float(np.mean(list(ev["agg"]["re"].values())))
        if not np.isfinite(score):
            raise FloatingPointError(f"[seed {args.seed}] non-finite val score "
                                     f"at epoch {ep}: {score}")
        val["epochs"].append(ep)
        val["score_mean_re"].append(score)
        for f in FIELD_NAMES:
            val["agg_re"][f].append(ev["agg"]["re"][f])
        val["per_geom_rows"].append({"epoch": ep, "per_geom": ev["rows"]})
        if score < best_val:
            best_val = score
            best_epoch = ep
            torch.save({"epoch": ep, "val_score": score, "seed": args.seed,
                        "objective": args.objective,
                        "protocol_deviations": snapshot["protocol_deviations"],
                        "model_state_dict": model.state_dict()}, best_path)
        return score

    t0 = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        lam = ramp_weight(epoch, args.warmup, args.ramp_epochs)
        order = rng.permutation(n_train)  # same geometry order across objectives
        ep_obs, ep_bc, ep_pde, ep_total = [], [], [], []
        for st in range(0, n_train, args.batch_geoms):
            ids = order[st:st + args.batch_geoms]
            xb, yb = Xt[ids], Yt[ids]
            # 三组都计算同一批真实 loss terms（含 BC/PDE），保证采样流一致
            terms = compute_loss_terms(model, xb, yb, y_mean_t, y_std_t, sx,
                                       whiten, args.n_colloc, phys)
            for key in GRAD_TERMS:
                if not torch.isfinite(terms[key]).all():
                    raise FloatingPointError(
                        f"[seed {args.seed}] non-finite loss term {key} "
                        f"at epoch {epoch}")
            for key in WHITEN_KEYS:
                if not np.isfinite(terms["rms"][key]):
                    raise FloatingPointError(
                        f"[seed {args.seed}] non-finite residual RMS {key} "
                        f"at epoch {epoch}")
            total = objective_total(terms, lam, args.objective)
            if not torch.isfinite(total):
                raise FloatingPointError(
                    f"[seed {args.seed}] non-finite total loss at epoch {epoch}")
            optimizer.zero_grad()
            total.backward()
            if args.clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            ep_obs.append(float(terms["obs"].detach()))
            ep_bc.append(float(terms["bc"].detach()))
            ep_pde.append(float(_pde_total(terms).detach()))
            ep_total.append(float(total.detach()))
            last_xb, last_yb = xb, yb
        curves["epochs"].append(epoch)
        curves["obs_loss"].append(float(np.mean(ep_obs)))
        curves["bc_loss"].append(float(np.mean(ep_bc)))
        curves["pde_loss"].append(float(np.mean(ep_pde)))
        curves["total_loss"].append(float(np.mean(ep_total)))
        curves["lambda"].append(float(lam))

        do_mon = (epoch % args.monitor_every == 0) or epoch == args.epochs
        if do_mon and last_xb is not None:
            mon = compute_monitor(model, last_xb, last_yb, y_mean_t, y_std_t,
                                  sx, whiten, args.n_colloc, phys)
            for key in GRAD_TERMS:
                if not torch.isfinite(mon["terms"][key]).all():
                    raise FloatingPointError(
                        f"[seed {args.seed}] non-finite monitor term {key} "
                        f"at epoch {epoch}")
                if not np.isfinite(mon["grad_norm"][key]):
                    raise FloatingPointError(
                        f"[seed {args.seed}] non-finite grad norm {key} "
                        f"at epoch {epoch}")
            for key in WHITEN_KEYS:
                if not np.isfinite(mon["terms"]["rms"][key]):
                    raise FloatingPointError(
                        f"[seed {args.seed}] non-finite monitor RMS {key} "
                        f"at epoch {epoch}")
            monitor["epochs"].append(epoch)
            for key in WHITEN_KEYS:
                monitor["rms"][key].append(mon["terms"]["rms"][key])
            for key in GRAD_TERMS:
                monitor["grad_norm"][key].append(mon["grad_norm"][key])
            r_obs = mon["grad_norm"]["obs"]
            pde_keys = ("pde_cont", "pde_momx", "pde_momy", "pde_energy")
            monitor["ratio_max_over_obs"].append(
                max(mon["grad_norm"][k] for k in pde_keys) / r_obs
                if r_obs > 0 else float("inf"))
            print(f"[mon] seed={args.seed} ep={epoch} lam={lam:.2f} "
                  "rms=" + " ".join(f"{k}:{mon['terms']['rms'][k]:.3e}"
                                    for k in WHITEN_KEYS)
                  + f" | grad-ratio={monitor['ratio_max_over_obs'][-1]:.2f}",
                  flush=True)

        do_val = (args.val_every is not None and (
            epoch % args.val_every == 0 or epoch == args.epochs))
        if do_val:
            score = eval_val(epoch)
            print(f"[val] seed={args.seed} ep={epoch} val_mean_RE={score:.6f} "
                  f"best={best_val:.6f}@{best_epoch}", flush=True)
    wall_s = time.monotonic() - t0
    n_ep_done = len(curves["epochs"])

    # 冻结 val 最佳权重 -> val 最终评估 + 原 test 20 一次性计分
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        best_epoch = int(ckpt["epoch"])
        best_val = float(ckpt["val_score"])
    model.eval()
    with torch.no_grad():
        val_final = thermal_io.eval_geometry_wise(
            lambda x: model(x), va_ids, X, Y, stats, device, chunk_points=None)
        test_final = thermal_io.eval_geometry_wise(
            lambda x: model(x), te_ids, X, Y, stats, device, chunk_points=None)
    test_metrics = {
        "n_geom": test_final["n_geom"],
        "agg": test_final["agg"],
        "rows": test_final["rows"],
        "evaluation_set": (f"original Protocol-A test 20 geoms (ids={te_ids}); "
                           "scored exactly once after training, never used for "
                           "selection or tuning"),
        "pressure_gauge": "raw stored pressure; no shift/translation",
    }
    for key in ("re", "mae", "max_err", "mae_bd"):
        for f, v in test_metrics["agg"][key].items():
            if v is not None and not np.isfinite(v):
                raise FloatingPointError(
                    f"[seed {args.seed}] non-finite test metric {key}/{f}: {v}")

    metrics = {
        "seed": int(args.seed),
        "objective": args.objective,
        "experiment_id": experiment_id(args.objective),
        "run_id": snapshot["run_id"],
        "task": snapshot["task"],
        "protocol": snapshot,
        "curves": curves,
        "monitor": monitor,
        "val": val,
        "val_best_metrics": {
            "n_geom": val_final["n_geom"],
            "agg": val_final["agg"],
            "rows": val_final["rows"],
            "evaluation_set": f"Protocol-A val subset (first {SUB_VAL} val geoms)",
        },
        "test": test_metrics,
        "best": {"epoch": int(best_epoch), "val_mean_re": float(best_val)},
        "checkpoint_best": "checkpoint_best.pth",
        "wall_time_s": round(wall_s, 3),
        "n_epochs_done": int(n_ep_done),
    }
    _write_json(os.path.join(seed_dir, f"metrics_seed{args.seed}.json"), metrics)
    report_path = os.path.join(seed_dir, f"report_seed{args.seed}.md")
    write_seed_report(report_path, snapshot, metrics)
    print(f"[done] objective={args.objective} seed={args.seed} "
          f"best_epoch={best_epoch} val_mean_RE={best_val:.6f} "
          f"test_RE(u/v/p/T)="
          + " ".join(f"{f}:{test_metrics['agg']['re'][f]:.4f}" for f in FIELD_NAMES),
          flush=True)
    print(f"[out] {seed_dir}", flush=True)
    return metrics


# ---------------------------------------------------------------------------
# 报告与跨 seed 汇总
# ---------------------------------------------------------------------------
def _fmt(v, nd=6):
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def write_seed_report(path, snapshot, metrics):
    agg = metrics["test"]["agg"]
    lines = [
        f"# R2-P1-MATCH-01 单 seed 报告：{snapshot['objective']} / seed {snapshot['seed']}",
        "",
        f"- run_id: `{snapshot['run_id']}`",
        f"- 协议: {snapshot['split_version']}，train {SUB_TRAIN} / val {SUB_VAL} / "
        f"test {len(snapshot['subset']['test_ids'])}（test 冻结后只计分一次）",
        f"- 主干: {snapshot['model']['name']}(hidden={snapshot['model']['hidden']}, "
        f"layers={snapshot['model']['num_layers']})，params={snapshot['model']['params']:,}",
        f"- 目标损失项（实际进入更新）: {', '.join(snapshot['loss_terms_actual']['terms_in_update'])}",
        f"- 系数: rho={snapshot['physics']['rho']}, mu=kappa={snapshot['physics']['mu']:.17g}, "
        f"cp={snapshot['physics']['cp']}, g_beta={snapshot['physics']['g_beta']}, "
        f"t_ref={snapshot['physics']['t_ref']}",
        f"- 无量纲推导: L={snapshot['dimensionless']['L']}, DeltaT={snapshot['dimensionless']['delta_T']}, "
        f"nu={snapshot['dimensionless']['nu']:.17g}, alpha={snapshot['dimensionless']['alpha']:.17g}, "
        f"Ra={snapshot['dimensionless']['Ra']:.6g}, Pr={snapshot['dimensionless']['Pr']:.6g}",
        f"- 白化 w（固定缩放，非实测 RMS）: {snapshot['whiten']}",
        f"- lambda: warmup={snapshot['lambda_schedule']['warmup_epochs']} ep, "
        f"ramp={snapshot['lambda_schedule']['ramp_epochs']} ep; 实际曲线见 metrics_seed*.json",
        f"- 精度: strict FP32={snapshot['precision']['strict_fp32']}, "
        f"TF32(matmul/cudnn)={snapshot['precision']['cuda_matmul_allow_tf32']}/"
        f"{snapshot['precision']['cudnn_allow_tf32']}; device={snapshot['backend']}",
        f"- epochs 实际执行: {metrics['n_epochs_done']}（正式协议 {FORMAL['epochs']}）",
        "",
        "## val（用于选 epoch；不接触 test）",
        "",
        f"- best epoch={metrics['best']['epoch']}，val 子集几何均值 RE={metrics['best']['val_mean_re']:.6f}",
        f"- val 子集各通道 RE: " + ", ".join(
            f"{f}={_fmt(metrics['val_best_metrics']['agg']['re'][f])}"
            for f in FIELD_NAMES),
        "",
        "## 冻结 test 20（训练后一次性计分）",
        "",
        "| channel | RE | MAE | max | boundary MAE |",
        "|---|---|---|---|---|",
    ]
    for f in FIELD_NAMES:
        lines.append(f"| {f} | {_fmt(agg['re'][f])} | {_fmt(agg['mae'][f])} | "
                     f"{_fmt(agg['max_err'][f])} | {_fmt(agg['mae_bd'][f])} |")
    lines += [
        "",
        "## 说明",
        "",
        "- 仅 val 3 几何选 epoch；test 20 在冻结 val 最佳权重后只评估一次，"
        "不用于调参、选路或改变权重。",
        "- 三组（supervised / bc / bc_pde）同 seed 使用相同初始化、训练几何顺序与"
        "内部 collocation 采样流；三组都计算相同的 loss terms，只有进入更新的项不同，"
        "因此墙钟时间不可作为训练成本比较。",
        "- 白化 w 为预先固定的缩放常数，不是实测 RMS；压力使用原始存储值，"
        "未做事后 gauge 平移。",
        "- 本实验为正确系数下的最小补证；不预设阳性，负结果同样保留。",
        "",
        f"- 文件: `protocol_snapshot.json`, `checkpoint_best.pth`, "
        f"`metrics_seed{snapshot['seed']}.json`, `report_seed{snapshot['seed']}.md`",
        "",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def _load_objective_docs(objective_dir):
    """Read only actual metrics files for the active seed set (never fake seeds)."""
    out = []
    for s in SEEDS:
        fp = os.path.join(objective_dir, f"seed{s}", f"metrics_seed{s}.json")
        if os.path.exists(fp):
            with open(fp, "r", encoding="utf-8") as fh:
                out.append((s, json.load(fh)))
    return out


def aggregate_and_register(out_root, objective):
    """三 seed 齐备时写 summary.json / Markdown 汇总并登记独立 registry ID。"""
    objective_dir = os.path.join(out_root, objective)
    docs = _load_objective_docs(objective_dir)
    present = [s for s, _ in docs]
    if len(docs) < len(SEEDS):
        print(f"[skip aggregate] {objective}: only seeds {present} present "
              f"({len(docs)}/{len(SEEDS)}); summary/registry not written",
              flush=True)
        return None
    agg_all = {key: _aggregate_field_values(docs, key) for key in REGISTRY_KEYS}
    first_proto = docs[0][1]["protocol"]
    reg_id = experiment_id(objective)
    summary = {
        "experiment_id": reg_id,
        "r2_task": R2_TASK,
        "objective": objective,
        "n_seeds": len(docs),
        "seeds": present,
        "split_version": SPLIT_VERSION,
        "subset": first_proto["subset"],
        "physics": first_proto["physics"],
        "dimensionless": first_proto["dimensionless"],
        "whiten": first_proto["whiten"],
        "whiten_note": first_proto["whiten_note"],
        "lambda_schedule": first_proto["lambda_schedule"],
        "loss_terms_actual": first_proto["loss_terms_actual"],
        "epochs": first_proto["training"]["epochs"],
        "model": first_proto["model"],
        "precision": first_proto["precision"],
        "test_policy": first_proto["selection_policy"],
        "agg": {key: {"mean": v["mean"], "std": v["std"],
                      "n_per_field": v["n_per_field"]}
                for key, v in agg_all.items()},
        "seed_files": [os.path.join(objective_dir, f"seed{s}",
                                    f"metrics_seed{s}.json") for s in present],
        "statistics_note": ("mean and sample std (ddof=1) across the 3 seeds; "
                            "per-channel independent; test scored once per seed on "
                            "the frozen weights"),
    }
    _write_json(os.path.join(objective_dir, "summary.json"), summary)
    write_objective_report(os.path.join(objective_dir, f"report_{objective}.md"),
                           summary)
    print(f"[aggregate] {objective}: seeds={present} -> summary.json", flush=True)

    re_a = agg_all["re"]
    protocol = {
        "r2_task": R2_TASK,
        "experiment_id": reg_id,
        "objective": objective,
        "split_version": SPLIT_VERSION,
        "split": first_proto["subset"]["rule"],
        "subset": first_proto["subset"],
        "split_seed": 123,
        "norm": first_proto["norm"]["source"],
        "physics": first_proto["physics"],
        "dimensionless": first_proto["dimensionless"],
        "whiten": first_proto["whiten"],
        "whiten_note": first_proto["whiten_note"],
        "lambda_schedule": first_proto["lambda_schedule"],
        "loss_terms_actual": first_proto["loss_terms_actual"],
        "model": first_proto["model"],
        "epochs": first_proto["training"]["epochs"],
        "lr": first_proto["training"]["lr"],
        "batch_geoms": first_proto["training"]["batch_geoms"],
        "val_every": first_proto["training"]["val_every"],
        "monitor_every": first_proto["training"]["monitor_every"],
        "n_colloc": first_proto["training"]["n_colloc"],
        "precision": first_proto["precision"],
        "seeds": present,
        "selection_policy": first_proto["selection_policy"],
        "pressure": first_proto["pressure"],
        "sources_sha256": first_proto["sources_sha256"],
        "data_sha256": first_proto["data_sha256"],
        "s_per_epoch_mean": float(np.mean([
            d["wall_time_s"] / d["protocol"]["training"]["epochs"]
            for _, d in docs if d["protocol"]["training"]["epochs"] > 0])),
        "hardware": first_proto["hardware"],
    }
    entry = new_experiment_entry(
        experiment_id=reg_id,
        protocol=protocol,
        seed=present,
        epochs=first_proto["training"]["epochs"],
        lr=first_proto["training"]["lr"],
        model=(f"CoordMLP(h{first_proto['model']['hidden']}, "
               f"L{first_proto['model']['num_layers']}) + matched NS-Boussinesq "
               f"soft constraint (objective={objective})"),
        metrics={
            "file": os.path.join(objective_dir, "summary.json"),
            "n_seeds": len(docs),
            "test_agg_re_mean": re_a["mean"],
            "test_agg_re_std": re_a["std"],
            "test_agg_mae_mean": agg_all["mae"]["mean"],
            "test_agg_max_err_mean": agg_all["max_err"]["mean"],
            "test_agg_mae_bd_mean": agg_all["mae_bd"]["mean"],
        },
        split_version=SPLIT_VERSION,
        model_params=first_proto["model"]["params"],
        hardware=first_proto["hardware"],
        torch_version=first_proto["torch_version"],
        tool="exp_r2_pinn_thermal.py",
        r2_task=R2_TASK,
        objective=objective,
        physical_coefficients=first_proto["physics"],
        dimensionless=first_proto["dimensionless"],
        loss_terms_actual=first_proto["loss_terms_actual"],
        selection_policy=first_proto["selection_policy"],
    )
    registry_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(registry.__file__)))), "results", "registry.json")
    registry_path = getattr(_ARGS, "registry_path", None) or registry_path
    n_reg = _locked_append_registry(entry, registry_path=registry_path)
    print(f"[registry] upsert {reg_id} -> {registry_path} "
          f"(total {n_reg} entries)", flush=True)
    return summary


def write_objective_report(path, summary):
    agg = summary["agg"]
    lines = [
        f"# R2-P1-MATCH-01 三 seed 汇总：{summary['objective']}",
        "",
        f"- registry ID: `{summary['experiment_id']}`",
        f"- seeds: {summary['seeds']}（mean ± sample std, ddof=1；每通道独立）",
        f"- 协议: {summary['split_version']}，train {SUB_TRAIN} / val {SUB_VAL} / "
        f"test {len(summary['subset']['test_ids'])}；test 冻结后每 seed 只计分一次",
        f"- 损失项（实际进入更新）: "
        f"{', '.join(summary['loss_terms_actual']['terms_in_update'])}",
        f"- 系数: mu=kappa={summary['physics']['mu']:.17g}，"
        f"Ra={summary['dimensionless']['Ra']:.6g}，Pr={summary['dimensionless']['Pr']:.6g}",
        f"- 白化 w（固定缩放，非实测 RMS）: {summary['whiten']}",
        f"- lambda: warmup {summary['lambda_schedule']['warmup_epochs']} ep，"
        f"ramp {summary['lambda_schedule']['ramp_epochs']} ep",
        "",
        "## 冻结 test 20：跨 3 seed 汇总",
        "",
        "| channel | RE | MAE | max | boundary MAE |",
        "|---|---|---|---|---|",
    ]
    for f in FIELD_NAMES:
        lines.append(
            f"| {f} | {_fmt(agg['re']['mean'][f])} ± {_fmt(agg['re']['std'][f])} | "
            f"{_fmt(agg['mae']['mean'][f])} ± {_fmt(agg['mae']['std'][f])} | "
            f"{_fmt(agg['max_err']['mean'][f])} ± {_fmt(agg['max_err']['std'][f])} | "
            f"{_fmt(agg['mae_bd']['mean'][f])} ± {_fmt(agg['mae_bd']['std'][f])} |")
    lines += [
        "",
        "## 说明",
        "",
        "- 三组（supervised / bc / bc_pde）同 seed 共用初始化、几何顺序与 collocation "
        "采样流；每组独立训练，不移植 checkpoint/权重/归一化。",
        "- 仅 val 3 几何选 epoch；test 20 冻结后一次性计分，不参与调参或选路。",
        "- 白化 w 为预先固定缩放，不是实测 RMS；压力原始存储值，无 gauge 平移。",
        "- 不预设阳性；负结果同样保留。墙钟时间不可作为训练成本比较（三组都计算同样 "
        "loss terms）。",
        "",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--objective", choices=OBJECTIVES, required=True,
                    help="supervised | bc | bc_pde")
    ap.add_argument("--seeds", default=None,
                    help="explicit seed set, e.g. 0,1,2,3,4; default historical (0,1,2)")
    ap.add_argument("--registry-path", default=None,
                    help="optional isolated registry JSON; default canonical results/registry.json")
    ap.add_argument("--seed", type=int, default=None, help="single seed within --seeds")
    ap.add_argument("--data", default=None, help="HeatTransferData.npy path")
    ap.add_argument("--out-root", default=None,
                    help="output root (default results/round2_pinn_thermal_matched)")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    # 显式调试参数（默认即正式协议；snapshot 记录实际值与偏差）
    ap.add_argument("--epochs", type=int, default=FORMAL["epochs"])
    ap.add_argument("--lr", type=float, default=FORMAL["lr"])
    ap.add_argument("--batch-geoms", type=int, default=FORMAL["batch_geoms"])
    ap.add_argument("--val-every", type=int, default=FORMAL["val_every"])
    ap.add_argument("--monitor-every", type=int, default=FORMAL["monitor_every"])
    ap.add_argument("--n-colloc", type=int, default=FORMAL["n_colloc"])
    ap.add_argument("--hidden", type=int, default=FORMAL["hidden"])
    ap.add_argument("--num-layers", type=int, default=FORMAL["num_layers"])
    ap.add_argument("--warmup", type=int, default=FORMAL["warmup"])
    ap.add_argument("--ramp-epochs", type=int, default=FORMAL["ramp_epochs"])
    ap.add_argument("--clip", type=float, default=None,
                    help="optional gradient clip; formal runs leave it None")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="only aggregate existing seeds 0..2 and register (no training)")
    return ap.parse_args()


_ARGS = None


def main():
    global SEEDS, _ARGS
    args = parse_args()
    _ARGS = args
    if args.seeds:
        vals = tuple(sorted(int(x) for x in str(args.seeds).split(',') if x.strip()))
        if not vals or any(v < 0 or v > 4 for v in vals):
            raise SystemExit('--seeds must be a non-empty subset of 0..4')
        SEEDS = vals
    if args.aggregate_only:
        _data, _splits, out_root = resolve_paths(args)
        aggregate_and_register(out_root, args.objective)
        return
    metrics = run_seed(args)
    _data, _splits, out_root = resolve_paths(args)
    aggregate_and_register(out_root, args.objective)
    return metrics


if __name__ == "__main__":
    main()
