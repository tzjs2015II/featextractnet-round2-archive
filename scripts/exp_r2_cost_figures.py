"""exp_r2_cost_figures.py —— R2 收口：当前实现成本 + 真实留出几何图（薄入口）。

范围（任务 R2-CLOSEOUT-COST-FIG-01）
------------------------------------
本脚本只做两件事，不训练、不调参、不改已有实验/模型/配置/结果：

  --task cost    在**实际 CUDA**上测当前实现的成本：参数量、严格 FP32 同步计时
                 的推理 mean/std、峰值显存、真实一次训练 step（前向+反向+Adam）
                 的 mean/std；热场与声学各取论文当前 token。所有形状/协议固定，
                 OOM 如实记录失败，**不缩小 shape 偷偷出数**。
  --task figures 只用 `results/round2_thermal_symmetric/feat_extract/seed0` 的真实
                 checkpoint/metrics + 训练统计，在 CPU 上对完整 5000 点做一次
                 symmetric 前向，按既有 test rows 的 u RE 选 typical/worst 两个
                 留出几何，出 4x3（GT/Pred/|Err| x u/v/p/T）图；明确标注选择来自
                 已有 test rows，不是独立验证。
  --task all     两者依次执行（提交脚本用 all，一个 30min 作业）。

复用与边界
----------
* 模型构造复用 scripts/exp_thermal.build_model / scripts.exp_acoustic.build_model，
  不复制网络、不改权重、不加载训练 checkpoint 做成本测量。
* 热场 token：feat_extract（pool_mode=symmetric）、mlp、pointnet、meshgraphnet
  （现有真实现 token）；声学 token：fourier_c / fourier_nc / unif_acous，均显式
  `fourier_layout='pointwise'`（不测旧错误布局）。
* 声学输入 batch=1、N=481*481=231361、cond=[0.6, 0.5]、输出 1 通道；热场
  batch=4、N=5000。随机输入 seed0，uniform[-1,1]，float32。
* 计时口径：warmup=2，推理 repeats=5，训练 step repeats=3；每次前/后
  torch.cuda.synchronize；峰值用 reset_peak_memory_stats，每个模型前清理引用。
  "net optimization time per epoch" 只按 25（热场）/6（声学）steps × 实测 step
  mean 折算**净优化时间**，明确不含数据加载、val、checkpoint 与指标评估，
  不是实际训练墙钟。
* 不 append 共享 registry：写 `registry_candidate.json` 供 Codex 合并。

输出
----
默认 out-root = <workspace>/results：
  cost    : <out-root>/round2_cost_figures/{protocol_snapshot.json, cost_results.json,
                                              registry_candidate.json}
  figures : <out-root>/figures/round2_thermal_current/
            fig_thermal_current_{typical,worst}_g<id>_seed0.{pdf,svg,png}
            figure_manifest.json（几何 id、每量 RE、路径、选择说明）
目标已存在即拒绝覆盖（exit 4），不覆盖旧图/旧结果。

CLI
---
  python scripts/exp_r2_cost_figures.py --task cost    --out-root results
  python scripts/exp_r2_cost_figures.py --task figures --out-root results
  python scripts/exp_r2_cost_figures.py --task all     --out-root results
（也可不传 --out-root，默认 <workspace>/results。）

约束：不安装依赖；本机无/CUDA 不可用或无 matplotlib 时明确跳过/报错，不做替代
测量；不 SSH/不提交；本脚本由 SBATCH 在服务器 CUDA+matplotlib 环境执行。
"""

import argparse
import gc
import hashlib
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

from src.utils.data_loader import set_seed  # noqa: E402
from src.utils.device import device_hardware_summary  # noqa: E402

# ---------------------------------------------------------------------------
# 冻结协议常量（与任务交接一致）
# ---------------------------------------------------------------------------
THERMAL_N = 5000
THERMAL_BATCH_GEOMS = 4
THERMAL_STEPS_PER_EPOCH = 25
ACOUSTIC_N = 481 * 481
ACOUSTIC_BATCH = 1
ACOUSTIC_COND = (0.6, 0.5)
ACOUSTIC_STEPS_PER_EPOCH = 6
WARMUP = 2
INFERENCE_REPEATS = 5
TRAIN_REPEATS = 3

THERMAL_TOKENS = (
    ("feat_extract", {"pool_context": "symmetric"}),
    ("mlp", {}),
    ("pointnet", {}),
    ("meshgraphnet", {}),
)
ACOUSTIC_TOKENS = (
    ("fourier_c", {"fourier_layout": "pointwise"}),
    ("fourier_nc", {"fourier_layout": "pointwise"}),
    ("unif_acous", {"fourier_layout": "pointwise"}),
)

COST_DIRNAME = "round2_cost_figures"
FIG_RELDIR = os.path.join("figures", "round2_thermal_current")
FIG_STEM = "fig_thermal_current"
FIELDS = ("u", "v", "p", "T")
SIGNED_FIELDS = ("u", "v", "p")
SEQ_FIELDS = ("T",)
CMAP_SIGNED = "coolwarm"
CMAP_SEQ = "viridis"
CMAP_ERR = "magma"
RE_TOL = 1e-6
METRICS_REL = os.path.join("results", "round2_thermal_symmetric", "feat_extract",
                           "seed0", "metrics_seed0.json")
CKPT_REL = os.path.join("results", "round2_thermal_symmetric", "feat_extract",
                        "seed0", "checkpoint_best.pth")

_ET = None
_EA = None


def _thermal_lib():
    """Lazy import: exp_thermal imports POSIX fcntl, defer until cost is needed."""
    global _ET
    if _ET is None:
        from scripts import exp_thermal as ET
        _ET = ET
    return _ET


def _acoustic_lib():
    global _EA
    if _EA is None:
        from scripts import exp_acoustic as EA
        _EA = EA
    return _EA


# ---------------------------------------------------------------------------
# 通用
# ---------------------------------------------------------------------------
def _require(cond, msg):
    if not cond:
        raise SystemExit("[refuse] " + msg)


def _now_local():
    from datetime import datetime
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path, doc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def resolve_paths(args):
    out_root = (os.path.abspath(args.out_root) if args.out_root
                else os.path.join(WS, "results"))
    cost_dir = os.path.join(out_root, COST_DIRNAME)
    fig_dir = os.path.join(out_root, FIG_RELDIR)
    return out_root, cost_dir, fig_dir


def find_data_path():
    repo_root = os.path.dirname(os.path.dirname(WS))
    for cand in (os.path.join(WS, "data", "thermal", "HeatTransferData.npy"),
                 os.path.join(repo_root, "HeatTransferData.npy")):
        if os.path.isfile(cand):
            return cand
    raise SystemExit("[fail] HeatTransferData.npy not found")


def strict_fp32():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "strict_fp32": True,
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _peak_bytes():
    return {
        "allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _timing_stats(samples_s):
    arr = np.asarray(samples_s, dtype=np.float64) * 1000.0
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "repeats": int(arr.size),
        "raw_ms": [float(v) for v in arr],
    }


# ---------------------------------------------------------------------------
# COST：真实 CUDA 测量
# ---------------------------------------------------------------------------
def _require_cuda():
    _require(torch.cuda.is_available(),
             "CUDA is not available; the cost task refuses any CPU substitution "
             "(explicitly skipped on this host)")
    _require(torch.get_default_dtype() == torch.float32,
             f"expected float32 default dtype, got {torch.get_default_dtype()}")


def _make_thermal_inputs(device):
    torch.manual_seed(0)
    x = torch.rand(THERMAL_BATCH_GEOMS, 2, THERMAL_N, device=device) * 2.0 - 1.0
    target = torch.zeros(THERMAL_BATCH_GEOMS, 4, THERMAL_N, device=device)
    return (x,), {}, target


def _make_acoustic_inputs(model, device):
    torch.manual_seed(0)
    x = torch.rand(ACOUSTIC_BATCH, 2, ACOUSTIC_N, device=device) * 2.0 - 1.0
    target = torch.zeros(ACOUSTIC_BATCH, 1, ACOUSTIC_N, device=device)
    cond = torch.tensor([list(ACOUSTIC_COND)], dtype=torch.float32, device=device)
    if hasattr(model, "set_cond"):
        model.set_cond(cond)
    return (x,), {}, target


def measure_inference(model, args, kwargs, device):
    model.eval()
    with torch.no_grad():
        for _ in range(WARMUP):
            model(*args, **kwargs)
        torch.cuda.synchronize()
        samples = []
        for _ in range(INFERENCE_REPEATS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(*args, **kwargs)
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - t0)
    stats = _timing_stats(samples)
    with torch.no_grad():
        out = model(*args, **kwargs)
    stats["output_shape"] = [int(v) for v in out.shape]
    stats["output_finite"] = bool(torch.isfinite(out).all())
    return stats


def measure_training_step(model, args, kwargs, target, device):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    def one_step():
        opt.zero_grad(set_to_none=True)
        out = model(*args, **kwargs)
        loss = F.mse_loss(out, target)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite training loss")
        loss.backward()
        opt.step()
        return loss

    for _ in range(WARMUP):
        one_step()
    torch.cuda.synchronize()
    samples = []
    for _ in range(TRAIN_REPEATS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = one_step()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    stats = _timing_stats(samples)
    stats["last_loss"] = float(loss.detach())
    stats["loss_finite"] = bool(torch.isfinite(loss.detach()))
    grads_ok = True
    for p in model.parameters():
        if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
            grads_ok = False
    stats["all_gradients_finite"] = grads_ok
    return stats


def _measure_model(task, token, kwargs, device):
    row = {"task": task, "token": token, "constructor_kwargs": dict(kwargs),
           "status": "pending"}
    model = None
    try:
        set_seed(0)
        if task == "thermal":
            model, label = _thermal_lib().build_model(token, **kwargs)
            args, call_kwargs, target = _make_thermal_inputs(device)
            steps_per_epoch = THERMAL_STEPS_PER_EPOCH
            shape = {"batch_geoms": THERMAL_BATCH_GEOMS, "N": THERMAL_N,
                     "input_channels": 2, "output_channels": 4}
        else:
            model, label = _acoustic_lib().build_model(token, **kwargs)
            model = model.to(device)
            args, call_kwargs, target = _make_acoustic_inputs(model, device)
            steps_per_epoch = ACOUSTIC_STEPS_PER_EPOCH
            shape = {"batch": ACOUSTIC_BATCH, "N": ACOUSTIC_N,
                     "cond": list(ACOUSTIC_COND), "input_channels": 2,
                     "output_channels": 1}
        model = model.to(device)
        row["model_label"] = label
        row["params"] = int(sum(p.numel() for p in model.parameters()))
        row["shape"] = shape

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        inf = measure_inference(model, args, call_kwargs, device)
        inf["peak"] = _peak_bytes()
        row["inference"] = inf
        row["inference"]["peak_gb"] = round(
            inf["peak"]["allocated_bytes"] / 1e9, 4)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        train = measure_training_step(model, args, call_kwargs, target, device)
        train["peak"] = _peak_bytes()
        train["steps_per_epoch"] = steps_per_epoch
        train["net_optimization_seconds_per_epoch"] = (
            train["mean_ms"] / 1000.0 * steps_per_epoch)
        train["net_optimization_note"] = (
            "only fwd+bwd+Adam steps x steps_per_epoch; excludes data loading, "
            "val, checkpoint and metric evaluation; not an actual training wall "
            "time and not measured with real training data")
        row["training_step"] = train
        row["status"] = "ok"
    except torch.cuda.OutOfMemoryError as exc:
        row["status"] = "error_oom"
        row["error"] = f"{type(exc).__name__}: {exc}"
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001 - one model must not kill the run
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
        torch.cuda.empty_cache()
    finally:
        del model
        _cleanup()
    return row


def _source_hashes():
    rels = ("scripts/exp_r2_cost_figures.py",
            "scripts/exp_thermal.py",
            "scripts/exp_acoustic.py",
            "src/core/thermal_io.py",
            "src/models/featextractnet.py",
            "src/models/baselines/coordinate_mlp.py",
            "src/models/baselines/pointnet.py",
            "src/models/baselines/meshgraphnet_style.py")
    out = {}
    for rel in rels:
        path = os.path.join(WS, rel)
        if os.path.isfile(path):
            out[rel] = sha256_file(path)
    return out


def run_cost(cost_dir):
    _require_cuda()
    check_cost_targets_clear(cost_dir)
    precision = strict_fp32()
    device = torch.device("cuda")
    env = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "device_total_memory_gb": round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 3),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "default_dtype": str(torch.get_default_dtype()),
        "precision": precision,
        "hardware": device_hardware_summary(),
    }
    snapshot = {
        "task": "cost",
        "r2_task": "R2-CLOSEOUT-COST-FIG-01",
        "created_local": _now_local(),
        "cost_only": True,
        "trains_to_convergence": False,
        "loads_training_checkpoints": False,
        "modifies_weights_or_results": False,
        "seed": 0,
        "random_input": "uniform[-1,1], float32, torch.manual_seed(0) per model",
        "warmup": WARMUP,
        "inference_repeats": INFERENCE_REPEATS,
        "training_step_repeats": TRAIN_REPEATS,
        "timing_note": ("CUDA events via perf_counter with torch.cuda.synchronize "
                        "before/after each repeat; warmup discarded; peak via "
                        "reset_peak_memory_stats with previous model references "
                        "deleted and cache emptied"),
        "net_optimization_note": (
            "net optimization time per epoch = measured training-step mean x "
            "steps/epoch (thermal 25, acoustic 6); excludes data loading, val, "
            "checkpoint and metric evaluation; not an actual training wall time"),
        "shapes": {
            "thermal": {"batch_geoms": THERMAL_BATCH_GEOMS, "N": THERMAL_N,
                        "steps_per_epoch": THERMAL_STEPS_PER_EPOCH},
            "acoustic": {"batch": ACOUSTIC_BATCH, "N": ACOUSTIC_N,
                         "cond": list(ACOUSTIC_COND),
                         "steps_per_epoch": ACOUSTIC_STEPS_PER_EPOCH},
        },
        "models": {
            "thermal": [{"token": t, "kwargs": k} for t, k in THERMAL_TOKENS],
            "acoustic": [{"token": t, "kwargs": k} for t, k in ACOUSTIC_TOKENS],
        },
        "precision_policy": "strict FP32 (TF32 disabled for matmul and cuDNN)",
        "env": env,
        "source_sha256": _source_hashes(),
    }
    _write_json(os.path.join(cost_dir, "protocol_snapshot.json"), snapshot)
    print(f"[cost] frozen snapshot -> {cost_dir}/protocol_snapshot.json",
          flush=True)

    rows = []
    for token, kwargs in THERMAL_TOKENS:
        row = _measure_model("thermal", token, kwargs, device)
        rows.append(row)
        print(f"[cost] thermal/{token}: status={row['status']} "
              f"params={row.get('params')} "
              f"inf_ms={row.get('inference', {}).get('mean_ms')} "
              f"train_ms={row.get('training_step', {}).get('mean_ms')}",
              flush=True)
    for token, kwargs in ACOUSTIC_TOKENS:
        row = _measure_model("acoustic", token, kwargs, device)
        rows.append(row)
        print(f"[cost] acoustic/{token}: status={row['status']} "
              f"params={row.get('params')} "
              f"inf_ms={row.get('inference', {}).get('mean_ms')} "
              f"train_ms={row.get('training_step', {}).get('mean_ms')}",
              flush=True)

    results = {
        "task": "cost",
        "r2_task": "R2-CLOSEOUT-COST-FIG-01",
        "created_local": _now_local(),
        "env": env,
        "conventions": {
            "seed": 0,
            "warmup": WARMUP,
            "inference_repeats": INFERENCE_REPEATS,
            "training_step_repeats": TRAIN_REPEATS,
            "rng_inputs": "uniform[-1,1]",
            "net_optimization_note": snapshot["net_optimization_note"],
            "no_flops": True,
            "no_scaling_sweep": True,
        },
        "rows": rows,
    }
    _write_json(os.path.join(cost_dir, "cost_results.json"), results)

    def candidate_entries():
        out = []
        for task in ("thermal", "acoustic"):
            task_rows = [r for r in rows if r["task"] == task]
            out.append({
                "id": f"R2_COST_{task}",
                "experiment_id": f"R2_COST_{task}",
                "type": "efficiency",
                "tool": os.path.basename(__file__),
                "r2_task": "R2-CLOSEOUT-COST-FIG-01",
                "protocol": {
                    "shapes": snapshot["shapes"][task],
                    "seed": 0,
                    "warmup": WARMUP,
                    "inference_repeats": INFERENCE_REPEATS,
                    "training_step_repeats": TRAIN_REPEATS,
                    "precision": precision,
                    "net_optimization_note": snapshot["net_optimization_note"],
                },
                "rows": task_rows,
                "file": os.path.join(
                    os.path.relpath(cost_dir, WS), "cost_results.json"),
                "hardware": env["hardware"],
                "note": ("REGISTRY CANDIDATE ONLY - not appended by this script; "
                         "Codex merges after acceptance"),
            })
        return out

    candidate = {
        "task": "registry_candidate",
        "r2_task": "R2-CLOSEOUT-COST-FIG-01",
        "appended": False,
        "note": ("Codex-only merge; this script never writes the shared "
                 "results/registry.json to avoid concurrent writers"),
        "entries": candidate_entries(),
    }
    _write_json(os.path.join(cost_dir, "registry_candidate.json"), candidate)
    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"[cost] done: {ok}/{len(rows)} models measured; results -> "
          f"{cost_dir}/cost_results.json", flush=True)
    return results


def check_cost_targets_clear(cost_dir):
    planned = ("protocol_snapshot.json", "cost_results.json",
               "registry_candidate.json")
    for name in planned:
        path = os.path.join(cost_dir, name)
        if os.path.exists(path):
            raise SystemExit(f"[refuse] cost target already exists: {path}; "
                             "refusing to overwrite accepted cost results")


# ---------------------------------------------------------------------------
# FIGURES：真实留出几何图
# ---------------------------------------------------------------------------
def select_holdout_geometries(metrics_doc):
    rows = metrics_doc.get("test", {}).get("rows") or []
    _require(bool(rows), "metrics JSON has no test.rows")
    us = [float(r["re_u"]) for r in rows]
    mean_u = float(np.mean(us))
    typical = min(rows, key=lambda r: abs(float(r["re_u"]) - mean_u))
    worst = max(rows, key=lambda r: float(r["re_u"]))
    return {
        "mean_re_u": mean_u,
        "picks": [
            {"tag": "typical", "row": typical,
             "rule": "closest re_u to the mean of existing test rows"},
            {"tag": "worst", "row": worst,
             "rule": "maximum re_u among existing test rows"},
        ],
        "selection_note": ("selected post hoc from the existing test rows of the "
                           "accepted metrics JSON; this is an illustrative "
                           "held-out geometry selection, NOT independent "
                           "validation"),
    }


def compute_figure_data(metrics_path, ckpt_path, data_path, tags=("typical",)):
    """CPU full-5000 forward on real symmetric checkpoint; returns per-tag arrays."""
    with open(metrics_path, "r", encoding="utf-8") as fh:
        metrics_doc = json.load(fh)
    selection = select_holdout_geometries(metrics_doc)
    picked = [p for p in selection["picks"] if p["tag"] in tags]
    proto = metrics_doc["protocol"]
    _require(proto.get("pool_context") == "symmetric",
             f"metrics pool_context={proto.get('pool_context')!r} != symmetric")
    stats = {k: np.asarray(proto[k], dtype=np.float32)
             for k in ("x_mean", "x_std", "y_mean", "y_std")}

    ET = _thermal_lib()
    from src.core import thermal_io
    model, label = ET.build_model("feat_extract", pool_context="symmetric")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    _require(isinstance(ckpt, dict) and "model_state_dict" in ckpt,
             f"checkpoint has no model_state_dict: {ckpt_path}")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    X, Y = thermal_io.load_thermal(data_path)
    _require(X.shape == (135, 5000, 2) and Y.shape == (135, 5000, 4),
             f"unexpected thermal data shapes X={X.shape} Y={Y.shape}")

    out = []
    for pick in picked:
        row = pick["row"]
        gid = int(row["geom_id"])
        xp = X[gid]
        yp = Y[gid].astype(np.float64)
        xn = thermal_io.normalize(xp, stats["x_mean"], stats["x_std"])
        xt = torch.from_numpy(np.ascontiguousarray(xn.T, dtype=np.float32))
        with torch.no_grad():
            pred_n = model(xt.unsqueeze(0))
        pred = thermal_io.denormalize(
            pred_n[0].T.cpu().numpy(), stats["y_mean"], stats["y_std"]
        ).astype(np.float64)
        re = {}
        err_vs_json = {}
        for j, field in enumerate(FIELDS):
            t = yp[:, j]
            p = pred[:, j]
            re[field] = float(np.linalg.norm(p - t) / np.linalg.norm(t))
            err_vs_json[field] = abs(re[field] - float(row[f"re_{field}"]))
        _require(all(v <= RE_TOL for v in err_vs_json.values()),
                 f"recomputed RE disagrees with metrics JSON for geometry "
                 f"{gid}: {err_vs_json}")
        out.append({"tag": pick["tag"], "row": row, "geom_id": gid,
                    "x": xp, "truth": yp, "pred": pred, "re": re,
                    "re_error_vs_json": err_vs_json})
    return {"metrics_doc": metrics_doc, "selection": selection,
            "model_label": label, "stats": stats,
            "checkpoint": ckpt_path, "metrics_path": metrics_path,
            "data_path": data_path, "entries": out}


def figure_target_paths(fig_dir):
    planned = [os.path.join(fig_dir, "figure_manifest.json")]
    for tag in ("typical", "worst"):
        for ext in ("pdf", "svg", "png"):
            planned.append(os.path.join(fig_dir, f"{FIG_STEM}_{tag}_g*_seed0.{ext}"))
    return planned


def check_figure_targets_clear(fig_dir, geom_ids):
    planned = [os.path.join(fig_dir, "figure_manifest.json")]
    for tag, gid in geom_ids.items():
        for ext in ("pdf", "svg", "png"):
            planned.append(os.path.join(fig_dir,
                                        f"{FIG_STEM}_{tag}_g{gid}_seed0.{ext}"))
    for path in planned:
        if os.path.exists(path):
            raise SystemExit(f"[refuse] figure target already exists: {path}; "
                             "refusing to overwrite figures")


def _render_figures(data, fig_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[skip] matplotlib is not available; figures were NOT rendered and "
              "no output was written. Run this task in the server matplotlib "
              "environment (no dependency installed here).", file=sys.stderr)
        raise SystemExit(3)

    matplotlib.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    os.makedirs(fig_dir, exist_ok=True)
    written = []
    for entry in data["entries"]:
        gid = entry["geom_id"]
        x = entry["x"]
        truth = entry["truth"]
        pred = entry["pred"]
        # 3 rows (GT / Pred / |Err|) x 4 columns (u, v, p, T)
        fig, axes = plt.subplots(3, 4, figsize=(11.5, 8.6))
        for c, field in enumerate(FIELDS):
            j = c
            gt = truth[:, j]
            pr = pred[:, j]
            err = np.abs(pr - gt)
            if field in SIGNED_FIELDS:
                vmax = float(max(np.abs(gt).max(), np.abs(pr).max()))
                norm = dict(vmin=-vmax, vmax=vmax, cmap=CMAP_SIGNED)
            else:
                vmin = float(min(gt.min(), pr.min()))
                vmax = float(max(gt.max(), pr.max()))
                norm = dict(vmin=vmin, vmax=vmax, cmap=CMAP_SEQ)
            evmax = float(err.max()) if err.size else 1.0
            panels = (
                ("GT", gt, norm),
                ("Pred", pr, norm),
                ("|Err|", err, dict(vmin=0.0, vmax=evmax, cmap=CMAP_ERR)),
            )
            for r, (title, arr, kwargs) in enumerate(panels):
                ax = axes[r][c]
                sc = ax.scatter(x[:, 0], x[:, 1], c=arr, s=1.0,
                                rasterized=True, **kwargs)
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(field, fontsize=9, pad=3)
                if c == 0:
                    ax.set_ylabel(title, fontsize=9)
                cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
                cb.ax.tick_params(labelsize=6)
        fig.suptitle(f"Geometry g{gid} ({entry['tag']}; stored scale)",
                     fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        for ext in ("pdf", "svg", "png"):
            path = os.path.join(
                fig_dir, f"{FIG_STEM}_{entry['tag']}_g{gid}_seed0.{ext}")
            fig.savefig(path, format=ext, dpi=300, bbox_inches="tight")
            written.append(path)
        plt.close(fig)
    print("[note] scatter points rasterized; text/axes/colorbar remain vector "
          "and editable (pdf.fonttype=42, svg.fonttype='none')")
    return written


def run_figures(fig_dir):
    metrics_path = os.path.join(WS, METRICS_REL)
    ckpt_path = os.path.join(WS, CKPT_REL)
    _require(os.path.isfile(metrics_path),
             f"metrics JSON not found: {metrics_path}")
    _require(os.path.isfile(ckpt_path), f"checkpoint not found: {ckpt_path}")
    with open(metrics_path, "r", encoding="utf-8") as fh:
        metrics_doc = json.load(fh)
    selection = select_holdout_geometries(metrics_doc)
    tags = ("typical", "worst")
    geom_ids = {p["tag"]: int(p["row"]["geom_id"])
                for p in selection["picks"] if p["tag"] in tags}
    check_figure_targets_clear(fig_dir, geom_ids)
    data_path = find_data_path()
    data = compute_figure_data(metrics_path, ckpt_path, data_path, tags=tags)
    files = _render_figures(data, fig_dir)

    manifest = {
        "task": "figures",
        "r2_task": "R2-CLOSEOUT-COST-FIG-01",
        "created_local": _now_local(),
        "model": "feat_extract (pool_mode=symmetric), seed0",
        "model_label": data["model_label"],
        "checkpoint": os.path.relpath(ckpt_path, WS),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "metrics": os.path.relpath(metrics_path, WS),
        "metrics_sha256": sha256_file(metrics_path),
        "data": data_path,
        "normalization": "training stats from metrics protocol (train-only)",
        "forward": "CPU full 5000 points, no chunking, no training/selection",
        "selection": {
            "rule": "typical = re_u closest to mean of existing test rows; "
                    "worst = maximum re_u",
            "mean_re_u_existing_test_rows": selection["mean_re_u"],
            "note": selection["selection_note"],
        },
        "geometries": [],
        "colormaps": {"signed": CMAP_SIGNED, "T": CMAP_SEQ, "error": CMAP_ERR},
        "shared_limits": "GT and Pred share limits per field; error starts at 0",
        "files": files,
        "notes": [
            "held-out geometries are selected post hoc from existing test rows "
            "and are not independent validation",
            "no original figure or accepted result was overwritten",
            "scatter rasterized; text/axes/colorbar vector",
        ],
    }
    for entry in data["entries"]:
        manifest["geometries"].append({
            "tag": entry["tag"],
            "geom_id": entry["geom_id"],
            "re": entry["re"],
            "re_from_metrics_json": {f: float(entry["row"][f"re_{f}"])
                                     for f in FIELDS},
            "abs_error_vs_metrics_json": entry["re_error_vs_json"],
            "files": [p for p in files if f"_{entry['tag']}_g{entry['geom_id']}_"
                      in os.path.basename(p)],
        })
    _write_json(os.path.join(fig_dir, "figure_manifest.json"), manifest)
    print(f"[figures] written {len(files)} image files + figure_manifest.json "
          f"-> {fig_dir}", flush=True)
    return manifest


# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=("cost", "figures", "all"), required=True)
    ap.add_argument("--out-root", default=None,
                    help="results root (default <workspace>/results); cost dir = "
                         "<root>/round2_cost_figures, figure dir = "
                         "<root>/figures/round2_thermal_current")
    return ap.parse_args()


def main():
    args = parse_args()
    out_root, cost_dir, fig_dir = resolve_paths(args)
    t0 = time.time()
    if args.task in ("cost", "all"):
        run_cost(cost_dir)
    if args.task in ("figures", "all"):
        run_figures(fig_dir)
    print(f"[done] task={args.task} in {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
