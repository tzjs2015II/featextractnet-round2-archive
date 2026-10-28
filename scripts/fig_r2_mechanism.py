"""fig_r2_mechanism.py —— 修复 gate-vs-physics 图的面板标题重叠，输出可编辑 PDF/SVG/PNG。

范围
----
本脚本只做**图**：用已验收的 `results/round2_mechanism/gate_summary.json` 与
`effective_protocol.json` 作为证据来源，读取 thermal symmetric 主模型 seed0 的
metrics/checkpoint（只读），在单个展开几何（geometry 17，seed 0）上按原始定义
重算面板散点所需的三组点级量，然后画 3 面板图。不重新计算、不覆盖 100-geometry
科学结果，也不修改 exp_mechanism.py 或任何门控数值。

面板与定义（沿用 effective_protocol.json）
------------------------------------------
* gate：真实 gate_low/mid/high 模块 sigmoid 输出拼成 448 通道后逐点求均值。
* (a) Boundary distance：到最近的**采样边界点**（每几何前 660 点，cKDTree k=1）
  的距离；边界点自身为 0。这是 sampled boundary points 的距离，不是连续壁面精确距离。
* (b) Vorticity：|dv/dx - du/dy|，由 reference（真值）u/v 的 k=24 局部线性拟合
  估计，ridge=1e-8。
* (c) Temperature gradient：||grad T||，由 reference T 经同一 kNN 局部线性拟合。
* 坐标与物理量都保持 stored scale；本图不声明 SI 单位。
* 散点为 geometry 17 / seed 0；跨 seed 汇总 r **只**作为图注信息在 stdout 给出，
  不在面板上标注，也不能写成该单 geometry 的相关性。

分位 bin 端点
-------------
对每个面板按 x 的 20 分位取 21 个端点；bin 规则为 [e_i, e_{i+1})（i=0..q-2），
最后一段 [e_{q-1}, e_q] 右端包含。每个点恰好归入一个 bin，脚本会核对计数总和，
避免端点双计。

输出
----
默认 `results/figures/round2_mechanism/final/gate_vs_physics.{pdf,svg,png}`；
不覆盖旧的不合格图（`results/figures/round2_mechanism/gate_vs_physics.png`）。
若目标三件套任一已存在则拒绝（exit 4）。PDF/SVG 文字保持可编辑（fonttype 42 /
svg.fonttype none），散点允许 rasterize（向量线仍保留），脚本会明确说明。

CLI
---
  python scripts/fig_r2_mechanism.py \
      --data HeatTransferData.npy \
      --checkpoint-root results/round2_thermal_symmetric/feat_extract \
      --summary results/round2_mechanism/gate_summary.json \
      --out-dir results/figures/round2_mechanism/final \
      --device cpu

约束：不安装依赖（本地无 matplotlib 时只做数据/配置检查并明确未渲染），
不本地 CUDA、不 SSH、不提交；渲染由 Codex 在服务器 matplotlib 环境完成。
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WS not in sys.path:
    sys.path.insert(0, WS)

import importlib.util  # noqa: E402

_EM_PATH = os.path.join(WS, "scripts", "exp_mechanism.py")
_spec = importlib.util.spec_from_file_location("exp_mechanism_fig", _EM_PATH)
if _spec is None or _spec.loader is None:
    raise SystemExit(f"[fail] cannot load {_EM_PATH}")
em = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(em)

GEOM_ID = 17
SEED = 0
Q_BINS = 20
FIG_W_IN = 7.1
FIG_H_IN = 2.7
SCATTER_COLOR = "#8fb0cf"     # 低饱和蓝
QUANTILE_COLOR = "#c0504d"    # 朱红
SCATTER_SIZE = 2.0
SCATTER_ALPHA = 0.22
QUANTILE_LW = 1.0
QUANTILE_MS = 2.5
FONT_SIZE = 8
TITLE_SIZE = 9
PNG_DPI = 300
STEM = "gate_vs_physics"

PANELS = (
    ("(a) Boundary distance",
     "Distance to sampled boundary\npoints (stored scale)", "wall_dist"),
    ("(b) Vorticity", "|dv/dx-du/dy|\n(stored scale)", "vorticity"),
    ("(c) Temperature gradient", "||grad T||\n(stored scale)", "gradT"),
)


# ---------------------------------------------------------------------------
# 路径、输入校验、traceability
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=None,
                    help="HeatTransferData.npy path (default: auto-detect)")
    ap.add_argument("--checkpoint-root", default=None,
                    help="root holding seed0/checkpoint_best.pth "
                         "(default results/round2_thermal_symmetric/feat_extract)")
    ap.add_argument("--summary", default=None,
                    help="accepted gate_summary.json "
                         "(default results/round2_mechanism/gate_summary.json)")
    ap.add_argument("--out-dir", default=None,
                    help="independent output dir "
                         "(default results/figures/round2_mechanism/final)")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"),
                    help="device for the single-geometry gate forward (default cpu)")
    return ap.parse_args()


def resolve_paths(args):
    data_path = em.resolve_data_path(args.data)
    ckpt_root = (os.path.abspath(args.checkpoint_root) if args.checkpoint_root
                 else os.path.join(WS, "results", "round2_thermal_symmetric",
                                   "feat_extract"))
    summary_path = (os.path.abspath(args.summary) if args.summary
                    else os.path.join(WS, "results", "round2_mechanism",
                                      "gate_summary.json"))
    out_dir = (os.path.abspath(args.out_dir) if args.out_dir
               else os.path.join(WS, "results", "figures", "round2_mechanism",
                                 "final"))
    if not os.path.isfile(summary_path):
        raise SystemExit(f"[fail] accepted summary not found: {summary_path}")
    for rel in (f"seed{SEED}/checkpoint_best.pth", f"seed{SEED}/metrics_seed{SEED}.json"):
        if not os.path.isfile(os.path.join(ckpt_root, rel)):
            raise SystemExit(f"[fail] checkpoint root missing {rel}: {ckpt_root}")
    return data_path, ckpt_root, summary_path, out_dir


def _require(cond, msg):
    if not cond:
        raise SystemExit("[refuse] " + msg)


def planned_outputs(out_dir):
    return [os.path.join(out_dir, f"{STEM}.{ext}") for ext in ("pdf", "svg", "png")]


def check_outputs_clear(out_dir):
    """拒绝覆盖旧不合格图/已有 final 输出；默认 final/ 与旧图目录不同。"""
    legacy_fig = os.path.normcase(os.path.realpath(
        os.path.join(WS, "results", "figures")))
    legacy_gate_dir = os.path.normcase(os.path.realpath(
        os.path.join(WS, "results", "figures", "round2_mechanism")))
    out_norm = os.path.normcase(os.path.realpath(os.path.abspath(out_dir)))
    _require(out_norm != legacy_fig,
             f"out-dir must not be the legacy figure root {legacy_fig}")
    _require(out_norm != legacy_gate_dir,
             f"out-dir must not be the old figure directory {legacy_gate_dir}; "
             "use the independent final/ directory")
    for path in planned_outputs(out_dir):
        if os.path.exists(path):
            raise SystemExit(f"[refuse] figure target already exists: {path}; "
                             "refusing to overwrite the accepted/old figure")
    print(f"[ok] output targets clear: {out_dir}")


def _sha256(path):
    return em.sha256_file(path)


def load_context(args):
    data_path, ckpt_root, summary_path, out_dir = resolve_paths(args)
    with open(summary_path, "r", encoding="utf-8") as fh:
        summary = json.load(fh)
    _require(summary.get("partial") is False,
             "summary is marked partial; refusing to plot a partial gate run")
    eff = summary.get("effective_protocol") or {}
    _require(bool(eff), "summary has no effective_protocol")
    _require(eff.get("pool_context") == "symmetric",
             f"effective pool_context={eff.get('pool_context')!r} != symmetric")
    for key, expected in (("n_points_full_field", 5000), ("boundary_n", 660),
                          ("knn_k", 24), ("ridge", 1e-8),
                          ("n_test_geom", 20)):
        _require(eff.get(key) == expected,
                 f"effective_protocol {key}={eff.get(key)!r} != {expected}")
    _require(GEOM_ID in [int(g) for g in eff.get("test_geoms", [])],
             f"geometry {GEOM_ID} is not in the accepted test_geoms list")

    # 已验收 JSON 的 sibling effective_protocol.json 必须与 summary 内嵌一致
    sibling = os.path.join(os.path.dirname(summary_path), "effective_protocol.json")
    if os.path.isfile(sibling):
        with open(sibling, "r", encoding="utf-8") as fh:
            sib = json.load(fh)
        for key in ("pool_context", "test_geoms", "n_points_full_field",
                    "boundary_n", "knn_k", "ridge"):
            _require(sib.get(key) == eff.get(key),
                     f"effective_protocol.json {key} disagrees with summary")

    metrics_path = os.path.join(ckpt_root, f"seed{SEED}",
                                f"metrics_seed{SEED}.json")
    with open(metrics_path, "r", encoding="utf-8") as fh:
        metrics_doc = json.load(fh)
    proto = metrics_doc.get("protocol", {})
    _splits_doc, ids = em.load_splits_config()
    em.validate_gate_protocol(proto, "symmetric", ids)

    # 图必须基于已验收的同一 checkpoint/metrics/exp_mechanism 版本
    hashes = eff.get("hashes") or {}
    _require(_sha256(_EM_PATH) == hashes.get("script_sha256"),
             "exp_mechanism.py hash does not match the accepted gate protocol")
    _require(_sha256(os.path.join(ckpt_root, f"seed{SEED}",
                                  "checkpoint_best.pth"))
             == (hashes.get("checkpoint_sha256") or {}).get(str(SEED)),
             f"seed{SEED} checkpoint hash does not match the accepted gate protocol")
    _require(_sha256(metrics_path)
             == (hashes.get("metrics_sha256") or {}).get(str(SEED)),
             f"seed{SEED} metrics hash does not match the accepted gate protocol")

    stats = {key: np.asarray(proto[key], dtype=np.float32)
             for key in ("x_mean", "x_std", "y_mean", "y_std")}
    X, Y = em.thermal_io.load_thermal(data_path)
    em.check_full_field(X)
    _require(X.shape[0] == 135 and Y.shape[2] == 4,
             f"unexpected thermal data shape X={X.shape} Y={Y.shape}")
    return {"data": data_path, "checkpoint_root": ckpt_root,
            "summary_path": summary_path, "out_dir": out_dir,
            "summary": summary, "effective": eff,
            "metrics_doc": metrics_doc, "stats": stats,
            "X": X, "Y": Y, "device": args.device}


# ---------------------------------------------------------------------------
# 单 geometry 面板数据（真实 gate hook + reference-field indicators）
# ---------------------------------------------------------------------------
def compute_panel_data(ctx):
    device = torch.device(ctx["device"])
    em.configure_precision("symmetric")
    model = em.build_gate_model("symmetric")
    em.load_gate_checkpoint(model, ctx["checkpoint_root"], SEED,
                            ctx["metrics_doc"])
    model.to(device).eval()

    buf = {}

    def mk(name):
        def hook(_m, _i, out):
            buf[name] = out.detach()
        return hook

    handles = [model.gate_low.register_forward_hook(mk("low")),
               model.gate_mid.register_forward_hook(mk("mid")),
               model.gate_high.register_forward_hook(mk("high"))]

    stats = ctx["stats"]
    xp = ctx["X"][GEOM_ID]
    xn = (xp - stats["x_mean"]) / stats["x_std"]
    xb = em.as_btc(xn, device)
    with torch.no_grad():
        model(xb)
    for h in handles:
        h.remove()

    gates = torch.cat([torch.sigmoid(buf["low"]), torch.sigmoid(buf["mid"]),
                       torch.sigmoid(buf["high"])], dim=1)
    _require(gates.shape[1] == 448,
             f"unexpected gate channel count {gates.shape[1]} (expected 448)")
    g_bar = gates.mean(dim=1)[0].cpu().numpy().astype(np.float64)

    Yg = ctx["Y"][GEOM_ID]
    u = Yg[:, 0].astype(np.float64)
    v = Yg[:, 1].astype(np.float64)
    T = Yg[:, 3].astype(np.float64)
    kd = em.wall_distance(xp)
    du_dx, du_dy = em.knn_local_gradient(xp, u, k=em.KNN_K, ridge=em.RIDGE)
    dv_dx, dv_dy = em.knn_local_gradient(xp, v, k=em.KNN_K, ridge=em.RIDGE)
    dT_dx, dT_dy = em.knn_local_gradient(xp, T, k=em.KNN_K, ridge=em.RIDGE)
    indicators = {
        "wall_dist": kd,
        "vorticity": np.abs(dv_dx - du_dy),
        "gradT": np.hypot(dT_dx, dT_dy),
    }
    for key, arr in indicators.items():
        _require(arr.shape == (em.N_POINTS,) and np.all(np.isfinite(arr)),
                 f"indicator {key} is not finite with shape {arr.shape}")
    _require(g_bar.shape == (em.N_POINTS,)
             and np.all(np.isfinite(g_bar)),
             f"gate mean is not finite with shape {g_bar.shape}")
    return {"geom": GEOM_ID, "seed": SEED, "g_bar": g_bar,
            "indicators": indicators, "n_points": int(g_bar.size)}


# ---------------------------------------------------------------------------
# 分位 bin（端点不双计：前 q-1 段右端开，末段右端闭）
# ---------------------------------------------------------------------------
def quantile_bins(x, y, q=Q_BINS):
    """返回 [(lo, hi, n, x_mean, gate_mean), ...]；每点恰属一个 bin。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    _require(x.ndim == 1 and y.shape == x.shape,
             f"x/y shape mismatch: {x.shape} vs {y.shape}")
    edges = np.quantile(x, np.linspace(0.0, 1.0, q + 1))
    # searchsorted(side='right')-1: x==e_i (i<q) 进入 [e_i, e_{i+1})；
    # x==e_q 被 clip 到最后一 bin（右端包含）。
    idx = np.searchsorted(edges, x, side="right") - 1
    idx = np.clip(idx, 0, q - 1)
    out = []
    for j in range(q):
        mask = idx == j
        n = int(mask.sum())
        if n == 0:
            continue
        out.append({"lo": float(edges[j]), "hi": float(edges[j + 1]),
                    "n": n, "x_mean": float(x[mask].mean()),
                    "gate_mean": float(y[mask].mean())})
    _require(sum(b["n"] for b in out) == int(x.size),
             "quantile binning lost or double-counted points")
    return out


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------
def render_figure(panel, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[refuse] matplotlib is not available; figure was NOT rendered. "
              "Run this script in the project's server matplotlib environment. "
              "No dependency was installed and no output was written.", file=sys.stderr)
        raise SystemExit(3)

    matplotlib.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "mathtext.fontset": "dejavusans",
    })

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W_IN, FIG_H_IN))
    for ax, (title, xlabel, key) in zip(axes, PANELS):
        x = panel["indicators"][key]
        y = panel["g_bar"]
        ax.scatter(x, y, s=SCATTER_SIZE, alpha=SCATTER_ALPHA,
                   linewidths=0, color=SCATTER_COLOR,
                   rasterized=True, zorder=1)
        bins = quantile_bins(x, y)
        ax.plot([b["x_mean"] for b in bins], [b["gate_mean"] for b in bins],
                color=QUANTILE_COLOR, lw=QUANTILE_LW, marker="o",
                ms=QUANTILE_MS, zorder=2)
        ax.set_title(title, loc="left", fontsize=TITLE_SIZE, pad=4.0)
        ax.set_xlabel(xlabel, fontsize=FONT_SIZE)
        ax.grid(True, alpha=0.25, linewidth=0.5, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("mean gate value (448 channels)", fontsize=FONT_SIZE)
    # 两行 x label：统一在底部留足空间，避免相邻面板标题/轴文字交叠
    fig.subplots_adjust(left=0.092, right=0.995, bottom=0.310, top=0.91,
                        wspace=0.30)

    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, f"{STEM}.pdf")
    svg_path = os.path.join(out_dir, f"{STEM}.svg")
    png_path = os.path.join(out_dir, f"{STEM}.png")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, format="png", bbox_inches="tight", dpi=PNG_DPI)
    plt.close(fig)
    print("[note] scatter points are rasterized in PDF/SVG (vector quantile "
          "lines and text remain editable: pdf.fonttype=42, svg.fonttype='none')")
    return [pdf_path, svg_path, png_path]


# ---------------------------------------------------------------------------
def caption_cross_seed_r(summary):
    """跨 seed 汇总 r 只用于图注；明确不是单 geometry 相关性。"""
    overall = summary.get("overall") or {}
    labels = {"gate_vs_wall_dist": "distance to sampled boundary points",
              "gate_vs_vorticity": "vorticity |dv/dx-du/dy|",
              "gate_vs_gradT": "temperature gradient ||grad T||"}
    out = {}
    for key, label in labels.items():
        entry = overall.get(key) or {}
        out[key] = {"label": label,
                    "pearson_r_cross_seed_mean":
                        entry.get("pearson_r_cross_seed_mean"),
                    "pearson_r_cross_seed_sample_std":
                        entry.get("pearson_r_cross_seed_sample_std")}
    return out


def main():
    args = parse_args()
    data_path, ckpt_root, summary_path, out_dir = resolve_paths(args)
    check_outputs_clear(out_dir)
    ctx = load_context(args)
    panel = compute_panel_data(ctx)
    caption = caption_cross_seed_r(ctx["summary"])
    print(f"[panel] geometry={panel['geom']} seed={panel['seed']} "
          f"n_points={panel['n_points']} gate=[{panel['g_bar'].min():.4f}, "
          f"{panel['g_bar'].max():.4f}]")
    for key in ("wall_dist", "vorticity", "gradT"):
        arr = panel["indicators"][key]
        bins = quantile_bins(arr, panel["g_bar"])
        print(f"[panel] {key}: range=[{arr.min():.6g}, {arr.max():.6g}] "
              f"bins={len(bins)} bin_n_sum={sum(b['n'] for b in bins)}")
    print("[caption] cross-seed mean Pearson r over 5 seeds x 20 test geometries "
          "(for the paper caption only; NOT this single-geometry panel):")
    for key, entry in caption.items():
        if entry["pearson_r_cross_seed_mean"] is None:
            continue
        print(f"  - {entry['label']}: "
              f"{entry['pearson_r_cross_seed_mean']:+.4f} "
              f"± {entry['pearson_r_cross_seed_sample_std']:.4f}")
    files = render_figure(panel, out_dir)
    for path in files:
        print(f"[fig] {path}")
    return {"files": files, "geom": panel["geom"], "seed": panel["seed"],
            "n_points": panel["n_points"], "caption_cross_seed_r": caption}


if __name__ == "__main__":
    main()
