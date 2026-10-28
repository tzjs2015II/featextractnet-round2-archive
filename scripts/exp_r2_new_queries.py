"""exp_r2_new_queries.py —— R2-NEW-QUERY-01：两个新节点分布 + 插值参考诊断。

协议（实验计划最后段，计算前冻结）
----------------------------------
* 固定 20 个 test 几何 × 5 个已训练 symmetric FeatExtractNet seed（0–4）；不训练/调参/选路。
* 每几何两个由**几何**而非误差决定的新坐标集合，每集合 5000 点：
    - Sobol：`qmc.Sobol(d=2, scramble=True, seed=20261022+gid)` 取 2^16 候选，
      过滤到流体区域（正方形 [-1,1]^2 内、正多边形外）后取前 4340 个流体内部点；
    - Jittered grid：90x90 细胞中心 + 每维 ±0.4 cell（格内 80% 全宽）均匀抖动，
      过滤到流体区域，
      用固定 rng（20261022+gid）打乱后取前 4340 个（不按网格行顺序偏取）。
  两集合都另加 492 个外正方形墙点与 168 个内多边形墙点（等弧长 half-step 偏移；
  外正方形周长 8；内多边形复用已验证的 regular_polygon_vertices）。
  同几何新坐标对所有 seed 相同；几何只用原 double 坐标，不猜相位。
* 新点 reference 固定使用原同几何已发布场的 k=8、反距离平方（IDW）插值；精确重合
  返回原标签；新墙 u=v=0、T 外 0/内 1 为已知 BC，pressure 仍 IDW 且无 offset。
  原始 reference/BC 标签只用于打分，不进入模型输入。
* 插值留出诊断：每几何 rng(20261022+1000+gid) 固定选 868 个原内部节点（20%）；
  source = 全 660 墙 + 其余 3472 内部；k=8 固定，不调参；报告 RE/MAE/max。
  这不是新位置标签误差的严格上界，也不是独立数值验证。
* 模型：原 train-only stats、strict float32 CPU、完整 5000 新支持点一次前向
  （不独立 chunk context）；先复现原 5000 点 GPU metrics RE 差 <=1e-6，再对两个
  新分布逐 seed×geometry 计算 all5000 与 wall 的 RE/MAE/max（零分母 RE=null），
  每几何 5 seed 样本 std，两分布分列。术语保持 interpolated reference。

输出（canonical results/round2_new_queries，任何目标已存在即拒绝）
------------------------------------------------------------------
  protocol_snapshot.json      计算前 JSON 协议（输入/哈希/分布/标签/指标定义）
  protocol.md                 计算前 Markdown 协议
  geom_<gid>.npz              每几何：coords/ref（两分布）、墙类型/标签来源、
                              留出 indices、5 seed 预测、墙距离/nearest-original
  raw_metrics.json            5 seed × 20 geom × 2 分布 all5000/wall 指标
  diagnostics.json            分布覆盖/墙距离/nearest-original/重合计数
  summary.json                geometry -> 5 seed mean/sample std；两分布分列
  report.md                   结果表与限制（interpolated reference）
  registry_candidate.json     Codex 合并用（本脚本不写共享 registry）

CLI
---
  python scripts/exp_r2_new_queries.py \
      --input-workspace /slurm/.../返稿修订工作区 \
      --data /slurm/.../PINN-AP/HeatTransferData.npy \
      --out-dir /slurm/.../返稿修订工作区/results/round2_new_queries

约束：CPU 4 线程、30min/16G/0GPU；不安装/不提交/不 SSH/不用本机 GPU；本机自查不写
canonical。
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
from scipy.spatial import cKDTree
from scipy.stats import qmc

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WS not in sys.path:
    sys.path.insert(0, WS)

# 复用已验证的几何/统计/模型 helper（不修改原文件）
from scripts import exp_r2_local_error as LOC  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
N_POINTS = 5000
N_OUTER = 492
N_INNER = 168
N_BOUNDARY = N_OUTER + N_INNER
N_INTERIOR = N_POINTS - N_BOUNDARY
DISTS = ("sobol", "jittered")
FIELDS = ("u", "v", "p", "T")
SQUARE_PERIMETER = 8.0
NEW_QUERY_SEED = 20261022
SOBOL_CANDIDATES = 2 ** 16
JITTER_CELLS = 90
JITTER_FRACTION = 0.8
IDW_K = 8
IDW_EXACT_TOL = 1e-12
HOLDOUT_FRACTION = 0.20
HOLDOUT_N = int(round(N_INTERIOR * HOLDOUT_FRACTION))  # 868
BOUNDARY_TOL = 1e-6
RE_METRICS_TOL = 1e-6
N_THREADS = 4
OUT_BASENAME = "round2_new_queries"
OUT_FILES = ("protocol_snapshot.json", "protocol.md", "raw_metrics.json",
             "diagnostics.json", "summary.json", "report.md",
             "registry_candidate.json")


def _require(cond, msg):
    if not cond:
        raise SystemExit("[fail] " + msg)


def _now_local():
    from datetime import datetime
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path, doc):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


# ---------------------------------------------------------------------------
# 几何 mask / 墙采样 / 分布生成
# ---------------------------------------------------------------------------
def inside_square(P):
    P = np.asarray(P, dtype=np.float64)
    return ((P[:, 0] >= -1.0) & (P[:, 0] <= 1.0) &
            (P[:, 1] >= -1.0) & (P[:, 1] <= 1.0))


def inside_polygon(P, spec):
    """凸正多边形（CCW 顶点）内部；边界也算 inside（随后被排除出流体）。"""
    P = np.asarray(P, dtype=np.float64)
    v = LOC.regular_polygon_vertices(spec)
    cross = []
    for k in range(v.shape[0]):
        a = v[k]
        b = v[(k + 1) % v.shape[0]]
        ab = b - a
        ap = P - a
        cross.append(ab[0] * ap[:, 1] - ab[1] * ap[:, 0])
    return np.all(np.stack(cross, axis=1) >= -1e-12, axis=1)


def fluid_mask(P, spec):
    return inside_square(P) & (~inside_polygon(P, spec))


def _sample_polyline_perimeter(vertices, n, half_step=True):
    """沿闭合折线等弧长采 n 点；half_step=True 用 (i+0.5)*P/n。"""
    vertices = np.asarray(vertices, dtype=np.float64)
    closed = np.vstack([vertices, vertices[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    perimeter = float(seg.sum())
    s = (np.arange(n, dtype=np.float64) + (0.5 if half_step else 0.0)) \
        * perimeter / n
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    out = np.empty((n, 2), dtype=np.float64)
    for i, si in enumerate(s):
        j = int(np.searchsorted(cum, si, side="right") - 1)
        j = min(max(j, 0), len(seg) - 1)
        t = (si - cum[j]) / seg[j]
        out[i] = closed[j] + t * (closed[j + 1] - closed[j])
    return out, perimeter


def outer_wall_points():
    sq = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    pts, per = _sample_polyline_perimeter(sq, N_OUTER)
    _require(abs(per - SQUARE_PERIMETER) < 1e-12, f"square perimeter {per}")
    return pts


def inner_wall_points(spec):
    v = LOC.regular_polygon_vertices(spec)
    pts, per = _sample_polyline_perimeter(v, N_INNER)
    return pts, per


def sobol_interior(spec, gid):
    seed = NEW_QUERY_SEED + int(gid)
    sob = qmc.Sobol(d=2, scramble=True, seed=seed)
    cand = sob.random(SOBOL_CANDIDATES) * 2.0 - 1.0
    mask = fluid_mask(cand, spec)
    accepted = cand[mask]
    _require(accepted.shape[0] >= N_INTERIOR,
             f"geom {gid}: only {accepted.shape[0]} fluid Sobol candidates")
    return accepted[:N_INTERIOR].copy(), {
        "candidates": SOBOL_CANDIDATES, "accepted": int(mask.sum()),
        "scramble_seed": seed, "take_rule": "first accepted in Sobol order",
    }


def jittered_interior(spec, gid):
    rng = np.random.default_rng(NEW_QUERY_SEED + int(gid))
    cell = 2.0 / JITTER_CELLS
    centers = np.stack(np.meshgrid(
        -1.0 + (np.arange(JITTER_CELLS) + 0.5) * cell,
        -1.0 + (np.arange(JITTER_CELLS) + 0.5) * cell), axis=-1).reshape(-1, 2)
    # JITTER_FRACTION=0.8 表示占满格宽的 80%；半宽 = 0.8*cell/2 = ±0.4 cell
    half = JITTER_FRACTION * cell / 2.0
    jitter = rng.uniform(-half, half, size=(centers.shape[0], 2))
    cand = centers + jitter
    mask = fluid_mask(cand, spec)
    accepted = cand[mask]
    _require(accepted.shape[0] >= N_INTERIOR,
             f"geom {gid}: only {accepted.shape[0]} fluid jittered candidates")
    perm = rng.permutation(accepted.shape[0])
    return accepted[perm[:N_INTERIOR]].copy(), {
        "cells": JITTER_CELLS, "candidates": int(centers.shape[0]),
        "accepted": int(mask.sum()), "jitter_fraction": JITTER_FRACTION,
        "jitter_half_width_cell": half,
        "rng_seed": NEW_QUERY_SEED + int(gid),
        "take_rule": "fixed-rng shuffled accepted candidates",
    }


def build_distribution(X_orig, spec, gid, kind):
    outer = outer_wall_points()
    inner, inner_per = inner_wall_points(spec)
    if kind == "sobol":
        interior, info = sobol_interior(spec, gid)
    elif kind == "jittered":
        interior, info = jittered_interior(spec, gid)
    else:
        raise ValueError(kind)
    coords = np.vstack([outer, inner, interior]).astype(np.float64)
    wall_kind = np.array(["outer"] * N_OUTER + ["inner"] * N_INNER +
                         ["interior"] * N_INTERIOR, dtype="U8")
    _require(coords.shape == (N_POINTS, 2), f"coords shape {coords.shape}")
    # 墙在连续边界上：外 492 只在外正方形、内 168 只在内多边形
    d_wall, d_inner_all, d_outer_all = LOC.boundary_distance(
        coords[:N_BOUNDARY], spec)
    _require(d_wall.max() < BOUNDARY_TOL,
             f"geom {gid}/{kind}: wall boundary dist {d_wall.max():.3e}")
    _require(d_outer_all[:N_OUTER].max() < BOUNDARY_TOL,
             f"geom {gid}/{kind}: outer wall not on square "
             f"({d_outer_all[:N_OUTER].max():.3e})")
    _require(d_inner_all[N_OUTER:].max() < BOUNDARY_TOL,
             f"geom {gid}/{kind}: inner wall not on polygon "
             f"({d_inner_all[N_OUTER:].max():.3e})")
    _require(bool(fluid_mask(coords[N_BOUNDARY:], spec).all()),
             f"geom {gid}/{kind}: interior point outside fluid region")
    _require(np.unique(coords, axis=0).shape[0] == N_POINTS,
             f"geom {gid}/{kind}: duplicate coordinates")
    info.update({"inner_perimeter": inner_per,
                 "boundary_wall_max_dist": float(d_wall.max()),
                 "outer_wall_max_outer_dist": float(d_outer_all[:N_OUTER].max()),
                 "inner_wall_max_inner_dist": float(d_inner_all[N_OUTER:].max())})
    return coords, wall_kind, info


# ---------------------------------------------------------------------------
# IDW reference
# ---------------------------------------------------------------------------
def idw_reference(targets, source_coords, source_fields, k=IDW_K):
    """k=8 反距离平方 IDW；精确重合返回原标签；返回 values, exact_overlap, nearest_d。"""
    targets = np.asarray(targets, dtype=np.float64)
    src = np.asarray(source_coords, dtype=np.float64)
    vals = np.asarray(source_fields, dtype=np.float64)
    tree = cKDTree(src)
    d, idx = tree.query(targets, k=k)
    if d.ndim == 1:
        d = d[:, None]
        idx = idx[:, None]
    out = np.empty((targets.shape[0], vals.shape[1]), dtype=np.float64)
    exact = np.zeros(targets.shape[0], dtype=bool)
    for i in range(targets.shape[0]):
        if d[i, 0] <= IDW_EXACT_TOL:
            out[i] = vals[idx[i, 0]]
            exact[i] = True
            continue
        w = 1.0 / (d[i] ** 2)
        w = w / w.sum()
        out[i] = w @ vals[idx[i]]
    return out, exact, d[:, 0]


def _label_source_4ch(wall_kind, exact):
    """每通道来源标签 [N,4] 固定宽度 Unicode，列顺序 u,v,p,T。

    墙：u/v/T（列 0,1,3）恒为 wall_bc，精确重合也不覆盖 BC 来源；
        p（列 2）恒为 idw 或 exact_original，绝不是 wall_bc。
    内部：各通道按 exact 为 exact_original / idw。
    """
    wall_kind = np.asarray(wall_kind, dtype="U8")
    exact = np.asarray(exact, dtype=bool)
    n = wall_kind.shape[0]
    lab = np.full((n, 4), "idw", dtype="U16")
    wall = wall_kind != "interior"
    interior = ~wall
    lab[interior & exact, :] = "exact_original"
    lab[wall, 0] = "wall_bc"  # u
    lab[wall, 1] = "wall_bc"  # v
    lab[wall, 3] = "wall_bc"  # T
    lab[wall & exact, 2] = "exact_original"  # p: exact only; otherwise idw
    return lab


def build_reference(coords, wall_kind, X_orig, Y_orig):
    """内部全量 IDW；墙 u/v/T 用已知 BC，p 仍 IDW（无 offset）。

    label_source 为逐通道 [N,4]：u/v/T 墙上恒为 wall_bc；p 恒为 idw/exact_original；
    精确重合仅影响对应通道的来源标签，不覆盖墙 BC 来源。
    """
    ref, exact, nearest = idw_reference(coords, X_orig, Y_orig)
    ref = ref.copy()
    wall = wall_kind != "interior"
    ref[wall, 0] = 0.0  # u
    ref[wall, 1] = 0.0  # v
    outer = wall_kind == "outer"
    inner = wall_kind == "inner"
    ref[outer, 3] = 0.0
    ref[inner, 3] = 1.0
    label_source = _label_source_4ch(wall_kind, exact)
    return ref, exact, nearest, label_source


# ---------------------------------------------------------------------------
# 插值留出诊断
# ---------------------------------------------------------------------------
def holdout_indices(gid):
    rng = np.random.default_rng(NEW_QUERY_SEED + 1000 + int(gid))
    interior = np.arange(N_BOUNDARY, N_POINTS)
    idx = rng.choice(interior, size=HOLDOUT_N, replace=False)
    return np.sort(idx.astype(np.int64))


def holdout_diagnostic(X_orig, Y_orig, holdout):
    interior = np.arange(N_BOUNDARY, N_POINTS)
    rest = np.setdiff1d(interior, holdout, assume_unique=True)
    source_ids = np.concatenate([np.arange(N_BOUNDARY), rest]).astype(np.int64)
    _require(source_ids.size == N_BOUNDARY + (N_INTERIOR - HOLDOUT_N),
             f"source size {source_ids.size}")
    _require(np.intersect1d(source_ids, holdout).size == 0, "holdout leaked into source")
    pred, exact, _ = idw_reference(X_orig[holdout], X_orig[source_ids],
                                   Y_orig[source_ids])
    _require(not exact.any(), "holdout target coincides with source")
    metrics = LOC.field_metrics(pred, Y_orig[holdout])
    return {"holdout_indices": [int(v) for v in holdout],
            "source_n": int(source_ids.size),
            "holdout_n": int(holdout.size),
            "metrics": metrics}


# ---------------------------------------------------------------------------
# 输入校验 / 协议
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-workspace", required=True,
                    help="canonical read-only workspace with accepted results/configs")
    ap.add_argument("--data", required=True, help="explicit HeatTransferData.npy")
    ap.add_argument("--out-dir", required=True,
                    help="<input-workspace>/results/round2_new_queries")
    return ap.parse_args()


def resolve_inputs(args):
    INPUT = os.path.abspath(args.input_workspace)
    OUT = os.path.abspath(args.out_dir)
    DATA = os.path.abspath(args.data)
    _require(os.path.isdir(INPUT), f"--input-workspace not a dir: {INPUT}")
    _require(os.path.isfile(DATA), f"--data not found: {DATA}")
    _require(os.path.normcase(os.path.realpath(OUT)).startswith(
                 os.path.normcase(os.path.realpath(INPUT)) + os.sep),
             f"--out-dir must be inside --input-workspace: {OUT}")
    _require(os.path.basename(os.path.normpath(OUT)) == OUT_BASENAME,
             f"--out-dir basename must be {OUT_BASENAME!r}")
    metrics = {}
    for seed in SEEDS:
        for name in ("checkpoint_best.pth", f"metrics_seed{seed}.json"):
            path = os.path.join(INPUT, "results", "round2_thermal_symmetric",
                                "feat_extract", f"seed{seed}", name)
            _require(os.path.isfile(path), f"missing accepted input: {path}")
        metrics[seed] = os.path.join(INPUT, "results", "round2_thermal_symmetric",
                                     "feat_extract", f"seed{seed}",
                                     f"metrics_seed{seed}.json")
    split_path = os.path.join(INPUT, "configs", "splits_thermal.json")
    spec_path = os.path.join(INPUT, "results", "thermal_geometry_spec.json")
    _require(os.path.isfile(split_path), f"missing splits: {split_path}")
    _require(os.path.isfile(spec_path), f"missing geometry spec: {spec_path}")
    return INPUT, OUT, DATA, metrics, split_path, spec_path


def load_inputs(INPUT, DATA, metrics_paths, split_path, spec_path):
    from src.core import thermal_io
    split = json.loads(open(split_path, encoding="utf-8").read())
    test_ids = [int(g) for g in split["protocol_A"]["test"]]
    _require(len(test_ids) == 20, f"test ids {len(test_ids)} != 20")
    _require(split.get("split_seed") == 123, "split_seed != 123")
    spec_doc = json.loads(open(spec_path, encoding="utf-8").read())
    per_geo = {int(r["geom"]): r for r in spec_doc["per_geometry"]}
    _require(all(g in per_geo for g in test_ids), "spec lacks test geoms")
    X, Y = thermal_io.load_thermal(DATA)
    _require(X.shape == (135, N_POINTS, 2) and Y.shape == (135, N_POINTS, 4),
             f"unexpected data shapes X={X.shape} Y={Y.shape}")
    raw = np.load(DATA)
    _require(raw.shape == (135, N_POINTS, 6), f"raw data shape {raw.shape}")
    X_double = raw[:, :, :2].astype(np.float64)
    Y_double = raw[:, :, 2:6].astype(np.float64)
    metrics_docs = {}
    stats_by_seed = {}
    labels = set()
    split_versions = set()
    for seed in SEEDS:
        doc = json.loads(open(metrics_paths[seed], encoding="utf-8").read())
        proto = doc["protocol"]
        _require(proto.get("pool_context") == "symmetric",
                 f"seed{seed} pool_context != symmetric")
        _require(proto.get("implementation_version") == "R2-IMPL-01B",
                 f"seed{seed} implementation_version mismatch")
        split_versions.add(str(proto.get("split_version")))
        stats = {k: np.asarray(proto[k], np.float32)
                 for k in ("x_mean", "x_std", "y_mean", "y_std")}
        if seed == 0:
            stats_by_seed[seed] = stats
        else:
            _require(all(np.array_equal(stats[k], stats_by_seed[0][k])
                         for k in stats),
                     f"seed{seed} stats differ from seed0")
            stats_by_seed[seed] = stats
        metrics_docs[seed] = doc
        labels.add(str(proto.get("model_label")))
    _require(len(labels) == 1, f"model labels differ: {labels}")
    _require(len(split_versions) == 1, f"split versions differ: {split_versions}")
    return {"split": split, "test_ids": test_ids, "spec": spec_doc,
            "per_geo": per_geo, "X": X, "Y": Y,
            "X_double": X_double, "Y_double": Y_double,
            "metrics_docs": metrics_docs, "stats_by_seed": stats_by_seed,
            "label": sorted(labels)[0], "split_version": split_versions.pop()}


def build_snapshot(INPUT, OUT, DATA, split_path, spec_path, inputs):
    source_rels = ("scripts/exp_r2_new_queries.py",
                   "scripts/exp_r2_local_error.py",
                   "src/core/thermal_io.py",
                   "scripts/exp_thermal.py",
                   "src/models/featextractnet.py")
    source_sha = {rel: sha256_file(os.path.join(WS, rel))
                  for rel in source_rels
                  if os.path.isfile(os.path.join(WS, rel))}
    ckpt_sha, metrics_sha = {}, {}
    for seed in SEEDS:
        base = os.path.join(INPUT, "results", "round2_thermal_symmetric",
                            "feat_extract", f"seed{seed}")
        ckpt_sha[str(seed)] = sha256_file(os.path.join(base, "checkpoint_best.pth"))
        metrics_sha[str(seed)] = sha256_file(
            os.path.join(base, f"metrics_seed{seed}.json"))
    return {
        "task": "new_query_distributions_interpolated_reference",
        "r2_task": "R2-NEW-QUERY-01",
        "created_local": _now_local(),
        "input_workspace": INPUT,
        "out_dir": OUT,
        "data": DATA,
        "data_sha256": sha256_file(DATA),
        "split_config": split_path,
        "split_sha256": sha256_file(split_path),
        "split_version": inputs["split_version"],
        "test_ids": inputs["test_ids"],
        "n_test_geom": len(inputs["test_ids"]),
        "geometry_spec": spec_path,
        "geometry_spec_sha256": sha256_file(spec_path),
        "geometry_reconstruction": {
            "rule": ("regular n-gon from n/R/phase_deg/centre via verified "
                     "regular_polygon_vertices; outer square [-1,1]^2; original "
                     "double coordinates, no phase guessing"),
            "boundary_check_max_by_geom": {},
        },
        "model": {"token": "feat_extract", "pool_context": "symmetric",
                  "label": inputs["label"], "seeds": list(SEEDS)},
        "checkpoint_sha256": ckpt_sha,
        "metrics_sha256": metrics_sha,
        "source_sha256": source_sha,
        "distributions": {
            "sobol": {
                "definition": ("qmc.Sobol(d=2, scramble=True, seed=20261022+gid), "
                               "2^16 candidates, filter to fluid region, take first "
                               "4340 accepted in Sobol order"),
                "scramble_seed_base": NEW_QUERY_SEED,
                "candidates": SOBOL_CANDIDATES,
                "n_interior": N_INTERIOR, "n_outer_wall": N_OUTER,
                "n_inner_wall": N_INNER, "n_total": N_POINTS,
                "selection_basis": "geometry only, not errors",
            },
            "jittered": {
                "definition": ("90x90 cell centers on [-1,1]^2, uniform +/-0.4 "
                               "cell (JITTER_FRACTION=0.8 of full cell width; "
                               "half-width 0.4*cell) jitter per axis, filter to "
                               "fluid region, fixed-rng shuffle, take first 4340"),
                "rng_seed_base": NEW_QUERY_SEED,
                "cells": JITTER_CELLS, "jitter_fraction": JITTER_FRACTION,
                "jitter_half_width_cells": JITTER_FRACTION / 2.0,
                "n_interior": N_INTERIOR, "n_outer_wall": N_OUTER,
                "n_inner_wall": N_INNER, "n_total": N_POINTS,
                "selection_basis": "geometry only, not errors",
            },
            "wall_sampling": {
                "outer": ("square perimeter 8, 492 points at (i+0.5)*8/492, "
                          "CCW from (-1,-1)"),
                "inner": ("polygon perimeter, 168 points at (i+0.5)*P/168, "
                          "CCW from first vertex"),
                "perimeter_rule": "equal arc length with half-step offset",
            },
        },
        "reference": {
            "interpolation": ("k=8 inverse-distance squared IDW from the same "
                              "geometry's published fields; exact coincidence "
                              "returns the original label; pressure has no offset"),
            "interior": "IDW all four fields",
            "walls": ("u=v=0; T=0 on outer wall, T=1 on inner wall (known BC); "
                      "p remains IDW"),
            "label_source": ("per-channel fixed-width Unicode [N,4] (u,v,p,T): "
                             "wall u/v/T are always wall_bc even at exact "
                             "coincidence; p is always idw/exact_original; "
                             "interior channels are idw/exact_original"),
            "labels_not_model_input": True,
            "terminology": "interpolated reference (not FEM re-mesh, not new truth)",
        },
        "holdout_diagnostic": {
            "rng": "np.random.default_rng(20261022 + 1000 + geom_id)",
            "n": HOLDOUT_N, "fraction": HOLDOUT_FRACTION,
            "source": "all 660 wall nodes + remaining 3472 interior nodes",
            "k": IDW_K,
            "note": ("not a strict upper bound for new-point label error and not "
                     "independent numerical validation; k not tuned"),
        },
        "metrics": {
            "normalization": "training-only stats saved in each seed's metrics",
            "dtype": "float32 CPU, strict float32",
            "forward": "single full-5000 symmetric context per distribution; no chunking",
            "all5000": "RE/MAE/max per field over 5000 new points",
            "wall": "RE/MAE/max per field over 660 wall points (zero denominator RE=null)",
            "statistics_primary": ("per seed: equal-weight mean over the 20 "
                                   "geometries (valid non-null only, count "
                                   "recorded); across seeds: mean/sample std "
                                   "(ddof=1). This is the main statistic."),
            "per_geometry_supplement": ("each geometry's 5-seed mean/std kept "
                                        "as supplement; median of per-geometry "
                                        "seed std is NOT the main seed std"),
            "aggregation": ("primary per-seed 20-geometry mean -> 5-seed "
                            "mean/sample std; per_geometry supplement; both "
                            "distributions separate"),
            "reproduction": "original 5000-point RE vs metrics JSON per seed x geom <=1e-6",
        },
        "cuda_or_training": "none (CPU analysis only; no weight updates)",
    }


def build_protocol_md(snapshot, inputs):
    lines = [
        "# R2-NEW-QUERY-01 协议（计算前冻结）",
        "",
        f"- 生成时间：{snapshot['created_local']}",
        f"- 输入工作区（只读）：`{snapshot['input_workspace']}`",
        f"- 模型：{snapshot['model']['label']}，seeds {snapshot['model']['seeds']}",
        f"- test 几何：{len(snapshot['test_ids'])} 个",
        f"- data SHA256：`{snapshot['data_sha256']}`",
        f"- split：`{snapshot['split_version']}`，SHA256 `{snapshot['split_sha256']}`",
        "",
        "## 新坐标分布",
        "",
    ]
    for kind in DISTS:
        d = snapshot["distributions"][kind]
        lines += [f"### {kind}", "", d["definition"], "",
                  f"- N interior/outer/inner/total = {d['n_interior']}/"
                  f"{d['n_outer_wall']}/{d['n_inner_wall']}/{d['n_total']}",
                  f"- selection: {d['selection_basis']}", ""]
    lines += [
        "## Reference（插值，不是新 truth）",
        "",
        "- k=8 inverse-distance squared from the same geometry's published fields; "
        "exact coincidence returns the original label.",
        "- interior: IDW four fields; wall: u=v=0, T=0 outer / 1 inner (known BC), "
        "p IDW without offset.",
        "- label_source is per-channel fixed-width Unicode [N,4] (u,v,p,T): wall "
        "u/v/T always wall_bc (exact coincidence does not override BC source); p is "
        "always idw/exact_original; interior channels idw/exact_original.",
        "- reference/BC labels never enter model input.",
        "",
        "## 插值留出诊断",
        "",
        f"- rng(20261022+1000+gid) selects {HOLDOUT_N} original interior nodes; "
        "source = all 660 wall + remaining 3472 interior; k=8 fixed.",
        "- Not a strict upper bound for new-point label error and not independent "
        "numerical validation.",
        "",
        "## 模型与指标",
        "",
        "- training-only stats; strict float32 CPU; single full-5000 symmetric "
        "context per distribution (no independent chunk context).",
        "- original 5000-point RE reproduced against metrics JSON <=1e-6 before "
        "new queries.",
        "- all5000 and wall RE/MAE/max per field; zero denominator RE=null; wall MAE "
        "reported separately.",
        "- main statistics: per seed equal-weight mean over the 20 geometries "
        "(valid non-null only, counts recorded), then 5-seed mean/sample std "
        "(ddof=1); per-geometry 5-seed mean/std is kept only as supplement.",
        "- No training, tuning, point selection by error, FEM re-mesh claim, or "
        "independent numerical validation claim.",
        "",
        "## 输出",
        "",
    ]
    lines += [f"- `{name}`" for name in OUT_FILES]
    lines += ["- `geom_<gid>.npz` for each of the 20 test geometries", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 指标与聚合
# ---------------------------------------------------------------------------
def metrics_for_points(pred, ref, scope_mask):
    return LOC.field_metrics(pred[scope_mask], ref[scope_mask])


def build_summary(raw):
    """主统计：每 seed 先 20 geometry 等权 mean，再 5 seed mean/sample std。

    per_geometry 保留为补充（每 geometry 的 5 seed mean/std）；
    median_per_geometry_seed_std 仅作补充诊断，不能冒充主表 seed std。
    raw 中 null 按实际有效 geometry 计数跳过。
    """
    geom_ids = sorted({r["geom_id"] for r in raw})
    lookup = {(r["seed"], r["geom_id"], r["distribution"]): r for r in raw}
    summary = {}
    for dist in DISTS:
        for scope in ("all5000", "wall"):
            domain = {}
            for field in FIELDS:
                domain[field] = {}
                for stat in ("re", "mae", "max"):
                    seed_means = []
                    for seed in SEEDS:
                        vals = [lookup[(seed, gid, dist)][scope][field][stat]
                                for gid in geom_ids]
                        valid = [v for v in vals if v is not None]
                        seed_means.append({
                            "seed": seed,
                            "mean": float(np.mean(valid)) if valid else None,
                            "n_valid_geom": len(valid),
                            "n_total_geom": len(geom_ids),
                        })
                    per_geom = {}
                    for gid in geom_ids:
                        vals = [lookup[(seed, gid, dist)][scope][field][stat]
                                for seed in SEEDS]
                        valid = [v for v in vals if v is not None]
                        per_geom[str(gid)] = {
                            "seed_values": vals,
                            "mean": float(np.mean(valid)) if valid else None,
                            "std_sample": (float(np.std(valid, ddof=1))
                                           if len(valid) > 1 else None),
                            "n_valid_seeds": len(valid),
                        }
                    primary_vals = [s["mean"] for s in seed_means
                                    if s["mean"] is not None]
                    domain[field][stat] = {
                        "primary_per_seed": {
                            "seed_means": [s["mean"] for s in seed_means],
                            "n_valid_geom_per_seed": [s["n_valid_geom"]
                                                      for s in seed_means],
                            "n_total_geom": len(geom_ids),
                            "mean": (float(np.mean(primary_vals))
                                     if primary_vals else None),
                            "std_sample": (float(np.std(primary_vals, ddof=1))
                                           if len(primary_vals) > 1 else None),
                            "n_seeds_valid": len(primary_vals),
                            "definition": ("per seed: equal-weight mean over "
                                           "valid geometries; across seeds: "
                                           "mean/sample std (ddof=1)"),
                        },
                        "per_geometry": per_geom,
                        "median_per_geometry_seed_std": _median(
                            [v["std_sample"] for v in per_geom.values()]),
                        "note": ("primary_per_seed is the main statistic; "
                                 "per_geometry and its median seed std are "
                                 "supplementary only"),
                    }
            summary.setdefault(dist, {})[scope] = domain
    return summary


def _fixture_checks():
    """统计与 label_source 小幅 fixture（运行时快速自检）。

    统计 fixture 设计为：每 geometry 的 5-seed std 与先 geometry 再 5-seed 的
    主 std 不同，可发现误把 median(per-geom std) 当主表误差条。
    """
    old_seeds = globals()["SEEDS"]
    try:
        globals()["SEEDS"] = (0, 1, 2, 3, 4)
        raw = []
        geom_vals = {1: [0.0, 10.0, 20.0, 30.0, 40.0],
                     2: [0.0, 0.0, 0.0, 0.0, 100.0]}
        for seed in range(5):
            for gid, vals in geom_vals.items():
                for dist in DISTS:
                    rec = {"seed": seed, "geom_id": gid, "distribution": dist}
                    for scope in ("all5000", "wall"):
                        rec[scope] = {
                            field: {stat: float(vals[seed])
                                    for stat in ("re", "mae", "max")}
                            for field in FIELDS}
                    raw.append(rec)
        summary = build_summary(raw)
        cell = summary["sobol"]["all5000"]["u"]["re"]
        primary = cell["primary_per_seed"]
        expected_means = [0.0, 5.0, 10.0, 15.0, 70.0]
        expected_std = float(np.std(expected_means, ddof=1))
        _require(np.allclose(primary["seed_means"], expected_means),
                 "fixture: primary per-seed means wrong")
        _require(np.isclose(primary["mean"], 20.0),
                 "fixture: primary mean wrong")
        _require(np.isclose(primary["std_sample"], expected_std),
                 "fixture: primary sample std wrong")
        _require(primary["n_valid_geom_per_seed"] == [2, 2, 2, 2, 2],
                 "fixture: valid geometry counts wrong")
        median_pg = cell["median_per_geometry_seed_std"]
        _require(median_pg is not None and
                 not np.isclose(median_pg, primary["std_sample"]),
                 "fixture: median(per-geom std) accidentally equals primary std")
        # 独立 label_source/reference fixture：原标签不等 [2,3,5,7]，
        # 覆盖 outer/inner/interior 与 exact true/false；按 u/v/p/T 字段名检查。
        src = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0],
                        [0.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
        src_fields = np.tile(np.array([2.0, 3.0, 5.0, 7.0]), (src.shape[0], 1))
        targets = np.array([[0.0, 0.0],    # outer exact
                            [1.0, 0.0],    # inner exact
                            [0.0, 1.0],    # interior exact
                            [0.15, 0.22],  # outer nonexact
                            [0.9, 0.1],    # inner nonexact
                            [0.3, -0.4]])  # interior nonexact
        wk = np.array(["outer", "inner", "interior",
                       "outer", "inner", "interior"], dtype="U8")
        exact_expected = np.array([True, True, True, False, False, False])
        ref, exact, _nearest, lab = build_reference(targets, wk, src, src_fields)
        fi = {name: i for i, name in enumerate(FIELDS)}
        wall = wk != "interior"
        _require(lab.shape == (6, 4) and lab.dtype.kind == "U",
                 "fixture: label_source must be fixed-width Unicode [N,4]")
        _require(np.array_equal(exact, exact_expected),
                 "fixture: exact-overlap flags wrong")
        _require(np.all(lab[wall, fi["u"]] == "wall_bc") and
                 np.all(lab[wall, fi["v"]] == "wall_bc") and
                 np.all(lab[wall, fi["T"]] == "wall_bc"),
                 "fixture: wall u/v/T columns must all be wall_bc")
        _require(np.all(lab[wall, fi["p"]] != "wall_bc"),
                 "fixture: wall p column must never be wall_bc")
        _require(np.all(lab[wall & exact_expected, fi["p"]] == "exact_original") and
                 np.all(lab[wall & (~exact_expected), fi["p"]] == "idw"),
                 "fixture: wall p source must be exact_original/idw only")
        _require(np.all(lab[~wall & exact_expected, :] == "exact_original") and
                 np.all(lab[~wall & (~exact_expected), :] == "idw"),
                 "fixture: interior label_source wrong")
        _require(np.allclose(ref[0], [0.0, 0.0, 5.0, 0.0]) and
                 np.allclose(ref[1], [0.0, 0.0, 5.0, 1.0]) and
                 np.allclose(ref[2], [2.0, 3.0, 5.0, 7.0]),
                 "fixture: wall BC / p exact / interior exact reference values wrong")
        return {"primary_std": float(primary["std_sample"]),
                "median_per_geom_std": float(median_pg),
                "label_source_4ch_ok": True}
    finally:
        globals()["SEEDS"] = old_seeds


def build_report(summary, raw, diagnostics, snapshot):
    lines = [
        "# R2-NEW-QUERY-01 结果：新节点分布与插值参考",
        "",
        "- reference 为 **interpolated reference**（k=8 IDW + 已知墙 BC），不是新坐标"
        "真值、不是独立 FEM 重网格或数值验证。",
        "- 两个分布均由几何决定，未按误差选点；同几何坐标对所有 seed 相同。",
        "- 模型：原 train-only stats、strict float32 CPU、完整 5000 点前向；原始 "
        "5000 点 metrics RE 已逐 seed×geom 复现（<=1e-6）。",
        "",
        "## 主统计（每 seed 先 20 geometry 等权 mean，再 5 seed mean±sample std）",
        "",
        "两种 scope 均使用该主统计；per-geometry 的 5 seed mean/std 仅作补充"
        "（summary.json 的 per_geometry），不用 median(per-geom std) 冒充主表误差条。",
        "",
        "| distribution | scope | field | RE mean±std | MAE mean±std | max mean±std |",
        "|---|---|---|---|---|---|",
    ]
    for dist in DISTS:
        for scope in ("all5000", "wall"):
            for field in FIELDS:
                cells = {stat: summary[dist][scope][field][stat]
                         for stat in ("re", "mae", "max")}
                lines.append(
                    f"| {dist} | {scope} | {field} | "
                    f"{_pm(cells['re'])} | {_pm(cells['mae'])} | "
                    f"{_pm(cells['max'])} |")
    lines += ["", "wall 的 u/v 真值为 0，RE=null（主表显示 n/a），MAE/max 仍有效；"
              "p 无 offset。", ""]
    lines += ["", "## 插值留出诊断（868 原内部节点；k=8 固定）", "",
              "| field | RE | MAE | max |", "|---|---|---|---|"]
    hold = diagnostics["holdout_aggregate"]
    for field in FIELDS:
        lines.append(f"| {field} | {_fmt(hold[field]['re'])} | "
                     f"{_fmt(hold[field]['mae'])} | {_fmt(hold[field]['max'])} |")
    lines += ["", "## 分布检查（选一个几何示例；完整见 diagnostics.json）", ""]
    g = snapshot["test_ids"][0]
    for dist in DISTS:
        d = diagnostics["per_geometry"][str(g)][dist]
        lines.append(f"- geom {g} / {dist}: bbox={d['interior_bbox']}, "
                     f"centroid={d['interior_centroid']}, "
                     f"nearest-original median={_fmt(d['nearest_median'])}, "
                     f"exact-overlap={d['exact_overlap_count']}, "
                     f"wall-point max dist={_fmt(d['wall_points_max_dist'])}")
    lines += [
        "",
        "## 限制",
        "",
        "- 新节点 reference 是插值，不是真解；不能据此声称 FEM 重网格完成或独立"
        " numerical validation。",
        "- 868 留出诊断只是插值参考质量检查，不是新位置标签误差严格上界。",
        "- 不同节点分布下的差异只说明该 interpolated-reference 测试对支撑集敏感，"
        "不证明网格收敛或物理精度。",
        "- wall u/v 真值为 0，RE=null，仅 MAE/max 有意义；p 无 offset。",
        "",
    ]
    return "\n".join(lines)


def _median(vals):
    vals = [v for v in vals if v is not None]
    return float(np.median(vals)) if vals else None


def _fmt(v, nd=6):
    return "n/a" if v is None else f"{v:.{nd}g}"


def _pm(cell):
    """主统计 mean ± sample std（primary_per_seed）。"""
    p = cell["primary_per_seed"]
    return f"{_fmt(p['mean'])} ± {_fmt(p['std_sample'])}"


def write_registry_candidate(out_dir, snapshot):
    entries = []
    for dist in DISTS:
        entries.append({
            "id": f"R2_NEW_QUERY_{dist.upper()}",
            "experiment_id": f"R2_NEW_QUERY_{dist.upper()}",
            "type": "analysis",
            "tool": "exp_r2_new_queries.py",
            "r2_task": "R2-NEW-QUERY-01",
            "protocol": {
                "distribution": dist,
                "n_points": N_POINTS,
                "n_interior": N_INTERIOR,
                "n_outer_wall": N_OUTER,
                "n_inner_wall": N_INNER,
                "reference": "k=8 inverse-distance squared interpolated reference",
                "label_source": ("per-channel fixed-width Unicode [N,4]; wall "
                                 "u/v/T wall_bc, p idw/exact_original"),
                "statistics_primary": ("per seed 20-geometry equal-weight mean -> "
                                       "5-seed mean/sample std; per_geometry is "
                                       "supplement"),
                "jitter_half_width_cells": (0.4 if dist == "jittered" else None),
                "model_seeds": list(SEEDS),
                "split_version": snapshot["split_version"],
                "data_sha256": snapshot["data_sha256"],
            },
            "metrics": {"file": os.path.join(out_dir, "summary.json")},
            "note": ("REGISTRY CANDIDATE ONLY - not appended; interpolated "
                     "reference, not an independent numerical validation"),
        })
    return {"appended": False,
            "note": "Codex merges; this script never writes results/registry.json",
            "entries": entries}


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    torch.set_num_threads(N_THREADS)
    _require(torch.get_default_dtype() == torch.float32,
             f"expected float32 default dtype, got {torch.get_default_dtype()}")
    fixture = _fixture_checks()
    print(f"[fixture] primary_std={fixture['primary_std']:.6f}, "
          f"median_per_geom_std={fixture['median_per_geom_std']:.6f}, "
          f"label_source_4ch={fixture['label_source_4ch_ok']}", flush=True)
    INPUT, OUT, DATA, metrics_paths, split_path, spec_path = resolve_inputs(args)
    if os.path.isdir(OUT):
        _require(not os.listdir(OUT),
                 f"output dir is not empty: {OUT}; refusing to overwrite")
    for name in OUT_FILES:
        _require(not os.path.exists(os.path.join(OUT, name)),
                 f"target exists: {os.path.join(OUT, name)}")
    inputs = load_inputs(INPUT, DATA, metrics_paths, split_path, spec_path)
    test_ids = inputs["test_ids"]
    # 原 double 坐标必须落在重建连续边界上（相位核验，不猜）
    boundary_check = {}
    for g in test_ids:
        d, _, _ = LOC.boundary_distance(
            inputs["X_double"][g][:N_BOUNDARY], inputs["per_geo"][g])
        max_d = float(d.max())
        boundary_check[str(g)] = max_d
        _require(max_d < BOUNDARY_TOL,
                 f"geom {g}: original first 660 nodes not on reconstructed "
                 f"boundary (max {max_d:.3e}); refusing to guess phase")
    snapshot = build_snapshot(INPUT, OUT, DATA, split_path, spec_path, inputs)
    snapshot["geometry_reconstruction"]["boundary_check_max_by_geom"] = boundary_check
    os.makedirs(OUT, exist_ok=True)
    _write_json(os.path.join(OUT, "protocol_snapshot.json"), snapshot)
    with open(os.path.join(OUT, "protocol.md"), "w", encoding="utf-8") as fh:
        fh.write(build_protocol_md(snapshot, inputs))
    print(f"[protocol] {OUT}/protocol_snapshot.json + protocol.md", flush=True)

    device = torch.device("cpu")
    diagnostics = {"per_geometry": {}, "holdout": {},
                   "exact_overlap": {}, "reproduction": {}}
    raw = []
    npz_plan = {}
    predictions = {g: {dist: np.zeros((len(SEEDS), N_POINTS, 4), np.float32)
                       for dist in DISTS} for g in test_ids}

    # 1) 生成两分布 + reference + 留出诊断（与 seed 无关）
    for g in test_ids:
        spec = inputs["per_geo"][g]
        X_orig = inputs["X_double"][g]
        Y_orig = inputs["Y_double"][g]
        hold = holdout_indices(g)
        hold_diag = holdout_diagnostic(X_orig, Y_orig, hold)
        diagnostics["holdout"][str(g)] = hold_diag
        orig_interior = X_orig[N_BOUNDARY:]
        orig_bbox = [[float(orig_interior[:, 0].min()),
                      float(orig_interior[:, 1].min())],
                     [float(orig_interior[:, 0].max()),
                      float(orig_interior[:, 1].max())]]
        orig_centroid = [float(orig_interior[:, 0].mean()),
                         float(orig_interior[:, 1].mean())]
        tree = cKDTree(X_orig)
        per_geom = {}
        diag_geom = {}
        for dist in DISTS:
            coords, wall_kind, info = build_distribution(
                X_orig, spec, g, dist)
            ref, exact, nearest, label_source = build_reference(
                coords, wall_kind, X_orig, Y_orig)
            d_wall, _, _ = LOC.boundary_distance(coords, spec)
            nd_all = tree.query(coords)[0]
            interior_xy = coords[N_BOUNDARY:]
            wall_d = d_wall[:N_BOUNDARY]
            int_d = d_wall[N_BOUNDARY:]
            per_geom[dist] = {
                "coords": coords, "ref": ref,
                "wall_kind": wall_kind, "label_source": label_source,
                "exact": exact, "nearest": nd_all,
                "wall_dist": d_wall, "info": info,
            }
            diag_geom[dist] = {
                "coords_sha256": sha256_bytes(coords.astype("<f8").tobytes()),
                "interior_bbox": [[float(interior_xy[:, 0].min()),
                                   float(interior_xy[:, 1].min())],
                                  [float(interior_xy[:, 0].max()),
                                   float(interior_xy[:, 1].max())]],
                "interior_centroid": [float(interior_xy[:, 0].mean()),
                                      float(interior_xy[:, 1].mean())],
                "original_interior_bbox": orig_bbox,
                "original_interior_centroid": orig_centroid,
                "nearest_min": float(nd_all.min()),
                "nearest_median": float(np.median(nd_all)),
                "nearest_p95": float(np.percentile(nd_all, 95)),
                "nearest_max": float(nd_all.max()),
                "wall_distance_min": float(d_wall.min()),
                "wall_distance_median": float(np.median(d_wall)),
                "wall_distance_p95": float(np.percentile(d_wall, 95)),
                "wall_distance_max": float(d_wall.max()),
                "wall_points_median_dist": float(np.median(wall_d)),
                "wall_points_max_dist": float(wall_d.max()),
                "interior_points_median_dist": float(np.median(int_d)),
                "interior_points_min_dist": float(int_d.min()),
                "exact_overlap_count": int(exact.sum()),
                "original_coord_overlap_count": int((nd_all <= 1e-12).sum()),
                "info": info,
            }
            diagnostics["exact_overlap"][f"{g}:{dist}"] = {
                "idw_exact": int(exact.sum()),
                "wall_idw_exact": int(exact[:N_BOUNDARY].sum()),
                "interior_idw_exact": int(exact[N_BOUNDARY:].sum()),
                "original_coord_overlap": int((nd_all <= 1e-12).sum()),
            }
        diagnostics["per_geometry"][str(g)] = diag_geom
        npz_plan[g] = {"holdout": hold, **per_geom}

    # 2) 5 seed：先复现原 5000 点，再新分布预测
    for seed in SEEDS:
        model, _label, _ck = LOC.load_seed_model(
            INPUT, seed, inputs["metrics_docs"][seed], device)
        rows = {int(r["geom_id"]): r for r in
                inputs["metrics_docs"][seed]["test"]["rows"]}
        rep_max = 0.0
        for g in test_ids:
            pred_orig = LOC.predict_full(
                model, inputs["X"][g], inputs["stats_by_seed"][seed], device)
            truth = inputs["Y"][g].astype(np.float64)
            for j, field in enumerate(FIELDS):
                re = float(np.linalg.norm(pred_orig[:, j] - truth[:, j]) /
                           np.linalg.norm(truth[:, j]))
                err = abs(re - float(rows[g][f"re_{field}"]))
                _require(err <= RE_METRICS_TOL,
                         f"seed{seed} geom{g} {field}: original RE mismatch "
                         f"({re} vs {rows[g][f're_{field}']}, err {err:.3e})")
                rep_max = max(rep_max, err)
            for dist in DISTS:
                data = npz_plan[g][dist]
                pred = LOC.predict_full(
                    model, data["coords"], inputs["stats_by_seed"][seed], device)
                all_metrics = metrics_for_points(
                    pred, data["ref"], np.ones(N_POINTS, dtype=bool))
                wall_metrics = metrics_for_points(
                    pred, data["ref"],
                    np.arange(N_POINTS) < N_BOUNDARY)
                raw.append({"seed": seed, "geom_id": g, "distribution": dist,
                            "all5000": all_metrics, "wall": wall_metrics})
                predictions[g][dist][seed] = pred.astype(np.float32)
        diagnostics["reproduction"][str(seed)] = {
            "max_abs_re_error_vs_metrics": rep_max}
        print(f"[seed {seed}] reproduction max RE err {rep_max:.3e}; "
              f"new queries predicted", flush=True)
        del model
        LOC._cleanup()

    # 3) 保存 NPZ / raw / summary / diagnostics / report
    for g in test_ids:
        plan = npz_plan[g]
        npz_path = os.path.join(OUT, f"geom_{g}.npz")
        payload = {"geom_id": np.int64(g),
                   "holdout_indices": plan["holdout"].astype(np.int64)}
        for dist in DISTS:
            d = plan[dist]
            payload[f"coords_{dist}"] = d["coords"]
            payload[f"ref_{dist}"] = d["ref"]
            payload[f"wall_kind_{dist}"] = np.asarray(d["wall_kind"], dtype="U8")
            payload[f"label_source_{dist}"] = np.asarray(
                d["label_source"], dtype="U16")
            payload[f"p_label_source_{dist}"] = np.asarray(
                d["label_source"][:, 2], dtype="U16")
            payload[f"nearest_original_{dist}"] = d["nearest"]
            payload[f"wall_distance_{dist}"] = d["wall_dist"]
            payload[f"pred_{dist}"] = predictions[g][dist]
        np.savez_compressed(npz_path, **payload)
    _write_json(os.path.join(OUT, "raw_metrics.json"), raw)
    summary = build_summary(raw)
    _write_json(os.path.join(OUT, "summary.json"), summary)
    # holdout aggregate over 20 geoms
    hold_agg = {}
    for field in FIELDS:
        vals = {stat: [diagnostics["holdout"][str(g)]["metrics"][field][stat]
                       for g in test_ids]
                for stat in ("re", "mae", "max")}
        hold_agg[field] = {
            stat: float(np.mean([v for v in vals[stat] if v is not None]))
            for stat in ("re", "mae", "max")}
    diagnostic_doc = {
        "task": "new_query_distributions_interpolated_reference",
        "r2_task": "R2-NEW-QUERY-01",
        "created_local": _now_local(),
        "per_geometry": diagnostics["per_geometry"],
        "holdout": diagnostics["holdout"],
        "holdout_aggregate": hold_agg,
        "exact_overlap": diagnostics["exact_overlap"],
        "reproduction": diagnostics["reproduction"],
        "notes": ["interpolated reference only; not FEM re-mesh or independent "
                  "numerical validation",
                  "holdout diagnostic is not a strict upper bound"],
    }
    _write_json(os.path.join(OUT, "diagnostics.json"), diagnostic_doc)
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(build_report(summary, raw, diagnostic_doc, snapshot))
    _write_json(os.path.join(OUT, "registry_candidate.json"),
                write_registry_candidate(OUT, snapshot))
    print(f"[done] {OUT}", flush=True)


if __name__ == "__main__":
    main()
