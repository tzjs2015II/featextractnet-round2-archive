"""exp_r2_local_error.py —— R2 二轮 Reviewer3 #8/#13 补证：局部误差/支持点敏感性。

只做描述性分析，不训练、不调参、不选路、不改任何已有结果/模型/配置/正文。

协议（运行前写入 protocol_snapshot.json 固化）
------------------------------------------------
* 输入（只读）：--input-workspace 下的
  results/round2_thermal_symmetric/feat_extract/seed{0..4}/{checkpoint_best.pth,
  metrics_seed{k}.json}、configs/splits_thermal.json、
  results/thermal_geometry_spec.json；--data 显式指到 HeatTransferData.npy。
  脚本自身 WS 是执行快照，只用于 import 代码；所有数据/结果路径显式传入。
* 全部 5 seed × 固定 20 test 几何，symmetric full 5000 点；记录 stats/权重/metrics/
  data/split/source SHA；先复算全场 RE 并与 metrics 逐量差 <=1e-6，否则停止。
* 几何：原 double 坐标 + thermal_geometry_spec.per_geometry 的 n/R/phase_deg/centre
  重建正多边形（顶点角度 = phase + pi/n + 2*pi*k/n，半径 R，中心 centre）；
  外边界为正方形 [-1,1]^2。点-线段投影 clamp 后取精确最小欧氏距离（内/外连续边界取小）。
  读入后核前 660 边界点距离 max < 1e-6；不匹配就报告并停止，**不猜相位**。
* 局部误差：原墙节点 0..659 独立组；内部 4340 点按到连续边界距离分
  [0,.02), [.02,.05), [.05,.1), [.1,inf) bands；每量 RE/MAE/max；零分母 RE=null。
  局部密度 h16 = 到第 16 个其它支持点距离（KDTree k=17，含全部 5000 点，排 self），
  density = 16 / (pi*h16^2)（points/m^2）；内部点按 h16 四等份（固定
  np.array_split；tie 稳定按原点 id），每组记录 h16/density 范围与同样误差。
  只在描述层面解释，不把相关当因果。
* 支持集敏感性（seeds 0–2，test 20）：每 geometry 固定 rng(20261021+geomID) 打乱内部
  索引；保留全部 660 墙节点，嵌套总 N=5000/2500/1250（内部前 N-660）。每个子集实际
  重算 symmetric context 并预测；对**同一 ids 的真实物理真值 Y_geom**分别打分
  subset_context_vs_reference 与 full_context_vs_reference，并保存配对差
  delta_subset_minus_full（subset-full，null 传播）。旧"以 full 预测为参照"的
  vs-full 比较仅保留为 drift_vs_full_same_subset 支持漂移诊断，不能回答真实误差。
  保存 geomID、子集哈希/原点 id、seed 结果；区分支撑变化与计分点变化；
  不称独立重网格或新坐标真值；不重训低密度。N=5000 应与 full 一致（容差内校验）。
* 聚合：先每 geometry 得到每 bin 指标（空 bin 跳过并记有效几何数），再每 seed 等权
  geometry 均值，最后 5 seed mean/sample std（ddof=1）；节点/几何内部点不当作独立重复。

输出（canonical results/round2_local_error，任何目标已存在即拒绝）
------------------------------------------------------------------
  protocol_snapshot.json       运行前协议/输入哈希/统计/几何重建定义
  raw_local_error.json         5 seed x 20 geom x (wall/bands/quartiles) 原始指标
  raw_support_sensitivity.json seeds0-2 x 20 geom x N 的真实 reference 评分/配对差/漂移
  support_subsets.json         每 geometry 的子集原点 id 与哈希
  summary.json                 geometry -> seed -> 5-seed 聚合
  report.md                    Markdown 说明与表
  registry_candidate.json      Codex 合并用（本脚本不 append 共享 registry）

CLI
---
  python scripts/exp_r2_local_error.py \
      --input-workspace /slurm/.../返稿修订工作区 \
      --out-dir /slurm/.../返稿修订工作区/results/round2_local_error \
      --data /slurm/.../PINN-AP/HeatTransferData.npy

约束：CPU（4 线程）且无 GPU；不训练/不安装/不 SSH/不提交；本机自查不写 canonical。
"""

import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WS not in sys.path:
    sys.path.insert(0, WS)

# 只在需要时 import 生产构造函数（其顶层 import POSIX fcntl；服务器 Linux 正常）
_ET = None


def _thermal_lib():
    global _ET
    if _ET is None:
        from scripts import exp_thermal as ET
        _ET = ET
    return _ET


SEEDS = (0, 1, 2, 3, 4)
SUPPORT_SEEDS = (0, 1, 2)
N_POINTS = 5000
N_BOUNDARY = 660
N_INTERIOR = N_POINTS - N_BOUNDARY
FIELDS = ("u", "v", "p", "T")
BANDS = ((0.0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, math.inf))
BAND_NAMES = ("[0,0.02)", "[0.02,0.05)", "[0.05,0.1)", "[0.1,inf)")
QUARTILE_LABELS = ("Q1", "Q2", "Q3", "Q4")
SUPPORT_TOTALS = (5000, 2500, 1250)
SUPPORT_SCORING_SETS = ("subset_context_vs_reference",
                        "full_context_vs_reference",
                        "delta_subset_minus_full",
                        "drift_vs_full_same_subset")
SUPPORT_RNG_BASE = 20261021
BOUNDARY_TOL = 1e-6
RE_METRICS_TOL = 1e-6
ZERO_DENOM_TOL = 1e-12
FULL_REORDER_TOL = 1e-4
N_THREADS = 4
OUT_BASENAME = "round2_local_error"
OUT_FILES = ("protocol_snapshot.json", "raw_local_error.json",
             "raw_support_sensitivity.json", "support_subsets.json",
             "summary.json", "report.md", "registry_candidate.json")


# ---------------------------------------------------------------------------
# 通用
# ---------------------------------------------------------------------------
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


def _norm_path(path):
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


# ---------------------------------------------------------------------------
# 连续边界距离：点-线段投影 clamp 精确欧氏距离
# ---------------------------------------------------------------------------
def point_segment_distance(P, A, B):
    """P [N,2], A/B [M,2] -> [N,M]，线段 AB 上的精确最小距离（端点 clamp）。"""
    P = np.asarray(P, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    AB = B - A
    AP = P[:, None, :] - A[None, :, :]
    denom = np.einsum("mj,mj->m", AB, AB)
    denom = np.where(denom == 0.0, 1.0, denom)
    t = np.einsum("nmj,mj->nm", AP, AB) / denom[None, :]
    t = np.clip(t, 0.0, 1.0)
    proj = A[None, :, :] + t[..., None] * AB[None, :, :]
    return np.linalg.norm(P[:, None, :] - proj, axis=2)


def regular_polygon_vertices(spec):
    """thermal_geometry_spec per_geometry -> 正多边形顶点（逆时针）。"""
    n = int(spec["n"])
    R = float(spec["R"])
    phi = math.radians(float(spec["phase_deg"]))
    centre = np.asarray(spec["centre"], dtype=np.float64)
    k = np.arange(n)
    ang = phi + math.pi / n + 2.0 * math.pi * k / n
    return centre + R * np.stack([np.cos(ang), np.sin(ang)], axis=1)


def polygon_edges(vertices):
    return np.stack([vertices, np.roll(vertices, -1, axis=0)], axis=1)


def square_edges():
    sq = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    return np.stack([sq, np.roll(sq, -1, axis=0)], axis=1)


def boundary_distance(P, spec):
    """到内（多边形）/外（正方形）连续边界的精确最小距离，返回 (min, inner, outer)。"""
    P = np.asarray(P, dtype=np.float64)
    edges = polygon_edges(regular_polygon_vertices(spec))
    d_inner = point_segment_distance(P, edges[:, 0, :], edges[:, 1, :]).min(axis=1)
    se = square_edges()
    d_outer = point_segment_distance(P, se[:, 0, :], se[:, 1, :]).min(axis=1)
    return np.minimum(d_inner, d_outer), d_inner, d_outer


# ---------------------------------------------------------------------------
# 密度 h16 / 分组
# ---------------------------------------------------------------------------
def h16_density(X_geom):
    """h16 = 到第 16 个其它支持点距离（k=17，排 self）；density=16/(pi h16^2)。"""
    X_geom = np.asarray(X_geom, dtype=np.float64)
    tree = cKDTree(X_geom)
    dist, idx = tree.query(X_geom, k=17)
    if not np.array_equal(idx[:, 0], np.arange(X_geom.shape[0])):
        raise SystemExit("[fail] cKDTree self-neighbour assumption failed")
    h16 = dist[:, 16]
    _require(np.all(h16 > 0.0), "non-positive h16 (duplicate support points)")
    density = 16.0 / (math.pi * h16 ** 2)
    return h16, density


def interior_quartiles(h16, density):
    """内部点按 h16 升序四等份；tie 稳定按原点 id。返回每组 interior-local index。"""
    interior_ids = np.arange(N_BOUNDARY, N_POINTS)
    h = h16[interior_ids]
    order = np.lexsort((interior_ids, h))
    groups = np.array_split(order, 4)
    out = []
    for label, pos in zip(QUARTILE_LABELS, groups):
        ids = interior_ids[pos]
        out.append({"quartile": label, "local_index": pos, "point_ids": ids,
                    "n": int(ids.size),
                    "h16_min": float(h16[ids].min()),
                    "h16_max": float(h16[ids].max()),
                    "density_min": float(density[ids].min()),
                    "density_max": float(density[ids].max())})
    return out


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def field_metrics(pred, truth):
    """pred/truth [N,4] -> {field: {re|None, mae, max, n, denominator_norm}}。

    RE 的分母是物理单位真值 L2 范数；当它 <= ZERO_DENOM_TOL（数值零，例如无滑移
    壁面 u/v ~1e-20）时 RE=null，避免把 1e-20 的分母放大成 1e18 的伪 RE；
    分母本身记录在 denominator_norm 供审计。MAE/max 始终保留。
    """
    out = {}
    for j, field in enumerate(FIELDS):
        diff = np.asarray(pred[:, j], np.float64) - np.asarray(truth[:, j], np.float64)
        denom = float(np.linalg.norm(np.asarray(truth[:, j], np.float64)))
        re = float(np.linalg.norm(diff) / denom) if denom > ZERO_DENOM_TOL else None
        out[field] = {"re": re, "mae": float(np.mean(np.abs(diff))),
                      "max": float(np.max(np.abs(diff))), "n": int(diff.size),
                      "denominator_norm": denom}
    return out


# ---------------------------------------------------------------------------
# 聚合：每 geometry -> 每 seed 等权均值 -> 5 seed mean/sample std
# ---------------------------------------------------------------------------
def aggregate_metric_values(values_by_seed):
    """values_by_seed: [(mean_value or None, n_geom)] -> 5-seed mean/std. """
    vals = [v for v, n in values_by_seed if v is not None]
    return {
        "seed_means": [None if v is None else float(v) for v, n in values_by_seed],
        "n_valid_geom_per_seed": [int(n) for v, n in values_by_seed],
        "mean": float(np.mean(vals)) if vals else None,
        "std_sample": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
        "n_seeds_valid": len(vals),
    }


def _cross_seed_values(values_by_seed):
    """[(seed mean or None, n_geom)] -> 3-seed mean/sample std（null 传播）。"""
    vals = [v for v, _n in values_by_seed if v is not None]
    return {
        "seed_means": [None if v is None else float(v) for v, _n in values_by_seed],
        "n_geom_per_seed": [int(n) for _v, n in values_by_seed],
        "mean": float(np.mean(vals)) if vals else None,
        "std_sample": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
        "n_seeds_valid": len(vals),
    }


# ---------------------------------------------------------------------------
# 输入与协议
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-workspace", required=True,
                    help="canonical (read-only) workspace with accepted results/configs")
    ap.add_argument("--out-dir", required=True,
                    help="canonical result dir (must be <input-workspace>/results/"
                         "round2_local_error)")
    ap.add_argument("--data", required=True, help="explicit HeatTransferData.npy path")
    return ap.parse_args()


def resolve_inputs(args):
    INPUT = os.path.abspath(args.input_workspace)
    OUT = os.path.abspath(args.out_dir)
    DATA = os.path.abspath(args.data)
    _require(os.path.isdir(INPUT), f"--input-workspace not a dir: {INPUT}")
    _require(os.path.isfile(DATA), f"--data not found: {DATA}")
    _require(_norm_path(OUT).startswith(_norm_path(INPUT) + os.sep),
             f"--out-dir must be inside --input-workspace: {OUT}")
    _require(os.path.basename(os.path.normpath(OUT)) == OUT_BASENAME,
             f"--out-dir basename must be {OUT_BASENAME!r}: {OUT}")
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


def load_and_validate_inputs(INPUT, DATA, metrics_paths, split_path, spec_path):
    from src.core import thermal_io
    with open(split_path, "r", encoding="utf-8") as fh:
        split_doc = json.load(fh)
    _require("protocol_A" in split_doc and len(split_doc["protocol_A"]["test"]) == 20,
             "splits protocol_A.test must have 20 geometries")
    _require(split_doc.get("split_seed") == 123, "split_seed != 123")
    test_ids = [int(i) for i in split_doc["protocol_A"]["test"]]
    _require(int(split_doc["protocol_A"]["train"].__len__()) == 100, "train != 100")
    _require(int(split_doc["protocol_A"]["val"].__len__()) == 15, "val != 15")

    with open(spec_path, "r", encoding="utf-8") as fh:
        spec_doc = json.load(fh)
    per_geo = {int(r["geom"]): r for r in spec_doc.get("per_geometry", [])}
    _require(all(g in per_geo for g in test_ids),
             "geometry spec lacks some test geometries")
    for g in test_ids:
        for key in ("n", "R", "phase_deg", "centre"):
            _require(key in per_geo[g], f"geometry spec missing {key} for geom {g}")

    X, Y = thermal_io.load_thermal(DATA)
    _require(X.shape == (135, N_POINTS, 2) and Y.shape == (135, N_POINTS, 4),
             f"unexpected data shapes X={X.shape} Y={Y.shape}")
    raw = np.load(DATA)
    _require(raw.shape == (135, N_POINTS, 6),
             f"unexpected raw data shape {raw.shape}")
    X_double = raw[:, :, :2].astype(np.float64)

    metrics_docs = {}
    stats_by_seed = {}
    labels = set()
    split_versions = set()
    for seed in SEEDS:
        with open(metrics_paths[seed], "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        proto = doc.get("protocol", {})
        _require(proto.get("pool_context") == "symmetric",
                 f"seed{seed} metrics pool_context != symmetric")
        _require(proto.get("implementation_version") == "R2-IMPL-01B",
                 f"seed{seed} unexpected implementation_version")
        split_versions.add(str(proto.get("split_version")))
        stats = {k: np.asarray(proto[k], dtype=np.float32)
                 for k in ("x_mean", "x_std", "y_mean", "y_std")}
        if seed == 0:
            stats_by_seed[seed] = stats
        else:
            _require(all(np.array_equal(stats[k], stats_by_seed[0][k])
                         for k in stats),
                     f"seed{seed} normalization stats differ from seed0")
            stats_by_seed[seed] = stats
        rows = {int(r["geom_id"]): r for r in doc["test"]["rows"]}
        _require(set(rows) == set(test_ids),
                 f"seed{seed} metrics test rows != split test ids")
        metrics_docs[seed] = doc
        labels.add(str(proto.get("model_label")))
    _require(len(labels) == 1, f"model labels differ across seeds: {labels}")
    _require(len(split_versions) == 1,
             f"metrics split_version differs across seeds: {split_versions}")
    return {"split_doc": split_doc, "test_ids": test_ids,
            "spec_doc": spec_doc, "per_geo": per_geo,
            "X": X, "Y": Y, "X_double": X_double,
            "metrics_docs": metrics_docs,
            "stats_by_seed": stats_by_seed, "labels": labels,
            "metrics_split_version": split_versions.pop()}


def build_snapshot(args, INPUT, OUT, DATA, metrics_paths, split_path, spec_path,
                   inputs, boundary_checks):
    """运行前固化协议；在任何模型预测/误差计算之前调用。"""
    stats0 = inputs["stats_by_seed"][0]
    source_rels = ("scripts/exp_r2_local_error.py",
                   "src/core/thermal_io.py",
                   "scripts/exp_thermal.py",
                   "src/models/featextractnet.py")
    source_sha = {rel: sha256_file(os.path.join(WS, rel))
                  for rel in source_rels if os.path.isfile(os.path.join(WS, rel))}
    checkpoint_sha = {}
    metrics_sha = {}
    for seed in SEEDS:
        ck = os.path.join(INPUT, "results", "round2_thermal_symmetric",
                          "feat_extract", f"seed{seed}", "checkpoint_best.pth")
        checkpoint_sha[str(seed)] = sha256_file(ck)
        metrics_sha[str(seed)] = sha256_file(metrics_paths[seed])
    return {
        "task": "local_error_support_sensitivity",
        "r2_task": "R2-LOCAL-ERROR-SUPPORT-01",
        "created_local": _now_local(),
        "input_workspace": INPUT,
        "out_dir": OUT,
        "data": DATA,
        "data_sha256": sha256_file(DATA),
        "split_config": split_path,
        "split_sha256": sha256_file(split_path),
        "split_version": inputs["metrics_split_version"],
        "splits_config_schema_version": inputs["split_doc"].get("version"),
        "split_seed": 123,
        "test_ids": inputs["test_ids"],
        "n_test_geom": len(inputs["test_ids"]),
        "geometry_spec": spec_path,
        "geometry_spec_sha256": sha256_file(spec_path),
        "geometry_reconstruction": {
            "inner_polygon": ("regular n-gon, circumradius R, phase phi=radians("
                              "phase_deg), centre, vertices at angle phi + pi/n + "
                              "2*pi*k/n"),
            "outer_boundary": "square [-1,1]^2 (four segments)",
            "distance": ("exact min Euclidean distance to clamped point-segment "
                         "projection over inner polygon edges and outer square "
                         "edges; boundary distance = min(inner, outer)"),
            "boundary_check": ("max continuous-boundary distance of the first 660 "
                               f"wall nodes must be < {BOUNDARY_TOL}; checked for "
                               "every test geometry"),
            "boundary_check_max_by_geom": boundary_checks,
        },
        "per_geometry_spec_used": {
            str(g): {"n": int(inputs["per_geo"][g]["n"]),
                      "R": float(inputs["per_geo"][g]["R"]),
                      "phase_deg": float(inputs["per_geo"][g]["phase_deg"]),
                      "centre": [float(v) for v in inputs["per_geo"][g]["centre"]]}
            for g in inputs["test_ids"]},
        "model": {"token": "feat_extract", "pool_context": "symmetric",
                  "label": sorted(inputs["labels"])[0],
                  "seeds": list(SEEDS)},
        "normalization": {
            "source": "training stats saved in each metrics protocol (train-only)",
            "identical_across_seeds": True,
            "x_mean": stats0["x_mean"].tolist(),
            "x_std": stats0["x_std"].tolist(),
            "y_mean": stats0["y_mean"].tolist(),
            "y_std": stats0["y_std"].tolist(),
        },
        "checkpoint_sha256": checkpoint_sha,
        "metrics_sha256": metrics_sha,
        "source_sha256": source_sha,
        "analysis": {
            "coordinate_dtype": {
                "geometry_distance_and_density": "float64 (original .npy coordinates)",
                "model_input": "arrays from src.core.thermal_io.load_thermal",
            },
            "full_field": "5000 points; CPU, no chunking; no training",
            "wall_group": f"point ids 0..{N_BOUNDARY - 1} (independent group)",
            "bands": [{"name": BAND_NAMES[i], "lo": BANDS[i][0],
                       "hi": None if math.isinf(BANDS[i][1]) else BANDS[i][1]}
                      for i in range(len(BANDS))],
            "band_rule": ("interior point ids 660..4999; lo <= d < hi for finite hi, "
                          "last band d >= 0.1"),
            "density": ("h16 = distance to the 16th other support point "
                        "(cKDTree k=17, all 5000 points, self excluded); "
                        "density = 16/(pi*h16^2) points/m^2"),
            "density_quartiles": ("interior points sorted by h16 ascending, ties by "
                                  "original point id; fixed np.array_split(...,4)"),
            "zero_denominator": (
                "RE is null when the truth L2 norm <= 1e-12 (numerical zero, "
                "e.g. no-slip wall u/v ~1e-20); the denominator norm is recorded; "
                "MAE/max kept"),
            "support_sensitivity": {
                "seeds": list(SUPPORT_SEEDS),
                "test_geoms": inputs["test_ids"],
                "rng": f"np.random.default_rng({SUPPORT_RNG_BASE} + geom_id)",
                "interior_shuffle": "permutation of interior ids 660..4999",
                "wall_retained": f"all {N_BOUNDARY} wall nodes 0..659",
                "nested_totals": list(SUPPORT_TOTALS),
                "subset_rule": ("subset = wall ids + first (N-660) shuffled interior "
                                "ids; each subset recomputes the symmetric context"),
                "comparison": ("for each subset and the same ids: "
                               "subset_context_vs_reference = field_metrics("
                               "subset_pred, Y_geom[ids]); "
                               "full_context_vs_reference = field_metrics("
                               "full_pred_on_subset, Y_geom[ids]); paired "
                               "delta_subset_minus_full = subset - full "
                               "(null propagates)"),
                "primary_real_reference": ("both context scores use the physical "
                                           "reference Y_geom; these are the main "
                                           "support tables"),
                "auxiliary_support_drift": ("drift_vs_full_same_subset keeps the "
                                            "old comparison against the full model "
                                            "output; it is a support-drift "
                                            "diagnostic only and cannot answer the "
                                            "true-reference error"),
                "not_independent": ("support change vs scoring-point change are "
                                    "separated; this is not an independent re-mesh, "
                                    "new-coordinate ground truth, or retraining"),
                "n5000_consistency_tol": FULL_REORDER_TOL,
            },
            "aggregation": {
                "order": "per geometry per bin -> per seed equal-weight geometry mean "
                         "-> 5-seed mean/sample std (ddof=1)",
                "empty_bin": "skipped; valid geometry count recorded",
                "unit_note": ("points/nodes within a geometry are not independent "
                              "replicates; seeds are the outer replication"),
            },
        },
        "re_metrics_tolerance": RE_METRICS_TOL,
        "cuda_or_training": "none (CPU analysis only; no weight updates)",
    }


# ---------------------------------------------------------------------------
# 单 seed：full predictions、metrics 复核、局部误差、支持敏感性
# ---------------------------------------------------------------------------
def load_seed_model(INPUT, seed, metrics_doc, device):
    ET = _thermal_lib()
    model, label = ET.build_model("feat_extract", pool_context="symmetric")
    ckpt_path = os.path.join(INPUT, "results", "round2_thermal_symmetric",
                             "feat_extract", f"seed{seed}", "checkpoint_best.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    _require(isinstance(ckpt, dict) and "model_state_dict" in ckpt,
             f"checkpoint missing model_state_dict: {ckpt_path}")
    _require(int(ckpt.get("seed", -1)) == seed,
             f"checkpoint seed={ckpt.get('seed')} != {seed}")
    best = metrics_doc.get("val_best") or {}
    _require(int(ckpt.get("epoch", -1)) == int(best.get("epoch", -2)),
             f"checkpoint epoch mismatch for seed{seed}")
    _require(np.isclose(float(ckpt.get("val_score", np.nan)),
                        float(best.get("score", np.nan)), rtol=1e-6, atol=1e-9),
             f"checkpoint val_score mismatch for seed{seed}")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, label, ckpt_path


def predict_full(model, X_geom, stats, device):
    from src.core import thermal_io
    xn = thermal_io.normalize(X_geom, stats["x_mean"], stats["x_std"])
    xt = torch.from_numpy(np.ascontiguousarray(xn.T, dtype=np.float32))
    with torch.no_grad():
        pred_n = model(xt.unsqueeze(0))[0]
    return thermal_io.denormalize(pred_n.T.cpu().numpy(),
                                  stats["y_mean"], stats["y_std"]).astype(np.float64)


def predict_subset(model, X_sub, stats, device):
    from src.core import thermal_io
    xn = thermal_io.normalize(X_sub, stats["x_mean"], stats["x_std"])
    xt = torch.from_numpy(np.ascontiguousarray(xn.T, dtype=np.float32))
    with torch.no_grad():
        pred_n = model(xt.unsqueeze(0))[0]
    return thermal_io.denormalize(pred_n.T.cpu().numpy(),
                                  stats["y_mean"], stats["y_std"]).astype(np.float64)


def full_metric_check(pred, truth, metrics_row, seed, geom):
    out = {}
    for j, field in enumerate(FIELDS):
        diff = pred[:, j] - truth[:, j]
        denom = float(np.linalg.norm(truth[:, j]))
        re = float(np.linalg.norm(diff) / denom) if denom > ZERO_DENOM_TOL else None
        json_re = metrics_row.get(f"re_{field}")
        _require(json_re is not None and re is not None,
                 f"seed{seed} geom{geom} {field}: null RE in metrics or recompute")
        err = abs(float(re) - float(json_re))
        _require(err <= RE_METRICS_TOL,
                 f"seed{seed} geom{geom} {field}: full-field RE mismatch "
                 f"{re} vs JSON {json_re} (err {err:.3e})")
        out[field] = {"re": float(re), "mae": float(np.mean(np.abs(diff))),
                      "max": float(np.max(np.abs(diff))),
                      "re_abs_error_vs_json": err}
    return out


def _band_index(d):
    edges = np.array([b[1] for b in BANDS[:-1]], dtype=np.float64)
    return np.searchsorted(edges, d, side="right")


def local_error_for_geometry(X_geom, pred, truth, spec, h16, density):
    d_bnd, _, _ = boundary_distance(X_geom, spec)
    wall = field_metrics(pred[:N_BOUNDARY], truth[:N_BOUNDARY])
    interior_slice = slice(N_BOUNDARY, N_POINTS)
    d_int = d_bnd[interior_slice]
    idx_band = _band_index(d_int)
    bands = []
    for b, name in enumerate(BAND_NAMES):
        mask = idx_band == b
        n = int(mask.sum())
        if n == 0:
            bands.append({"band": name, "n": 0, "metrics": None})
            continue
        bands.append({
            "band": name, "n": n,
            "lo": float(BANDS[b][0]),
            "hi": None if math.isinf(BANDS[b][1]) else float(BANDS[b][1]),
            "d_min": float(d_int[mask].min()),
            "d_max": float(d_int[mask].max()),
            "metrics": field_metrics(pred[interior_slice][mask],
                                     truth[interior_slice][mask]),
        })
    grp = interior_quartiles(h16, density)
    quartiles = []
    for g in grp:
        local = g["local_index"]
        quartiles.append({
            "quartile": g["quartile"], "n": g["n"],
            "h16_min": g["h16_min"], "h16_max": g["h16_max"],
            "density_min": g["density_min"], "density_max": g["density_max"],
            "metrics": field_metrics(pred[interior_slice][local],
                                     truth[interior_slice][local]),
        })
    return {
        "boundary_distance_max_660": float(d_bnd[:N_BOUNDARY].max()),
        "boundary_distance_max_interior": float(d_int.max()),
        "wall": {"n": N_BOUNDARY, "metrics": wall},
        "bands": bands,
        "density_quartiles": quartiles,
    }


def build_support_subsets(test_ids):
    """固定 rng(20261021+geom) 的嵌套子集；每 geometry 与训练 seed 无关。"""
    subsets = {}
    for g in test_ids:
        rng = np.random.default_rng(SUPPORT_RNG_BASE + int(g))
        shuffled = rng.permutation(np.arange(N_BOUNDARY, N_POINTS))
        per_n = {}
        for total in SUPPORT_TOTALS:
            interior_n = total - N_BOUNDARY
            ids = np.concatenate([np.arange(N_BOUNDARY),
                                  shuffled[:interior_n]]).astype(np.int64)
            _require(len(np.unique(ids)) == total,
                     f"subset N={total} has duplicate ids for geom {g}")
            _require(set(range(N_BOUNDARY)) <= set(ids.tolist()),
                     f"subset N={total} lost wall ids for geom {g}")
            if total == N_POINTS:
                _require(set(ids.tolist()) == set(range(N_POINTS)),
                         f"subset N=5000 is not the full support for geom {g}")
            per_n[str(total)] = {
                "ids": [int(v) for v in ids],
                "sha256": sha256_bytes(ids.astype("<i8").tobytes()),
            }
        subsets[g] = per_n
    return subsets


def _delta_metrics(a, b):
    """a - b for re/mae/max；任一端为 null 时传播 null。"""
    out = {}
    for field in FIELDS:
        out[field] = {}
        for stat in ("re", "mae", "max"):
            av = a[field].get(stat)
            bv = b[field].get(stat)
            out[field][stat] = (None if (av is None or bv is None)
                                else float(av) - float(bv))
    return out


def support_for_geometry(model, X_geom, stats, device, subsets_for_geom,
                         full_pred, Y_geom):
    """每个 N：真实 reference 评分 + full-context 基线 + 配对差 + 漂移诊断。

    * subset_context_vs_reference = field_metrics(sub_pred, Y_geom[ids])
    * full_context_vs_reference  = field_metrics(full_pred[ids], Y_geom[ids])
    * delta_subset_minus_full    = 前者 - 后者（null 传播）
    * drift_vs_full_same_subset  = 旧诊断（把 full 模型当参照），不能回答真实误差
    """
    out = {}
    for total, payload in subsets_for_geom.items():
        total_int = int(total)
        ids = np.asarray(payload["ids"], dtype=np.int64)
        sub_pred = predict_subset(model, X_geom[ids], stats, device)
        full_on_sub = full_pred[ids]
        ref = np.asarray(Y_geom[ids], dtype=np.float64)
        sub_vs_ref = field_metrics(sub_pred, ref)
        full_vs_ref = field_metrics(full_on_sub, ref)
        out[total] = {
            "n_total": total_int,
            "n_interior": int(total_int - N_BOUNDARY),
            "subset_sha256": payload["sha256"],
            "subset_context_vs_reference": sub_vs_ref,
            "full_context_vs_reference": full_vs_ref,
            "delta_subset_minus_full": _delta_metrics(sub_vs_ref, full_vs_ref),
            "drift_vs_full_same_subset": field_metrics(sub_pred, full_on_sub),
            "max_abs_diff_vs_full": float(np.max(np.abs(sub_pred - full_on_sub))),
        }
        if total_int == N_POINTS:
            _require(out[total]["max_abs_diff_vs_full"] <= FULL_REORDER_TOL,
                     f"reordered full-support prediction exceeds tolerance "
                     f"({out[total]['max_abs_diff_vs_full']:.3e} > "
                     f"{FULL_REORDER_TOL})")
            out[total]["consistent_with_full"] = True
    return out


def build_summary(raw_local, raw_support, test_ids, labels):
    """per geometry -> per seed -> 5-seed mean/std；空 bin 记有效几何数。"""
    # ---------------- local error ----------------
    bins_meta = [{"kind": "wall", "bin": "wall"}]
    for name in BAND_NAMES:
        bins_meta.append({"kind": "band", "bin": name})
    for label in QUARTILE_LABELS:
        bins_meta.append({"kind": "density_quartile", "bin": label})
    local_summary = {}
    for meta in bins_meta:
        kind, bin_name = meta["kind"], meta["bin"]
        per_seed = []
        for seed in SEEDS:
            per_geom = []
            for rec in raw_local[str(seed)]:
                if kind == "wall":
                    payload = rec["wall"]["metrics"]
                elif kind == "band":
                    payload = next((b["metrics"] for b in rec["bands"]
                                    if b["band"] == bin_name), None)
                else:
                    payload = next((q["metrics"] for q in rec["density_quartiles"]
                                    if q["quartile"] == bin_name), None)
                per_geom.append(payload)
            valid = [p for p in per_geom if p is not None]
            seed_mean = {}
            for field in FIELDS:
                vals = [p[field]["re"] for p in valid if p[field]["re"] is not None]
                maes = [p[field]["mae"] for p in valid]
                maxs = [p[field]["max"] for p in valid]
                seed_mean[field] = {
                    "re": float(np.mean(vals)) if vals else None,
                    "re_n_valid_geom": len(vals),
                    "mae": float(np.mean(maes)) if maes else None,
                    "max": float(np.mean(maxs)) if maxs else None,
                }
            per_seed.append({"seed": seed, "n_valid_geom": len(valid),
                             "mean_over_geoms": seed_mean})
        agg = {}
        for field in FIELDS:
            agg[field] = {
                "re": aggregate_metric_values(
                    [(s["mean_over_geoms"][field]["re"], s["n_valid_geom"])
                     for s in per_seed]),
                "mae": aggregate_metric_values(
                    [(s["mean_over_geoms"][field]["mae"], s["n_valid_geom"])
                     for s in per_seed]),
                "max": aggregate_metric_values(
                    [(s["mean_over_geoms"][field]["max"], s["n_valid_geom"])
                     for s in per_seed]),
            }
        local_summary[f"{kind}:{bin_name}"] = {
            "kind": kind, "bin": bin_name, "per_seed": per_seed,
            "cross_seed": agg,
        }
    # ---------------- support sensitivity ----------------
    # 两套真实 reference 评分（subset / full-context）与配对差是主结果；
    # drift_vs_full_same_subset 是旧诊断，仅用于分离支撑变化，不能回答真实误差。
    support_summary = {}
    for total in SUPPORT_TOTALS:
        per_seed = []
        for seed in SUPPORT_SEEDS:
            per_geom = [rec["subsets"][str(total)]
                        for rec in raw_support[str(seed)]]
            means = {}
            for key in SUPPORT_SCORING_SETS:
                means[key] = {}
                for field in FIELDS:
                    means[key][field] = {}
                    for stat in ("re", "mae", "max"):
                        vals = [p[key][field][stat] for p in per_geom]
                        vals = [v for v in vals if v is not None]
                        means[key][field][stat] = (float(np.mean(vals))
                                                   if vals else None)
            per_seed.append({
                "seed": seed, "n_geom": len(per_geom),
                "max_abs_diff_vs_full_over_geoms": float(max(
                    p["max_abs_diff_vs_full"] for p in per_geom)),
                "mean_over_geoms": means,
            })
        cross = {}
        for key in SUPPORT_SCORING_SETS:
            cross[key] = {}
            for field in FIELDS:
                cross[key][field] = {}
                for stat in ("re", "mae", "max"):
                    cross[key][field][stat] = _cross_seed_values(
                        [(s["mean_over_geoms"][key][field][stat], s["n_geom"])
                         for s in per_seed])
        support_summary[str(total)] = {"n_total": total,
                                       "per_seed": per_seed,
                                       "cross_seed": cross}
    return {
        "task": "local_error_support_sensitivity",
        "r2_task": "R2-LOCAL-ERROR-SUPPORT-01",
        "created_local": _now_local(),
        "model": {"token": "feat_extract", "pool_context": "symmetric",
                  "label": labels},
        "n_seeds": len(SEEDS),
        "test_ids": test_ids,
        "local_error": local_summary,
        "support_sensitivity": support_summary,
        "support_scoring": {
            "primary_real_reference": ["subset_context_vs_reference",
                                       "full_context_vs_reference"],
            "paired_delta": "delta_subset_minus_full (subset - full; null propagates)",
            "auxiliary_support_drift": ("drift_vs_full_same_subset (full model "
                                        "output used as pseudo-reference; NOT a "
                                        "true-reference error)"),
            "note": ("main support tables score against the physical reference "
                     "Y_geom; the old vs-full comparison is kept only to separate "
                     "support change from scoring-point change"),
        },
        "aggregation_order": ("per geometry per bin -> per seed equal-weight "
                              "geometry mean -> cross-seed mean/sample std "
                              "(ddof=1)"),
        "unit_note": ("nodes/points inside one geometry are not independent "
                      "replicates; geometry-level means and seed-level "
                      "replication are reported separately"),
        "zero_denominator": ("RE null when truth L2 norm <= 1e-12 (numerical zero); "
                             "denominator norm recorded; MAE/max kept"),
        "descriptive_only": True,
    }


# ---------------------------------------------------------------------------
# 报告 / registry candidate
# ---------------------------------------------------------------------------
def _fmt(v, nd=6):
    return "n/a" if v is None else f"{v:.{nd}g}"


def write_report(path, summary, snapshot):
    lines = [
        "# R2 局部误差与支持点敏感性（Reviewer3 #8/#13 补证）",
        "",
        f"- 生成时间：{summary['created_local']}",
        f"- 输入工作区（只读）：`{snapshot['input_workspace']}`",
        f"- 模型：{summary['model']['label']}，seeds {snapshot['model']['seeds']}",
        f"- test 几何：{len(snapshot['test_ids'])} 个；full 5000 点/symmetric",
        f"- 全 5 seed 全场 RE 与 metrics JSON 的最大允许差：{RE_METRICS_TOL}；"
        "本地已在每个 seed×geometry×通道复核通过",
        f"- 边界验证：前 660 墙节点到连续边界最大距离 < {BOUNDARY_TOL}（逐几何记录）",
        "",
        "## 聚合与解释",
        "",
        "- 顺序：每 geometry 每 bin -> 每 seed 等权 geometry 均值 -> 5 seed "
        "mean/sample std（ddof=1）。",
        "- 空 bin 跳过并记录有效几何数；真值范数 <=1e-12（数值零，如壁面无滑移 u/v）"
        "时 RE=null，MAE/max 保留并记录分母范数。",
        "- 节点/几何内部点不是独立重复；相关只作描述，不作因果解释、不选路。",
        "",
        "## 墙节点组（0..659）",
        "",
        "| field | RE mean±std | MAE mean | max mean |",
        "|---|---|---|---|",
    ]
    wall = summary["local_error"]["wall:wall"]["cross_seed"]
    for field in FIELDS:
        re = wall[field]["re"]
        lines.append(f"| {field} | {_fmt(re['mean'])} ± {_fmt(re['std_sample'])} | "
                     f"{_fmt(wall[field]['mae']['mean'])} | "
                     f"{_fmt(wall[field]['max']['mean'])} |")
    lines += ["", "## 到连续边界距离的 bands（内部 4340 点）", "",
              "| band | field | RE mean±std | MAE mean | max mean | 有效几何数 |",
              "|---|---|---|---|---|---|"]
    for name in BAND_NAMES:
        entry = summary["local_error"][f"band:{name}"]
        n_geom = max((s["n_valid_geom"] for s in entry["per_seed"]), default=0)
        for field in FIELDS:
            re = entry["cross_seed"][field]["re"]
            lines.append(f"| {name} | {field} | {_fmt(re['mean'])} ± "
                         f"{_fmt(re['std_sample'])} | "
                         f"{_fmt(entry['cross_seed'][field]['mae']['mean'])} | "
                         f"{_fmt(entry['cross_seed'][field]['max']['mean'])} | "
                         f"{n_geom} |")
    lines += ["", "## 局部密度四等份（按 h16 排序；内部点）", "",
              "| quartile | field | RE mean±std | MAE mean | max mean | 有效几何数 |",
              "|---|---|---|---|---|---|"]
    for label in QUARTILE_LABELS:
        entry = summary["local_error"][f"density_quartile:{label}"]
        n_geom = max((s["n_valid_geom"] for s in entry["per_seed"]), default=0)
        for field in FIELDS:
            re = entry["cross_seed"][field]["re"]
            lines.append(f"| {label} | {field} | {_fmt(re['mean'])} ± "
                         f"{_fmt(re['std_sample'])} | "
                         f"{_fmt(entry['cross_seed'][field]['mae']['mean'])} | "
                         f"{_fmt(entry['cross_seed'][field]['max']['mean'])} | "
                         f"{n_geom} |")
    lines += ["", "## 支持点敏感性：真实 reference 误差（主表；seeds 0–2）", "",
              "主表以 Y_geom 物理真值打分；full-context 行是 full5000 预测在同"
              "一子集点上的真实误差。", "",
              "| N | context | field | RE mean±std | MAE mean | max mean |",
              "|---|---|---|---|---|---|"]
    for total in SUPPORT_TOTALS:
        entry = summary["support_sensitivity"][str(total)]
        for ctx, key in (("subset-context", "subset_context_vs_reference"),
                         ("full-context", "full_context_vs_reference")):
            for field in FIELDS:
                cell = entry["cross_seed"][key][field]
                re = cell["re"]
                lines.append(f"| {total} | {ctx} | {field} | "
                             f"{_fmt(re['mean'])} ± {_fmt(re['std_sample'])} | "
                             f"{_fmt(cell['mae']['mean'])} | "
                             f"{_fmt(cell['max']['mean'])} |")
    lines += ["", "## 配对差（subset − full，同一 ids；null 传播）", "",
              "| N | field | ΔRE mean±std | ΔMAE mean | Δmax mean |",
              "|---|---|---|---|---|"]
    for total in SUPPORT_TOTALS:
        entry = summary["support_sensitivity"][str(total)]
        for field in FIELDS:
            cell = entry["cross_seed"]["delta_subset_minus_full"][field]
            lines.append(f"| {total} | {field} | "
                         f"{_fmt(cell['re']['mean'])} ± "
                         f"{_fmt(cell['re']['std_sample'])} | "
                         f"{_fmt(cell['mae']['mean'])} | "
                         f"{_fmt(cell['max']['mean'])} |")
    lines += ["", "## 支持漂移诊断（full 模型当参照；不是真实误差）", "",
              "| N | field | RE(vs full) mean±std | MAE | max | "
              "max abs diff vs full(全体几何) |",
              "|---|---|---|---|---|---|"]
    for total in SUPPORT_TOTALS:
        entry = summary["support_sensitivity"][str(total)]
        max_diff = max(s["max_abs_diff_vs_full_over_geoms"]
                       for s in entry["per_seed"])
        for field in FIELDS:
            cell = entry["cross_seed"]["drift_vs_full_same_subset"][field]
            lines.append(f"| {total} | {field} | "
                         f"{_fmt(cell['re']['mean'])} ± "
                         f"{_fmt(cell['re']['std_sample'])} | "
                         f"{_fmt(cell['mae']['mean'])} | "
                         f"{_fmt(cell['max']['mean'])} | {_fmt(max_diff)} |")
    lines += [
        "",
        "## 说明",
        "",
        "- 支持集子集为同一 fixed rng(20261021+geomID) 打乱内部索引后嵌套截取；"
        "wall 660 点全部保留；每子集实际重算 symmetric context，不重训、不重网格。",
        "- 主表/配对差以物理真值 Y_geom 打分；full-context 是 full5000 预测在"
        "同一子集点上的真实 reference 误差，配对差才是支撑变化对应的真实误差差；"
        "旧 vs-full 表仅为支持漂移诊断，不能回答真实误差。",
        "- 结果不是独立重网格、新坐标真值或重训低密度。",
        "- N=5000 行应与 full 一致（容差内校验并记录）。",
        "- 局部距离是到**连续**内/外边界（多边形/正方形线段）的精确投影距离；"
        "band 只描述误差随距离变化，不作因果或选路。",
        "",
        f"- 原始数据：`raw_local_error.json`、`raw_support_sensitivity.json`、"
        f"`support_subsets.json`；协议：`protocol_snapshot.json`。",
        "",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def write_registry_candidate(out_dir, summary, snapshot):
    entries = [
        {
            "id": "R2_LOCAL_ERROR",
            "experiment_id": "R2_LOCAL_ERROR",
            "type": "analysis",
            "tool": "exp_r2_local_error.py",
            "r2_task": "R2-LOCAL-ERROR-SUPPORT-01",
            "protocol": {
                "split_version": snapshot["split_version"],
                "test_ids": snapshot["test_ids"],
                "model": snapshot["model"],
                "geometry_spec_sha256": snapshot["geometry_spec_sha256"],
                "data_sha256": snapshot["data_sha256"],
                "bands": snapshot["analysis"]["bands"],
                "density": snapshot["analysis"]["density"],
                "aggregation": snapshot["analysis"]["aggregation"],
            },
            "metrics": {"file": os.path.join(out_dir, "summary.json")},
            "note": "REGISTRY CANDIDATE ONLY - not appended by this script",
        },
        {
            "id": "R2_SUPPORT_SENSITIVITY",
            "experiment_id": "R2_SUPPORT_SENSITIVITY",
            "type": "analysis",
            "tool": "exp_r2_local_error.py",
            "r2_task": "R2-LOCAL-ERROR-SUPPORT-01",
            "protocol": {
                "seeds": list(SUPPORT_SEEDS),
                "nested_totals": list(SUPPORT_TOTALS),
                "rng": f"np.random.default_rng({SUPPORT_RNG_BASE}+geom_id)",
                "scoring": ("real physical reference Y_geom for subset_context and "
                            "full_context on the same ids; paired delta subset-full "
                            "(null propagates); drift_vs_full is auxiliary only"),
                "note": "support-change vs scoring-point-change; not retraining",
            },
            "metrics": {"file": os.path.join(out_dir, "summary.json")},
            "note": "REGISTRY CANDIDATE ONLY - not appended by this script",
        },
    ]
    return {"appended": False,
            "note": "Codex merges; this script never writes results/registry.json",
            "entries": entries}


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    torch.set_num_threads(N_THREADS)
    INPUT, OUT, DATA, metrics_paths, split_path, spec_path = resolve_inputs(args)
    if os.path.isdir(OUT):
        _require(not os.listdir(OUT),
                 f"output dir is not empty: {OUT}; refusing to overwrite")
    for name in OUT_FILES:
        path = os.path.join(OUT, name)
        _require(not os.path.exists(path),
                 f"target already exists: {path}; refusing to overwrite")
    inputs = load_and_validate_inputs(INPUT, DATA, metrics_paths, split_path,
                                      spec_path)
    X, Y = inputs["X"], inputs["Y"]
    X_double = inputs["X_double"]
    test_ids = inputs["test_ids"]
    device = torch.device("cpu")

    # 1) 几何重建/边界验证（原 double 坐标；在任何预测前记录，失败即停）
    boundary_checks = {}
    specs = {}
    for g in test_ids:
        spec = inputs["per_geo"][g]
        specs[g] = spec
        d, _, _ = boundary_distance(X_double[g][:N_BOUNDARY], spec)
        max_d = float(d.max())
        boundary_checks[str(g)] = max_d
        _require(max_d < BOUNDARY_TOL,
                 f"geom {g}: first 660 nodes are not on the reconstructed "
                 f"continuous boundary (max distance {max_d:.3e} >= {BOUNDARY_TOL}); "
                 "refusing to guess the polygon phase")

    # 2) 运行前协议快照
    os.makedirs(OUT, exist_ok=True)
    snapshot = build_snapshot(args, INPUT, OUT, DATA, metrics_paths, split_path,
                              spec_path, inputs, boundary_checks)
    _write_json(os.path.join(OUT, "protocol_snapshot.json"), snapshot)
    print(f"[snapshot] {OUT}/protocol_snapshot.json", flush=True)

    # 3) 每 geometry 的密度/距离（与 seed 无关，先算一次）
    h16 = {}
    density = {}
    for g in test_ids:
        h16[g], density[g] = h16_density(X_double[g])
    subsets = build_support_subsets(test_ids)
    _write_json(os.path.join(OUT, "support_subsets.json"), subsets)

    raw_local = {}
    raw_support = {}
    labels = sorted(inputs["labels"])[0]
    for seed in SEEDS:
        model, _label, _ckpt_path = load_seed_model(
            INPUT, seed, inputs["metrics_docs"][seed], device)
        rows = {int(r["geom_id"]): r for r in
                inputs["metrics_docs"][seed]["test"]["rows"]}
        seed_local = []
        seed_support = []
        for g in test_ids:
            X_geom = X[g]
            truth = Y[g].astype(np.float64)
            pred = predict_full(model, X_geom, inputs["stats_by_seed"][seed],
                                device)
            full_check = full_metric_check(pred, truth, rows[g], seed, g)
            local = local_error_for_geometry(
                X_double[g], pred, truth, specs[g], h16[g], density[g])
            local["geom_id"] = int(g)
            local["full_field_metrics"] = full_check
            seed_local.append(local)
            if seed in SUPPORT_SEEDS:
                sup = {"geom_id": int(g),
                       "subsets": support_for_geometry(
                           model, X_geom, inputs["stats_by_seed"][seed], device,
                           subsets[g], pred, truth)}
                seed_support.append(sup)
        raw_local[str(seed)] = seed_local
        if seed in SUPPORT_SEEDS:
            raw_support[str(seed)] = seed_support
        print(f"[seed {seed}] {len(seed_local)} geometries analysed", flush=True)
        del model
        _cleanup()

    _write_json(os.path.join(OUT, "raw_local_error.json"), raw_local)
    _write_json(os.path.join(OUT, "raw_support_sensitivity.json"), raw_support)
    summary = build_summary(raw_local, raw_support, test_ids, labels)
    _write_json(os.path.join(OUT, "summary.json"), summary)
    write_report(os.path.join(OUT, "report.md"), summary, snapshot)
    candidate = write_registry_candidate(OUT, summary, snapshot)
    _write_json(os.path.join(OUT, "registry_candidate.json"), candidate)
    print(f"[done] {OUT}", flush=True)


def _cleanup():
    import gc
    gc.collect()


if __name__ == "__main__":
    main()
