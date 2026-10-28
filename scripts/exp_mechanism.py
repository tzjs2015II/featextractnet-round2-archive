"""Mechanism analysis for the two claims challenged in the review (WP1).

Implements ``docs/M_机理分析设计.md``:

  --mode gate   (M1) CGF gate values vs wall distance / vorticity / |grad T|,
                using the frozen Protocol-A checkpoints. Test geometries are only
                *analysed*, never used for any selection.
  --mode grad   (M2) Controlled pooling comparison: layer-wise gradient norms as a
                function of the number of points N, for the global-pooling context
                and the spatial-grid context, at random init and after a short
                controlled training. A synthetic coordinate set is used and is
                labelled as such; it probes the architecture's N-dependence, it is
                not a physical case.
  --mode grad-traj (M2b) Per-layer gradient-norm trajectories during a short real
                training on the thermal train split (100 epochs, seed 0).

Why the gradient probe is designed this way
-------------------------------------------
A layer-wise gradient-norm curve alone cannot separate "global pooling attenuates
gradients as N grows" from ordinary depth-related attenuation. The discriminating
comparison is therefore a controlled one: same data, same weights, same loss, only
the context branch switched (``use_spatial_grid`` False/True), swept over N. The
decision rule is fixed in advance (design doc §2.2): the claim holds only if the
global-pooling per-point gradient norm falls off with N while the spatial-grid one
does not.

Inputs (--mode gate, corrected R2 entry)
----------------------------------------
  default checkpoint root: results/round2_thermal_symmetric/feat_extract
  results/round2_thermal_symmetric/feat_extract/seed{0..4}/checkpoint_best.pth
  .../seed{k}/metrics_seed{k}.json  (pool_context / split counts / train-only stats)
  configs/splits_thermal.json       (compared with the metrics-recorded snapshot)
  HeatTransferData.npy              (auto-detected, or --data)

Outputs (--mode gate, corrected)
--------------------------------
  results/round2_mechanism/gate_stats_seed{k}.json
  results/round2_mechanism/gate_summary.json      (5-seed aggregate + protocol)
  results/round2_mechanism/effective_protocol.json
  results/round2_mechanism/M1_R2_symmetric_correctcurl_report.md
  results/figures/round2_mechanism/gate_vs_physics.png  (skipped if no matplotlib)
  results/registry.json entry M1_R2_symmetric_correctcurl (full 5-seed run only)

Legacy / untouched paths
------------------------
  --mode grad keeps the historical defaults and outputs:
  results/mechanism/grad_norm_vs_N.json, grad_traj_*.json,
  results/figures/grad_attenuation_vs_N.png.
  The legacy gate results (results/mechanism/gate_stats_seed*.json) and the
  legacy figure (results/figures/gate_vs_physics.png) are never overwritten.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WS)

from scipy.spatial import cKDTree  # noqa: E402
from scipy import stats as sps  # noqa: E402

from src.core import thermal_io  # noqa: E402
from src.core.registry import append_registry  # noqa: E402
from src.models.featextractnet import FeatExtractNet  # noqa: E402
from src.utils.data_loader import set_seed  # noqa: E402
from src.utils.device import device_hardware_summary, get_device  # noqa: E402

PROTO_DIR = os.path.join(WS, "results", "thermal_protocolA")
CKPT_DIR = os.path.join(PROTO_DIR, "feat_extract")
OUT_DIR = os.path.join(WS, "results", "mechanism")
FIG_DIR = os.path.join(WS, "results", "figures")
SPLIT_PATH = os.path.join(WS, "configs", "splits_thermal.json")
FIELD_NAMES = ("u", "v", "p", "T")
SEEDS = (0, 1, 2, 3, 4)
BOUNDARY_N = thermal_io.DEFAULT_BOUNDARY_N
KNN_K = 24
N_POINTS = 5000
RIDGE = 1e-8
SPLIT_VERSION = "splits_thermal_v1_protocolA"
POOL_CONTEXTS = ("legacy_index_bins", "symmetric")

# Corrected R2 gate defaults: independent round2 outputs; legacy dirs are protected.
R2_SYMMETRIC_CKPT_ROOT = os.path.join(WS, "results", "round2_thermal_symmetric",
                                      "feat_extract")
R2_GATE_OUT_DIR = os.path.join(WS, "results", "round2_mechanism")
R2_GATE_FIG_DIR = os.path.join(WS, "results", "figures", "round2_mechanism")
REGISTRY_ID = "M1_R2_symmetric_correctcurl"
GATE_RELATIONS = (
    ("gate_vs_wall_dist", "distance to sampled boundary points"),
    ("gate_vs_vorticity", "|dv/dx - du/dy| from reference u,v"),
    ("gate_vs_gradT", "||grad T|| from reference T"),
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def load_thermal_data():
    for cand in (os.path.join(WS, "data", "thermal", "HeatTransferData.npy"),
                 os.path.join(os.path.dirname(os.path.dirname(WS)),
                              "HeatTransferData.npy")):
        if os.path.exists(cand):
            return thermal_io.load_thermal(cand)
    raise FileNotFoundError("HeatTransferData.npy not found")


def load_protocol_a():
    with open(SPLIT_PATH, "r", encoding="utf-8") as fh:
        sp = json.load(fh)
    pa = sp["protocol_A"]
    return ([int(i) for i in pa["train"]], [int(i) for i in pa["val"]],
            [int(i) for i in pa["test"]])


def load_ckpt_stats(seed):
    """Normalisation stats recorded by the run (train-only z-score)."""
    path = os.path.join(CKPT_DIR, f"seed{seed}", f"metrics_seed{seed}.json")
    with open(path, "r", encoding="utf-8") as fh:
        proto = json.load(fh)["protocol"]
    return {k: np.asarray(proto[k], dtype=np.float32)
            for k in ("x_mean", "x_std", "y_mean", "y_std")}


def build_featextract():
    return FeatExtractNet(input_channels=2, output_channels=4, scaling=1.0)


def as_btc(arr, device):
    """Convert to the model's [B, C, N] input layout on ``device``.

    Accepts ``[N, C]`` (single geometry -> ``[1, C, N]``) or ``[B, N, C]``
    (geometric batch -> ``[B, C, N]``).
    """
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    if t.ndim == 2:
        return t.T.unsqueeze(0).to(device)
    if t.ndim == 3:
        return t.permute(0, 2, 1).to(device)
    raise ValueError(f"expected 2-D or 3-D array, got shape {tuple(t.shape)}")


def load_state(model, seed):
    path = os.path.join(CKPT_DIR, f"seed{seed}", "checkpoint_best.pth")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt


def knn_local_gradient(pts, vals, k=KNN_K, ridge=1e-8):
    """Per-point gradient of a scalar field on a scattered point cloud.

    Fits ``v ~ a + b*dx + c*dy`` over the k nearest neighbours of each point by
    ridge-regularised least squares; ``(b, c)`` estimates ``grad v``. Scattered
    finite differences are unreliable here because the cloud is unstructured.
    """
    pts = np.asarray(pts, dtype=np.float64)
    vals = np.asarray(vals, dtype=np.float64)
    n = pts.shape[0]
    k = min(k, n)
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=k)
    if idx.ndim == 1:
        idx = idx[:, None]
    d = pts[idx] - pts[:, None, :]                      # [N,k,2]
    A = np.concatenate([np.ones((n, k, 1)), d], axis=2)  # [N,k,3]
    ata = np.einsum("nki,nkj->nij", A, A) + ridge * np.eye(3)
    atv = np.einsum("nki,nk->ni", A, vals[idx])
    # solve expects a stacked (..., m, m) / (..., m, n) pair, so keep the
    # right-hand side as (N, 3, 1) instead of (N, 3)
    coef = np.linalg.solve(ata, atv[..., None])[..., 0]   # [N,3]
    return coef[:, 1], coef[:, 2]


def wall_distance(pts, n_boundary=BOUNDARY_N):
    """Distance from each point to the nearest *sampled* boundary point.

    The first ``n_boundary`` points of every geometry are the boundary/near-wall
    points (environment fact). ``cKDTree`` is queried with ``k=1``: a boundary
    point is its own nearest sampled point, so its distance is exactly 0; the
    second-nearest point is deliberately not used. This is a distance to sampled
    boundary points, not an exact continuous wall distance.
    """
    pts = np.asarray(pts, dtype=np.float64)
    boundary = pts[:n_boundary]
    tree = cKDTree(boundary)
    d, _ = tree.query(pts, k=1)
    return np.asarray(d, dtype=np.float64)


# ---------------------------------------------------------------------------
# corrected R2 gate helpers (M1_R2_symmetric_correctcurl)
# ---------------------------------------------------------------------------
def _require(cond, msg):
    if not cond:
        raise SystemExit("[refuse] " + msg)


def _normpath(path):
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def sha256_file(path):
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_data_path(explicit=None):
    if explicit is not None:
        path = _normpath(explicit)
        _require(os.path.isfile(path), f"--data not found: {explicit}")
        return path
    for cand in (os.path.join(WS, "data", "thermal", "HeatTransferData.npy"),
                 os.path.join(os.path.dirname(os.path.dirname(WS)),
                              "HeatTransferData.npy")):
        if os.path.isfile(cand):
            return _normpath(cand)
    raise SystemExit("[fail] HeatTransferData.npy not found; pass --data")


def load_splits_config():
    with open(SPLIT_PATH, "r", encoding="utf-8") as fh:
        sp = json.load(fh)
    _require(str(sp.get("version", "")) != "", "splits config has no version field")
    _require(int(sp.get("split_seed", -1)) == 123, "splits split_seed != 123")
    _require(int(sp.get("n_geometries", -1)) == 135, "splits n_geometries != 135")
    pa = sp["protocol_A"]
    ids = {k: [int(i) for i in pa[k]] for k in ("train", "val", "test")}
    for k, n in (("train", 100), ("val", 15), ("test", 20)):
        _require(len(ids[k]) == n,
                 f"protocol_A {k} has {len(ids[k])} geoms, expected {n}")
        _require(len(set(ids[k])) == len(ids[k]),
                 f"protocol_A {k} contains duplicate ids")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        _require(not (set(ids[a]) & set(ids[b])), f"protocol_A {a}/{b} overlap")
    _require(set().union(*(set(v) for v in ids.values())) <= set(range(135)),
             "protocol_A ids outside 0..134")
    return sp, ids


def compare_splits_snapshot(proto, ids):
    """Compare the split ids used here with the config snapshot recorded by the run.

    The split ids themselves are not stored in metrics_seed*.json, so the recorded
    ``splits_config`` snapshot is the only exact evidence of which geometries were
    trained on. If it cannot be read, the analysis refuses instead of silently
    falling back to the workspace split file.
    """
    snap = proto.get("splits_config")
    _require(bool(snap),
             "metrics protocol records no splits_config; cannot verify train/val/"
             "test ids")
    norm = str(snap).replace("\\", "/")
    _require(os.path.basename(norm) == "splits_thermal.json",
             f"metrics splits_config basename is not splits_thermal.json: {snap}")
    _require(norm.endswith("configs/splits_thermal.json"),
             f"metrics splits_config is not configs/splits_thermal.json: {snap}")
    _require(os.path.isfile(snap),
             f"metrics-recorded split snapshot not found: {snap}; cannot verify "
             "train/val/test ids, refusing")
    with open(snap, "r", encoding="utf-8") as fh:
        snap_doc = json.load(fh)
    for k in ("train", "val", "test"):
        got = [int(i) for i in snap_doc["protocol_A"][k]]
        _require(got == ids[k],
                 f"split snapshot {k} ids differ from configs/splits_thermal.json")
    return {"metrics_splits_config": snap, "snapshot_found": True,
            "snapshot_matches": True}


def validate_gate_protocol(proto, pool_context, ids):
    """Refuse a gate run whose metrics protocol does not match the requested model."""
    _require(pool_context in POOL_CONTEXTS, f"unknown pool_context {pool_context!r}")
    _require(proto.get("split_version") == SPLIT_VERSION,
             f"metrics split_version {proto.get('split_version')!r} != "
             f"{SPLIT_VERSION!r}")
    for key, name in (("n_train_geom", "train"), ("n_val_geom", "val"),
                      ("n_test_geom", "test")):
        _require(int(proto.get(key, -1)) == len(ids[name]),
                 f"metrics {key}={proto.get(key)!r} != {len(ids[name])} ({name})")
    _require(int(proto.get("split_seed", -1)) == 123, "metrics split_seed != 123")
    ctx = proto.get("pool_context", "legacy_index_bins")
    _require(ctx == pool_context,
             f"metrics pool_context={ctx!r} != requested {pool_context!r}; "
             "refusing to mix model families")
    label = str(proto.get("model_label", ""))
    if pool_context == "symmetric":
        _require("pool_mode=symmetric" in label,
                 f"metrics model_label {label!r} does not record pool_mode=symmetric")
        _require(proto.get("cuda_matmul_allow_tf32") is False
                 and proto.get("cudnn_allow_tf32") is False,
                 "symmetric metrics record TF32 enabled; the R2 symmetric analysis "
                 "requires TF32 off")
    else:
        _require("pool_mode=symmetric" not in label,
                 f"metrics model_label {label!r} is a symmetric run; "
                 "legacy_index_bins was requested")
    _require("train-only" in str(proto.get("norm", "")),
             f"metrics norm source is not train-only: {proto.get('norm')!r}")
    return {"pool_context": ctx, "model_label": label,
            "implementation_version": proto.get("implementation_version"),
            "tf32_recorded_off": (proto.get("cuda_matmul_allow_tf32") is False
                                  and proto.get("cudnn_allow_tf32") is False)}


def validate_norm_stats_consistency(stats_list):
    """The 5 seeds must share one train-only normalization (bit-identical floats)."""
    _require(len(stats_list) > 0, "no normalization stats to compare")
    base = stats_list[0]
    for key in ("x_mean", "x_std", "y_mean", "y_std"):
        ref = np.asarray(base[key], dtype=np.float32)
        for i, stats in enumerate(stats_list[1:], start=1):
            cur = np.asarray(stats[key], dtype=np.float32)
            _require(cur.shape == ref.shape,
                     f"norm stat {key} shape differs at seed entry {i}")
            if not np.array_equal(cur, ref):
                raise SystemExit(
                    f"[refuse] normalization stat {key} differs across seeds; the "
                    "5-seed gate analysis requires one shared train-only "
                    "normalization")
    return True


def verify_norm_stats_source(stats, X, Y, train_ids):
    """Recompute the train-only fit from the data and refuse if it disagrees."""
    fitted = thermal_io.fit_stats(X, Y, geom_ids=train_ids)
    for key in ("x_mean", "x_std", "y_mean", "y_std"):
        ref = np.asarray(fitted[key], dtype=np.float64)
        rec = np.asarray(stats[key], dtype=np.float64)
        if rec.shape != ref.shape or not np.allclose(rec, ref, rtol=1e-5, atol=1e-7):
            raise SystemExit(
                f"[refuse] metrics norm stat {key} cannot be reproduced from the "
                "train-only fit on the training geometries")
    return True


def assert_model_pool_context(model, pool_context):
    actual = getattr(model, "pool_mode", None)
    if actual != pool_context:
        raise SystemExit(
            f"[refuse] model pool_mode={actual!r} != requested {pool_context!r}; "
            "refusing to pair a checkpoint with a mismatched model configuration")
    return actual


def build_gate_model(pool_context):
    model = FeatExtractNet(input_channels=2, output_channels=4, scaling=1.0,
                           pool_mode=pool_context)
    assert_model_pool_context(model, pool_context)
    return model


def verify_checkpoint_metadata(ckpt, metrics_doc, seed):
    _require(isinstance(ckpt, dict) and "model_state_dict" in ckpt,
             "checkpoint has no model_state_dict entry")
    _require(int(ckpt.get("seed", -1)) == int(seed),
             f"checkpoint seed={ckpt.get('seed')!r} != metrics seed {seed}")
    best = metrics_doc.get("val_best") or {}
    _require(int(ckpt.get("epoch", -1)) == int(best.get("epoch", -2)),
             f"checkpoint epoch={ckpt.get('epoch')!r} != metrics "
             f"val_best.epoch={best.get('epoch')!r}")
    score = ckpt.get("val_score")
    _require(score is not None
             and np.isclose(float(score), float(best.get("score", np.nan)),
                            rtol=1e-6, atol=1e-9),
             f"checkpoint val_score={score!r} != metrics "
             f"val_best.score={best.get('score')!r}")
    return True


def check_state_dict_keys(state, model, label="checkpoint"):
    """Strict key-set match: refuse a checkpoint built for another configuration."""
    expected = set(model.state_dict().keys())
    got = set(state.keys())
    if got != expected:
        raise SystemExit(
            f"[refuse] {label} state_dict keys do not match the model "
            f"configuration (missing={sorted(expected - got)[:5]}, "
            f"unexpected={sorted(got - expected)[:5]})")
    return True


def load_gate_checkpoint(model, ckpt_root, seed, metrics_doc):
    path = os.path.join(ckpt_root, f"seed{seed}", "checkpoint_best.pth")
    _require(os.path.isfile(path), f"checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    _require(isinstance(ckpt, dict) and "model_state_dict" in ckpt,
             f"checkpoint {path} has no model_state_dict entry")
    check_state_dict_keys(ckpt["model_state_dict"], model, label=path)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    verify_checkpoint_metadata(ckpt, metrics_doc, seed)
    return path, ckpt


def gate_report_name(pool_context):
    if pool_context == "symmetric":
        return f"{REGISTRY_ID}_report.md"
    return f"M1_gate_{pool_context}_report.md"


def check_gate_outputs_clear(out_dir, fig_dir, pool_context="symmetric"):
    """Refuse legacy dirs and any existing corrected seed JSON / report / figure."""
    legacy_out = _normpath(OUT_DIR)
    legacy_fig = _normpath(FIG_DIR)
    out_n = _normpath(out_dir)
    fig_n = _normpath(fig_dir)
    _require(not (out_n == legacy_out or out_n.startswith(legacy_out + os.sep)),
             f"gate --out-dir points into the legacy mechanism results {OUT_DIR}; "
             "corrected output must be an independent round2 directory and must "
             "not overwrite legacy mechanism results")
    _require(fig_n != legacy_fig,
             f"gate --fig-dir must not be the legacy figure root {FIG_DIR}")
    for label, path in (("--out-dir", out_n), ("--fig-dir", fig_n)):
        parts = [p for p in path.replace("\\", "/").split("/") if p]
        has_marker = any(("round2" in p.lower() or "r2" in p.lower())
                         for p in parts)
        _require(has_marker, f"gate {label} has no round2/r2 path component: {path}")
    planned = [os.path.join(out_dir, f"gate_stats_seed{s}.json") for s in SEEDS]
    planned += [os.path.join(out_dir, "gate_summary.json"),
                os.path.join(out_dir, "effective_protocol.json"),
                os.path.join(out_dir, gate_report_name(pool_context)),
                os.path.join(fig_dir, "gate_vs_physics.png")]
    for path in planned:
        if os.path.exists(path):
            raise SystemExit(
                f"[refuse] gate target already exists: {path}; refusing to "
                "overwrite (choose a fresh --out-dir/--fig-dir)")
    return True


def configure_precision(pool_context):
    """Symmetric analysis runs full FP32 (TF32 off); returns the effective flags."""
    if pool_context == "symmetric":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return {"cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32)}


def check_full_field(X):
    _require(X.ndim == 3 and X.shape[1] == N_POINTS,
             f"gate analysis requires the full {N_POINTS}-point field per "
             f"geometry (no chunking); got shape {tuple(X.shape)}")
    return int(X.shape[1])


def _seed_summary(rows):
    out = {}
    for key, _label in GATE_RELATIONS:
        out[key] = {"n_geom": int(len(rows))}
        for stat in ("pearson_r", "spearman_r"):
            vals = np.asarray([r[key][stat] for r in rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            out[key][f"{stat}_geom_mean"] = (float(vals.mean()) if vals.size
                                             else float("nan"))
            out[key][f"{stat}_geom_std_sample"] = (float(vals.std(ddof=1))
                                                   if vals.size > 1
                                                   else float("nan"))
    return out


def _overall_summary(per_seed):
    """Cross-seed mean and sample std of the 5 per-seed geometry means."""
    out = {}
    for key, _label in GATE_RELATIONS:
        out[key] = {}
        for stat in ("pearson_r", "spearman_r"):
            seed_means = np.asarray(
                [rec["summary"][key][f"{stat}_geom_mean"] for rec in per_seed],
                dtype=np.float64)
            finite = seed_means[np.isfinite(seed_means)]
            out[key][f"{stat}_cross_seed_mean"] = (float(finite.mean())
                                                   if finite.size else float("nan"))
            out[key][f"{stat}_cross_seed_sample_std"] = (
                float(finite.std(ddof=1)) if finite.size > 1 else float("nan"))
            out[key][f"{stat}_seed_geom_means"] = [float(v) for v in seed_means]
    out["n_seeds"] = int(len(per_seed))
    out["n_geom_per_seed"] = (int(per_seed[0]["summary"][GATE_RELATIONS[0][0]]["n_geom"])
                              if per_seed else 0)
    out["statistical_unit"] = (
        "Each seed value is the mean over the 20 test geometries; the cross-seed "
        "mean and sample std (ddof=1) are over the 5 seeds. Per-geometry p-values "
        "describe within-geometry association, not global significance; 5 seeds x "
        "20 geometries are not 100 independent samples.")
    return out


def _write_json(path, doc):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def _build_effective_protocol(pool_context, ckpt_root, out_dir, fig_dir, data_path,
                              ids, seed_entries, test_ids, tf32, ridge, n_pts,
                              dev_type, snap_info):
    return {
        "analysis": f"M1 corrected gate-vs-physics ({pool_context})",
        "mode": "gate",
        "pool_context": pool_context,
        "checkpoint_root": ckpt_root,
        "out_dir": out_dir,
        "fig_dir": fig_dir,
        "seeds": list(SEEDS),
        "test_geoms": list(test_ids),
        "n_test_geom": int(len(test_ids)),
        "n_geometry_dataset": 135,
        "split_version": SPLIT_VERSION,
        "split_seed": 123,
        "split_counts": {k: len(v) for k, v in ids.items()},
        "splits_snapshot": snap_info,
        "data": data_path,
        "n_points_full_field": int(n_pts),
        "chunked": False,
        "boundary_n": int(BOUNDARY_N),
        "knn_k": int(KNN_K),
        "ridge": float(ridge),
        "gate_def": ("sigmoid of the real outputs of gate_low (64 ch), gate_mid "
                     "(128 ch) and gate_high (256 ch), concatenated (448 channels) "
                     "and averaged over those 448 channels per point; hooks on the "
                     "actual gate Conv1d modules"),
        "curl_def": ("|dv/dx - du/dy| per point, estimated by the k=24 nearest-"
                     "neighbour local linear fit from the reference (ground-truth) "
                     "u and v fields, not from model predictions"),
        "gradT_def": ("||grad T|| per point from the reference (ground-truth) T "
                      "field, k=24 nearest-neighbour local linear fit"),
        "wall_dist_def": (
            "Euclidean distance to the nearest sampled boundary point (the first "
            f"{BOUNDARY_N} points of each geometry; cKDTree k=1). Boundary points "
            "themselves are 0 because they are their own nearest sampled point; "
            "the second-nearest point is not used. This is distance to sampled "
            "boundary points, not an exact continuous wall distance"),
        "tf32": tf32,
        "device_type": dev_type,
        "norm": {
            "source": ("per-seed metrics protocol (train-only z-score, fit on the "
                       "100 Protocol-A train geoms)"),
            "identical_across_seeds": True,
            "reproduced_from_train_geoms": True,
        },
        "statistics_note": (
            "Descriptive association only; no causal interpretation and the sign "
            "of any correlation is not presupposed. Statistical unit: seed mean "
            "over the 20 test geometries; cross-seed mean/sample-std over 5 seeds."),
        "hashes": {
            "script_sha256": sha256_file(os.path.abspath(__file__)),
            "data_sha256": sha256_file(data_path),
            "checkpoint_sha256": {str(e["seed"]): sha256_file(
                os.path.join(ckpt_root, f"seed{e['seed']}",
                             "checkpoint_best.pth")) for e in seed_entries},
            "metrics_sha256": {str(e["seed"]): sha256_file(e["metrics_path"])
                               for e in seed_entries},
        },
    }


def write_gate_summary(out_dir, effective, per_seed, overall, fig_path):
    doc = {
        "registry_id": REGISTRY_ID,
        "generated_local": effective["generated_local"],
        "partial": bool(effective.get("partial", False)),
        "effective_protocol": effective,
        "per_seed": [{"seed": rec["seed"], "n_test_geom": rec["n_test_geom"],
                      "summary": rec["summary"]} for rec in per_seed],
        "overall": overall,
        "figure": (os.path.relpath(fig_path, WS) if fig_path else None),
        "notes": {
            "reading": ("descriptive associations; no causal claim; the sign of "
                        "any correlation is not presupposed"),
            "statistics": overall["statistical_unit"],
        },
    }
    _write_json(os.path.join(out_dir, "gate_summary.json"), doc)
    _write_json(os.path.join(out_dir, "effective_protocol.json"), effective)
    return doc


def write_gate_report(out_dir, pool_context, effective, per_seed, overall, fig_path):
    path = os.path.join(out_dir, gate_report_name(pool_context))
    keys = [k for k, _ in GATE_RELATIONS]
    labels = dict(GATE_RELATIONS)
    seeds = [rec["seed"] for rec in per_seed]
    lines = [
        f"# M1 corrected gate-vs-physics ({pool_context})",
        "",
        f"- generated: {effective['generated_local']}",
        f"- checkpoint root: `{effective['checkpoint_root']}`",
        f"- seeds: {effective['seeds']} (n={len(per_seed)})",
        f"- test geometries: {effective['n_test_geom']} per seed "
        f"({effective['split_version']}, split_seed 123)",
        f"- full-field points: {effective['n_points_full_field']} (no chunking)",
        f"- device: {effective['device_type']}; TF32 flags: {effective['tf32']}",
        f"- gate definition: {effective['gate_def']}",
        f"- curl: {effective['curl_def']}",
        f"- grad T: {effective['gradT_def']}",
        f"- wall distance: {effective['wall_dist_def']}",
        "",
        "## Pearson / Spearman r",
        "",
        "| relation | " + " | ".join(f"seed{s} geom mean" for s in seeds)
        + " | cross-seed mean ± sample std (n=5) |",
        "|---" * (len(seeds) + 2) + "|",
    ]
    for key in keys:
        for stat, name in (("pearson_r", "Pearson"), ("spearman_r", "Spearman")):
            vals = " | ".join(f"{rec['summary'][key][stat + '_geom_mean']:+.4f}"
                              for rec in per_seed)
            mean = overall[key][stat + "_cross_seed_mean"]
            std = overall[key][stat + "_cross_seed_sample_std"]
            lines.append(f"| {labels[key]} ({name}) | {vals} | "
                         f"{mean:+.4f} ± {std:.4f} |")
    lines += [
        "",
        "## Reading notes",
        "",
        "- Correlations are descriptive associations; no causal interpretation and "
        "the sign is not presupposed.",
        "- Statistical unit: each seed value is the mean over 20 test geometries; "
        "the cross-seed mean and sample std (ddof=1) are over 5 seeds. Per-geometry "
        "p-values describe within-geometry association only and are not interpreted "
        "as global significance.",
        "- Vorticity uses the corrected 2-D curl magnitude |dv/dx - du/dy| from the "
        "reference u,v (the legacy |du/dx - dv/dy| is not a curl).",
        "- Wall distance is distance to sampled boundary points, not exact continuous "
        "wall distance; boundary points are 0 (cKDTree k=1, no second-nearest "
        "fallback).",
        "",
        "## Sources",
        "",
        f"- script sha256: `{effective['hashes']['script_sha256']}`",
        f"- data sha256: `{effective['hashes']['data_sha256']}`",
        "- per-seed checkpoint/metrics sha256: see `effective_protocol.json`",
        f"- figure: {os.path.relpath(fig_path, WS) if fig_path else 'not generated'}",
        "",
        "## Registry",
        "",
        (f"- id: `{REGISTRY_ID}` (registered only after a complete 5-seed run)"
         if pool_context == "symmetric" else "- legacy rerun: not registered"),
        "",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


def _corr(a, b):
    if a.size < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return {"pearson_r": float("nan"), "pearson_p": float("nan"),
                "spearman_r": float("nan"), "spearman_p": float("nan")}
    pr, pp = sps.pearsonr(a, b)
    sr, sp = sps.spearmanr(a, b)
    return {"pearson_r": float(pr), "pearson_p": float(pp),
            "spearman_r": float(sr), "spearman_p": float(sp)}


def _quantile_bins(x, y, q=10):
    edges = np.quantile(x, np.linspace(0, 1, q + 1))
    edges = np.unique(edges)
    out = []
    for i in range(len(edges) - 1):
        m = (x >= edges[i]) & (x < edges[i + 1] if i < len(edges) - 2
                               else x <= edges[i + 1])
        if m.sum() >= 5:
            out.append({"lo": float(edges[i]), "hi": float(edges[i + 1]),
                        "n": int(m.sum()), "gate_mean": float(y[m].mean()),
                        "gate_std": float(y[m].std())})
    return out


# ---------------------------------------------------------------------------
# M1: gate values vs physics
# ---------------------------------------------------------------------------
def run_gate(args):
    """M1 corrected: CGF gate value vs reference-field physics.

    Gate values come from hooks on the real gate modules of the frozen
    checkpoints; vorticity / temperature gradient come from the reference
    (ground-truth) fields; the full 5000-point field is used without chunking.
    The 5-seed run is aggregated only after every seed completes.
    """
    pool_context = args.pool_context or "symmetric"
    _require(pool_context in POOL_CONTEXTS, f"unknown pool_context {pool_context!r}")
    _require(int(args.geoms) >= 0, "--geoms must be >= 0")
    ckpt_root = (_normpath(args.checkpoint_root) if args.checkpoint_root
                 else _normpath(R2_SYMMETRIC_CKPT_ROOT))
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else R2_GATE_OUT_DIR
    fig_dir = os.path.abspath(args.fig_dir) if args.fig_dir else R2_GATE_FIG_DIR
    ridge = float(args.ridge)
    _require(ridge >= 0.0, "--ridge must be >= 0")

    check_gate_outputs_clear(out_dir, fig_dir, pool_context)
    tf32 = configure_precision(pool_context)
    _require(pool_context != "symmetric"
             or (tf32["cuda_matmul_allow_tf32"] is False
                 and tf32["cudnn_allow_tf32"] is False),
             "symmetric analysis requires TF32 disabled")

    data_path = resolve_data_path(args.data)
    _split_doc, ids = load_splits_config()
    test_ids = list(ids["test"])
    if args.geoms:
        test_ids = test_ids[: args.geoms]
    partial = bool(args.geoms)
    if partial:
        print(f"[M1] partial debug run on the first {len(test_ids)} test "
              "geometries; no aggregate/report/registry will be written",
              flush=True)

    X, Y = thermal_io.load_thermal(data_path)
    n_pts = check_full_field(X)
    _require(Y.shape[2] == len(FIELD_NAMES), "unexpected output channel count")

    seed_entries = []
    for seed in SEEDS:
        metrics_path = os.path.join(ckpt_root, f"seed{seed}",
                                    f"metrics_seed{seed}.json")
        _require(os.path.isfile(metrics_path), f"metrics not found: {metrics_path}")
        with open(metrics_path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        proto = doc.get("protocol", {})
        validate_gate_protocol(proto, pool_context, ids)
        snap_info = compare_splits_snapshot(proto, ids)
        stats = {key: np.asarray(proto[key], dtype=np.float32)
                 for key in ("x_mean", "x_std", "y_mean", "y_std")}
        seed_entries.append({"seed": seed, "metrics": doc, "proto": proto,
                             "stats": stats, "metrics_path": metrics_path,
                             "snapshot": snap_info})
    validate_norm_stats_consistency([e["stats"] for e in seed_entries])
    verify_norm_stats_source(seed_entries[0]["stats"], X, Y, ids["train"])

    # Only after every protocol / normalization check passed: create the
    # independent round2 output directories.
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    device, dev_type = get_device(args.device)

    per_seed = []
    panel = None
    for entry in seed_entries:
        seed = entry["seed"]
        stats = entry["stats"]
        doc = entry["metrics"]
        model = build_gate_model(pool_context)
        ckpt_path, _ckpt = load_gate_checkpoint(model, ckpt_root, seed, doc)
        model.to(device).eval()

        buf = {}

        def mk(name):
            def hook(_m, _i, out):
                buf[name] = out.detach()
            return hook

        handles = [model.gate_low.register_forward_hook(mk("low")),
                   model.gate_mid.register_forward_hook(mk("mid")),
                   model.gate_high.register_forward_hook(mk("high"))]

        rows = []
        for g in test_ids:
            xp = X[g]
            xn = (xp - stats["x_mean"]) / stats["x_std"]
            xb = as_btc(xn, device)
            with torch.no_grad():
                model(xb)
            gates = torch.cat([torch.sigmoid(buf["low"]),
                               torch.sigmoid(buf["mid"]),
                               torch.sigmoid(buf["high"])], dim=1)
            _require(gates.shape[1] == 448,
                     f"unexpected gate channel count {gates.shape[1]} (expected 448)")
            g_bar = gates.mean(dim=1)[0].cpu().numpy().astype(np.float64)

            # Physics from the reference fields (never from model predictions).
            u = Y[g][:, 0].astype(np.float64)
            v = Y[g][:, 1].astype(np.float64)
            T = Y[g][:, 3].astype(np.float64)
            kd = wall_distance(xp)
            du_dx, du_dy = knn_local_gradient(xp, u, k=KNN_K, ridge=ridge)
            dv_dx, dv_dy = knn_local_gradient(xp, v, k=KNN_K, ridge=ridge)
            dT_dx, dT_dy = knn_local_gradient(xp, T, k=KNN_K, ridge=ridge)
            vorticity = np.abs(dv_dx - du_dy)  # corrected 2-D curl magnitude
            gradT = np.hypot(dT_dx, dT_dy)

            rows.append({
                "geom": int(g),
                "gate_vs_wall_dist": _corr(g_bar, kd),
                "gate_vs_vorticity": _corr(g_bar, vorticity),
                "gate_vs_gradT": _corr(g_bar, gradT),
                "gate_overall": {"mean": float(g_bar.mean()),
                                 "std": float(g_bar.std()),
                                 "min": float(g_bar.min()),
                                 "max": float(g_bar.max())},
                "bins_wall_dist": _quantile_bins(kd, g_bar),
                "bins_vorticity": _quantile_bins(vorticity, g_bar),
                "bins_gradT": _quantile_bins(gradT, g_bar),
            })
            if panel is None and seed == SEEDS[0]:
                panel = {"g_bar": g_bar, "wall": kd, "vort": vorticity,
                         "gradT": gradT, "geom": int(g)}
        for h in handles:
            h.remove()

        summary = _seed_summary(rows)
        rec = {"seed": seed, "n_test_geom": len(rows), "partial": partial,
               "summary": summary, "per_geom": rows,
               "protocol": {
                   "checkpoint": os.path.relpath(ckpt_path, WS),
                   "checkpoint_sha256": sha256_file(ckpt_path),
                   "metrics": os.path.relpath(entry["metrics_path"], WS),
                   "metrics_sha256": sha256_file(entry["metrics_path"]),
                   "pool_context": pool_context,
                   "model_label": doc.get("protocol", {}).get("model_label"),
                   "model_params": doc.get("protocol", {}).get("model_params"),
                   "tf32": tf32,
                   "test_geoms": test_ids,
                   "n_test_geom": len(test_ids),
                   "n_points_full_field": n_pts,
                   "chunked": False,
                   "boundary_n": int(BOUNDARY_N),
                   "knn_k": int(KNN_K),
                   "ridge": ridge,
                   "gate_def": ("sigmoid of the real gate_low/gate_mid/gate_high "
                                "outputs, 448 channels averaged per point"),
                   "wall_dist_def": ("distance to the nearest sampled boundary "
                                     f"point (first {BOUNDARY_N} points, cKDTree "
                                     "k=1); boundary points are 0; not exact "
                                     "continuous wall distance"),
                   "vorticity_def": ("|dv/dx - du/dy| from reference u,v, kNN "
                                     "local linear fit"),
                   "gradT_def": ("||grad T|| from reference T, kNN local linear "
                                 "fit"),
                   "splits_snapshot": entry["snapshot"],
                   "device": dev_type,
                   "hardware": device_hardware_summary(),
               }}
        with open(os.path.join(out_dir, f"gate_stats_seed{seed}.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(rec, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        per_seed.append(rec)
        keys = [k for k, _ in GATE_RELATIONS]
        print(f"[M1] seed {seed}: " + " | ".join(
            f"{k} r={summary[k]['pearson_r_geom_mean']:+.3f}" for k in keys),
            flush=True)

    if partial:
        print("[M1] partial run complete; skipping aggregate/report/registry",
              flush=True)
        return 0

    overall = _overall_summary(per_seed)
    fig_path = _plot_gate(panel, overall, fig_dir) if panel is not None else None
    effective = _build_effective_protocol(
        pool_context, ckpt_root, out_dir, fig_dir, data_path, ids, seed_entries,
        test_ids, tf32, ridge, n_pts, dev_type, seed_entries[0]["snapshot"])
    effective["splits_config_version"] = _split_doc.get("version")
    from datetime import datetime
    effective["generated_local"] = datetime.now().astimezone().isoformat(
        timespec="seconds")
    effective["partial"] = False
    write_gate_summary(out_dir, effective, per_seed, overall, fig_path)
    report_path = write_gate_report(out_dir, pool_context, effective, per_seed,
                                    overall, fig_path)
    print(f"[M1] report -> {report_path}", flush=True)

    if pool_context == "symmetric" and len(per_seed) == len(SEEDS) \
            and len(test_ids) == len(ids["test"]):
        out_rel = os.path.relpath(out_dir, WS)
        append_registry({
            "id": REGISTRY_ID,
            "experiment_id": REGISTRY_ID,
            "type": "mechanism",
            "tool": os.path.basename(__file__),
            "mode": "gate",
            "pool_context": pool_context,
            "checkpoint_root": os.path.relpath(ckpt_root, WS),
            "seeds": [rec["seed"] for rec in per_seed],
            "n_seeds": len(per_seed),
            "n_test_geom": len(test_ids),
            "split_version": SPLIT_VERSION,
            "summary": overall,
            "statistical_unit": overall["statistical_unit"],
            "note": ("descriptive association only; no causal claim; the sign is "
                     "not presupposed; corrected curl and wall-distance "
                     "definitions; aggregate over the full 5-seed run, not a "
                     "seed0 summary"),
            "files": [os.path.join(out_rel, f"gate_stats_seed{s}.json")
                      for s in SEEDS]
                     + [os.path.join(out_rel, "gate_summary.json"),
                        os.path.join(out_rel, "effective_protocol.json"),
                        os.path.relpath(report_path, WS)],
            "script_sha256": effective["hashes"]["script_sha256"],
        })
    else:
        print("[M1] legacy/partial configuration: no registry entry written",
              flush=True)
    return 0


def _plot_gate(panel, overall, fig_dir):
    """Corrected 3-panel gate-vs-physics figure; the legacy figure is untouched.

    matplotlib is imported lazily so that the rest of the analysis (and the local
    probe) does not depend on a plotting install. Returns the figure path or None.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[M1] matplotlib unavailable; figure skipped (recorded as absent)",
              flush=True)
        return None
    os.makedirs(fig_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    specs = [("distance to sampled boundary points", "wall", "gate_vs_wall_dist"),
             ("|dv/dx - du/dy| (reference u,v)", "vort", "gate_vs_vorticity"),
             ("||grad T|| (reference T)", "gradT", "gate_vs_gradT")]
    for ax, (lab, key, skey) in zip(axes, specs):
        ax.scatter(panel[key], panel["g_bar"], s=2, alpha=0.15, edgecolors="none")
        q = np.quantile(panel[key], np.linspace(0, 1, 21))
        cx, cy = [], []
        for i in range(len(q) - 1):
            m = (panel[key] >= q[i]) & (panel[key] <= q[i + 1])
            if m.sum() >= 5:
                cx.append(panel[key][m].mean())
                cy.append(panel["g_bar"][m].mean())
        ax.plot(cx, cy, "o-", color="crimson", lw=1.6, ms=3,
                label="20-quantile mean")
        r = overall[skey]["pearson_r_cross_seed_mean"]
        ax.set_title(f"{lab}\ncross-seed mean Pearson r = {r:+.3f} "
                     "(5 seeds x 20 geoms)", fontsize=10)
        ax.set_xlabel(lab)
        ax.set_ylabel("mean CGF gate value")
        ax.legend(fontsize=8)
    fig.suptitle(f"M1: CGF gate value vs physical quantities "
                 f"(test geometry id {panel['geom']}, seed 0; single-geometry "
                 "panel, aggregate r shown)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(fig_dir, "gate_vs_physics.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[M1] figure -> {out}", flush=True)
    return out


# ---------------------------------------------------------------------------
# M2 / M2b: gradient norms
# ---------------------------------------------------------------------------
def _analytic_target(pts):
    """Fixed synthetic target field for the N-sweep (4 channels)."""
    x, y = pts[:, 0], pts[:, 1]
    return np.stack([np.sin(np.pi * x) * np.cos(np.pi * y),
                     np.cos(np.pi * x) * np.sin(np.pi * y),
                     np.sin(2 * np.pi * x),
                     np.cos(2 * np.pi * y)], axis=1)


def _grad_norms_at_N(model, n, device, seed=0):
    """Forward/backward one synthetic batch of N points; return gradient norms."""
    g = torch.Generator().manual_seed(seed)
    pts = torch.rand((n, 2), generator=g) * 2.0 - 1.0
    tgt = torch.from_numpy(_analytic_target(pts.numpy())).float()
    xb = pts.T.unsqueeze(0).to(device)                  # [1,2,N]
    yb = tgt.T.unsqueeze(0).to(device)                  # [1,4,N]

    buf = {}

    def mk(name):
        def hook(_m, _i, out):
            out.retain_grad()
            buf[name] = out
        return hook

    hs = [model.conv_high.register_forward_hook(mk("f_high")),
          model.input_fc.register_forward_hook(mk("f_low"))]
    sem = getattr(model, "global_conv2", None) or getattr(model, "spatial_conv2")
    hs.append(sem.register_forward_hook(mk("context")))

    model.zero_grad(set_to_none=True)
    pred = model(xb)
    loss = torch.nn.functional.mse_loss(pred, yb)
    loss.backward()
    out = {"N": int(n), "loss": float(loss.detach()), "per_point": {}}
    for name in ("f_low", "f_high", "context"):
        t = buf.get(name)
        if t is None or t.grad is None:
            out["per_point"][name] = float("nan")
            continue
        gnorm = float(t.grad.norm())
        # per-point scale: the global-pooling context is an average over N, so
        # the meaningful quantity is ||g|| / sqrt(N)
        out["per_point"][name] = gnorm / float(np.sqrt(n))
        out[f"{name}_total"] = gnorm
    w = model.input_fc.weight.grad
    out["input_fc_weight_grad"] = float(w.norm()) if w is not None else float("nan")
    for h in hs:
        h.remove()
    del buf
    return out


def _train_probe(ctx, train_ids, X, Y, stats, device, epochs, lr, tag):
    """Short controlled training on the thermal train split, logging per-layer
    parameter-gradient norms each epoch (M2b)."""
    use_grid = (ctx == "spatial_grid")
    model = FeatExtractNet(input_channels=2, output_channels=4, scaling=1.0,
                           use_spatial_grid=use_grid)
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    n_batches = 0
    for ep in range(1, epochs + 1):
        ep_norms = {}
        for start in range(0, len(train_ids), 4):
            gids = train_ids[start:start + 4]
            xb = np.stack([(X[g] - stats["x_mean"]) / stats["x_std"]
                           for g in gids])
            yb = np.stack([(Y[g] - stats["y_mean"]) / stats["y_std"]
                           for g in gids])
            xt = as_btc(xb, device)
            yt = as_btc(yb, device)
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(xt), yt)
            loss.backward()
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                key = name.split(".")[0]
                ep_norms[key] = ep_norms.get(key, 0.0) + float(p.grad.norm()) ** 2
            opt.step()
            n_batches += 1
        hist.append({"epoch": ep,
                     "grad_norm": {k: float(np.sqrt(v / max(1, n_batches // epochs)))
                                   for k, v in ep_norms.items()}})
        if ep == 1 or ep % 20 == 0:
            print(f"[{tag}] {ctx} ep{ep} done", flush=True)
    ckpt = os.path.join(OUT_DIR, f"probe_ckpt_{ctx}_seed0.pth")
    torch.save({"model_state_dict": model.state_dict(), "ctx": ctx,
                "epochs": epochs, "tag": tag}, ckpt)
    return model, hist, ckpt


def args_lr():
    return 1e-3


def run_grad(args):
    X, Y = load_thermal_data()
    train_ids, val_ids, test_ids = load_protocol_a()
    os.makedirs(OUT_DIR, exist_ok=True)
    device, dev_type = get_device(args.device)
    set_seed(0)
    stats = load_ckpt_stats(0)

    Ns = [int(v) for v in args.ns.split(",")]
    contexts = ("global_pool", "spatial_grid")
    result = {"Ns": Ns, "device": dev_type,
              "hardware": device_hardware_summary(),
              "synthetic_target": "y = [sin(pi x)cos(pi y), cos(pi x)sin(pi y), "
                                  "sin(2pi x), cos(2pi y)] on uniform random "
                                  "points in [-1,1]^2 (architecture-level probe, "
                                  "not a physical case)",
              "per_point_def": "||dL/dh||_2 / sqrt(N)",
              "contexts": {}}

    for ctx in contexts:
        use_grid = (ctx == "spatial_grid")
        # (a) random init
        set_seed(0)
        m0 = FeatExtractNet(input_channels=2, output_channels=4, scaling=1.0,
                            use_spatial_grid=use_grid).to(device).eval()
        init_rows = []
        for n in Ns:
            init_rows.append(_grad_norms_at_N(m0, n, device))
            print(f"[M2] {ctx} init N={n} done", flush=True)
        # (b) short controlled training (M2b) then sweep again
        model, hist, ckpt = _train_probe(ctx, train_ids, X, Y, stats, device,
                                        args.train_epochs, args.lr, "M2b")
        model.eval()
        trained_rows = []
        for n in Ns:
            trained_rows.append(_grad_norms_at_N(model, n, device))
            print(f"[M2] {ctx} trained N={n} done", flush=True)
        with open(os.path.join(OUT_DIR, f"grad_traj_{ctx}_seed0.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"context": ctx, "epochs": args.train_epochs,
                       "train_geoms": train_ids, "history": hist,
                       "per_layer_def": "sqrt(sum_batches ||dL/dW_layer||^2 / "
                                        "n_batches)",
                       "probe_checkpoint": os.path.relpath(ckpt, WS)},
                      fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        result["contexts"][ctx] = {"use_spatial_grid": use_grid,
                                   "random_init": init_rows,
                                   "after_short_training": trained_rows}

    # decision rule fixed in the design doc before running
    def slope(rows, key):
        n = np.array([r["N"] for r in rows], dtype=float)
        y = np.array([r["per_point"][key] for r in rows], dtype=float)
        m = np.isfinite(y) & (y > 0)
        if m.sum() < 2:
            return float("nan")
        return float(np.polyfit(np.log(n[m]), np.log(y[m]), 1)[0])

    verdict = {}
    for key in ("f_low", "f_high", "context"):
        s_gp = slope(result["contexts"]["global_pool"]["random_init"], key)
        s_sg = slope(result["contexts"]["spatial_grid"]["random_init"], key)
        verdict[key] = {
            "log_log_slope_global_pool": s_gp,
            "log_log_slope_spatial_grid": s_sg,
            "global_pool_falls_off_faster": bool(np.isfinite(s_gp)
                                                 and np.isfinite(s_sg)
                                                 and s_gp < s_sg - 0.05),
        }
    result["verdict"] = verdict
    result["decision_rule"] = ("claim 'global pooling attenuates gradients as N "
                              "grows' counts as supported only if the global-pooling "
                              "per-point gradient norm falls off faster with N than "
                              "the spatial-grid one (log-log slope difference > "
                              "0.05); otherwise the corresponding sentence in "
                              "main.tex is removed (A#10)")

    with open(os.path.join(OUT_DIR, "grad_norm_vs_N.json"), "w",
              encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    _plot_grad(result)
    for key, v in verdict.items():
        print(f"[M2] {key}: slope(global_pool)={v['log_log_slope_global_pool']:+.3f} "
              f"slope(spatial_grid)={v['log_log_slope_spatial_grid']:+.3f} "
              f"-> supported={v['global_pool_falls_off_faster']}", flush=True)
    append_registry({
        "id": "M2_grad_vs_N",
        "experiment_id": "M2_grad_vs_N",
        "type": "mechanism",
        "tool": os.path.basename(__file__),
        "Ns": Ns,
        "verdict": verdict,
        "decision_rule": result["decision_rule"],
        "short_training_epochs": args.train_epochs,
        "splits_version": "splits_thermal_v1_protocolA",
        "files": ["results/mechanism/grad_norm_vs_N.json",
                  "results/mechanism/grad_traj_global_pool_seed0.json",
                  "results/mechanism/grad_traj_spatial_grid_seed0.json"],
    })
    return 0


def _plot_grad(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    keys = [("f_low", "grad wrt f_low (input block)"),
            ("f_high", "grad wrt f_high (deep block)"),
            ("context", "grad wrt context branch output")]
    for ax, (key, lab) in zip(axes, keys):
        for ctx, style in (("global_pool", "o-"), ("spatial_grid", "s--")):
            for which, ls in (("random_init", ":"), ("after_short_training", "-")):
                rows = result["contexts"][ctx][which]
                n = [r["N"] for r in rows]
                y = [r["per_point"][key] for r in rows]
                ax.plot(n, y, style, ls=ls, ms=4,
                        label=f"{ctx} / {which.replace('_', ' ')}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("number of points N")
        ax.set_title(lab, fontsize=10)
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("||dL/dh|| / sqrt(N)")
    axes[0].legend(fontsize=7)
    fig.suptitle("M2: controlled pooling comparison - gradient norm vs N", fontsize=11)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "grad_attenuation_vs_N.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[M2] figure -> {out}", flush=True)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("gate", "grad"), required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--geoms", type=int, default=0,
                    help="limit the number of test geometries (0 = all 20)")
    ap.add_argument("--ns", default="2500,5000,25000,100000,231361")
    ap.add_argument("--train-epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    # corrected gate interface (gate mode only; the default grad behaviour is
    # unchanged and any gate-only option passed with --mode grad is refused)
    ap.add_argument("--checkpoint-root", default=None,
                    help="gate: directory holding seed{k}/checkpoint_best.pth "
                         "(default results/round2_thermal_symmetric/feat_extract)")
    ap.add_argument("--out-dir", default=None,
                    help="gate: independent round2 output directory "
                         "(default results/round2_mechanism)")
    ap.add_argument("--fig-dir", default=None,
                    help="gate: figure directory "
                         "(default results/figures/round2_mechanism)")
    ap.add_argument("--data", default=None,
                    help="gate: HeatTransferData.npy path (default: auto-detect)")
    ap.add_argument("--pool-context", choices=POOL_CONTEXTS, default=None,
                    help="gate: pool context of the checkpoint root (default "
                         "symmetric; legacy_index_bins only with an explicit "
                         "legacy checkpoint root and a fresh round2 out-dir)")
    ap.add_argument("--ridge", type=float, default=RIDGE,
                    help="gate: ridge lambda of the kNN local linear gradient fit")
    args = ap.parse_args()
    if args.mode == "grad":
        gate_only = [name for name in ("checkpoint_root", "out_dir", "fig_dir",
                                       "data", "pool_context")
                     if getattr(args, name) is not None]
        if gate_only:
            raise SystemExit(
                "[refuse] gate-only options passed with --mode grad: "
                + ", ".join(f"--{name.replace('_', '-')}" for name in gate_only)
                + "; the default grad behaviour is unchanged")
    t0 = time.time()
    rc = run_gate(args) if args.mode == "gate" else run_grad(args)
    print(f"[done] mode={args.mode} in {time.time() - t0:.1f} s", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
