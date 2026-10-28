#!/usr/bin/env python3
"""R3.16 route A: original 135 pressure-field forensics (read-only).

Computes per-geometry pressure signatures, public-outer-boundary values,
near-zero thresholds, and a predeclared train/val/test candidate-reference
selection.  This is a data description, NOT proof of the historical solver
gauge; only an exact source/code record can make that claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime

import numpy as np


WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(WS, "results", "round2_pressure_reference_closure")
TAU_RULE = "1e-12*max(1,max_abs_p_per_geometry)"


def _now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def split_ids(config):
    with open(os.path.join(WS, config["splits"]), encoding="utf-8") as fh:
        doc = json.load(fh)
    a = doc["protocol_A"]
    return [int(v) for v in a["train"]], [int(v) for v in a["val"]], [int(v) for v in a["test"]]


def stats(v):
    v = np.asarray(v, dtype=np.float64)
    return {
        "n": int(v.size),
        "min": float(v.min()), "max": float(v.max()),
        "mean": float(v.mean()), "std": float(v.std(ddof=1)),
        "median": float(np.median(v)),
        "mean_abs": float(np.mean(np.abs(v))),
        "max_abs": float(np.max(np.abs(v))),
        "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)),
    }


def near_zero(v, threshold):
    a = np.abs(np.asarray(v, dtype=np.float64))
    return {"threshold": float(threshold), "count": int((a <= threshold).sum()),
            "fraction": float((a <= threshold).mean()),
            "median_abs": float(np.median(a)),
            "max_abs": float(a.max())}


def candidate_metrics(p_at, ids, threshold_by_geom):
    x = p_at[ids, :]
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=1)
    maxabs = np.abs(x).max(axis=0)
    near = (np.abs(x) <= threshold_by_geom[ids, None]).mean(axis=0)
    score = np.abs(mean) + std
    return {"mean": mean, "std": std, "max_abs": maxabs,
            "near_zero_fraction": near, "score": score}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-workspace", default=WS)
    ap.add_argument("--config", default=os.path.join(WS, "configs", "r2_pressure_reference_closure.json"))
    ap.add_argument("--data", default=None)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    args = ap.parse_args()
    inp = os.path.abspath(args.input_workspace)
    out = os.path.abspath(args.out_dir)
    os.makedirs(out, exist_ok=True)
    cfg = load_config(args.config)
    data_path = os.path.abspath(args.data or cfg["data"])
    for name in ("protocol_snapshot.json", "all135_pressure_signatures.json"):
        if os.path.exists(os.path.join(out, name)):
            raise SystemExit(f"[refuse] target exists: {out}/{name}")

    A = np.load(data_path)
    if A.ndim != 3 or A.shape[2] != 6:
        raise SystemExit(f"unexpected data shape {A.shape}")
    n_geom = A.shape[0]
    X = A[:, :, :2].astype(np.float64)
    P = A[:, :, 4].astype(np.float64)
    train, val, test = split_ids(cfg)
    if sorted(train + val + test) != list(range(n_geom)):
        raise SystemExit("split is not a partition of all geometries")

    outer = (np.isclose(np.abs(X[:, :, 0]), 1.0, atol=1e-12)
             | np.isclose(np.abs(X[:, :, 1]), 1.0, atol=1e-12))
    outer_counts = outer.sum(axis=1)
    if not np.all(outer_counts == cfg["public_outer_boundary_count"]):
        raise SystemExit(f"outer boundary count mismatch: {sorted(set(outer_counts.tolist()))}")

    # Common outer coordinates present in every geometry (coordinate, not index).
    coord_sets = []
    for i in range(n_geom):
        coord_sets.append(set(zip(np.round(X[i, outer[i], 0], 10),
                                  np.round(X[i, outer[i], 1], 10))))
    common = sorted(set.intersection(*coord_sets))
    if len(common) != cfg["public_outer_boundary_count"]:
        raise SystemExit(f"common outer coordinate count {len(common)} != 492")
    common_arr = np.asarray(common, dtype=np.float64)
    p_at = np.empty((n_geom, len(common)), dtype=np.float64)
    for i in range(n_geom):
        idx = {tuple(k): j for j, k in enumerate(zip(np.round(X[i, :, 0], 10),
                                                      np.round(X[i, :, 1], 10)))}
        for j, key in enumerate(common):
            p_at[i, j] = P[i, idx[key]]

    # Per-geometry signatures; all thresholds predeclared.
    tau = 1e-12 * np.maximum(1.0, np.abs(P).max(axis=1))
    signatures = []
    for i in range(n_geom):
        s = stats(P[i])
        s.update({
            "geom_id": i,
            "stage": ("train" if i in train else "val" if i in val else "test"),
            "tau": float(tau[i]),
            "tau_rule": TAU_RULE,
            "near_zero_outer": {
                "tau": near_zero(P[i, outer[i]], tau[i]),
                "1e-10": near_zero(P[i, outer[i]], 1e-10),
                "1e-8": near_zero(P[i, outer[i]], 1e-8),
            },
            "outer_boundary": stats(P[i, outer[i]]),
        })
        signatures.append(s)

    tr = np.asarray(train, dtype=int)
    va = np.asarray(val, dtype=int)
    te = np.asarray(test, dtype=int)
    mtr = candidate_metrics(p_at, tr, tau)
    mva = candidate_metrics(p_at, va, tau)
    mte = candidate_metrics(p_at, te, tau)
    best = int(np.argmin(mtr["score"]))
    selected = {
        "coord": [float(common_arr[best, 0]), float(common_arr[best, 1])],
        "selection_rule": cfg["candidate_selection"],
        "train": {"mean": float(mtr["mean"][best]), "std_sample": float(mtr["std"][best]),
                  "max_abs": float(mtr["max_abs"][best]),
                  "near_zero_fraction": float(mtr["near_zero_fraction"][best]),
                  "score": float(mtr["score"][best])},
        "val_after_freeze": {"mean": float(mva["mean"][best]), "std_sample": float(mva["std"][best]),
                             "max_abs": float(mva["max_abs"][best]),
                             "near_zero_fraction": float(mva["near_zero_fraction"][best])},
        "test_after_freeze": {"mean": float(mte["mean"][best]), "std_sample": float(mte["std"][best]),
                              "max_abs": float(mte["max_abs"][best]),
                              "near_zero_fraction": float(mte["near_zero_fraction"][best])},
        "descriptive_all135": {
            "mean": float(p_at[:, best].mean()),
            "std_sample": float(p_at[:, best].std(ddof=1)),
            "max_abs": float(np.abs(p_at[:, best]).max()),
        },
    }
    order = np.argsort(mtr["score"])[:10]
    top = [{"rank": int(r + 1),
            "coord": [float(common_arr[j, 0]), float(common_arr[j, 1])],
            "train_mean": float(mtr["mean"][j]), "train_std": float(mtr["std"][j]),
            "train_max_abs": float(mtr["max_abs"][j]),
            "train_near_zero_fraction": float(mtr["near_zero_fraction"][j]),
            "val_mean": float(mva["mean"][j]), "val_max_abs": float(mva["max_abs"][j]),
            "test_mean": float(mte["mean"][j]), "test_max_abs": float(mte["max_abs"][j])}
           for r, j in enumerate(order)]
    out_doc = {
        "task": "R3.16 route A original pressure forensics",
        "protocol": cfg["protocol"], "version": cfg["version"],
        "created_local": _now(), "data": data_path, "data_sha256": sha256_file(data_path),
        "shape": [int(v) for v in A.shape], "dtype": str(A.dtype),
        "columns": ["x", "y", "u", "v", "p", "T"],
        "split": {"train": train, "val": val, "test": test,
                  "n_train": len(train), "n_val": len(val), "n_test": len(test)},
        "public_outer_boundary_count": int(outer_counts[0]),
        "common_outer_coordinates": len(common),
        "tau_rule": TAU_RULE,
        "sensitivity_thresholds": cfg["sensitivity_thresholds"],
        "selected_reference": selected,
        "top_train_candidates": top,
        "per_geometry": signatures,
        "all135_mean": float(P.mean()), "all135_std_sample": float(P.std(ddof=1)),
        "all135_mean_range": [float(P.mean(axis=1).min()), float(P.mean(axis=1).max())],
        "all135_mean_abs_outer": float(np.mean(np.abs(p_at))),
        "interpretation": {
            "data_description_only": True,
            "historical_reference_identified": False,
            "reason": ("node arithmetic means are not zero; a coordinate candidate is only "
                       "a data-compatible descriptive anchor, not a recovered historical gauge"),
            "mean_rule": "arithmetic equal node weight, not FEM area integral",
        },
    }
    protocol = {
        "task": "R3.16 pressure reference closure",
        "protocol": cfg["protocol"], "version": cfg["version"],
        "created_local": _now(), "config": os.path.abspath(args.config),
        "config_sha256": sha256_file(args.config),
        "data": data_path, "data_sha256": sha256_file(data_path),
        "splits": os.path.join(inp, cfg["splits"]),
        "splits_sha256": sha256_file(os.path.join(inp, cfg["splits"])),
        "source_sha256": {rel: sha256_file(os.path.join(WS, rel))
                          for rel in ("scripts/exp_r2_pressure_reference_audit.py",
                                      "configs/r2_pressure_reference_closure.json")},
        "route_A": {
            "task": "original pressure forensics + source trace",
            "tau_rule": TAU_RULE,
            "sensitivity_thresholds": cfg["sensitivity_thresholds"],
            "candidate_selection": cfg["candidate_selection"],
            "public_outer_boundary_count": cfg["public_outer_boundary_count"],
            "train_val_test_only": True,
            "test_after_freeze_only": True,
            "historical_reference_not_claimed": True,
        },
        "route_B": cfg["gauge"],
        "outputs_planned": ["all135_pressure_signatures.json", "source_evidence_index.json",
                            "exact_gauge_comparison.json", "model_metric_sensitivity.json",
                            "job_ledger.json", "self_acceptance.json", "FINAL_DELIVERY.md"],
        "training": False, "registry_modified": False,
    }
    with open(os.path.join(out, "protocol_snapshot.json"), "w", encoding="utf-8") as fh:
        json.dump(protocol, fh, indent=2, ensure_ascii=False)
    with open(os.path.join(out, "all135_pressure_signatures.json"), "w", encoding="utf-8") as fh:
        json.dump(out_doc, fh, indent=2, ensure_ascii=False)
    print(json.dumps({"selected": selected, "top": top[:3],
                      "all135_mean_range": out_doc["all135_mean_range"]}, indent=2))
    print(f"[done] {out}/all135_pressure_signatures.json", flush=True)


if __name__ == "__main__":
    main()
