"""R2-THERMAL-FEM-02 frozen-model evaluation on fixed FEM-derived support.

Protocol (frozen before model errors are inspected)
---------------------------------------------------
* Geometries: seven 46, eight 17, nine 96 (Protocol-A test minima).
* Reference field: independent FEM solution on the h=0.04 mesh, transferred to
  a fixed 5000-point support (4340 fluid Sobol + 492 outer-wall + 168 inner-wall
  points) by piecewise-linear/Clough-Tocher interpolation of the FEM DOFs.
  This is an FEM reference transfer, not interpolation of the original labels.
* Models: accepted FEN (FeatExtractNet symmetric) and MeshGraphNet-style graph
  checkpoints, seeds 0..4, frozen; no training or model selection.
* Support coordinates are generated from geometry only (Sobol seed
  20261022+geom), never from model errors.  Protocol/support hashes are written
  before any model prediction.
* Pressure is reported raw and point-mean-centered; denominators are recorded;
  zero denominators are null, never epsilon-padded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WS not in sys.path:
    sys.path.insert(0, WS)

from scripts import exp_r2_local_error as LOC  # noqa: E402
from scripts import exp_r2_new_queries as NQ  # noqa: E402

GEOMS = (46, 17, 96)
SEEDS = (0, 1, 2, 3, 4)
FIELDS = ("u", "v", "p", "T")
N_POINTS = 5000
N_BOUNDARY = 660
FEN_BASE = "results/round2_thermal_symmetric/feat_extract"
GRAPH_BASE = "results/round2_graph_five_seed_main/meshgraphnet"


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path, doc):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def _metrics(pred, ref):
    out = {}
    for j, name in enumerate(FIELDS):
        truth = ref[:, j]
        if name == "p":
            centered_truth = truth - truth.mean()
            centered_pred = pred[:, j] - pred[:, j].mean()
            out["p_raw"] = _block(truth, pred[:, j])
            out["p_centered"] = _block(centered_truth, centered_pred)
        else:
            out[name] = _block(truth, pred[:, j])
    return out


def _block(truth, prediction):
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    denom = float(np.linalg.norm(truth))
    diff = prediction - truth
    return {"n": int(len(truth)), "re": (float(np.linalg.norm(diff) / denom)
                                          if denom > 0 else None),
            "mae": float(np.mean(np.abs(diff))),
            "max_err": float(np.max(np.abs(diff))),
            "denominator": denom}


def support_for_geometry(data, spec_doc, geom):
    """Fixed support built from geometry; same as new-query distribution."""
    per_geo = {int(r["geom"]): r for r in spec_doc["per_geometry"]}
    coords, wall_kind, info = NQ.build_distribution(
        data[int(geom), :, :2].astype(np.float64),
        per_geo[int(geom)], int(geom), "sobol")
    return coords, wall_kind, info


def _interp(points, coords, values):
    from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator
    from scipy.spatial import cKDTree
    q = np.asarray(points, dtype=np.float64)[:, :2]
    c = np.asarray(coords, dtype=np.float64)[:, :2]
    v = np.asarray(values, dtype=np.float64)
    try:
        f = CloughTocher2DInterpolator(c, v)
        out = f(q)
    except Exception:  # noqa: BLE001
        f = LinearNDInterpolator(c, v)
        out = f(q)
    out = np.asarray(out, dtype=np.float64)
    bad = ~np.isfinite(out)
    if np.any(bad):
        tree = cKDTree(c)
        _, idx = tree.query(q[bad])
        out[bad] = v[idx]
    return out


def load_fem_reference(out_dir, geom, h):
    """Load actual dolfinx-evaluated FEM reference on the frozen support."""
    path = os.path.join(WS, "results/round2_thermal_fem/v2_reference",
                        f"fem_reference_geom{int(geom)}_h{h:g}.npz")
    if not os.path.isfile(path):
        raise RuntimeError(f"missing actual FEM support reference: {path}")
    d = np.load(path)
    coords = d["coords"]
    ref = np.column_stack([d["u"], d["v"], d["p"], d["T"]])
    valid = d["valid"].astype(bool)
    if not np.all(valid):
        raise RuntimeError(f"FEM support has {int((~valid).sum())} outside points")
    if not np.all(np.isfinite(ref)):
        raise RuntimeError("non-finite FEM support reference")
    return coords, ref, path


def load_fen(seed, metrics_doc, device):
    from src.fem_reference.featextractnet_symmetric import FeatExtractNet
    model = FeatExtractNet(input_channels=2, output_channels=4, scaling=1.0,
                           pool_mode="symmetric")
    ckpt_path = os.path.join(WS, FEN_BASE, f"seed{seed}", "checkpoint_best.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    if int(ckpt.get("seed", -1)) != seed:
        raise RuntimeError(f"FEN seed mismatch {seed}")
    model.to(device).eval()
    return model, str(metrics_doc["protocol"].get("model_label")), ckpt_path


def load_graph(seed, metrics_doc, device):
    from src.models.baselines.meshgraphnet_style import MeshGraphNetStyleBaseline
    model = MeshGraphNetStyleBaseline(input_channels=2, output_channels=4)
    label = str(metrics_doc["protocol"].get("model_label"))
    ckpt_path = os.path.join(WS, GRAPH_BASE, f"seed{seed}", "checkpoint_best.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    best = metrics_doc["val_best"]
    if int(ckpt.get("epoch", -2)) != int(best["epoch"]):
        raise RuntimeError(f"graph checkpoint epoch mismatch seed{seed}")
    model.to(device).eval()
    return model, label, ckpt_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=os.path.join(
        WS, "results", "round2_thermal_fem", "model_eval"))
    ap.add_argument("--h", type=float, default=0.04)
    ap.add_argument("--geoms", type=int, nargs="*", default=list(GEOMS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    out = os.path.abspath(args.out_dir)
    if os.path.exists(out) and os.listdir(out) and not args.force:
        raise SystemExit(f"[fail] output exists (use --force): {out}")
    os.makedirs(out, exist_ok=True)
    data = np.load(os.path.join(os.path.dirname(os.path.dirname(WS)), "HeatTransferData.npy"))
    split = json.load(open(os.path.join(WS, "configs/splits_thermal.json"),
                           encoding="utf-8"))
    spec = json.load(open(os.path.join(WS, "results/thermal_geometry_spec.json"),
                          encoding="utf-8"))
    per_geo = {int(r["geom"]): r for r in spec["per_geometry"]}
    device = torch.device("cpu")
    torch.set_num_threads(4)

    protocol = {
        "task": "R2-THERMAL-FEM-02-model-eval",
        "created_local": NQ._now_local(),
        "reference": ("independent FEM h=%.2f, transferred to fixed support by "
                      "CloughTocher/linear interpolation of FEM DOFs; not label interpolation" % args.h),
        "support": {"n": N_POINTS, "boundary": N_BOUNDARY,
                    "sobol_seed": NQ.NEW_QUERY_SEED,
                    "generator": "exp_r2_new_queries.build_distribution"},
        "geometries": [int(g) for g in args.geoms],
        "seeds": list(SEEDS),
        "models": {"FEN": FEN_BASE, "graph": GRAPH_BASE},
        "metrics": ["re", "mae", "max_err", "p_raw", "p_centered"],
        "source_hashes": {},
        "input_hashes": {},
    }
    for rel in ("src/fem_reference/fenics_solver.py", "src/fem_reference/geometry.py",
                "scripts/exp_r2_fem_model_eval.py", "configs/splits_thermal.json",
                "results/thermal_geometry_spec.json"):
        protocol["source_hashes"][rel] = _sha(os.path.join(WS, rel))
    protocol["input_hashes"]["data"] = _sha(os.path.join(
        os.path.dirname(os.path.dirname(WS)), "HeatTransferData.npy"))

    supports = {}
    for geom in args.geoms:
        coords, wall_kind, info = support_for_geometry(
            data, spec, int(geom))
        path = os.path.join(out, f"support_geom{geom}.npz")
        np.savez_compressed(path, coords=coords, wall_kind=wall_kind)
        ref_coords, ref, fem_path = load_fem_reference(out, geom, args.h)
        if not np.array_equal(coords, ref_coords):
            raise RuntimeError(f"support coordinates do not match FEM support geom{geom}")
        np.savez_compressed(os.path.join(out, f"fem_reference_geom{geom}.npz"),
                            coords=coords, ref=ref, wall_kind=wall_kind)
        protocol["support"][f"geom{geom}"] = {
            "coords_sha256": _sha(path), "fem_reference": fem_path,
            "fem_reference_sha256": _sha(fem_path), "info": info,
            "evaluation": "actual dolfinx mesh/Function.eval; outside points rejected"}
        supports[int(geom)] = (coords, ref)
    # Same-coordinate three-h FEM comparison (before model errors).
    discretization = {}
    for geom in args.geoms:
        refs = {}
        for h in (0.08, 0.04, 0.02):
            try:
                _, ref, _ = load_fem_reference(out, geom, h)
                refs[h] = ref
            except Exception:
                refs[h] = None
        if refs.get(0.04) is not None:
            block = {}
            for h in (0.08, 0.02):
                if refs.get(h) is None:
                    block[str(h)] = None
                    continue
                metrics = {}
                for j, name in enumerate(FIELDS):
                    truth = refs[0.04][:, j]
                    pred = refs[h][:, j]
                    if name == "p":
                        truth = truth - truth.mean()
                        pred = pred - pred.mean()
                    metrics[name] = _block(truth, pred)
                block[str(h)] = metrics
            discretization[str(geom)] = block
    protocol["same_coordinate_h_comparison_vs_h004"] = discretization
    _json(os.path.join(out, "protocol_snapshot.json"), protocol)
    print("[protocol] frozen before model predictions", flush=True)

    rows = []
    stats = {}
    for model_name, loader, base, pool in (
            ("FEN", load_fen, FEN_BASE, "symmetric"),
            ("graph", load_graph, GRAPH_BASE, "legacy_index_bins")):
        for seed in SEEDS:
            metrics_path = os.path.join(WS, base, f"seed{seed}",
                                        f"metrics_seed{seed}.json")
            metrics_doc = json.load(open(metrics_path, encoding="utf-8"))
            if metrics_doc["protocol"]["pool_context"] != pool:
                raise RuntimeError(f"{model_name} seed{seed} pool mismatch")
            s = {k: np.asarray(metrics_doc["protocol"][k], np.float32)
                 for k in ("x_mean", "x_std", "y_mean", "y_std")}
            model, label, ckpt_path = loader(seed, metrics_doc, device)
            for geom in args.geoms:
                coords, ref = supports[int(geom)]
                pred = LOC.predict_full(model, coords.astype(np.float32), s, device)
                block = _metrics(pred, ref)
                rows.append({"model": model_name, "seed": seed,
                             "geom": int(geom), "metrics": block})
                npz_path = os.path.join(out, f"pred_{model_name}_seed{seed}_geom{geom}.npz")
                np.savez_compressed(npz_path, prediction=pred.astype(np.float32),
                                    coords=coords.astype(np.float32))
            stats.setdefault(model_name, {})[seed] = {
                "checkpoint": ckpt_path, "checkpoint_sha256": _sha(ckpt_path),
                "metrics_sha256": _sha(metrics_path), "label": label}
            del model
    summary = {}
    for model_name in ("FEN", "graph"):
        summary[model_name] = {}
        for geom in args.geoms:
            summary[model_name][str(geom)] = {}
            for key in ("u", "v", "p_raw", "p_centered", "T"):
                vals = [r["metrics"][key]["re"] for r in rows
                        if r["model"] == model_name and r["geom"] == geom
                        and r["metrics"][key]["re"] is not None]
                summary[model_name][str(geom)][key] = {
                    "mean_re": float(np.mean(vals)) if vals else None,
                    "std_re": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                    "n": len(vals)}
    _json(os.path.join(out, "raw_metrics.json"), {"rows": rows,
                                                  "model_stats": stats})
    _json(os.path.join(out, "summary.json"), summary)
    lines = ["# R2-THERMAL-FEM-02 model evaluation on FEM support", "",
             "Reference: independent FEM h=%.2f, transferred to fixed 5000-point support; not label interpolation." % args.h,
             "", "## Summary (5 seeds)", ""]
    for model_name in ("FEN", "graph"):
        lines.append(f"### {model_name}")
        for geom in args.geoms:
            block = summary[model_name][str(geom)]
            lines.append(f"- geom {geom}: " + ", ".join(
                f"{k} RE={v['mean_re']:.4f}±{v['std_re']:.4f}" if v['mean_re'] is not None
                else f"{k} RE=undefined" for k, v in block.items()))
        lines.append("")
    lines += ["## Limitations", "",
              "- FEM transfer interpolation is not the historical solver; it does not validate historical labels.",
              "- Only three representative geometries have independent FEM support.",
              "- No retraining/model selection; pressure raw and centered metrics use different denominators."]
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(json.dumps({"status": "model_eval_done", "out": out,
                      "rows": len(rows), "geoms": args.geoms}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
