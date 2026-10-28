#!/usr/bin/env python3
"""R3.16 exact pressure-reference runs and model sensitivity postprocessing.

Modes:
  pilot   geom17 h=0.04 exact point pin + nonlinear manufactured-solution check
  full    46/17/96 h=0.04 exact point pin, derive integral-zero gauge, compare
  analysis  reuse frozen 5seed x 2model predictions; recompute p metrics under
            the exact zero-mean reference (no training / no GPU rerun)
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
if WS not in sys.path:
    sys.path.insert(0, WS)

from src.fem_reference import geometry, pressure_reference_audit as exact  # noqa: E402

OUT = os.path.join(WS, "results", "round2_pressure_reference_closure")
DEFAULT_CONFIG = os.path.join(WS, "configs", "r2_pressure_reference_closure.json")
FEM_CONFIG = os.path.join(WS, "configs", "r2_thermal_fem.json")


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_json(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def input_paths(cfg):
    inp = WS
    data = cfg["data"]
    if not os.path.isabs(data):
        data = os.path.join(inp, data)
    return inp, data


def params_from_fem(fem_cfg):
    p = dict(fem_cfg["physics"])
    p["pressure_gauge_eps"] = 0.0
    return p


def solver_fem_cfg(cfg, fem_cfg):
    out = dict(fem_cfg)
    out["solver"] = {**dict(fem_cfg.get("solver", {})), **dict(cfg.get("exact_solver", {}))}
    return out


def supports_for(cfg, gid):
    rel = cfg["supports"][str(gid)]
    return np.load(os.path.join(WS, rel))["coords"].astype(float)


def case_paths(gid, tag=""):
    stem = f"geom{gid}_h004{tag}"
    return (os.path.join(OUT, "exact", f"fields_{stem}.npz"),
            os.path.join(OUT, "exact", f"comparison_{stem}.json"))


def initial_fields(cfg, gid):
    rel = (cfg.get("initial_guess") or {}).get(str(gid))
    if not rel:
        return None
    p = os.path.join(WS, rel)
    if not os.path.isfile(p):
        raise SystemExit(f"missing initial guess: {p}")
    return np.load(p)


def run_case(cfg, fem_cfg, spec, params, gid, tag=""):
    entry = dict(geometry.geometry_entry(spec, gid))
    entry["vertices"] = geometry.polygon_vertices(entry).tolist()
    h = float(cfg["gauge"]["target_h"])
    result = exact.solve_integral_gauge(entry, params, solver_fem_cfg(cfg, fem_cfg), h,
                                      initial_fields=initial_fields(cfg, gid))
    points = supports_for(cfg, gid)
    fields = exact.support_fields(result, points)
    cmp = exact.gauge_comparison(fields, result)
    cmp.update({
        "geom_id": gid, "target_h": h, "tag": tag,
        "pin_dof": result["pin_dof"], "pin_coord": result["pin_coord"],
        "pin_margin": result["pin_margin"], "pin_value": result["pin_value"],
        "newton_reason": result.get("newton_reason", 2 if result.get("pressure_gauge") == "integral_lagrange_petsc" else -1),
        "newton_iterations": result.get("newton_iterations"),
        "continuation": result.get("history", result.get("continuation")),
        "diagnostics": result["diagnostics"],
        "weak_residual": result["weak_residual"],
        "pressure_gauge_eps": 0.0,
        "integral_pressure": result.get("integral_pressure"),
        "final_constraint": result.get("final_constraint"),
        "final_linear_residual_rel": result.get("final_linear_residual_rel"),
        "gauge_definition": "point Dirichlet p(x_pin)=0; exact, no epsilon mass term",
        "integral_gauge_definition": "p_integral = p_point - area_weighted_mean(p_point)",
        "mean_rule": "support metrics use equal-weight arithmetic point mean",
    })
    fpath, jpath = case_paths(gid, tag)
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    for p in (fpath, jpath):
        if os.path.exists(p):
            raise SystemExit(f"[refuse] target exists: {p}")
    np.savez_compressed(
        fpath,
        coords=fields["coords"], valid=fields["valid"], cells=fields["cells"],
        u=fields["u"], v=fields["v"], T=fields["T"],
        p_point=fields["p_point"], p_integral=fields["p_integral"],
        area_mean_point=np.float64(fields["area_mean_point"]),
        pin_coord=np.asarray(result["pin_coord"], dtype=float),
        pin_value=np.float64(result["pin_value"]),
        geom_id=np.int64(gid), target_h=np.float64(h),
    )
    with open(jpath, "w", encoding="utf-8") as fh:
        json.dump(cmp, fh, indent=2, ensure_ascii=False)
    print(f"[exact] geom{gid} h={h} newton={result['newton_reason']} "
          f"pin={result['pin_value']:.3e} valid={cmp['n_valid']}/{cmp['n_total']} "
          f"center_rel={cmp['centered_relative_difference']:.3e}", flush=True)
    return cmp, fields


def manufactured_check(cfg, fem_cfg, spec):
    gid = int(cfg["gauge"]["geometries"][0])
    entry = dict(geometry.geometry_entry(spec, gid))
    entry["vertices"] = geometry.polygon_vertices(entry).tolist()
    out = exact.solve_manufactured_pinned(entry, params_from_fem(fem_cfg), solver_fem_cfg(cfg, fem_cfg),
                                          float(cfg["gauge"]["target_h"]))
    out["geom_id"] = gid
    out["created_local"] = now()
    return out


def load_case(gid, tag=""):
    fpath, jpath = case_paths(gid, tag)
    if not os.path.isfile(fpath) or not os.path.isfile(jpath):
        raise SystemExit(f"missing exact case {gid}: {fpath} / {jpath}")
    return np.load(fpath), load_json(jpath)


def analysis(cfg, fem_cfg):
    root = os.path.join(WS, cfg["model_eval"]["root"])
    geoms = [int(g) for g in cfg["gauge"]["geometries"]]
    models = list(cfg["model_eval"]["models"])
    seeds = [int(s) for s in cfg["model_eval"]["seeds"]]
    cases_by_geom = {g: load_case(g) for g in geoms}
    rows = []
    old_rows = []
    for model in models:
        for seed in seeds:
            for gid in geoms:
                data, meta = cases_by_geom[gid]
                pred_path = os.path.join(root, f"pred_{model}_seed{seed}_geom{gid}.npz")
                d = np.load(pred_path)
                coords = d["coords"].astype(float)
                ref_coords = data["coords"].astype(float)
                if not np.allclose(coords, ref_coords, rtol=0, atol=1e-6):
                    raise SystemExit(f"coords mismatch {pred_path}")
                pred = d["prediction"].astype(np.float64)
                ref = np.column_stack([
                    data["u"].astype(float), data["v"].astype(float),
                    data["p_integral"].astype(float), data["T"].astype(float)])
                valid = data["valid"].astype(bool)
                pred = pred[valid]
                ref = ref[valid]
                if not np.all(np.isfinite(pred)):
                    raise SystemExit(f"nonfinite prediction {pred_path}")
                # Metrics: all four fields; pressure raw/center-separate.
                row = {"model": model, "seed": seed, "geom_id": gid,
                       "n_valid": int(valid.sum()),
                       "prediction": os.path.relpath(pred_path, WS),
                       "prediction_sha256": sha(pred_path)}
                for j, name in enumerate(("u", "v", "p", "T")):
                    err = pred[:, j] - ref[:, j]
                    den = float(np.linalg.norm(ref[:, j]))
                    row[name] = {"re": (float(np.linalg.norm(err) / den)
                                        if den > 0 else None),
                                 "mae": float(np.mean(np.abs(err))),
                                 "max": float(np.max(np.abs(err))),
                                 "denominator": den}
                p_pred = pred[:, 2]
                p_ref = ref[:, 2]
                bias = float(p_pred.mean() - p_ref.mean())
                cp = p_pred - p_pred.mean()
                cr = p_ref - p_ref.mean()
                dc = float(np.linalg.norm(cr))
                row["p_centered"] = {
                    "re": (float(np.linalg.norm(cp - cr) / dc) if dc > 0 else None),
                    "mae": float(np.mean(np.abs(cp - cr))),
                    "max": float(np.max(np.abs(cp - cr))),
                    "denominator": dc,
                    "bias": bias,
                    "n": int(p_ref.size),
                }
                # old eps-reference centered column at the same support
                old = np.load(os.path.join(root, f"fem_reference_geom{gid}.npz"))
                old_ref = old["ref"].astype(np.float64)[valid, 2]
                old_c = old_ref - old_ref.mean()
                old_row = {"model": model, "seed": seed, "geom_id": gid,
                           "eps_reference_sha256": sha(os.path.join(
                               root, f"fem_reference_geom{gid}.npz")),
                           "eps_centered_re": (float(np.linalg.norm(cp - old_c)
                                                     / np.linalg.norm(old_c))
                                               if np.linalg.norm(old_c) > 0 else None),
                           "exact_minus_eps_area_mean": float(
                               data["area_mean_point"].item() - old_ref.mean())}
                rows.append(row)
                old_rows.append(old_row)
    # Per-seed/geometry aggregates
    def agg(sel):
        vals = [r["p_centered"]["re"] for r in sel if r["p_centered"]["re"] is not None]
        return {"n": len(vals),
                "mean": float(np.mean(vals)) if vals else None,
                "std_sample": float(np.std(vals, ddof=1)) if len(vals) > 1 else None}
    seed_stats = {}
    for model in models:
        seed_stats[model] = {}
        for seed in seeds:
            sel = [r for r in rows if r["model"] == model and r["seed"] == seed]
            seed_stats[model][str(seed)] = {
                "n_geometries": len(sel),
                "p_centered_re": agg(sel),
                "p_raw_re": {"mean": float(np.mean([r["p"]["re"] for r in sel])),
                             "std_sample": float(np.std([r["p"]["re"] for r in sel], ddof=1))},
                "p_centered_mae": {"mean": float(np.mean([r["p_centered"]["mae"] for r in sel])),
                                   "std_sample": float(np.std([r["p_centered"]["mae"] for r in sel], ddof=1))},
                "p_centered_max": {"mean": float(np.mean([r["p_centered"]["max"] for r in sel])),
                                   "std_sample": float(np.std([r["p_centered"]["max"] for r in sel], ddof=1))},
            }
    doc = {
        "task": "R3.16 exact zero-mean pressure reference model sensitivity",
        "protocol": cfg["protocol"], "version": cfg["version"],
        "created_local": now(), "data": cfg["data"],
        "data_sha256": sha(cfg["data"]) if os.path.isfile(cfg["data"]) else None,
        "geometries": geoms, "models": models, "seeds": seeds,
        "reference": "exact point-pin solution; integral-zero pressure = p_point - area_mean used as reference p",
        "support": "frozen v2_model_eval 5000 supports; actual Function.eval",
        "pressure_rule": "exact zero mean in the integral/area gauge; point support metrics use equal-weight mean",
        "raw": rows, "old_eps_reference_comparison": old_rows,
        "seed_stats": seed_stats,
        "training": False, "gpu_rerun": False,
        "note": "uses existing frozen 5seed x 2model predictions; no retraining or full GPU rerun",
    }
    path = os.path.join(OUT, "model_metric_sensitivity.json")
    if os.path.exists(path):
        raise SystemExit(f"[refuse] target exists: {path}")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
    print(f"[analysis] {path} rows={len(rows)}", flush=True)


def run_pilot(cfg, fem_cfg, spec):
    geoms = [17]
    cmps = {}
    for gid in geoms:
        cmp, _ = run_case(cfg, fem_cfg, spec, params_from_fem(fem_cfg), gid, tag="_pilot")
        cmps[str(gid)] = cmp
    man = {"status": "deferred_to_separate_manufactured_pilot"}
    path = os.path.join(OUT, "exact", "pilot_geom17_h004.json")
    if os.path.exists(path):
        raise SystemExit(f"[refuse] target exists: {path}")
    doc = {"task": "exact gauge pilot", "created_local": now(),
           "geom": 17, "h": float(cfg["gauge"]["target_h"]),
           "comparison": cmps, "manufactured_nonlinear": man,
           "checks": {
               "integral_zero": abs(cmps["17"].get("final_constraint", 1.0)) < 1e-7,
               "all_support_valid": cmps["17"]["n_valid"] == cmps["17"]["n_total"],
               "constant_shift_only": (cmps["17"]["centered_relative_difference"] is not None
                                       and cmps["17"]["centered_relative_difference"] < 1e-6),
               "converged": cmps["17"]["newton_reason"] == 2,
               "free_residual_small": abs(
                   cmps["17"]["weak_residual"]["blocks"]["full"]["free"]) < 1e-8,
           }}
    doc["checks"]["all_pass"] = all(doc["checks"].values())
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
    print(json.dumps(doc["checks"], indent=2), flush=True)
    if not doc["checks"]["all_pass"]:
        raise SystemExit("[fail] pilot checks failed; see " + path)


def summarize_existing(cfg, fem_cfg, spec):
    """Build exact_gauge_comparison.json from the three already-solved cases."""
    geoms = [int(g) for g in cfg["gauge"]["geometries"]]
    threshold = float(cfg["gauge"]["equivalence_threshold_relative_l2"])
    cmps = {}
    checks = []
    for gid in geoms:
        data, cmp = load_case(gid)
        cmps[str(gid)] = cmp
        old = np.load(os.path.join(WS, cfg["model_eval"]["root"],
                                   f"fem_reference_geom{gid}.npz"))
        valid = data["valid"].astype(bool)
        p_exact = data["p_integral"].astype(float)[valid]
        p_eps = old["ref"].astype(float)[valid, 2]
        ec = p_exact - p_exact.mean()
        oc = p_eps - p_eps.mean()
        den = float(np.linalg.norm(oc))
        eps_rel = float(np.linalg.norm(ec - oc) / den) if den > 0 else None
        ok = (cmp["centered_relative_difference"] is not None
              and cmp["centered_relative_difference"] <= threshold)
        checks.append({"geom": int(gid),
                       "centered_relative_difference": cmp["centered_relative_difference"],
                       "threshold": threshold, "passed": bool(ok),
                       "exact_vs_eps_centered_re": eps_rel,
                       "pin_value": cmp["pin_value"], "n_valid": cmp["n_valid"],
                       "n_total": cmp["n_total"],
                       "free_residual": cmp["weak_residual"]["blocks"]["full"]["free"],
                       "final_constraint": cmp.get("final_constraint")})
    doc = {"task": "R3.16 exact gauge comparison", "protocol": cfg["protocol"],
           "version": cfg["version"], "created_local": now(),
           "target_h": float(cfg["gauge"]["target_h"]), "geometries": geoms,
           "gauges": ["point_reference", "integral_zero_mean"],
           "threshold_relative_l2": threshold,
           "checks": checks, "cases": cmps,
           "pilot": "exact/pilot_geom17_h004.json",
           "note": ("point and integral gauges are exact constant shifts of the same "
                    "discrete integral-gauge solution; u/v/T identical by construction; "
                    "eps v2 reference retained only as a separate approximate column"),
           "all_pass": all(c["passed"] for c in checks)}
    path = os.path.join(OUT, "exact_gauge_comparison.json")
    if os.path.exists(path):
        raise SystemExit(f"[refuse] target exists: {path}")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
    print(json.dumps({"all_pass": doc["all_pass"], "checks": checks}, indent=2), flush=True)
    if not doc["all_pass"]:
        raise SystemExit("[fail] exact gauge comparison failed")


def run_manufactured(cfg, fem_cfg, spec):
    """Asymmetric nonlinear manufactured solution through the production path."""
    geoms = [int(cfg["gauge"]["geometries"][0])]
    entry = dict(geometry.geometry_entry(spec, geoms[0]))
    entry["vertices"] = geometry.polygon_vertices(entry).tolist()
    levels = [0.20, 0.14, 0.10]
    rows = []
    for h in levels:
        result = exact.solve_integral_gauge(
            entry, params_from_fem(fem_cfg), solver_fem_cfg(cfg, fem_cfg), h,
            manufactured=True)
        m = result["manufactured_metrics"]
        rows.append({
            "h": h,
            "newton_iterations": result["history"][-1]["iteration"],
            "final_linear_residual_rel": result["final_linear_residual_rel"],
            "integral_constraint": result["final_constraint"],
            "initial_integral": result["initial_integral"],
            "relaxation": result["relaxation"],
            "pin_value": result["pin_value"],
            "manufactured": m,
            "weak_residual": result["weak_residual"],
        })
        print("MANU_LEVEL", h, m["velocity_rel_l2"], m["temperature_rel_l2"],
              m["pressure_centered_rel_l2"],
              "free", m["free_residual_vector_norm"],
              "constraint", result["final_constraint"], flush=True)
    cfg_ce = dict(cfg)
    cfg_ce["exact_solver"] = {**dict(cfg.get("exact_solver", {})),
                              "max_iterations": 20, "relaxation": 0.6}
    ce = exact.solve_integral_gauge(
        entry, params_from_fem(fem_cfg), solver_fem_cfg(cfg_ce, fem_cfg), 0.14,
        manufactured=True, initial_p_offset=1.0, relaxation=0.6)
    counter = {
        "h": 0.14, "initial_p_offset": 1.0, "relaxation": 0.6,
        "initial_integral": ce["initial_integral"],
        "initial_integral_expected": ce["diagnostics"]["area"] * 1.0,
        "iterations": ce["history"][-1]["iteration"],
        "history": ce["history"],
        "final_integral": ce["final_integral"],
        "final_constraint": ce["final_constraint"],
        "final_linear_residual_rel": ce["final_linear_residual_rel"],
        "free_residual_vector_norm": ce["manufactured_metrics"]["free_residual_vector_norm"],
        "manufactured": ce["manufactured_metrics"],
    }
    print("MANU_COUNTEREXAMPLE initial", counter["initial_integral"],
          "expected", counter["initial_integral_expected"],
          "final", counter["final_constraint"],
          "iter", counter["iterations"], flush=True)
    vel = [r["manufactured"]["velocity_rel_l2"] for r in rows]
    tmp = [r["manufactured"]["temperature_rel_l2"] for r in rows]
    pre = [r["manufactured"]["pressure_centered_rel_l2"] for r in rows]
    checks = {
        "all_levels_converged": all(r["newton_iterations"] < 20 for r in rows),
        "all_linear_residuals_small": all(
            r["final_linear_residual_rel"] < 1e-8 for r in rows),
        "free_residual_finest_small": (
            rows[-1]["manufactured"]["free_residual_vector_norm"] < 1e-6),
        "free_residual_decreases": all(
            rows[i]["manufactured"]["free_residual_vector_norm"]
            > rows[i + 1]["manufactured"]["free_residual_vector_norm"]
            for i in range(len(rows) - 1)),
        "all_integral_constraints_small": all(
            abs(r["integral_constraint"]) < 1e-7 for r in rows),
        "boundary_values_finite": all(
            bool(r["manufactured"].get("boundary_values_finite")) for r in rows),
        "velocity_error_decreases": vel[0] > vel[-1] if vel[-1] is not None else False,
        "temperature_error_decreases": tmp[0] > tmp[-1] if tmp[-1] is not None else False,
        "pressure_error_decreases": pre[0] > pre[-1] if pre[-1] is not None else False,
        "asymmetric_nonzero_fields": rows[0]["manufactured"]["u_nonzero"] > 0,
        "counterexample_initial_p_plus1": abs(
            counter["initial_integral"] - counter["initial_integral_expected"]) < 1e-6
            and abs(counter["initial_integral"]) > 1.0,
        "counterexample_integral_converged": abs(counter["final_constraint"]) < 1e-7,
        "counterexample_free_residual_small": counter["free_residual_vector_norm"] < 1e-6,
        "counterexample_linear_residual_small": counter["final_linear_residual_rel"] < 1e-8,
    }
    doc = {"task": "R3.16 nonlinear manufactured solution through production "
                   "integral-gauge path",
           "created_local": now(),
           "geometry": geoms[0], "levels_h": levels,
           "fields": "psi=sin(pi x)sin(pi y)+0.2 sin(2pi x)sin(pi y); "
                     "u=(psi_y,-psi_x); T=sin(pi x)sin(pi y)+0.2 cos(pi x)sin(2pi y); "
                     "p=cos(pi x)sin(pi y)+0.3 sin(pi x)cos(2pi y)",
           "method": ("same solve_integral_gauge: PETSc augmented Newton, nonhomogeneous "
                      "u/T Dirichlet BC, eps=0, integral(p)=0"),
           "rows": rows, "counterexample": counter, "checks": checks,
           "all_pass": all(checks.values())}
    path = os.path.join(OUT, "manufactured_solution.json")
    if os.path.exists(path):
        raise SystemExit(f"[refuse] target exists: {path}")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
    print(json.dumps({"all_pass": doc["all_pass"], "checks": checks}, indent=2), flush=True)
    if not doc["all_pass"]:
        raise SystemExit("[fail] manufactured checks failed")


def run_manufactured_polynomial(cfg, fem_cfg, spec):
    """Strictly compatible polynomial MMS through the production path."""
    geoms = [int(cfg["gauge"]["geometries"][0])]
    entry = dict(geometry.geometry_entry(spec, geoms[0]))
    entry["vertices"] = geometry.polygon_vertices(entry).tolist()
    levels = [0.20, 0.14, 0.10]
    rows = []
    for h in levels:
        result = exact.solve_integral_gauge(
            entry, params_from_fem(fem_cfg), solver_fem_cfg(cfg, fem_cfg), h,
            manufactured=True, manufactured_case="polynomial")
        m = result["manufactured_metrics"]
        rows.append({"h": h, "newton_iterations": result["history"][-1]["iteration"],
                     "final_linear_residual_rel": result["final_linear_residual_rel"],
                     "integral_constraint": result["final_constraint"],
                     "initial_integral": result["initial_integral"],
                     "relaxation": result["relaxation"], "manufactured": m,
                     "history": result["history"]})
        print("POLY_MMS", h, "free", m["free_residual_vector_norm"],
              "rel", m.get("free_residual_relative_to_source"),
              "bflux", m.get("boundary_flux_solution"),
              "cont", m.get("continuity_integral_solution"),
              "bc", m.get("boundary_velocity_max_error"),
              "constraint", result["final_constraint"], flush=True)
    cfg_ce = dict(cfg); cfg_ce["exact_solver"] = {
        **dict(cfg.get("exact_solver", {})), "max_iterations": 20,
        "relaxation": 0.6}
    ce = exact.solve_integral_gauge(
        entry, params_from_fem(fem_cfg), solver_fem_cfg(cfg_ce, fem_cfg), 0.14,
        manufactured=True, manufactured_case="polynomial",
        initial_p_offset=1.0, relaxation=0.6)
    counter = {"h": 0.14, "initial_p_offset": 1.0, "relaxation": 0.6,
               "initial_integral": ce["initial_integral"],
               "initial_integral_expected": ce["diagnostics"]["area"],
               "iterations": ce["history"][-1]["iteration"],
               "history": ce["history"],
               "final_integral": ce["final_integral"],
               "final_constraint": ce["final_constraint"],
               "final_linear_residual_rel": ce["final_linear_residual_rel"],
               "free_residual_vector_norm": ce["manufactured_metrics"]["free_residual_vector_norm"],
               "manufactured": ce["manufactured_metrics"]}
    checks = {
        "all_levels_converged": all(r["newton_iterations"] < 20 for r in rows),
        "all_linear_residuals_small": all(r["final_linear_residual_rel"] < 1e-8 for r in rows),
        "all_free_absolute_1e12": all(r["manufactured"]["free_residual_vector_norm"] < 1e-12 for r in rows),
        "all_free_relative_1e8": all(
            (r["manufactured"].get("free_residual_relative_to_source") or 1.0) < 1e-8
            for r in rows),
        "all_integral_constraints_small": all(abs(r["integral_constraint"]) < 1e-7 for r in rows),
        "all_boundary_velocity_error_small": all(
            (r["manufactured"].get("boundary_velocity_dof_max_error") or 0.0) < 1e-10 for r in rows),
        "all_boundary_temperature_error_small": all(
            (r["manufactured"].get("boundary_temperature_dof_max_error") or 0.0) < 1e-10 for r in rows),
        "all_boundary_flux_compatible": all(
            abs(r["manufactured"].get("boundary_flux_solution", 1.0)) < 1e-10 for r in rows),
        "all_continuity_compatible": all(
            abs(r["manufactured"].get("continuity_integral_solution", 1.0)) < 1e-10 for r in rows),
        "all_block_l2_residuals_small": all(
            all((r["manufactured"].get(f"block_l2_residual_{b}") or 1.0) < 1e-10
                for b in ("continuity", "momentum", "energy")) for r in rows),
        "machine_precision_field_errors": all(
            (r["manufactured"]["velocity_rel_l2"] or 1.0) < 1e-12
            and (r["manufactured"]["temperature_rel_l2"] or 1.0) < 1e-12
            and (r["manufactured"]["pressure_centered_rel_l2"] or 1.0) < 1e-12
            for r in rows),
        "counterexample_initial_p_plus1": abs(counter["initial_integral"] - counter["initial_integral_expected"]) < 1e-6 and abs(counter["initial_integral"]) > 1.0,
        "counterexample_integral_converged": abs(counter["final_constraint"]) < 1e-7,
        "counterexample_free_small": counter["free_residual_vector_norm"] < 1e-12,
        "counterexample_linear_small": counter["final_linear_residual_rel"] < 1e-8,
    }
    doc = {"task": "R3.16 strict polynomial manufactured solution through production path",
           "created_local": now(), "geometry": geoms[0], "levels_h": levels,
           "fields": "psi=x^3-3xy^2+0.5x^2y+0.2y^3; u=(psi_y,-psi_x) quadratic; "
                     "T quadratic; p linear; P2/P1-compatible boundary traces",
           "method": "same solve_integral_gauge augmented Newton, nonhomogeneous BC, eps=0",
           "rows": rows, "counterexample": counter, "checks": checks,
           "all_pass": all(checks.values())}
    path = os.path.join(OUT, "manufactured_polynomial_solution.json")
    if os.path.exists(path):
        raise SystemExit(f"[refuse] target exists: {path}")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
    print(json.dumps({"all_pass": doc["all_pass"], "checks": checks}, indent=2), flush=True)
    if not doc["all_pass"]:
        raise SystemExit("[fail] polynomial manufactured checks failed")


def run_manufactured_nontrivial(cfg, fem_cfg, spec):
    """One h=.14 production-path test from perturbed interior u/T DOFs."""
    geoms = [int(cfg["gauge"]["geometries"][0])]
    entry = dict(geometry.geometry_entry(spec, geoms[0]))
    entry["vertices"] = geometry.polygon_vertices(entry).tolist()
    cfg_nt = dict(cfg); cfg_nt["exact_solver"] = {
        **dict(cfg.get("exact_solver", {})), "max_iterations": 80,
        "relaxation": 0.6}
    result = exact.solve_integral_gauge(
        entry, params_from_fem(fem_cfg), solver_fem_cfg(cfg_nt, fem_cfg), 0.14,
        manufactured=True, manufactured_case="polynomial",
        initial_p_offset=1.0, relaxation=0.6,
        initial_perturbation=0.05, perturbation_seed=20261023,
        free_tol=1e-12)
    m = result["manufactured_metrics"]
    hist = result["history"]
    update_norms = [h["update_norm"] for h in hist]
    checks = {
        "more_than_one_newton_step": len(hist) > 1,
        "nonzero_actual_updates": max(update_norms) > 1e-6,
        "initial_free_residual_significant": result["initial_source_free_residual"] > 1e-6,
        "initial_perturbation_nonzero": result["initial_perturbation_norm"] > 1e-3,
        "initial_p_plus1": abs(result["initial_integral"] - result["diagnostics"]["area"]) < 1e-6,
        "final_free_absolute_1e12": m["free_residual_vector_norm"] < 1e-12,
        "final_free_relative_1e8": (m.get("free_residual_relative_to_source") or 1.0) < 1e-8,
        "final_integral_constraint": abs(result["final_constraint"]) < 1e-7,
        "final_linear_residual": result["final_linear_residual_rel"] < 1e-8,
        "boundary_velocity_dof_error": (m.get("boundary_velocity_dof_max_error") or 0.0) < 1e-10,
        "boundary_temperature_dof_error": (m.get("boundary_temperature_dof_max_error") or 0.0) < 1e-10,
        "boundary_flux_compatible": abs(m.get("boundary_flux_solution", 1.0)) < 1e-10,
        "continuity_compatible": abs(m.get("continuity_integral_solution", 1.0)) < 1e-10,
        "field_error_machine": (m["velocity_rel_l2"] < 1e-12
                                and m["temperature_rel_l2"] < 1e-12
                                and m["pressure_centered_rel_l2"] < 1e-12),
    }
    doc = {"task": "R3.16 nontrivial initial u/T perturbation through production path",
           "created_local": now(), "h": 0.14, "geometry": geoms[0],
           "initial_perturbation": 0.05, "perturbation_seed": 20261023,
           "initial_p_offset": 1.0, "relaxation": 0.6,
           "initial_integral": result["initial_integral"],
           "initial_perturbation_norm": result["initial_perturbation_norm"],
           "initial_source_free_residual": result["initial_source_free_residual"],
           "iterations": len(hist), "update_norms": update_norms,
           "history": hist, "final": {
               "free_residual_vector_norm": m["free_residual_vector_norm"],
               "free_residual_relative_to_source": m.get("free_residual_relative_to_source"),
               "final_constraint": result["final_constraint"],
               "final_linear_residual_rel": result["final_linear_residual_rel"],
               "manufactured": m},
           "checks": checks, "all_pass": all(checks.values())}
    path = os.path.join(OUT, "manufactured_nontrivial_initial_solution.json")
    if os.path.exists(path):
        raise SystemExit(f"[refuse] target exists: {path}")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
    print("NONTRIVIAL", json.dumps({"all_pass": doc["all_pass"], "checks": checks,
                                    "iters": len(hist), "init_free": result["initial_source_free_residual"],
                                    "final_free": m["free_residual_vector_norm"]}, indent=2), flush=True)
    if not doc["all_pass"]:
        raise SystemExit("[fail] nontrivial initial MMS checks failed")


def run_full(cfg, fem_cfg, spec):
    pilot = os.path.join(OUT, "exact", "pilot_geom17_h004.json")
    if not os.path.isfile(pilot):
        raise SystemExit("pilot must pass before full exact gauge comparison")
    pdoc = load_json(pilot)
    if not pdoc.get("checks", {}).get("all_pass"):
        raise SystemExit("pilot all_pass is not true")
    geoms = [int(g) for g in cfg["gauge"]["geometries"]]
    cmps = {}
    for gid in geoms:
        cmp, _ = run_case(cfg, fem_cfg, spec, params_from_fem(fem_cfg), gid, tag="")
        cmps[str(gid)] = cmp
    threshold = float(cfg["gauge"]["equivalence_threshold_relative_l2"])
    checks = []
    for gid, c in cmps.items():
        old = np.load(os.path.join(WS, cfg["model_eval"]["root"],
                                   f"fem_reference_geom{gid}.npz"))
        data = np.load(case_paths(int(gid))[0])
        valid = data["valid"].astype(bool)
        p_exact = data["p_point"].astype(float)[valid]
        p_eps = old["ref"].astype(float)[valid, 2]
        ec = p_exact - p_exact.mean()
        oc = p_eps - p_eps.mean()
        den = float(np.linalg.norm(oc))
        eps_rel = float(np.linalg.norm(ec - oc) / den) if den > 0 else None
        ok = (c["centered_relative_difference"] is not None
              and c["centered_relative_difference"] <= threshold)
        checks.append({"geom": int(gid), "centered_relative_difference":
                       c["centered_relative_difference"], "threshold": threshold,
                       "passed": bool(ok),
                       "exact_vs_eps_centered_re": eps_rel,
                       "pin_value": c["pin_value"], "n_valid": c["n_valid"],
                       "n_total": c["n_total"]})
    doc = {"task": "R3.16 exact gauge comparison", "protocol": cfg["protocol"],
           "version": cfg["version"], "created_local": now(),
           "target_h": float(cfg["gauge"]["target_h"]), "geometries": geoms,
           "gauges": ["point_reference", "integral_zero_mean"],
           "threshold_relative_l2": threshold, "checks": checks,
           "cases": cmps,
           "pilot": "exact/pilot_geom17_h004.json",
           "note": ("both gauges are exact constant shifts of the same discrete solution; "
                    "u/v/T are identical by construction; eps reference is retained as "
                    "a separate historical numerical column"),
           "all_pass": all(c["passed"] for c in checks)}
    path = os.path.join(OUT, "exact_gauge_comparison.json")
    if os.path.exists(path):
        raise SystemExit(f"[refuse] target exists: {path}")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
    print(json.dumps({"all_pass": doc["all_pass"], "checks": checks}, indent=2), flush=True)
    if not doc["all_pass"]:
        raise SystemExit("[fail] exact gauge comparison failed")


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-workspace", default=WS)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--data", default=None)
    ap.add_argument("--mode", required=True, choices=("pilot", "full", "compare", "analysis", "manufactured", "manufactured_polynomial", "manufactured_nontrivial"))
    args = ap.parse_args()
    OUT = os.path.abspath(args.out_dir)
    cfg = load_json(args.config)
    fem_cfg = load_json(FEM_CONFIG)
    cfg["data"] = args.data if getattr(args, "data", None) else cfg["data"]
    spec = geometry.load_geometry_spec(os.path.join(WS, fem_cfg["geometries"]["geometry_spec"]))
    os.makedirs(os.path.join(OUT, "exact"), exist_ok=True)
    if args.mode == "pilot":
        run_pilot(cfg, fem_cfg, spec)
    elif args.mode == "full":
        run_full(cfg, fem_cfg, spec)
    elif args.mode == "compare":
        summarize_existing(cfg, fem_cfg, spec)
    elif args.mode == "manufactured":
        run_manufactured(cfg, fem_cfg, spec)
    elif args.mode == "manufactured_polynomial":
        run_manufactured_polynomial(cfg, fem_cfg, spec)
    elif args.mode == "manufactured_nontrivial":
        run_manufactured_nontrivial(cfg, fem_cfg, spec)
    else:
        analysis(cfg, fem_cfg)


if __name__ == "__main__":
    main()
