"""R2-THERMAL-FEM-02 experiment entry (review implementation).

Modes
-----
probe        local geometry/protocol/dependency probe (no formal FEM solve)
manufactured manufactured Stokes/advection-diffusion verification (needs FEM env)
pilot        one steady natural-convection solve (needs FEM env; SLURM only)
levels       natural-convection solves at the three pre-registered h values
representatives  seven 46 / eight 17 / nine 96 remeshing review entry
                  (first h by default; explicit --levels expands the run)

No mode runs a formal natural-convection solve on a login node; Codex schedules
SLURM execution after reviewing this implementation.  The script never downloads
dependencies and never writes outside results/round2_thermal_fem/.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time

import numpy as np

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from src.fem_reference import fenics_solver, geometry  # noqa: E402

CONFIG_PATH = os.path.join(WORKSPACE, "configs", "r2_thermal_fem.json")
OUT_ROOT = os.path.join(WORKSPACE, "results", "round2_thermal_fem")
SUPPORT_ROOT = os.path.join(WORKSPACE, "results", "round2_thermal_fem", "model_eval")
REFERENCE_ROOT = os.path.join(WORKSPACE, "results", "round2_thermal_fem", "v2_reference")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config() -> dict:
    return geometry.load_json(CONFIG_PATH)


def geometry_entry(config: dict, geom_id: int) -> dict:
    spec = geometry.load_geometry_spec(
        os.path.join(WORKSPACE, config["geometries"]["geometry_spec"]))
    entry = dict(geometry.geometry_entry(spec, int(geom_id)))
    entry["vertices"] = geometry.polygon_vertices(entry).tolist()
    return entry


def _file_hash(path: str) -> str | None:
    return sha256_file(path) if os.path.isfile(path) else None


def protocol_snapshot(config: dict, mode: str) -> dict:
    deps = fenics_solver.dependency_status()
    source_files = (
        "src/fem_reference/__init__.py",
        "src/fem_reference/geometry.py",
        "src/fem_reference/fenics_solver.py",
        "scripts/exp_r2_thermal_fem.py",
        "scripts/probe_r2_thermal_fem.py",
        "scripts/sbatch/r2_thermal_fem.sbatch",
    )
    geometry_spec = os.path.join(WORKSPACE,
                                 config["geometries"]["geometry_spec"])
    split_config = os.path.join(WORKSPACE,
                                config["geometries"]["split_config"])
    boundary_data = os.path.abspath(os.path.join(
        WORKSPACE, config["geometries"]["boundary_data"]))
    return {
        "protocol": config["protocol_id"],
        "protocol_version": config["protocol_version"],
        "module": config["module"],
        "mode": mode,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_sha256": sha256_file(CONFIG_PATH),
        "source_hashes": {rel: _file_hash(os.path.join(WORKSPACE, rel))
                          for rel in source_files},
        "input_hashes": {
            "geometry_spec": _file_hash(geometry_spec),
            "split_config": _file_hash(split_config),
            "boundary_data": _file_hash(boundary_data),
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "fem_dependency_status": deps,
        "physics": config["physics"],
        "geometries": config["geometries"],
        "mesh": config["mesh"],
        "elements": config["elements"],
        "solver": config["solver"],
        "acceptance_thresholds": config["acceptance_thresholds"],
        "note": ("Implementation protocol snapshot only; no formal result is "
                 "claimed by this file."),
    }


def mesh_metadata(result: dict) -> dict:
    """Return only JSON-serialisable mesh metadata (not dolfinx objects)."""
    mesh = result.get("mesh", {})
    keys = ("n_cells", "n_vertices", "target_h", "h_max")
    return {key: mesh.get(key) for key in keys}


def write_json(path: str, document: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=False)


def write_report(path: str, lines: list) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def compare_original_nodes(result: dict, entry: dict, config: dict) -> dict:
    """Evaluate the FEM solution at the original 5000 stored nodes for diagnosis."""
    data_path = os.path.abspath(
        os.path.join(WORKSPACE, config["geometries"]["boundary_data"]))
    data = np.load(data_path)
    geom = int(entry["geom"])
    points = np.asarray(data[geom, :, :2], dtype=float)
    reference = np.asarray(data[geom, :, 2:6], dtype=float)
    boundary_count = int(config["geometries"].get("boundary_n", 660))
    return fenics_solver.compare_with_reference(
        result, points, reference, boundary_count=boundary_count)


def evaluate_support(result: dict, entry: dict, config: dict,
                     target_h: float, suffix: str = "") -> dict:
    """Evaluate the FEM solution at the frozen 5000-point support with dolfinx."""
    geom = int(entry["geom"])
    support_path = os.path.join(SUPPORT_ROOT, f"support_geom{geom}.npz")
    if not os.path.isfile(support_path):
        raise RuntimeError(f"missing frozen support: {support_path}")
    coords = np.load(support_path)["coords"]
    values = fenics_solver.evaluate_at_points(result, coords)
    valid = values["valid"]
    payload = {
        "coords": coords, "valid": valid.astype(np.int8),
        "cells": values["cells"], "u": values["u"], "v": values["v"],
        "p": values["p"], "T": values["T"],
    }
    path = os.path.join(REFERENCE_ROOT,
                        f"fem_reference_geom{geom}_h{target_h:g}{suffix}.npz")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **payload)
    return {"path": os.path.relpath(path, WORKSPACE),
            "n_points": int(len(coords)),
            "n_valid": int(valid.sum()),
            "n_outside": int((~valid).sum()),
            "n_nonfinite": int(np.sum(~np.isfinite(values["u"]))
                               + np.sum(~np.isfinite(values["v"]))
                               + np.sum(~np.isfinite(values["p"]))
                               + np.sum(~np.isfinite(values["T"])))}


def run_probe(args) -> int:
    probe_path = os.path.join(WORKSPACE, "scripts", "probe_r2_thermal_fem.py")
    spec = importlib.util.spec_from_file_location("probe_r2_thermal_fem",
                                                  probe_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main(["--output", args.output or os.path.join(
        OUT_ROOT, "probe", "probe_report.json"),
        *(["--run-fem-probes"] if args.run_fem_probes else [])])


def run_manufactured(args, config: dict) -> int:
    entry = geometry_entry(config, args.geom or config["geometries"]["pilot"])
    levels = args.levels or [0.20, 0.10, 0.05]
    records = []
    for target_h in levels:
        result = fenics_solver.solve_manufactured(entry, config["physics"],
                                                  config, target_h,
                                                  mode="stokes")
        records.append({"target_h": target_h,
                        "mesh": result["mesh"],
                        "errors": result["errors"],
                        "newton_reason": result["newton_reason"]})
    orders = {}
    for key in records[0]["errors"]:
        orders[key] = [
            math.log(max(records[i]["errors"][key], 1e-300)
                     / max(records[i + 1]["errors"][key], 1e-300))
            / math.log(2.0)
            for i in range(len(records) - 1)
        ]
    out = {"protocol_snapshot": protocol_snapshot(config, "manufactured"),
           "geom": int(entry["geom"]), "records": records,
           "observed_orders": orders,
           "status": "manufactured_ran_not_natural_convection_acceptance"}
    path = args.output or os.path.join(OUT_ROOT, "manufactured",
                                       f"manufactured_geom{entry['geom']}.json")
    write_json(path, out)
    print(json.dumps({"status": out["status"], "orders": orders,
                      "output": os.path.abspath(path)}, indent=2))
    return 0


def write_fields(result: dict, path: str) -> str:
    """Write collapsed FEM fields and their DOF coordinates to a compressed NPZ."""
    solution = result["solution"]
    arrays = {}
    for name, sub_index in (("velocity", 0), ("pressure", 1), ("temperature", 2)):
        function = solution.sub(sub_index).collapse()
        space = function.function_space
        arrays[f"{name}_dof_coords"] = np.asarray(
            space.tabulate_dof_coordinates())
        arrays[f"{name}_values"] = np.asarray(function.x.array)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def run_pilot(args, config: dict) -> int:
    entry = geometry_entry(config, args.geom or config["geometries"]["pilot"])
    target_h = args.target_h or config["mesh"]["levels"][0]["target_h"]
    result = fenics_solver.solve_natural_convection(entry, config["physics"],
                                                    config, target_h)
    diagnostics = result["diagnostics"]
    path = args.output or os.path.join(OUT_ROOT, "pilot",
                                       f"pilot_geom{entry['geom']}_h{target_h}.json")
    fields_path = os.path.join(os.path.dirname(os.path.abspath(path)),
                               os.path.splitext(os.path.basename(path))[0] + "_fields.npz")
    write_fields(result, fields_path)
    out = {"protocol_snapshot": protocol_snapshot(config, "pilot"),
           "geom": int(entry["geom"]), "target_h": target_h,
           "mesh": mesh_metadata(result),
           "newton_reason": result["newton_reason"],
           "newton_iterations": result.get("newton_iterations"),
           "continuation": result.get("continuation"),
           "weak_residual": result.get("weak_residual"),
           "diagnostics": diagnostics,
           "fields_file": os.path.relpath(fields_path, WORKSPACE),
           "comparison_original_nodes": compare_original_nodes(
               result, entry, config) if args.compare_original else None,
           "fem_support": evaluate_support(result, entry, config, target_h)
           if args.eval_support else None,
           "acceptance": evaluate_natural_acceptance(
               diagnostics, config, result["newton_reason"],
               result.get("weak_residual"))}
    write_json(path, out)
    print(json.dumps({"status": out["acceptance"], "output": os.path.abspath(path)},
                     indent=2))
    return 0


def evaluate_natural_acceptance(diagnostics: dict, config: dict,
                                newton_reason=None,
                                weak_residual=None) -> dict:
    thresholds = config["acceptance_thresholds"]["natural_convection"]
    boundary = diagnostics.get("boundary", {})
    weak = (weak_residual or {}).get("blocks", {})
    full = weak.get("full", {})
    free = full.get("free")
    raw = full.get("full")
    free_rel = (free / raw) if (free is not None and raw not in (None, 0)) else None
    newton_converged = (newton_reason is not None and int(newton_reason) > 0)
    checks = {
        "finite_all_fields": bool(diagnostics.get("finite")),
        "newton_converged": bool(newton_converged)
        if thresholds.get("newton_reason_must_be_converged") else True,
        "galerkin_free_absolute": free is not None
        and free <= thresholds["galerkin_free_absolute_max"],
        "galerkin_free_relative": free_rel is not None
        and free_rel <= thresholds["galerkin_free_relative_max"],
        "wall_velocity": boundary.get("wall_velocity_max", 1e9)
        <= thresholds["wall_velocity_max"],
        "temperature_bc": max(boundary.get("inner_temperature_max_dev", 1e9),
                              boundary.get("outer_temperature_max_dev", 1e9))
        <= thresholds["temperature_bc_max_dev"],
        "mass_flux": abs(diagnostics.get("mass_flux", 1.0))
        <= thresholds["mass_flux_relative_max"]
        * max(diagnostics.get("velocity_l2", 1.0), 1e-300),
        "pressure_reported": "pressure_mean" in diagnostics,
    }
    return {"checks": checks, "all_passed": all(checks.values()),
            "weak_residual_used": weak_residual is not None,
            "cell_projection_diagnostic": {
                key: weak.get(key) for key in ("continuity", "momentum", "energy")},
            "note": "true gate is assembled Galerkin residual (full block, all "
                    "Dirichlet rows zeroed), absolute and relative; cell "
                    "projection numbers are diagnostics only, not nonlinear "
                    "weak residual; Newton reason cannot override the free "
                    "residual. Stokes/finite-only never count as acceptance."}


def run_levels(args, config: dict) -> int:
    entry = geometry_entry(config, args.geom or config["geometries"]["pilot"])
    records = []
    for level in config["mesh"]["levels"]:
        result = fenics_solver.solve_natural_convection(
            entry, config["physics"], config, level["target_h"])
        fields_path = os.path.join(
            OUT_ROOT, "levels", f"fields_geom{entry['geom']}_{level['name']}.npz")
        write_fields(result, fields_path)
        records.append({"level": level["name"], "target_h": level["target_h"],
                        "mesh": mesh_metadata(result),
                        "newton_reason": result["newton_reason"],
                        "newton_iterations": result.get("newton_iterations"),
                        "continuation": result.get("continuation"),
                        "weak_residual": result.get("weak_residual"),
                        "diagnostics": result["diagnostics"],
                        "acceptance": evaluate_natural_acceptance(
                            result["diagnostics"], config,
                            result["newton_reason"],
                            result.get("weak_residual")),
                        "fields_file": os.path.relpath(fields_path, WORKSPACE),
                        "comparison_original_nodes": compare_original_nodes(
                            result, entry, config)
                        if args.compare_original else None,
                        "fem_support": evaluate_support(
                            result, entry, config, level["target_h"])
                        if args.eval_support else None})
    out = {"protocol_snapshot": protocol_snapshot(config, "levels"),
           "geom": int(entry["geom"]), "records": records,
           "status": "levels_ran_review_against_thresholds"}
    path = args.output or os.path.join(OUT_ROOT, "levels",
                                       f"levels_geom{entry['geom']}.json")
    write_json(path, out)
    print(json.dumps({"status": out["status"], "output": os.path.abspath(path)},
                     indent=2))
    return 0


def run_representatives(args, config: dict) -> int:
    """Three representative geometry classes (seven 46, eight 17, nine 96).

    By default this runs the first pre-registered h only, which is the review
    entry point for the three-class remeshing experiment.  Passing
    ``--levels`` explicitly expands to the requested h values; Codex decides
    whether the full nine-solve expansion is authorised after the pilot.
    """
    geometry_ids = [int(config["geometries"]["representative"][key])
                    for key in ("seven", "eight", "nine")]
    if getattr(args, "geoms", None):
        geometry_ids = [int(g) for g in args.geoms]
    elif args.geom is not None:
        geometry_ids = [int(args.geom)]
    target_hs = args.levels or [config["mesh"]["levels"][0]["target_h"]]
    records = []
    for geom_id in geometry_ids:
        entry = geometry_entry(config, geom_id)
        for target_h in target_hs:
            result = fenics_solver.solve_natural_convection(
                entry, config["physics"], config, target_h)
            fields_path = os.path.join(
                OUT_ROOT, "representatives",
                f"fields_geom{entry['geom']}_h{target_h}.npz")
            write_fields(result, fields_path)
            records.append({
                "geom": int(entry["geom"]),
                "fields_file": os.path.relpath(fields_path, WORKSPACE),
                "comparison_original_nodes": compare_original_nodes(
                    result, entry, config) if args.compare_original else None,
                "fem_support": evaluate_support(
                    result, entry, config, target_h)
                if args.eval_support else None,
                "geometry_class": int(entry["n"]),
                "target_h": target_h,
                "mesh": mesh_metadata(result),
                "newton_reason": result["newton_reason"],
                "newton_iterations": result.get("newton_iterations"),
                "continuation": result.get("continuation"),
                "weak_residual": result.get("weak_residual"),
                "diagnostics": result["diagnostics"],
                "acceptance": evaluate_natural_acceptance(
                    result["diagnostics"], config, result["newton_reason"],
                    result.get("weak_residual")),
            })
    out = {"protocol_snapshot": protocol_snapshot(config, "representatives"),
           "geometry_ids": geometry_ids, "target_h_values": target_hs,
           "records": records,
           "status": "representatives_ran_review_against_thresholds"}
    path = args.output or os.path.join(
        OUT_ROOT, "representatives",
        "representatives_" + "_".join(str(g) for g in geometry_ids) + ".json")
    write_json(path, out)
    print(json.dumps({"status": out["status"], "n_records": len(records),
                      "output": os.path.abspath(path)}, indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="probe",
                        choices=("probe", "manufactured", "pilot", "levels",
                                 "representatives"))
    parser.add_argument("--geom", type=int, default=None)
    parser.add_argument("--geoms", type=int, nargs="*", default=None)
    parser.add_argument("--target-h", type=float, default=None)
    parser.add_argument("--levels", type=float, nargs="*", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--out-root", default=None,
                        help="optional versioned result root (default canonical)")
    parser.add_argument("--run-fem-probes", action="store_true")
    parser.add_argument("--eval-support", action="store_true",
                        help="evaluate the FEM solution at the frozen 5000-point "
                             "support with actual dolfinx Function.eval")
    parser.add_argument("--compare-original", action="store_true",
                        help="evaluate the FEM solution at the original 5000 "
                             "stored nodes (source-consistency diagnostic)")
    args = parser.parse_args(argv)
    config = load_config()
    global OUT_ROOT
    if args.out_root:
        OUT_ROOT = os.path.abspath(args.out_root)
    os.makedirs(OUT_ROOT, exist_ok=True)
    try:
        if args.mode == "probe":
            return run_probe(args)
        if args.mode == "manufactured":
            return run_manufactured(args, config)
        if args.mode == "pilot":
            return run_pilot(args, config)
        if args.mode == "representatives":
            return run_representatives(args, config)
        return run_levels(args, config)
    except fenics_solver.FemEnvironmentError as exc:
        path = args.output or os.path.join(OUT_ROOT, "blocked",
                                           f"blocked_{args.mode}.json")
        write_json(path, {"mode": args.mode, "status": "blocked",
                          "error": str(exc),
                          "protocol_snapshot": protocol_snapshot(config, args.mode)})
        print(json.dumps({"status": "blocked", "error": str(exc),
                          "output": os.path.abspath(path)}, indent=2))
        return 2


if __name__ == "__main__":
    sys.exit(main())
