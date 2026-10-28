"""R2-THERMAL-FEM-02 pre-flight probe (no formal natural-convection run).

Checks locally verifiable obligations before any SLURM FEM run:

* protocol parameters and the Ra/Pr identities are exactly the frozen values;
* representative test-geometry IDs (seven 46, eight 17, nine 96; pilot 17);
* geometry reconstruction from the stored 660 boundary points (outer square
  Hausdorff-like residual, inner point-to-polygon residual, boundary counts);
* wall velocity ~ 0 and inner/outer temperature conditions of the stored labels;
* the independent polar mesh: orientation, area conservation, boundary tags,
  quality metrics;
* exact dependency status of the FEM stack (dolfinx/ufl/basix/petsc4py/mpi4py/
  gmsh).  If a dependency is missing, the FEM numerical probes are recorded as
  BLOCKED with exact packages and an isolated-environment plan; this is not a
  pass and no formal solve is attempted here.

Optionally, when ``--run-fem-probes`` is given and the environment is complete,
manufactured Stokes/advection-diffusion convergence probes are run on the pilot
geometry (never a natural-convection acceptance run).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys

import numpy as np

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from src.fem_reference import fenics_solver, geometry  # noqa: E402

CONFIG_PATH = os.path.join(WORKSPACE, "configs", "r2_thermal_fem.json")
DATA_PATH = os.path.join(os.path.dirname(WORKSPACE), "HeatTransferData.npy")
OUT_DIR = os.path.join(WORKSPACE, "results", "round2_thermal_fem", "probe")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check(checks: list, name: str, ok: bool, detail: dict) -> None:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})


def protocol_checks(config: dict) -> dict:
    physics = config["physics"]
    rho = physics["rho"]
    cp = physics["cp"]
    gbeta = physics["g_beta"]
    mu = physics["mu"]
    kappa = physics["kappa"]
    length = 2.0
    delta_t = 1.0
    ra = rho ** 2 * cp * gbeta * delta_t * length ** 3 / (mu * kappa)
    pr = mu / kappa
    expected_mu = 2.0 * math.sqrt(2.0) * 10 ** -2.5
    return {
        "Ra": ra,
        "Ra_expected": 100000.0,
        "Ra_relative_error": abs(ra - 100000.0) / 100000.0,
        "Pr": pr,
        "Pr_expected": 1.0,
        "mu": mu,
        "kappa": kappa,
        "mu_expected": expected_mu,
        "mu_relative_error": abs(mu - expected_mu) / expected_mu,
        "domain": physics["domain"],
        "boundary_conditions": physics["boundary_conditions"],
        "sign_convention": physics["sign_check"],
    }


def geometry_checks(config: dict) -> list:
    spec = geometry.load_geometry_spec(
        os.path.join(WORKSPACE, config["geometries"]["geometry_spec"]))
    split = geometry.split_geometry_ids(
        os.path.join(WORKSPACE, config["geometries"]["split_config"]),
        config["geometries"]["split_protocol"])
    data_path = os.path.abspath(
        os.path.join(WORKSPACE, config["geometries"]["boundary_data"]))
    representatives = config["geometries"]["representative"]
    pilot = config["geometries"]["pilot"]
    rows = []
    for label, geom_id in representatives.items():
        entry = geometry.geometry_entry(spec, int(geom_id))
        boundary = geometry.load_boundary_points(
            data_path, int(geom_id), config["geometries"]["boundary_n"])
        verify = geometry.verify_geometry(entry, boundary)
        values = geometry.boundary_value_checks(boundary)
        verify.update(values)
        verify["representative_label"] = label
        verify["in_protocol_A_test"] = int(geom_id) in split["test"]
        verify["is_pilot"] = int(geom_id) == int(pilot)
        rows.append(verify)
        mesh = geometry.build_polar_mesh(
            entry, **config["mesh"]["probe"])
        verify["mesh"] = {key: mesh[key] for key in (
            "n_nodes", "n_triangles", "h_max", "area", "expected_area",
            "area_relative_error", "min_triangle_angle_deg", "max_edge_ratio")}
        inner_nodes = mesh["nodes"][mesh["boundary_side"] == 0]
        outer_nodes = mesh["nodes"][mesh["boundary_side"] == 1]
        verify["mesh_boundary"] = {
            "inner_max_distance_to_polygon": float(np.max(
                geometry._point_segment_distances(
                    inner_nodes, geometry.polygon_segments(entry)))),
            "outer_max_distance_to_square": float(np.max(
                geometry._point_segment_distances(
                    outer_nodes, geometry.square_segments()))),
            "inner_nodes": int(len(inner_nodes)),
            "outer_nodes": int(len(outer_nodes)),
        }
    return rows


def representative_id_checks(config: dict) -> dict:
    """Independently recompute the minimal test-geometry ID for each class."""
    spec = geometry.load_geometry_spec(
        os.path.join(WORKSPACE, config["geometries"]["geometry_spec"]))
    split = geometry.split_geometry_ids(
        os.path.join(WORKSPACE, config["geometries"]["split_config"]),
        config["geometries"]["split_protocol"])
    by_class = {}
    for geom_id in split["test"]:
        sides = int(geometry.geometry_entry(spec, geom_id)["n"])
        by_class.setdefault(sides, []).append(geom_id)
    min_by_class = {sides: min(ids) for sides, ids in by_class.items()}
    configured = {int(v) for v in config["geometries"]["representative"].values()}
    expected = {7: 46, 8: 17, 9: 96}
    return {
        "test_count": len(split["test"]),
        "min_id_by_sides": {str(k): v for k, v in sorted(min_by_class.items())},
        "expected_min_id_by_sides": {str(k): v for k, v in expected.items()},
        "configured_representatives": sorted(configured),
        "pilot": int(config["geometries"]["pilot"]),
        "ok": bool(min_by_class == expected
                   and configured == set(expected.values())
                   and int(config["geometries"]["pilot"]) == expected[8]),
    }


def manufactured_formula_checks(config: dict) -> dict:
    """Finite-difference check of the manufactured Stokes/energy source formulas.

    This is a local analytic check of the exact fields used by the FEM backend;
    it does not replace an FE solve.
    """
    mu = float(config["physics"]["mu"])
    kappa = float(config["physics"]["kappa"])
    points = np.array([[0.23, -0.37], [-0.41, 0.22], [0.12, 0.48]])
    step = 1e-5

    def u(point):
        x, y = point
        return np.array([math.sin(math.pi * x) * math.cos(math.pi * y),
                         -math.cos(math.pi * x) * math.sin(math.pi * y)])

    def p(point):
        return math.sin(math.pi * point[0]) * math.sin(math.pi * point[1])

    def t(point):
        return math.sin(math.pi * point[0]) * math.sin(math.pi * point[1])

    records = []
    for x, y in points:
        point = np.array([float(x), float(y)])
        uxx = (u(point + [step, 0])[0] - 2 * u(point)[0]
               + u(point - [step, 0])[0]) / step ** 2
        uyy = (u(point + [0, step])[0] - 2 * u(point)[0]
               + u(point - [0, step])[0]) / step ** 2
        vxx = (u(point + [step, 0])[1] - 2 * u(point)[1]
               + u(point - [step, 0])[1]) / step ** 2
        vyy = (u(point + [0, step])[1] - 2 * u(point)[1]
               + u(point - [0, step])[1]) / step ** 2
        px = (p(point + [step, 0]) - p(point - [step, 0])) / (2 * step)
        py = (p(point + [0, step]) - p(point - [0, step])) / (2 * step)
        ux = (u(point + [step, 0])[0] - u(point - [step, 0])[0]) / (2 * step)
        uy = (u(point + [0, step])[0] - u(point - [0, step])[0]) / (2 * step)
        vx = (u(point + [step, 0])[1] - u(point - [step, 0])[1]) / (2 * step)
        vy = (u(point + [0, step])[1] - u(point - [0, step])[1]) / (2 * step)
        tx = (t(point + [step, 0]) - t(point - [step, 0])) / (2 * step)
        ty = (t(point + [0, step]) - t(point - [0, step])) / (2 * step)
        txx = (t(point + [step, 0]) - 2 * t(point)
               + t(point - [step, 0])) / step ** 2
        tyy = (t(point + [0, step]) - 2 * t(point)
               + t(point - [0, step])) / step ** 2
        pi = math.pi
        sin_x, cos_x = math.sin(pi * x), math.cos(pi * x)
        sin_y, cos_y = math.sin(pi * y), math.cos(pi * y)
        expected_fx = (2.0 * mu * pi ** 2 * sin_x * cos_y
                       + pi * cos_x * sin_y
                       + pi * sin_x * cos_x)
        expected_fy = (-2.0 * mu * pi ** 2 * cos_x * sin_y
                       + pi * sin_x * cos_y
                       + pi * sin_y * cos_y
                       - sin_x * sin_y)
        expected_fT = (u(point)[0] * tx + u(point)[1] * ty
                       + 2.0 * kappa * pi ** 2 * t(point))
        fd_fx = (-mu * (uxx + uyy) + px
                 + u(point)[0] * ux + u(point)[1] * uy)
        fd_fy = (-mu * (vxx + vyy) + py
                 + u(point)[0] * vx + u(point)[1] * vy - t(point))
        divergence = ux + vy
        fd_fT = (u(point)[0] * tx + u(point)[1] * ty
                 - kappa * (txx + tyy))
        records.append({
            "point": [float(x), float(y)],
            "divergence_abs": float(abs(divergence)),
            "fx_abs_error": float(abs(fd_fx - expected_fx)),
            "fy_abs_error": float(abs(fd_fy - expected_fy)),
            "fT_abs_error": float(abs(fd_fT - expected_fT)),
        })
    max_error = max(max(item["fx_abs_error"], item["fy_abs_error"],
                        item["fT_abs_error"]) for item in records)
    max_div = max(item["divergence_abs"] for item in records)
    return {"records": records, "max_source_abs_error": float(max_error),
            "max_divergence_abs": float(max_div),
            "ok": bool(max_error < 1e-5 and max_div < 1e-8)}


def protocol_snapshot_check(config: dict) -> dict:
    """Verify the saved protocol snapshot binds the current config and sources."""
    path = os.path.join(WORKSPACE, "results", "round2_thermal_fem",
                        "protocol_snapshot.json")
    if not os.path.isfile(path):
        return {"ok": False, "reason": "protocol_snapshot.json missing"}
    snapshot = geometry.load_json(path)
    checks = []
    checks.append(("config_sha256",
                   snapshot.get("config_sha256") == sha256_file(CONFIG_PATH)))
    current_hashes = {}
    for rel in snapshot.get("source_hashes", {}):
        if rel == "scripts/probe_r2_thermal_fem.py":
            continue  # probe is not part of its own snapshot by design
        full = os.path.join(WORKSPACE, rel)
        current_hashes[rel] = sha256_file(full) if os.path.isfile(full) else None
    for rel, saved in snapshot.get("source_hashes", {}).items():
        if rel == "scripts/probe_r2_thermal_fem.py":
            continue
        checks.append((f"source:{rel}", current_hashes.get(rel) == saved))
    for name, rel in (("geometry_spec", config["geometries"]["geometry_spec"]),
                      ("split_config", config["geometries"]["split_config"])):
        full = os.path.join(WORKSPACE, rel)
        checks.append((f"input:{name}",
                       snapshot.get("input_hashes", {}).get(name)
                       == (sha256_file(full) if os.path.isfile(full) else None)))
    return {"ok": all(ok for _, ok in checks),
            "checks": {name: bool(ok) for name, ok in checks},
            "path": os.path.relpath(path, WORKSPACE)}


def acceptance_logic_checks(config: dict) -> dict:
    """Exercise the acceptance gate with synthetic diagnostics (no FEM solve).

    A finite/Stokes-only field must not pass natural-convection acceptance, and a
    missing/zero Newton convergence reason must fail.
    """
    import importlib.util
    exp_path = os.path.join(WORKSPACE, "scripts", "exp_r2_thermal_fem.py")
    spec = importlib.util.spec_from_file_location("exp_r2_thermal_fem_probe",
                                                  exp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    good = {
        "finite": True, "divergence_l2": 0.0, "mass_flux": 0.0,
        "velocity_l2": 1.0, "pressure_mean": 0.0,
        "residual": {"momentum_l2": 0.0, "energy_l2": 0.0},
        "boundary": {"wall_velocity_max": 0.0,
                     "inner_temperature_max_dev": 0.0,
                     "outer_temperature_max_dev": 0.0},
        "weak_residual": {"blocks": {
            "full": {"free": 1e-15, "full": 0.1},
            "continuity": {"relative_free": 0.0},
            "momentum": {"relative_free": 0.0},
            "energy": {"relative_free": 0.0}}},
    }
    finite_only = {"finite": True, "divergence_l2": 0.0,
                   "mass_flux": 0.0, "velocity_l2": 1.0,
                   "pressure_mean": 0.0}
    large_free = json.loads(json.dumps(good))
    large_free["weak_residual"]["blocks"]["full"]["free"] = 1.0
    good_result = module.evaluate_natural_acceptance(
        good, config, 2, good["weak_residual"])
    finite_result = module.evaluate_natural_acceptance(
        finite_only, config, 2, None)
    no_newton_result = module.evaluate_natural_acceptance(
        good, config, None, good["weak_residual"])
    large_free_result = module.evaluate_natural_acceptance(
        large_free, config, 2, large_free["weak_residual"])
    return {
        "good_all_passed": bool(good_result["all_passed"]),
        "finite_only_all_passed": bool(finite_result["all_passed"]),
        "no_newton_all_passed": bool(no_newton_result["all_passed"]),
        "large_galerkin_free_all_passed": bool(large_free_result["all_passed"]),
        "ok": bool(good_result["all_passed"]
                   and not finite_result["all_passed"]
                   and not no_newton_result["all_passed"]
                   and not large_free_result["all_passed"]),
    }


def mesh_entity_checks(rows: list, config: dict) -> list:
    thresholds = config["acceptance_thresholds"]["mesh"]
    checks = []
    for row in rows:
        mesh = row["mesh"]
        checks.append({
            "name": f"mesh_area_geom{row['geom']}",
            "ok": mesh["area_relative_error"] <= thresholds["area_relative_error"],
            "detail": mesh,
        })
        checks.append({
            "name": f"mesh_quality_geom{row['geom']}",
            "ok": (mesh["min_triangle_angle_deg"] >= thresholds["min_triangle_angle_deg"]
                   and mesh["max_edge_ratio"] <= thresholds["max_edge_ratio"]),
            "detail": mesh,
        })
    return checks


def comparison_metric_checks() -> dict:
    """Check the source-consistency metric helpers on synthetic arrays."""
    truth = np.array([1.0, 2.0, 3.0])
    exact = fenics_solver._metric_block(truth, truth.copy(),
                                        np.ones(3, dtype=bool))
    scaled = fenics_solver._metric_block(truth, 2.0 * truth,
                                         np.ones(3, dtype=bool))
    zero = fenics_solver._metric_block(np.zeros(3), np.zeros(3),
                                       np.ones(3, dtype=bool))
    ok = (exact["re"] == 0.0 and exact["mae"] == 0.0
          and abs(scaled["re"] - 1.0) < 1e-12
          and zero["re"] is None)
    return {"exact": exact, "scaled": scaled, "zero_denominator": zero,
            "ok": bool(ok)}


def operator_bug_checks() -> dict:
    """Independent analytic evidence for the old dot(u, grad(u)) error."""
    import math
    def u(x, y):
        return (x + 2.0 * y, 3.0 * x - y)
    # grad u = [[1, 2], [3, -1]] (non-symmetric, div u = 0)
    conv = lambda x, y: (7.0 * x, 7.0 * y)
    old = lambda x, y: (10.0 * x - y, -x + 5.0 * y)  # 0.5 grad |u|^2
    rows = []
    for x, y in ((0.2, -0.3), (-0.4, 0.5), (0.3, 0.1)):
        c = conv(x, y)
        o = old(x, y)
        rows.append({"point": [x, y], "correct": list(c), "old": list(o),
                     "difference": [c[0] - o[0], c[1] - o[1]]})
    max_diff = max(max(abs(r["difference"][0]), abs(r["difference"][1]))
                   for r in rows)
    return {"field": "u=(x+2y, 3x-y), grad(u)=[[1,2],[3,-1]]",
            "rows": rows, "max_difference": max_diff,
            "ok": bool(max_diff > 1e-6)}


def _run_manufactured_levels(entry, params, config, mode, levels):
    errors = []
    for target_h in levels:
        out = fenics_solver.solve_manufactured(
            entry, params, config, target_h, mode=mode)
        errors.append({"target_h": target_h,
                       "mesh": {key: out["mesh"][key] for key in (
                           "n_cells", "n_vertices", "target_h")},
                       "errors": out["errors"],
                       "newton_reason": out["newton_reason"],
                       "newton_iterations": out.get("newton_iterations")})
    orders = {}
    for key in errors[0]["errors"]:
        ratio = [errors[i]["errors"][key]
                 / max(errors[i + 1]["errors"][key], 1e-300)
                 for i in range(len(errors) - 1)]
        orders[key] = [math.log(v) / math.log(2.0) for v in ratio]
    return errors, orders


def dependency_checks(run_fem_probes: bool) -> dict:
    status = fenics_solver.dependency_status()
    result = {"dependency_status": status, "manufactured": None,
              "blocked": not status["all_present"]}
    if status["all_present"]:
        if not run_fem_probes:
            result["manufactured"] = {"status": "not_requested"}
            return result
        config = geometry.load_json(CONFIG_PATH)
        spec = geometry.load_geometry_spec(
            os.path.join(WORKSPACE, config["geometries"]["geometry_spec"]))
        entry = dict(geometry.geometry_entry(
            spec, int(config["geometries"]["pilot"])))
        entry["vertices"] = geometry.polygon_vertices(entry).tolist()
        params = config["physics"]
        levels = [0.20, 0.10, 0.05]
        errors, orders = _run_manufactured_levels(
            entry, params, config, "stokes", levels)
        b_errors, b_orders = _run_manufactured_levels(
            entry, params, config, "boussinesq", levels)
        nullspace = fenics_solver.pressure_nullspace_probe(entry, params, 0.20)
        thresholds = config["acceptance_thresholds"]["manufactured"]
        order_thresholds = {
            "velocity_l2": thresholds["stokes_velocity_l2_order_min"],
            "velocity_h1": thresholds["stokes_velocity_h1_order_min"],
            "pressure_l2": thresholds["stokes_pressure_l2_order_min"],
            "temperature_l2": thresholds["energy_temperature_l2_order_min"],
            "temperature_h1": 0.0,
        }
        def evaluate(errors, orders, thresholds):
            order_ok = all(len(orders.get(key, [])) > 0
                           and min(orders[key]) >= threshold
                           for key, threshold in thresholds.items())
            decrease = all(
                all(errors[i + 1]["errors"][key] < errors[i]["errors"][key]
                    for i in range(len(errors) - 1))
                for key in errors[0]["errors"])
            return bool(order_ok and decrease), order_ok, decrease
        accepted, order_ok, decrease = evaluate(errors, orders, order_thresholds)
        b_accepted, b_order_ok, b_decrease = evaluate(
            b_errors, b_orders, order_thresholds)
        result["manufactured"] = {
            "levels": errors, "observed_orders": orders,
            "order_thresholds": order_thresholds,
            "errors_decrease": decrease,
            "accepted": accepted,
            "status": "ran"}
        result["manufactured_boussinesq"] = {
            "levels": b_errors, "observed_orders": b_orders,
            "order_thresholds": order_thresholds,
            "errors_decrease": b_decrease,
            "accepted": b_accepted,
            "status": "ran"}
        result["pressure_nullspace"] = nullspace
        result["pressure_nullspace"]["accepted"] = bool(
            abs(nullspace["constant_pressure_rel_residual"])
            <= config["acceptance_thresholds"]["pressure"][
                "constant_mode_null_residual_rel"])
        result["pressure_nullspace"]["applicable"] = bool(
            float(params.get("pressure_gauge_eps", 0.0)) == 0.0)
        result["pressure_nullspace"]["gauge_mode"] = (
            "mass_penalty_zero_mean"
            if float(params.get("pressure_gauge_eps", 0.0)) > 0.0
            else "fixed_reference_dof")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=os.path.join(OUT_DIR,
                                                         "probe_report.json"))
    parser.add_argument("--run-fem-probes", action="store_true",
                        help="run manufactured probes when the FEM env exists")
    args = parser.parse_args(argv)
    config = geometry.load_json(CONFIG_PATH)
    geometry.load_geometry_spec(os.path.join(WORKSPACE,
                                             config["geometries"]["geometry_spec"]))
    checks = []
    protocol = protocol_checks(config)
    _check(checks, "Ra_identity", protocol["Ra_relative_error"] < 1e-12, protocol)
    _check(checks, "Pr_identity", abs(protocol["Pr"] - 1.0) < 1e-12, protocol)
    _check(checks, "mu_value", protocol["mu_relative_error"] < 1e-12, protocol)
    rows = geometry_checks(config)
    thresholds = config["acceptance_thresholds"]["geometry"]
    for row in rows:
        _check(checks, f"geometry_residuals_geom{row['geom']}",
               (row["outer_hausdorff_like_max"] <= thresholds["outer_hausdorff_max"]
                and row["inner_point_to_polygon_rms"] <= thresholds["inner_point_to_polygon_rms_max"]),
               row)
        _check(checks, f"boundary_counts_geom{row['geom']}",
               (row["outer_count"] == thresholds["boundary_counts"]["outer"]
                and row["inner_count"] == thresholds["boundary_counts"]["inner"]),
               row)
        _check(checks, f"boundary_values_geom{row['geom']}",
               (row["wall_velocity_max"] < 1e-9
                and abs(row["inner_temperature_min"] - 1.0) < 1e-9
                and abs(row["inner_temperature_max"] - 1.0) < 1e-9
                and abs(row["outer_temperature_min"]) < 1e-9
                and abs(row["outer_temperature_max"]) < 1e-9),
               row)
        _check(checks, f"representative_in_test_geom{row['geom']}",
               row["in_protocol_A_test"], row)
        _check(checks, f"mesh_boundary_conformity_geom{row['geom']}",
               (row["mesh_boundary"]["inner_max_distance_to_polygon"] < 1e-9
                and row["mesh_boundary"]["outer_max_distance_to_square"] < 1e-9),
               row["mesh_boundary"])
        _check(checks, f"independent_polygon_fit_geom{row['geom']}",
               (row["independent_polygon_fit"]["rms"] < 1e-9
                and row["fit_radius_abs_error"] < 1e-9
                and row["fit_side_abs_error"] < 1e-9
                and row["fit_phase_abs_error_deg"] < 1e-9),
               row["independent_polygon_fit"])
    checks.extend(mesh_entity_checks(rows, config))
    representative = representative_id_checks(config)
    _check(checks, "representative_minimal_test_ids", representative["ok"],
           representative)
    formula = manufactured_formula_checks(config)
    _check(checks, "manufactured_source_formulas_fd", formula["ok"], formula)
    acceptance = acceptance_logic_checks(config)
    _check(checks, "acceptance_gate_rejects_finite_only", acceptance["ok"],
           acceptance)
    metrics = comparison_metric_checks()
    _check(checks, "comparison_metric_helpers", metrics["ok"], metrics)
    snapshot = protocol_snapshot_check(config)
    _check(checks, "protocol_snapshot_current", snapshot["ok"], snapshot)
    _check(checks, "pressure_gauge_declared",
           bool((config["physics"].get("pressure") or {}).get("gauge")),
           config["physics"].get("pressure"))
    fem = dependency_checks(args.run_fem_probes)
    bug = operator_bug_checks()
    _check(checks, "operator_bug_analytic_evidence",
           bug["ok"], bug)
    if fem.get("manufactured") and fem["manufactured"].get("status") == "ran":
        _check(checks, "manufactured_stokes_convergence",
               fem["manufactured"].get("accepted") is True,
               fem["manufactured"].get("observed_orders"))
        _check(checks, "manufactured_boussinesq_convergence",
               fem.get("manufactured_boussinesq", {}).get("accepted") is True,
               fem.get("manufactured_boussinesq", {}).get("observed_orders"))
        nullspace = fem.get("pressure_nullspace", {})
        if nullspace.get("applicable", True):
            _check(checks, "pressure_nullspace",
                   nullspace.get("accepted") is True, nullspace)
        else:
            _check(checks, "pressure_gauge_regularization",
                   float(config["physics"].get("pressure_gauge_eps", 0.0)) > 0.0,
                   nullspace)
    status = "geometry_protocol_passed"
    if fem["blocked"]:
        status = "geometry_protocol_passed_fem_environment_blocked"
    elif fem["manufactured"] is not None and fem["manufactured"].get("status") == "ran":
        status = "geometry_protocol_and_manufactured_probes_ran"
    report = {
        "protocol": config["protocol_id"],
        "protocol_version": config["protocol_version"],
        "module": config["module"],
        "status": status,
        "created_utc": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ",
                                                   __import__("time").gmtime()),
        "config_sha256": sha256_file(CONFIG_PATH),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "checks": checks,
        "all_local_checks_passed": all(item["ok"] for item in checks),
        "geometry": rows,
        "representative_id_check": representative,
        "manufactured_formula_check": formula,
        "acceptance_logic_check": acceptance,
        "comparison_metric_check": metrics,
        "protocol_snapshot_check": snapshot,
        "protocol": protocol,
        "fem_environment": fem,
        "not_run": [
            "formal natural-convection FEM pilot",
            "three-level natural-convection convergence (h=0.08/0.04/0.02)",
            "SLURM submission (not authorised in this phase)",
            "model prediction on new FEM support",
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    # short Markdown companion
    md_path = os.path.join(os.path.dirname(os.path.abspath(args.output)),
                           "probe_report.md")
    lines = [
        "# R2-THERMAL-FEM-02 probe report",
        "",
        f"Protocol: {config['protocol_id']} ({config['protocol_version']})",
        f"Status: **{status}**",
        f"Local checks passed: {report['all_local_checks_passed']}",
        "",
        "## Geometry / boundary probes",
    ]
    for row in rows:
        lines.append(
            f"- geom {row['geom']} (n={row['n']}): outer={row['outer_count']}, "
            f"inner={row['inner_count']}, inner RMS={row['inner_point_to_polygon_rms']:.3e}, "
            f"wall |u|max={row['wall_velocity_max']:.3e}, "
            f"mesh h={row['mesh']['h_max']:.3f}, cells={row['mesh']['n_triangles']}, "
            f"min angle={row['mesh']['min_triangle_angle_deg']:.2f} deg")
    lines += [
        "",
        "## Representative IDs / manufactured formula checks",
        "",
        f"- minimal test IDs by class: {representative['min_id_by_sides']} "
        f"(configured {representative['configured_representatives']})",
        f"- manufactured source formula max abs error: "
        f"{formula['max_source_abs_error']:.3e}, "
        f"max |div u|: {formula['max_divergence_abs']:.3e}",
        "",
        "## FEM environment",
        "",
        f"- all FEM packages present: {fem['dependency_status']['all_present']}",
        f"- missing: {[name for name in fenics_solver.FEM_PACKAGES if not fem['dependency_status'][name]]}",
        f"- isolation command: `{fem['dependency_status']['isolation_command']}`",
        "",
        "## Not run",
    ]
    lines += [f"- {item}" for item in report["not_run"]]
    lines.append("")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(json.dumps({"status": status,
                      "all_local_checks_passed": report["all_local_checks_passed"],
                      "fem_blocked": fem["blocked"],
                      "output": os.path.abspath(args.output)}, indent=2))
    return 0 if report["all_local_checks_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
