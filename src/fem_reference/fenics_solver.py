"""FEniCSx backend for R2-THERMAL-FEM-02 (review implementation).

The laboratory environment currently has no FEniCS/FEM installation (verified
2026-10-22); this module therefore imports lazily and raises an explicit
``FemEnvironmentError`` listing the exact missing packages.  It implements the
steady incompressible Boussinesq problem with Taylor-Hood P2/P1 velocity-pressure
and P2 temperature, a real nonlinear Newton solve, weak-residual diagnostics and
manufactured-solution verification.  It does not interpolate stored labels and
does not replace FEM with PINN or finite differences.

Nothing here writes results outside ``results/round2_thermal_fem/``.
"""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

FEM_PACKAGES = ("dolfinx", "ufl", "basix", "petsc4py", "mpi4py", "gmsh")
ISOLATION_ENV = "pinnap-fem"
ISOLATION_COMMAND = (
    "conda create -n pinnap-fem -c conda-forge "
    "fenics-dolfinx=0.8.0 python-gmsh mpi4py petsc4py pyvista "
    "numpy scipy  # requires network and explicit installation authorisation"
)


class FemEnvironmentError(RuntimeError):
    """Raised when the review FEM environment is not installed."""


def dependency_status() -> dict:
    """Report which FEM packages are importable (no install, no side effects)."""
    import importlib.util

    status = {}
    for name in FEM_PACKAGES:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            spec = None
        status[name] = spec is not None
    status["isolated_environment"] = ISOLATION_ENV
    status["isolation_command"] = ISOLATION_COMMAND
    status["all_present"] = all(status[name] for name in FEM_PACKAGES)
    return status


def _import_fenics():
    status = dependency_status()
    missing = [name for name in FEM_PACKAGES if not status[name]]
    if missing:
        raise FemEnvironmentError(
            "FEniCSx/gmsh environment is incomplete; missing packages: "
            + ", ".join(missing)
            + ".  Suggested isolated environment: " + ISOLATION_COMMAND)
    from mpi4py import MPI
    from dolfinx import fem, mesh as dmesh, default_scalar_type
    from dolfinx.io import gmshio
    from dolfinx.fem.petsc import NonlinearProblem
    from dolfinx.nls.petsc import NewtonSolver
    from basix.ufl import element, mixed_element
    import ufl
    import gmsh

    return {
        "MPI": MPI, "fem": fem, "mesh": dmesh,
        "default_scalar_type": default_scalar_type, "gmshio": gmshio,
        "NonlinearProblem": NonlinearProblem, "NewtonSolver": NewtonSolver,
        "element": element, "mixed_element": mixed_element, "ufl": ufl,
        "gmsh": gmsh,
    }


def _polygon_boundary_distance(x: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Distance to polygon edges for vectorised points (2,n)."""
    points = x.T
    if points.shape[1] > 2:
        points = points[:, :2]
    starts = vertices[None, :, :]
    ends = np.roll(vertices, -1, axis=0)[None, :, :]
    direction = ends - starts
    denom = np.sum(direction * direction, axis=2)
    denom = np.where(denom > 0, denom, 1.0)
    rel = points[:, None, :] - starts
    t = np.clip(np.sum(rel * direction, axis=2) / denom, 0.0, 1.0)
    closest = starts + t[..., None] * direction
    return np.linalg.norm(points[:, None, :] - closest, axis=2).min(axis=1)


def build_gmsh_mesh(entry: dict, target_h: float,
                    verbose: bool = False) -> dict:
    """Build an independent gmsh mesh of square-minus-polygon via boolean cut."""
    f = _import_fenics()
    gmsh = f["gmsh"]
    gmsh.initialize()
    try:
        if verbose:
            gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add("r2_thermal_fem")
        occ = gmsh.model.occ
        square = occ.addRectangle(-1.0, -1.0, 0.0, 2.0, 2.0)
        vertices = entry["vertices"]
        points = [occ.addPoint(float(x), float(y), 0.0, target_h)
                  for x, y in vertices]
        lines = [occ.addLine(points[i], points[(i + 1) % len(points)])
                 for i in range(len(points))]
        loop = occ.addCurveLoop(lines)
        hole = occ.addPlaneSurface([loop])
        occ.synchronize()
        cut = occ.cut([(2, square)], [(2, hole)])
        occ.synchronize()
        if not cut:
            raise RuntimeError("gmsh boolean cut returned no surface")
        surface = cut[0][0][1]
        gmsh.model.addPhysicalGroup(2, [surface], 1)
        boundaries = gmsh.model.getBoundary([(2, surface)], oriented=False)
        outer_curves = []
        inner_curves = []
        for dim, tag in boundaries:
            centre = gmsh.model.occ.getCenterOfMass(dim, tag)
            radius = max(abs(centre[0]), abs(centre[1]))
            (outer_curves if radius > 0.99 else inner_curves).append(tag)
        if not outer_curves or not inner_curves:
            raise RuntimeError("could not classify gmsh boundary curves")
        gmsh.model.addPhysicalGroup(1, outer_curves, 2)
        gmsh.model.addPhysicalGroup(1, inner_curves, 3)
        gmsh.option.setNumber("Mesh.MeshSizeMin", target_h)
        gmsh.option.setNumber("Mesh.MeshSizeMax", target_h)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(1)
        mesh_data = f["gmshio"].model_to_mesh(gmsh.model, f["MPI"].COMM_WORLD,
                                              0, gdim=2)
        domain, cell_tags, facet_tags = mesh_data
        try:
            actual_h = float(f["mesh"].hmax(domain))
        except Exception:  # noqa: BLE001 - hmax availability is version-dependent
            actual_h = None
        return {"mesh": domain, "cell_tags": cell_tags,
                "facet_tags": facet_tags,
                "n_cells": domain.topology.index_map(2).size_local,
                "n_vertices": domain.topology.index_map(0).size_local,
                "target_h": float(target_h),
                "h_max": actual_h}
    finally:
        gmsh.finalize()


def _spaces(mesh, element_namespace, mixed_element, fem_namespace):
    P2 = element_namespace("Lagrange", mesh.basix_cell(), 2, shape=(2,))
    P1 = element_namespace("Lagrange", mesh.basix_cell(), 1)
    P2t = element_namespace("Lagrange", mesh.basix_cell(), 2)
    W = fem_namespace.functionspace(mesh, mixed_element([P2, P1, P2t]))
    return W


def _boundary_conditions(mesh, W, entry: dict, fem, ufl,
                          exact_stokes_source=None, exact_energy_source=None):
    """Velocity no-slip on all walls; inner T=1, outer T=0; pressure pin.

    Boundary conditions on mixed subspaces follow the FEniCSx 0.8 demo pattern:
    collapse each subspace, locate its dofs through ``(W.sub(i), V_collapsed)``,
    and attach the BC to the mixed subspace ``W.sub(i)``.
    """
    V0, Q0, T0 = W.sub(0), W.sub(1), W.sub(2)
    Vu, _ = V0.collapse()
    Qc, _ = Q0.collapse()
    Tc, _ = T0.collapse()
    vertices = np.asarray(entry["vertices"], dtype=float)

    def on_wall(x):
        outer = (np.isclose(np.abs(x[0]), 1.0, atol=1e-10)
                 | np.isclose(np.abs(x[1]), 1.0, atol=1e-10))
        inner = _polygon_boundary_distance(x, vertices) < 2.0e-9
        return outer | inner

    def on_inner(x):
        return _polygon_boundary_distance(x, vertices) < 2.0e-9

    def on_outer(x):
        return ((np.isclose(np.abs(x[0]), 1.0, atol=1e-10)
                 | np.isclose(np.abs(x[1]), 1.0, atol=1e-10))
                & ~on_inner(x))

    u_bc = fem.Function(Vu)
    wall_raw = np.asarray(
        fem.locate_dofs_geometrical((V0, Vu), on_wall), dtype=np.int32)
    wall_dofs = [wall_raw[0], wall_raw[1]]
    if exact_stokes_source is None:
        u_bc.interpolate(lambda x: np.vstack([np.zeros_like(x[0]),
                                              np.zeros_like(x[0])]))
    else:
        u_bc.interpolate(exact_stokes_source)
    bcs = [fem.dirichletbc(u_bc, wall_dofs, V0)]
    if exact_energy_source is None:
        outer_vals = fem.Function(Tc)
        outer_vals.x.array[:] = 0.0
        inner_vals = fem.Function(Tc)
        inner_vals.x.array[:] = 1.0
        bcs.append(fem.dirichletbc(
            outer_vals,
            list(np.asarray(fem.locate_dofs_geometrical((T0, Tc), on_outer),
                            dtype=np.int32)), T0))
        bcs.append(fem.dirichletbc(
            inner_vals,
            list(np.asarray(fem.locate_dofs_geometrical((T0, Tc), on_inner),
                            dtype=np.int32)), T0))
    else:
        t_bc = fem.Function(Tc)
        t_bc.interpolate(exact_energy_source)
        bcs.append(fem.dirichletbc(
            t_bc, list(np.asarray(
                fem.locate_dofs_geometrical((T0, Tc), on_wall),
                dtype=np.int32)), T0))
    # Pressure gauge is handled by a small mass regularization in _residual
    # (params["pressure_gauge_eps"]) for this environment; output pressure is
    # still post-processed with an explicit mean/centering convention.
    return bcs, on_wall, on_inner, on_outer


def _convective_components(u, ufl):
    """Correct component-wise (u·∇)u for 2D/3D vector u.

    UFL ``grad(u)[i, j] = ∂u_i/∂x_j``; ``dot(u, grad(u))`` is therefore
    ``0.5 grad(|u|^2)``, not convection.  The explicit components below are
    ``Σ_j u_j ∂u_i/∂x_j`` and are also used for the weak residual.
    """
    dim = u.ufl_shape[0]
    return ufl.as_vector([
        ufl.dot(u, ufl.grad(u[i])) for i in range(dim)
    ])


def _residual(mesh, w, W, params, ufl, source_velocity=None,
              source_temperature=None, linear_stokes: bool = False):
    u, p, T = ufl.split(w)
    v, q, s = ufl.TestFunctions(W)
    mu = params["mu"]
    rho = params["rho"]
    cp = params["cp"]
    kappa = params["kappa"]
    g_beta = params["g_beta"]
    gauge_eps = float(params.get("pressure_gauge_eps", 0.0))
    F = (mu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + q * ufl.div(u) * ufl.dx
         + kappa * ufl.dot(ufl.grad(T), ufl.grad(s)) * ufl.dx)
    if gauge_eps > 0.0:
        F += gauge_eps * p * q * ufl.dx
    if not linear_stokes:
        F += (rho * ufl.dot(_convective_components(u, ufl), v) * ufl.dx
              - rho * g_beta * T * v[1] * ufl.dx
              + rho * cp * ufl.dot(u, ufl.grad(T)) * s * ufl.dx)
    if source_velocity is not None:
        F = F - ufl.dot(source_velocity, v) * ufl.dx
    if source_temperature is not None:
        F = F - source_temperature * s * ufl.dx
    return F


def _solve_newton(w, bcs, F, cfg, nonlinear_problem_cls, newton_cls, MPI):
    problem = nonlinear_problem_cls(F, w, bcs=bcs)
    solver = newton_cls(MPI.COMM_WORLD, problem)
    solver.rtol = float(cfg.get("convergence", {}).get("update_rel_tolerance", 1e-8))
    solver.atol = float(cfg.get("convergence", {}).get("absolute_update_tolerance",
                                                      1e-12))
    solver.max_it = int(cfg.get("max_iterations", 50))
    solver.convergence_criterion = "incremental"
    solver.report = bool(cfg.get("report", False))
    try:
        result = solver.solve(w)
    except RuntimeError:
        raise
    if isinstance(result, tuple):
        iterations, converged = result[0], bool(result[1])
    else:
        iterations, converged = result, True
    return (2 if converged else -1), (int(iterations)
                                     if iterations is not None else None)


def _assemble_scalar(form, fem):
    return float(fem.assemble_scalar(form))


def evaluate_at_points(result: dict, points: np.ndarray) -> dict:
    """Evaluate collapsed velocity/pressure/temperature at arbitrary points.

    Uses dolfinx geometry collision detection to select one containing cell per
    point.  Points outside the mesh are flagged and returned as NaN.
    """
    f = _import_fenics()
    mesh = result["mesh"]["mesh"]
    solution = result["solution"]
    from dolfinx.geometry import (bb_tree, compute_colliding_cells,
                                  compute_collisions_points)
    points = np.asarray(points, dtype=float)
    # dolfinx 0.8 bounding-box queries require 3D coordinates even for 2D
    # meshes; pad the planar points with z=0 before collision/eval.
    gdim = 3
    if points.shape[1] < gdim:
        points = np.pad(points, ((0, 0), (0, gdim - points.shape[1])),
                        mode="constant")
    elif points.shape[1] > gdim:
        points = points[:, :gdim]
    tree = bb_tree(mesh, mesh.topology.dim)
    candidates = compute_collisions_points(tree, points)
    colliding = compute_colliding_cells(mesh, candidates, points)
    cells = np.full(len(points), -1, dtype=np.int32)
    for index in range(len(points)):
        row = colliding.links(index)
        row = row[row >= 0]
        if row.size:
            cells[index] = int(row[0])
    valid = cells >= 0
    velocity = solution.sub(0).collapse()
    pressure = solution.sub(1).collapse()
    temperature = solution.sub(2).collapse()
    uv = np.full((len(points), 2), np.nan)
    pv = np.full(len(points), np.nan)
    tv = np.full(len(points), np.nan)
    if np.any(valid):
        uv[valid] = np.asarray(velocity.eval(points[valid], cells[valid]))
        pv[valid] = np.asarray(pressure.eval(points[valid], cells[valid])).reshape(-1)
        tv[valid] = np.asarray(temperature.eval(points[valid], cells[valid])).reshape(-1)
    return {"points": points, "valid": valid, "cells": cells,
            "u": uv[:, 0], "v": uv[:, 1], "p": pv, "T": tv}


def compare_with_reference(result: dict, reference_points: np.ndarray,
                           reference_values: np.ndarray,
                           boundary_count: int | None = None) -> dict:
    """Source-consistency diagnostics against stored labels at original nodes.

    Reports all-node and (when ``boundary_count`` is given) boundary-layer metrics.
    This is a diagnostic only: it does not validate the historical labels and it
    does not replace the independent FEM convergence assessment.
    """
    pred = evaluate_at_points(result, reference_points)
    fields = {"u": 0, "v": 1, "p": 2, "T": 3}
    valid = pred["valid"]
    predictions = {"u": pred["u"], "v": pred["v"],
                   "p": pred["p"], "T": pred["T"]}

    def blocks(mask: np.ndarray) -> dict:
        metrics = {}
        for name, column in fields.items():
            truth = np.asarray(reference_values[:, column], dtype=float)
            prediction = predictions[name]
            if name == "p":
                truth_centered = truth - np.nanmean(truth[mask])
                prediction_centered = prediction - np.nanmean(prediction[mask])
                metrics["p_raw"] = _metric_block(truth, prediction, mask)
                metrics["p_centered"] = _metric_block(
                    truth_centered, prediction_centered, mask)
            else:
                metrics[name] = _metric_block(truth, prediction, mask)
        return metrics

    result = {"n_points": int(len(reference_points)),
              "n_evaluated": int(np.sum(valid)),
              "n_outside_mesh": int(np.sum(~valid)),
              "metrics": blocks(valid),
              "pressure_note": "raw and equal-weight point-mean-centered pressure "
                               "are reported separately; this is not an FE area "
                               "gauge normalization"}
    if boundary_count is not None:
        boundary_mask = np.zeros(len(reference_points), dtype=bool)
        boundary_mask[:min(int(boundary_count), len(boundary_mask))] = True
        boundary_mask &= valid
        result["boundary_layer"] = {
            "n_points": int(np.sum(boundary_mask)),
            "first_n_original_nodes": int(boundary_count),
            "metrics": blocks(boundary_mask),
        }
    return result


def _metric_block(truth: np.ndarray, prediction: np.ndarray,
                  valid: np.ndarray) -> dict:
    truth = np.asarray(truth, dtype=float)[valid]
    prediction = np.asarray(prediction, dtype=float)[valid]
    denominator = float(np.linalg.norm(truth))
    error = prediction - truth
    return {
        "n": int(len(truth)),
        "re": (float(np.linalg.norm(error) / denominator)
               if denominator > 0 else None),
        "mae": float(np.mean(np.abs(error))) if len(error) else None,
        "max_err": float(np.max(np.abs(error))) if len(error) else None,
        "denominator": denominator,
    }


def _solve_picard(mesh, w, W, params, bcs, cfg, fem, ufl,
                   linear_problem_cls, MPI) -> tuple[int, int]:
    """Picard (Oseen) iterations with under-relaxation for steady Boussinesq.

    This is the fallback/primary nonlinear path for the natural-convection run;
    the manufactured verification uses a linear problem and therefore Newton.
    Returns (reason, iterations), reason=2 on convergence, -1 otherwise.
    """
    u_old, p_old, T_old = ufl.split(w)
    u_trial, p_trial, T_trial = ufl.TrialFunctions(W)
    v, q, s = ufl.TestFunctions(W)
    mu = params["mu"]
    rho = params["rho"]
    cp = params["cp"]
    kappa = params["kappa"]
    g_beta = params["g_beta"]
    gauge_eps = float(params.get("pressure_gauge_eps", 0.0))
    dx = ufl.dx(domain=mesh)
    convection_trial = ufl.as_vector([
        ufl.dot(u_old, ufl.grad(u_trial[i])) for i in range(2)
    ])
    a = (rho * ufl.dot(convection_trial, v) * dx
         + mu * ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * dx
         - p_trial * ufl.div(v) * dx
         - rho * g_beta * T_trial * v[1] * dx
         + q * ufl.div(u_trial) * dx
         + rho * cp * ufl.dot(u_old, ufl.grad(T_trial)) * s * dx
         + kappa * ufl.dot(ufl.grad(T_trial), ufl.grad(s)) * dx)
    if gauge_eps > 0.0:
        a += gauge_eps * p_trial * q * dx
    zero = fem.Constant(mesh, (0.0, 0.0))
    L = (ufl.dot(zero, v) * dx
         + ufl.as_ufl(0.0) * q * dx
         + ufl.as_ufl(0.0) * s * dx)
    problem = linear_problem_cls(a, L, bcs=bcs)
    max_it = int(cfg.get("max_iterations", 100))
    relax = float(cfg.get("relaxation", 0.7))
    tol = float(cfg.get("convergence", {}).get(
        "update_rel_tolerance", 1e-8))
    for iteration in range(1, max_it + 1):
        w_new = problem.solve()
        old = w.x.array.copy()
        trial = w_new.x.array
        diff = trial - old
        scale = max(float(np.linalg.norm(old)), 1e-12)
        relative = float(np.linalg.norm(diff)) / scale
        w.x.array[:] = old + relax * diff
        w.x.scatter_forward()
        if relative < tol:
            return 2, iteration
    return -1, max_it


def _mixed_dirichlet_dofs(W, fem, on_wall, on_inner, on_outer) -> np.ndarray:
    """Mixed-space dof indices for all velocity and temperature Dirichlet BCs."""
    V0 = W.sub(0)
    Vu, _ = V0.collapse()
    T0 = W.sub(2)
    Tc, _ = T0.collapse()
    dofs = []
    for space, collapsed, marker in ((V0, Vu, on_wall),
                                     (T0, Tc, on_inner),
                                     (T0, Tc, on_outer)):
        raw = np.asarray(fem.locate_dofs_geometrical(
            (space, collapsed), marker), dtype=np.int32)
        dofs.append(raw[0])
    return np.unique(np.concatenate(dofs)).astype(np.int32)


def weak_residual_norm(mesh, W, w, params, fem, ufl,
                       on_wall, on_inner, on_outer,
                       bcs=None) -> dict:
    """Variational residual at the FE solution with all Dirichlet rows zeroed."""
    from dolfinx.fem.petsc import assemble_vector
    from petsc4py import PETSc

    u, p, T = ufl.split(w)
    v, q, s = ufl.TestFunctions(W)
    rho = params["rho"]
    cp = params["cp"]
    mu = params["mu"]
    kappa = params["kappa"]
    g_beta = params["g_beta"]
    dx = ufl.dx(domain=mesh)
    conv = _convective_components(u, ufl)
    mom = (rho * conv - mu * ufl.div(ufl.grad(u))
           + ufl.grad(p) - ufl.as_vector([0.0, rho * g_beta * T]))
    energy = rho * cp * ufl.dot(u, ufl.grad(T)) - kappa * ufl.div(ufl.grad(T))
    forms = {
        "full": _residual(mesh, w, W, params, ufl),
        "continuity": q * ufl.div(u) * dx,
        "momentum": ufl.dot(mom, v) * dx,
        "energy": energy * s * dx,
    }
    dofs = _mixed_dirichlet_dofs(W, fem, on_wall, on_inner, on_outer)
    out = {"n_dirichlet_dofs": int(len(dofs)), "blocks": {}}
    vector = None
    scales = {
        "velocity": math.sqrt(max(_assemble_scalar(
            fem.form(ufl.dot(u, u) * dx), fem), 0.0)),
        "temperature": math.sqrt(max(_assemble_scalar(
            fem.form(T * T * dx), fem), 0.0)),
    }
    for name, form in forms.items():
        vector = assemble_vector(fem.form(form))
        vector.ghostUpdate(addv=PETSc.InsertMode.ADD,
                           mode=PETSc.ScatterMode.REVERSE)
        vector.assemble()
        full = float(vector.norm())
        arr = vector.array
        for idx in dofs:
            if 0 <= int(idx) < len(arr):
                arr[int(idx)] = 0.0
        vector.assemble()
        free = float(vector.norm())
        block = {"full": full, "free": free}
        if name == "continuity":
            block["relative_free"] = free / max(scales["velocity"], 1e-30)
        elif name == "momentum":
            block["relative_free"] = free / max(scales["velocity"], 1e-30)
        elif name == "energy":
            block["relative_free"] = free / max(scales["temperature"], 1e-30)
        out["blocks"][name] = block
    out["dirichlet_zeroed"] = out["blocks"]["full"]["free"]
    free = out["blocks"]["full"]["free"]
    raw = out["blocks"]["full"]["full"]
    out["galerkin"] = {
        "free_absolute": free,
        "free_relative_to_raw": (free / raw) if raw > 0 else None,
        "definition": "assembled nonlinear Galerkin residual after zeroing all "
                      "velocity/temperature Dirichlet rows",
    }
    out["cell_strong_against_test_projection"] = {
        key: out["blocks"][key] for key in ("continuity", "momentum", "energy")
    }
    return out


def diagnostics(mesh, W, w, params, bcs, fem, ufl,
                on_wall=None, on_inner=None, on_outer=None) -> dict:
    u, p, T = ufl.split(w)
    area = _assemble_scalar(fem.form(ufl.as_ufl(1.0) * ufl.dx(domain=mesh)), fem)
    div_l2 = math.sqrt(max(_assemble_scalar(
        fem.form(ufl.div(u) * ufl.div(u) * ufl.dx), fem), 0.0))
    mass_flux = _assemble_scalar(fem.form(ufl.div(u) * ufl.dx), fem)
    p_mean = _assemble_scalar(fem.form(p * ufl.dx), fem) / area
    p_var = _assemble_scalar(fem.form((p - p_mean) * (p - p_mean) * ufl.dx),
                             fem) / area
    u_l2 = math.sqrt(max(_assemble_scalar(
        fem.form(ufl.dot(u, u) * ufl.dx), fem), 0.0))
    t_l2 = math.sqrt(max(_assemble_scalar(
        fem.form(T * T * ufl.dx), fem), 0.0))
    rho = params["rho"]
    mu = params["mu"]
    cp = params["cp"]
    kappa = params["kappa"]
    g_beta = params["g_beta"]
    lap_u = ufl.div(ufl.grad(u))
    rx = rho * ufl.dot(u, ufl.grad(u[0])) + p.dx(0) - mu * lap_u[0]
    ry = (rho * ufl.dot(u, ufl.grad(u[1])) + p.dx(1) - mu * lap_u[1]
          - rho * g_beta * T)
    re = (rho * cp * ufl.dot(u, ufl.grad(T))
          - kappa * ufl.div(ufl.grad(T)))
    residual = {
        "continuity_l2": div_l2,
        "momentum_l2": math.sqrt(max(_assemble_scalar(
            fem.form((rx * rx + ry * ry) * ufl.dx), fem), 0.0)),
        "energy_l2": math.sqrt(max(_assemble_scalar(
            fem.form(re * re * ufl.dx), fem), 0.0)),
    }
    u_array = w.sub(0).collapse().x.array
    t_array = w.sub(2).collapse().x.array
    finite = bool(np.all(np.isfinite(u_array)) and np.all(np.isfinite(t_array)))
    boundary = {}
    if on_wall is not None:
        Vu, _ = W.sub(0).collapse()
        Vt, _ = W.sub(2).collapse()
        u_collapsed = w.sub(0).collapse()
        t_collapsed = w.sub(2).collapse()
        wall_raw = np.asarray(
            fem.locate_dofs_geometrical((W.sub(0), Vu), on_wall),
            dtype=np.int32)
        wall_collapsed = wall_raw[1]
        boundary["wall_velocity_max"] = float(np.max(np.abs(
            u_collapsed.x.array[wall_collapsed]))) if len(wall_collapsed) else 0.0
        inner_raw = np.asarray(
            fem.locate_dofs_geometrical((W.sub(2), Vt), on_inner),
            dtype=np.int32)
        outer_raw = np.asarray(
            fem.locate_dofs_geometrical((W.sub(2), Vt), on_outer),
            dtype=np.int32)
        inner_collapsed = inner_raw[1]
        outer_collapsed = outer_raw[1]
        boundary["inner_temperature_max_dev"] = float(np.max(np.abs(
            t_collapsed.x.array[inner_collapsed] - 1.0))) if len(inner_collapsed) else 0.0
        boundary["outer_temperature_max_dev"] = float(np.max(np.abs(
            t_collapsed.x.array[outer_collapsed]))) if len(outer_collapsed) else 0.0
    return {
        "area": area,
        "divergence_l2": div_l2,
        "mass_flux": mass_flux,
        "pressure_mean": p_mean,
        "pressure_std": math.sqrt(max(p_var, 0.0)),
        "velocity_l2": u_l2,
        "temperature_l2": t_l2,
        "residual": residual,
        "boundary": boundary,
        "finite": finite,
    }


def solve_natural_convection(entry: dict, params: dict, cfg: dict,
                             target_h: float) -> dict:
    """Solve steady Boussinesq with Ra continuation (Newton, Picard fallback)."""
    f = _import_fenics()
    fem = f["fem"]
    ufl = f["ufl"]
    mesh_data = build_gmsh_mesh(entry, target_h)
    mesh = mesh_data["mesh"]
    W = _spaces(mesh, f["element"], f["mixed_element"], fem)
    bcs, on_wall, on_inner, on_outer = _boundary_conditions(
        mesh, W, entry, fem, ufl)
    w = fem.Function(W)
    solver_cfg = cfg.get("solver", cfg)
    stages = solver_cfg.get("continuation_gbeta", [0.01, 0.1, 0.5, 1.0])
    history = []
    for fraction in stages:
        stage_params = dict(params)
        stage_params["g_beta"] = float(params["g_beta"]) * float(fraction)
        stage_reason = None
        try:
            F = _residual(mesh, w, W, stage_params, ufl)
            stage_reason, stage_iterations = _solve_newton(
                w, bcs, F, solver_cfg, f["NonlinearProblem"],
                f["NewtonSolver"], f["MPI"])
        except RuntimeError:
            stage_reason, stage_iterations = -1, None
        history.append({"g_beta_fraction": float(fraction),
                        "reason": int(stage_reason),
                        "iterations": (int(stage_iterations)
                                       if stage_iterations is not None else None)})
        print("CONT_STAGE", fraction, stage_reason, stage_iterations, flush=True)
        if int(stage_reason) < 0:
            break
    final_reason = history[-1]["reason"] if history else -1
    final_iterations = history[-1]["iterations"] if history else None
    return {"mesh": mesh_data, "solution": w, "newton_reason": final_reason,
            "newton_iterations": final_iterations,
            "continuation": history,
            "weak_residual": weak_residual_norm(
                mesh, W, w, params, fem, ufl, on_wall, on_inner, on_outer),
            "diagnostics": diagnostics(mesh, W, w, params, bcs, fem, ufl,
                                       on_wall, on_inner, on_outer)}


def manufactured_fields(mesh, params: dict, ufl, nonlinear: bool = False):
    """Exact fields and the full Boussinesq source terms that they satisfy.

    The source includes convection and buoyancy in the momentum block, so the
    manufactured test exercises the same nonlinear weak form used by the natural-
    convection solver (rho=cp=g_beta=1 in this project).
    """
    x = ufl.SpatialCoordinate(mesh)
    pi = ufl.pi
    rho = params["rho"]
    g_beta = params["g_beta"]
    cp = params["cp"]
    mu = params["mu"]
    kappa = params["kappa"]
    u_exact = ufl.as_vector([
        ufl.sin(pi * x[0]) * ufl.cos(pi * x[1]),
        -ufl.cos(pi * x[0]) * ufl.sin(pi * x[1]),
    ])
    p_exact = ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    T_exact = ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    if nonlinear:
        buoyancy = ufl.as_vector([0.0, -rho * g_beta * T_exact])
        # Independent component expansion (not dot(u, grad(u))); this must
        # match the production weak operator and expose the old operator bug.
        conv_exact = ufl.as_vector([
            u_exact[0] * u_exact[0].dx(0) + u_exact[1] * u_exact[0].dx(1),
            u_exact[0] * u_exact[1].dx(0) + u_exact[1] * u_exact[1].dx(1),
        ])
        f_velocity = (rho * conv_exact
                      - mu * ufl.div(ufl.grad(u_exact))
                      + ufl.grad(p_exact) + buoyancy)
        f_temperature = (rho * cp * ufl.dot(u_exact, ufl.grad(T_exact))
                         - kappa * ufl.div(ufl.grad(T_exact)))
    else:
        f_velocity = (-mu * ufl.div(ufl.grad(u_exact))
                      + ufl.grad(p_exact))
        f_temperature = -kappa * ufl.div(ufl.grad(T_exact))
    return u_exact, p_exact, T_exact, f_velocity, f_temperature


def solve_manufactured(mesh_entry: dict, params: dict, cfg: dict,
                       target_h: float, mode: str = "stokes") -> dict:
    """Manufactured Stokes (with optional energy) verification on one mesh."""
    f = _import_fenics()
    fem = f["fem"]
    ufl = f["ufl"]
    mesh_data = build_gmsh_mesh(mesh_entry, target_h)
    mesh = mesh_data["mesh"]
    W = _spaces(mesh, f["element"], f["mixed_element"], fem)
    nonlinear = (mode == "boussinesq")
    u_exact, p_exact, T_exact, f_vel, f_T = manufactured_fields(
        mesh, params, ufl, nonlinear=nonlinear)
    # exact boundary expressions for the manufactured test
    def u_expr(x):
        pix = np.pi * x[0]
        piy = np.pi * x[1]
        return np.vstack([np.sin(pix) * np.cos(piy),
                          -np.cos(pix) * np.sin(piy)])

    def t_expr(x):
        return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])

    bcs, _, _, _ = _boundary_conditions(mesh, W, mesh_entry, fem, ufl,
                                        exact_stokes_source=u_expr,
                                        exact_energy_source=t_expr)
    w = fem.Function(W)
    F = _residual(mesh, w, W, params, ufl, source_velocity=f_vel,
                  source_temperature=f_T, linear_stokes=not nonlinear)
    reason, iterations = _solve_newton(
        w, bcs, F, cfg.get("solver", cfg), f["NonlinearProblem"],
        f["NewtonSolver"], f["MPI"])
    u, p, T = ufl.split(w)
    area = _assemble_scalar(fem.form(ufl.as_ufl(1.0) * ufl.dx(domain=mesh)), fem)
    u_err = fem.form(ufl.dot(u - u_exact, u - u_exact) * ufl.dx)
    gu_err = fem.form(ufl.inner(ufl.grad(u - u_exact),
                                ufl.grad(u - u_exact)) * ufl.dx)
    p_mean = _assemble_scalar(fem.form(p * ufl.dx), fem) / area
    pe_mean = _assemble_scalar(fem.form(p_exact * ufl.dx), fem) / area
    p_err = fem.form((p - p_mean - (p_exact - pe_mean)) ** 2 * ufl.dx)
    t_err = fem.form((T - T_exact) ** 2 * ufl.dx)
    gt_err = fem.form(ufl.dot(ufl.grad(T - T_exact),
                              ufl.grad(T - T_exact)) * ufl.dx)
    return {
        "mesh": mesh_data,
        "newton_reason": reason,
        "newton_iterations": iterations,
        "errors": {
            "velocity_l2": math.sqrt(max(_assemble_scalar(u_err, fem), 0.0)),
            "velocity_h1": math.sqrt(max(_assemble_scalar(gu_err, fem), 0.0)),
            "pressure_l2": math.sqrt(max(_assemble_scalar(p_err, fem), 0.0)),
            "temperature_l2": math.sqrt(max(_assemble_scalar(t_err, fem), 0.0)),
            "temperature_h1": math.sqrt(max(_assemble_scalar(gt_err, fem), 0.0)),
        },
        "pressure_mean": p_mean,
    }

def pressure_nullspace_probe(entry: dict, params: dict, target_h: float) -> dict:
    """Assemble the unconstrained Stokes operator and test the constant-pressure
    nullspace mode (the pressure pin is the production gauge choice)."""
    f = _import_fenics()
    fem = f["fem"]
    ufl = f["ufl"]
    mesh_data = build_gmsh_mesh(entry, target_h)
    mesh = mesh_data["mesh"]
    W = _spaces(mesh, f["element"], f["mixed_element"], fem)
    from dolfinx.fem.petsc import assemble_matrix
    from petsc4py import PETSc
    trial = ufl.TrialFunction(W)
    v, q, s = ufl.TestFunctions(W)
    u, p, T = ufl.split(trial)
    form = (params["mu"] * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
            - p * ufl.div(v) * ufl.dx
            + q * ufl.div(u) * ufl.dx
            + params["kappa"] * ufl.dot(ufl.grad(T), ufl.grad(s)) * ufl.dx)
    bcs, _, _, _ = _boundary_conditions(mesh, W, entry, fem, ufl)
    matrix = assemble_matrix(fem.form(form), bcs=bcs)
    matrix.assemble()
    vec = fem.Function(W)
    vec.sub(1).x.array[:] = 1.0
    out = vec.x.petsc_vec.duplicate()
    matrix.mult(vec.x.petsc_vec, out)
    residual = float(out.norm())
    scale = float(vec.x.petsc_vec.norm())
    return {"mesh": {key: mesh_data[key] for key in
                     ("n_cells", "n_vertices", "target_h")},
            "constant_pressure_null_residual": residual,
            "constant_pressure_rel_residual": residual / max(scale, 1e-300),
            "matrix_nnz": int(matrix.getInfo()["nz_used"])}

