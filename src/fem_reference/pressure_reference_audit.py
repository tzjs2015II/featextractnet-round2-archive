#!/usr/bin/env python3
"""R3.16 exact pressure gauge experiments (new owned module).

Reuses the accepted production Boussinesq weak form and mesh/BC builders but
adds a *pressure Dirichlet point* so the pressure constant nullspace is removed
exactly (no epsilon mass penalty).  The integral-zero-mean gauge is an exact
constant shift of the point-reference solution; velocity and temperature are
therefore unchanged.  The original solver is NOT modified.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any

import numpy as np

from . import fenics_solver as base


def _fenics():
    return base._import_fenics()


def _linear_problem():
    from dolfinx.fem.petsc import LinearProblem
    return LinearProblem


def _pin_dof(mesh, W, entry, fem):
    """Choose an interior pressure DOF far from both walls; return (dof, coord)."""
    Q0 = W.sub(1)
    Qc, _ = Q0.collapse()
    coords = np.asarray(Qc.tabulate_dof_coordinates(), dtype=float)[:, :2]
    vertices = np.asarray(entry["vertices"], dtype=float)
    inner_dist = base._polygon_boundary_distance(coords.T, vertices)
    outer_dist = np.minimum(1.0 - np.abs(coords[:, 0]),
                            1.0 - np.abs(coords[:, 1]))
    margin = np.minimum(inner_dist, outer_dist)
    idx = int(np.argmax(margin))
    x0, y0 = float(coords[idx, 0]), float(coords[idx, 1])
    # Use exact DOF coordinate lookup; fall back to collapsed index if needed.
    raw = np.asarray(fem.locate_dofs_geometrical(
        (Q0, Qc), lambda x: ((x[0] - x0) ** 2 + (x[1] - y0) ** 2) < 1e-24),
        dtype=np.int32)
    if raw.size:
        dofs_pair = list(np.asarray(raw, dtype=np.int32))
        collapsed_dof = int(raw[1, 0]) if raw.ndim == 2 and raw.shape[0] > 1 else int(raw[0])
    else:
        # Fallback: collapsed index is valid for the mixed subspace in dolfinx
        # when passed as a single-element pair.
        dofs_pair = [np.asarray([idx], dtype=np.int32)]
        collapsed_dof = int(idx)
    return dofs_pair, collapsed_dof, [x0, y0], float(margin[idx])


def _solve_newton_relaxed(w, bcs, F, cfg, nonlinear_problem_cls,
                          newton_cls, MPI, relaxation=0.5):
    problem = nonlinear_problem_cls(F, w, bcs=bcs)
    solver = newton_cls(MPI.COMM_WORLD, problem)
    solver.rtol = float(cfg.get("convergence", {}).get("update_rel_tolerance", 1e-8))
    solver.atol = float(cfg.get("convergence", {}).get("absolute_update_tolerance", 1e-12))
    solver.max_it = int(cfg.get("max_iterations", 50))
    solver.convergence_criterion = "incremental"
    solver.relaxation_parameter = float(relaxation)
    solver.report = bool(cfg.get("report", False))
    result = solver.solve(w)
    if isinstance(result, tuple):
        iterations, converged = result[0], bool(result[1])
    else:
        iterations, converged = result, True
    return (2 if converged else -1), (int(iterations) if iterations is not None else None)


def solve_point_reference(entry: dict, params: dict, cfg: dict,
                          target_h: float, initial_fields: dict | None = None) -> dict:
    """Solve the steady Boussinesq system with a pressure point Dirichlet BC.

    Exact gauge: p(x_pin)=0.  The residual has no epsilon mass term; the only
    pressure condition is the point pin, so this removes exactly the constant
    nullspace and does not alter momentum/energy/divergence physics.
    """
    f = _fenics()
    fem, ufl = f["fem"], f["ufl"]
    mesh_data = base.build_gmsh_mesh(entry, target_h)
    mesh = mesh_data["mesh"]
    W = base._spaces(mesh, f["element"], f["mixed_element"], fem)
    bcs, on_wall, on_inner, on_outer = base._boundary_conditions(
        mesh, W, entry, fem, ufl)
    pin_dofs, pin_dof, pin_coord, margin = _pin_dof(mesh, W, entry, fem)
    Q0 = W.sub(1)
    Qc, _ = Q0.collapse()
    pin_bc = fem.dirichletbc(fem.Function(Qc), pin_dofs, Q0)
    bcs = list(bcs) + [pin_bc]
    p_params = dict(params)
    p_params["pressure_gauge_eps"] = 0.0
    solver_cfg = cfg.get("solver", cfg)
    stages = solver_cfg.get("continuation_gbeta", [0.01, 0.1, 0.5, 1.0])
    w = fem.Function(W)
    if mfd is not None:
        for sub, key, expr in ((0, "u", mfd["u_expr"]),
                               (1, "p", mfd["p_expr"]),
                               (2, "T", mfd["t_expr"])):
            Vsub, _ = W.sub(sub).collapse()
            fsub = fem.Function(Vsub)
            fsub.interpolate(expr)
            if key == "p":
                area = float(fem.assemble_scalar(fem.form(ufl.as_ufl(1.0) * ufl.dx(domain=mesh))))
                pm = float(fem.assemble_scalar(fem.form(fsub * ufl.dx(domain=mesh)))) / area
                fsub.x.array[:] -= pm
                fsub.x.array[:] += float(initial_p_offset)
            w.sub(sub).interpolate(fsub)
        w.x.scatter_forward()
        if initial_perturbation > 0.0:
            rng = np.random.default_rng(int(perturbation_seed))
            # Perturb only interior u/T DOFs; wall dofs stay at exact manufactured BC.
            Vu, _ = W.sub(0).collapse(); Vt, _ = W.sub(2).collapse()
            u_fn = w.sub(0).collapse(); t_fn = w.sub(2).collapse()
            uw = np.asarray(fem.locate_dofs_geometrical((W.sub(0), Vu), on_wall), dtype=np.int32)
            tw = np.asarray(fem.locate_dofs_geometrical((W.sub(2), Vt), on_wall), dtype=np.int32)
            ucols = uw[1]; tcols = tw[1]
            u_arr = np.asarray(u_fn.x.array).reshape(-1, 2)
            t_arr = np.asarray(t_fn.x.array)
            u_wall = np.zeros(u_arr.size, dtype=bool); u_wall[ucols] = True
            t_wall = np.zeros(t_arr.size, dtype=bool); t_wall[tcols] = True
            uscale = initial_perturbation * max(1.0, float(np.max(np.abs(u_arr))))
            tscale = initial_perturbation * max(1.0, float(np.max(np.abs(t_arr))))
            u_noise = rng.standard_normal(u_arr.size) * uscale
            t_noise = rng.standard_normal(t_arr.size) * tscale
            u_noise[u_wall] = 0.0; t_noise[t_wall] = 0.0
            u_arr += u_noise.reshape(-1, 2); t_arr += t_noise
            w.sub(0).interpolate(u_fn); w.sub(2).interpolate(t_fn)
            w.x.scatter_forward()
            initial_perturbation_norm = float(np.linalg.norm(u_noise) + np.linalg.norm(t_noise))
        else:
            initial_perturbation_norm = 0.0
    elif initial_fields is not None:
        for sub, key in ((0, "velocity"), (1, "pressure"), (2, "temperature")):
            vals = np.asarray(initial_fields[f"{key}_values"], dtype=float)
            Vsub, _ = W.sub(sub).collapse()
            fsub = fem.Function(Vsub)
            if vals.size != fsub.x.array.size:
                raise RuntimeError(f"initial {key} size {vals.size} != {fsub.x.array.size}")
            fsub.x.array[:] = vals
            w.sub(sub).interpolate(fsub)
        w.x.scatter_forward()
        # Exact initial gauge: shift only pressure constant so p(x_pin)=0.
        pc = w.sub(1).collapse()
        init_pin = float(pc.x.array[pin_dof])
        pc.x.array[:] = pc.x.array - init_pin
        w.sub(1).interpolate(pc)
        w.x.scatter_forward()
        print("INIT_INJECT", "vel", float(np.linalg.norm(w.sub(0).collapse().x.array)),
              "p_after_pin_shift", float(np.linalg.norm(w.sub(1).collapse().x.array)),
              "init_pin", init_pin, "pin_now", float(w.sub(1).collapse().x.array[pin_dof]),
              "T", float(np.linalg.norm(w.sub(2).collapse().x.array)), flush=True)
    history = []
    for fraction in stages:
        stage_params = dict(p_params)
        stage_params["g_beta"] = float(p_params["g_beta"]) * float(fraction)
        F = base._residual(mesh, w, W, stage_params, ufl)
        try:
            # Picard with the accepted eps field as initial guess; pin BC is
            # validated separately on the linear Stokes system.
            reason, iterations = base._solve_picard(
                mesh, w, W, stage_params, bcs, solver_cfg, fem, ufl,
                _linear_problem(), f["MPI"])
        except RuntimeError:
            reason, iterations = -1, None
        print("EXACT_PICARD", fraction, "reason", reason,
              "iterations", iterations,
              "finite", bool(np.all(np.isfinite(w.x.array))), flush=True)
        if int(reason) < 0 or not bool(np.all(np.isfinite(w.x.array))):
            if not bool(np.all(np.isfinite(w.x.array))):
                w.x.array[:] = 0.0
            try:
                reason, iterations = base._solve_newton(
                    w, bcs, F, solver_cfg, f["NonlinearProblem"],
                    f["NewtonSolver"], f["MPI"])
            except RuntimeError:
                reason, iterations = -1, None
            print("EXACT_NEWTON", fraction, "reason", reason,
                  "iterations", iterations,
                  "finite", bool(np.all(np.isfinite(w.x.array))), flush=True)
        if int(reason) < 0 or not bool(np.all(np.isfinite(w.x.array))):
            history.append({"g_beta_fraction": float(fraction),
                            "reason": int(reason),
                            "iterations": (int(iterations) if iterations is not None else None)})
            break
        history.append({"g_beta_fraction": float(fraction),
                        "reason": int(reason),
                        "iterations": (int(iterations) if iterations is not None else None)})
        history.append({"g_beta_fraction": float(fraction),
                        "reason": int(reason),
                        "iterations": (int(iterations) if iterations is not None else None)})
        if int(reason) < 0:
            break
    if not history or history[-1]["reason"] < 0:
        raise RuntimeError("exact point-pin solve failed: " + repr(history))
    if not bool(np.all(np.isfinite(w.x.array))):
        raise RuntimeError("exact point-pin solve produced non-finite solution")
    diag = base.diagnostics(mesh, W, w, p_params, bcs, fem, ufl,
                            on_wall, on_inner, on_outer)
    p_collapsed = w.sub(1).collapse()
    p_pin = float(p_collapsed.x.array[pin_dof]) if pin_dof < len(p_collapsed.x.array) else None
    return {
        "mesh": mesh_data, "solution": w, "W": W, "mesh_data": mesh_data,
        "pin_dof": pin_dof, "pin_dofs_pair": [a.tolist() for a in pin_dofs],
        "pin_coord": pin_coord, "pin_margin": margin,
        "pin_value": p_pin,
        "newton_reason": history[-1]["reason"] if history else -1,
        "newton_iterations": history[-1]["iterations"] if history else None,
        "continuation": history,
        "diagnostics": diag,
        "weak_residual": base.weak_residual_norm(
            mesh, W, w, p_params, fem, ufl, on_wall, on_inner, on_outer),
        "pressure_gauge": "point_reference_exact",
        "pressure_gauge_eps": 0.0,
        "params": p_params,
    }


def support_fields(result: dict, points: np.ndarray) -> dict:
    """Evaluate actual dolfinx Function at frozen support coordinates."""
    ev = base.evaluate_at_points(result, np.asarray(points, dtype=float))
    area_mean = float(result["diagnostics"]["pressure_mean"])
    p_raw = np.asarray(ev["p"], dtype=float)
    if result.get("pressure_gauge") == "integral_lagrange_petsc":
        p_integral = p_raw
        p_point = p_raw - float(result.get("pin_value", 0.0))
    else:
        p_point = p_raw
        p_integral = p_point - area_mean
    return {
        "coords": np.asarray(ev["points"])[:, :2],
        "valid": ev["valid"],
        "cells": ev["cells"],
        "u": ev["u"], "v": ev["v"], "T": ev["T"],
        "p_point": p_point, "p_integral": p_integral,
        "area_mean_point": area_mean,
        "support_equal_weight_mean_point": (float(np.nanmean(p_point))
                                            if np.any(ev["valid"]) else None),
        "support_equal_weight_mean_integral": (float(np.nanmean(p_integral))
                                               if np.any(ev["valid"]) else None),
        "pin_value": result.get("pin_value"),
    }


def gauge_comparison(fields: dict, result: dict) -> dict:
    valid = np.asarray(fields["valid"], dtype=bool)
    a = np.asarray(fields["p_point"], dtype=float)[valid]
    b = np.asarray(fields["p_integral"], dtype=float)[valid]
    center_a = a - a.mean()
    center_b = b - b.mean()
    diff = a - b
    denominator = float(np.linalg.norm(center_b))
    return {
        "n_valid": int(valid.sum()),
        "n_total": int(valid.size),
        "area_mean_point": float(fields["area_mean_point"]),
        "mean_pA_minus_pB": float(diff.mean()) if diff.size else None,
        "centered_relative_difference": (float(np.linalg.norm(center_a - center_b))
                                         / denominator) if denominator > 0 else None,
        "centered_absolute_difference": float(np.linalg.norm(center_a - center_b)),
        "max_abs_pA_minus_pB": float(np.max(np.abs(diff))) if diff.size else None,
        "pin_value": result.get("pin_value"),
        "pressure_gauge": "point_reference_vs_integral_zero_mean",
        "note": "integral gauge is exact constant shift; u/v/T identical by construction",
    }



def _manufactured_polynomial_fields(mesh, params, ufl):
    """Cubic streamfunction -> quadratic velocity; P2/P1-compatible MMS.

    psi = x^3 - 3 x y^2 + 0.5 x^2 y + 0.2 y^3
    u = (psi_y, -psi_x) is quadratic and divergence-free; its trace on straight
    polygon/square edges is exactly representable by P2.  T is quadratic (P2),
    p is linear (P1).  Nonhomogeneous BCs are therefore discretely compatible.
    """
    x = ufl.SpatialCoordinate(mesh)
    psi = (x[0] ** 3 - 3.0 * x[0] * x[1] ** 2
           + 0.5 * x[0] ** 2 * x[1] + 0.2 * x[1] ** 3)
    u_exact = ufl.as_vector([psi.dx(1), -psi.dx(0)])
    T_exact = (0.3 + x[0] + 0.5 * x[1] + 0.7 * x[0] ** 2
               + 0.2 * x[0] * x[1] - 0.4 * x[1] ** 2)
    p_exact = 0.2 + 0.4 * x[0] - 0.3 * x[1]
    conv = ufl.as_vector([
        u_exact[0] * u_exact[0].dx(0) + u_exact[1] * u_exact[0].dx(1),
        u_exact[0] * u_exact[1].dx(0) + u_exact[1] * u_exact[1].dx(1)])
    lap_u = ufl.div(ufl.grad(u_exact))
    f_vel = (params["rho"] * conv - params["mu"] * lap_u
             + ufl.grad(p_exact)
             + ufl.as_vector([0.0, -params["rho"] * params["g_beta"] * T_exact]))
    f_T = (params["rho"] * params["cp"] * ufl.dot(u_exact, ufl.grad(T_exact))
           - params["kappa"] * ufl.div(ufl.grad(T_exact)))

    def u_expr(q):
        x0, x1 = q[0], q[1]
        return np.vstack([-6.0 * x0 * x1 + 0.5 * x0 ** 2 + 0.6 * x1 ** 2,
                          -3.0 * x0 ** 2 + 3.0 * x1 ** 2 - x0 * x1])

    def t_expr(q):
        x0, x1 = q[0], q[1]
        return 0.3 + x0 + 0.5 * x1 + 0.7 * x0 ** 2 + 0.2 * x0 * x1 - 0.4 * x1 ** 2

    def p_expr(q):
        return 0.2 + 0.4 * q[0] - 0.3 * q[1]
    return {"u": u_exact, "p": p_exact, "T": T_exact,
            "f_vel": f_vel, "f_T": f_T,
            "u_expr": u_expr, "t_expr": t_expr, "p_expr": p_expr,
            "case": "polynomial_cubic_streamfunction"}


def _manufactured_fields(mesh, params, ufl):
    """Asymmetric, non-zero exact fields and nonlinear Boussinesq sources."""
    x = ufl.SpatialCoordinate(mesh)
    pi = ufl.pi
    psi = (ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
           + 0.2 * ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1]))
    u_exact = ufl.as_vector([psi.dx(1), -psi.dx(0)])
    T_exact = (ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
               + 0.2 * ufl.cos(pi * x[0]) * ufl.sin(2 * pi * x[1]))
    p_exact = (ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
               + 0.3 * ufl.sin(pi * x[0]) * ufl.cos(2 * pi * x[1]))
    conv = ufl.as_vector([
        u_exact[0] * u_exact[0].dx(0) + u_exact[1] * u_exact[0].dx(1),
        u_exact[0] * u_exact[1].dx(0) + u_exact[1] * u_exact[1].dx(1)])
    lap_u = ufl.div(ufl.grad(u_exact))
    f_vel = (params["rho"] * conv - params["mu"] * lap_u
             + ufl.grad(p_exact)
             + ufl.as_vector([0.0, -params["rho"] * params["g_beta"] * T_exact]))
    f_T = (params["rho"] * params["cp"] * ufl.dot(u_exact, ufl.grad(T_exact))
           - params["kappa"] * ufl.div(ufl.grad(T_exact)))

    def u_expr(q):
        return np.vstack([
            np.pi * np.sin(np.pi * q[0]) * np.cos(np.pi * q[1])
            + 0.2 * np.pi * np.sin(2 * np.pi * q[0]) * np.cos(np.pi * q[1]),
            -np.pi * np.cos(np.pi * q[0]) * np.sin(np.pi * q[1])
            - 0.4 * np.pi * np.cos(2 * np.pi * q[0]) * np.sin(np.pi * q[1])])

    def t_expr(q):
        return (np.sin(np.pi * q[0]) * np.sin(np.pi * q[1])
                + 0.2 * np.cos(np.pi * q[0]) * np.sin(2 * np.pi * q[1]))

    def p_expr(q):
        return (np.cos(np.pi * q[0]) * np.sin(np.pi * q[1])
                + 0.3 * np.sin(np.pi * q[0]) * np.cos(2 * np.pi * q[1]))
    return {"u": u_exact, "p": p_exact, "T": T_exact,
            "f_vel": f_vel, "f_T": f_T,
            "u_expr": u_expr, "t_expr": t_expr, "p_expr": p_expr}


def solve_integral_gauge(entry: dict, params: dict, cfg: dict,
                      target_h: float, initial_fields: dict | None = None,
                      manufactured: bool = False, initial_p_offset: float = 0.0,
                      relaxation: float | None = None,
                      manufactured_case: str = "trig",
                      initial_perturbation: float = 0.0,
                      perturbation_seed: int = 20261023,
                      free_tol: float | None = None) -> dict:
    """Exact integral-zero pressure by Newton correction with a PETSc saddle row.

    At each Newton step: J = dF/dw at eps=0, R = -F, plus the scalar constraint
    integral(p) dx = 0.  The augmented matrix is [J, b; b^T, 0], where b is the
    continuity-test mass vector.  No pressure test row is deleted and no
    epsilon mass penalty is used.
    """
    f = _fenics()
    fem, ufl = f["fem"], f["ufl"]
    from petsc4py import PETSc
    from dolfinx.fem.petsc import (assemble_matrix, assemble_vector,
                                   apply_lifting, set_bc)
    mesh_data = base.build_gmsh_mesh(entry, target_h)
    mesh = mesh_data["mesh"]
    W = base._spaces(mesh, f["element"], f["mixed_element"], fem)
    p_params = dict(params)
    p_params["pressure_gauge_eps"] = 0.0
    if manufactured:
        if manufactured_case == "polynomial":
            mfd = _manufactured_polynomial_fields(mesh, p_params, ufl)
        else:
            mfd = _manufactured_fields(mesh, p_params, ufl)
    else:
        mfd = None
    bcs, on_wall, on_inner, on_outer = base._boundary_conditions(
        mesh, W, entry, fem, ufl,
        exact_stokes_source=None if mfd is None else mfd["u_expr"],
        exact_energy_source=None if mfd is None else mfd["t_expr"])
    pin_dofs, pin_dof, pin_coord, margin = _pin_dof(mesh, W, entry, fem)
    w = fem.Function(W)
    if mfd is not None:
        for sub, key, expr in ((0, "u", mfd["u_expr"]),
                               (1, "p", mfd["p_expr"]),
                               (2, "T", mfd["t_expr"])):
            Vsub, _ = W.sub(sub).collapse()
            fsub = fem.Function(Vsub)
            fsub.interpolate(expr)
            if key == "p":
                area = float(fem.assemble_scalar(fem.form(ufl.as_ufl(1.0) * ufl.dx(domain=mesh))))
                pm = float(fem.assemble_scalar(fem.form(fsub * ufl.dx(domain=mesh)))) / area
                fsub.x.array[:] -= pm
                fsub.x.array[:] += float(initial_p_offset)
            w.sub(sub).interpolate(fsub)
        w.x.scatter_forward()
        if initial_perturbation > 0.0:
            rng = np.random.default_rng(int(perturbation_seed))
            # Perturb only interior u/T DOFs; wall dofs stay at exact manufactured BC.
            Vu, _ = W.sub(0).collapse(); Vt, _ = W.sub(2).collapse()
            u_fn = w.sub(0).collapse(); t_fn = w.sub(2).collapse()
            uw = np.asarray(fem.locate_dofs_geometrical((W.sub(0), Vu), on_wall), dtype=np.int32)
            tw = np.asarray(fem.locate_dofs_geometrical((W.sub(2), Vt), on_wall), dtype=np.int32)
            ucols = uw[1]; tcols = tw[1]
            u_arr = np.asarray(u_fn.x.array).reshape(-1, 2)
            t_arr = np.asarray(t_fn.x.array)
            u_wall = np.zeros(u_arr.size, dtype=bool); u_wall[ucols] = True
            t_wall = np.zeros(t_arr.size, dtype=bool); t_wall[tcols] = True
            uscale = initial_perturbation * max(1.0, float(np.max(np.abs(u_arr))))
            tscale = initial_perturbation * max(1.0, float(np.max(np.abs(t_arr))))
            u_noise = rng.standard_normal(u_arr.size) * uscale
            t_noise = rng.standard_normal(t_arr.size) * tscale
            u_noise[u_wall] = 0.0; t_noise[t_wall] = 0.0
            u_arr += u_noise.reshape(-1, 2); t_arr += t_noise
            w.sub(0).interpolate(u_fn); w.sub(2).interpolate(t_fn)
            w.x.scatter_forward()
            initial_perturbation_norm = float(np.linalg.norm(u_noise) + np.linalg.norm(t_noise))
        else:
            initial_perturbation_norm = 0.0
    elif initial_fields is not None:
        for sub, key in ((0, "velocity"), (1, "pressure"), (2, "temperature")):
            vals = np.asarray(initial_fields[f"{key}_values"], dtype=float)
            Vsub, _ = W.sub(sub).collapse()
            fsub = fem.Function(Vsub)
            if vals.size != fsub.x.array.size:
                raise RuntimeError(f"initial {key} size {vals.size} != {fsub.x.array.size}")
            fsub.x.array[:] = vals
            w.sub(sub).interpolate(fsub)
        w.x.scatter_forward()
    trial = ufl.TrialFunction(W)
    testv, testq, tests = ufl.TestFunctions(W)
    dx = ufl.dx(domain=mesh)
    b_form = fem.form(testq * dx)
    b_vec = assemble_vector(b_form)
    b_vec.assemble()
    b_arr = b_vec.getArray().copy()
    initial_integral = float(np.dot(b_arr, w.x.petsc_vec.getArray()))
    initial_perturbation_norm = locals().get("initial_perturbation_norm", 0.0)
    n = int(w.x.petsc_vec.getSize())
    solver_cfg = cfg.get("solver", cfg)
    max_it = int(solver_cfg.get("max_iterations", 20))
    relax = float(solver_cfg.get("relaxation", 0.5) if relaxation is None else relaxation)
    tol = float(solver_cfg.get("convergence", {}).get("update_rel_tolerance", 1e-8))
    F = base._residual(
        mesh, w, W, p_params, ufl,
        source_velocity=None if mfd is None else mfd["f_vel"],
        source_temperature=None if mfd is None else mfd["f_T"])
    J = ufl.derivative(F, w, trial)
    R0 = assemble_vector(fem.form(F))
    R0arr = np.abs(R0.getArray().copy())
    dofs0 = base._mixed_dirichlet_dofs(W, fem, on_wall, on_inner, on_outer)
    mask0 = np.ones(R0arr.size, dtype=bool); mask0[dofs0] = False
    initial_source_free_residual = float(np.linalg.norm(R0arr[mask0]))
    initial_source_full_residual = float(np.linalg.norm(R0arr))
    history = []
    for iteration in range(1, max_it + 1):
        A = assemble_matrix(fem.form(J), bcs=bcs)
        A.assemble()
        R = assemble_vector(fem.form(F))
        # Match dolfinx.fem.petsc.NonlinearProblem.F exactly: absolute BCs
        # lifted against the current x0=w, and BC rows set to the residual
        # (homogeneous increment) with scale=-1.
        apply_lifting(R, [fem.form(J)], bcs=[bcs], x0=[w.x.petsc_vec], scale=-1.0)
        R.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(R, bcs, w.x.petsc_vec, -1.0)
        # R is the assembled residual; Newton correction solves J delta = -R.
        R.scale(-1.0)
        print("R_DEBUG", "norm", float(R.norm()),
              "arrnorm", float(np.linalg.norm(R.getArray())), flush=True)
        p_cur = w.sub(1).collapse()
        C = base._assemble_scalar(fem.form(p_cur * dx), fem)
        A_aug = PETSc.Mat().createAIJ((n + 1, n + 1), comm=A.getComm())
        A_aug.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        for i in range(n):
            cols, vals = A.getRow(i)
            if len(cols):
                A_aug.setValues(i, cols, vals)
        bscale = 1.0 / max(float(np.linalg.norm(b_arr)), 1e-30)
        for i in range(n):
            bi = float(b_arr[i]) * bscale
            if bi != 0.0:
                A_aug.setValue(i, n, bi)
                A_aug.setValue(n, i, bi)
        # Exact constraint: multiplier diagonal is zero; row and RHS are
        # consistently scaled by bscale.
        A_aug.setValue(n, n, 0.0)
        A_aug.assemble()
        A_check = A_aug.copy()
        rhs_aug = PETSc.Vec().createSeq(n + 1, comm=A_aug.getComm())
        ra = rhs_aug.getArray()
        ra[:n] = R.getArray()
        # Unknown delta must satisfy bscale*b^T delta = -bscale*C.
        ra[n] = -float(C) * bscale
        rhs_aug.assemble()
        print("RHS_DEBUG", "arrnorm", float(np.linalg.norm(ra)),
              "vecnorm", float(rhs_aug.norm()), flush=True)
        sol = rhs_aug.duplicate()
        ksp = PETSc.KSP().create(comm=A_aug.getComm())
        ksp.setOperators(A_aug)
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        try:
            pc.setFactorSolverType("mumps")
        except Exception:
            pass
        ksp.solve(rhs_aug, sol)
        print("KSP_REASON", int(ksp.getConvergedReason()), "iters", int(ksp.getIterationNumber()), flush=True)
        lin = sol.duplicate()
        A_check.mult(sol, lin)
        lin.axpy(-1.0, rhs_aug)
        lin_rel = float(lin.norm() / max(float(rhs_aug.norm()), 1e-30))
        print("LIN_RESIDUAL_INDEPENDENT", lin_rel, "rhs_norm", float(rhs_aug.norm()), flush=True)
        if not np.isfinite(lin_rel) or lin_rel > 1e-8:
            raise RuntimeError(f"augmented linear solve untrusted: rel residual {lin_rel:.3e}")
        sol_arr = sol.getArray()
        old_arr = w.x.petsc_vec.getArray().copy()
        delta = sol_arr[:n]
        rel = float(np.linalg.norm(delta) / max(float(np.linalg.norm(old_arr)), 1e-30))
        w.x.petsc_vec.setArray(old_arr + relax * delta)
        w.x.scatter_forward()
        c_mult = float(sol_arr[n])
        pre_proj = float(np.dot(b_arr, w.x.petsc_vec.getArray()))
        # Exact post-update gauge projection; only pressure constant is shifted.
        area_c = base._assemble_scalar(fem.form(ufl.as_ufl(1.0) * dx), fem)
        p_corr = w.sub(1).collapse()
        shift = pre_proj / area_c
        p_corr.x.array[:] -= shift
        w.sub(1).interpolate(p_corr)
        w.x.scatter_forward()
        constraint = float(np.dot(b_arr, w.x.petsc_vec.getArray()))
        history.append({"iteration": iteration, "update_rel": rel,
                        "update_norm": float(np.linalg.norm(delta)),
                        "linear_residual_rel": lin_rel,
                        "pre_projection_integral": pre_proj,
                        "projection_constant_shift": shift,
                        "constraint": constraint, "c": c_mult})
        print("INTEGRAL_NEWTON", iteration, "rel", rel, "lin", lin_rel,
              "constraint", constraint, "c", c_mult,
              "Rnorm_after", float(R.norm()), flush=True)
        if not bool(np.all(np.isfinite(w.x.array))):
            raise RuntimeError("integral gauge produced non-finite solution")
        # Hard gate: update_rel alone is insufficient.  Assemble the actual
        # source-bearing free residual before accepting convergence.
        Rchk = assemble_vector(fem.form(F))
        rchk = np.abs(Rchk.getArray().copy())
        free_now = float(np.linalg.norm(rchk[mask0]))
        history[-1]["free_residual_vector_norm"] = free_now
        if rel < tol and (free_tol is None or free_now < float(free_tol)):
            break
    if not history or history[-1]["update_rel"] >= tol:
        raise RuntimeError("integral gauge Newton did not converge: " + repr(history[-3:]))
    if abs(history[-1]["constraint"]) > 1e-7:
        raise RuntimeError("integral constraint not converged: " + repr(history[-1]))
    if free_tol is not None and history[-1].get("free_residual_vector_norm", 1e30) >= float(free_tol):
        raise RuntimeError("free residual not converged: " + repr(history[-1]))
    diag = base.diagnostics(mesh, W, w, p_params, bcs, fem, ufl,
                            on_wall, on_inner, on_outer)
    p_collapsed = w.sub(1).collapse()
    pin_value = float(p_collapsed.x.array[pin_dof])
    integral = base._assemble_scalar(fem.form(p_collapsed * dx), fem)
    free_residual = base.weak_residual_norm(
        mesh, W, w, p_params, fem, ufl, on_wall, on_inner, on_outer)
    manufactured_metrics = None
    if mfd is not None:
        u, p, T = ufl.split(w)
        dx = ufl.dx(domain=mesh)
        area = float(fem.assemble_scalar(fem.form(ufl.as_ufl(1.0) * dx)))
        pm_sol = float(fem.assemble_scalar(fem.form(p * dx))) / area
        pm_ex = float(fem.assemble_scalar(fem.form(mfd["p"] * dx))) / area
        u_err = float(np.sqrt(max(fem.assemble_scalar(
            fem.form(ufl.dot(u - mfd["u"], u - mfd["u"]) * dx)), 0.0)))
        t_err = float(np.sqrt(max(fem.assemble_scalar(
            fem.form((T - mfd["T"]) ** 2 * dx)), 0.0)))
        p_err = float(np.sqrt(max(fem.assemble_scalar(
            fem.form((p - pm_sol - (mfd["p"] - pm_ex)) ** 2 * dx)), 0.0)))
        u_ref = float(np.sqrt(max(fem.assemble_scalar(
            fem.form(ufl.dot(mfd["u"], mfd["u"]) * dx)), 0.0)))
        t_ref = float(np.sqrt(max(fem.assemble_scalar(
            fem.form(mfd["T"] ** 2 * dx)), 0.0)))
        p_ref = float(np.sqrt(max(fem.assemble_scalar(
            fem.form((mfd["p"] - pm_ex) ** 2 * dx)), 0.0)))
        Rvec = assemble_vector(fem.form(F))
        Rarr = np.abs(Rvec.getArray().copy())
        dofs = base._mixed_dirichlet_dofs(W, fem, on_wall, on_inner, on_outer)
        mask = np.ones(Rarr.size, dtype=bool); mask[dofs] = False
        manufactured_metrics = {
            "velocity_l2": u_err, "velocity_rel_l2": u_err / u_ref if u_ref > 0 else None,
            "temperature_l2": t_err, "temperature_rel_l2": t_err / t_ref if t_ref > 0 else None,
            "pressure_centered_l2": p_err,
            "pressure_centered_rel_l2": p_err / p_ref if p_ref > 0 else None,
            "free_residual_vector_norm": float(np.linalg.norm(Rarr[mask])),
            "full_residual_vector_norm": float(np.linalg.norm(Rarr)),
            "n_dirichlet": int(len(dofs)),
            "boundary_wall_velocity_max": diag["boundary"].get("wall_velocity_max"),
            "boundary_inner_temperature_max_dev": diag["boundary"].get("inner_temperature_max_dev"),
            "boundary_outer_temperature_max_dev": diag["boundary"].get("outer_temperature_max_dev"),
            "u_nonzero": float(np.max(np.abs(np.asarray(mfd["u_expr"]([0.3, 0.2]))))),
            "p_nonzero": float(np.max(np.abs(np.asarray(mfd["p_expr"]([0.3, 0.2]))))),
        }
        manufactured_metrics["boundary_values_finite"] = bool(
            np.all(np.isfinite(w.x.array)))
        # Actual boundary Dirichlet error at the frozen boundary support points.
        support_root = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))),
            "results", "round2_thermal_fem", "v2_model_eval")
        gid = int(entry.get("geom", 17))
        sp = np.load(os.path.join(support_root, f"support_geom{gid}.npz"))
        bmask = sp["wall_kind"] != "interior"
        bpts = sp["coords"][bmask].astype(float)
        ev = base.evaluate_at_points({"mesh": mesh_data, "solution": w}, bpts)
        valid = ev["valid"]
        ub = np.column_stack([ev["u"], ev["v"]])[valid]
        tb = np.asarray(ev["T"])[valid]
        if np.any(valid):
            ux = np.asarray(mfd["u_expr"](bpts[valid].T)).T
            tx = np.asarray(mfd["t_expr"](bpts[valid].T)).reshape(-1)
            manufactured_metrics["boundary_velocity_max_error"] = float(
                np.max(np.abs(ub - ux)))
            manufactured_metrics["boundary_temperature_max_error"] = float(
                np.max(np.abs(tb - tx)))
        else:
            manufactured_metrics["boundary_velocity_max_error"] = None
            manufactured_metrics["boundary_temperature_max_error"] = None
        manufactured_metrics["boundary_points"] = int(bpts.shape[0])
        manufactured_metrics["boundary_valid"] = int(valid.sum())
        # Per actual Dirichlet DOF error (not just boundary point sampling).
        Vu, _ = W.sub(0).collapse(); Vt, _ = W.sub(2).collapse()
        uw = np.asarray(fem.locate_dofs_geometrical((W.sub(0), Vu), on_wall), dtype=np.int32)
        tw = np.asarray(fem.locate_dofs_geometrical((W.sub(2), Vt), on_wall), dtype=np.int32)
        ucols = uw[1]; tcols = tw[1]
        u_fn = w.sub(0).collapse(); t_fn = w.sub(2).collapse()
        u_arr = np.asarray(u_fn.x.array).reshape(-1, 2)
        ucoords = Vu.tabulate_dof_coordinates()[ucols // 2, :2]
        u_ex = np.asarray(mfd["u_expr"](ucoords.T)).T
        u_act = u_arr[ucols // 2, ucols % 2]
        u_exd = u_ex[np.arange(ucols.size), ucols % 2]
        tcoords = Vt.tabulate_dof_coordinates()[tcols, :2]
        t_ex = np.asarray(mfd["t_expr"](tcoords.T)).reshape(-1)
        t_act = np.asarray(t_fn.x.array)[tcols]
        manufactured_metrics["boundary_velocity_dof_max_error"] = float(
            np.max(np.abs(u_act - u_exd)))
        manufactured_metrics["boundary_temperature_dof_max_error"] = float(
            np.max(np.abs(t_act - t_ex)))
        manufactured_metrics["boundary_velocity_dofs"] = int(ucols.size)
        manufactured_metrics["boundary_temperature_dofs"] = int(tcols.size)
        # Closed-boundary flux and q=1 continuity compatibility.
        nrm = ufl.FacetNormal(mesh)
        ds = ufl.ds(domain=mesh)
        manufactured_metrics["boundary_flux_solution"] = float(
            base._assemble_scalar(fem.form(ufl.dot(u, nrm) * ds), fem))
        manufactured_metrics["boundary_flux_exact"] = float(
            base._assemble_scalar(fem.form(ufl.dot(mfd["u"], nrm) * ds), fem))
        manufactured_metrics["continuity_integral_solution"] = float(
            base._assemble_scalar(fem.form(ufl.div(u) * dx), fem))
        manufactured_metrics["continuity_integral_exact"] = float(
            base._assemble_scalar(fem.form(ufl.div(mfd["u"]) * dx), fem))
        # Correct source-bearing residual block norms on free DOFs.
        uu, pp, TT = ufl.split(w)
        conv_s = ufl.as_vector([
            uu[0] * uu[0].dx(0) + uu[1] * uu[0].dx(1),
            uu[0] * uu[1].dx(0) + uu[1] * uu[1].dx(1)])
        F_mom = (p_params["rho"] * conv_s
                 - p_params["mu"] * ufl.div(ufl.grad(uu))
                 + ufl.grad(pp)
                 + ufl.as_vector([0.0, -p_params["rho"] * p_params["g_beta"] * TT])
                 - mfd["f_vel"])
        F_cont = ufl.div(uu)
        F_ener = (p_params["rho"] * p_params["cp"] * ufl.dot(uu, ufl.grad(TT))
                  - p_params["kappa"] * ufl.div(ufl.grad(TT)) - mfd["f_T"])
        # Block L2 residuals of the source-bearing equations.
        cont_sq = fem.assemble_scalar(fem.form(F_cont * F_cont * dx))
        mom_sq = 0.0
        for comp in range(2):
            mom_sq += fem.assemble_scalar(fem.form(F_mom[comp] * F_mom[comp] * dx))
        ener_sq = fem.assemble_scalar(fem.form(F_ener * F_ener * dx))
        manufactured_metrics["block_l2_residual_continuity"] = float(
            np.sqrt(max(cont_sq, 0.0)))
        manufactured_metrics["block_l2_residual_momentum"] = float(
            np.sqrt(max(mom_sq, 0.0)))
        manufactured_metrics["block_l2_residual_energy"] = float(
            np.sqrt(max(ener_sq, 0.0)))
        source_norm = float(np.sqrt(max(fem.assemble_scalar(
            fem.form((ufl.dot(mfd["f_vel"], mfd["f_vel"])
                      + mfd["f_T"] ** 2) * dx)), 0.0)))
        manufactured_metrics["source_l2_norm"] = source_norm
        manufactured_metrics["free_residual_relative_to_source"] = (
            manufactured_metrics["free_residual_vector_norm"]
            / max(source_norm, 1e-30))
    return {
        "mesh": mesh_data, "solution": w, "W": W, "pin_dof": pin_dof,
        "pin_coord": pin_coord, "pin_margin": margin, "pin_value": pin_value,
        "integral_pressure": integral, "history": history,
        "diagnostics": diag, "weak_residual": free_residual,
        "pressure_gauge": "integral_lagrange_petsc",
        "pressure_gauge_eps": 0.0,
        "manufactured_case": manufactured_case if manufactured else None,
        "initial_p_offset": float(initial_p_offset),
        "initial_integral": initial_integral,
        "initial_perturbation_norm": initial_perturbation_norm,
        "initial_source_free_residual": initial_source_free_residual,
        "initial_source_full_residual": initial_source_full_residual,
        "relaxation": relax,
        "final_integral": float(np.dot(b_arr, w.x.petsc_vec.getArray())),
        "final_linear_residual_rel": history[-1]["linear_residual_rel"],
        "final_constraint": history[-1]["constraint"],
        "newton_reason": 2, "newton_iterations": history[-1]["iteration"],
        "params": p_params,
        "manufactured_metrics": manufactured_metrics,
    }


def solve_manufactured_pinned(entry: dict, params: dict, cfg: dict,
                              target_h: float) -> dict:
    """One real nonlinear manufactured-solution solve with the exact pin.

    The manufactured pressure has a nonzero constant; the pin removes it.  We
    compare velocity/temperature directly and pressure only after exact constant
    centering (the physics is invariant to this constant).
    """
    f = _fenics()
    fem, ufl = f["fem"], f["ufl"]
    mesh_data = base.build_gmsh_mesh(entry, target_h)
    mesh = mesh_data["mesh"]
    W = base._spaces(mesh, f["element"], f["mixed_element"], fem)
    u_exact, p_exact, T_exact, f_vel, f_T = base.manufactured_fields(
        mesh, params, ufl, nonlinear=True)
    bcs, _, _, _ = base._boundary_conditions(
        mesh, W, entry, fem, ufl,
        exact_stokes_source=lambda x: np.vstack([
            np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]),
            -np.cos(np.pi * x[0]) * np.sin(np.pi * x[1])]),
        exact_energy_source=lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    pin_dofs, pin_dof, pin_coord, margin = _pin_dof(mesh, W, entry, fem)
    Q0 = W.sub(1)
    Qc, _ = Q0.collapse()
    bcs = list(bcs) + [fem.dirichletbc(fem.Function(Qc), pin_dofs, Q0)]
    p_params = dict(params)
    p_params["pressure_gauge_eps"] = 0.0
    w = fem.Function(W)
    F = base._residual(mesh, w, W, p_params, ufl,
                       source_velocity=f_vel, source_temperature=f_T,
                       linear_stokes=False)
    solver_cfg = cfg.get("solver", cfg)
    try:
        reason, iterations = base._solve_newton(
            w, bcs, F, solver_cfg, f["NonlinearProblem"],
            f["NewtonSolver"], f["MPI"])
    except RuntimeError:
        reason, iterations = base._solve_picard(
            mesh, w, W, p_params, bcs, solver_cfg, fem, ufl,
            _linear_problem(), f["MPI"])
    u, p, T = ufl.split(w)
    area = base._assemble_scalar(
        fem.form(ufl.as_ufl(1.0) * ufl.dx(domain=mesh)), fem)
    p_mean = base._assemble_scalar(fem.form(p * ufl.dx), fem) / area
    pe_mean = base._assemble_scalar(fem.form(p_exact * ufl.dx), fem) / area
    u_err = base._assemble_scalar(
        fem.form(ufl.dot(u - u_exact, u - u_exact) * ufl.dx), fem)
    t_err = base._assemble_scalar(
        fem.form((T - T_exact) ** 2 * ufl.dx), fem)
    p_err = base._assemble_scalar(
        fem.form((p - p_mean - (p_exact - pe_mean)) ** 2 * ufl.dx), fem)
    u_ref = base._assemble_scalar(
        fem.form(ufl.dot(u_exact, u_exact) * ufl.dx), fem)
    t_ref = base._assemble_scalar(
        fem.form(T_exact ** 2 * ufl.dx), fem)
    p_ref = base._assemble_scalar(
        fem.form((p_exact - pe_mean) ** 2 * ufl.dx), fem)
    return {
        "mesh": {k: mesh_data[k] for k in ("n_cells", "n_vertices", "target_h")},
        "pin_dof": pin_dof, "pin_dofs_pair": [a.tolist() for a in pin_dofs],
        "pin_coord": pin_coord, "pin_margin": margin,
        "newton_reason": int(reason), "newton_iterations": int(iterations)
        if iterations is not None else None,
        "pressure_relative_l2_centered": math.sqrt(max(p_err / p_ref, 0.0))
        if p_ref > 0 else None,
        "velocity_relative_l2": math.sqrt(max(u_err / u_ref, 0.0))
        if u_ref > 0 else None,
        "temperature_relative_l2": math.sqrt(max(t_err / t_ref, 0.0))
        if t_ref > 0 else None,
        "pressure_area_mean": float(p_mean),
        "pressure_exact_area_mean": float(pe_mean),
    }
