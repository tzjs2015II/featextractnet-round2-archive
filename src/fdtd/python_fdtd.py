"""2-D acoustic FDTD solver: NumPy port of the MATLAB *discretisation* in
``codematlab/`` (read-only reference for the FeatExtractNet 返稿工作区).

Scope of the claim
------------------
This module reproduces the numerical scheme of the MATLAB reference so that the
physical formulas and coefficients are the same: same PML damping terms, same
4-stage RK4, same 2nd-order central differences, same hard line source. It is
**not** a pointwise-validated reproduction: no per-node comparison against
``codematlab/*.mat`` was performed (author decision D3), the reference main
script itself only advances 2 steps before loading external ``.mat`` fields and
adding noise, and several deviations are deliberate (listed at the end of this
docstring). Wording therefore says "port of the discretisation", not
"faithful/line-by-line equivalent" — see docs/审计_协议合规核对.md P1-4.

Design intent
-------------
This module reproduces, line by line, the numerical scheme of the MATLAB
reference implementation so that physical formulas and coefficients are
identical. It solves the linearised Euler / convected-wave equations on a
2-D Cartesian grid:

  * total fields   ``(p, wx, wy)`` : pressure + acoustic particle velocity
    propagating through a time-invariant Lamb-Oseen vortex background
    ``(vx, vy)`` (the vortex moves the sound wave => scattered field);
  * incident field ``(P, Wx, Wy)`` : the same plane wave in a quiescent
    homogeneous medium (linear advection-free equations);
  * scattered pressure ``PP = p - P``.

Time integration is a classical 4-stage Runge-Kutta (RK4) update applied
sequentially to each field, with 2nd-order central differences in space.
An absorbing perfectly-matched-layer-like damping term ``-sigma * w``
(constant ``sigma``) plus one-sided stencils is applied in the outer PML
cells of the particle-velocity fields (mirroring the reference code; it is
*not* a graded/analytic PML).

Source injection reproduces the reference: a hard (Dirichlet) plane-wave
line source ``sin(2*pi*f*t)`` is written onto the first physical-air column
(or the last one, see ``source_side``) over the ``pjs:pje`` row range of
*both* ``p`` and ``P`` after every time step.

MATLAB file -> Python method / block mapping
--------------------------------------------
* define_problem_space_parameters_2d.m        -> ``FDTDAcoustic2D.__init__``
  (n_steps, courant_factor, dx/dy, PML & air-buffer layers, dt) and defaults
  of ``run_case``.
* initialize_fdtd_grid_2d.m                   -> ``__init__``: domain
  extents, ``nx/ny``, node coordinates and the physical-air index ranges
  ``pis/pie/pjs/pje`` (MATLAB 1-based) stored as 0-based slices.
* initialize_fdtd_parameters_and_arrays_2d.m  -> ``__init__``: allocation of
  all field / RK4-stage / forcing arrays plus ``b=1/rho0``, ``kap=rho0*c0^2``.
* initialize_vortex_2d.m                       -> ``_init_vortex``: Lamb-Oseen
  mean flow ``(vx, vy)`` (Gamma = 2.8*Ma*pi*L*c0, alpha = 1.256431).
* update_pressure_fields_2d.m                  -> ``_update_pressure_fields``
* update_velocity_x_fields_2d.m                -> ``_update_velocity_x_fields``
* update_velocity_y_fields_2d.m                -> ``_update_velocity_y_fields``
* update_velocity_x_fields_for_PML_2d.m        -> ``_update_velocity_x_fields_for_pml``
* update_velocity_y_fields_for_PML_2d.m        -> ``_update_velocity_y_fields_for_pml``
* update_P1ressure_fields_2d.m                 -> ``_update_incident_pressure_fields``
* update_V1elocity_x_fields_2d.m               -> ``_update_incident_velocity_x_fields``
* update_V1elocity_y_fields_2d.m               -> ``_update_incident_velocity_y_fields``
* update_V1elocity_x_fields_for_PML_2d.m       -> ``_update_incident_velocity_x_fields_for_pml``
* update_V1elocity_y_fields_for_PML_2d.m       -> ``_update_incident_velocity_y_fields_for_pml``
* run_fdtd_time_marching_loop_2d.m             -> ``solve`` (per-step calling
  order and source injection are identical; plotting/noise parts skipped).

Faithfully reproduced reference quirks (deliberately kept for numerical
equivalence with the legacy MATLAB data)
---------------------------------------------
* ``update_velocity_x_fields_for_PML_2d.m``: on the *left* and *below* PML
  bands the final combination reads ``fx1+2*fx1+2*fx1+fx1`` (=> dt*fx1) and
  ``fx1+2*fx2+2*fx1+fx1`` respectively, instead of ``fx1+2*fx2+2*fx3+fx4``.
* ``update_V1elocity_x_fields_for_PML_2d.m``: identical typos on left/below
  bands of the incident velocity field.
* ``update_velocity_y_fields_for_PML_2d.m`` / ``update_V1elocity_y_fields_for_PML_2d.m``:
  on the *right* band the 2nd RK4 intermediate ``wy2/Wy2`` is advanced with
  ``fy1/Fy1`` (not ``fy2/Fy2``).
* ``update_velocity_x_fields_for_PML_2d.m`` below band line 60 indexes
  ``2:pis-1`` in y where ``2:pjs-1`` is meant; equal for the square domain
  (pis == pjs), harmless.
* The ``current_time = current_time + dt`` statement that each update_* file
  executes is a display-only counter in the reference; here time is derived
  from the step index instead (no effect on the fields).
* The reference builds the vortex on a grid with step ``0.1251`` (intended
  ``0.125``) which shifts the vortex core slightly off the node lattice.
  This port uses the acoustic node lattice with the core placed exactly at
  the origin (the analytic Lamb-Oseen field is smooth and the difference is
  ~0.1 % of the domain), keeping the *physical* formulas/coefficients exact;
  a core-node r->0 limiting value replaces the 0/0 NaN of the reference.

Author decision (per 返稿工作区 D3 / A0 workflow): no pointwise comparison
against ``codematlab/*.mat`` is performed; only smoke validation is required.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np


# Number of acoustic periods averaged when deciding whether the field has
# reached a time-harmonic steady state (see ``_steady_stats``).
STEADY_N_PERIOD = 4

# Divergence guard. The physical fields here are O(1) (the hard source amplitude
# is 1), so any |p| above this is numerical blow-up rather than physics; it is far
# below float64 overflow, which is the point -- a diverging run stays "finite" for
# thousands of steps and would otherwise be reported as merely "not steady" (the
# 2026-09-11 calibration probe reached ~1e70 with ``isfinite`` still true).
DIVERGENCE_MAX_ABS_P = 1e8


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _mslice(a: int, b: int) -> slice:
    """Convert a MATLAB 1-based inclusive range ``a:b`` to a 0-based Python
    half-open slice (MATLAB indices ``a..b`` inclusive)."""
    return slice(a - 1, b)


def _lamb_oseen_vortex(x: np.ndarray, y: np.ndarray,
                       gamma: float, alpha: float, L: float) -> Tuple[np.ndarray, np.ndarray]:
    """Divergence-free Lamb-Oseen vortex velocity (vx, vy) on node meshes.

    vx = -Gamma*y*(1-exp(-alpha r^2/L^2))/(2*pi*r^2)
    vy = +Gamma*x*(1-exp(-alpha r^2/L^2))/(2*pi*r^2)
    with the finite limit alpha/L^2 used at the core node r=0.

    Equivalently v_theta = Gamma/(2*pi*r)*(1-exp(-alpha r^2/L^2)) with
    (vx, vy) = (-sin(theta), cos(theta)) * v_theta, i.e. the standard rotating
    Lamb-Oseen vortex.

    Deviation from the reference, and why
    -------------------------------------
    ``codematlab/initialize_vortex_2d.m:4-5`` writes ``vx1 = -Gamma*x.*(...)`` /
    ``vy1 = Gamma*y.*(...)`` -- the (x, y) components are transposed relative to the
    standard vortex. That field is ``v = c*term*(-x, +y)``: a **straining
    (stagnation-point) flow**, which is *compressible*
    (``div v = 2*c*term'*(y^2 - x^2) != 0``, magnitude comparable to the shear) and
    is therefore **not** a Lamb-Oseen vortex, despite the name.

    That compressibility is not a harmless detail: the advection terms in the
    acoustic update then act as a spurious energy source, and the solver grows
    without bound for any Ma > 0 (Ma = 0 is unaffected because v = 0). The growth is
    interior, at low wavenumber (domain scale) and insensitive to the PML -- all
    measured; see ``docs/A3_对流项稳定化设计.md`` §1.1. The transposed version is
    faithfully reproduced by the port, so this function is the **single** place that
    intentionally deviates from the reference.

    Falsifiable check (must hold after the fix, and did not before):
    the discrete divergence of ``v`` must tend to zero under dx refinement, whereas
    the transposed field keeps ``max|div v| / max|shear| ~ 1`` at every dx.
    """
    L2 = L * L
    r2 = x * x + y * y
    term = np.zeros_like(r2)
    ok = r2 > 0.0
    np.divide(1.0 - np.exp(-alpha * r2 / L2), r2, out=term, where=ok)
    term[~ok] = alpha / L2  # limit of (1-exp(-a r^2/L^2))/r^2 as r -> 0
    c = gamma / (2.0 * np.pi)
    vx = -c * y * term
    vy = c * x * term
    return vx, vy


def _steady_stats(energy_tail: np.ndarray, dt: float, f: float,
                  n_period: int = 4, tol: float = 0.02) -> Dict[str, Any]:
    """Assess steady state from phase-locked, integer-period averaged energy.

    The instantaneous air-region energy ``E(t) = mean(p_air**2)`` oscillates at
    ``2*omega``. Sampling it every ``monitor_every`` steps therefore aliases that
    oscillation, and a linear trend fit over the aliased samples cannot separate
    a genuine growth from the oscillation: a pure linear ramp always yields
    ``rel_growth = k/(N-k/2)`` regardless of how many steps are run. Averaging
    the *per-step* energy over an integer number of acoustic periods removes the
    ``2*omega`` component exactly, up to the residual phase error of the rounded
    period length (reported as ``phase_err``).

    Parameters
    ----------
    energy_tail : per-step ``mean(p_air**2)`` covering the final ``n_period``
        acoustic periods.
    dt, f : time step and source frequency, used to snap the period to whole steps.

    Returns ``steady_ok`` (all consecutive period means within ``tol``),
    ``rel_growth`` (largest consecutive relative change), ``energy_per_period``,
    ``steps_per_period`` and ``phase_err``.
    """
    n = int(len(energy_tail))
    spp = int(round(1.0 / (f * dt))) if (f > 0 and dt > 0) else 0
    base = {"method": "phase-locked integer-period mean", "n_samples": n,
            "steps_per_period": spp, "n_period": int(n_period)}
    # Finiteness is a precondition, not an afterthought. A diverging run can
    # produce a tail whose consecutive period means happen to agree within the
    # tolerance (e.g. while it is still in the early transient, or at machine
    # overflow), which would be reported as "steady". The calibration probe on
    # 2026-09-11 hit exactly that failure mode: it reported steady_ok at ~1.1-1.25
    # crossings while the field was diverging to ~1e70. See
    # docs/A_声学数据与任务设计.md §2.3.1 lesson 2.
    tail_arr = np.asarray(energy_tail, dtype=float)
    if tail_arr.size and not np.isfinite(tail_arr).all():
        return dict(base, steady_ok=False, rel_growth=float("nan"),
                    reason="energy tail contains non-finite values (field "
                           "diverged); steady_ok must not be trusted")
    if spp < 8:
        return dict(base, steady_ok=False, rel_growth=float("nan"),
                    reason="acoustic period shorter than 8 steps")
    need = spp * n_period
    if n < need:
        return dict(base, steady_ok=False, rel_growth=float("nan"),
                    reason="tail shorter than n_period acoustic periods")
    tail = tail_arr[-need:]
    per = tail.reshape(n_period, spp).mean(axis=1)
    rel = np.full(n_period - 1, np.nan)
    for j in range(n_period - 1):
        if abs(per[j]) > 1e-30:
            rel[j] = abs(per[j + 1] - per[j]) / abs(per[j])
    finite = np.isfinite(rel)
    rel_growth = float(np.max(rel[finite])) if finite.any() else float("nan")
    return dict(base,
                steady_ok=bool(finite.all() and np.all(rel <= tol)),
                rel_growth=rel_growth,
                phase_err=float(abs(1.0 / (f * dt) - spp) * f * dt),
                tol=float(tol),
                energy_per_period=[float(v) for v in per],
                rel_per_period=[float(v) for v in rel],
                mean_energy_last=float(per[-1]))


def fit_complex_amplitude(snapshots: np.ndarray, steps: np.ndarray,
                          dt: float, f: float) -> Dict[str, np.ndarray]:
    """Least-squares phasor extraction from phase-locked field snapshots.

    Fits ``p(t) ~ a*cos(omega t) + b*sin(omega t) + c + d*t`` per grid node and
    returns the complex amplitude ``a + i*b`` (convention: real part on cos) plus
    the residual linear-drift coefficient ``d`` (a drift that is not small
    relative to the amplitude means the snapshots were not taken in a
    time-harmonic regime).

    Parameters
    ----------
    snapshots : array ``[K, H, W]`` of phase-locked pressure snapshots.
    steps : array ``[K]`` of integer step indices at which they were taken.
    """
    snaps = np.asarray(snapshots, dtype=np.float64)
    k = snaps.shape[0]
    if k < 4:
        raise ValueError("need at least 4 snapshots for a 4-parameter fit")
    t = np.asarray(steps, dtype=np.float64) * float(dt)
    omega = 2.0 * np.pi * float(f)
    design = np.stack([np.cos(omega * t), np.sin(omega * t),
                       np.ones_like(t), t], axis=1)          # [K,4]
    flat = snaps.reshape(k, -1)                               # [K, H*W]
    coef, *_ = np.linalg.lstsq(design, flat, rcond=None)       # [4, H*W]
    hw = snaps.shape[1:]
    a = coef[0].reshape(hw)
    b = coef[1].reshape(hw)
    c = coef[2].reshape(hw)
    d = coef[3].reshape(hw)
    amp = np.sqrt(a * a + b * b)
    with np.errstate(divide="ignore", invalid="ignore"):
        drift_rel = np.where(amp > 1e-30, np.abs(d) * (1.0 / float(f)) / amp,
                             np.inf)
    return {"phasor_re": a, "phasor_im": b, "offset": c, "drift": d,
            "amplitude": amp, "drift_rel": drift_rel}


# --------------------------------------------------------------------------
# solver
# --------------------------------------------------------------------------
class FDTDAcoustic2D:
    """2-D convected-acoustics FDTD solver (RK4 / PML), MATLAB-scheme port.

    Parameters
    ----------
    f : float
        Source frequency in Hz (also stored as ``omega = 2*pi*f``).
    Ma : float
        Vortex Mach number; circulation Gamma = gamma_coef*Ma*pi*L*c0.
    c0 : float
        Adiabatic sound speed (m/s).
    rho0 : float
        Background density (kg/m^3).
    L : float
        Lamb-Oseen core-radius scale (m).
    alpha : float
        Lamb-Oseen profile exponent coefficient (default 1.256431).
    gamma_coef : float
        Circulation prefactor (default 2.8, as in the MATLAB file).
    dx, dy : float
        Cell size (m); default 0.125 m (801x801 nodes -> 100 m domain).
    courant_factor : float
        dt = courant_factor / (c0*sqrt(1/dx^2 + 1/dy^2)); default 0.1.
    n_pml : int
        PML cell count on each side (default 160).
    n_buffer : int
        Physical air-buffer cell count on each side (default 240).
    nx, ny : int, optional
        Total cell counts. Default 2*(n_pml + n_buffer) each. If given they
        must be even and >= 2*n_pml; the buffer count is then derived.
    sigma : float
        Constant PML damping coefficient (default 100, as reference).
    n_steps : int
        Number of RK4 steps in a full solve (default 8000).
    source_side : str
        'left' (reference default; injection column = pis, x ~ -30 m) or
        'right' (injection column = pie, x ~ +30 m, wave travels leftwards).
    source_y_frac : tuple(float, float)
        Fraction (start, stop) of the pjs..pje line that carries the source.
    """

    def __init__(self, f: float = 400.0, Ma: float = 0.5, c0: float = 340.0,
                 rho0: float = 1.0, L: float = 1.0, alpha: float = 1.256431,
                 gamma_coef: float = 2.8, dx: float = 0.125, dy: float = 0.125,
                 courant_factor: float = 0.1, n_pml: int = 160,
                 n_buffer: int = 240, nx: Optional[int] = None,
                 ny: Optional[int] = None, sigma: float = 100.0,
                 n_steps: int = 8000, source_side: str = "left",
                 source_y_frac: Tuple[float, float] = (0.0, 1.0)) -> None:

        # --- physical / numerical parameters ----------------------------
        self.f = float(f)
        self.omega = 2.0 * np.pi * self.f
        self.Ma = float(Ma)
        self.c0 = float(c0)
        self.rho0 = float(rho0)
        self.L = float(L)
        self.alpha = float(alpha)
        self.gamma_coef = float(gamma_coef)
        self.Gamma = self.gamma_coef * self.Ma * np.pi * self.L * self.c0
        self.dx = float(dx)
        self.dy = float(dy)
        self.courant_factor = float(courant_factor)
        self.n_pml = int(n_pml)
        self.sigma = float(sigma)
        self.n_steps = int(n_steps)
        if source_side not in ("left", "right"):
            raise ValueError("source_side must be 'left' or 'right'")
        self.source_side = source_side
        self.source_y_frac = tuple(source_y_frac)

        # grid size (cells)
        base = 2 * (self.n_pml + int(n_buffer))
        self.nx = base if nx is None else int(nx)
        self.ny = base if ny is None else int(ny)
        for nn, name in ((self.nx, "nx"), (self.ny, "ny")):
            if nn % 2 != 0 or nn < 2 * self.n_pml:
                raise ValueError(f"{name} must be even and >= 2*n_pml")
        self.nxp1 = self.nx + 1
        self.nyp1 = self.ny + 1

        # dt from the CFL-style formula of the MATLAB file
        self.dt = (self.courant_factor
                   / (self.c0 * np.sqrt(1.0 / self.dx**2 + 1.0 / self.dy**2)))
        self.t_final = self.n_steps * self.dt

        # physical-air (air buffer) index ranges, MATLAB 1-based
        self.pis = self.n_pml + 1
        self.pie = self.nx - self.n_pml + 1
        self.pjs = self.n_pml + 1
        self.pje = self.ny - self.n_pml + 1

        # node coordinates (0-based: node i sits at min_x + i*dx)
        self.min_x = -(self.nx / 2.0) * self.dx
        self.max_x = +(self.nx / 2.0) * self.dx
        self.min_y = -(self.ny / 2.0) * self.dy
        self.max_y = +(self.ny / 2.0) * self.dy
        self.x_node = self.min_x + np.arange(self.nxp1) * self.dx
        self.y_node = self.min_y + np.arange(self.nyp1) * self.dy

        # --- allocate fields (names follow the MATLAB files) -------------
        shape = (self.nxp1, self.nyp1)
        z = np.zeros
        # total fields
        self.p, self.p1, self.p2, self.p3 = z(shape), z(shape), z(shape), z(shape)
        self.fp1, self.fp2, self.fp3, self.fp4 = z(shape), z(shape), z(shape), z(shape)
        self.wx, self.wx1, self.wx2, self.wx3 = z(shape), z(shape), z(shape), z(shape)
        self.fx1, self.fx2, self.fx3, self.fx4 = z(shape), z(shape), z(shape), z(shape)
        self.wy, self.wy1, self.wy2, self.wy3 = z(shape), z(shape), z(shape), z(shape)
        self.fy1, self.fy2, self.fy3, self.fy4 = z(shape), z(shape), z(shape), z(shape)
        # incident fields (capitalised names as in the reference)
        self.P, self.P1, self.P2, self.P3 = z(shape), z(shape), z(shape), z(shape)
        self.Fp1, self.Fp2, self.Fp3, self.Fp4 = z(shape), z(shape), z(shape), z(shape)
        self.Wx, self.Wx1, self.Wx2, self.Wx3 = z(shape), z(shape), z(shape), z(shape)
        self.Fx1, self.Fx2, self.Fx3, self.Fx4 = z(shape), z(shape), z(shape), z(shape)
        self.Wy, self.Wy1, self.Wy2, self.Wy3 = z(shape), z(shape), z(shape), z(shape)
        self.Fy1, self.Fy2, self.Fy3, self.Fy4 = z(shape), z(shape), z(shape), z(shape)

        # medium coefficients
        self.b = (1.0 / self.rho0) * np.ones(shape)
        self.kap = (self.rho0 * self.c0 * self.c0) * np.ones(shape)

        # Lamb-Oseen background flow (vx, vy) -- time invariant
        X, Y = np.meshgrid(self.x_node, self.y_node, indexing="ij")
        self.vx, self.vy = _lamb_oseen_vortex(X, Y, self.Gamma, self.alpha, self.L)

        # --- precomputed 0-based slices (MATLAB 1-based ranges) ----------
        # pressure interior update range 2:nx, 2:ny
        self.s_px = _mslice(2, self.nx)
        self.s_py = _mslice(2, self.ny)
        # physical-air region pis:pie, pjs:pje (used by velocity updates)
        self.s_ax = _mslice(self.pis, self.pie)
        self.s_ay = _mslice(self.pjs, self.pje)
        # neighbours along each axis inside the air region
        self.s_axp = _mslice(self.pis + 1, self.pie + 1)   # i+1 (stencil)
        self.s_axm = _mslice(self.pis - 1, self.pie - 1)   # i-1
        self.s_ayp = _mslice(self.pjs + 1, self.pje + 1)   # j+1
        self.s_aym = _mslice(self.pjs - 1, self.pje - 1)   # j-1
        # PML x bands (left / right) and full-height y interior
        self.s_lx = _mslice(2, self.pis - 1)               # 2:pis-1
        self.s_lxp = _mslice(3, self.pis)                  # 3:pis
        self.s_lxm = _mslice(1, self.pis - 2)              # 1:pis-2
        self.s_rx = _mslice(self.pie + 1, self.nx - 1)     # pie+1:nx-1 (wx)
        self.s_rxw = _mslice(self.pie + 1, self.nx)        # pie+1:nx   (wy)
        self.s_rxp = _mslice(self.pie + 2, self.nx)        # pie+2:nx
        self.s_rxmm = _mslice(self.pie, self.nx - 2)       # pie:nx-2   (wx)
        self.s_rxwm = _mslice(self.pie, self.nx - 1)       # pie:nx-1   (wy)
        self.s_rxwp = _mslice(self.pie + 2, self.nxp1)     # pie+2:nxp1 (wy)
        # PML y bands (below / above) spanning the air x-range
        self.s_by = _mslice(2, self.pjs - 1)               # 2:pjs-1
        self.s_byp = _mslice(3, self.pjs)                  # 3:pjs
        self.s_bym = _mslice(1, self.pjs - 2)              # 1:pjs-2
        self.s_at = _mslice(self.pje + 1, self.ny)         # pje+1:ny (wx)
        self.s_atp = _mslice(self.pje + 2, self.nyp1)      # pje+2:nyp1 (wx)
        self.s_atm = _mslice(self.pje, self.ny - 1)        # pje:ny-1  (wx)
        self.s_at1 = _mslice(self.pje + 1, self.ny - 1)    # pje+1:ny-1 (wy)
        self.s_at1p = _mslice(self.pje + 2, self.ny)       # pje+2:ny   (wy)
        self.s_at1pp = _mslice(self.pje + 2, self.ny)      # pje+2:ny   (wy d/dy+)
        self.s_at1m = _mslice(self.pje, self.ny - 2)       # pje:ny-2   (wy d/dy-)
        # PML full-height y ranges used by left/right velocity bands
        self.s_ly = _mslice(2, self.ny)                    # 2:ny     (wx bands)
        self.s_lyp = _mslice(3, self.nyp1)                 # 3:nyp1
        self.s_lym = _mslice(1, self.ny - 1)               # 1:ny-1
        self.s_ly1 = _mslice(2, self.ny - 1)               # 2:ny-1   (wy bands)
        self.s_ly1p = _mslice(3, self.ny)                  # 3:ny
        self.s_ly1m = _mslice(1, self.ny - 2)              # 1:ny-2

        # source injection column (0-based) and row range
        if self.source_side == "left":
            self.src_col = self.pis - 1
        else:
            self.src_col = self.pie - 1
        j0 = self.pjs - 1 + int(self.source_y_frac[0] * (self.pje - self.pjs))
        j1 = self.pjs - 1 + int(self.source_y_frac[1] * (self.pje - self.pjs))
        j0 = min(max(j0, self.pjs - 1), self.pje - 1)
        j1 = min(max(j1, self.pjs - 1), self.pje - 1)
        if j1 < j0:
            j0, j1 = j1, j0
        self.src_rows = np.arange(j0, j1 + 1)

    # ------------------------------------------------------------------ #
    # pressure -- total field  (update_pressure_fields_2d.m)              #
    # ------------------------------------------------------------------ #
    def _update_pressure_fields(self) -> None:
        p, fp1, fp2, fp3, fp4 = self.p, self.fp1, self.fp2, self.fp3, self.fp4
        p1, p2, p3 = self.p1, self.p2, self.p3
        wx, wy, vx, vy, kap = self.wx, self.wy, self.vx, self.vy, self.kap
        ix, iy = self.s_px, self.s_py
        xp, xm = _mslice(3, self.nxp1), _mslice(1, self.nx - 1)
        yp, ym = _mslice(3, self.nyp1), _mslice(1, self.ny - 1)
        dt, dx, dy = self.dt, self.dx, self.dy
        vxij = vx[ix, iy] + vx[xm, iy]
        vyij = vy[ix, iy] + vy[ix, ym]
        kapij = kap[ix, iy]
        divw = (wx[ix, iy] - wx[xm, iy]) / dx + (wy[ix, iy] - wy[ix, ym]) / dy

        def rhs(pst: np.ndarray) -> np.ndarray:
            # fp(2:nx,2:ny) -- lines 5-9 of the MATLAB file
            return (-vxij * (pst[xp, iy] - pst[xm, iy]) / (4.0 * dx)
                    - vyij * (pst[ix, yp] - pst[ix, ym]) / (4.0 * dy)
                    - kapij * divw)

        fp1[ix, iy] = rhs(p)
        p1[ix, iy] = p[ix, iy] + dt * fp1[ix, iy] / 2.0
        fp2[ix, iy] = rhs(p1)
        p2[ix, iy] = p[ix, iy] + dt * fp2[ix, iy] / 2.0
        fp3[ix, iy] = rhs(p2)
        p3[ix, iy] = p[ix, iy] + dt * fp3[ix, iy]
        fp4[ix, iy] = rhs(p3)
        p[ix, iy] += dt * (fp1[ix, iy] + 2.0 * fp2[ix, iy]
                           + 2.0 * fp3[ix, iy] + fp4[ix, iy]) / 6.0

    # ------------------------------------------------------------------ #
    # velocity x -- total field, air region (update_velocity_x_fields_2d.m)
    # ------------------------------------------------------------------ #
    def _update_velocity_x_fields(self) -> None:
        wx, wy, p = self.wx, self.wy, self.p
        vx, vy, b = self.vx, self.vy, self.b
        wx1, wx2, wx3 = self.wx1, self.wx2, self.wx3
        fx1, fx2, fx3, fx4 = self.fx1, self.fx2, self.fx3, self.fx4
        ax, ay = self.s_ax, self.s_ay
        axp, axm = self.s_axp, self.s_axm
        ayp, aym = self.s_ayp, self.s_aym
        dt, dx, dy = self.dt, self.dx, self.dy
        wy_avg = (wy[axp, ay] + wy[ax, ay] + wy[axp, aym] + wy[ax, aym])
        vy_avg = (vy[axp, ay] + vy[ax, ay] + vy[axp, aym] + vy[ax, aym])
        dvx_dx = (vx[axp, ay] - vx[axm, ay]) / (2.0 * dx)
        dvx_dy = (vx[ax, ayp] - vx[ax, aym]) / (8.0 * dy)
        b_p = b[axp, ay] + b[ax, ay]
        p_grad = (p[axp, ay] - p[ax, ay]) / (2.0 * dx)

        def rhs(ws: np.ndarray) -> np.ndarray:
            # fx(air) -- lines 5-11 of the MATLAB file
            return (-ws[ax, ay] * dvx_dx
                    - wy_avg * dvx_dy
                    - vx[ax, ay] * (ws[axp, ay] - ws[axm, ay]) / (2.0 * dx)
                    - vy_avg * (ws[ax, ayp] - ws[ax, aym]) / (8.0 * dy)
                    - b_p * p_grad)

        fx1[ax, ay] = rhs(wx)
        wx1[ax, ay] = wx[ax, ay] + dt * fx1[ax, ay] / 2.0
        fx2[ax, ay] = rhs(wx1)
        wx2[ax, ay] = wx[ax, ay] + dt * fx2[ax, ay] / 2.0
        fx3[ax, ay] = rhs(wx2)
        wx3[ax, ay] = wx[ax, ay] + dt * fx3[ax, ay]
        fx4[ax, ay] = rhs(wx3)
        wx[ax, ay] += dt * (fx1[ax, ay] + 2.0 * fx2[ax, ay]
                            + 2.0 * fx3[ax, ay] + fx4[ax, ay]) / 6.0

    # ------------------------------------------------------------------ #
    # velocity y -- total field, air region (update_velocity_y_fields_2d.m)
    # ------------------------------------------------------------------ #
    def _update_velocity_y_fields(self) -> None:
        wx, wy, p = self.wx, self.wy, self.p
        vx, vy, b = self.vx, self.vy, self.b
        wy1, wy2, wy3 = self.wy1, self.wy2, self.wy3
        fy1, fy2, fy3, fy4 = self.fy1, self.fy2, self.fy3, self.fy4
        ax, ay = self.s_ax, self.s_ay
        axp, axm = self.s_axp, self.s_axm
        ayp, aym = self.s_ayp, self.s_aym
        dt, dx, dy = self.dt, self.dx, self.dy
        wx_avg = (wx[ax, ayp] + wx[ax, ay] + wx[axm, ayp] + wx[axm, ay])
        vx_avg = (vx[ax, ayp] + vx[ax, ay] + vx[axm, ayp] + vx[axm, ay])
        dvy_dy = (vy[ax, ayp] - vy[ax, aym]) / (2.0 * dy)
        dvy_dx = (vy[axp, ay] - vy[axm, ay]) / (8.0 * dx)
        b_p = b[ax, ayp] + b[ax, ay]
        p_grad = (p[ax, ayp] - p[ax, ay]) / (2.0 * dy)

        def rhs(ws: np.ndarray) -> np.ndarray:
            # fy(air) -- lines 5-11 of the MATLAB file
            return (-ws[ax, ay] * dvy_dy
                    - wx_avg * dvy_dx
                    - vy[ax, ay] * (ws[ax, ayp] - ws[ax, aym]) / (2.0 * dy)
                    - vx_avg * (ws[axp, ay] - ws[axm, ay]) / (8.0 * dx)
                    - b_p * p_grad)

        fy1[ax, ay] = rhs(wy)
        wy1[ax, ay] = wy[ax, ay] + dt * fy1[ax, ay] / 2.0
        fy2[ax, ay] = rhs(wy1)
        wy2[ax, ay] = wy[ax, ay] + dt * fy2[ax, ay] / 2.0
        fy3[ax, ay] = rhs(wy2)
        wy3[ax, ay] = wy[ax, ay] + dt * fy3[ax, ay]
        fy4[ax, ay] = rhs(wy3)
        wy[ax, ay] += dt * (fy1[ax, ay] + 2.0 * fy2[ax, ay]
                            + 2.0 * fy3[ax, ay] + fy4[ax, ay]) / 6.0

    # ------------------------------------------------------------------ #
    # velocity x -- PML bands (update_velocity_x_fields_for_PML_2d.m)     #
    # ------------------------------------------------------------------ #
    def _update_velocity_x_fields_for_pml(self) -> None:
        wx, p = self.wx, self.p
        vx, vy, b = self.vx, self.vy, self.b
        wx1, wx2, wx3 = self.wx1, self.wx2, self.wx3
        fx1, fx2, fx3, fx4 = self.fx1, self.fx2, self.fx3, self.fx4
        dt, dx, dy = self.dt, self.dx, self.dy
        sig = self.sigma

        # ---- left band: region (2:pis-1, 2:ny), lines 5-28 ----------
        lx, lxp, lxm = self.s_lx, self.s_lxp, self.s_lxm
        ly, lyp, lym = self.s_ly, self.s_lyp, self.s_lym
        bl = b[lxp, ly] + b[lx, ly]
        pl = (p[lxp, ly] - p[lx, ly]) / (2.0 * dx)
        vy_avg = (vy[lxp, ly] + vy[lx, ly] + vy[lxp, lym] + vy[lx, lym])

        def f_left(ws_: np.ndarray) -> np.ndarray:
            return (-sig * ws_[lx, ly] * bl / 2.0
                    - bl * pl
                    - (ws_[lxp, ly] - ws_[lxm, ly]) * vx[lx, ly] / (2.0 * dx)
                    - vy_avg * (ws_[lx, lyp] - ws_[lx, lym]) / (8.0 * dy))

        fx1[lx, ly] = f_left(wx)
        wx1[lx, ly] = wx[lx, ly] + dt * fx1[lx, ly] / 2.0
        fx2[lx, ly] = f_left(wx1)
        wx2[lx, ly] = wx[lx, ly] + dt * fx2[lx, ly] / 2.0
        fx3[lx, ly] = f_left(wx2)
        wx3[lx, ly] = wx[lx, ly] + dt * fx3[lx, ly]
        fx4[lx, ly] = f_left(wx3)
        # reference final line 28: fx1+2*fx1+2*fx1+fx1 == 6*fx1 (typo kept)
        wx[lx, ly] += dt * fx1[lx, ly]

        # ---- right band: region (pie+1:nx-1, 2:ny), lines 31-54 -----
        rx = self.s_rx
        rxp, rxmm = self.s_rxp, self.s_rxmm
        br = b[rxp, ly] + b[rx, ly]
        pr = (p[rxp, ly] - p[rx, ly]) / (2.0 * dx)
        vy_avg_r = (vy[rxp, ly] + vy[rx, ly] + vy[rxp, lym] + vy[rx, lym])

        def f_right(ws_: np.ndarray) -> np.ndarray:
            return (-sig * ws_[rx, ly] * br / 2.0
                    - br * pr
                    - (ws_[rxp, ly] - ws_[rxmm, ly]) * vx[rx, ly] / (2.0 * dx)
                    - vy_avg_r * (ws_[rx, lyp] - ws_[rx, lym]) / (8.0 * dy))

        fx1[rx, ly] = f_right(wx)
        wx1[rx, ly] = wx[rx, ly] + dt * fx1[rx, ly] / 2.0
        fx2[rx, ly] = f_right(wx1)
        wx2[rx, ly] = wx[rx, ly] + dt * fx2[rx, ly] / 2.0
        fx3[rx, ly] = f_right(wx2)
        wx3[rx, ly] = wx[rx, ly] + dt * fx3[rx, ly]
        fx4[rx, ly] = f_right(wx3)
        wx[rx, ly] += dt * (fx1[rx, ly] + 2.0 * fx2[rx, ly]
                            + 2.0 * fx3[rx, ly] + fx4[rx, ly]) / 6.0

        # ---- below band: region (pis:pie, 2:pjs-1), lines 57-80 -----
        ax = self.s_ax
        axp, axm = self.s_axp, self.s_axm
        by, byp, bym = self.s_by, self.s_byp, self.s_bym
        bb = b[axp, by] + b[ax, by]
        pb = (p[axp, by] - p[ax, by]) / (2.0 * dx)
        # note: reference line 60 uses 2:pis-1 in y == by for the square
        # domain (pis == pjs); reproduced as by here.
        vy_avg_b = (vy[axp, by] + vy[ax, by] + vy[axp, bym] + vy[ax, bym])

        def f_below(ws_: np.ndarray) -> np.ndarray:
            return (-sig * ws_[ax, by] * bb / 2.0
                    - bb * pb
                    - (ws_[axp, by] - ws_[axm, by]) * vx[ax, by] / (2.0 * dx)
                    - vy_avg_b * (ws_[ax, byp] - ws_[ax, bym]) / (8.0 * dy))

        fx1[ax, by] = f_below(wx)
        wx1[ax, by] = wx[ax, by] + dt * fx1[ax, by] / 2.0
        fx2[ax, by] = f_below(wx1)
        wx2[ax, by] = wx[ax, by] + dt * fx2[ax, by] / 2.0
        fx3[ax, by] = f_below(wx2)
        wx3[ax, by] = wx[ax, by] + dt * fx3[ax, by]
        fx4[ax, by] = f_below(wx3)
        # reference final line 80: fx1+2*fx2+2*fx1+fx1 (typo kept)
        wx[ax, by] += dt * (fx1[ax, by] + 2.0 * fx2[ax, by]
                            + 2.0 * fx1[ax, by] + fx1[ax, by]) / 6.0

        # ---- above band: region (pis:pie, pje+1:ny), lines 82-105 ----
        at, atp, atm = self.s_at, self.s_atp, self.s_atm
        bt = b[axp, at] + b[ax, at]
        pt = (p[axp, at] - p[ax, at]) / (2.0 * dx)
        vy_avg_t = (vy[axp, at] + vy[ax, at] + vy[axp, atm] + vy[ax, atm])

        def f_above(ws_: np.ndarray) -> np.ndarray:
            return (-sig * ws_[ax, at] * bt / 2.0
                    - bt * pt
                    - (ws_[axp, at] - ws_[axm, at]) * vx[ax, at] / (2.0 * dx)
                    - vy_avg_t * (ws_[ax, atp] - ws_[ax, atm]) / (8.0 * dy))

        fx1[ax, at] = f_above(wx)
        wx1[ax, at] = wx[ax, at] + dt * fx1[ax, at] / 2.0
        fx2[ax, at] = f_above(wx1)
        wx2[ax, at] = wx[ax, at] + dt * fx2[ax, at] / 2.0
        fx3[ax, at] = f_above(wx2)
        wx3[ax, at] = wx[ax, at] + dt * fx3[ax, at]
        fx4[ax, at] = f_above(wx3)
        wx[ax, at] += dt * (fx1[ax, at] + 2.0 * fx2[ax, at]
                            + 2.0 * fx3[ax, at] + fx4[ax, at]) / 6.0

    # ------------------------------------------------------------------ #
    # velocity y -- PML bands (update_velocity_y_fields_for_PML_2d.m)     #
    # ------------------------------------------------------------------ #
    def _update_velocity_y_fields_for_pml(self) -> None:
        wx, wy, p = self.wx, self.wy, self.p
        vx, vy, b = self.vx, self.vy, self.b
        wy1, wy2, wy3 = self.wy1, self.wy2, self.wy3
        fy1, fy2, fy3, fy4 = self.fy1, self.fy2, self.fy3, self.fy4
        dt, dx, dy = self.dt, self.dx, self.dy
        sig = self.sigma

        # ---- left band: region (2:pis-1, 2:ny-1), lines 5-28 ---------
        lx, lxp, lxm = self.s_lx, self.s_lxp, self.s_lxm
        ly1, ly1p, ly1m = self.s_ly1, self.s_ly1p, self.s_ly1m
        bl = b[lx, ly1p] + b[lx, ly1]
        pl = (p[lx, ly1p] - p[lx, ly1]) / (2.0 * dy)
        vx_avg = (vx[lx, ly1p] + vx[lx, ly1] + vx[lxm, ly1p] + vx[lxm, ly1])

        def f_left(ws_: np.ndarray) -> np.ndarray:
            return (-sig * ws_[lx, ly1] * bl / 2.0
                    - bl * pl
                    - vx_avg * (ws_[lxp, ly1] - ws_[lxm, ly1]) / (2.0 * dx)
                    - vy[lx, ly1] * (ws_[lx, ly1p] - ws_[lx, ly1m]) / (2.0 * dy))

        fy1[lx, ly1] = f_left(wy)
        wy1[lx, ly1] = wy[lx, ly1] + dt * fy1[lx, ly1] / 2.0
        fy2[lx, ly1] = f_left(wy1)
        wy2[lx, ly1] = wy[lx, ly1] + dt * fy2[lx, ly1] / 2.0
        fy3[lx, ly1] = f_left(wy2)
        wy3[lx, ly1] = wy[lx, ly1] + dt * fy3[lx, ly1]
        fy4[lx, ly1] = f_left(wy3)
        wy[lx, ly1] += dt * (fy1[lx, ly1] + 2.0 * fy2[lx, ly1]
                             + 2.0 * fy3[lx, ly1] + fy4[lx, ly1]) / 6.0

        # ---- right band: region (pie+1:nx, 2:ny-1), lines 30-53 ------
        rxw = self.s_rxw
        rxwp, rxwm = self.s_rxwp, self.s_rxwm
        br = b[rxw, ly1p] + b[rxw, ly1]
        pr = (p[rxw, ly1p] - p[rxw, ly1]) / (2.0 * dy)
        vx_avg_r = (vx[rxw, ly1p] + vx[rxw, ly1] + vx[rxwm, ly1p] + vx[rxwm, ly1])

        def f_right(ws_: np.ndarray) -> np.ndarray:
            return (-sig * ws_[rxw, ly1] * br / 2.0
                    - br * pr
                    - vx_avg_r * (ws_[rxwp, ly1] - ws_[rxwm, ly1]) / (8.0 * dx)
                    - vy[rxw, ly1] * (ws_[rxw, ly1p] - ws_[rxw, ly1m]) / (2.0 * dy))

        fy1[rxw, ly1] = f_right(wy)
        wy1[rxw, ly1] = wy[rxw, ly1] + dt * fy1[rxw, ly1] / 2.0
        fy2[rxw, ly1] = f_right(wy1)
        # reference line 41: wy2 advanced with fy1 (typo kept)
        wy2[rxw, ly1] = wy[rxw, ly1] + dt * fy1[rxw, ly1] / 2.0
        fy3[rxw, ly1] = f_right(wy2)
        wy3[rxw, ly1] = wy[rxw, ly1] + dt * fy3[rxw, ly1]
        fy4[rxw, ly1] = f_right(wy3)
        wy[rxw, ly1] += dt * (fy1[rxw, ly1] + 2.0 * fy2[rxw, ly1]
                              + 2.0 * fy3[rxw, ly1] + fy4[rxw, ly1]) / 6.0

        # ---- below band: region (pis:pie, 2:pjs-1), lines 55-78 ------
        ax, axp, axm = self.s_ax, self.s_axp, self.s_axm
        by, byp, bym = self.s_by, self.s_byp, self.s_bym
        bb = b[ax, byp] + b[ax, by]
        pb = (p[ax, byp] - p[ax, by]) / (2.0 * dy)
        vx_avg_b = (vx[ax, byp] + vx[ax, by] + vx[axm, byp] + vx[axm, by])

        def f_below(ws_: np.ndarray) -> np.ndarray:
            return (-sig * ws_[ax, by] * bb / 2.0
                    - bb * pb
                    - vx_avg_b * (ws_[axp, by] - ws_[axm, by]) / (8.0 * dx)
                    - vy[ax, by] * (ws_[ax, byp] - ws_[ax, bym]) / (2.0 * dy))

        fy1[ax, by] = f_below(wy)
        wy1[ax, by] = wy[ax, by] + dt * fy1[ax, by] / 2.0
        fy2[ax, by] = f_below(wy1)
        wy2[ax, by] = wy[ax, by] + dt * fy2[ax, by] / 2.0
        fy3[ax, by] = f_below(wy2)
        wy3[ax, by] = wy[ax, by] + dt * fy3[ax, by]
        fy4[ax, by] = f_below(wy3)
        wy[ax, by] += dt * (fy1[ax, by] + 2.0 * fy2[ax, by]
                            + 2.0 * fy3[ax, by] + fy4[ax, by]) / 6.0

        # ---- above band: region (pis:pie, pje+1:ny-1), lines 80-103 --
        at1, at1p, at1m = self.s_at1, self.s_at1p, self.s_at1m
        bt = b[ax, at1p] + b[ax, at1]
        pt = (p[ax, at1p] - p[ax, at1]) / (2.0 * dy)
        vx_avg_t = (vx[ax, at1p] + vx[ax, at1] + vx[axm, at1p] + vx[axm, at1])

        def f_above(ws_: np.ndarray) -> np.ndarray:
            return (-sig * ws_[ax, at1] * bt / 2.0
                    - bt * pt
                    - vx_avg_t * (ws_[axp, at1] - ws_[axm, at1]) / (8.0 * dx)
                    - vy[ax, at1] * (ws_[ax, at1p] - ws_[ax, at1m]) / (2.0 * dy))

        fy1[ax, at1] = f_above(wy)
        wy1[ax, at1] = wy[ax, at1] + dt * fy1[ax, at1] / 2.0
        fy2[ax, at1] = f_above(wy1)
        wy2[ax, at1] = wy[ax, at1] + dt * fy2[ax, at1] / 2.0
        fy3[ax, at1] = f_above(wy2)
        wy3[ax, at1] = wy[ax, at1] + dt * fy3[ax, at1]
        fy4[ax, at1] = f_above(wy3)
        wy[ax, at1] += dt * (fy1[ax, at1] + 2.0 * fy2[ax, at1]
                             + 2.0 * fy3[ax, at1] + fy4[ax, at1]) / 6.0

    # ------------------------------------------------------------------ #
    # incident pressure  (update_P1ressure_fields_2d.m)                   #
    # ------------------------------------------------------------------ #
    def _update_incident_pressure_fields(self) -> None:
        P, Fp1, Fp2, Fp3, Fp4 = self.P, self.Fp1, self.Fp2, self.Fp3, self.Fp4
        P1, P2, P3 = self.P1, self.P2, self.P3
        Wx, Wy, kap = self.Wx, self.Wy, self.kap
        ix, iy = self.s_px, self.s_py
        xm = _mslice(1, self.nx - 1)
        ym = _mslice(1, self.ny - 1)
        dt, dx, dy = self.dt, self.dx, self.dy
        divW = ((Wx[ix, iy] - Wx[xm, iy]) / dx
                + (Wy[ix, iy] - Wy[ix, ym]) / dy)
        kapij = kap[ix, iy]

        def rhs(_pst: np.ndarray) -> np.ndarray:
            # Fp(2:nx,2:ny) -- lines 5-11 of the MATLAB file
            return -kapij * divW

        Fp1[ix, iy] = rhs(P)
        P1[ix, iy] = P[ix, iy] + dt * Fp1[ix, iy] / 2.0
        Fp2[ix, iy] = rhs(P1)
        P2[ix, iy] = P[ix, iy] + dt * Fp2[ix, iy] / 2.0
        Fp3[ix, iy] = rhs(P2)
        P3[ix, iy] = P[ix, iy] + dt * Fp3[ix, iy]
        Fp4[ix, iy] = rhs(P3)
        P[ix, iy] += dt * (Fp1[ix, iy] + 2.0 * Fp2[ix, iy]
                           + 2.0 * Fp3[ix, iy] + Fp4[ix, iy]) / 6.0

    # ------------------------------------------------------------------ #
    # incident velocity x, air region (update_V1elocity_x_fields_2d.m)    #
    # ------------------------------------------------------------------ #
    def _update_incident_velocity_x_fields(self) -> None:
        Wx, P, b = self.Wx, self.P, self.b
        Wx1, Wx2, Wx3 = self.Wx1, self.Wx2, self.Wx3
        Fx1, Fx2, Fx3, Fx4 = self.Fx1, self.Fx2, self.Fx3, self.Fx4
        ax, ay = self.s_ax, self.s_ay
        axp = self.s_axp
        dt, dx = self.dt, self.dx
        b_p = b[axp, ay] + b[ax, ay]
        p_grad = (P[axp, ay] - P[ax, ay]) / (2.0 * dx)

        def rhs(_ws: np.ndarray) -> np.ndarray:
            return -b_p * p_grad

        Fx1[ax, ay] = rhs(Wx)
        Wx1[ax, ay] = Wx[ax, ay] + dt * Fx1[ax, ay] / 2.0
        Fx2[ax, ay] = rhs(Wx1)
        Wx2[ax, ay] = Wx[ax, ay] + dt * Fx2[ax, ay] / 2.0
        Fx3[ax, ay] = rhs(Wx2)
        Wx3[ax, ay] = Wx[ax, ay] + dt * Fx3[ax, ay]
        Fx4[ax, ay] = rhs(Wx3)
        Wx[ax, ay] += dt * (Fx1[ax, ay] + 2.0 * Fx2[ax, ay]
                            + 2.0 * Fx3[ax, ay] + Fx4[ax, ay]) / 6.0

    # ------------------------------------------------------------------ #
    # incident velocity y, air region (update_V1elocity_y_fields_2d.m)    #
    # ------------------------------------------------------------------ #
    def _update_incident_velocity_y_fields(self) -> None:
        Wy, P, b = self.Wy, self.P, self.b
        Wy1, Wy2, Wy3 = self.Wy1, self.Wy2, self.Wy3
        Fy1, Fy2, Fy3, Fy4 = self.Fy1, self.Fy2, self.Fy3, self.Fy4
        ax, ay = self.s_ax, self.s_ay
        ayp = self.s_ayp
        dt, dy = self.dt, self.dy
        b_p = b[ax, ayp] + b[ax, ay]
        p_grad = (P[ax, ayp] - P[ax, ay]) / (2.0 * dy)

        def rhs(_ws: np.ndarray) -> np.ndarray:
            return -b_p * p_grad

        Fy1[ax, ay] = rhs(Wy)
        Wy1[ax, ay] = Wy[ax, ay] + dt * Fy1[ax, ay] / 2.0
        Fy2[ax, ay] = rhs(Wy1)
        Wy2[ax, ay] = Wy[ax, ay] + dt * Fy2[ax, ay] / 2.0
        Fy3[ax, ay] = rhs(Wy2)
        Wy3[ax, ay] = Wy[ax, ay] + dt * Fy3[ax, ay]
        Fy4[ax, ay] = rhs(Wy3)
        Wy[ax, ay] += dt * (Fy1[ax, ay] + 2.0 * Fy2[ax, ay]
                            + 2.0 * Fy3[ax, ay] + Fy4[ax, ay]) / 6.0

    # ------------------------------------------------------------------ #
    # incident velocity x -- PML (update_V1elocity_x_fields_for_PML_2d.m) #
    # ------------------------------------------------------------------ #
    def _update_incident_velocity_x_fields_for_pml(self) -> None:
        Wx, P, b = self.Wx, self.P, self.b
        Wx1, Wx2, Wx3 = self.Wx1, self.Wx2, self.Wx3
        Fx1, Fx2, Fx3, Fx4 = self.Fx1, self.Fx2, self.Fx3, self.Fx4
        dt, dx = self.dt, self.dx
        sig = self.sigma

        # ---- left band (2:pis-1, 2:ny), lines 5-16 ------------------
        lx, lxp = self.s_lx, self.s_lxp
        ly = self.s_ly
        bl = b[lxp, ly] + b[lx, ly]
        pl = (P[lxp, ly] - P[lx, ly]) / (2.0 * dx)

        def f_left(ws_: np.ndarray) -> np.ndarray:
            return -sig * ws_[lx, ly] * bl / 2.0 - bl * pl

        Fx1[lx, ly] = f_left(Wx)
        Wx1[lx, ly] = Wx[lx, ly] + dt * Fx1[lx, ly] / 2.0
        Fx2[lx, ly] = f_left(Wx1)
        Wx2[lx, ly] = Wx[lx, ly] + dt * Fx2[lx, ly] / 2.0
        Fx3[lx, ly] = f_left(Wx2)
        Wx3[lx, ly] = Wx[lx, ly] + dt * Fx3[lx, ly]
        Fx4[lx, ly] = f_left(Wx3)
        # reference line 16 typo: all Fx1 -> dt*Fx1
        Wx[lx, ly] += dt * Fx1[lx, ly]

        # ---- right band (pie+1:nx-1, 2:ny), lines 19-30 -------------
        rx, rxp = self.s_rx, self.s_rxp
        br = b[rxp, ly] + b[rx, ly]
        pr = (P[rxp, ly] - P[rx, ly]) / (2.0 * dx)

        def f_right(ws_: np.ndarray) -> np.ndarray:
            return -sig * ws_[rx, ly] * br / 2.0 - br * pr

        Fx1[rx, ly] = f_right(Wx)
        Wx1[rx, ly] = Wx[rx, ly] + dt * Fx1[rx, ly] / 2.0
        Fx2[rx, ly] = f_right(Wx1)
        Wx2[rx, ly] = Wx[rx, ly] + dt * Fx2[rx, ly] / 2.0
        Fx3[rx, ly] = f_right(Wx2)
        Wx3[rx, ly] = Wx[rx, ly] + dt * Fx3[rx, ly]
        Fx4[rx, ly] = f_right(Wx3)
        Wx[rx, ly] += dt * (Fx1[rx, ly] + 2.0 * Fx2[rx, ly]
                            + 2.0 * Fx3[rx, ly] + Fx4[rx, ly]) / 6.0

        # ---- below band (pis:pie, 2:pjs-1), lines 33-44 -------------
        ax, axp = self.s_ax, self.s_axp
        by = self.s_by
        bb = b[axp, by] + b[ax, by]
        pb = (P[axp, by] - P[ax, by]) / (2.0 * dx)

        def f_below(ws_: np.ndarray) -> np.ndarray:
            return -sig * ws_[ax, by] * bb / 2.0 - bb * pb

        Fx1[ax, by] = f_below(Wx)
        Wx1[ax, by] = Wx[ax, by] + dt * Fx1[ax, by] / 2.0
        Fx2[ax, by] = f_below(Wx1)
        Wx2[ax, by] = Wx[ax, by] + dt * Fx2[ax, by] / 2.0
        Fx3[ax, by] = f_below(Wx2)
        Wx3[ax, by] = Wx[ax, by] + dt * Fx3[ax, by]
        Fx4[ax, by] = f_below(Wx3)
        # reference line 44 typo: Fx1+2*Fx2+2*Fx1+Fx1
        Wx[ax, by] += dt * (Fx1[ax, by] + 2.0 * Fx2[ax, by]
                            + 2.0 * Fx1[ax, by] + Fx1[ax, by]) / 6.0

        # ---- above band (pis:pie, pje+1:ny), lines 46-57 ------------
        at = self.s_at
        bt = b[axp, at] + b[ax, at]
        pt = (P[axp, at] - P[ax, at]) / (2.0 * dx)

        def f_above(ws_: np.ndarray) -> np.ndarray:
            return -sig * ws_[ax, at] * bt / 2.0 - bt * pt

        Fx1[ax, at] = f_above(Wx)
        Wx1[ax, at] = Wx[ax, at] + dt * Fx1[ax, at] / 2.0
        Fx2[ax, at] = f_above(Wx1)
        Wx2[ax, at] = Wx[ax, at] + dt * Fx2[ax, at] / 2.0
        Fx3[ax, at] = f_above(Wx2)
        Wx3[ax, at] = Wx[ax, at] + dt * Fx3[ax, at]
        Fx4[ax, at] = f_above(Wx3)
        Wx[ax, at] += dt * (Fx1[ax, at] + 2.0 * Fx2[ax, at]
                            + 2.0 * Fx3[ax, at] + Fx4[ax, at]) / 6.0

    # ------------------------------------------------------------------ #
    # incident velocity y -- PML (update_V1elocity_y_fields_for_PML_2d.m) #
    # ------------------------------------------------------------------ #
    def _update_incident_velocity_y_fields_for_pml(self) -> None:
        Wy, P, b = self.Wy, self.P, self.b
        Wy1, Wy2, Wy3 = self.Wy1, self.Wy2, self.Wy3
        Fy1, Fy2, Fy3, Fy4 = self.Fy1, self.Fy2, self.Fy3, self.Fy4
        dt, dy = self.dt, self.dy
        sig = self.sigma

        # ---- left band (2:pis-1, 2:ny-1), lines 5-16 ----------------
        lx = self.s_lx
        ly1, ly1p = self.s_ly1, self.s_ly1p
        bl = b[lx, ly1p] + b[lx, ly1]
        pl = (P[lx, ly1p] - P[lx, ly1]) / (2.0 * dy)

        def f_left(ws_: np.ndarray) -> np.ndarray:
            return -sig * ws_[lx, ly1] * bl / 2.0 - bl * pl

        Fy1[lx, ly1] = f_left(Wy)
        Wy1[lx, ly1] = Wy[lx, ly1] + dt * Fy1[lx, ly1] / 2.0
        Fy2[lx, ly1] = f_left(Wy1)
        Wy2[lx, ly1] = Wy[lx, ly1] + dt * Fy2[lx, ly1] / 2.0
        Fy3[lx, ly1] = f_left(Wy2)
        Wy3[lx, ly1] = Wy[lx, ly1] + dt * Fy3[lx, ly1]
        Fy4[lx, ly1] = f_left(Wy3)
        Wy[lx, ly1] += dt * (Fy1[lx, ly1] + 2.0 * Fy2[lx, ly1]
                             + 2.0 * Fy3[lx, ly1] + Fy4[lx, ly1]) / 6.0

        # ---- right band (pie+1:nx, 2:ny-1), lines 18-29 ------------
        rxw, rxwp, rxwm = self.s_rxw, self.s_rxwp, self.s_rxwm
        br = b[rxw, ly1p] + b[rxw, ly1]
        pr = (P[rxw, ly1p] - P[rxw, ly1]) / (2.0 * dy)

        def f_right(ws_: np.ndarray) -> np.ndarray:
            return -sig * ws_[rxw, ly1] * br / 2.0 - br * pr

        Fy1[rxw, ly1] = f_right(Wy)
        Wy1[rxw, ly1] = Wy[rxw, ly1] + dt * Fy1[rxw, ly1] / 2.0
        Fy2[rxw, ly1] = f_right(Wy1)
        # reference line 23 typo: Wy2 advanced with Fy1
        Wy2[rxw, ly1] = Wy[rxw, ly1] + dt * Fy1[rxw, ly1] / 2.0
        Fy3[rxw, ly1] = f_right(Wy2)
        Wy3[rxw, ly1] = Wy[rxw, ly1] + dt * Fy3[rxw, ly1]
        Fy4[rxw, ly1] = f_right(Wy3)
        Wy[rxw, ly1] += dt * (Fy1[rxw, ly1] + 2.0 * Fy2[rxw, ly1]
                              + 2.0 * Fy3[rxw, ly1] + Fy4[rxw, ly1]) / 6.0

        # ---- below band (pis:pie, 2:pjs-1), lines 31-42 ------------
        ax = self.s_ax
        by, byp = self.s_by, self.s_byp
        bb = b[ax, byp] + b[ax, by]
        pb = (P[ax, byp] - P[ax, by]) / (2.0 * dy)

        def f_below(ws_: np.ndarray) -> np.ndarray:
            return -sig * ws_[ax, by] * bb / 2.0 - bb * pb

        Fy1[ax, by] = f_below(Wy)
        Wy1[ax, by] = Wy[ax, by] + dt * Fy1[ax, by] / 2.0
        Fy2[ax, by] = f_below(Wy1)
        Wy2[ax, by] = Wy[ax, by] + dt * Fy2[ax, by] / 2.0
        Fy3[ax, by] = f_below(Wy2)
        Wy3[ax, by] = Wy[ax, by] + dt * Fy3[ax, by]
        Fy4[ax, by] = f_below(Wy3)
        Wy[ax, by] += dt * (Fy1[ax, by] + 2.0 * Fy2[ax, by]
                            + 2.0 * Fy3[ax, by] + Fy4[ax, by]) / 6.0

        # ---- above band (pis:pie, pje+1:ny-1), lines 44-55 ---------
        at1, at1p = self.s_at1, self.s_at1p
        bt = b[ax, at1p] + b[ax, at1]
        pt = (P[ax, at1p] - P[ax, at1]) / (2.0 * dy)

        def f_above(ws_: np.ndarray) -> np.ndarray:
            return -sig * ws_[ax, at1] * bt / 2.0 - bt * pt

        Fy1[ax, at1] = f_above(Wy)
        Wy1[ax, at1] = Wy[ax, at1] + dt * Fy1[ax, at1] / 2.0
        Fy2[ax, at1] = f_above(Wy1)
        Wy2[ax, at1] = Wy[ax, at1] + dt * Fy2[ax, at1] / 2.0
        Fy3[ax, at1] = f_above(Wy2)
        Wy3[ax, at1] = Wy[ax, at1] + dt * Fy3[ax, at1]
        Fy4[ax, at1] = f_above(Wy3)
        Wy[ax, at1] += dt * (Fy1[ax, at1] + 2.0 * Fy2[ax, at1]
                             + 2.0 * Fy3[ax, at1] + Fy4[ax, at1]) / 6.0

    # ------------------------------------------------------------------ #
    # source injection (run_fdtd_time_marching_loop_2d.m, last lines)     #
    # ------------------------------------------------------------------ #
    def _inject_source(self, step: int) -> None:
        s = np.sin(2.0 * np.pi * self.f * step * self.dt)
        rows = self.src_rows
        self.p[self.src_col, rows] = s
        self.P[self.src_col, rows] = s

    # ------------------------------------------------------------------ #
    # time marching                                                       #
    # ------------------------------------------------------------------ #
    def solve(self, monitor_every: int = 200, early_stop_on_nan: bool = True,
              verbose: bool = False, phase_snapshots: bool = False,
              n_phase: int = 8) -> Dict[str, Any]:
        """Run the full loop: 10 update calls + source injection per step.

        The calling order matches ``run_fdtd_time_marching_loop_2d.m``.
        ``monitor_every > 0`` records the air-region acoustic energy
        ``mean(p^2)`` (and max |p|) every N steps for the coarse trajectory
        record. In addition the energy is recorded **every step** over the final
        ``n_period`` acoustic periods, which is what the steady-state check
        consumes (see ``_steady_stats``); that per-step series is returned as
        ``energy_tail`` so the convergence can be re-inspected afterwards.

        ``phase_snapshots`` additionally stores ``n_phase`` phase-locked air-region
        pressure snapshots spanning one acoustic period at the end of the run,
        used to extract the complex amplitude (``fit_complex_amplitude``).
        """
        t0 = time.perf_counter()
        energies: list = []
        maxps: list = []
        energy_tail: list = []
        snap: Dict[int, np.ndarray] = {}
        steps_done = 0
        broke_nan = False
        diverged = False
        max_abs_p_seen = 0.0
        mon = max(1, int(monitor_every))
        ax, ay = self.s_ax, self.s_ay

        spp = int(round(1.0 / (self.f * self.dt))) if self.f > 0 else 0
        n_tail = STEADY_N_PERIOD * spp
        tail_start = max(0, self.n_steps - n_tail)
        if phase_snapshots and spp >= n_phase:
            for k in range(n_phase):
                snap[self.n_steps - spp + int(round(k * spp / n_phase))] = k

        for step in range(1, self.n_steps + 1):
            self._update_pressure_fields()                       # 1
            self._update_velocity_x_fields()                     # 2
            self._update_velocity_y_fields()                     # 3
            self._update_velocity_x_fields_for_pml()             # 4
            self._update_velocity_y_fields_for_pml()             # 5
            self._update_incident_pressure_fields()              # 6
            self._update_incident_velocity_x_fields()            # 7
            self._update_incident_velocity_y_fields()            # 8
            self._update_incident_velocity_x_fields_for_pml()    # 9
            self._update_incident_velocity_y_fields_for_pml()    # 10
            self._inject_source(step)
            steps_done = step
            if step in snap:
                snap[step] = np.array(self.p[ax, ay], copy=True)
            if step > tail_start:
                p_air = self.p[ax, ay]
                energy_tail.append(float(np.mean(p_air * p_air)))
            if step % mon == 0:
                p_air = self.p[ax, ay]
                energies.append(float(np.mean(p_air * p_air)))
                m = float(np.max(np.abs(self.p)))
                maxps.append(m)
                max_abs_p_seen = max(max_abs_p_seen, m)
                if not np.isfinite(p_air).all() or not np.isfinite(m):
                    broke_nan = True
                    if early_stop_on_nan:
                        break
                elif m > DIVERGENCE_MAX_ABS_P:
                    # large-but-finite blow-up: flag it explicitly instead of
                    # letting it pass as "merely not steady" (see the constant's
                    # comment). Stopping here saves the remaining compute.
                    diverged = True
                    if early_stop_on_nan:
                        break
            if verbose and step % max(1000, mon) == 0:
                print(f"  step {step}/{self.n_steps}  "
                      f"mean p^2(air)={energies[-1]:.3e}", flush=True)

        wall_s = time.perf_counter() - t0
        energy = np.asarray(energies) if energies else np.empty(0)
        maxp = np.asarray(maxps) if maxps else np.empty(0)
        tail_arr = np.asarray(energy_tail) if energy_tail else np.empty(0)
        snaps = [v for _, v in sorted(snap.items())] if snap else []
        snap_steps = [s for s in sorted(snap)] if snap else []
        diag = {
            "n_steps_done": steps_done,
            "wall_s": wall_s,
            "seconds_per_step": wall_s / steps_done if steps_done else float("nan"),
            "broke_nan": broke_nan,
            "diverged": bool(diverged),
            "divergence_threshold": DIVERGENCE_MAX_ABS_P,
            "max_abs_p_seen": float(max_abs_p_seen),
            "energy_samples": energy.tolist(),
            "energy_tail": tail_arr.tolist(),
            "maxp_samples": maxp.tolist(),
            "steady": _steady_stats(tail_arr, self.dt, self.f),
            "phase_steps": snap_steps,
            "phase_snapshots": snaps,
            "max_abs_p_final": float(np.max(np.abs(self.p))) if steps_done else float("nan"),
            "max_abs_P_final": float(np.max(np.abs(self.P))) if steps_done else float("nan"),
            "rms_p_air_final": float(np.sqrt(np.mean(self.p[ax, ay] ** 2))) if steps_done else float("nan"),
        }
        return diag


# --------------------------------------------------------------------------
# top-level convenience API
# --------------------------------------------------------------------------
def run_case(f: float = 400.0, Ma: float = 0.5, c0: float = 340.0,
             rho0: float = 1.0, L: float = 1.0, alpha: float = 1.256431,
             gamma_coef: float = 2.8, dx: float = 0.125, dy: float = 0.125,
             courant_factor: float = 0.1, n_pml: int = 160,
             n_buffer: int = 240, nx: Optional[int] = None,
             ny: Optional[int] = None, sigma: float = 100.0,
             n_steps: int = 8000, source_side: str = "left",
             source_y_frac: Tuple[float, float] = (0.0, 1.0),
             monitor_every: int = 200, early_stop_on_nan: bool = True,
             crop_air: bool = True, return_velocities: bool = False,
             phase_snapshots: bool = False, n_phase: int = 8,
             verbose: bool = False) -> Dict[str, Any]:
    """Run one acoustic case and return cropped pressure fields + metadata.

    Returns a dict with ``p_total``, ``p_incident``, ``p_scattered``,
    ``x`` / ``y`` (physical-air coordinates when ``crop_air``), the parameter
    dictionary ``params`` and the diagnostics dict ``diag``. Arrays are the
    2-D physical-air region ``[pis-1:pie] x [pjs-1:pje]``.

    ``phase_snapshots`` additionally returns ``p_phase`` (``[n_phase, H, W]``
    phase-locked snapshots spanning the final acoustic period) and ``phasor``
    (least-squares complex amplitude ``phasor_re + i*phasor_im`` plus the
    residual drift diagnostic), which is what a frequency-domain physics
    formulation needs.
    """
    sol = FDTDAcoustic2D(
        f=f, Ma=Ma, c0=c0, rho0=rho0, L=L, alpha=alpha,
        gamma_coef=gamma_coef, dx=dx, dy=dy, courant_factor=courant_factor,
        n_pml=n_pml, n_buffer=n_buffer, nx=nx, ny=ny, sigma=sigma,
        n_steps=n_steps, source_side=source_side, source_y_frac=source_y_frac)
    diag = sol.solve(monitor_every=monitor_every,
                     early_stop_on_nan=early_stop_on_nan, verbose=verbose,
                     phase_snapshots=phase_snapshots, n_phase=n_phase)
    snaps = diag.pop("phase_snapshots", [])
    snap_steps = [int(s) for s in diag.get("phase_steps", [])]
    phasor = {}
    if snaps:
        phasor = fit_complex_amplitude(np.stack(snaps), np.asarray(snap_steps),
                                       sol.dt, f)
        diag["phasor_drift_rel_max"] = float(np.max(phasor["drift_rel"]))
        diag["phasor_amplitude_mean"] = float(np.mean(phasor["amplitude"]))

    i0, i1 = sol.pis - 1, sol.pie       # 0-based python slices
    j0, j1 = sol.pjs - 1, sol.pje
    if crop_air:
        p_total = np.array(sol.p[i0:i1, j0:j1], copy=True)
        p_inc = np.array(sol.P[i0:i1, j0:j1], copy=True)
        p_scat = p_total - p_inc
        x_air = np.array(sol.x_node[i0:i1], copy=True)
        y_air = np.array(sol.y_node[j0:j1], copy=True)
    else:
        p_total = np.array(sol.p, copy=True)
        p_inc = np.array(sol.P, copy=True)
        p_scat = p_total - p_inc
        x_air = np.array(sol.x_node, copy=True)
        y_air = np.array(sol.y_node, copy=True)

    vel = {}
    if return_velocities:
        vel = {"wx_total": np.array(sol.wx[i0:i1, j0:j1], copy=True),
               "wy_total": np.array(sol.wy[i0:i1, j0:j1], copy=True),
               "Wx_incident": np.array(sol.Wx[i0:i1, j0:j1], copy=True),
               "Wy_incident": np.array(sol.Wy[i0:i1, j0:j1], copy=True)}

    params = {
        "f_hz": float(f), "omega_rad_s": 2.0 * np.pi * float(f),
        "Ma": float(Ma), "c0": float(c0), "rho0": float(rho0), "L": float(L),
        "alpha": float(alpha), "gamma_coef": float(gamma_coef),
        "Gamma": float(sol.Gamma), "dx": float(dx), "dy": float(dy),
        "courant_factor": float(courant_factor), "dt_s": float(sol.dt),
        "n_pml": int(n_pml), "n_buffer": int(n_buffer),
        "nx_cells": int(sol.nx), "ny_cells": int(sol.ny),
        "nxp1": int(sol.nxp1), "nyp1": int(sol.nyp1),
        "n_steps": int(n_steps), "t_final_s": float(sol.t_final),
        "sigma_pml": float(sigma), "source_side": source_side,
        "source_y_frac": [float(source_y_frac[0]), float(source_y_frac[1])],
        "src_col_0based": int(sol.src_col),
        "src_rows_0based": [int(sol.src_rows[0]), int(sol.src_rows[-1])],
        "pis_pie_pjs_pje_matlab": [int(sol.pis), int(sol.pie),
                                   int(sol.pjs), int(sol.pje)],
        "air_node_range_0based": [int(i0), int(i1), int(j0), int(j1)],
        "crop_air": bool(crop_air),
        "lambda_m": float(c0 / f),
        "cells_per_lambda": float(c0 / f / dx),
        "grid_min_x_m": float(sol.min_x), "grid_max_x_m": float(sol.max_x),
    }

    return {"p_total": p_total, "p_incident": p_inc, "p_scattered": p_scat,
            "x": x_air, "y": y_air, "params": params, "diag": diag,
            "velocities": vel, "p_phase": snaps, "phasor": phasor}


def case_params_to_dict(f, Ma, **kw) -> Dict[str, Any]:
    """Serialisable parameter dict for a case (used by the generator)."""
    # quick json-safe wrapper around the subset of run_case keyword args
    allowed = ["c0", "rho0", "L", "alpha", "gamma_coef", "dx", "dy",
               "courant_factor", "n_pml", "n_buffer", "sigma", "n_steps",
               "source_side", "source_y_frac"]
    out: Dict[str, Any] = {"f": float(f), "Ma": float(Ma)}
    for k in allowed:
        if k in kw and kw[k] is not None:
            out[k] = kw[k]
    return out
