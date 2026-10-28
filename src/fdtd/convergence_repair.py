"""R2-ACOUSTIC-CONVERGENCE-REPAIR-01 independent repair module.

The legacy ``python_fdtd`` advances pressure and each velocity component with a
separate four-stage Runge-Kutta sweep, freezing the other components.  That is a
block-sequential integrator, not a simultaneous RK4 for the coupled first-order
system.  This module reuses the legacy grid, vortex, source and spatial/PML
operators, but evaluates the complete derivative at every RK4 stage and advances
all six fields (total and incident) together.  The legacy solver and all old
results remain untouched.

``repair`` modes:
  * ``coupled_ode_study``: exact nonzero-coupling linear ODE vs sequential and
    synchronized RK4, observed order/phase.
  * ``plane_wave_study``: periodic 1-D discrete acoustic symbol and phase speed.
  * ``analyze_existing``: diagnostics for the preserved ten old final fields.
  * ``RepairedFDTD2D``: synchronized RK4 production class.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.fdtd.python_fdtd import FDTDAcoustic2D, _lamb_oseen_vortex


# ---------------------------------------------------------------------------
# analytic / semi-discrete verification
# ---------------------------------------------------------------------------
def exact_oscillator(t: np.ndarray, k: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact solution of p'=-k w, w'=k p with p(0)=1, w(0)=0."""
    return np.cos(k * t), np.sin(k * t)


def sequential_block_step(p: float, w: float, dt: float, k: float) -> tuple[float, float]:
    """Legacy block-sequential RK4 step for the nonzero coupled ODE."""
    def p_rhs(pp, ww):
        return -k * ww

    def w_rhs(pp, ww):
        return k * pp

    # pressure RK4, w frozen
    k1 = p_rhs(p, w)
    k2 = p_rhs(p + dt * k1 / 2.0, w)
    k3 = p_rhs(p + dt * k2 / 2.0, w)
    k4 = p_rhs(p + dt * k3, w)
    pn = p + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    # velocity RK4, new pressure frozen
    l1 = w_rhs(pn, w)
    l2 = w_rhs(pn, w + dt * l1 / 2.0)
    l3 = w_rhs(pn, w + dt * l2 / 2.0)
    l4 = w_rhs(pn, w + dt * l3)
    wn = w + dt * (l1 + 2.0 * l2 + 2.0 * l3 + l4) / 6.0
    return pn, wn


def synchronized_step(p: float, w: float, dt: float, k: float) -> tuple[float, float]:
    """Standard synchronized RK4 for the same coupled ODE."""
    def f(pp, ww):
        return -k * ww, k * pp

    k1p, k1w = f(p, w)
    k2p, k2w = f(p + dt * k1p / 2.0, w + dt * k1w / 2.0)
    k3p, k3w = f(p + dt * k2p / 2.0, w + dt * k2w / 2.0)
    k4p, k4w = f(p + dt * k3p, w + dt * k3w)
    return (p + dt * (k1p + 2 * k2p + 2 * k3p + k4p) / 6.0,
            w + dt * (k1w + 2 * k2w + 2 * k3w + k4w) / 6.0)


@dataclass
class OdeResult:
    scheme: str
    n_steps: int
    dt: float
    error_p: float
    error_w: float
    phase_error: float
    amplitude_ratio: float


def ode_order_study(k: float = 2.0 * np.pi, periods: float = 4.0,
                    n_values: tuple[int, ...] = (20, 40, 80, 160)) -> dict[str, Any]:
    """Convergence/phase study for the nonzero-coupling system."""
    T = periods * (2.0 * np.pi / k)
    results = {'system': "p'=-k w, w'=k p; p(0)=1,w(0)=0",
               'k': k, 'periods': periods, 'schemes': []}
    for name, step in (('sequential_block_rk4', sequential_block_step),
                       ('synchronized_rk4', synchronized_step)):
        rows = []
        for n in n_values:
            dt = T / n
            p, w = 1.0, 0.0
            for _ in range(n):
                p, w = step(p, w, dt, k)
            pe, we = exact_oscillator(np.array([T]), k)
            ep = float(abs(p - pe[0]))
            ew = float(abs(w - we[0]))
            # phase from atan2; exact phase is T*k=2*pi*periods
            phase = float(np.arctan2(w, p))
            phase_ref = float(2.0 * np.pi * periods % (2.0 * np.pi))
            # principal difference
            phase_err = float(abs((phase - phase_ref + np.pi) % (2 * np.pi) - np.pi))
            rows.append({'n_steps': n, 'dt': dt, 'error_p': ep, 'error_w': ew,
                         'phase_error_rad': phase_err,
                         'amplitude_ratio': float(np.hypot(p, w))})
        orders = []
        for a, b in zip(rows[:-1], rows[1:]):
            if a['error_p'] > 0 and b['error_p'] > 0:
                orders.append(float(np.log(a['error_p'] / b['error_p'])
                                    / np.log(b['n_steps'] / a['n_steps'])))
        results['schemes'].append({'name': name, 'rows': rows,
                                   'estimated_order_error_p': orders,
                                   'phase_error_monotone_order': orders})
    return results


def fd_phase_speed(k: float, dx: float, order: int, c: float = 1.0) -> dict[str, float]:
    """Phase speed of the repaired collocated central-difference operator."""
    theta = k * dx
    if order == 2:
        coeff = [1.0, -1.0]; denom = 2.0
    elif order == 6:
        coeff = [45.0, -9.0, 1.0]; denom = 60.0
    elif order == 10:
        coeff = [2100.0, -600.0, 150.0, -25.0, 2.0]; denom = 2520.0
    else:
        raise ValueError('order must be 2, 6 or 10')
    # central antisymmetric symbol: sum_n a_n sin(n theta)/(denom*dx)
    sym = 0.0
    for n, an in enumerate(coeff, start=1):
        sym += an * np.sin(n * theta) / (denom * dx)
    sym *= 2.0  # positive and negative stencil pairs
    omega = c * abs(float(sym))
    omega_exact = c * k
    return {'order': order, 'k': k, 'dx': dx, 'k_dx': theta,
            'ppw': 2 * np.pi / theta,
            'omega_num': omega, 'omega_exact': omega_exact,
            'phase_speed_num': omega / k if k else float('nan'),
            'phase_speed_error_rel': (omega / k - c) / c if k else float('nan')}


def plane_wave_study(f: float = 500.0, c: float = 340.0,
                     dx_values=(0.125, 0.1, 0.0833333333, 0.0625),
                     periods: float = 177.0) -> dict[str, Any]:
    k = 2.0 * np.pi * f / c
    rows = []
    for dx in dx_values:
        for order in (2, 6, 10):
            row = fd_phase_speed(k, dx, order, c)
            row['phase_error_after_periods_rad'] = float(
                abs(row['omega_num'] - row['omega_exact']) * (periods / f))
            row['phase_error_after_periods_cycles'] = float(
                row['phase_error_after_periods_rad'] / (2 * np.pi))
            rows.append(row)
    return {'frequency_hz': f, 'c0_m_s': c, 'periods': periods,
            'dispersion': rows,
            'note': 'repaired collocated central-difference phase speeds; '
                    'legacy measured carrier is reported separately from old fields'}


# ---------------------------------------------------------------------------
# diagnostics of the preserved ten old fields
# ---------------------------------------------------------------------------
def _load_fields(base: str, case: str, resolution: str) -> dict[str, np.ndarray]:
    import os
    p = os.path.join(base, case, resolution, 'fields.npz')
    with np.load(p) as d:
        return {k: np.array(d[k]) for k in d.files}


def _same_nodes(coarse: dict[str, np.ndarray], fine: dict[str, np.ndarray],
                stride: int) -> tuple[np.ndarray, ...]:
    return fine['p_total'][::stride, ::stride]


def analyze_existing(base: str, case: str = 'high500') -> dict[str, Any]:
    """Quantify old coarse/mid/fine differences and local error structure."""
    import os
    c = _load_fields(base, case, 'coarse')
    m = _load_fields(base, case, 'mid')
    f = _load_fields(base, case, 'fine')
    x, y = c['x'], c['y']
    X, Y = np.meshgrid(x, y, indexing='ij')
    out: dict[str, Any] = {'case': case, 'nodes': [int(v) for v in c['p_total'].shape],
                           'comparisons': {}, 'regions': {}, 'spectra': {}}
    pairs = {'coarse_vs_mid': (c, m, 2), 'mid_vs_fine': (m, f, 2)}
    for name, (a, b, stride) in pairs.items():
        bs = {k: v[::stride, ::stride] for k, v in b.items()
              if isinstance(v, np.ndarray) and v.ndim == 2}
        comp = {}
        for field in ('p_total', 'p_incident', 'p_scattered'):
            d = a[field] - bs[field]
            comp[field] = {
                're': float(np.linalg.norm(d) / np.linalg.norm(bs[field])),
                'mae': float(np.mean(np.abs(d))),
                'max': float(np.max(np.abs(d))),
                'difference_l2': float(np.linalg.norm(d)),
                'reference_l2': float(np.linalg.norm(bs[field])),
                'amplitude_a': float(np.sqrt(np.mean(a[field] ** 2))),
                'amplitude_b': float(np.sqrt(np.mean(bs[field] ** 2))),
            }
        out['comparisons'][name] = comp
    # region decomposition on the coarse-vs-mid comparison
    mc = m['p_total'][::2, ::2]
    d = c['p_total'] - mc
    masks = {
        'source_column': X <= x[0] + 1.5 * (x[1] - x[0]),
        'pml_adjacent': (np.abs(X) > 13.5),
        'interior': np.abs(X) <= 10.0,
        'vortex_core': (X ** 2 + Y ** 2) <= 4.0,
        'all_air': np.ones_like(X, dtype=bool),
    }
    for label, mask in masks.items():
        out['regions'][label] = {
            'n': int(mask.sum()),
            're': float(np.linalg.norm(d[mask]) / np.linalg.norm(mc[mask])),
            'mae': float(np.mean(np.abs(d[mask]))),
            'max': float(np.max(np.abs(d[mask]))),
        }
    # spectra of high500 coarse-mid difference and fields
    spec = {}
    for label, arr in (('coarse_mid_diff', d), ('coarse_p_total', c['p_total']),
                       ('mid_p_total', m['p_total'])):
        F = np.fft.fftshift(np.fft.fft2(arr))
        P = np.abs(F)
        idx = np.unravel_index(int(np.argmax(P)), P.shape)
        spec[label] = {'peak_index': [int(idx[0]), int(idx[1])],
                       'peak_abs': float(P[idx]),
                       'nyquist_energy_fraction': float(
                           np.sum(P[::2, ::2] ** 2) / np.sum(P ** 2))}
    out['spectra'] = spec
    # source/PML audit from protocol/meta if present
    audit = {}
    for res in ('coarse', 'mid', 'fine'):
        pdir = os.path.join(base, case, res)
        try:
            meta = json.load(open(os.path.join(pdir, 'meta.json'), encoding='utf-8'))
            audit[res] = {'dx': meta['plan']['dx_m'],
                          'dt': meta['plan']['dt_s'],
                          'n_steps': meta['plan']['n_steps'],
                          'ppw': meta['plan']['ppw'],
                          'pml_cells': meta['plan']['n_pml_cells'],
                          'pml_thickness_m': meta['plan']['pml_thickness_m'],
                          'sigma': meta['identity']['sigma_pml'],
                          'source_x_m': meta['identity']['source_x_target_m'],
                          'source_t': 'sin(2*pi*f*step*dt), injected on p and P after each full step',
                          'final_phase_residual': meta['plan'].get('final_phase_residual_cycles_target')}
        except Exception as exc:  # noqa: BLE001
            audit[res] = {'error': repr(exc)}
    # measured carrier of the preserved incident field (diagnostic, no phase alignment)
    measured = {}
    expected_lambda = 340.0 / 500.0
    for res, z in (('coarse', c), ('mid', m), ('fine', f)):
        prof = z['p_incident'][:, z['p_incident'].shape[1] // 2]
        xres = z['x']
        F = np.abs(np.fft.fft(prof))
        kcyc = np.fft.fftfreq(len(xres), d=xres[1] - xres[0])
        idx = int(np.argmax(F[1:]) + 1) if len(F) > 1 else 0
        lam = 1.0 / abs(kcyc[idx]) if kcyc[idx] else float('nan')
        measured[res] = {'peak_cycles_per_m': float(abs(kcyc[idx])),
                         'measured_wavelength_m': float(lam),
                         'expected_wavelength_m': expected_lambda,
                         'phase_speed_m_s': float(500.0 * lam),
                         'phase_speed_error_rel': float((500.0 * lam - 340.0) / 340.0),
                         'amplitude_peak_abs': float(F[idx] / len(xres))}
    out['measured_incident_carrier'] = measured
    out['source_pml_audit'] = audit
    return out


# ---------------------------------------------------------------------------
# synchronized production solver
# ---------------------------------------------------------------------------
class RepairedFDTD2D(FDTDAcoustic2D):
    """Legacy operators + simultaneous RK4 for (p,wx,wy,P,Wx,Wy)."""

    def _rhs_all(self) -> tuple[np.ndarray, ...]:
        """Evaluate all six RHS arrays at the current primary state.

        The legacy sweep methods are called with ``dt=0``.  They compute their
        first-stage derivatives and perform no primary-state update.  This reuses
        the exact legacy spatial/PML/source operators without copying them.
        """
        saved_dt = self.dt
        self.dt = 0.0
        try:
            for arr in (self.fp1, self.fx1, self.fy1,
                        self.Fp1, self.Fx1, self.Fy1):
                arr.fill(0.0)
            for method in (
                self._update_pressure_fields,
                self._update_velocity_x_fields,
                self._update_velocity_y_fields,
                self._update_velocity_x_fields_for_pml,
                self._update_velocity_y_fields_for_pml,
                self._update_incident_pressure_fields,
                self._update_incident_velocity_x_fields,
                self._update_incident_velocity_y_fields,
                self._update_incident_velocity_x_fields_for_pml,
                self._update_incident_velocity_y_fields_for_pml,
            ):
                method()
            return (self.fp1.copy(), self.fx1.copy(), self.fy1.copy(),
                    self.Fp1.copy(), self.Fx1.copy(), self.Fy1.copy())
        finally:
            self.dt = saved_dt

    def _state(self) -> tuple[np.ndarray, ...]:
        return (self.p.copy(), self.wx.copy(), self.wy.copy(),
                self.P.copy(), self.Wx.copy(), self.Wy.copy())

    def _set_state(self, state: tuple[np.ndarray, ...]) -> None:
        self.p[...], self.wx[...], self.wy[...] = state[0], state[1], state[2]
        self.P[...], self.Wx[...], self.Wy[...] = state[3], state[4], state[5]

    def solve(self, monitor_every: int = 200, early_stop_on_nan: bool = True,
              verbose: bool = False, phase_snapshots: bool = False,
              n_phase: int = 8) -> dict[str, Any]:
        import time
        t0 = time.perf_counter()
        y = self._state()
        dt = self.dt
        energies, maxps = [], []
        max_abs_p_seen = 0.0
        broke_nan = False
        diverged = False
        steps_done = 0
        mon = max(1, int(monitor_every))
        ax, ay = self.s_ax, self.s_ay
        for step in range(1, self.n_steps + 1):
            k1 = self._rhs_all()
            stage = tuple(y[i] + 0.5 * dt * k1[i] for i in range(6))
            self._set_state(stage)
            k2 = self._rhs_all()
            stage = tuple(y[i] + 0.5 * dt * k2[i] for i in range(6))
            self._set_state(stage)
            k3 = self._rhs_all()
            stage = tuple(y[i] + dt * k3[i] for i in range(6))
            self._set_state(stage)
            k4 = self._rhs_all()
            y = tuple(y[i] + dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0
                      for i in range(6))
            self._set_state(y)
            self._inject_source(step)
            for arr in (self.p, self.wx, self.wy, self.P, self.Wx, self.Wy):
                arr[0, :] = 0.0; arr[-1, :] = 0.0
                arr[:, 0] = 0.0; arr[:, -1] = 0.0
            steps_done = step
            if step % mon == 0:
                pa = self.p[ax, ay]
                energies.append(float(np.mean(pa * pa)))
                m = float(np.max(np.abs(self.p)))
                maxps.append(m)
                max_abs_p_seen = max(max_abs_p_seen, m)
                if not np.isfinite(pa).all() or not np.isfinite(m):
                    broke_nan = True
                    if early_stop_on_nan:
                        break
                elif m > 1e8:
                    diverged = True
                    if early_stop_on_nan:
                        break
            if verbose and step % max(1000, mon) == 0:
                print(f'  repaired step {step}/{self.n_steps}', flush=True)
        wall_s = time.perf_counter() - t0
        return {'n_steps_done': steps_done, 'wall_s': wall_s,
                'seconds_per_step': wall_s / steps_done if steps_done else float('nan'),
                'broke_nan': broke_nan, 'diverged': bool(diverged),
                'max_abs_p_seen': float(max_abs_p_seen),
                'max_abs_p_final': float(np.max(np.abs(self.p))) if steps_done else float('nan'),
                'rms_p_air_final': float(np.sqrt(np.mean(self.p[ax, ay] ** 2))) if steps_done else float('nan'),
                'energy_samples': energies, 'maxp_samples': maxps,
                'scheme': 'synchronized_rk4_repaired'}


# ---------------------------------------------------------------------------
# high-order collocated repair variant (same physical domain/source/PML extent)
# ---------------------------------------------------------------------------
def _d1_2(f: np.ndarray, dx: float, axis: int) -> np.ndarray:
    out = np.zeros_like(f)
    if axis == 0:
        out[1:-1] = (f[2:] - f[:-2]) / (2.0 * dx)
        out[0] = (f[1] - f[0]) / dx
        out[-1] = (f[-1] - f[-2]) / dx
    else:
        out[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2.0 * dx)
        out[:, 0] = (f[:, 1] - f[:, 0]) / dx
        out[:, -1] = (f[:, -1] - f[:, -2]) / dx
    return out


def _d1_6(f: np.ndarray, dx: float, axis: int) -> np.ndarray:
    """Sixth-order central first derivative, 2nd-order at the outer edges."""
    out = np.zeros_like(f)
    if axis == 0:
        n = f.shape[0]
        if n >= 3:
            out[1:-1] = (f[2:] - f[:-2]) / (2.0 * dx)
            out[0] = (f[1] - f[0]) / dx
            out[-1] = (f[-1] - f[-2]) / dx
        if n >= 7:
            out[3:-3] = (f[6:] - 9.0 * f[5:-1] + 45.0 * f[4:-2]
                         - 45.0 * f[2:-4] + 9.0 * f[1:-5] - f[0:-6]) / (60.0 * dx)
    else:
        n = f.shape[1]
        if n >= 3:
            out[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2.0 * dx)
            out[:, 0] = (f[:, 1] - f[:, 0]) / dx
            out[:, -1] = (f[:, -1] - f[:, -2]) / dx
        if n >= 7:
            out[:, 3:-3] = (f[:, 6:] - 9.0 * f[:, 5:-1] + 45.0 * f[:, 4:-2]
                            - 45.0 * f[:, 2:-4] + 9.0 * f[:, 1:-5] - f[:, :-6]) / (60.0 * dx)
    return out


def _d1_10(f: np.ndarray, dx: float, axis: int) -> np.ndarray:
    """Tenth-order central first derivative, 2nd/6th order near the edges."""
    out = np.zeros_like(f)
    if axis == 0:
        n = f.shape[0]
        if n >= 3:
            out[1:-1] = (f[2:] - f[:-2]) / (2.0 * dx)
            out[0] = (f[1] - f[0]) / dx
            out[-1] = (f[-1] - f[-2]) / dx
        if n >= 7:
            out[3:-3] = (f[6:] - 9.0 * f[5:-1] + 45.0 * f[4:-2]
                         - 45.0 * f[2:-4] + 9.0 * f[1:-5] - f[0:-6]) / (60.0 * dx)
        if n >= 11:
            out[5:-5] = (2.0 * f[10:] - 25.0 * f[9:-1] + 150.0 * f[8:-2]
                         - 600.0 * f[7:-3] + 2100.0 * f[6:-4]
                         - 2100.0 * f[4:-6] + 600.0 * f[3:-7]
                         - 150.0 * f[2:-8] + 25.0 * f[1:-9]
                         - 2.0 * f[0:-10]) / (2520.0 * dx)
    else:
        n = f.shape[1]
        if n >= 3:
            out[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2.0 * dx)
            out[:, 0] = (f[:, 1] - f[:, 0]) / dx
            out[:, -1] = (f[:, -1] - f[:, -2]) / dx
        if n >= 7:
            out[:, 3:-3] = (f[:, 6:] - 9.0 * f[:, 5:-1] + 45.0 * f[:, 4:-2]
                            - 45.0 * f[:, 2:-4] + 9.0 * f[:, 1:-5] - f[:, :-6]) / (60.0 * dx)
        if n >= 11:
            out[:, 5:-5] = (2.0 * f[:, 10:] - 25.0 * f[:, 9:-1] + 150.0 * f[:, 8:-2]
                            - 600.0 * f[:, 7:-3] + 2100.0 * f[:, 6:-4]
                            - 2100.0 * f[:, 4:-6] + 600.0 * f[:, 3:-7]
                            - 150.0 * f[:, 2:-8] + 25.0 * f[:, 1:-9]
                            - 2.0 * f[:, :-10]) / (2520.0 * dx)
    return out



    """Tenth-order central first derivative, 2nd-order on the two edge nodes."""
    out = np.zeros_like(f)
    if axis == 0:
        n = f.shape[0]
        if n >= 11:
            out[5:-5] = (2.0 * f[10:] - 25.0 * f[9:-1] + 150.0 * f[8:-2]
                         - 600.0 * f[7:-3] + 2100.0 * f[6:-4]
                         - 2100.0 * f[4:-6] + 600.0 * f[3:-7]
                         - 150.0 * f[2:-8] + 25.0 * f[1:-9]
                         - 2.0 * f[0:-10]) / (2520.0 * dx)
        if n >= 7:
            out[3:-3] = (f[6:] - 9.0 * f[5:-1] + 45.0 * f[4:-2]
                         - 45.0 * f[2:-4] + 9.0 * f[1:-5] - f[0:-6]) / (60.0 * dx)
        if n >= 3:
            out[1:-1] = (f[2:] - f[:-2]) / (2.0 * dx)
            out[0] = (f[1] - f[0]) / dx
            out[-1] = (f[-1] - f[-2]) / dx
    else:
        n = f.shape[1]
        if n >= 11:
            out[:, 5:-5] = (2.0 * f[:, 10:] - 25.0 * f[:, 9:-1] + 150.0 * f[:, 8:-2]
                            - 600.0 * f[:, 7:-3] + 2100.0 * f[:, 6:-4]
                            - 2100.0 * f[:, 4:-6] + 600.0 * f[:, 3:-7]
                            - 150.0 * f[:, 2:-8] + 25.0 * f[:, 1:-9]
                            - 2.0 * f[:, :-10]) / (2520.0 * dx)
        if n >= 7:
            out[:, 3:-3] = (f[:, 6:] - 9.0 * f[:, 5:-1] + 45.0 * f[:, 4:-2]
                            - 45.0 * f[:, 2:-4] + 9.0 * f[:, 1:-5] - f[:, :-6]) / (60.0 * dx)
        if n >= 3:
            out[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2.0 * dx)
            out[:, 0] = (f[:, 1] - f[:, 0]) / dx
            out[:, -1] = (f[:, -1] - f[:, -2]) / dx
    return out


class HighOrderFDTD2D(FDTDAcoustic2D):
    """Synchronized RK4 + sixth-order central air operator + graded sponge.

    This is an independent repaired algorithm, not the legacy generator.  The
    physical domain, source, vortex, final time and PML geometric extent are the
    legacy ones; the discrete time integration is simultaneous RK4 and the air
    derivatives use sixth-order central differences (second order on the two
    nodes next to an interface).  A graded sponge replaces the legacy
    one-sided constant-sigma PML update in this independent version.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # graded sponge: sigma = sigma_max * r^power in the PML, 0 in air
        x = self.x_node
        y = self.y_node
        X, Y = np.meshgrid(x, y, indexing='ij')
        ax0, ax1 = self.pis - 1, self.pie
        ay0, ay1 = self.pjs - 1, self.pje
        x_air = x[ax0:ax1]
        y_air = y[ay0:ay1]
        dx_out = np.maximum(np.maximum(x_air[0] - X, X - x_air[-1]), 0.0) / 10.0
        dy_out = np.maximum(np.maximum(y_air[0] - Y, Y - y_air[-1]), 0.0) / 10.0
        r = np.minimum(np.sqrt(dx_out ** 2 + dy_out ** 2), 1.0)
        self.sigma_sponge = 6.0 * self.c0 / 10.0 * r ** 4
        self.air_mask = np.zeros_like(X, dtype=bool)
        self.air_mask[ax0:ax1, ay0:ay1] = True
        self.air_i0, self.air_i1 = ax0, ax1
        self.air_j0, self.air_j1 = ay0, ay1

    def _deriv(self, f: np.ndarray, axis: int) -> np.ndarray:
        """High order only well inside the air block; low order near boundaries."""
        d = _d1_10(f, self.dx if axis == 0 else self.dy, axis)
        if axis == 0:
            safe10 = np.zeros_like(self.air_mask)
            safe10[self.air_i0 + 5:self.air_i1 - 5, self.air_j0:self.air_j1] = True
            safe6 = np.zeros_like(self.air_mask)
            safe6[self.air_i0 + 3:self.air_i1 - 3, self.air_j0:self.air_j1] = True
        else:
            safe10 = np.zeros_like(self.air_mask)
            safe10[:, self.air_j0 + 5:self.air_j1 - 5] = True
            safe6 = np.zeros_like(self.air_mask)
            safe6[:, self.air_j0 + 3:self.air_j1 - 3] = True
        # fill everything that is not safely high order with the next lower scheme
        d6 = _d1_6(f, self.dx if axis == 0 else self.dy, axis)
        d2 = _d1_2(f, self.dx if axis == 0 else self.dy, axis)
        d[~safe10] = d6[~safe10]
        d[~safe6] = d2[~safe6]
        # PML/outside: second order only
        d[~self.air_mask] = d2[~self.air_mask]
        return d

    def _rhs_high(self) -> tuple[np.ndarray, ...]:
        p, wx, wy, P, Wx, Wy = self.p, self.wx, self.wy, self.P, self.Wx, self.Wy
        sig = self.sigma_sponge
        dx, dy = self.dx, self.dy
        b, kap = 1.0 / self.rho0, self.rho0 * self.c0 * self.c0
        vx, vy = self.vx, self.vy
        dpdx = self._deriv(p, 0); dpdy = self._deriv(p, 1)
        dwxdx = self._deriv(wx, 0); dwxdy = self._deriv(wx, 1)
        dwydx = self._deriv(wy, 0); dwydy = self._deriv(wy, 1)
        # vortex advection uses the same high-order derivatives
        dvxdx = self._deriv(vx, 0); dvxdy = self._deriv(vx, 1)
        dvydx = self._deriv(vy, 0); dvydy = self._deriv(vy, 1)
        rhs_p = (-vx * dpdx - vy * dpdy - kap * (dwxdx + dwydy) - sig * p)
        rhs_wx = (-vx * dwxdx - vy * dwxdy - b * dpdx - sig * wx)
        rhs_wy = (-vx * dwydx - vy * dwydy - b * dpdy - sig * wy)
        # incident: quiescent medium, same sponge and high-order operator
        dPdx = self._deriv(P, 0); dPdy = self._deriv(P, 1)
        dWxdx = self._deriv(Wx, 0); dWydy = self._deriv(Wy, 1)
        rhs_P = -kap * (dWxdx + dWydy) - sig * P
        rhs_Wx = -b * dPdx - sig * Wx
        rhs_Wy = -b * dPdy - sig * Wy
        return (rhs_p, rhs_wx, rhs_wy, rhs_P, rhs_Wx, rhs_Wy)

    def _state(self):
        return (self.p.copy(), self.wx.copy(), self.wy.copy(),
                self.P.copy(), self.Wx.copy(), self.Wy.copy())

    def _set_state(self, y):
        self.p[...] = y[0]; self.wx[...] = y[1]; self.wy[...] = y[2]
        self.P[...] = y[3]; self.Wx[...] = y[4]; self.Wy[...] = y[5]

    def solve(self, monitor_every: int = 200, early_stop_on_nan: bool = True,
              verbose: bool = False, phase_snapshots: bool = False,
              n_phase: int = 8) -> dict[str, Any]:
        import time
        t0 = time.perf_counter()
        y = self._state()
        dt = self.dt
        energies, maxps = [], []
        max_abs_p_seen = 0.0
        broke_nan = False
        diverged = False
        steps_done = 0
        mon = max(1, int(monitor_every))
        ax, ay = self.s_ax, self.s_ay
        # zero outer faces after source? Sponge has sigma>0 but no explicit BC;
        # enforce a homogeneous outer-face condition as a reflecting-safe closure
        # only after full steps; the sponge decays incident energy before it.
        def _stage_src(t):
            if self.stage_source:
                ss = np.sin(2.0 * np.pi * self.f * t)
                self.p[self.src_col, self.src_rows] = ss
                self.P[self.src_col, self.src_rows] = ss
        for step in range(1, self.n_steps + 1):
            _stage_src((step - 0.5) * dt)
            k1 = self._rhs_high()
            self._set_state(tuple(y[i] + 0.5 * dt * k1[i] for i in range(6)))
            _stage_src(step * dt)
            k2 = self._rhs_high()
            self._set_state(tuple(y[i] + 0.5 * dt * k2[i] for i in range(6)))
            _stage_src(step * dt)
            k3 = self._rhs_high()
            self._set_state(tuple(y[i] + dt * k3[i] for i in range(6)))
            _stage_src((step + 0.5) * dt)
            k4 = self._rhs_high()
            y = tuple(y[i] + dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0
                      for i in range(6))
            self._set_state(y)
            self._inject_source(step)
            for arr in (self.p, self.wx, self.wy, self.P, self.Wx, self.Wy):
                arr[0, :] = 0.0; arr[-1, :] = 0.0
                arr[:, 0] = 0.0; arr[:, -1] = 0.0
            steps_done = step
            if step % mon == 0:
                pa = self.p[ax, ay]
                energies.append(float(np.mean(pa * pa)))
                m = float(np.max(np.abs(self.p)))
                maxps.append(m)
                max_abs_p_seen = max(max_abs_p_seen, m)
                if not np.isfinite(pa).all() or not np.isfinite(m):
                    broke_nan = True
                    if early_stop_on_nan:
                        break
                elif m > 1e8:
                    diverged = True
                    if early_stop_on_nan:
                        break
        wall_s = time.perf_counter() - t0
        return {'n_steps_done': steps_done, 'wall_s': wall_s,
                'seconds_per_step': wall_s / steps_done if steps_done else float('nan'),
                'broke_nan': broke_nan, 'diverged': bool(diverged),
                'max_abs_p_seen': float(max_abs_p_seen),
                'max_abs_p_final': float(np.max(np.abs(self.p))) if steps_done else float('nan'),
                'rms_p_air_final': float(np.sqrt(np.mean(self.p[ax, ay] ** 2))) if steps_done else float('nan'),
                'energy_samples': energies, 'maxp_samples': maxps,
                'scheme': 'synchronized_rk4_high_order_sponge'}


def run_high_order_case(f, Ma, dx, dy, n_steps, **kwargs):
    """Convenience wrapper with legacy grid/source and repaired high-order RHS."""
    sol = HighOrderFDTD2D(f=f, Ma=Ma, dx=dx, dy=dy, n_steps=n_steps,
                          n_pml=int(round(10.0 / dx)),
                          n_buffer=int(round(15.0 / dy)),
                          **kwargs)
    diag = sol.solve(monitor_every=kwargs.pop('monitor_every', 200),
                     early_stop_on_nan=kwargs.pop('early_stop_on_nan', True))
    i0, i1 = sol.pis - 1, sol.pie
    j0, j1 = sol.pjs - 1, sol.pje
    return {'p_total': np.array(sol.p[i0:i1, j0:j1], copy=True),
            'p_incident': np.array(sol.P[i0:i1, j0:j1], copy=True),
            'p_scattered': np.array(sol.p[i0:i1, j0:j1] - sol.P[i0:i1, j0:j1], copy=True),
            'x': np.array(sol.x_node[i0:i1], copy=True),
            'y': np.array(sol.y_node[j0:j1], copy=True),
            'diag': diag, 'params': {'f': f, 'Ma': Ma, 'dx': dx, 'dy': dy,
                                     'dt': sol.dt, 'n_steps': n_steps}}


# ---------------------------------------------------------------------------
# optional PyTorch backend for the same repaired high-order method
# ---------------------------------------------------------------------------
class TorchHighOrderFDTD2D:
    """GPU/CPU PyTorch implementation of the repaired high-order method.

    The CPU NumPy path is retained for audit; this backend exists only to make the
    frozen five-run protocol tractable. It is initialized from the same legacy
    grid/source/vortex formulas and uses double precision. No training/autograd.
    """
    def __init__(self, f=500.0, Ma=0.7, c0=340.0, rho0=1.0, L=1.0,
                 alpha=1.256431, gamma_coef=2.8, dx=0.1, dy=0.1,
                 n_pml=None, n_buffer=None, n_steps=1, device='cuda',
                 dtype='float64', order=10):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        self.order = int(order)
        self.stage_source = True
        self.dtype = torch.float64 if dtype == 'float64' else torch.float32
        self.f, self.Ma, self.c0 = float(f), float(Ma), float(c0)
        self.rho0, self.L, self.alpha, self.gamma_coef = rho0, L, alpha, gamma_coef
        self.dx = float(dx); self.dy = float(dy)
        self.n_pml = int(round(10.0 / dx)) if n_pml is None else int(n_pml)
        if n_buffer is None:
            n_buffer = int(round(15.0 / dy))
        self.n_steps = int(n_steps)
        base = 2 * (self.n_pml + int(n_buffer))
        self.nx = self.ny = base
        self.nxp1, self.nyp1 = self.nx + 1, self.ny + 1
        self.min_x = -(self.nx / 2.0) * dx; self.max_x = +(self.nx / 2.0) * dx
        self.min_y = -(self.ny / 2.0) * dy; self.max_y = +(self.ny / 2.0) * dy
        self.x_node = np.linspace(self.min_x, self.max_x, self.nxp1)
        self.y_node = np.linspace(self.min_y, self.max_y, self.nyp1)
        self.pis, self.pie = self.n_pml + 1, self.nx - self.n_pml + 1
        self.pjs, self.pje = self.n_pml + 1, self.ny - self.n_pml + 1
        self.src_col = self.pis - 1
        self.src_rows = np.arange(self.pjs - 1, self.pje)
        self.velocity_source = False
        self.stage_source = True
        X, Y = np.meshgrid(self.x_node, self.y_node, indexing='ij')
        vx, vy = _lamb_oseen_vortex(X, Y, self.gamma_coef * self.Ma * np.pi * L * c0,
                                     self.alpha, L)
        self.vx = torch.tensor(vx, dtype=self.dtype, device=self.device)
        self.vy = torch.tensor(vy, dtype=self.dtype, device=self.device)
        ax0, ax1 = self.pis - 1, self.pie
        ay0, ay1 = self.pjs - 1, self.pje
        self.ax0, self.ax1, self.ay0, self.ay1 = ax0, ax1, ay0, ay1
        x_air = self.x_node[ax0:ax1]; y_air = self.y_node[ay0:ay1]
        dx_out = np.maximum(np.maximum(x_air[0] - X, X - x_air[-1]), 0.0) / 10.0
        dy_out = np.maximum(np.maximum(y_air[0] - Y, Y - y_air[-1]), 0.0) / 10.0
        r = np.minimum(np.sqrt(dx_out ** 2 + dy_out ** 2), 1.0)
        self.sigma = torch.tensor(6.0 * c0 / 10.0 * r ** 4, dtype=self.dtype, device=self.device)
        mask = np.zeros_like(X, dtype=bool); mask[ax0:ax1, ay0:ay1] = True
        self.air = torch.tensor(mask, dtype=torch.bool, device=self.device)
        # safety masks: high order only well inside the air block
        safe10 = np.zeros_like(mask); safe10[ax0 + 5:ax1 - 5, ay0 + 5:ay1 - 5] = True
        safe6 = np.zeros_like(mask); safe6[ax0 + 3:ax1 - 3, ay0 + 3:ay1 - 3] = True
        self.safe10 = torch.tensor(safe10, dtype=torch.bool, device=self.device)
        self.safe6 = torch.tensor(safe6, dtype=torch.bool, device=self.device)
        self.p = torch.zeros((self.nxp1, self.nyp1), dtype=self.dtype, device=self.device)
        self.wx = torch.zeros_like(self.p); self.wy = torch.zeros_like(self.p)
        self.P = torch.zeros_like(self.p); self.Wx = torch.zeros_like(self.p); self.Wy = torch.zeros_like(self.p)
        self.dt = 1.0 / (f * max(1, int(round(1.0 / (f * (0.1 / (c0 * (2 ** 0.5) / dx)))))))
        self.t_final = n_steps * self.dt

    def _d(self, f, axis, order):
        t = self.torch
        out = t.zeros_like(f)
        if axis == 0:
            n = f.shape[0]
            if order >= 2:
                out[1:-1] = (f[2:] - f[:-2]) / (2.0 * self.dx)
                out[0] = (f[1] - f[0]) / self.dx; out[-1] = (f[-1] - f[-2]) / self.dx
            if order >= 6 and n >= 7:
                out[3:-3] = (f[6:] - 9 * f[5:-1] + 45 * f[4:-2] - 45 * f[2:-4] + 9 * f[1:-5] - f[0:-6]) / (60 * self.dx)
            if order >= 10 and n >= 11:
                out[5:-5] = (2 * f[10:] - 25 * f[9:-1] + 150 * f[8:-2] - 600 * f[7:-3] + 2100 * f[6:-4]
                             - 2100 * f[4:-6] + 600 * f[3:-7] - 150 * f[2:-8] + 25 * f[1:-9] - 2 * f[0:-10]) / (2520 * self.dx)
        else:
            n = f.shape[1]
            if order >= 2:
                out[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2.0 * self.dy)
                out[:, 0] = (f[:, 1] - f[:, 0]) / self.dy; out[:, -1] = (f[:, -1] - f[:, -2]) / self.dy
            if order >= 6 and n >= 7:
                out[:, 3:-3] = (f[:, 6:] - 9 * f[:, 5:-1] + 45 * f[:, 4:-2] - 45 * f[:, 2:-4] + 9 * f[:, 1:-5] - f[:, :-6]) / (60 * self.dy)
            if order >= 10 and n >= 11:
                out[:, 5:-5] = (2 * f[:, 10:] - 25 * f[:, 9:-1] + 150 * f[:, 8:-2] - 600 * f[:, 7:-3] + 2100 * f[:, 6:-4]
                                - 2100 * f[:, 4:-6] + 600 * f[:, 3:-7] - 150 * f[:, 2:-8] + 25 * f[:, 1:-9] - 2 * f[:, :-10]) / (2520 * self.dy)
        return out

    def _deriv(self, f, axis):
        d10 = self._d(f, axis, self.order); d6 = self._d(f, axis, 6); d2 = self._d(f, axis, 2)
        d10[~self.safe10] = d6[~self.safe10]
        d10[~self.safe6] = d2[~self.safe6]
        d10[~self.air] = d2[~self.air]
        return d10

    def _rhs(self):
        p, wx, wy, P, Wx, Wy = self.p, self.wx, self.wy, self.P, self.Wx, self.Wy
        sig = self.sigma; b = 1.0 / self.rho0; kap = self.rho0 * self.c0 ** 2
        vx, vy = self.vx, self.vy
        dpdx = self._deriv(p, 0); dpdy = self._deriv(p, 1)
        dwxdx = self._deriv(wx, 0); dwxdy = self._deriv(wx, 1)
        dwydx = self._deriv(wy, 0); dwydy = self._deriv(wy, 1)
        dvxdx = self._deriv(vx, 0); dvxdy = self._deriv(vx, 1)
        dvydx = self._deriv(vy, 0); dvydy = self._deriv(vy, 1)
        dPdx = self._deriv(P, 0); dPdy = self._deriv(P, 1)
        dWxdx = self._deriv(Wx, 0); dWydy = self._deriv(Wy, 1)
        return (-vx * dpdx - vy * dpdy - kap * (dwxdx + dwydy) - sig * p,
                -vx * dwxdx - vy * dwxdy - b * dpdx - sig * wx,
                -vx * dwydx - vy * dwydy - b * dpdy - sig * wy,
                -kap * (dWxdx + dWydy) - sig * P,
                -b * dPdx - sig * Wx,
                -b * dPdy - sig * Wy)

    def solve(self, monitor_every=200):
        import time
        t = self.torch; t0 = time.perf_counter()
        y = [self.p.clone(), self.wx.clone(), self.wy.clone(), self.P.clone(), self.Wx.clone(), self.Wy.clone()]
        dt = self.dt; energies = []; steps = 0; maxp = 0.0
        def sety(v):
            self.p.copy_(v[0]); self.wx.copy_(v[1]); self.wy.copy_(v[2])
            self.P.copy_(v[3]); self.Wx.copy_(v[4]); self.Wy.copy_(v[5])
        def apply_stage_source(t):
            if self.stage_source:
                ss = np.sin(2.0 * np.pi * self.f * t)
                self.p[self.src_col, self.src_rows] = ss
                self.P[self.src_col, self.src_rows] = ss
        for step in range(1, self.n_steps + 1):
            apply_stage_source((step - 0.5) * dt)
            k1 = self._rhs(); sety([y[i] + 0.5 * dt * k1[i] for i in range(6)])
            apply_stage_source(step * dt)
            k2 = self._rhs(); sety([y[i] + 0.5 * dt * k2[i] for i in range(6)])
            apply_stage_source(step * dt)
            k3 = self._rhs(); sety([y[i] + dt * k3[i] for i in range(6)])
            apply_stage_source((step + 0.5) * dt)
            k4 = self._rhs()
            y = [y[i] + dt * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6.0 for i in range(6)]
            sety(y)
            s = np.sin(2.0 * np.pi * self.f * step * dt)
            self.p[self.src_col, self.src_rows] = s
            self.P[self.src_col, self.src_rows] = s
            steps = step
        self.torch.cuda.synchronize() if self.device.type == 'cuda' else None
        wall = time.perf_counter() - t0
        p_air = self.p[self.ax0:self.ax1, self.ay0:self.ay1].detach().cpu().numpy()
        P_air = self.P[self.ax0:self.ax1, self.ay0:self.ay1].detach().cpu().numpy()
        return {'n_steps_done': steps, 'wall_s': wall,
                'seconds_per_step': wall / steps if steps else float('nan'),
                'max_abs_p_final': float(np.max(np.abs(p_air))),
                'rms_p_air_final': float(np.sqrt(np.mean(p_air ** 2))),
                'scheme': 'torch_synchronized_rk4_high_order_sponge',
                'cfl': self.c0 * dt / self.dx * np.sqrt(2.0)}, p_air, P_air


# ---------------------------------------------------------------------------
# staggered face/node solver: a valid remap of the legacy layout
# ---------------------------------------------------------------------------
class TorchStaggeredFDTD2D:
    """Synchronized RK4 on the legacy staggered layout.

    p lives at integer nodes; wx lives at right x-faces (i+1/2,j), wy at
    top y-faces (i,j+1/2).  This is the layout implied by the legacy formulas:
    pressure uses ``(wx[i]-wx[i-1])/dx`` and velocity uses
    ``(p[i+1]-p[i])/dx`` (the legacy 2dx with b_i+b_{i+1} is equivalent for
    constant b).  A graded sponge is used outside the physical air block.
    """
    def __init__(self, f=500.0, Ma=0.7, c0=340.0, rho0=1.0, L=1.0,
                 alpha=1.256431, gamma_coef=2.8, dx=0.1, dy=0.1,
                 n_pml=None, n_buffer=None, n_steps=1, device='cuda',
                 dtype='float64'):
        import torch
        self.torch = torch; self.device = torch.device(device)
        self.dtype = torch.float64 if dtype == 'float64' else torch.float32
        self.f, self.Ma, self.c0, self.rho0 = float(f), float(Ma), float(c0), rho0
        self.dx, self.dy = float(dx), float(dy)
        self.n_pml = int(round(10.0 / dx)) if n_pml is None else int(n_pml)
        n_buffer = int(round(15.0 / dy)) if n_buffer is None else int(n_buffer)
        self.n_steps = int(n_steps)
        self.nx = self.ny = 2 * (self.n_pml + n_buffer)
        self.nxp1, self.nyp1 = self.nx + 1, self.ny + 1
        self.x_node = np.linspace(-self.nx / 2 * dx, self.nx / 2 * dx, self.nxp1)
        self.y_node = np.linspace(-self.ny / 2 * dy, self.ny / 2 * dy, self.nyp1)
        self.pis, self.pie = self.n_pml + 1, self.nx - self.n_pml + 1
        self.pjs, self.pje = self.n_pml + 1, self.ny - self.n_pml + 1
        self.ax0, self.ax1 = self.pis - 1, self.pie
        self.ay0, self.ay1 = self.pjs - 1, self.pje
        self.src_col = self.pis - 1
        self.src_rows = np.arange(self.pjs - 1, self.pje)
        self.velocity_source = False
        self.stage_source = True
        X, Y = np.meshgrid(self.x_node, self.y_node, indexing='ij')
        vx, vy = _lamb_oseen_vortex(X, Y, gamma_coef * Ma * np.pi * L * c0, alpha, L)
        self.vx = torch.tensor(vx, dtype=self.dtype, device=self.device)
        self.vy = torch.tensor(vy, dtype=self.dtype, device=self.device)
        x_air = self.x_node[self.ax0:self.ax1]; y_air = self.y_node[self.ay0:self.ay1]
        dxo = np.maximum(np.maximum(x_air[0] - X, X - x_air[-1]), 0.0) / 10.0
        dyo = np.maximum(np.maximum(y_air[0] - Y, Y - y_air[-1]), 0.0) / 10.0
        r = np.minimum(np.sqrt(dxo ** 2 + dyo ** 2), 1.0)
        self.sigma = torch.tensor(6.0 * c0 / 10.0 * r ** 4, dtype=self.dtype, device=self.device)
        self.p = torch.zeros((self.nxp1, self.nyp1), dtype=self.dtype, device=self.device)
        self.wx = torch.zeros_like(self.p); self.wy = torch.zeros_like(self.p)
        self.P = torch.zeros_like(self.p); self.Wx = torch.zeros_like(self.p); self.Wy = torch.zeros_like(self.p)
        self.dt = 1.0 / (f * 20)
        self.t_final = 0.0

    def _rhs(self):
        p, wx, wy, P, Wx, Wy = self.p, self.wx, self.wy, self.P, self.Wx, self.Wy
        b, kap = 1.0 / self.rho0, self.rho0 * self.c0 ** 2
        sig = self.sigma
        # interior node/face slices
        rhs_p = self.torch.zeros_like(p); rhs_wx = self.torch.zeros_like(wx); rhs_wy = self.torch.zeros_like(wy)
        rhs_P = self.torch.zeros_like(P); rhs_Wx = self.torch.zeros_like(Wx); rhs_Wy = self.torch.zeros_like(Wy)
        # p: backward divergence of face velocities
        rhs_p[1:-1, 1:-1] = (-kap * ((wx[1:-1, 1:-1] - wx[:-2, 1:-1]) / self.dx
                                     + (wy[1:-1, 1:-1] - wy[1:-1, :-2]) / self.dy)
                             - sig[1:-1, 1:-1] * p[1:-1, 1:-1])
        # wx at face (i+1/2,j): forward p difference
        rhs_wx[1:-1, 1:-1] = (-b * (p[2:, 1:-1] - p[1:-1, 1:-1]) / self.dx
                              - sig[1:-1, 1:-1] * wx[1:-1, 1:-1])
        # wy at face (i,j+1/2)
        rhs_wy[1:-1, 1:-1] = (-b * (p[1:-1, 2:] - p[1:-1, 1:-1]) / self.dy
                              - sig[1:-1, 1:-1] * wy[1:-1, 1:-1])
        rhs_P[1:-1, 1:-1] = (-kap * ((Wx[1:-1, 1:-1] - Wx[:-2, 1:-1]) / self.dx
                                     + (Wy[1:-1, 1:-1] - Wy[1:-1, :-2]) / self.dy)
                             - sig[1:-1, 1:-1] * P[1:-1, 1:-1])
        rhs_Wx[1:-1, 1:-1] = (-b * (P[2:, 1:-1] - P[1:-1, 1:-1]) / self.dx
                              - sig[1:-1, 1:-1] * Wx[1:-1, 1:-1])
        rhs_Wy[1:-1, 1:-1] = (-b * (P[1:-1, 2:] - P[1:-1, 1:-1]) / self.dy
                              - sig[1:-1, 1:-1] * Wy[1:-1, 1:-1])
        if self.Ma != 0.0:
            # full linearized-Euler background terms on the staggered layout:
            # (U.grad)w_x + (w.grad)U_x, and the corresponding y equation.
            # face values are averages of nodal background fields
            vx_fx = 0.5 * (self.vx[1:-1, 1:-1] + self.vx[2:, 1:-1])
            vy_fx = 0.5 * (self.vy[1:-1, 1:-1] + self.vy[2:, 1:-1])
            vx_fy = 0.5 * (self.vx[1:-1, 1:-1] + self.vx[1:-1, 2:])
            vy_fy = 0.5 * (self.vy[1:-1, 1:-1] + self.vy[1:-1, 2:])
            dpdx = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * self.dx)
            dpdy = (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * self.dy)
            dwxdx = (wx[2:, 1:-1] - wx[:-2, 1:-1]) / (2 * self.dx)
            dwxdy = (wx[1:-1, 2:] - wx[1:-1, :-2]) / (2 * self.dy)
            dwydx = (wy[2:, 1:-1] - wy[:-2, 1:-1]) / (2 * self.dx)
            dwydy = (wy[1:-1, 2:] - wy[1:-1, :-2]) / (2 * self.dy)
            dvxdx = (self.vx[2:, 1:-1] - self.vx[:-2, 1:-1]) / (2 * self.dx)
            dvxdy = (self.vx[1:-1, 2:] - self.vx[1:-1, :-2]) / (2 * self.dy)
            dvydx = (self.vy[2:, 1:-1] - self.vy[:-2, 1:-1]) / (2 * self.dx)
            dvydy = (self.vy[1:-1, 2:] - self.vy[1:-1, :-2]) / (2 * self.dy)
            # average the transverse face velocity to the other face
            wy_at_wx = 0.25 * (wy[1:-1, 1:-1] + wy[2:, 1:-1]
                               + wy[1:-1, :-2] + wy[2:, :-2])
            wx_at_wy = 0.25 * (wx[1:-1, 1:-1] + wx[1:-1, 2:]
                               + wx[:-2, 1:-1] + wx[:-2, 2:])
            rhs_p[1:-1, 1:-1] += (-self.vx[1:-1, 1:-1] * dpdx
                                  - self.vy[1:-1, 1:-1] * dpdy)
            rhs_wx[1:-1, 1:-1] += (-(vx_fx * dwxdx + vy_fx * dwxdy)
                                   - (wx[1:-1, 1:-1] * dvxdx + wy_at_wx * dvxdy))
            rhs_wy[1:-1, 1:-1] += (-(vx_fy * dwydx + vy_fy * dwydy)
                                   - (wx_at_wy * dvydx + wy[1:-1, 1:-1] * dvydy))
        return rhs_p, rhs_wx, rhs_wy, rhs_P, rhs_Wx, rhs_Wy

    def solve(self):
        import time
        t = self.torch; t0 = time.perf_counter()
        y = [self.p.clone(), self.wx.clone(), self.wy.clone(), self.P.clone(), self.Wx.clone(), self.Wy.clone()]
        dt = self.dt
        def sety(v):
            self.p.copy_(v[0]); self.wx.copy_(v[1]); self.wy.copy_(v[2])
            self.P.copy_(v[3]); self.Wx.copy_(v[4]); self.Wy.copy_(v[5])
        def apply_stage_source(t):
            if self.stage_source:
                ss = np.sin(2.0 * np.pi * self.f * t)
                self.p[self.src_col, self.src_rows] = ss
                self.P[self.src_col, self.src_rows] = ss
        for step in range(1, self.n_steps + 1):
            t0 = (step - 1) * dt
            apply_stage_source(t0)
            k1 = self._rhs(); sety([y[i] + 0.5 * dt * k1[i] for i in range(6)])
            apply_stage_source(t0 + 0.5 * dt)
            k2 = self._rhs(); sety([y[i] + 0.5 * dt * k2[i] for i in range(6)])
            apply_stage_source(t0 + 0.5 * dt)
            k3 = self._rhs(); sety([y[i] + dt * k3[i] for i in range(6)])
            apply_stage_source(t0 + dt)
            k4 = self._rhs()
            y = [y[i] + dt * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6.0 for i in range(6)]
            sety(y)
            s = np.sin(2.0 * np.pi * self.f * (t0 + dt))
            self.p[self.src_col, self.src_rows] = s
            self.P[self.src_col, self.src_rows] = s
            y[0][self.src_col, self.src_rows] = s
            y[3][self.src_col, self.src_rows] = s
            if self.velocity_source:
                wsrc = s / (self.rho0 * self.c0)
                self.wx[self.src_col, self.src_rows] = wsrc
                self.Wx[self.src_col, self.src_rows] = wsrc
            if getattr(self, 'probe_index', None) is not None:
                i, j = self.probe_index
                if not hasattr(self, 'probe_trace'):
                    self.probe_trace = []
                self.probe_trace.append(float(self.p[i, j].item()))
        self.torch.cuda.synchronize() if self.device.type == 'cuda' else None
        wall = time.perf_counter() - t0
        p_air = self.p[self.ax0:self.ax1, self.ay0:self.ay1].detach().cpu().numpy()
        P_air = self.P[self.ax0:self.ax1, self.ay0:self.ay1].detach().cpu().numpy()
        return {'n_steps_done': self.n_steps, 'wall_s': wall,
                'seconds_per_step': wall / self.n_steps if self.n_steps else float('nan'),
                'max_abs_p_final': float(np.max(np.abs(p_air))),
                'rms_p_air_final': float(np.sqrt(np.mean(p_air ** 2))),
                'probe_trace': getattr(self, 'probe_trace', None),
                'scheme': 'torch_staggered_synchronized_rk4'}, p_air, P_air


# ---------------------------------------------------------------------------
# production-RHS verification (uniform flow + constant shear)
# ---------------------------------------------------------------------------
def validate_production_rhs(device='cpu') -> dict[str, Any]:
    """Compare the production staggered RHS with analytic linearized-Euler RHS.

    Uniform flow checks all three wave terms.  A constant shear background
    additionally exercises the four background-gradient couplings that were
    missing in the v1 implementation.
    """
    import torch
    out = {}
    dx = 0.2
    s = TorchStaggeredFDTD2D(f=500.0, Ma=0.7, dx=dx, dy=dx, n_pml=10,
                             n_buffer=10, n_steps=1, device=device)
    s.sigma.zero_()
    X, Y = np.meshgrid(s.x_node, s.y_node, indexing='ij')
    c0, rho = s.c0, s.rho0
    kap = rho * c0 * c0
    b = 1.0 / rho
    k = 2.0 * np.pi / 2.0
    U0 = 0.2 * c0
    phase = k * (X - (U0 + c0) * 0.0)
    p = np.cos(phase)
    wx = (c0 / kap) * np.cos(k * (X + dx / 2.0))
    wy = np.zeros_like(p)
    s.p[:] = torch.tensor(p, dtype=s.dtype, device=s.device)
    s.wx[:] = torch.tensor(wx, dtype=s.dtype, device=s.device)
    s.wy[:] = torch.tensor(wy, dtype=s.dtype, device=s.device)
    s.P.zero_(); s.Wx.zero_(); s.Wy.zero_()
    s.vx[:] = torch.tensor(np.full_like(X, U0), dtype=s.dtype, device=s.device)
    s.vy.zero_()
    rhs = s._rhs()
    p_t = rhs[0].detach().cpu().numpy()
    wx_t = rhs[1].detach().cpu().numpy()
    wy_t = rhs[2].detach().cpu().numpy()
    # independent discrete right-hand side for U=(U0,0), using the same
    # production stencils but explicitly assembled outside the production loop
    pn = np.cos(k * X)
    wxn = (c0 / kap) * np.cos(k * (X + dx / 2.0))
    p_x = (pn[2:, 1:-1] - pn[:-2, 1:-1]) / (2 * dx)
    wx_x = (wxn[2:, 1:-1] - wxn[:-2, 1:-1]) / (2 * dx)
    div = (wxn[1:-1, 1:-1] - wxn[:-2, 1:-1]) / dx
    p_t_ex = -U0 * p_x - kap * div
    wx_t_ex = -U0 * wx_x - b * (pn[2:, 1:-1] - pn[1:-1, 1:-1]) / dx
    # align manually assembled arrays to interior (1:-1)
    p_t_prod = p_t[1:-1, 1:-1]
    wx_t_prod = wx_t[1:-1, 1:-1]
    def rel(a, b_):
        return float(np.linalg.norm(a - b_) / max(np.linalg.norm(b_), 1e-30))
    out['uniform_flow'] = {
        'p_rel_error': rel(p_t_prod, p_t_ex),
        'wx_rel_error': rel(wx_t_prod, wx_t_ex),
        'wy_max_abs': float(np.max(np.abs(wy_t[1:-1, 1:-1]))),
        'note': 'independent discrete assembly uses the same declared stencils; not a continuum analytic derivative',
    }
    # constant shear U=(a y, 0): nonzero dvx/dy, four coupling terms
    a_shear = 0.3 * c0
    s.p[:] = torch.tensor(np.cos(k * X) * np.cos(k * Y), dtype=s.dtype, device=s.device)
    s.wx[:] = torch.tensor(0.03 * np.sin(k * X) * np.cos(k * Y), dtype=s.dtype, device=s.device)
    s.wy[:] = torch.tensor(0.03 * np.cos(k * X) * np.sin(k * Y), dtype=s.dtype, device=s.device)
    s.vx[:] = torch.tensor(a_shear * Y, dtype=s.dtype, device=s.device)
    s.vy.zero_()
    rhs = s._rhs()
    # only check finite and magnitudes; full face-interpolation analytic check is
    # reported separately in the probe's component tests
    out['constant_shear'] = {
        'p_finite': bool(np.isfinite(rhs[0].detach().cpu().numpy()).all()),
        'wx_finite': bool(np.isfinite(rhs[1].detach().cpu().numpy()).all()),
        'wy_finite': bool(np.isfinite(rhs[2].detach().cpu().numpy()).all()),
        'nonzero_shear_term': bool(float(np.max(np.abs(rhs[1].detach().cpu().numpy()[1:-1, 1:-1]))) > 0),
    }
    return out


# ---------------------------------------------------------------------------
# high-order staggered operators (generic moment-condition weights)
# ---------------------------------------------------------------------------
def _stag_weights(order: int) -> np.ndarray:
    """Weights for a first derivative at a half-node using `order` face values.

    Positions are relative to the target point: -order/2+1/2 ... order/2-1/2.
    Solved from moment conditions sum w_k x_k^p = delta_{p,1} p!.
    """
    n = int(order)
    pos = -n / 2.0 + 0.5 + np.arange(n)
    A = np.vstack([pos ** p for p in range(n)])
    rhs = np.zeros(n); rhs[1] = 1.0
    return np.linalg.solve(A, rhs)


def _apply_stag_div(f, dx, axis: int, w: np.ndarray):
    """Divergence-like derivative at nodes from face values (torch/NumPy array)."""
    import torch
    n = len(w); s = n // 2
    out = torch.zeros_like(f) if isinstance(f, torch.Tensor) else np.zeros_like(f)
    if axis == 0:
        if f.shape[0] > 2 * s:
            out[s:-s] = sum(w[k] * f[k:f.shape[0] - 2 * s + k] for k in range(n)) / dx
    else:
        if f.shape[1] > 2 * s:
            out[:, s:-s] = sum(w[k] * f[:, k:f.shape[1] - 2 * s + k] for k in range(n)) / dx
    return out


def _apply_stag_grad(f, dx, axis: int, w: np.ndarray):
    """Gradient-like derivative at faces from node values."""
    import torch
    n = len(w); s = n // 2 - 1
    out = torch.zeros_like(f) if isinstance(f, torch.Tensor) else np.zeros_like(f)
    if axis == 0:
        if f.shape[0] > n - 1:
            out[s:f.shape[0] - s - 2] = sum(
                w[k] * f[k:f.shape[0] - n + k] for k in range(n)) / dx
    else:
        if f.shape[1] > n - 1:
            out[:, s:f.shape[1] - s - 2] = sum(
                w[k] * f[:, k:f.shape[1] - n + k] for k in range(n)) / dx
    return out



def _central_weights(points=9):
    pos = np.arange(points) - (points // 2)
    A = np.vstack([pos ** p for p in range(points)])
    rhs = np.zeros(points); rhs[1] = 1.0
    return np.linalg.solve(A, rhs)


def _apply_central(f, dx, axis, w):
    import torch
    n = len(w); s = n // 2
    out = torch.zeros_like(f) if isinstance(f, torch.Tensor) else np.zeros_like(f)
    if axis == 0:
        if f.shape[0] > n - 1:
            count = f.shape[0] - 2 * s
            out[s:f.shape[0] - s] = sum(w[k] * f[k:k + count] for k in range(n)) / dx
    else:
        if f.shape[1] > n - 1:
            count = f.shape[1] - 2 * s
            out[:, s:f.shape[1] - s] = sum(w[k] * f[:, k:k + count] for k in range(n)) / dx
    return out

class HighOrderStaggeredFDTD2D(TorchStaggeredFDTD2D):
    """Staggered solver with a user-selectable high-order gradient/divergence pair.

    The 4th-order pair is the ``9/8, -1/24`` interface check; higher orders use
    moment-condition weights.  Advection/background gradients remain explicit;
    the sponge is kept and is *not* called a PML.
    """
    def __init__(self, *args, order=4, sponge_scale=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.sponge_scale = float(sponge_scale)
        if self.sponge_scale != 1.0:
            self.sigma.mul_(self.sponge_scale)
        self.order = int(order)
        self._w = _stag_weights(self.order)
        self._w4 = _stag_weights(4)
        self._wc = _central_weights(9)

    def _div(self, f, dx, axis):
        return _apply_stag_div(f, dx, axis, self._w)

    def _grad(self, f, dx, axis):
        return _apply_stag_grad(f, dx, axis, self._w)

    def _cen(self, f, dx, axis):
        return _apply_central(f, dx, axis, self._wc)

    def _rhs(self):
        p, wx, wy, P, Wx, Wy = self.p, self.wx, self.wy, self.P, self.Wx, self.Wy
        b, kap = 1.0 / self.rho0, self.rho0 * self.c0 ** 2
        sig = self.sigma
        rhs_p = self.torch.zeros_like(p); rhs_wx = self.torch.zeros_like(wx); rhs_wy = self.torch.zeros_like(wy)
        rhs_P = self.torch.zeros_like(P); rhs_Wx = self.torch.zeros_like(Wx); rhs_Wy = self.torch.zeros_like(Wy)
        # high-order staggered divergence/gradient in the whole grid; sponge
        # damps outside air.  Boundary rows are kept zero by the operator.
        rhs_p = -kap * (self._div(wx, self.dx, 0) + self._div(wy, self.dy, 1)) - sig * p
        rhs_wx = -b * self._grad(p, self.dx, 0) - sig * wx
        rhs_wy = -b * self._grad(p, self.dy, 1) - sig * wy
        rhs_P = -kap * (self._div(Wx, self.dx, 0) + self._div(Wy, self.dy, 1)) - sig * P
        rhs_Wx = -b * self._grad(P, self.dx, 0) - sig * Wx
        rhs_Wy = -b * self._grad(P, self.dy, 1) - sig * Wy
        if self.Ma != 0.0:
            vx_fx = 0.5 * (self.vx[1:-1, 1:-1] + self.vx[2:, 1:-1])
            vy_fx = 0.5 * (self.vy[1:-1, 1:-1] + self.vy[2:, 1:-1])
            vx_fy = 0.5 * (self.vx[1:-1, 1:-1] + self.vx[1:-1, 2:])
            vy_fy = 0.5 * (self.vy[1:-1, 1:-1] + self.vy[1:-1, 2:])
            dpdx = self._cen(p, self.dx, 0)[1:-1, 1:-1]
            dpdy = self._cen(p, self.dy, 1)[1:-1, 1:-1]
            dwxdx = self._cen(wx, self.dx, 0)[1:-1, 1:-1]
            dwxdy = self._cen(wx, self.dy, 1)[1:-1, 1:-1]
            dwydx = self._cen(wy, self.dx, 0)[1:-1, 1:-1]
            dwydy = self._cen(wy, self.dy, 1)[1:-1, 1:-1]
            dvxdx_node = self._cen(self.vx, self.dx, 0)
            dvxdy_node = self._cen(self.vx, self.dy, 1)
            dvydx_node = self._cen(self.vy, self.dx, 0)
            dvydy_node = self._cen(self.vy, self.dy, 1)
            dvxdx = 0.5 * (dvxdx_node[1:-1, 1:-1] + dvxdx_node[2:, 1:-1])
            # cross gradients live at the same face y (wx) / x (wy) as the
            # transported velocity, not at the cell corner
            dvxdy = 0.5 * (dvxdy_node[1:-1, 1:-1] + dvxdy_node[2:, 1:-1])
            dvydx = 0.5 * (dvydx_node[1:-1, 1:-1] + dvydx_node[1:-1, 2:])
            dvydy = 0.5 * (dvydy_node[1:-1, 1:-1] + dvydy_node[1:-1, 2:])
            wy_at_wx = 0.25 * (wy[1:-1, 1:-1] + wy[2:, 1:-1]
                               + wy[1:-1, :-2] + wy[2:, :-2])
            wx_at_wy = 0.25 * (wx[1:-1, 1:-1] + wx[1:-1, 2:]
                               + wx[:-2, 1:-1] + wx[:-2, 2:])
            rhs_p[1:-1, 1:-1] += (-self.vx[1:-1, 1:-1] * dpdx
                                  - self.vy[1:-1, 1:-1] * dpdy)
            rhs_wx[1:-1, 1:-1] += (-(vx_fx * dwxdx + vy_fx * dwxdy)
                                   - (wx[1:-1, 1:-1] * dvxdx + wy_at_wx * dvxdy))
            rhs_wy[1:-1, 1:-1] += (-(vx_fy * dwydx + vy_fy * dwydy)
                                   - (wx_at_wy * dvydx + wy[1:-1, 1:-1] * dvydy))
        return rhs_p, rhs_wx, rhs_wy, rhs_P, rhs_Wx, rhs_Wy


# ---------------------------------------------------------------------------
# quantitative manufactured check for all four background-gradient terms
# ---------------------------------------------------------------------------
def validate_mms_gradients(device='cpu', dx_values=(0.2, 0.1, 0.05)) -> dict[str, Any]:
    """Polynomial manufactured solution with nonzero u',v',p' and full shear.

    The forcing is the analytic linearized-Euler residual; code RHS is compared
    with the independent analytic spatial residual on the same staggered points.
    """
    import torch
    results = {}
    for dx in dx_values:
        out = {}
        for case in ('dUx_dx', 'dUx_dy', 'dUy_dx', 'dUy_dy', 'all'):
            s = HighOrderStaggeredFDTD2D(f=500.0, Ma=0.7, dx=dx, dy=dx,
                                         n_pml=5, n_buffer=5, n_steps=1,
                                         device=device, order=8)
            X, Y = np.meshgrid(s.x_node, s.y_node, indexing='ij')
            ax = 0.4 if case in ('dUx_dx', 'all') else 0.0
            ay = 0.35 if case in ('dUx_dy', 'all') else 0.0
            bx = 0.30 if case in ('dUy_dx', 'all') else 0.0
            by = 0.45 if case in ('dUy_dy', 'all') else 0.0
            Ux0, Uy0 = 0.2 * s.c0, -0.1 * s.c0
            Ux = Ux0 + ax * X + ay * Y
            Uy = Uy0 + bx * X + by * Y
            # nonzero asymmetric u', v', p' with linear spatial structure
            px, py, wx0, wy0 = 0.31, -0.27, 0.22, -0.19
            wxx, wxy = 0.13, -0.11
            wyx, wyy = -0.17, 0.09
            p = px * X + py * Y + 0.07
            # face values must be sampled at the staggered positions, not nodes
            Xf, Yf = X + dx / 2.0, Y + dx / 2.0
            wx = wx0 + wxx * Xf + wxy * Y
            wy = wy0 + wyx * X + wyy * Yf
            s.p[:] = torch.tensor(p, dtype=s.dtype, device=s.device)
            s.wx[:] = torch.tensor(wx, dtype=s.dtype, device=s.device)
            s.wy[:] = torch.tensor(wy, dtype=s.dtype, device=s.device)
            s.P.zero_(); s.Wx.zero_(); s.Wy.zero_()
            s.vx[:] = torch.tensor(Ux, dtype=s.dtype, device=s.device)
            s.vy[:] = torch.tensor(Uy, dtype=s.dtype, device=s.device)
            s.sigma.zero_()
            rhs = s._rhs()
            # analytic spatial residual at nodes (p) and faces (wx/wy)
            kap = s.rho0 * s.c0 ** 2; b = 1.0 / s.rho0
            p_ex = -Ux * px - Uy * py - kap * (wxx + wyy)
            # advecting background velocity is sampled at the same face as wx/wy
            wx_ex = (-(Ux0 + ax * Xf + ay * Y) * wxx
                     - (Uy0 + bx * Xf + by * Y) * wxy
                     - (wx * ax + (wy0 + wyx * Xf + wyy * Y) * ay) - b * px)
            wy_ex = (-(Ux0 + ax * X + ay * Yf) * wyx
                     - (Uy0 + bx * X + by * Yf) * wyy
                     - ((wx0 + wxx * X + wxy * Yf) * bx + wy * by) - b * py)
            # interior masks; code/outputs align at staggered points
            sl = slice(6, -6)
            p_code = rhs[0][sl, sl].detach().cpu().numpy()
            w = rhs[1][sl, sl].detach().cpu().numpy()
            wy_code = rhs[2][sl, sl].detach().cpu().numpy()
            p_ref = p_ex[sl, sl]; wx_ref = wx_ex[sl, sl]; wy_ref = wy_ex[sl, sl]
            def metrics(a, ref):
                d = a - ref
                return {'l2': float(np.linalg.norm(d)), 'max': float(np.max(np.abs(d))),
                        'rel_l2': float(np.linalg.norm(d) / max(np.linalg.norm(ref), 1e-30))}
            out[case] = {'p': metrics(p_code, p_ref), 'wx': metrics(w, wx_ref),
                         'wy': metrics(wy_code, wy_ref),
                         'forcing_nonzero': bool(np.max(np.abs(wx_ref)) > 0 or np.max(np.abs(wy_ref)) > 0)}
        results[str(dx)] = out
    return results
