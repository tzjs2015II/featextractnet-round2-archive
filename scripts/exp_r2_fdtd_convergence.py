"""scripts/exp_r2_fdtd_convergence.py — R2-FDTD-CONV-01 pressure space/time convergence.

Purpose
-------
Single-configuration entry for the R2-FDTD-CONV-01 reviewer-validation experiment
(返稿修订 · 实验计划.md, section ``R2-FDTD-CONV-01``). It runs exactly ONE fixed
configuration

    --case {low100,high500} --resolution {coarse,mid,fine,mid_dt2,mid_dt4}

with the *real production solver* (``src.fdtd.run_case``, the same RK4 / constant-
sigma PML / Lamb-Oseen vortex code that generated ``data/acoustic``), freezes the
protocol JSON before the solve, saves the real final-time pressure snapshots, and
(with ``--aggregate-only``) verifies protocol identity / node alignment and reports
space/time convergence metrics for the complete 2 cases x 5 resolutions = 10 solves.

Fixed physical protocol (from 实验计划.md R2-FDTD-CONV-01; NOT adjustable here)
-------------------------------------------------------------------------------
* cases: (f=100 Hz, Ma=0.3) and (f=500 Hz, Ma=0.7); hard left line source at x=-15 m
* c0=340 m/s, rho0=1, L=1, alpha=1.256431, gamma_coef=2.8, sigma=100
* air domain [-15,15]^2, PML physical thickness 10 m per side
* space: dx=dy=1/8, 1/16, 1/32 m -> 401/801/1601 nodes total, 241/481/961 air nodes
* termination: t_final = Ncycle/f with Ncycle = ceil(f*4*30/c0) (>= 4 air crossings)
* time stepping: coarse steps-per-period s0 = ceil(c0*sqrt(2)/(f*dx_coarse*0.1));
  mid = 2*s0, fine = 4*s0; the extra temporal points on the fixed mid grid are
  mid_dt2 = 4*s0 and mid_dt4 = 8*s0. dt = 1/(f*spp) exactly, n_steps = Ncycle*spp,
  so t_final = Ncycle/f and the final source phase is a whole number of cycles for
  every configuration. ``courant_factor`` is back-solved from the required dt and
  the realized dt / t_final / phase / source position are verified after the solve.
* coarse 500 Hz has only 5.44 points per wavelength (under-resolved diagnostic);
  it is kept, not excluded.

Output layout
-------------
``<out-root>/<case>/<resolution>/``

    protocol.json   written BEFORE the solve (fixed protocol + identity + plan + hashes)
    fields.npz      p_total, p_incident, p_scattered, x, y (+ diagnostics arrays,
                    + optional cropped phasor when --save-phasor)
    meta.json       actual solver params, achieved dt/t_final/phase, checks, diagnostics
    report.md       human-readable per-run report

``<out-root>/aggregate.json`` / ``aggregate.md``
    written only after all 10 runs are present, ok and identity/coordinate verified.
``<out-root>/aggregate_pending.json``
    written instead when runs are missing/failed (pending listed, nothing fabricated).

Guards
------
* Solving requires SLURM (``SLURM_JOB_ID``). Outside SLURM the solve entry REFUSES:
  the login node must not run the heavy FDTD solves. ``--dry-run`` only prints the
  fixed protocol and writes nothing; ``--aggregate-only`` is light I/O analysis and
  is allowed outside SLURM.
* Existing key files are never overwritten (failure products are preserved); use a
  new ``--out-root``.
* Deployed frozen-snapshot run: when this entry is executed from
  ``<canonical>/results/round2_execution_<version>/`` (the v11+ deployment layout,
  where ``<canonical>`` is the real 返稿修订工作区), the only accepted output root
  outside that snapshot is ``<canonical>/results/round2_fdtd_convergence``; the
  canonical path is derived only for that recognizable layout (see
  :func:`resolve_out_root`). Arbitrary external directories stay refused.
* ``results/registry.json`` receives an independent ID only after the complete 10/10
  solve set passes identity/coordinate checks. Missing items are reported as pending,
  never fabricated.
* If the solver interface cannot realize the fixed final time / phase (realized dt,
  t_final or source position deviates beyond tolerance), the run is marked
  ``interface_mismatch`` and reported; physical parameters are NOT adjusted.

Interpretation boundary
-----------------------
The fine grid is a *numerical reference*, not an analytic solution; one final-time
snapshot at >=4 air crossings is not a steady-state measurement. Finite / steady /
blow-up diagnostics from the solver are recorded verbatim but do NOT substitute for
the pressure convergence comparison. Observed order is reported only for a monotone
error sequence with a non-zero denominator; otherwise it is explicitly ``undefined``.
A positive or negative outcome is not an acceptance criterion.

Usage
-----
  python3 scripts/exp_r2_fdtd_convergence.py --dry-run
  python3 scripts/exp_r2_fdtd_convergence.py --case low100 --resolution coarse   # SLURM
  python3 scripts/exp_r2_fdtd_convergence.py --aggregate-only                   # after 10/10
"""

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import platform
import socket
import sys
import time

import numpy as np

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WS not in sys.path:
    sys.path.insert(0, WS)

from src.fdtd import run_case  # noqa: E402
from src.fdtd.python_fdtd import fit_complex_amplitude  # noqa: E402
from src.core.registry import append_registry  # noqa: E402

# ---------------------------------------------------------------------------
# fixed identifiers / paths
# ---------------------------------------------------------------------------
TASK_ID = 'R2-FDTD-CONV-01'
PROTOCOL_VERSION = 'R2_FDTD_CONV_v1'
SCRIPT_REL = 'scripts/exp_r2_fdtd_convergence.py'
SOLVER_REL = 'src/fdtd/python_fdtd.py'
CONFIG_REL = 'configs/acoustic_cases.json'
DEFAULT_OUT_ROOT = 'results/round2_fdtd_convergence'

# Frozen-snapshot deployment layout (job 50906 pilot record):
#   <canonical>/results/round2_execution_<version>/...   <- frozen source snapshot = WS
#   <canonical>/results/round2_fdtd_convergence           <- canonical write-back output
FROZEN_SNAPSHOT_PREFIX = 'round2_execution_'
CANONICAL_WORKSPACE_DIRNAME = '返稿修订工作区'
CANONICAL_OUTPUT_SUBPATH = ('results', 'round2_fdtd_convergence')
RULES_ENTRY_RELPATH_FROM_CANONICAL = os.path.join(os.pardir, os.pardir, 'AGENTS.md')

# ---------------------------------------------------------------------------
# fixed physical / numerical protocol (实验计划.md R2-FDTD-CONV-01)
# ---------------------------------------------------------------------------
C0 = 340.0                 # m/s
RHO0 = 1.0                 # kg/m^3
CORE_L = 1.0               # m, Lamb-Oseen core-radius scale
ALPHA = 1.256431           # Lamb-Oseen profile exponent
GAMMA_COEF = 2.8           # circulation prefactor
SIGMA_PML = 100.0          # constant PML damping of the production solver
AIR_HALF_M = 15.0          # air domain is [-15, 15]^2
PML_THICKNESS_M = 10.0     # PML physical thickness per side
AIR_SPAN_M = 2.0 * AIR_HALF_M
N_TRANSIT = 4.0            # >= 4 crossings of the air domain at c0
CFL_MAX = 0.10             # required effective CFL target
PPW_MIN = 10.0             # points-per-wavelength reference (under-resolution flag only)
SOURCE_SIDE = 'left'
SOURCE_Y_FRAC = (0.0, 1.0)
N_PHASE = 8                # phase-locked snapshots for the optional phasor

S0_DX_DENOM = 8            # s0 is defined on the coarse dx = 1/8 m grid

CASES = {
    'low100': {'f_hz': 100.0, 'Ma': 0.3, 'config_case_id': '100_0.3'},
    'high500': {'f_hz': 500.0, 'Ma': 0.7, 'config_case_id': '500_0.7'},
}

# series: 'space' = dx refinement; 'time' = dt refinement on the fixed mid grid
RESOLUTIONS = {
    'coarse': {'dx_denom': 8, 'spp_mult': 1, 'series': 'space'},
    'mid': {'dx_denom': 16, 'spp_mult': 2, 'series': 'space'},
    'fine': {'dx_denom': 32, 'spp_mult': 4, 'series': 'space'},
    'mid_dt2': {'dx_denom': 16, 'spp_mult': 4, 'series': 'time'},
    'mid_dt4': {'dx_denom': 16, 'spp_mult': 8, 'series': 'time'},
}
SPATIAL_SERIES = ('coarse', 'mid', 'fine')
TEMPORAL_SERIES = ('mid', 'mid_dt2', 'mid_dt4')
PRIMARY_FIELD = 'p_total'
SECONDARY_FIELDS = ('p_scattered',)
FIELDS = (PRIMARY_FIELD,) + SECONDARY_FIELDS

KEY_FILES = ('protocol.json', 'fields.npz', 'meta.json', 'report.md',
             'diagnostics.npz')

# convergence protocol constants
REFINEMENT_RATIO = 2.0     # dx (and dt) ratio between consecutive series points
COORD_ATOL_M = 1e-12       # nested-grid node coordinates must agree to this
ALIGN_SPACING_RTOL = 1e-9  # target spacing must be an integer multiple of the axis
DT_REL_TOL = 1e-9
T_REL_TOL = 1e-9
PHASE_CYCLE_TOL = 1e-9
SOURCE_POS_TOL_M = 1e-12
PHASOR_MIN_SNAPSHOTS = 4


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------
class R2FdtdError(RuntimeError):
    """Base class for controlled refusals / protocol errors."""


class Refused(R2FdtdError):
    """A guard refused the requested operation."""


class SolveRefused(Refused):
    """Solving is not allowed in the current environment (non-SLURM)."""


class ProtocolError(R2FdtdError):
    """A protocol/identity/solver-interface requirement is not satisfied."""


class IdentityMismatch(ProtocolError):
    """Two runs that must share a protocol identity do not."""


class IncompleteResults(R2FdtdError):
    """The 10-solve set is not complete/ok; aggregation is deferred."""


class RunFailed(R2FdtdError):
    """One solve produced a non-ok result (evidence retained)."""


# ---------------------------------------------------------------------------
# generic helpers
# ---------------------------------------------------------------------------
def now_utc():
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds')


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def ws_path(path):
    """Resolve a user path against the workspace root (relative paths are WS-relative)."""
    path = os.path.expanduser(str(path))
    if not os.path.isabs(path):
        path = os.path.join(WS, path)
    return os.path.abspath(path)


def rel_ws(path):
    return os.path.relpath(os.path.abspath(path), WS).replace(os.sep, '/')


def _json_safe(obj):
    """Convert numpy scalars and non-finite floats so the JSON stays strict."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def write_json(path, doc):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(_json_safe(doc), fh, indent=2, ensure_ascii=False, allow_nan=False)
        fh.write('\n')
    os.replace(tmp, path)


def read_json(path):
    with open(path, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def write_text(path, text):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        fh.write(text)
    os.replace(tmp, path)


def write_npz(path, arrays):
    tmp = path + '.tmp.npz'
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def environment_snapshot():
    slurm = {
        'SLURM_JOB_ID': os.environ.get('SLURM_JOB_ID'),
        'SLURM_JOBID': os.environ.get('SLURM_JOBID'),
        'SLURM_JOB_NODELIST': os.environ.get('SLURM_JOB_NODELIST'),
        'SLURM_JOB_PARTITION': os.environ.get('SLURM_JOB_PARTITION'),
        'SLURM_CPUS_PER_TASK': os.environ.get('SLURM_CPUS_PER_TASK'),
    }
    return {
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'python': platform.python_version(),
        'numpy': np.__version__,
        'cpu_count': os.cpu_count(),
        'slurm': slurm,
    }


# ---------------------------------------------------------------------------
# environment / out-root guards
# ---------------------------------------------------------------------------
def slurm_job_id(env=None):
    env = os.environ if env is None else env
    for key in ('SLURM_JOB_ID', 'SLURM_JOBID'):
        val = env.get(key)
        if val:
            return str(val)
    return None


def check_solve_environment(env=None):
    """Refuse to solve outside SLURM (login nodes must not run heavy FDTD solves)."""
    env = os.environ if env is None else env
    job_id = slurm_job_id(env)
    if not job_id:
        raise SolveRefused(
            'refusing to run the FDTD solve outside SLURM (SLURM_JOB_ID is not set); '
            'login nodes must not run the heavy solves. Submit via sbatch/srun, or use '
            '--dry-run to inspect the fixed physical protocol without solving.')
    return {
        'slurm_job_id': job_id,
        'slurm_job_nodelist': env.get('SLURM_JOB_NODELIST'),
        'slurm_job_partition': env.get('SLURM_JOB_PARTITION'),
        'slurm_cpus_per_task': env.get('SLURM_CPUS_PER_TASK'),
    }


def canonical_output_root_from_frozen_ws(ws):
    """Derive the canonical write-back root when ``ws`` is a deployed frozen snapshot.

    Deployment layout (confirmed by the job 50906 pilot record):

        <canonical>/results/round2_execution_<version>/...   <- frozen source snapshot = WS
        <canonical>/results/round2_fdtd_convergence           <- canonical output root

    A snapshot is only recognized when all of the following hold, so that no other
    external directory can be targeted:

    * ``ws`` sits directly under a directory literally named ``results``;
    * ``ws`` basename starts with ``round2_execution_``;
    * the derived canonical directory's *real* basename is ``返稿修订工作区``;
    * the original project rules entry still exists at the project root
      (``<canonical>/../../AGENTS.md``; it is not part of the frozen snapshot).

    Returns the realpath of the single allowed canonical output root, or ``None``
    when ``ws`` is not a recognizable frozen snapshot.
    """
    ws_real = os.path.realpath(ws)
    results_dir = os.path.dirname(ws_real)
    canonical = os.path.dirname(results_dir)
    if os.path.basename(results_dir) != 'results':
        return None
    if not os.path.basename(ws_real).startswith(FROZEN_SNAPSHOT_PREFIX):
        return None
    canonical_real = os.path.realpath(canonical)
    if os.path.basename(canonical_real) != CANONICAL_WORKSPACE_DIRNAME:
        return None
    rules_entry = os.path.realpath(os.path.join(
        canonical_real, RULES_ENTRY_RELPATH_FROM_CANONICAL))
    if not os.path.isfile(rules_entry):
        return None
    return os.path.realpath(os.path.join(canonical_real, *CANONICAL_OUTPUT_SUBPATH))


def resolve_out_root(path):
    """Resolve --out-root; refuse writes outside the workspace / into raw data dirs.

    The output root must normally be inside the running workspace. The one
    documented exception is the deployed frozen-snapshot run: the entry executes
    from ``<canonical>/results/round2_execution_<version>`` but the results must be
    written back to ``<canonical>/results/round2_fdtd_convergence`` (the same
    project's canonical 返稿修订工作区). Only that exact canonical path is accepted,
    and only when :func:`canonical_output_root_from_frozen_ws` proves the layout;
    arbitrary external directories remain refused.

    Requiring a round2/r2 path component mirrors the R2 execution layout and keeps
    this experiment from writing next to legacy result roots.
    """
    root = os.path.realpath(ws_path(path))
    ws = os.path.realpath(WS)
    if root == ws or root.startswith(ws + os.sep):
        data_root = os.path.realpath(os.path.join(ws, 'data'))
        if root == data_root or root.startswith(data_root + os.sep):
            raise Refused(f'--out-root is inside data/ (raw data is read-only): {root}')
        parts = [p for p in root.replace('\\', '/').split('/') if p]
        if not any(('round2' in p.lower() or 'r2' in p.lower()) for p in parts):
            raise Refused('--out-root has no round2/r2 path component; refusing to write '
                          f'near legacy results: {root}')
        return root
    canonical_out = canonical_output_root_from_frozen_ws(ws)
    if canonical_out is not None and root == canonical_out:
        return root
    raise Refused(
        f'--out-root is outside the workspace: {root} '
        f'(workspace={ws}; allowed canonical frozen-snapshot output root='
        f'{canonical_out if canonical_out is not None else "n/a"})')


# ---------------------------------------------------------------------------
# input config identity (configs/acoustic_cases.json)
# ---------------------------------------------------------------------------
_CONFIG_CACHE = {}


def _config_path():
    return os.path.join(WS, CONFIG_REL)


def load_acoustic_config():
    path = _config_path()
    if 'doc' not in _CONFIG_CACHE:
        if not os.path.exists(path):
            raise ProtocolError(f'required config is missing: {path}')
        _CONFIG_CACHE['doc'] = read_json(path)
        _CONFIG_CACHE['sha256'] = sha256_file(path)
    return _CONFIG_CACHE['doc'], _CONFIG_CACHE['sha256']


def validate_config_identity():
    """Verify the two fixed cases and shared physical constants against the config.

    The config is the provenance of the (f, Ma) case table used by the acoustic
    dataset; the R2 protocol must reuse the same physical constants and the same
    source side. A mismatch is an interface problem and is reported, not patched.
    """
    doc, digest = load_acoustic_config()
    cases_cfg = doc.get('cases')
    proto = doc.get('protocol')
    if not isinstance(cases_cfg, dict) or not isinstance(proto, dict):
        raise ProtocolError(f'{CONFIG_REL}: missing "cases" or "protocol" block')
    shared = {
        'c0': (float(proto.get('c0')), C0),
        'rho0': (float(proto.get('rho0')), RHO0),
        'L': (float(proto.get('L')), CORE_L),
        'alpha': (float(proto.get('alpha')), ALPHA),
        'gamma_coef': (float(proto.get('gamma_coef')), GAMMA_COEF),
        'sigma_pml': (float(proto.get('sigma_pml')), SIGMA_PML),
    }
    mismatched = []
    for name, (cfg_val, fixed_val) in shared.items():
        if not math.isclose(cfg_val, fixed_val, rel_tol=0.0, abs_tol=1e-12):
            mismatched.append(f'{name}: config={cfg_val!r} fixed={fixed_val!r}')
    if mismatched:
        raise ProtocolError(
            f'{CONFIG_REL} protocol constants disagree with the fixed R2 protocol: '
            + '; '.join(mismatched))
    out = {'path': CONFIG_REL, 'sha256': digest, 'cases': {}}
    for name, spec in CASES.items():
        cid = spec['config_case_id']
        rec = cases_cfg.get(cid)
        if not isinstance(rec, dict):
            raise ProtocolError(f'{CONFIG_REL}: required case {cid!r} is missing')
        prm = rec.get('params')
        if not isinstance(prm, dict):
            raise ProtocolError(f'{CONFIG_REL}: case {cid!r} has no params block')
        checks = {
            'f_hz': float(prm.get('f_hz')),
            'Ma': float(prm.get('Ma')),
            'source_side': prm.get('source_side'),
            'status': rec.get('status'),
        }
        if not math.isclose(checks['f_hz'], spec['f_hz'], rel_tol=0.0, abs_tol=1e-12):
            raise ProtocolError(f'{CONFIG_REL}: case {cid!r} f_hz={checks["f_hz"]} '
                                f'!= fixed {spec["f_hz"]}')
        if not math.isclose(checks['Ma'], spec['Ma'], rel_tol=0.0, abs_tol=1e-12):
            raise ProtocolError(f'{CONFIG_REL}: case {cid!r} Ma={checks["Ma"]} '
                                f'!= fixed {spec["Ma"]}')
        if checks['source_side'] != SOURCE_SIDE:
            raise ProtocolError(f'{CONFIG_REL}: case {cid!r} source_side='
                                f'{checks["source_side"]!r} != {SOURCE_SIDE!r}')
        out['cases'][cid] = checks
    return out


# ---------------------------------------------------------------------------
# fixed plan construction
# ---------------------------------------------------------------------------
def _require_case(case_name):
    if case_name not in CASES:
        raise Refused(f'unknown case {case_name!r}; choices={tuple(CASES)}')
    return CASES[case_name]


def _require_resolution(res_name):
    if res_name not in RESOLUTIONS:
        raise Refused(f'unknown resolution {res_name!r}; choices={tuple(RESOLUTIONS)}')
    return RESOLUTIONS[res_name]


def ncycle_for_case(f_hz):
    """Ncycle = ceil(f * 4 * 30 / c0) (>= 4 crossings of the 30 m air span)."""
    return int(math.ceil(float(f_hz) * N_TRANSIT * AIR_SPAN_M / C0))


def s0_for_case(f_hz):
    """Coarse steps-per-period: ceil(c0*sqrt(2) / (f * dx_coarse * 0.1))."""
    dx_coarse = 1.0 / float(S0_DX_DENOM)
    return int(math.ceil(C0 * math.sqrt(2.0) / (float(f_hz) * dx_coarse * CFL_MAX)))


def resolution_plan(case_name, res_name):
    """Return the full fixed plan for one (case, resolution) configuration.

    The plan is deterministic and derived only from the fixed protocol constants;
    it is written to protocol.json before the solve and re-used by the checks.
    """
    case = _require_case(case_name)
    res = _require_resolution(res_name)
    denom = int(res['dx_denom'])
    dx = 1.0 / float(denom)
    dy = dx
    f_hz = float(case['f_hz'])
    Ma = float(case['Ma'])

    Ncycle = ncycle_for_case(f_hz)
    s0 = s0_for_case(f_hz)
    spp = int(s0 * int(res['spp_mult']))
    dt_s = 1.0 / (f_hz * spp)                 # exact protocol dt
    n_pml = int(round(PML_THICKNESS_M / dx))
    n_buffer = int(round(AIR_HALF_M / dx))
    nx = 2 * (n_pml + n_buffer)
    ny = nx
    n_steps = int(Ncycle * spp)
    t_final_target = float(Ncycle) / f_hz
    courant = dt_s * C0 * math.sqrt(1.0 / dx ** 2 + 1.0 / dy ** 2)
    min_x = -(nx / 2.0) * dx
    source_x = min_x + n_pml * dx
    air_nodes = nx - 2 * n_pml + 1
    lambda_m = C0 / f_hz
    ppw = lambda_m / dx
    plan = {
        'task_id': TASK_ID,
        'protocol_version': PROTOCOL_VERSION,
        'case': case_name,
        'resolution': res_name,
        'series': res['series'],
        'f_hz': f_hz,
        'Ma': Ma,
        'c0_m_s': C0,
        'rho0_kg_m3': RHO0,
        'L_m': CORE_L,
        'alpha': ALPHA,
        'gamma_coef': GAMMA_COEF,
        'sigma_pml': SIGMA_PML,
        'dx_m': dx,
        'dy_m': dy,
        'dx_denom': denom,
        'dy_denom': denom,
        'n_pml_cells': n_pml,
        'n_buffer_cells': n_buffer,
        'nx_cells': nx,
        'ny_cells': ny,
        'nxp1': nx + 1,
        'nyp1': ny + 1,
        'air_nodes_x': air_nodes,
        'air_nodes_y': air_nodes,
        'air_node_range_0based': [n_pml, nx - n_pml + 1,
                                  n_pml, ny - n_pml + 1],
        'grid_min_x_m': min_x,
        'grid_max_x_m': -(min_x),
        'air_x_min_m': -AIR_HALF_M,
        'air_x_max_m': AIR_HALF_M,
        'pml_thickness_m': n_pml * dx,
        'air_half_width_m': n_buffer * dx,
        'air_span_m': 2.0 * n_buffer * dx,
        'Ncycle': Ncycle,
        's0': s0,
        'spp_mult': int(res['spp_mult']),
        'spp': spp,
        'dt_s': dt_s,
        'n_steps': n_steps,
        't_final_target_s': t_final_target,
        't_final_steps_s': n_steps * dt_s,
        'final_phase_cycles': Ncycle,
        'final_phase_residual_cycles_target': float(f_hz * t_final_target - Ncycle),
        'courant_factor': courant,
        'cfl_max': CFL_MAX,
        'source_side': SOURCE_SIDE,
        'source_y_frac': [SOURCE_Y_FRAC[0], SOURCE_Y_FRAC[1]],
        'source_x_m': source_x,
        'lambda_m': lambda_m,
        'ppw': ppw,
        'ppw_min_reference': PPW_MIN,
        'under_resolved_ppw': bool(ppw < PPW_MIN),
    }
    # protocol-internal invariants (a failure here is a code bug, not a solver issue)
    assert plan['pml_thickness_m'] == PML_THICKNESS_M, plan['pml_thickness_m']
    assert plan['air_half_width_m'] == AIR_HALF_M, plan['air_half_width_m']
    assert plan['air_span_m'] == AIR_SPAN_M, plan['air_span_m']
    assert plan['nxp1'] == nx + 1
    assert plan['air_nodes_x'] == 2 * n_buffer + 1
    assert plan['n_steps'] == Ncycle * spp
    assert plan['courant_factor'] <= CFL_MAX * (1.0 + 1e-12)
    return plan


def expected_air_coords(plan):
    """Air-region node coordinates exactly as the production solver computes them.

    Replicates ``FDTDAcoustic2D.__init__`` + ``run_case`` crop: x = min_x + i*dx
    for i in [pis-1, pie). The same arithmetic order is used so the values are
    bit-identical to the arrays returned by ``run_case``.
    """
    dx = float(plan['dx_m'])
    nx = int(plan['nx_cells'])
    n_pml = int(plan['n_pml_cells'])
    min_x = -(nx / 2.0) * dx
    idx = np.arange(n_pml, nx - n_pml + 1)
    x = min_x + idx * dx
    y = min_x + idx * dx
    return x, y


def expected_source_x(plan):
    nx = int(plan['nx_cells'])
    dx = float(plan['dx_m'])
    n_pml = int(plan['n_pml_cells'])
    return -(nx / 2.0) * dx + n_pml * dx


def make_run_kwargs(plan, save_phasor=False):
    """Production ``run_case`` keyword arguments realizing one fixed plan."""
    return {
        'f': plan['f_hz'],
        'Ma': plan['Ma'],
        'c0': plan['c0_m_s'],
        'rho0': plan['rho0_kg_m3'],
        'L': plan['L_m'],
        'alpha': plan['alpha'],
        'gamma_coef': plan['gamma_coef'],
        'dx': plan['dx_m'],
        'dy': plan['dy_m'],
        'courant_factor': plan['courant_factor'],
        'n_pml': plan['n_pml_cells'],
        'n_buffer': plan['n_buffer_cells'],
        'sigma': plan['sigma_pml'],
        'n_steps': plan['n_steps'],
        'source_side': plan['source_side'],
        'source_y_frac': tuple(plan['source_y_frac']),
        'monitor_every': max(1, int(plan['spp'])),
        'early_stop_on_nan': True,
        'crop_air': True,
        'return_velocities': False,
        'phase_snapshots': bool(save_phasor),
        'n_phase': N_PHASE,
        'verbose': False,
    }


def build_identity(case_name):
    """Case-level fixed protocol identity shared by all 5 resolutions of a case."""
    case = _require_case(case_name)
    f_hz = float(case['f_hz'])
    Ncycle = ncycle_for_case(f_hz)
    _, config_sha = load_acoustic_config()
    return {
        'task_id': TASK_ID,
        'protocol_version': PROTOCOL_VERSION,
        'case': case_name,
        'config_case_id': case['config_case_id'],
        'f_hz': f_hz,
        'Ma': float(case['Ma']),
        'c0_m_s': C0,
        'rho0_kg_m3': RHO0,
        'L_m': CORE_L,
        'alpha': ALPHA,
        'gamma_coef': GAMMA_COEF,
        'sigma_pml': SIGMA_PML,
        'air_domain_x_m': [-AIR_HALF_M, AIR_HALF_M],
        'air_domain_y_m': [-AIR_HALF_M, AIR_HALF_M],
        'pml_thickness_m': PML_THICKNESS_M,
        'source_side': SOURCE_SIDE,
        'source_x_target_m': -AIR_HALF_M,
        'Ncycle': Ncycle,
        't_final_target_s': float(Ncycle) / f_hz,
        'final_phase_cycles': Ncycle,
        'solver_path': SOLVER_REL,
        'solver_sha256': sha256_file(os.path.join(WS, SOLVER_REL)),
        'config_path': CONFIG_REL,
        'config_sha256': config_sha,
    }


def build_fixed_protocol():
    """The frozen physical protocol snapshot (shared by all 10 configurations)."""
    _, config_sha = load_acoustic_config()
    return {
        'task_id': TASK_ID,
        'protocol_version': PROTOCOL_VERSION,
        'purpose': 'reviewer-requested pressure space/time convergence of the real '
                   'Python-FDTD production solver; not a retraining or data change',
        'cases': {name: {'f_hz': spec['f_hz'], 'Ma': spec['Ma'],
                         'config_case_id': spec['config_case_id']}
                  for name, spec in CASES.items()},
        'physical': {
            'c0_m_s': C0, 'rho0_kg_m3': RHO0, 'L_m': CORE_L, 'alpha': ALPHA,
            'gamma_coef': GAMMA_COEF, 'sigma_pml': SIGMA_PML,
            'air_domain_m': [-AIR_HALF_M, AIR_HALF_M],
            'pml_thickness_m_per_side': PML_THICKNESS_M,
        },
        'grid': {
            'dx_denoms': {name: spec['dx_denom'] for name, spec in RESOLUTIONS.items()},
            'n_pml_formula': 'round(10 m / dx)',
            'n_buffer_formula': 'round(15 m / dx)',
            'total_nodes_space_series': {'coarse': 401, 'mid': 801, 'fine': 1601},
            'air_nodes_space_series': {'coarse': 241, 'mid': 481, 'fine': 961},
        },
        'time': {
            'Ncycle_formula': 'ceil(f * 4 * 30 / c0)',
            'Ncycle_by_case': {name: ncycle_for_case(spec['f_hz'])
                               for name, spec in CASES.items()},
            't_final_formula': 'Ncycle / f',
            's0_formula': 'ceil(c0*sqrt(2) / (f * (1/8) * 0.1))',
            'spp_by_resolution': {name: f's0*{spec["spp_mult"]}'
                                  for name, spec in RESOLUTIONS.items()},
            'dt_formula': '1 / (f * spp)',
            'n_steps_formula': 'Ncycle * spp',
            'final_phase': 'whole number of source cycles (sin(2*pi*f*t_final)=0)',
            'courant_factor': 'back-solved from the required dt; realized value verified',
        },
        'source': {
            'side': SOURCE_SIDE,
            'physical_x_m': -AIR_HALF_M,
            'temporal': 'hard line source sin(2*pi*f*t) injected on p and P',
            'y_fraction': [SOURCE_Y_FRAC[0], SOURCE_Y_FRAC[1]],
        },
        'comparison_contract': {
            'primary_field': PRIMARY_FIELD,
            'secondary_fields': list(SECONDARY_FIELDS),
            'spatial_domain': 'coarse air nodes (241x241), direct node extraction',
            'temporal_domain': 'mid air nodes (481x481), fixed grid',
            'interpolation': False,
            'phase_shift': False,
            'metrics': 'RE=||num-ref||_2/||ref||_2, MAE=|num-ref| mean, max=|num-ref| max',
            'observed_order': 'log2(||p_coarse-p_mid||_2 / ||p_mid-p_fine||_2)/log2(2), '
                              'i.e. the same-node ABSOLUTE L2 difference-norm ratio '
                              '(not the RE ratio: the two pairs have different reference '
                              'norms); reported only for a strictly decreasing difference '
                              'sequence with a non-zero denominator, else undefined',
            'reference': 'next finer grid is a numerical reference, not an analytic solution',
            'fine_grid_is_analytic_truth': False,
            'phasor': 'supplementary only; fitted on actual phase-locked snapshot '
                      'times and cropped to the air region with air_node_range',
            'convergence_acceptance_threshold': None,
            'single_final_time_snapshot': True,
            'steady_state_measurement': False,
            'under_resolved_note': 'coarse 500 Hz ppw=5.44 is an under-resolved '
                                   'diagnostic and is retained, not excluded',
        },
        'inputs': {
            'solver_path': SOLVER_REL,
            'solver_sha256': sha256_file(os.path.join(WS, SOLVER_REL)),
            'config_path': CONFIG_REL,
            'config_sha256': config_sha,
        },
    }


def build_protocol(case_name, res_name, save_phasor=False):
    """Protocol snapshot written to disk BEFORE the solve."""
    validate_config_identity()
    plan = resolution_plan(case_name, res_name)
    fixed = build_fixed_protocol()
    return {
        'task_id': TASK_ID,
        'protocol_version': PROTOCOL_VERSION,
        'created_utc': now_utc(),
        'status': 'written_before_solve',
        'matrix_position': {
            'case': case_name,
            'resolution': res_name,
            'series': plan['series'],
        },
        'fixed_protocol': fixed,
        'identity': build_identity(case_name),
        'plan': plan,
        'save_phasor': bool(save_phasor),
        'hashes': {
            'script_path': SCRIPT_REL,
            'script_sha256': sha256_file(os.path.join(WS, SCRIPT_REL)),
            'solver_path': SOLVER_REL,
            'solver_sha256': fixed['inputs']['solver_sha256'],
            'config_path': CONFIG_REL,
            'config_sha256': fixed['inputs']['config_sha256'],
        },
        'solve_gate': {
            'requires_slurm': True,
            'slurm_job_id': slurm_job_id(),
            'note': 'the heavy solve runs only inside SLURM; else --dry-run/--aggregate-only',
        },
        'environment': environment_snapshot(),
    }


# ---------------------------------------------------------------------------
# result consistency checks (fixed final time / phase / source / coords)
# ---------------------------------------------------------------------------
def _max_dev(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return None
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)))


def evaluate_checks(res, protocol):
    """Verify the realized solve against the frozen protocol.

    Returns a list of ``{name, ok, detail, value, expected}`` entries. The checks
    cover the realized dt / t_final / final phase / source position / CFL / grid /
    air coordinates / physical constants / finiteness. A failure is reported as an
    ``interface_mismatch``; physical parameters are never adjusted to make it pass.
    """
    plan = protocol['plan']
    ident = protocol['identity']
    checks = []

    def add(name, ok, detail='', value=None, expected=None):
        checks.append({'name': name, 'ok': bool(ok), 'detail': detail,
                       'value': _json_safe(value), 'expected': _json_safe(expected)})

    if not isinstance(res, dict):
        add('result_schema', False, f'run_case returned {type(res).__name__}')
        return checks
    missing = [k for k in ('p_total', 'p_incident', 'p_scattered', 'x', 'y',
                           'params', 'diag') if k not in res]
    if missing:
        add('result_schema', False, f'missing keys {missing}')
        return checks
    add('result_schema', True, 'run_case returned the documented keys')
    params = res['params']
    diag = res['diag']

    # --- realized time step / final time / final phase -----------------
    dt_act = float(params['dt_s'])
    dt_rel = abs(dt_act - plan['dt_s']) / plan['dt_s']
    add('dt_matches_plan', dt_rel <= DT_REL_TOL,
        f'|dt_actual-dt_plan|/dt_plan={dt_rel:.3e} (tol {DT_REL_TOL:g})',
        dt_act, plan['dt_s'])
    t_act = float(params['t_final_s'])
    t_rel = abs(t_act - ident['t_final_target_s']) / ident['t_final_target_s']
    add('t_final_matches_identity', t_rel <= T_REL_TOL,
        f'|t_final-Ncycle/f|/(Ncycle/f)={t_rel:.3e} (tol {T_REL_TOL:g})',
        t_act, ident['t_final_target_s'])
    phase_cycles = abs(plan['f_hz'] * t_act - plan['final_phase_cycles'])
    add('final_phase_cycles', phase_cycles <= PHASE_CYCLE_TOL,
        f'|f*t_final-Ncycle|={phase_cycles:.3e} cycles (tol {PHASE_CYCLE_TOL:g})',
        plan['f_hz'] * t_act, float(plan['final_phase_cycles']))
    sin_phase = abs(math.sin(2.0 * math.pi * plan['f_hz'] * t_act))
    add('final_source_phase_sin', sin_phase <= 1e-8,
        f'|sin(2*pi*f*t_final)|={sin_phase:.3e}', sin_phase, 0.0)
    cf_rel = abs(float(params['courant_factor']) - plan['courant_factor']) / plan['courant_factor']
    add('courant_factor_matches_plan', cf_rel <= DT_REL_TOL,
        f'rel diff={cf_rel:.3e}', float(params['courant_factor']),
        plan['courant_factor'])
    add('effective_cfl_within_target',
        float(params['courant_factor']) <= CFL_MAX * (1.0 + 1e-12),
        f'courant_factor={float(params["courant_factor"]):.6f} <= {CFL_MAX:g}',
        float(params['courant_factor']), CFL_MAX)

    # --- grid / air region --------------------------------------------
    add('grid_cells', int(params['nx_cells']) == plan['nx_cells']
        and int(params['ny_cells']) == plan['ny_cells'],
        f'actual nx,ny={params["nx_cells"]},{params["ny_cells"]}',
        [params['nx_cells'], params['ny_cells']],
        [plan['nx_cells'], plan['ny_cells']])
    add('grid_dx', math.isclose(float(params['dx']), plan['dx_m'], rel_tol=0.0,
                                abs_tol=1e-15)
        and math.isclose(float(params['dy']), plan['dy_m'], rel_tol=0.0, abs_tol=1e-15),
        f'actual dx,dy={params["dx"]},{params["dy"]}',
        [params['dx'], params['dy']], [plan['dx_m'], plan['dy_m']])
    add('pml_buffer_cells',
        int(params['n_pml']) == plan['n_pml_cells']
        and int(params['n_buffer']) == plan['n_buffer_cells'],
        f'actual n_pml,n_buffer={params["n_pml"]},{params["n_buffer"]}',
        [params['n_pml'], params['n_buffer']],
        [plan['n_pml_cells'], plan['n_buffer_cells']])
    add('air_node_range',
        list(params['air_node_range_0based']) == plan['air_node_range_0based'],
        f'actual={params["air_node_range_0based"]}',
        list(params['air_node_range_0based']), plan['air_node_range_0based'])
    add('n_steps_done', int(diag.get('n_steps_done', -1)) == plan['n_steps'],
        f'actual={diag.get("n_steps_done")} (early stop would show here)',
        diag.get('n_steps_done'), plan['n_steps'])

    # --- source location (left boundary at x=-15 m) --------------------
    src_x = float(params['grid_min_x_m']) + int(params['src_col_0based']) * float(params['dx'])
    src_dev = abs(src_x - expected_source_x(plan))
    add('source_side', params.get('source_side') == SOURCE_SIDE,
        f'actual={params.get("source_side")!r}', params.get('source_side'), SOURCE_SIDE)
    add('source_physical_position', src_dev <= SOURCE_POS_TOL_M,
        f'src_x={src_x:.15f} m, |dev|={src_dev:.3e} m (tol {SOURCE_POS_TOL_M:g})',
        src_x, expected_source_x(plan))

    # --- coordinates returned by run_case -----------------------------
    x_exp, y_exp = expected_air_coords(plan)
    x_dev = _max_dev(res['x'], x_exp)
    y_dev = _max_dev(res['y'], y_exp)
    add('air_coords_x_exact', x_dev is not None and x_dev <= COORD_ATOL_M,
        f'max|x-x_expected|={x_dev} m (tol {COORD_ATOL_M:g})', x_dev, 0.0)
    add('air_coords_y_exact', y_dev is not None and y_dev <= COORD_ATOL_M,
        f'max|y-y_expected|={y_dev} m (tol {COORD_ATOL_M:g})', y_dev, 0.0)

    # --- field shapes / finiteness ------------------------------------
    air_shape = (int(plan['air_nodes_x']), int(plan['air_nodes_y']))
    shapes = {k: list(np.asarray(res[k]).shape)
              for k in ('p_total', 'p_incident', 'p_scattered')}
    add('field_shapes', all(tuple(shapes[k]) == air_shape for k in shapes),
        f'shapes={shapes}, expected={air_shape}', shapes, list(air_shape))
    finite = {k: bool(np.isfinite(np.asarray(res[k], dtype=np.float64)).all())
              for k in ('p_total', 'p_incident', 'p_scattered')}
    add('fields_finite', all(finite.values()), f'finite={finite}', finite, True)
    add('diag_not_broke_nan', not bool(diag.get('broke_nan', False)),
        f'broke_nan={diag.get("broke_nan")}', diag.get('broke_nan'), False)
    add('diag_not_diverged', not bool(diag.get('diverged', False)),
        f'diverged={diag.get("diverged")}, max|p|_seen={diag.get("max_abs_p_seen")}',
        diag.get('diverged'), False)

    # --- physical constants actually used -----------------------------
    consts = {
        'c0': (float(params['c0']), C0),
        'rho0': (float(params['rho0']), RHO0),
        'L': (float(params['L']), CORE_L),
        'alpha': (float(params['alpha']), ALPHA),
        'gamma_coef': (float(params['gamma_coef']), GAMMA_COEF),
        'sigma_pml': (float(params['sigma_pml']), SIGMA_PML),
        'f_hz': (float(params['f_hz']), float(ident['f_hz'])),
        'Ma': (float(params['Ma']), float(ident['Ma'])),
    }
    bad = [k for k, (a, b) in consts.items()
           if not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12)]
    add('physical_constants_match_identity', not bad,
        f'mismatched={bad}', {k: consts[k][0] for k in consts},
        {k: consts[k][1] for k in consts})

    # --- optional phasor availability ---------------------------------
    if protocol.get('save_phasor'):
        snaps = res.get('p_phase') or []
        steps = (diag or {}).get('phase_steps') or []
        add('phasor_snapshots_available', len(snaps) >= PHASOR_MIN_SNAPSHOTS
            and len(steps) == len(snaps),
            f'n_snapshots={len(snaps)} n_steps={len(steps)}',
            len(snaps), f'>={PHASOR_MIN_SNAPSHOTS}')
    return checks


def classify_status(res, checks):
    """Map solver diagnostics + failed checks to a run status."""
    diag = res.get('diag') or {}
    arrays = [res.get(k) for k in ('p_total', 'p_incident', 'p_scattered')]
    nonfinite = any(a is None or not np.isfinite(np.asarray(a, dtype=np.float64)).all()
                    for a in arrays)
    if bool(diag.get('broke_nan', False)) or nonfinite:
        return 'nan'
    if bool(diag.get('diverged', False)):
        return 'diverged'
    plan_steps = int((res.get('params') or {}).get('n_steps', -1))
    if int(diag.get('n_steps_done', -1)) != plan_steps:
        return 'early_stop'
    failed = [c['name'] for c in checks if not c['ok']]
    if failed:
        return 'interface_mismatch'
    return 'ok'


def summarize_diag(res):
    """Solver diagnostics recorded verbatim (finite/steady/blow-up), JSON-safe."""
    diag = res.get('diag') or {}
    arrays = {}
    for key in ('p_total', 'p_incident', 'p_scattered'):
        arr = res.get(key)
        arrays[key] = bool(np.isfinite(np.asarray(arr, dtype=np.float64)).all()) \
            if arr is not None else None
    out = {
        'n_steps_done': diag.get('n_steps_done'),
        'seconds_per_step': diag.get('seconds_per_step'),
        'wall_s': diag.get('wall_s'),
        'broke_nan': diag.get('broke_nan'),
        'diverged': diag.get('diverged'),
        'divergence_threshold': diag.get('divergence_threshold'),
        'max_abs_p_seen': diag.get('max_abs_p_seen'),
        'max_abs_p_final': diag.get('max_abs_p_final'),
        'max_abs_P_final': diag.get('max_abs_P_final'),
        'rms_p_air_final': diag.get('rms_p_air_final'),
        'steady': diag.get('steady'),
        'phase_steps': diag.get('phase_steps'),
        'phasor_drift_rel_max': diag.get('phasor_drift_rel_max'),
        'finite_arrays': arrays,
        'note': 'diagnostic only (finite/steady/blow-up); NOT convergence evidence',
    }
    return out


# ---------------------------------------------------------------------------
# phasor handling: fit on actual snapshot times, then crop/align to the air grid
# ---------------------------------------------------------------------------
def align_phasor_to_air(phasor, params):
    """Crop a phasor to the physical-air region using ``params['air_node_range_0based']``.

    The task protocol warns that ``run_case`` may expose phasor / ``p_phase``
    arrays on the full grid while the cropped pressure fields use the air crop.
    Mixing the two coordinate systems would silently shift the comparison, so both
    shapes are handled explicitly and anything else is refused:
      * full-domain shape ``(nxp1, nyp1)`` -> crop with
        ``[i0:i1, j0:j1]`` from ``air_node_range_0based``;
      * already-air shape ``(i1-i0, j1-j0)`` -> accepted unchanged.

    Returns ``(cropped_array, alignment_method)``.
    """
    arr = np.asarray(phasor, dtype=np.float64)
    if arr.ndim != 2:
        raise ProtocolError(f'phasor must be 2-D, got shape {arr.shape}')
    rng = params.get('air_node_range_0based')
    if not isinstance(rng, (list, tuple)) or len(rng) != 4:
        raise ProtocolError('params.air_node_range_0based missing/invalid')
    i0, i1, j0, j1 = (int(v) for v in rng)
    air_shape = (i1 - i0, j1 - j0)
    if arr.shape == air_shape:
        return arr, 'already_air_cropped'
    full_shape = (int(params.get('nxp1', -1)), int(params.get('nyp1', -1)))
    if arr.shape == full_shape:
        return np.array(arr[i0:i1, j0:j1], copy=True), 'cropped_from_full_domain'
    raise ProtocolError(
        f'phasor shape {arr.shape} matches neither the air crop {air_shape} nor the '
        f'full grid {full_shape}; refusing to mix full-domain and cropped coordinates')


def fit_phasor_air(res, plan):
    """Re-fit the complex pressure amplitude from the actual phase-locked snapshots.

    The fit uses ``fit_complex_amplitude`` (production solver function) on the
    snapshots returned by ``run_case`` at their actual step indices and dt, so the
    phasor is not just copied from a differently-timed array. The result is then
    cropped/aligned to the air grid with ``align_phasor_to_air``.
    """
    snaps = res.get('p_phase') or []
    steps = np.asarray((res.get('diag') or {}).get('phase_steps') or [],
                       dtype=np.int64)
    params = res.get('params') or {}
    if len(snaps) < PHASOR_MIN_SNAPSHOTS:
        raise ProtocolError(f'phase snapshots unavailable ({len(snaps)})')
    if steps.size != len(snaps):
        raise ProtocolError(f'phase_steps ({steps.size}) != snapshots ({len(snaps)})')
    dt_s = float(params.get('dt_s'))
    stack = np.stack([np.asarray(s, dtype=np.float64) for s in snaps])
    phz = fit_complex_amplitude(stack, steps, dt_s, float(plan['f_hz']))
    re, how_re = align_phasor_to_air(phz['phasor_re'], params)
    im, how_im = align_phasor_to_air(phz['phasor_im'], params)
    drift, how_drift = align_phasor_to_air(phz['drift_rel'], params)
    if not (how_re == how_im == how_drift):
        raise ProtocolError('inconsistent phasor alignment between components')
    drift = np.asarray(drift, dtype=np.float64)
    finite = np.isfinite(drift)
    return {
        'phasor_re': re,
        'phasor_im': im,
        'phasor_drift_rel': drift,
        'phase_steps': steps,
        'alignment': how_re,
        'n_snapshots': int(stack.shape[0]),
        'drift_rel_max_finite': float(drift[finite].max()) if finite.any() else None,
        'n_nonfinite_drift': int((~finite).sum()),
        'fit': 'fit_complex_amplitude(snapshots, actual phase_steps, dt_s, f_hz)',
    }


# ---------------------------------------------------------------------------
# one solve
# ---------------------------------------------------------------------------
def run_dir(out_root, case_name, res_name):
    return os.path.join(out_root, case_name, res_name)


def _report_run_md(meta):
    lines = [
        f'# R2-FDTD-CONV-01 / {meta["case"]} / {meta["resolution"]}',
        '',
        f'- status: **{meta["status"]}**',
        f'- written: {meta["finished_utc"]}',
        f'- wall time: {meta["wall_s"]:.2f} s',
        '',
        '## Fixed physical protocol',
        '',
        f'- f = {meta["identity"]["f_hz"]:g} Hz, Ma = {meta["identity"]["Ma"]:g}, '
        f'source side = {meta["identity"]["source_side"]} at x = '
        f'{meta["identity"]["source_x_target_m"]:g} m',
        f'- c0 = {meta["identity"]["c0_m_s"]:g} m/s, rho0 = '
        f'{meta["identity"]["rho0_kg_m3"]:g}, L = {meta["identity"]["L_m"]:g}, '
        f'alpha = {meta["identity"]["alpha"]:g}, gamma_coef = '
        f'{meta["identity"]["gamma_coef"]:g}, sigma = {meta["identity"]["sigma_pml"]:g}',
        f'- air domain = {meta["identity"]["air_domain_x_m"]} m, PML thickness = '
        f'{meta["identity"]["pml_thickness_m"]:g} m/side',
        '',
        '## Numerical plan (fixed, written before the solve)',
        '',
        f'- dx = dy = {meta["plan"]["dx_m"]:.6g} m -> {meta["plan"]["nxp1"]} nodes total, '
        f'{meta["plan"]["air_nodes_x"]} air nodes',
        f'- n_pml = {meta["plan"]["n_pml_cells"]} cells, n_buffer = '
        f'{meta["plan"]["n_buffer_cells"]} cells',
        f'- spp = {meta["plan"]["spp"]} steps/period, n_steps = {meta["plan"]["n_steps"]}',
        f'- dt = {meta["plan"]["dt_s"]:.12e} s, t_final = '
        f'{meta["plan"]["t_final_target_s"]:.12f} s = Ncycle/f '
        f'(Ncycle = {meta["plan"]["Ncycle"]})',
        f'- courant_factor (back-solved) = {meta["plan"]["courant_factor"]:.9f} '
        f'(target CFL <= {meta["plan"]["cfl_max"]:g})',
        f'- points per wavelength = {meta["plan"]["ppw"]:.2f}'
        + ('  **[under-resolved diagnostic: retained, not excluded]**'
           if meta["plan"]["under_resolved_ppw"] else ''),
        '',
        '## Realized solver parameters and interface checks',
        '',
        '| check | ok | actual | expected |',
        '|---|---|---|---|',
    ]
    for chk in meta['checks']:
        lines.append(f'| {chk["name"]} | {chk["ok"]} | '
                     f'{json.dumps(chk["value"], ensure_ascii=False)} | '
                     f'{json.dumps(chk["expected"], ensure_ascii=False)} |')
    diag = meta.get('diagnostics')
    lines += [
        '',
        '## Solver diagnostics (recorded, not convergence evidence)',
        '',
    ]
    if diag is None:
        lines.append('- diagnostics unavailable (the solve raised before returning)')
    else:
        lines += [
            f'- n_steps_done = {diag.get("n_steps_done")}',
            f'- broke_nan = {diag.get("broke_nan")}, diverged = '
            f'{diag.get("diverged")}',
            f'- max |p| seen = {diag.get("max_abs_p_seen")}, '
            f'rms p_air final = {diag.get("rms_p_air_final")}',
            f'- steady = {json.dumps(_json_safe(diag.get("steady")), ensure_ascii=False)}',
        ]
    lines += [
        '',
        'These finite / steady / blow-up diagnostics are model-side evidence only. '
        'The convergence comparison of this experiment uses the real ``p_total`` '
        'final-time snapshots, not the steady flag.',
        '',
        '## Outputs',
        '',
        f'- protocol (written pre-solve): {meta["protocol_path"]} '
        f'(sha256 {meta["protocol_sha256"][:16]}...)',
    ]
    if meta['outputs'].get('fields_npz'):
        lines.append(f'- fields: {meta["outputs"]["fields_npz"]} '
                     f'(sha256 {meta["outputs"]["sha256"][:16]}...)')
    if meta['outputs'].get('diagnostics_npz'):
        lines.append(f'- diagnostics only: {meta["outputs"]["diagnostics_npz"]} '
                     f'(sha256 {meta["outputs"]["sha256"][:16]}...)')
    if meta.get('error'):
        lines += ['', f'- error: `{meta["error"]}`']
    lines += ['', f'- environment: {json.dumps(_json_safe(meta["environment"]), ensure_ascii=False)}', '']
    return '\n'.join(lines)


def run_one(out_root, case_name, res_name, save_phasor=False):
    """Run one fixed configuration and persist protocol/fields/meta/report."""
    gate = check_solve_environment()           # refuses outside SLURM (heavy solve)
    protocol = build_protocol(case_name, res_name, save_phasor)
    plan = protocol['plan']
    d = run_dir(out_root, case_name, res_name)
    existing = [name for name in KEY_FILES if os.path.exists(os.path.join(d, name))]
    if existing:
        raise Refused(f'{rel_ws(d)} already contains {existing}; refusing to '
                      'overwrite (failure products are preserved -- use a new --out-root)')
    os.makedirs(d, exist_ok=True)
    protocol_path = os.path.join(d, 'protocol.json')
    write_json(protocol_path, protocol)
    protocol_sha = sha256_file(protocol_path)
    meta_path = os.path.join(d, 'meta.json')
    report_path = os.path.join(d, 'report.md')

    started = now_utc()
    t0 = time.perf_counter()
    res = None
    error = None
    try:
        res = run_case(**make_run_kwargs(plan, save_phasor))
    except Exception as exc:                    # noqa: BLE001 - evidence must be kept
        error = repr(exc)
    wall_s = time.perf_counter() - t0

    if error is not None:
        meta = {
            'task_id': TASK_ID,
            'protocol_version': PROTOCOL_VERSION,
            'case': case_name,
            'resolution': res_name,
            'status': 'error',
            'started_utc': started,
            'finished_utc': now_utc(),
            'wall_s': wall_s,
            'protocol_path': rel_ws(protocol_path),
            'protocol_sha256': protocol_sha,
            'plan': plan,
            'identity': protocol['identity'],
            'environment': protocol['environment'],
            'slurm': gate,
            'error': error,
            'checks': [],
            'checks_all_ok': False,
            'diagnostics': None,
            'outputs': {},
            'phasor': {'saved': False, 'reason': 'solve raised'},
        }
        write_json(meta_path, meta)
        write_text(report_path, _report_run_md(meta))
        raise RunFailed(f'{case_name}/{res_name}: solve raised {error}; '
                        f'evidence kept in {rel_ws(d)}')

    diag = res.get('diag') or {}
    try:
        checks = evaluate_checks(res, protocol)
    except Exception as exc:                    # noqa: BLE001 - evidence must be kept
        checks = [{'name': 'check_evaluation', 'ok': False,
                   'detail': f'check evaluation raised {exc!r}',
                   'value': None, 'expected': None}]
    status = classify_status(res, checks)

    phasor = None
    if status == 'ok' and save_phasor:
        try:
            phasor = fit_phasor_air(res, plan)
        except Exception as exc:                # noqa: BLE001 - evidence must be kept
            checks.append({'name': 'phasor_fit', 'ok': False,
                           'detail': f'phasor re-fit failed: {exc!r}',
                           'value': None, 'expected': 'fit from actual snapshots'})
            status = 'interface_mismatch'

    diag_payload = summarize_diag(res)
    outputs = {}
    if status == 'ok':
        fields_path = os.path.join(d, 'fields.npz')
        arrays = {
            'p_total': np.asarray(res['p_total'], dtype=np.float64),
            'p_incident': np.asarray(res['p_incident'], dtype=np.float64),
            'p_scattered': np.asarray(res['p_scattered'], dtype=np.float64),
            'x': np.asarray(res['x'], dtype=np.float64),
            'y': np.asarray(res['y'], dtype=np.float64),
            'energy_tail': np.asarray(diag.get('energy_tail') or [], dtype=np.float64),
            'energy_samples': np.asarray(diag.get('energy_samples') or [], dtype=np.float64),
            'maxp_samples': np.asarray(diag.get('maxp_samples') or [], dtype=np.float64),
        }
        if phasor is not None:
            arrays.update({
                'p_phasor_re': np.asarray(phasor['phasor_re'], dtype=np.float64),
                'p_phasor_im': np.asarray(phasor['phasor_im'], dtype=np.float64),
                'p_phasor_drift_rel': np.asarray(phasor['phasor_drift_rel'],
                                                  dtype=np.float64),
                'phase_steps': np.asarray(phasor['phase_steps'], dtype=np.int64),
            })
        write_npz(fields_path, arrays)
        outputs = {
            'fields_npz': rel_ws(fields_path),
            'sha256': sha256_file(fields_path),
            'keys': sorted(arrays),
            'shapes': {k: list(np.asarray(v).shape) for k, v in arrays.items()},
        }
    else:
        diagnostics_path = os.path.join(d, 'diagnostics.npz')
        arrays = {
            'p_total': np.asarray(res.get('p_total'), dtype=np.float64),
            'p_incident': np.asarray(res.get('p_incident'), dtype=np.float64),
            'p_scattered': np.asarray(res.get('p_scattered'), dtype=np.float64),
            'x': np.asarray(res.get('x'), dtype=np.float64),
            'y': np.asarray(res.get('y'), dtype=np.float64),
            'energy_tail': np.asarray(diag.get('energy_tail') or [], dtype=np.float64),
            'energy_samples': np.asarray(diag.get('energy_samples') or [], dtype=np.float64),
            'maxp_samples': np.asarray(diag.get('maxp_samples') or [], dtype=np.float64),
        }
        write_npz(diagnostics_path, arrays)
        outputs = {
            'diagnostics_npz': rel_ws(diagnostics_path),
            'sha256': sha256_file(diagnostics_path),
            'keys': sorted(arrays),
            'shapes': {k: list(np.asarray(v).shape) for k, v in arrays.items()},
        }

    meta = {
        'task_id': TASK_ID,
        'protocol_version': PROTOCOL_VERSION,
        'case': case_name,
        'resolution': res_name,
        'series': plan['series'],
        'status': status,
        'started_utc': started,
        'finished_utc': now_utc(),
        'wall_s': wall_s,
        'slurm': gate,
        'environment': protocol['environment'],
        'protocol_path': rel_ws(protocol_path),
        'protocol_sha256': protocol_sha,
        'hashes': protocol['hashes'],
        'identity': protocol['identity'],
        'plan': plan,
        'solver_params': res.get('params'),
        'save_phasor': bool(save_phasor),
        'phasor': ({'saved': phasor is not None, 'alignment': phasor['alignment'],
                    'n_snapshots': phasor['n_snapshots'],
                    'fit': phasor['fit'],
                    'drift_rel_max_finite': phasor['drift_rel_max_finite'],
                    'n_nonfinite_drift': phasor['n_nonfinite_drift']}
                   if phasor is not None else
                   {'saved': False,
                    'reason': ('run without --save-phasor' if not save_phasor
                               else 'not computed')}),
        'checks': checks,
        'checks_all_ok': all(c['ok'] for c in checks),
        'n_checks': len(checks),
        'n_failed_checks': sum(1 for c in checks if not c['ok']),
        'diagnostics': diag_payload,
        'outputs': outputs,
        'interpretation': 'single fixed configuration; convergence requires the full '
                          '10-solve aggregate',
    }
    write_json(meta_path, meta)
    write_text(report_path, _report_run_md(meta))

    print(f'[solve] {case_name}/{res_name}: status={status} '
          f'({wall_s:.1f} s, {diag.get("seconds_per_step")} s/step, '
          f'n_steps={diag.get("n_steps_done")}/{plan["n_steps"]}, '
          f't_final={float(res["params"]["t_final_s"]):.9f} s, '
          f'dt={float(res["params"]["dt_s"]):.6e} s)', flush=True)
    if status != 'ok':
        failed = [c['name'] for c in checks if not c['ok']]
        raise RunFailed(f'{case_name}/{res_name}: status={status}, failed checks={failed}; '
                        f'evidence kept in {rel_ws(d)}')
    return meta


# ---------------------------------------------------------------------------
# aggregation: node alignment, metrics, observed order
# ---------------------------------------------------------------------------
def _axis_indices(axis, target, label):
    """Map target coordinates onto axis node indices WITHOUT interpolation.

    The target coordinates must be an exact, regularly-strided subset of the axis
    nodes; otherwise a ``ProtocolError`` is raised (wrong grid / misaligned run).
    """
    axis = np.asarray(axis, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if axis.ndim != 1 or target.ndim != 1:
        raise ProtocolError(f'{label}: 1-D coordinates required')
    if axis.size < 2 or target.size < 2:
        raise ProtocolError(f'{label}: need >=2 coordinates')
    diffs = np.diff(axis)
    if not np.all(diffs > 0):
        raise ProtocolError(f'{label}: axis coordinates are not strictly increasing')
    spacing = float(diffs[0])
    if np.max(np.abs(diffs - spacing)) > spacing * 1e-9:
        raise ProtocolError(f'{label}: axis coordinates are not uniformly spaced')
    tdiffs = np.diff(target)
    if not np.all(tdiffs > 0):
        raise ProtocolError(f'{label}: target coordinates are not strictly increasing')
    ratio = float(tdiffs[0]) / spacing
    k = int(round(ratio))
    if k < 1 or abs(ratio - k) > ALIGN_SPACING_RTOL * max(1, k):
        raise ProtocolError(
            f'{label}: target spacing {float(tdiffs[0]):.12g} is not an integer multiple '
            f'of the axis spacing {spacing:.12g} (no interpolation allowed)')
    if np.max(np.abs(tdiffs - k * spacing)) > spacing * 1e-9:
        raise ProtocolError(f'{label}: target coordinates are not regularly spaced')
    idx = np.rint((target - axis[0]) / spacing).astype(np.int64)
    if idx.min() < 0 or idx.max() >= axis.size:
        raise ProtocolError(f'{label}: target coordinates outside the source axis')
    dev = np.abs(axis[idx] - target)
    if float(np.max(dev)) > COORD_ATOL_M:
        raise ProtocolError(
            f'{label}: target nodes do not coincide with source nodes '
            f'(max deviation {float(np.max(dev)):.3e} m > {COORD_ATOL_M:g} m); '
            'refusing to interpolate')
    return idx, k


def align_field_to_nodes(field, x, y, target_x, target_y, label=''):
    """Extract ``field`` on ``target_x``/``target_y`` by exact node indexing.

    No interpolation and no coordinate shift: the target coordinates must coincide
    with source nodes. Returns ``(cropped_field, ix, iy)``.
    """
    arr = np.asarray(field, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if arr.ndim != 2:
        raise ProtocolError(f'{label}: field must be 2-D, got {arr.shape}')
    if arr.shape != (x.size, y.size):
        raise ProtocolError(f'{label}: field shape {arr.shape} != grid '
                            f'({x.size}, {y.size})')
    ix, _ = _axis_indices(x, target_x, f'{label}-x')
    iy, _ = _axis_indices(y, target_y, f'{label}-y')
    return np.array(arr[np.ix_(ix, iy)], copy=True), ix, iy


def pair_metrics(num, ref):
    """RE / MAE / max error between ``num`` and ``ref`` on identical nodes.

    RE = ||num-ref||_2 / ||ref||_2. A zero reference norm makes RE undefined
    (reported as ``None`` with a reason), not infinity or 0.
    """
    num = np.asarray(num, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if num.shape != ref.shape:
        raise ProtocolError(f'pair shape mismatch {num.shape} vs {ref.shape}')
    diff = num - ref
    norm_num = float(np.sqrt(np.sum(diff * diff)))
    norm_ref = float(np.sqrt(np.sum(ref * ref)))
    out = {
        'n_points': int(num.size),
        'norm_diff': norm_num,
        'norm_ref': norm_ref,
        'mae': float(np.mean(np.abs(diff))) if num.size else None,
        'max': float(np.max(np.abs(diff))) if num.size else None,
        're': None,
        're_status': 'undefined',
        're_reason': None,
    }
    if not (math.isfinite(norm_num) and math.isfinite(norm_ref)):
        out['re_reason'] = 'non-finite norm'
    elif norm_ref <= 0.0:
        out['re_reason'] = 'zero denominator: ||ref||_2 == 0'
    else:
        out['re'] = norm_num / norm_ref
        out['re_status'] = 'defined'
    return out


def complex_pair_metrics(num_re, num_im, ref_re, ref_im):
    """Complex-valued analogue of :func:`pair_metrics` (phasor supplementary)."""
    num = np.asarray(num_re, dtype=np.float64) + 1j * np.asarray(num_im, dtype=np.float64)
    ref = np.asarray(ref_re, dtype=np.float64) + 1j * np.asarray(ref_im, dtype=np.float64)
    if num.shape != ref.shape:
        raise ProtocolError(f'complex pair shape mismatch {num.shape} vs {ref.shape}')
    diff = num - ref
    norm_num = float(np.sqrt(np.sum(np.abs(diff) ** 2)))
    norm_ref = float(np.sqrt(np.sum(np.abs(ref) ** 2)))
    out = {
        'n_points': int(num.size),
        'norm_diff': norm_num,
        'norm_ref': norm_ref,
        'mae': float(np.mean(np.abs(diff))) if num.size else None,
        'max': float(np.max(np.abs(diff))) if num.size else None,
        're': None,
        're_status': 'undefined',
        're_reason': None,
    }
    if not (math.isfinite(norm_num) and math.isfinite(norm_ref)):
        out['re_reason'] = 'non-finite norm'
    elif norm_ref <= 0.0:
        out['re_reason'] = 'zero denominator: ||ref||_2 == 0'
    else:
        out['re'] = norm_num / norm_ref
        out['re_status'] = 'defined'
    return out


def observed_order(d_coarser, d_finer, refinement_ratio=REFINEMENT_RATIO):
    """Observed order from the same-node absolute L2 difference norms.

    The two consecutive pairs are ``||p_coarse - p_mid||_2`` (``d_coarser``) and
    ``||p_mid - p_fine||_2`` (``d_finer``), evaluated on the same nodes. These are
    absolute difference norms in pressure units, as required by the protocol: the
    RE normalizations of the two pairs use *different* reference norms
    (``||p_mid||`` vs ``||p_fine||``), so ``log2(RE_cm/RE_mf)`` is NOT the absolute
    L2 difference ratio. With ``d ~ C h^p`` and refinement ratio ``r``:
    ``p = log(d_coarser/d_finer)/log(r)``.

    Reported only when the difference-norm sequence is strictly decreasing and the
    denominator is non-zero; otherwise explicitly ``undefined`` with the reason.
    RE remains the primary per-pair score and is not changed by this function.
    """
    out = {
        'l2_diff_coarser': d_coarser,
        'l2_diff_finer': d_finer,
        'refinement_ratio': float(refinement_ratio),
        'log2_diff_ratio': None,
        'observed_order': None,
        'status': 'undefined',
        'reason': None,
    }
    if d_coarser is None or d_finer is None:
        out['reason'] = 'missing absolute L2 difference norm'
        return out
    try:
        d1 = float(d_coarser)
        d2 = float(d_finer)
    except (TypeError, ValueError):
        out['reason'] = 'non-numeric absolute L2 difference norm'
        return out
    if not (math.isfinite(d1) and math.isfinite(d2)):
        out['reason'] = 'non-finite absolute L2 difference norm'
        return out
    if d2 == 0.0:
        out['reason'] = 'zero denominator: ||p_mid-p_fine||_2 == 0'
        return out
    if d1 <= d2:
        out['reason'] = (f'non-monotonic L2 difference sequence: '
                         f'||p_coarse-p_mid||_2={d1:.6e} <= '
                         f'||p_mid-p_fine||_2={d2:.6e}; observed order not defined')
        return out
    ratio = d1 / d2
    out['log2_diff_ratio'] = math.log2(ratio)
    out['observed_order'] = math.log(ratio) / math.log(float(refinement_ratio))
    out['status'] = 'defined'
    return out


# ---------------------------------------------------------------------------
# aggregation scan / load
# ---------------------------------------------------------------------------
def expected_runs():
    """The fixed 2 cases x 5 resolutions = 10 solve matrix."""
    return [(case, res) for case in CASES for res in RESOLUTIONS]


def scan_runs(out_root):
    """Classify each expected run as complete / pending / invalid without writing."""
    complete, pending, invalid = [], [], []
    for case, res in expected_runs():
        d = run_dir(out_root, case, res)
        names = {name: os.path.join(d, name) for name in KEY_FILES}
        if not os.path.isdir(d):
            pending.append({'case': case, 'resolution': res,
                            'reason': 'run directory missing'})
            continue
        if not os.path.exists(names['meta.json']):
            pending.append({'case': case, 'resolution': res,
                            'reason': 'meta.json missing (run not finished)'})
            continue
        try:
            meta = read_json(names['meta.json'])
        except Exception as exc:                # noqa: BLE001
            invalid.append({'case': case, 'resolution': res, 'status': 'unreadable',
                            'reason': f'meta.json unreadable: {exc!r}'})
            continue
        status = meta.get('status')
        if status != 'ok':
            invalid.append({'case': case, 'resolution': res, 'status': status,
                            'reason': meta.get('error') or f'run status={status}'})
            continue
        if not meta.get('checks_all_ok'):
            invalid.append({'case': case, 'resolution': res, 'status': status,
                            'reason': 'interface checks failed'})
            continue
        for need in ('protocol.json', 'fields.npz', 'report.md'):
            if not os.path.exists(names[need]):
                invalid.append({'case': case, 'resolution': res, 'status': status,
                                'reason': f'{need} missing'})
                break
        else:
            out = meta.get('outputs') or {}
            sha = out.get('sha256')
            if not sha or sha256_file(names['fields.npz']) != sha:
                invalid.append({'case': case, 'resolution': res, 'status': status,
                                'reason': 'fields.npz sha256 mismatch'})
                continue
            complete.append({'case': case, 'resolution': res,
                             'dir': rel_ws(d), 'meta': meta})
    return {'complete': complete, 'pending': pending, 'invalid': invalid,
            'n_expected': len(expected_runs()),
            'n_complete': len(complete)}


def _load_npz(path):
    with np.load(path, allow_pickle=False) as fh:
        return {key: np.array(fh[key], copy=True) for key in fh.files}, set(fh.files)


def load_run(out_root, case_name, res_name):
    """Load one finished run (protocol + meta + fields) for aggregation."""
    d = run_dir(out_root, case_name, res_name)
    protocol = read_json(os.path.join(d, 'protocol.json'))
    meta = read_json(os.path.join(d, 'meta.json'))
    arrays, keys = _load_npz(os.path.join(d, 'fields.npz'))
    if meta.get('status') != 'ok':
        raise ProtocolError(f'{case_name}/{res_name}: run status={meta.get("status")}')
    if sha256_file(os.path.join(d, 'protocol.json')) != meta.get('protocol_sha256'):
        raise ProtocolError(f'{case_name}/{res_name}: protocol.json changed after the solve')
    if 'p_total' not in arrays or 'x' not in arrays or 'y' not in arrays:
        raise ProtocolError(f'{case_name}/{res_name}: fields.npz is missing required arrays')
    expected_sha = (meta.get('outputs') or {}).get('sha256')
    if expected_sha and sha256_file(os.path.join(d, 'fields.npz')) != expected_sha:
        raise ProtocolError(f'{case_name}/{res_name}: fields.npz sha256 mismatch')
    return {
        'case': case_name,
        'resolution': res_name,
        'dir': d,
        'protocol': protocol,
        'meta': meta,
        'arrays': arrays,
        'npz_keys': keys,
        'has_phasor': {'p_phasor_re', 'p_phasor_im',
                       'p_phasor_drift_rel'}.issubset(keys),
    }


def validate_identity(protocols):
    """Require the exact same case-level protocol identity across a run set."""
    if not protocols:
        raise ProtocolError('no protocols supplied')
    ref = protocols[0].get('identity')
    if not isinstance(ref, dict) or not ref:
        raise ProtocolError('protocol has no identity block')
    for i, proto in enumerate(protocols[1:], start=1):
        other = proto.get('identity')
        if not isinstance(other, dict):
            raise IdentityMismatch(f'protocol #{i} has no identity block')
        diffs = [key for key in sorted(set(ref) | set(other))
                 if ref.get(key) != other.get(key)]
        if diffs:
            detail = '; '.join(f'{key}: {ref.get(key)!r} != {other.get(key)!r}'
                               for key in diffs)
            raise IdentityMismatch(f'protocol identity mismatch: {detail}')
    canonical = json.dumps(ref, sort_keys=True, separators=(',', ':'))
    return {
        'n_protocols': len(protocols),
        'identity_sha256': hashlib.sha256(canonical.encode('utf-8')).hexdigest(),
        'identity': ref,
        'keys': sorted(ref),
    }


def _current_hashes():
    return {
        'script_sha256': sha256_file(os.path.join(WS, SCRIPT_REL)),
        'solver_sha256': sha256_file(os.path.join(WS, SOLVER_REL)),
        'config_sha256': sha256_file(os.path.join(WS, CONFIG_REL)),
    }


def _verify_protocol_integrity(runs):
    """All runs must share the frozen code/config hashes and match the current files."""
    current = _current_hashes()
    ref = None
    for run in runs:
        hashes = run['protocol'].get('hashes') or {}
        subset = {key: hashes.get(key) for key in current}
        if None in subset.values():
            raise ProtocolError(f'{run["case"]}/{run["resolution"]}: protocol hashes missing')
        if subset != current:
            raise ProtocolError(
                f'{run["case"]}/{run["resolution"]}: code/config changed since the solve '
                f'(protocol={subset}, current={current}); refusing to aggregate mixed '
                'protocol versions')
        if ref is None:
            ref = subset
        elif subset != ref:
            raise ProtocolError('run set mixes different code/config hashes')
    return {'hashes': current, 'n_runs_checked': len(runs)}


def _verify_run_meta(run, expected_identity):
    """Re-check per-run meta against the protocol plan and the case identity."""
    meta = run['meta']
    proto = run['protocol']
    plan = proto['plan']
    if meta.get('plan') != plan:
        raise ProtocolError(f'{run["case"]}/{run["resolution"]}: meta plan != protocol plan')
    if meta.get('identity') != expected_identity:
        raise ProtocolError(f'{run["case"]}/{run["resolution"]}: identity != case identity')
    params = meta.get('solver_params') or {}
    target_t = float(expected_identity['t_final_target_s'])
    t_act = float(params.get('t_final_s'))
    if abs(t_act - target_t) > T_REL_TOL * target_t:
        raise ProtocolError(f'{run["case"]}/{run["resolution"]}: t_final={t_act} '
                            f'deviates from Ncycle/f={target_t}')
    dt_act = float(params.get('dt_s'))
    if abs(dt_act - float(plan['dt_s'])) > DT_REL_TOL * float(plan['dt_s']):
        raise ProtocolError(f'{run["case"]}/{run["resolution"]}: dt deviates from plan')
    checks = meta.get('checks') or []
    failed = [c['name'] for c in checks if not c.get('ok')]
    if failed:
        raise ProtocolError(f'{run["case"]}/{run["resolution"]}: failed checks {failed}')
    return {'case': run['case'], 'resolution': run['resolution'],
            't_final_s': t_act, 'dt_s': dt_act,
            'n_steps': int(params.get('n_steps', -1)),
            'checks_ok': True}


def _align_bundle(run, target_x, target_y, field_names, phasor=False):
    """Align all requested real fields (and optional phasor) to target nodes."""
    out = {}
    ix = iy = None
    for field in field_names:
        arr, ix, iy = align_field_to_nodes(
            run['arrays'][field], run['arrays']['x'], run['arrays']['y'],
            target_x, target_y, label=f'{run["case"]}/{run["resolution"]}->target {field}')
        out[field] = arr
    if phasor:
        out['p_phasor_re'], ix, iy = align_field_to_nodes(
            run['arrays']['p_phasor_re'], run['arrays']['x'], run['arrays']['y'],
            target_x, target_y, label=f'{run["case"]}/{run["resolution"]}->target phasor_re')
        out['p_phasor_im'], _, _ = align_field_to_nodes(
            run['arrays']['p_phasor_im'], run['arrays']['x'], run['arrays']['y'],
            target_x, target_y, label=f'{run["case"]}/{run["resolution"]}->target phasor_im')
    strides = {'ix_stride': int(ix[1] - ix[0]) if ix.size > 1 else None,
               'iy_stride': int(iy[1] - iy[0]) if iy.size > 1 else None}
    return out, strides


def compare_space(runs_by_res, phasor=False):
    """Spatial convergence at the common coarse air nodes (direct extraction)."""
    coarse = runs_by_res['coarse']
    mid = runs_by_res['mid']
    fine = runs_by_res['fine']
    xc = coarse['arrays']['x']
    yc = coarse['arrays']['y']
    out = {
        'series': list(SPATIAL_SERIES),
        'target_nodes': {
            'description': 'coarse air nodes (direct node extraction, no interpolation)',
            'n_x': int(xc.size), 'n_y': int(yc.size),
            'x_min_m': float(np.min(xc)), 'x_max_m': float(np.max(xc)),
        },
        'fields': {},
        'alignment': {},
    }
    mid_aligned, mid_strides = _align_bundle(mid, xc, yc, FIELDS, phasor=phasor)
    fine_aligned, fine_strides = _align_bundle(fine, xc, yc, FIELDS, phasor=phasor)
    out['alignment'] = {
        'mid_to_coarse': dict(mid_strides, coords_equal_max_dev_m=0.0),
        'fine_to_coarse': dict(fine_strides, coords_equal_max_dev_m=0.0),
    }
    for field in FIELDS:
        pc = coarse['arrays'][field]
        pm = mid_aligned[field]
        pf = fine_aligned[field]
        pairs = {
            'coarse_vs_mid': {'num': 'coarse', 'ref': 'mid on coarse nodes',
                              'metrics': pair_metrics(pc, pm)},
            'mid_vs_fine': {'num': 'mid on coarse nodes',
                            'ref': 'fine on coarse nodes',
                            'metrics': pair_metrics(pm, pf)},
            'coarse_vs_fine': {'num': 'coarse', 'ref': 'fine on coarse nodes',
                               'metrics': pair_metrics(pc, pf)},
        }
        # Total plan: observed order uses the same-node ABSOLUTE L2 difference
        # norms ||p_coarse-p_mid||_2 and ||p_mid-p_fine||_2, because the two pairs
        # have different reference norms and log2(RE_cm/RE_mf) would not be the
        # absolute-difference ratio. RE remains the primary per-pair score.
        order = observed_order(pairs['coarse_vs_mid']['metrics']['norm_diff'],
                               pairs['mid_vs_fine']['metrics']['norm_diff'],
                               REFINEMENT_RATIO)
        out['fields'][field] = {
            'pairs': pairs,
            'observed_order': order,
            'coarse_to_fine_error': pairs['coarse_vs_fine']['metrics']['re'],
        }
    if phasor:
        c_re = coarse['arrays']['p_phasor_re']
        c_im = coarse['arrays']['p_phasor_im']
        m_re, m_im = mid_aligned['p_phasor_re'], mid_aligned['p_phasor_im']
        f_re, f_im = fine_aligned['p_phasor_re'], fine_aligned['p_phasor_im']
        pairs = {
            'coarse_vs_mid': {'metrics': complex_pair_metrics(c_re, c_im, m_re, m_im)},
            'mid_vs_fine': {'metrics': complex_pair_metrics(m_re, m_im, f_re, f_im)},
            'coarse_vs_fine': {'metrics': complex_pair_metrics(c_re, c_im, f_re, f_im)},
        }
        order = observed_order(pairs['coarse_vs_mid']['metrics']['norm_diff'],
                               pairs['mid_vs_fine']['metrics']['norm_diff'],
                               REFINEMENT_RATIO)
        out['phasor_supplementary'] = {'pairs': pairs, 'observed_order': order,
                                       'note': 'supplementary: complex phase-locked '
                                               'amplitude, not the primary snapshot metric'}
    return out


def compare_time(runs_by_res, phasor=False):
    """Temporal convergence on the fixed mid grid (same nodes, same final phase)."""
    mid = runs_by_res['mid']
    dt2 = runs_by_res['mid_dt2']
    dt4 = runs_by_res['mid_dt4']
    xm = mid['arrays']['x']
    ym = mid['arrays']['y']
    coords = {}
    for name, run in (('mid_dt2', dt2), ('mid_dt4', dt4)):
        dev_x = _max_dev(run['arrays']['x'], xm)
        dev_y = _max_dev(run['arrays']['y'], ym)
        if dev_x is None or dev_y is None or dev_x > COORD_ATOL_M or dev_y > COORD_ATOL_M:
            raise ProtocolError(f'{name} coordinates do not match the mid grid '
                                f'(dx_dev={dev_x}, dy_dev={dev_y}); no interpolation allowed')
        coords[name] = {'x_max_dev_m': dev_x, 'y_max_dev_m': dev_y}
    out = {
        'series': list(TEMPORAL_SERIES),
        'target_nodes': {
            'description': 'fixed mid air nodes (no interpolation, no phase shift)',
            'n_x': int(xm.size), 'n_y': int(ym.size),
            'x_min_m': float(np.min(xm)), 'x_max_m': float(np.max(xm)),
        },
        'coords_equal': coords,
        'fields': {},
    }
    for field in FIELDS:
        p_mid = mid['arrays'][field]
        p_dt2 = dt2['arrays'][field]
        p_dt4 = dt4['arrays'][field]
        pairs = {
            'mid_vs_mid_dt2': {'num': 'mid (dt)', 'ref': 'mid_dt2 (dt/2)',
                               'metrics': pair_metrics(p_mid, p_dt2)},
            'mid_dt2_vs_mid_dt4': {'num': 'mid_dt2 (dt/2)', 'ref': 'mid_dt4 (dt/4)',
                                   'metrics': pair_metrics(p_dt2, p_dt4)},
            'mid_vs_mid_dt4': {'num': 'mid (dt)', 'ref': 'mid_dt4 (dt/4)',
                               'metrics': pair_metrics(p_mid, p_dt4)},
        }
        order = observed_order(pairs['mid_vs_mid_dt2']['metrics']['norm_diff'],
                               pairs['mid_dt2_vs_mid_dt4']['metrics']['norm_diff'],
                               REFINEMENT_RATIO)
        out['fields'][field] = {'pairs': pairs, 'observed_order': order}
    if phasor:
        pairs = {
            'mid_vs_mid_dt2': {
                'metrics': complex_pair_metrics(
                    mid['arrays']['p_phasor_re'], mid['arrays']['p_phasor_im'],
                    dt2['arrays']['p_phasor_re'], dt2['arrays']['p_phasor_im'])},
            'mid_dt2_vs_mid_dt4': {
                'metrics': complex_pair_metrics(
                    dt2['arrays']['p_phasor_re'], dt2['arrays']['p_phasor_im'],
                    dt4['arrays']['p_phasor_re'], dt4['arrays']['p_phasor_im'])},
            'mid_vs_mid_dt4': {
                'metrics': complex_pair_metrics(
                    mid['arrays']['p_phasor_re'], mid['arrays']['p_phasor_im'],
                    dt4['arrays']['p_phasor_re'], dt4['arrays']['p_phasor_im'])},
        }
        order = observed_order(pairs['mid_vs_mid_dt2']['metrics']['norm_diff'],
                               pairs['mid_dt2_vs_mid_dt4']['metrics']['norm_diff'],
                               REFINEMENT_RATIO)
        out['phasor_supplementary'] = {'pairs': pairs, 'observed_order': order,
                                       'note': 'supplementary: complex phase-locked '
                                               'amplitude, not the primary snapshot metric'}
    return out


def _diagnostics_table(runs_by_res):
    table = {}
    for name, run in runs_by_res.items():
        diag = run['meta'].get('diagnostics') or {}
        table[name] = {
            'status': run['meta'].get('status'),
            'n_steps_done': diag.get('n_steps_done'),
            'broke_nan': diag.get('broke_nan'),
            'diverged': diag.get('diverged'),
            'max_abs_p_seen': diag.get('max_abs_p_seen'),
            'rms_p_air_final': diag.get('rms_p_air_final'),
            'steady_ok': (diag.get('steady') or {}).get('steady_ok'),
            'rel_growth': (diag.get('steady') or {}).get('rel_growth'),
            'ppw': run['protocol']['plan']['ppw'],
            'under_resolved_ppw': run['protocol']['plan']['under_resolved_ppw'],
            'phasor_saved': bool(run['meta'].get('phasor', {}).get('saved')),
            'note': 'diagnostic only; not convergence evidence',
        }
    return table


# ---------------------------------------------------------------------------
# aggregate report
# ---------------------------------------------------------------------------
def _fmt_re(metrics):
    return 'undefined' if metrics.get('re') is None else f'{metrics["re"]:.6e}'


def _fmt_order(order):
    if order.get('status') == 'defined':
        return f'{order["observed_order"]:.4f}'
    return f'undefined ({order.get("reason")})'


def _aggregate_md(agg):
    lines = [
        '# R2-FDTD-CONV-01：压力解空间/时间收敛聚合报告',
        '',
        f'- task: {agg["task_id"]} / protocol {agg["protocol_version"]}',
        f'- created (UTC): {agg["created_utc"]}',
        f'- solves: {agg["n_solves"]}/10 complete & ok',
        f'- out-root: `{agg["out_root"]}`',
        f'- solver: `{agg["solver_path"]}` (sha256 {agg["hashes"]["solver_sha256"][:16]}...)',
        f'- script sha256: {agg["hashes"]["script_sha256"][:16]}..., '
        f'config sha256: {agg["hashes"]["config_sha256"][:16]}...',
        '',
        '## 约定（聚合前已核）',
        '',
        f'- 主字段: `{agg["contract"]["primary_field"]}`（真实终时压力快照）；'
        f'补充字段: {", ".join("`%s`" % f for f in agg["contract"]["secondary_fields"])}',
        '- 空间 3 点：coarse/mid/fine 全部在 **coarse 气域节点**上直接按整数索引抽取，'
        '无插值、无坐标平移。',
        '- 时间 3 点：mid/mid_dt2/mid_dt4 固定 **mid 气域节点**，同物理终时 `Ncycle/f`、'
        '整数个源周期（终相 sin(2π f t)=0），无插值、无任意相位平移。',
        '- 指标：RE=||num-ref||₂/||ref||₂，MAE=mean|num-ref|，max=max|num-ref|；'
        '参考 = 相邻更细一档（细网格只是数值参考，不是解析真解）。',
        '- observed order 使用**同节点绝对 L2 差范数**之比：'
        'log₂(‖p_coarse−p_mid‖₂ / ‖p_mid−p_fine‖₂) / log₂(2)（不是 RE 之比，'
        '因为两 pair 的参考场范数不同）；仅当差范数严格单调下降且分母非零时报，'
        '否则显式 `undefined` 并给原因。RE 仍是主计分，不受此影响。',
        '',
        '## 协议身份与坐标核查',
        '',
        '| case | solves | identity 一致 | t_final 实测一致 | 空间 coords 嵌套精确 | 时间 coords 精确 |',
        '|---|---|---|---|---|---|',
    ]
    for case, info in agg['identity_checks'].items():
        lines.append(
            f'| {case} | {info["n_runs"]} | {info["all_equal"]} | '
            f'{info["t_final_all_equal"]} | {info["space_coords_nested_exact"]} | '
            f'{info["time_coords_equal"]} |')
    lines += ['', '## 空间收敛（共同节点抽取）', '']
    for case, blk in agg['space'].items():
        lines += [f'### {case}', '',
                  f'共同节点：{blk["target_nodes"]["n_x"]}×{blk["target_nodes"]["n_y"]}'
                  f'（{blk["target_nodes"]["description"]}）', '',
                  '| 字段 | pair | RE | MAE | max | ||num-ref|| | ||ref|| |',
                  '|---|---|---|---|---|---|---|']
        for field, fblk in blk['fields'].items():
            for pname, pair in fblk['pairs'].items():
                m = pair['metrics']
                lines.append(f'| {field} | {pname} | {_fmt_re(m)} | '
                             f'{m["mae"]:.6e} | {m["max"]:.6e} | '
                             f'{m["norm_diff"]:.6e} | {m["norm_ref"]:.6e} |')
        lines += ['', '| 字段 | ‖p_coarse−p_mid‖₂ | ‖p_mid−p_fine‖₂ | observed order |']
        lines += ['|---|---|---|---|']
        for field, fblk in blk['fields'].items():
            d_cm = fblk['pairs']['coarse_vs_mid']['metrics'].get('norm_diff')
            d_mf = fblk['pairs']['mid_vs_fine']['metrics'].get('norm_diff')
            lines.append(f'| {field} | '
                         f'{"undefined" if d_cm is None else f"{d_cm:.6e}"} | '
                         f'{"undefined" if d_mf is None else f"{d_mf:.6e}"} | '
                         f'{_fmt_order(fblk["observed_order"])} |')
        lines.append('')
    lines += ['## 时间收敛（固定 mid 网格）', '']
    for case, blk in agg['time'].items():
        lines += [f'### {case}', '',
                  f'固定节点：{blk["target_nodes"]["n_x"]}×{blk["target_nodes"]["n_y"]}'
                  f'（{blk["target_nodes"]["description"]}）', '',
                  '| 字段 | pair | RE | MAE | max | ||num-ref|| | ||ref|| |',
                  '|---|---|---|---|---|---|---|']
        for field, fblk in blk['fields'].items():
            for pname, pair in fblk['pairs'].items():
                m = pair['metrics']
                lines.append(f'| {field} | {pname} | {_fmt_re(m)} | '
                             f'{m["mae"]:.6e} | {m["max"]:.6e} | '
                             f'{m["norm_diff"]:.6e} | {m["norm_ref"]:.6e} |')
        lines += ['', '| 字段 | ‖p_mid−p_mid_dt2‖₂ | ‖p_mid_dt2−p_mid_dt4‖₂ | observed order |']
        lines += ['|---|---|---|---|']
        for field, fblk in blk['fields'].items():
            d_12 = fblk['pairs']['mid_vs_mid_dt2']['metrics'].get('norm_diff')
            d_23 = fblk['pairs']['mid_dt2_vs_mid_dt4']['metrics'].get('norm_diff')
            lines.append(f'| {field} | '
                         f'{"undefined" if d_12 is None else f"{d_12:.6e}"} | '
                         f'{"undefined" if d_23 is None else f"{d_23:.6e}"} | '
                         f'{_fmt_order(fblk["observed_order"])} |')
        lines.append('')
    phz = agg.get('phasor_supplementary')
    lines += ['## 复相量（补充）', '']
    if not phz or phz.get('status') != 'computed':
        lines.append(f'未计算：{phz.get("reason") if phz else "无"}')
    else:
        lines.append('复相量由实际 phase-locked snapshot 时刻经 `fit_complex_amplitude` '
                     '重新拟合，并按 `air_node_range_0based` 裁切到气域后比较；'
                     '仅作补充，不改变主快照指标。')
        lines.append('')
        for case, blk in phz['space'].items():
            lines.append(f'- {case} 空间：complex RE coarse→mid '
                         f'{_fmt_re(blk["pairs"]["coarse_vs_mid"]["metrics"])}, '
                         f'mid→fine {_fmt_re(blk["pairs"]["mid_vs_fine"]["metrics"])}; '
                         f'observed order {_fmt_order(blk["observed_order"])}')
        for case, blk in phz['time'].items():
            lines.append(f'- {case} 时间：complex RE mid→mid_dt2 '
                         f'{_fmt_re(blk["pairs"]["mid_vs_mid_dt2"]["metrics"])}, '
                         f'mid_dt2→mid_dt4 {_fmt_re(blk["pairs"]["mid_dt2_vs_mid_dt4"]["metrics"])}; '
                         f'observed order {_fmt_order(blk["observed_order"])}')
    lines += ['', '## 求解器诊断（原样记录）', '',
              '| case/res | ppw | under-resolved | steady_ok | rel_growth |'
              ' max\\|p\\| seen | broke_nan | diverged |',
              '|---|---|---|---|---|---|---|---|']
    for case, blk in agg['run_diagnostics'].items():
        for resname, row in blk.items():
            lines.append(f'| {case}/{resname} | {row["ppw"]:.2f} | '
                         f'{row["under_resolved_ppw"]} | {row["steady_ok"]} | '
                         f'{row["rel_growth"]} | {row["max_abs_p_seen"]:.6e} | '
                         f'{row["broke_nan"]} | {row["diverged"]} |')
    lines += ['', '## 边界与注意', '']
    for obs in agg['observations']:
        lines.append(f'- {obs}')
    lines.append('')
    return '\n'.join(lines)


def aggregate(out_root, register=True):
    """Verify the complete 10-solve set and write the convergence aggregate."""
    agg_json = os.path.join(out_root, 'aggregate.json')
    agg_md = os.path.join(out_root, 'aggregate.md')
    for path in (agg_json, agg_md):
        if os.path.exists(path):
            raise Refused(f'{rel_ws(path)} already exists; refusing to overwrite '
                          'the aggregate (use a new --out-root)')
    scan = scan_runs(out_root)
    pending_all = scan['pending'] + scan['invalid']
    if pending_all:
        doc = {
            'task_id': TASK_ID,
            'protocol_version': PROTOCOL_VERSION,
            'created_utc': now_utc(),
            'complete': False,
            'n_expected': scan['n_expected'],
            'n_complete': scan['n_complete'],
            'complete_runs': [{'case': r['case'], 'resolution': r['resolution'],
                               'dir': r['dir']} for r in scan['complete']],
            'pending': scan['pending'],
            'invalid': scan['invalid'],
            'note': 'aggregation and registry entry are deferred until all 10 solves '
                    'are complete & ok; nothing is fabricated for missing/failed runs',
        }
        os.makedirs(out_root, exist_ok=True)
        write_json(os.path.join(out_root, 'aggregate_pending.json'), doc)
        missing = [f'{p["case"]}/{p["resolution"]}: {p["reason"]}' for p in pending_all]
        raise IncompleteResults(
            f'{scan["n_complete"]}/{scan["n_expected"]} runs complete; pending/failed: '
            + '; '.join(missing)
            + f'. Wrote {rel_ws(os.path.join(out_root, "aggregate_pending.json"))}')

    runs = {}
    for case in CASES:
        for res in RESOLUTIONS:
            runs[(case, res)] = load_run(out_root, case, res)
    all_runs = list(runs.values())
    integrity = _verify_protocol_integrity(all_runs)

    identity_checks = {}
    case_identity = {}
    run_meta = {}
    for case in CASES:
        case_runs = [runs[(case, res)] for res in RESOLUTIONS]
        identity = validate_identity([r['protocol'] for r in case_runs])
        case_identity[case] = identity['identity']
        for run in case_runs:
            run_meta[f'{case}/{run["resolution"]}'] = _verify_run_meta(
                run, identity['identity'])
        identity_checks[case] = {'n_runs': identity['n_protocols'],
                                 'all_equal': True,
                                 'identity_sha256': identity['identity_sha256'],
                                 'keys': identity['keys']}
    # cross-resolution t_final / dt consistency for each case
    t_finals = {}
    for case in CASES:
        case_runs = [runs[(case, res)] for res in RESOLUTIONS]
        ts = [float((r['meta']['solver_params'] or {}).get('t_final_s'))
              for r in case_runs]
        target = float(case_identity[case]['t_final_target_s'])
        max_dev = max(abs(t - target) for t in ts)
        if max_dev > T_REL_TOL * target:
            raise ProtocolError(f'{case}: cross-resolution t_final deviation '
                                f'{max_dev:.3e} s exceeds tolerance')
        t_finals[case] = {'target_s': target, 'values': ts,
                          'max_abs_dev_s': max_dev}
        identity_checks[case]['t_final_all_equal'] = True

    # coordinate alignment evidence
    for case in CASES:
        space_runs = {res: runs[(case, res)] for res in SPATIAL_SERIES}
        xc = space_runs['coarse']['arrays']['x']
        yc = space_runs['coarse']['arrays']['y']
        for res in ('mid', 'fine'):
            x = space_runs[res]['arrays']['x']
            y = space_runs[res]['arrays']['y']
            stride = int(round((xc[1] - xc[0]) / (x[1] - x[0])))
            expected = x[::stride]
            if expected.shape != xc.shape or _max_dev(expected, xc) > COORD_ATOL_M:
                raise ProtocolError(f'{case}: {res} air coords are not an exact nested '
                                    'superset of the coarse air nodes')
            if _max_dev(y[::stride], yc) > COORD_ATOL_M:
                raise ProtocolError(f'{case}: {res} air y coords are not nested')
        identity_checks[case]['space_coords_nested_exact'] = True
        time_runs = {res: runs[(case, res)] for res in TEMPORAL_SERIES}
        xm = time_runs['mid']['arrays']['x']
        ym = time_runs['mid']['arrays']['y']
        for res in ('mid_dt2', 'mid_dt4'):
            if _max_dev(time_runs[res]['arrays']['x'], xm) > COORD_ATOL_M or \
                    _max_dev(time_runs[res]['arrays']['y'], ym) > COORD_ATOL_M:
                raise ProtocolError(f'{case}: {res} coords differ from the mid grid')
        identity_checks[case]['time_coords_equal'] = True

    phasor_available = [run['has_phasor'] for run in all_runs]
    if all(phasor_available):
        phasor_state = {'status': 'computed'}
    elif any(phasor_available):
        phasor_state = {'status': 'omitted',
                        'reason': 'inconsistent phasor availability across the 10 runs; '
                                  'supplementary phasor comparison skipped'}
    else:
        phasor_state = {'status': 'omitted',
                        'reason': 'all runs were saved without --save-phasor; '
                                  'supplementary phasor comparison skipped'}

    # comparisons per case
    space_all, time_all, diagnostics = {}, {}, {}
    for case in CASES:
        space_runs = {res: runs[(case, res)] for res in SPATIAL_SERIES}
        time_runs = {res: runs[(case, res)] for res in TEMPORAL_SERIES}
        space = compare_space(space_runs, phasor=(phasor_state['status'] == 'computed'))
        time_blk = compare_time(time_runs, phasor=(phasor_state['status'] == 'computed'))
        space_all[case] = space
        time_all[case] = time_blk
        diagnostics[case] = _diagnostics_table({res: runs[(case, res)]
                                                for res in RESOLUTIONS})
        if phasor_state['status'] == 'computed':
            phasor_state.setdefault('space', {})[case] = space['phasor_supplementary']
            phasor_state.setdefault('time', {})[case] = time_blk['phasor_supplementary']

    hashes = integrity['hashes']
    fixed_protocol = all_runs[0]['protocol']['fixed_protocol']
    contract = fixed_protocol['comparison_contract']
    observations = [
        f'完整 {scan["n_complete"]}/{scan["n_expected"]} 次真实求解均 status=ok，'
        '协议身份、代码/配置哈希、coords 嵌套与终时相位均已核验后才聚合。',
        '细网格（fine）只是数值参考，不是解析真解；本实验只比较同一物理终时快照。',
        '稳态/有限值/blow-up 诊断按原样记录，不作为压力收敛的证据；'
        'single final-time snapshot at >=4 air crossings 不等同稳态测量。',
        'observed order 由同节点绝对 L2 差范数之比计算'
        '（log₂(‖p_coarse−p_mid‖₂/‖p_mid−p_fine‖₂)/log₂(2)），不使用 RE 之比；'
        '仅对差范数严格单调下降且分母非零报告，非单调/零分母显式 undefined。',
        '未使用插值或任意相位平移；空间抽取为整数节点索引，时间三点固定同一 mid 网格。',
        '不设收敛通过阈值：不因出现正收敛或负收敛改变协议或挑选参数。',
    ]
    under = sorted({f'{case}/{res}' for case in CASES for res in RESOLUTIONS
                    if runs[(case, res)]['protocol']['plan']['under_resolved_ppw']})
    if under:
        observations.append(
            'ppw<10 的欠分辨诊断点仍保留在比较中，不剔除：' + ', '.join(under))
    steady_false = sorted(
        f'{case}/{res}' for case in CASES for res in RESOLUTIONS
        if not ((runs[(case, res)]['meta'].get('diagnostics') or {})
                .get('steady') or {}).get('steady_ok', False))
    if steady_false:
        observations.append('steady_ok=False 的配置（原样记录，不影响主比较）：'
                            + ', '.join(steady_false))

    created = now_utc()
    agg = {
        'task_id': TASK_ID,
        'protocol_version': PROTOCOL_VERSION,
        'created_utc': created,
        'complete': True,
        'n_solves': scan['n_complete'],
        'out_root': rel_ws(out_root),
        'solver_path': SOLVER_REL,
        'config_path': CONFIG_REL,
        'hashes': hashes,
        'fixed_protocol': fixed_protocol,
        'contract': contract,
        'identity_checks': identity_checks,
        'protocol_identity': case_identity,
        't_final_checks': t_finals,
        'run_meta': run_meta,
        'space': space_all,
        'time': time_all,
        'phasor_supplementary': phasor_state,
        'run_diagnostics': diagnostics,
        'observations': observations,
        'interpretation': 'convergence metrics computed from real p_total final-time '
                          'snapshots; fine grid is a numerical reference only; '
                          'per-pair metrics and undefined observed orders are '
                          'reported as computed, no pass/fail threshold',
    }
    write_json(agg_json, agg)
    write_text(agg_md, _aggregate_md(agg))
    stale_pending = os.path.join(out_root, 'aggregate_pending.json')
    if os.path.exists(stale_pending):
        os.remove(stale_pending)     # a completed aggregate supersedes the pending status file
    print(f'[aggregate] complete {scan["n_complete"]}/10; wrote '
          f'{rel_ws(agg_json)} and {rel_ws(agg_md)}', flush=True)

    if register:
        utc_tag = created.replace('+00:00', 'Z').replace('-', '').replace(':', '')
        registry_id = f'{TASK_ID}_{utc_tag}'
        entry = {
            'id': registry_id,
            'experiment_id': registry_id,
            'type': 'numerical_convergence',
            'tool': SCRIPT_REL,
            'created_utc': created,
            'task_id': TASK_ID,
            'protocol_version': PROTOCOL_VERSION,
            'n_solves': scan['n_complete'],
            'cases': list(CASES),
            'resolutions': list(RESOLUTIONS),
            'protocol': {
                'task_id': TASK_ID,
                'fixed_protocol': fixed_protocol,
                'comparison_contract': contract,
                'identity_checks': identity_checks,
                't_final_checks': t_finals,
                'hashes': hashes,
                'note': 'registered only after the complete 10/10 solve set passed '
                        'identity/coordinate checks; fine grid is a numerical '
                        'reference, not analytic truth',
            },
            'metrics': rel_ws(agg_json),
            'report': rel_ws(agg_md),
            'out_root': rel_ws(out_root),
            'slurm_job_ids': sorted({str((r['meta'].get('slurm') or {}).get('slurm_job_id'))
                                     for r in all_runs}),
            'interpretation': 'space/time pressure convergence metrics; no convergence '
                              'acceptance threshold; see aggregate report for caveats',
        }
        append_registry(entry)
        print(f'[aggregate] registry entry {registry_id} appended', flush=True)
        agg['registry_id'] = registry_id
    return agg


# ---------------------------------------------------------------------------
# dry run
# ---------------------------------------------------------------------------
DRY_RUN_COLUMNS = (
    'case', 'resolution', 'dx_m', 'n_pml_cells', 'n_buffer_cells', 'nxp1',
    'air_nodes_x', 'spp', 'dt_s', 'n_steps', 't_final_target_s',
    'courant_factor', 'ppw', 'source_x_m', 'under_resolved_ppw',
)


def dry_run_rows(case_name=None, res_name=None):
    """Rows for --dry-run: the fixed physical protocol, no solve and no writes."""
    cases = [case_name] if case_name else list(CASES)
    res_list = [res_name] if res_name else list(RESOLUTIONS)
    rows = []
    for case in cases:
        for res in res_list:
            plan = resolution_plan(case, res)
            rows.append({key: plan[key] for key in DRY_RUN_COLUMNS})
    return rows


def print_dry_run(case_name=None, res_name=None):
    identity = validate_config_identity()
    print(f'[{TASK_ID}] fixed physical protocol (dry run: no solve, no files written)')
    print(f'  config: {CONFIG_REL} sha256={identity["sha256"][:16]}... '
          f'(cases {sorted(identity["cases"])})')
    for case in ([case_name] if case_name else list(CASES)):
        plan = resolution_plan(case, 'coarse')
        print(f'  case {case}: f={plan["f_hz"]:g} Hz, Ma={plan["Ma"]:g}, '
              f'Ncycle={plan["Ncycle"]}, t_final=Ncycle/f={plan["t_final_target_s"]:.12f} s, '
              f's0={plan["s0"]}, source=left x={plan["source_x_m"]:g} m')
    rows = dry_run_rows(case_name, res_name)
    widths = {key: max(len(key), *(len(f'{row[key]:.6g}') if isinstance(row[key], float)
                                     else len(str(row[key])) for row in rows))
              for key in DRY_RUN_COLUMNS}
    header = '  ' + '  '.join(key.rjust(widths[key]) for key in DRY_RUN_COLUMNS)
    print(header)
    print('  ' + '  '.join('-' * widths[key] for key in DRY_RUN_COLUMNS))
    for row in rows:
        cells = []
        for key in DRY_RUN_COLUMNS:
            val = row[key]
            text = f'{val:.6g}' if isinstance(val, float) else str(val)
            cells.append(text.rjust(widths[key]))
        print('  ' + '  '.join(cells))
    print(f'  rows: {len(rows)}; the heavy solve is only allowed inside SLURM '
          f'(SLURM_JOB_ID), otherwise the entry refuses.')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--case', choices=tuple(CASES),
                    help='fixed case (required for a solve)')
    ap.add_argument('--resolution', choices=tuple(RESOLUTIONS),
                    help='fixed resolution (required for a solve)')
    ap.add_argument('--out-root', default=DEFAULT_OUT_ROOT,
                    help=f'output root (default {DEFAULT_OUT_ROOT}; must contain a '
                         'round2/r2 path component)')
    ap.add_argument('--aggregate-only', action='store_true',
                    help='only aggregate/verify all 10 solves (light I/O; no solve)')
    ap.add_argument('--dry-run', action='store_true',
                    help='print the fixed physical protocol; no solve, no writes')
    ap.add_argument('--save-phasor', action='store_true',
                    help='additionally save the air-cropped phase-locked complex '
                         'amplitude (supplementary; does not change primary metrics)')
    ap.add_argument('--version', action='version',
                    version=f'{TASK_ID} {PROTOCOL_VERSION}')
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        out_root = resolve_out_root(args.out_root)
        if args.dry_run:
            if args.aggregate_only:
                raise Refused('--dry-run and --aggregate-only are mutually exclusive')
            print_dry_run(args.case, args.resolution)
            return 0
        if args.aggregate_only:
            if args.case or args.resolution:
                raise Refused('--aggregate-only does not take --case/--resolution')
            aggregate(out_root)
            return 0
        if not args.case or not args.resolution:
            raise Refused('solve mode requires both --case and --resolution')
        run_one(out_root, args.case, args.resolution, save_phasor=args.save_phasor)
        return 0
    except IncompleteResults as exc:
        print(f'[pending] {exc}', file=sys.stderr)
        return 3
    except R2FdtdError as exc:
        print(f'[refuse] {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
