"""Generate the acoustic FDTD dataset into data/acoustic/ (Opt-1 grid).

Purpose
-------
Create the parametric acoustic scattering cases used by the FeatExtractNet
返稿 experiments. Grid and case table follow
``docs/A_声学数据与任务设计.md`` (author decision 2026-09-10):

  * f in {100, 200, 300, 400, 500} Hz  x  Ma in {0.2, 0.3, 0.5, 0.7}
    -> 20 cases
  * one variable-source case: f=400 Hz, Ma=0.5, plane wave injected from the
    right edge of the air region (backward travelling)              -> 1 case

Resolution (Opt-1): dx = dy = 1/16 m, n_pml = 160 cells (10 m PML),
n_buffer = 240 cells -> air region 2*240*dx = 30 m, total grid 800x800 cells
(801x801 nodes). The node count equals the previous 0.125 m dataset, so the
cost per step is unchanged, while cells-per-wavelength improves by 2x:

    f (Hz)   :  100    200    300    400    500
    ppw      : 54.4   27.2   18.1   13.6   10.9     (>= 10 required)

800 Hz is deliberately excluded: at dx=0.125 m it had only 3.4 points per
wavelength and was dominated by numerical dispersion (see the design doc §4).

Step count: the run must cover ``--n-transit`` (default 3) crossings of the air
region at c0, and is rounded up to a whole number of ``4`` acoustic periods so
that the phase-locked period average used by the steady-state check is exact.

Self-check (replaces the previous aliased trend test)
-----------------------------------------------------
The instantaneous air-region energy ``mean(p^2)`` oscillates at ``2*omega``.
It is now recorded **every step** over the final 4 acoustic periods and averaged
per integer period; the field counts as steady when consecutive period means
differ by <= 2 %. A case whose per-step energy series is persisted
(``energy_tail`` in the npz) so the convergence can be re-inspected.

A case that goes non-finite is recorded as ``status="nan"`` and is **not**
written to disk; the batch then exits non-zero, so an unconverged or broken case
can no longer pass silently.

Phase-locked snapshots
----------------------
8 pressure snapshots spanning the final acoustic period are stored
(``p_phase``) together with the least-squares complex amplitude
(``p_phasor_re`` / ``p_phasor_im``) and the relative linear-drift diagnostic
(``p_phasor_drift_rel``). These are the inputs a frequency-domain
(Helmholtz / convected Helmholtz) physics formulation needs, so the dataset
does not have to be regenerated for it.

Usage
-----
  python3 scripts/generate_acoustic_cases.py --workers 10            # all
  python3 scripts/generate_acoustic_cases.py --cases 400_0.5        # subset
  python3 scripts/generate_acoustic_cases.py --cases 400_0.5,200_0.3

Outputs
-------
  data/acoustic/case_{case_id}.npz
  configs/acoustic_cases.json       (machine-readable, per-case protocol+roles)
  data/acoustic/generation_report.md (human-readable summary)
  results/registry.json             (experiment registry entry, AGENTS §3.4)
"""

import argparse
import datetime as _dt
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WS)

from src.fdtd import run_case  # noqa: E402
from src.fdtd.python_fdtd import STEADY_N_PERIOD  # noqa: E402

DATA_DIR = os.path.join(WS, "data", "acoustic")
CONFIG_PATH = os.path.join(WS, "configs", "acoustic_cases.json")
REPORT_PATH = os.path.join(DATA_DIR, "generation_report.md")
REGISTRY_PATH = os.path.join(WS, "results", "registry.json")

# --- Opt-1 grid / protocol constants (see docs/A_声学数据与任务设计.md §4) ---
DEFAULT_RES = 16            # dx = 1/RES metres  -> 0.0625 m
DEFAULT_N_TRANSIT = 3.0     # air-region crossings required before the tail window
N_PML = 160                 # PML cells per side (10 m at RES=16)
N_BUFFER = 240              # air cells per side -> air span 2*N_BUFFER*dx = 30 m
C0 = 340.0
COURANT = 0.1
PPW_MIN = 10.0              # minimum points per wavelength accepted

CASE_F = (100.0, 200.0, 300.0, 400.0, 500.0)
CASE_MA = (0.2, 0.3, 0.5, 0.7)
SRC_R_CASE = {"case_id": "400_0.5_srcR", "f": 400.0, "Ma": 0.5,
              "source_side": "right"}

# train/val/test roles (docs/A_声学数据与任务设计.md §5.1).
# test cases never share an (f, Ma) pair with train.
ROLES = {
    "train": ["200_0.3", "200_0.5", "200_0.7",
              "300_0.5", "400_0.3", "400_0.5"],
    "val": ["400_0.7"],
    "test": ["100_0.3", "100_0.5", "100_0.7",
             "300_0.3", "300_0.7",
             "500_0.2", "500_0.3", "500_0.5", "500_0.7",
             "400_0.5_srcR"],
    # reserved for the frequency/Ma sweep extension (A4), not used by A2
    "unused": ["100_0.2", "200_0.2", "300_0.2", "400_0.2"],
}


def _dt_for(dx: float) -> float:
    """CFL-style time step of the MATLAB reference formula."""
    return COURANT / (C0 * np.sqrt(1.0 / dx**2 + 1.0 / dy_for(dx) ** 2))


def dy_for(dx: float) -> float:
    return dx


def steps_for_case(f: float, dx: float, n_transit: float) -> tuple:
    """Return ``(n_steps, dt, steps_per_period)`` for one case.

    ``n_steps`` covers ``n_transit`` air-region crossings and is rounded up to a
    whole number of ``STEADY_N_PERIOD`` acoustic periods.
    """
    dt = _dt_for(dx)
    air_span = 2.0 * N_BUFFER * dx
    t_need = n_transit * air_span / C0
    sph = int(round(1.0 / (f * dt)))                 # steps per acoustic period
    block = STEADY_N_PERIOD * sph
    n_steps = int(np.ceil((t_need / dt) / block) * block)
    return n_steps, dt, sph


# ---------------------------------------------------------------------------
# case table
# ---------------------------------------------------------------------------
def build_cases(res: int = DEFAULT_RES, n_transit: float = DEFAULT_N_TRANSIT,
                n_steps_override: int = 0) -> list:
    """Return the ordered case list (5 f x 4 Ma + 1 variable-source case)."""
    dx = 1.0 / float(res)
    cases = []
    for f in CASE_F:
        for Ma in CASE_MA:
            cid = f"{f:.0f}_{Ma:.1f}"
            n_steps, dt, sph = steps_for_case(f, dx, n_transit)
            cases.append({
                "case_id": cid,
                "f": float(f), "Ma": float(Ma), "source_side": "left",
                "res": int(res), "dx": dx, "n_buffer": N_BUFFER,
                "n_steps": int(n_steps_override or n_steps),
                "monitor_every": sph,
                "dt": dt,
                "description": f"f={f:.0f} Hz, Ma={Ma:.1f}, plane wave from left",
            })
    n_steps, dt, sph = steps_for_case(SRC_R_CASE["f"], dx, n_transit)
    cases.append({
        **SRC_R_CASE,
        "res": int(res), "dx": dx, "n_buffer": N_BUFFER,
        "n_steps": int(n_steps_override or n_steps),
        "monitor_every": sph, "dt": dt,
        "description": f"f={SRC_R_CASE['f']:.0f} Hz, Ma={SRC_R_CASE['Ma']:.1f}, "
                       "plane wave injected from the right edge (travels leftwards)",
    })
    for c in cases:
        c["role"] = next((r for r, ids in ROLES.items() if c["case_id"] in ids),
                         "unassigned")
    return cases


def case_filename(case_id: str) -> str:
    return os.path.join(DATA_DIR, f"case_{case_id}.npz")


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------
def _run_one(case: dict) -> dict:
    """Run one case on a worker process and save its npz."""
    t0 = time.perf_counter()
    rec = {"case_id": case["case_id"], "role": case["role"]}
    try:
        res = run_case(
            f=case["f"], Ma=case["Ma"], n_steps=case["n_steps"],
            dx=case["dx"], dy=case["dx"],
            n_pml=N_PML, n_buffer=case["n_buffer"],
            source_side=case["source_side"],
            monitor_every=case["monitor_every"],
            early_stop_on_nan=case.get("early_stop_on_nan", True),
            phase_snapshots=True, n_phase=8,
        )
        prm = res["params"]
        prm["source_side"] = case["source_side"]
        prm["role"] = case["role"]
        d = res["diag"]
        ppw = float(prm["cells_per_lambda"])

        if (d.get("diverged") or d["broke_nan"]
                or not np.isfinite(res["p_total"]).all()):
            # A large-but-finite blow-up (`diverged`) is treated exactly like a
            # NaN: never written to disk. Before the divergence guard existed the
            # 2026-09-11 calibration probe passed an ~1e70 field through as
            # "finite" and its steady-state verdict was a false positive (see
            # docs/A_声学数据与任务设计.md §2.3.1).
            bad = ("nan" if (d["broke_nan"]
                             or not np.isfinite(res["p_total"]).all())
                   else "diverged")
            rec.update({"status": bad,
                        "wall_s": round(time.perf_counter() - t0, 2),
                        "n_steps_done": int(d["n_steps_done"]),
                        "cells_per_lambda": ppw,
                        "diverged": bool(d.get("diverged")),
                        "max_abs_p_seen": d.get("max_abs_p_seen"),
                        "steady": d["steady"]})
            print(f"[gen] {case['case_id']}: {bad.upper()}, not saved "
                  f"(max|p|={d.get('max_abs_p_seen')})", flush=True)
            return rec

        meta_str = json.dumps(prm, indent=1, sort_keys=True)
        phz = res.get("phasor") or {}
        path = case_filename(case["case_id"])
        payload = {
            "p_total": res["p_total"],
            "p_incident": res["p_incident"],
            "p_scattered": res["p_scattered"],
            "x": res["x"], "y": res["y"],
            "meta": np.array(meta_str),
            "energy_tail": np.asarray(d["energy_tail"], dtype=np.float64),
            "phase_steps": np.asarray(d["phase_steps"], dtype=np.int64),
        }
        if res.get("p_phase"):
            payload["p_phase"] = np.stack(res["p_phase"]).astype(np.float32)
        if phz:
            payload["p_phasor_re"] = phz["phasor_re"].astype(np.float32)
            payload["p_phasor_im"] = phz["phasor_im"].astype(np.float32)
            payload["p_phasor_drift_rel"] = phz["drift_rel"].astype(np.float32)
        np.savez(path, **payload)

        rec.update({
            "file": os.path.relpath(path, WS),
            "status": "ok",
            "shape": [int(res["p_total"].shape[0]), int(res["p_total"].shape[1])],
            "wall_s": round(time.perf_counter() - t0, 2),
            "n_steps_done": int(d["n_steps_done"]),
            "seconds_per_step": d["seconds_per_step"],
            "broke_nan": bool(d["broke_nan"]),
            "max_abs_p_final": d["max_abs_p_final"],
            "rms_p_air_final": d["rms_p_air_final"],
            "steady": d["steady"],
            "cells_per_lambda": ppw,
            "res_ok": bool(ppw >= PPW_MIN),
            "phasor_drift_rel_max": d.get("phasor_drift_rel_max"),
            "params": prm,
        })
        print(f"[gen] {case['case_id']}: OK in {rec['wall_s']:.1f} s "
              f"({d['seconds_per_step'] * 1e3:.1f} ms/step) | "
              f"ppw={ppw:.1f} | steady_ok={d['steady']['steady_ok']} "
              f"(rel_growth={d['steady']['rel_growth']:.4f}) | "
              f"drift_rel={rec['phasor_drift_rel_max']:.3e}", flush=True)
    except Exception as exc:  # noqa: BLE001 - worker must not kill the pool
        rec.update({"status": "error", "error": repr(exc),
                    "wall_s": round(time.perf_counter() - t0, 2)})
        print(f"[gen] {case['case_id']}: ERROR {exc!r}", flush=True)
    return rec


# ---------------------------------------------------------------------------
# registry / report helpers
# ---------------------------------------------------------------------------
def _append_registry(entry: dict) -> None:
    reg = []
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "r", encoding="utf-8") as fh:
            reg = json.load(fh)
    reg.append(entry)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as fh:
        json.dump(reg, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def _write_report(cases_done: list, protocol: dict, started_utc: str,
                  wall_s: float) -> None:
    lines = [
        "# Acoustic FDTD dataset generation report",
        "",
        f"- generated (UTC): {started_utc}",
        f"- wall time (batch): {wall_s / 60:.1f} min",
        "- solver: src/fdtd/python_fdtd.py (port of the codematlab/ discretisation: "
        "same PML coefficients, 4-stage RK4, 2nd-order central differences and hard "
        "line source; NOT a pointwise-validated reproduction, see module docstring)",
        f"- protocol snapshot: {json.dumps(protocol, sort_keys=True)}",
        f"- roles: {json.dumps(ROLES, sort_keys=True)}",
        "",
        "## Per-case summary",
        "",
        "| case | role | f (Hz) | Ma | source | ppw | steps | spp | wall (s) | "
        "ms/step | max\\|p\\| | steady_ok | rel_growth | phase_err | drift_rel | file |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in cases_done:
        if r["status"] != "ok":
            lines.append(f"| {r['case_id']} | {r.get('role', '-')} | - | - | - | - | "
                         f"- | - | {r.get('wall_s', float('nan'))} | - | - | - | "
                         f"**{r['status'].upper()}** | - | - | "
                         f"{r.get('error', '')} |")
            continue
        st = r["steady"]
        p = r["params"]
        lines.append(
            f"| {r['case_id']} | {r['role']} | {p['f_hz']:.0f} | {p['Ma']:.1f} | "
            f"{p['source_side']} | {r['cells_per_lambda']:.1f} | {r['n_steps_done']} | "
            f"{st.get('steps_per_period', 0)} | {r['wall_s']:.1f} | "
            f"{r['seconds_per_step'] * 1e3:.1f} | {r['max_abs_p_final']:.3e} | "
            f"{st['steady_ok']} | {st['rel_growth']:.4f} | "
            f"{st.get('phase_err', float('nan')):.2e} | "
            f"{(r.get('phasor_drift_rel_max') or float('nan')):.3e} | {r['file']} |")
    lines += [
        "",
        "Notes:",
        "- steady_ok is decided by a **phase-locked integer-period average**: the",
        "  per-step air-region energy is averaged over each of the final 4 acoustic",
        "  periods and the run counts as steady when consecutive period means agree",
        "  within 2 %. This replaces the previous linear-trend test, which was",
        "  aliased by the 2*omega oscillation of the instantaneous energy.",
        "- phase_err is the residual phase error of rounding the acoustic period to",
        "  a whole number of steps; drift_rel is the least-squares linear-drift",
        "  coefficient of the phase-locked snapshots divided by the amplitude",
        "  (large values mean the snapshots were not taken in a time-harmonic regime).",
        f"- ppw = cells per wavelength = c0/(f*dx); PPW_MIN = {PPW_MIN:g} is required.",
        "- A non-finite case is recorded as NAN and is not written to disk; the batch",
        "  then exits non-zero.",
    ]
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", default="all",
                    help="comma-separated case ids or 'all'")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--res", type=int, default=DEFAULT_RES,
                    help="dx = 1/res metres (default 16 -> 0.0625 m)")
    ap.add_argument("--n-transit", type=float, default=DEFAULT_N_TRANSIT,
                    help="air-region crossings the run must cover (default 3)")
    ap.add_argument("--n-steps", type=int, default=0,
                    help="override the derived step count (0 = derive per case)")
    ap.add_argument("--no-early-stop", action="store_true",
                    help="do not abort a case if a NaN appears")
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)

    dx = 1.0 / float(args.res)
    cases = build_cases(args.res, args.n_transit, args.n_steps)
    if args.cases != "all":
        wanted = {c.strip() for c in args.cases.split(",") if c.strip()}
        cases = [c for c in cases if c["case_id"] in wanted]
    if not cases:
        print("no matching cases; available:",
              [c["case_id"] for c in build_cases(args.res, args.n_transit)])
        return 2

    for c in cases:
        c["early_stop_on_nan"] = not args.no_early_stop

    protocol = {
        "solver": "src/fdtd/python_fdtd.py",
        "grid": f"{2 * (N_PML + N_BUFFER) + 1}x{2 * (N_PML + N_BUFFER) + 1} nodes "
                f"(PML {N_PML} + air buffer {N_BUFFER} cells/side)",
        "crop": f"[pis:pie] x [pjs:pje] = {2 * N_BUFFER + 1}x{2 * N_BUFFER + 1} nodes, "
                f"x/y in [-{N_BUFFER * dx:g},{N_BUFFER * dx:g}] m",
        "dx_dy_m": dx, "courant_factor": COURANT, "c0": C0, "rho0": 1.0,
        "L": 1.0, "alpha": 1.256431, "gamma_coef": 2.8,
        "sigma_pml": 100.0, "n_pml": N_PML, "n_buffer": N_BUFFER,
        "n_steps": "per-case, derived from n_transit and rounded to whole periods",
        "n_transit": args.n_transit,
        "steady_criterion": "phase-locked integer-period mean over "
                            f"{STEADY_N_PERIOD} periods, tol 2%",
        "ppw_min": PPW_MIN,
        "phase_snapshots": 8,
        "source": "hard plane-wave line source sin(2*pi*f*t) on p and P",
        "legacy_matlab": "codematlab/ (read-only; discretisation port, not "
                         "pointwise-validated)",
        "seed": None,
    }

    started_utc = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    t_all = time.perf_counter()
    max_steps = max(c["n_steps"] for c in cases)
    print(f"[gen] launching {len(cases)} cases with {args.workers} workers | "
          f"dx=1/{args.res} m | max n_steps={max_steps}", flush=True)

    if args.workers > 1:
        with mp.Pool(processes=args.workers) as pool:
            records = pool.map(_run_one, cases)
    else:
        records = [_run_one(c) for c in cases]

    wall_s = time.perf_counter() - t_all
    n_ok = sum(1 for r in records if r["status"] == "ok")
    n_bad = len(records) - n_ok
    print(f"[gen] done: {n_ok}/{len(records)} ok, {n_bad} failed "
          f"(nan/error), wall {wall_s:.1f} s", flush=True)

    doc = {
        "generator": os.path.basename(__file__),
        "created_utc": started_utc,
        "protocol": protocol,
        "roles": ROLES,
        "n_cases": len(records),
        "n_ok": n_ok,
        "n_error": n_bad,
        "batch_wall_s": round(wall_s, 2),
        "cases": {r["case_id"]: r for r in records},
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    _write_report(records, protocol, started_utc, wall_s)

    _append_registry({
        "id": f"acoustic_fdtd_dataset_{started_utc}",
        "type": "data_generation",
        "tool": os.path.basename(__file__),
        "created_utc": started_utc,
        "protocol": protocol,
        "roles": ROLES,
        "cases": {r["case_id"]: {k: r[k] for k in
                                 ("role", "status", "file", "shape", "wall_s",
                                  "n_steps_done", "broke_nan", "steady",
                                  "cells_per_lambda", "res_ok")
                                 if k in r}
                  for r in records},
        "config_file": os.path.relpath(CONFIG_PATH, WS),
    })

    print(f"[gen] config written to {os.path.relpath(CONFIG_PATH, WS)}")
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
