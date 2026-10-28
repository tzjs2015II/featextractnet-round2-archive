"""Post-process saved v2 FEM diagnostics with corrected acceptance criteria.

No solve is re-run. The true discrete Galerkin residual is
``weak_residual.blocks.full.free`` (the full assembled nonlinear residual after
zeroing every velocity/temperature Dirichlet row). The previous
momentum/energy "weak" numbers are cell-wise strong residual projections against
test functions; they are retained and renamed as a diagnostic, not used as the
nonlinear residual gate.

The gate is absolute free residual <= 1e-12 (solver absolute update tolerance)
and free/full <= 1e-8 (the original weak-residual tolerance), plus existing
field/BC/mass checks and Newton convergence. A deliberately large free residual
with converged Newton and small cell projection must be rejected.
"""
import hashlib, json, os, sys

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_LEVELS = os.path.join(WS, "results/round2_thermal_fem/v2/levels/levels_geom17.json")
IN_REPS = os.path.join(WS, "results/round2_thermal_fem/v2/representatives/representatives_46_96.json")
OUT = os.path.join(WS, "results/round2_thermal_fem/v3_postprocess")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluate(record):
    w = record.get("weak_residual", {})
    blocks = w.get("blocks", {})
    full = blocks.get("full", {})
    free_abs = full.get("free")
    raw = full.get("full")
    free_rel = (free_abs / raw) if (free_abs is not None and raw not in (None, 0)) else None
    old = record.get("acceptance", {}).get("checks", {})
    checks = {
        "galerkin_free_absolute": free_abs is not None and free_abs <= 1e-12,
        "galerkin_free_relative": free_rel is not None and free_rel <= 1e-8,
        "newton_converged": int(record.get("newton_reason", -1)) == 2,
        "wall_velocity": bool(old.get("wall_velocity", False)),
        "temperature_bc": bool(old.get("temperature_bc", False)),
        "mass_flux": bool(old.get("mass_flux", False)),
        "finite_all_fields": bool(old.get("finite_all_fields", False)),
    }
    cell = {name: blocks.get(name) for name in ("continuity", "momentum", "energy")}
    return {"checks": checks, "all_passed": all(checks.values()),
            "galerkin_free_absolute": free_abs,
            "galerkin_free_relative": free_rel,
            "galerkin_raw_full": raw,
            "cell_strong_against_test_projection": cell,
            "note": "free is the assembled nonlinear residual after zeroing all "
                    "Dirichlet rows; cell projection is diagnostic only"}


def main():
    records = []
    for path in (IN_LEVELS, IN_REPS):
        doc = json.load(open(path, encoding="utf-8"))
        for r in doc["records"]:
            item = dict(r)
            item["saved_file"] = os.path.relpath(path, WS)
            item["saved_file_sha256"] = sha(path)
            item["postprocess"] = evaluate(r)
            records.append(item)
    bad = [r for r in records if not r["postprocess"]["all_passed"]]
    fake = {"newton_reason": 2,
            "acceptance": {"checks": {"wall_velocity": True,
                                       "temperature_bc": True,
                                       "mass_flux": True,
                                       "finite_all_fields": True}},
            "weak_residual": {"blocks": {"full": {"free": 1.0, "full": 0.1},
                                          "momentum": {"relative_free": 0.0},
                                          "energy": {"relative_free": 0.0}}}}
    fake_check = not evaluate(fake)["all_passed"]
    out = {"n_records": len(records), "n_passed": len(records) - len(bad),
           "n_failed": len(bad), "failed": [
               {"geom": r.get("geom", 17), "h": r.get("target_h")} for r in bad],
           "intentionally_large_free_rejected": bool(fake_check),
           "records": records,
           "postprocess_source_sha256": sha(__file__),
           "saved_inputs": {os.path.relpath(p, WS): sha(p)
                            for p in (IN_LEVELS, IN_REPS)},
           "note": "No solve was rerun. Old v2 acceptance statements are archived "
                   "in archive_v2_prepost; this is a post-process re-acceptance."}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "reacceptance.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    lines = ["# v3 post-process re-acceptance (no re-solve)", "",
             f"Records: {out['n_records']}; passed: {out['n_passed']}; failed: {out['n_failed']}",
             f"Intentional large-free rejection works: {out['intentionally_large_free_rejected']}", "",
             "| geom | h | free absolute | free/full | all_passed |",
             "|---|---|---|---|---|"]
    for r in records:
        p = r["postprocess"]
        lines.append(f"| {r.get('geom',17)} | {r.get('target_h')} | {p['galerkin_free_absolute']:.3e} | "
                     f"{p['galerkin_free_relative']:.3e} | {p['all_passed']} |")
    lines += ["", "Cell projection numbers are retained as diagnostics only; they are not the Galerkin residual."]
    with open(os.path.join(OUT, "reacceptance.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(json.dumps({"n_records": out["n_records"], "n_passed": out["n_passed"],
                      "n_failed": out["n_failed"],
                      "large_free_rejected": out["intentionally_large_free_rejected"]},
                     indent=2))
    return 0 if out["n_failed"] == 0 and fake_check else 1


if __name__ == "__main__":
    sys.exit(main())
