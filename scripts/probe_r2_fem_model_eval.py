"""Pre-flight probe for R2-THERMAL-FEM-02 frozen-model FEM-support eval."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WS not in sys.path:
    sys.path.insert(0, WS)

from scripts import exp_r2_new_queries as NQ  # noqa: E402
from src.fem_reference import geometry  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
GEOMS = (46, 17, 96)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(
        WS, "results", "round2_thermal_fem", "model_eval"))
    args = ap.parse_args(argv)
    out = os.path.abspath(args.out_dir)
    checks = []
    protocol = os.path.join(out, "protocol_snapshot.json")
    if not os.path.isfile(protocol):
        print(json.dumps({"ok": False, "reason": "protocol_snapshot.json missing"}))
        return 1
    doc = json.load(open(protocol, encoding="utf-8"))
    checks.append(("protocol_hash_files", all(
        os.path.isfile(os.path.join(WS, rel)) and
        sha(os.path.join(WS, rel)) == h
        for rel, h in doc.get("source_hashes", {}).items())))
    spec = json.load(open(os.path.join(WS, "results/thermal_geometry_spec.json"),
                          encoding="utf-8"))
    per_geo = {int(r["geom"]): r for r in spec["per_geometry"]}
    data = np.load(os.path.join(os.path.dirname(os.path.dirname(WS)), "HeatTransferData.npy"))
    for geom in doc.get("geometries", list(GEOMS)):
        support = os.path.join(out, f"support_geom{geom}.npz")
        fem = os.path.join(out, f"fem_reference_geom{geom}.npz")
        checks.append((f"support_{geom}", os.path.isfile(support)))
        checks.append((f"fem_ref_{geom}", os.path.isfile(fem)))
        if os.path.isfile(support) and os.path.isfile(fem):
            s = np.load(support)
            f = np.load(fem)
            coords = s["coords"]
            checks.append((f"support_shape_{geom}",
                           coords.shape == (5000, 2)))
            checks.append((f"support_finite_{geom}",
                           bool(np.all(np.isfinite(coords)))))
            checks.append((f"fem_ref_finite_{geom}",
                           bool(np.all(np.isfinite(f["ref"])))))
            checks.append((f"fem_ref_shape_{geom}", f["ref"].shape == (5000, 4)))
            checks.append((f"support_fem_coords_match_{geom}",
                           np.array_equal(coords, f["coords"])))
    for model_name, base in (
            ("FEN", "results/round2_thermal_symmetric/feat_extract"),
            ("graph", "results/round2_graph_five_seed_main/meshgraphnet")):
        for seed in SEEDS:
            ckpt = os.path.join(WS, base, f"seed{seed}", "checkpoint_best.pth")
            metrics = os.path.join(WS, base, f"seed{seed}", f"metrics_seed{seed}.json")
            checks.append((f"{model_name}_seed{seed}_ckpt", os.path.isfile(ckpt)))
            checks.append((f"{model_name}_seed{seed}_metrics", os.path.isfile(metrics)))
            if os.path.isfile(metrics):
                m = json.load(open(metrics, encoding="utf-8"))
                checks.append((f"{model_name}_seed{seed}_stats",
                               all(k in m["protocol"] for k in
                                   ("x_mean", "x_std", "y_mean", "y_std"))))
    for model_name in ("FEN", "graph"):
        for seed in SEEDS:
            for geom in doc.get("geometries", list(GEOMS)):
                path = os.path.join(out, f"pred_{model_name}_seed{seed}_geom{geom}.npz")
                checks.append((f"pred_{model_name}_{seed}_{geom}", os.path.isfile(path)))
                if os.path.isfile(path):
                    pred = np.load(path)
                    checks.append((f"pred_shape_{model_name}_{seed}_{geom}",
                                   pred["prediction"].shape == (5000, 4)))
                    checks.append((f"pred_finite_{model_name}_{seed}_{geom}",
                                   bool(np.all(np.isfinite(pred["prediction"])))))
    fen = os.path.join(WS, "src/fem_reference/featextractnet_symmetric.py")
    checks.append(("fen_independent_copy_sha_9f8d2",
                   os.path.isfile(fen) and sha(fen).startswith("9f8d2")))
    failed = [name for name, ok in checks if not ok]
    result = {"ok": not failed, "n_checks": len(checks), "failed": failed,
              "out_dir": out}
    print(json.dumps(result, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
