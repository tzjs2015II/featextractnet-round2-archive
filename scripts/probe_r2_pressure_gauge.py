"""probe_r2_pressure_gauge.py —— R2-PRESSURE-GAUGE-02 轻量解析 + 真实单点 probe。

不做批量 GPU 推理、不训练：
  1. 解析检查压力指标实现：平移不变性（预测与参考统一算子）、零中心化参考 norm 标
     undefined、独立恒等式 ||p̂-p||² = ||Cp̂-Cp||² + N*bias²、N 点等权均值；
  2. 坏实现可识别：只中心化预测不中心化参考、用 epsilon 把 undefined 伪造成有限值；
  3. 真实轻量：对 FEN/graph seed0 × 第一个 test 几何各做一次 CPU full-5000 前向
     （若环境缺 graph 依赖则明确记录 skipped，不安装），复算 p 原 RE/MAE/max 与
     metrics JSON 一致，并给出中心化指标/bias/分母/恒等式残差。
输出仅 probe_report.{json,md} 到 --out-dir（默认 results/round2_pressure_gauge/probe），
不写主协议/raw/summary，不覆盖已有 probe 报告。
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch

# 本地 Windows probe 没有 POSIX fcntl；仅 probe 进程注入 no-op stub，便于加载
# 生产模块。服务器 Linux 不会走此分支，不改变生产行为。
try:
    import fcntl  # noqa: F401
except ImportError:
    import types
    _fcntl = types.ModuleType("fcntl")
    _fcntl.flock = lambda *a, **k: 0
    _fcntl.LOCK_EX = 1
    _fcntl.LOCK_UN = 2
    sys.modules["fcntl"] = _fcntl

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WS not in sys.path:
    sys.path.insert(0, WS)


def _load_main():
    path = os.path.join(WS, "scripts", "exp_r2_pressure_gauge.py")
    spec = importlib.util.spec_from_file_location("pressure_gauge_main", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PG = _load_main()


def _close(a, b, tol=1e-10):
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))


def analytic_checks():
    rng = np.random.default_rng(20261022)
    ref = rng.normal(size=37).astype(np.float64)
    pred = ref + rng.normal(scale=0.3, size=37)
    base = PG.p_metrics(pred, ref)
    shifts = (-3.7, 2.5, 0.125)
    trans = {}
    for c in shifts:
        m = PG.p_metrics(pred, ref, shift=c)
        trans[str(c)] = {
            "original_re_same": _close(m["original"]["re"], base["original"]["re"]),
            "original_mae_same": _close(m["original"]["mae"], base["original"]["mae"]),
            "original_max_same": _close(m["original"]["max"], base["original"]["max"]),
            "centered_re_same": _close(m["centered"]["re"], base["centered"]["re"]),
            "centered_mae_same": _close(m["centered"]["mae"], base["centered"]["mae"]),
            "centered_max_same": _close(m["centered"]["max"], base["centered"]["max"]),
            "bias_same": _close(m["bias_pred_minus_ref"], base["bias_pred_minus_ref"]),
            "mean_ref_shifted": _close(m["mean_ref"], base["mean_ref"] + c),
        }
    translation_ok = all(
        (d["original_mae_same"] and d["original_max_same"]
         and d["centered_re_same"] and d["centered_mae_same"]
         and d["centered_max_same"] and d["bias_same"]
         and d["mean_ref_shifted"])
        for d in trans.values())
    original_re_changes_as_expected = all(
        not d["original_re_same"] for d in trans.values())

    # zero centered norm: constant reference; centered RE must be None (no epsilon)
    ref_const = np.full(11, 5.0)
    pred_const = np.full(11, 7.0)
    m_const = PG.p_metrics(pred_const, ref_const)
    zero_undefined = m_const["centered"]["re"] is None
    wrong_eps_re = float(np.linalg.norm(pred_const - ref_const)
                         / max(float(np.linalg.norm(ref_const - ref_const.mean())), 1e-12))
    epsilon_bad_detected = zero_undefined and wrong_eps_re > 0.0

    # identity on a biased pure-shift field
    ref_shift = rng.normal(size=101)
    pred_shift = ref_shift + 2.5
    m_shift = PG.p_metrics(pred_shift, ref_shift)
    identity_ok = abs(m_shift["identity"]["relative_residual"]) <= 1e-12
    centered_zero = m_shift["centered"]["re"] is not None and abs(m_shift["centered"]["re"]) <= 1e-12
    bias_ok = _close(m_shift["bias_pred_minus_ref"], 2.5)

    # equal-weight arithmetic mean (not area weighted)
    ref_unequal = np.array([1.0, 2.0, 100.0])
    m_mean = PG.p_metrics(ref_unequal, ref_unequal)
    arithmetic_ok = _close(m_mean["mean_ref"], float(np.mean(ref_unequal)))
    area_weighted = np.average(ref_unequal, weights=np.array([100.0, 1.0, 1.0]))
    area_differs = not _close(area_weighted, m_mean["mean_ref"])

    # bad implementation: center prediction only, leave reference uncentered
    ref_bad = np.array([1.0, 2.0, 3.0, 4.0])
    pred_bad = ref_bad + 1.0
    correct = PG.p_metrics(pred_bad, ref_bad)
    wrong_one_sided = float(np.linalg.norm((pred_bad - pred_bad.mean()) - ref_bad)
                            / np.linalg.norm(ref_bad))
    one_sided_detected = (abs(correct["centered"]["re"]) <= 1e-12
                          and wrong_one_sided > 1e-6)

    checks = {
        "translation_invariance": {
            "pass": translation_ok and original_re_changes_as_expected,
            "centered_and_mae_max_invariant": translation_ok,
            "original_re_changes_as_expected": original_re_changes_as_expected,
            "note": ("original RE denominator ||ref|| changes under translation; "
                     "centered metrics are translation invariant"),
            "shifts": trans,
        },
        "zero_centered_norm_undefined": {
            "pass": zero_undefined, "centered_re": m_const["centered"]["re"],
            "epsilon_bad_detected": epsilon_bad_detected,
            "epsilon_bad_re": wrong_eps_re,
        },
        "identity_decomposition": {
            "pass": identity_ok and centered_zero and bias_ok,
            "residual": m_shift["identity"]["residual"],
            "relative_residual": m_shift["identity"]["relative_residual"],
            "centered_re": m_shift["centered"]["re"],
            "bias": m_shift["bias_pred_minus_ref"],
        },
        "equal_weight_mean": {
            "pass": arithmetic_ok and area_differs,
            "mean": m_mean["mean_ref"], "area_weighted_mean": float(area_weighted),
        },
        "bad_implementation_detectable": {
            "pass": one_sided_detected,
            "correct_centered_re": correct["centered"]["re"],
            "wrong_one_sided_centered_re": wrong_one_sided,
        },
    }
    checks["all_pass"] = all(v["pass"] for v in checks.values())
    return checks


def verify_artifact_binding(input_ws, spec):
    """不构建模型也核对真实 checkpoint/metrics/hash/val_best/协议身份。"""
    ckpt, met = PG.model_paths(input_ws, spec, 0)
    for p in (ckpt, met):
        if not os.path.isfile(p):
            raise SystemExit(f"[probe] missing artifact: {p}")
    m = json.load(open(met, encoding="utf-8"))
    proto = m.get("protocol") or {}
    checks = {
        "model": proto.get("model") == spec["expected_class"],
        "model_label": proto.get("model_label") == spec["expected_label"],
        "model_params": int(proto.get("model_params", -1)) == spec["expected_params"],
        "strict_fp32": (proto.get("cuda_matmul_allow_tf32") is False
                        and proto.get("cudnn_allow_tf32") is False),
        "epochs": int(proto.get("epochs", -1)) == 500,
        "pool_context": proto.get("pool_context") == spec["pool_context"],
        "seed": int(m.get("seed", -1)) == 0,
        "test_20": (int((m.get("test") or {}).get("n_geom", -1)) == 20
                    and len((m.get("test") or {}).get("rows") or []) == 20),
    }
    vb = m.get("val_best") or {}
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    checks["checkpoint_seed"] = int(ck.get("seed", -1)) == 0
    checks["checkpoint_epoch_binds_val_best"] = int(ck.get("epoch", -1)) == int(vb.get("epoch", -2))
    checks["checkpoint_score_binds_val_best"] = abs(
        float(ck.get("val_score")) - float(vb.get("score"))) <= 1e-9
    checks["all_pass"] = all(v for k, v in checks.items() if k != "all_pass")
    return {
        "pass": checks["all_pass"],
        "checks": checks,
        "checkpoint": ckpt,
        "checkpoint_sha256": PG.sha256_file(ckpt),
        "metrics": met,
        "metrics_sha256": PG.sha256_file(met),
        "val_best": vb,
    }


def real_checks(input_ws, data, max_geoms=1):
    from src.core import thermal_io
    split_path, test_ids = PG.load_split(input_ws)
    gid = test_ids[0]
    X, Y = thermal_io.load_thermal(data)
    out = {"split": {"path": split_path, "sha256": PG.sha256_file(split_path),
                     "test_ids": test_ids},
           "data": {"path": data, "sha256": PG.sha256_file(data)},
           "models": {}}
    for spec in PG.MODEL_SPECS:
        entry = {"name": spec["name"], "status": "ok", "geoms": []}
        entry["artifact_binding"] = verify_artifact_binding(input_ws, spec)
        try:
            model, metrics, stats, ckpt, met = PG.load_model_and_metrics(
                input_ws, spec, 0, torch.device("cpu"))
        except (ImportError, ModuleNotFoundError) as exc:
            # 仅对本地缺失 graph 可选依赖明确跳过；FEN/协议/绑定错误必须失败。
            entry["status"] = "skipped"
            entry["reason"] = f"{type(exc).__name__}: {exc}"
            out["models"][spec["name"]] = entry
            continue
        except RuntimeError as exc:
            msg = str(exc)
            if "torch_geometric" in msg or "not importable" in msg:
                entry["status"] = "skipped"
                entry["reason"] = f"missing optional dependency: {type(exc).__name__}: {msg}"
                out["models"][spec["name"]] = entry
                continue
            raise
        rows = {int(r["geom_id"]): r for r in metrics["test"]["rows"]}
        for _ in range(max_geoms):
            g = int(test_ids[0]) if _ == 0 else int(test_ids[_])
            pred = PG.predict_geometry(model, X[g], stats, torch.device("cpu"))
            ref = Y[g].astype(np.float64)
            fm = PG._field_metrics(pred, ref)
            row = rows[g]
            max_err = 0.0
            for field in PG.FIELDS:
                for stat in ("re", "mae", "max"):
                    got = fm[field][stat]
                    want = row.get(f"{stat}_{field}")
                    if got is None or want is None:
                        entry["status"] = "failed"
                        entry["reason"] = f"missing {field}.{stat}"
                        out["models"][spec["name"]] = entry
                        return out
                    max_err = max(max_err, abs(float(got) - float(want)))
            pm = PG.p_metrics(pred[:, PG.P_INDEX], ref[:, PG.P_INDEX])
            entry["geoms"].append({
                "geom_id": g,
                "checkpoint": ckpt,
                "checkpoint_sha256": PG.sha256_file(ckpt),
                "metrics": met,
                "metrics_sha256": PG.sha256_file(met),
                "original_reproduction_max_abs_error": max_err,
                "p": pm,
            })
        del model
        out["models"][spec["name"]] = entry
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-workspace", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out-dir", required=True,
                    help="probe output dir (e.g. results/round2_pressure_gauge/probe)")
    args = ap.parse_args()
    inp = os.path.abspath(args.input_workspace)
    data = os.path.abspath(args.data)
    out = os.path.abspath(args.out_dir)
    os.makedirs(out, exist_ok=True)
    report_path = os.path.join(out, "probe_report.json")
    md_path = os.path.join(out, "probe_report.md")
    for p in (report_path, md_path):
        if os.path.exists(p):
            raise SystemExit(f"[refuse] probe target exists: {p}")
    analytic = analytic_checks()
    real = real_checks(inp, data, max_geoms=1)
    report = {
        "task": "pressure_gauge_probe",
        "protocol": "R2-THERMAL-REFERENCE-02 A",
        "created_local": PG._now_local(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "workspace": inp,
        "data": data,
        "analytic": analytic,
        "real_smoke": real,
        "notes": [
            "CPU only; no training; no batch GPU inference",
            "graph skip is explicit if optional dependency unavailable locally",
            "probe does not write protocol/raw/summary main outputs",
        ],
    }
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    lines = [
        "# R2-PRESSURE-GAUGE-02 probe",
        "",
        f"- created: {report['created_local']}",
        f"- analytic all_pass: {analytic['all_pass']}",
        "",
        "## analytic",
        "",
    ]
    for k, v in analytic.items():
        if k == "all_pass":
            continue
        lines.append(f"- {k}: pass={v.get('pass')}")
    lines += ["", "## real smoke (CPU)", ""]
    for name, entry in real["models"].items():
        lines.append(f"- {name}: status={entry['status']}; "
                     f"artifact_binding_pass={entry['artifact_binding']['pass']}")
        if entry.get("reason"):
            lines.append(f"  - reason: {entry['reason']}")
        for g in entry.get("geoms", []):
            lines.append(f"  - geom {g['geom_id']} original reproduction max abs err="
                         f"{g['original_reproduction_max_abs_error']:.3e}; "
                         f"p original RE={g['p']['original']['re']}; "
                         f"centered RE={g['p']['centered']['re']}; "
                         f"bias={g['p']['bias_pred_minus_ref']}")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[probe] {report_path}")
    print(f"[probe] analytic all_pass={analytic['all_pass']}")
    if not analytic["all_pass"]:
        raise SystemExit(1)
    for entry in real["models"].values():
        if entry["status"] == "failed":
            raise SystemExit(1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
