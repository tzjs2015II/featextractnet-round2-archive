"""exp_r2_pressure_gauge.py —— 压力常数偏置敏感性（R2-PRESSURE-GAUGE-02 / R2-THERMAL-REFERENCE-02 A）。

冻结口径（docs/热场压力规范与独立FEM补证执行计划.md §A）
--------------------------------------------------------
* 模型：当前已验收主 FEN（feat_extract，pool_mode=symmetric）seeds0–4；
  fresh strict-FP32 graph（meshgraphnet legacy）seeds0–4；均在同一 20 个
  Protocol-A test 几何上评估。无训练、无调参、无选路。
* 对每个 seed×geometry 保存 p 的：
  - 原始：RE=||p̂-p||/||p||、MAE、max，分母 ||p||；
  - 逐场独立去均值：Cp=p-mean(p)，Cp̂=p̂-mean(p̂)，
    RE_c=||Cp̂-Cp||/||Cp||、MAE_c、max_c，分母 ||Cp||；
  - mean(p)、mean(p̂)、bias=mean(p̂)-mean(p)、N；
  - 零中心化参考 norm 严格标 undefined（None），不加 epsilon；
  - 独立恒等式 ||p̂-p||² = ||Cp̂-Cp||² + N*bias²，保存残差；
  - N 点等权算术均值，明确不是有限元面积积分零均值。
* 原 RE 与中心化 RE 的分母不同，禁止把两者直接相减称为"提升"；报告并列。
* 严格绑定：checkpoint SHA、metrics SHA、data/split SHA、metrics 中 train-only
  statistics、checkpoint seed/epoch/val_score 与 val_best 绑定、模型类/参数量、
  strictFP32 flags；原 full-5000 评分先复算并与 metrics JSON 逐通道一致。
* 统计：逐 geometry 后逐 seed mean/sample std；FEN vs graph 的 20-geometry
  paired bootstrap（20000 次、analysis seed 20261021），结果条件于固定训练 seeds；
  这是查看 test 后的补充敏感性分析，不包装为独立确认。
* 其他基线只有在当前同协议权重/入口可核时才纳入；本脚本仅纳入 FEN 与 fresh graph，
  不拼接历史 3seed。

输出（默认 canonical results/round2_pressure_gauge/，任何计划文件已存在即拒绝）
--------------------------------------------------------------------------------
  protocol_snapshot.json   计算前协议/输入/哈希/口径
  raw_results.json         逐 model×seed×geometry 原始/中心化指标、bias/分母/恒等式
  summary.json             逐 geometry seed 统计、跨 geometry 汇总、paired bootstrap
  report.md                表格与限制

CLI
---
  python scripts/exp_r2_pressure_gauge.py \
      --input-workspace <canonical WS> \
      --data <HeatTransferData.npy> \
      --out-dir <WS>/results/round2_pressure_gauge --device cuda

约束：只做冻结模型推理；不训练、不安装、不写共享 registry；CUDA 不可用时拒绝。
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WS not in sys.path:
    sys.path.insert(0, WS)

FIELDS = ("u", "v", "p", "T")
P_INDEX = 2
N_POINTS = 5000
SEEDS = (0, 1, 2, 3, 4)
BOOTSTRAP_REPS = 20000
BOOTSTRAP_SEED = 20261021
BOOTSTRAP_CI = (2.5, 97.5)
RE_TOL = 1e-6
MODEL_SPECS = (
    {
        "name": "fen",
        "token": "feat_extract",
        "pool_context": "symmetric",
        "root": os.path.join("results", "round2_thermal_symmetric", "feat_extract"),
        "expected_params": 1149512,
        "expected_class": "FeatExtractNet",
        "expected_label": "FeatExtractNet(scale=1.0, pool_mode=symmetric)",
    },
    {
        "name": "graph",
        "token": "meshgraphnet",
        "pool_context": "legacy_index_bins",
        "root": os.path.join("results", "round2_graph_five_seed_main", "meshgraphnet"),
        "expected_params": 976516,
        "expected_class": "MeshGraphNetStyleBaseline",
        "expected_label": ("MeshGraphNet-style encode-process-decode GNN "
                           "(assembled, hidden=128, layers=8, k=16) 976,516 params"),
    },
)
OUT_FILES = ("protocol_snapshot.json", "raw_results.json", "summary.json", "report.md")
POSTREVIEW_FILES = ("protocol_snapshot_postreview.json",
                    "summary_postreview.json", "report_postreview.md",
                    "delivery_postreview.json", "baseline_inventory.json")
BASELINE_TOKENS = ("unif_therm", "mlp", "pointnet", "pointnetpp", "dgcnn",
                   "pipn", "siren", "pointtransformer", "meshgraphnet",
                   "feat_extract")
BASELINE_ROOTS = (
    os.path.join("results", "baselines_ext", "thermal_protocolA"),
    os.path.join("results", "thermal_protocolA"),
    os.path.join("results", "thermal"),
    os.path.join("results", "round2_thermal_symmetric"),
    os.path.join("results", "round2_graph_five_seed_main"),
)


def _require(cond, msg):
    if not cond:
        raise SystemExit("[fail] " + msg)


def _now_local():
    from datetime import datetime
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path, doc):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


# ---------------------------------------------------------------------------
# 指标：干净实现 + 严格 undefined/恒等式
# ---------------------------------------------------------------------------
def p_metrics(pred_p, ref_p, shift=0.0):
    """单场 p 指标。pred/ref 等长一维 float；shift 用于平移不变性 probe。"""
    pred = np.asarray(pred_p, dtype=np.float64).reshape(-1) + float(shift)
    ref = np.asarray(ref_p, dtype=np.float64).reshape(-1) + float(shift)
    _require(pred.shape == ref.shape and pred.size > 0, "p arrays shape/empty")
    n = int(pred.size)
    mean_pred = float(pred.mean())
    mean_ref = float(ref.mean())
    bias = mean_pred - mean_ref
    diff = pred - ref
    err_sq = float(np.dot(diff, diff))
    norm_ref = float(np.linalg.norm(ref))
    orig = {
        "re": (float(np.linalg.norm(diff) / norm_ref) if norm_ref > 0.0 else None),
        "mae": float(np.mean(np.abs(diff))),
        "max": float(np.max(np.abs(diff))),
        "denominator": norm_ref,
    }
    cp = ref - mean_ref
    cpred = pred - mean_pred
    cdiff = cpred - cp
    cnorm = float(np.linalg.norm(cp))
    centered = {
        "re": (float(np.linalg.norm(cdiff) / cnorm) if cnorm > 0.0 else None),
        "mae": float(np.mean(np.abs(cdiff))),
        "max": float(np.max(np.abs(cdiff))),
        "denominator": cnorm,
    }
    identity = {
        "err_sq": err_sq,
        "centered_err_sq": float(np.dot(cdiff, cdiff)),
        "n_times_bias_sq": float(n * bias * bias),
        "residual": float(err_sq - np.dot(cdiff, cdiff) - n * bias * bias),
        "relative_residual": (float(abs(err_sq - np.dot(cdiff, cdiff) - n * bias * bias)
                                  / err_sq) if err_sq > 0.0 else 0.0),
    }
    _require(bool(np.isfinite(pred).all() and np.isfinite(ref).all()),
             "non-finite p prediction/reference")
    _require(all(np.isfinite(v) for v in (
        mean_pred, mean_ref, bias, orig["mae"], orig["max"],
        centered["mae"], centered["max"], identity["err_sq"],
        identity["centered_err_sq"], identity["n_times_bias_sq"],
        identity["residual"], identity["relative_residual"])),
        "non-finite pressure metric value")
    return {
        "n": n,
        "mean_ref": mean_ref,
        "mean_pred": mean_pred,
        "bias_pred_minus_ref": bias,
        "original": orig,
        "centered": centered,
        "identity": identity,
        "mean_rule": "arithmetic_equal_node_weight (not FEM area integral)",
    }


def _field_metrics(pred, ref):
    out = {}
    for j, field in enumerate(FIELDS):
        diff = np.asarray(pred[:, j], np.float64) - np.asarray(ref[:, j], np.float64)
        denom = float(np.linalg.norm(np.asarray(ref[:, j], np.float64)))
        out[field] = {
            "re": float(np.linalg.norm(diff) / denom) if denom > 0.0 else None,
            "mae": float(np.mean(np.abs(diff))),
            "max": float(np.max(np.abs(diff))),
        }
    return out


# ---------------------------------------------------------------------------
# 冻结输入/模型载入
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-workspace", required=True,
                    help="canonical read-only workspace")
    ap.add_argument("--data", required=True, help="explicit HeatTransferData.npy")
    ap.add_argument("--out-dir", required=True,
                    help="<input-workspace>/results/round2_pressure_gauge")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--postprocess-only", action="store_true",
                    help="use existing raw_results.json; write postreview summary/report only")
    ap.add_argument("--export-array-probe", action="store_true",
                    help="export FEN/graph seed0 geom17 actual pred/ref arrays (SLURM)")
    return ap.parse_args()


def resolve_paths(args):
    INPUT = os.path.abspath(args.input_workspace)
    DATA = os.path.abspath(args.data)
    OUT = os.path.abspath(args.out_dir)
    _require(os.path.isdir(INPUT), f"input workspace missing: {INPUT}")
    _require(os.path.isfile(DATA), f"data missing: {DATA}")
    _require(os.path.normcase(os.path.realpath(OUT)).startswith(
                 os.path.normcase(os.path.realpath(INPUT)) + os.sep),
             f"out-dir must be inside input workspace: {OUT}")
    _require(os.path.basename(os.path.normpath(OUT)) == "round2_pressure_gauge",
             "out-dir basename must be round2_pressure_gauge")
    splits = os.path.join(INPUT, "configs", "splits_thermal.json")
    _require(os.path.isfile(splits), f"splits missing: {splits}")
    return INPUT, DATA, OUT, splits


def load_split(input_ws):
    path = os.path.join(input_ws, "configs", "splits_thermal.json")
    doc = json.load(open(path, encoding="utf-8"))
    pa = doc.get("protocol_A") or {}
    test_ids = [int(g) for g in pa.get("test", [])]
    _require(doc.get("split_seed") == 123, "split_seed != 123")
    _require(int(doc.get("n_geometries", -1)) == 135, "n_geometries != 135")
    _require(len(pa.get("train", [])) == 100 and len(pa.get("val", [])) == 15
             and len(test_ids) == 20, "protocol_A counts != 100/15/20")
    return path, test_ids


def model_paths(input_ws, spec, seed):
    base = os.path.join(input_ws, spec["root"], f"seed{seed}")
    return (os.path.join(base, "checkpoint_best.pth"),
            os.path.join(base, f"metrics_seed{seed}.json"))


def scan_baseline_inventory(input_ws):
    """实际检查主基线候选根/seed 存在性；不凭先验写“缺少权重”。"""
    inv = {"checked_at": _now_local(), "input_workspace": os.path.abspath(input_ws),
           "roots_checked": [os.path.join(input_ws, r) for r in BASELINE_ROOTS],
           "tokens": {}, "included": ["fen", "graph"],
           "rule": ("complete_five_seed requires seed0..4 current metrics+checkpoint; "
                    "strict_fp32 requires every found current metrics flag false; "
                    "not-found means not found under these checked roots, not a global absence claim")}
    for token in BASELINE_TOKENS:
        entry = {"roots": {}, "complete_five_seed": False,
                 "strict_fp32_all_found": False,
                 "included": token in ("feat_extract", "meshgraphnet")}
        for root_rel in BASELINE_ROOTS:
            root = os.path.join(input_ws, root_rel, token)
            if not os.path.isdir(root):
                continue
            seeds = {}
            for name in sorted(os.listdir(root)):
                if not name.startswith("seed"):
                    continue
                try:
                    seed = int(name[4:])
                except ValueError:
                    continue
                sd = os.path.join(root, name)
                mp = os.path.join(sd, f"metrics_seed{seed}.json")
                cp = os.path.join(sd, "checkpoint_best.pth")
                rec = {"metrics": os.path.isfile(mp), "checkpoint": os.path.isfile(cp)}
                if os.path.isfile(mp):
                    try:
                        m = json.load(open(mp, encoding="utf-8"))
                        p = m.get("protocol") or {}
                        rec.update({"model": p.get("model"),
                                    "model_label": p.get("model_label"),
                                    "model_params": p.get("model_params"),
                                    "epochs": p.get("epochs"),
                                    "pool_context": p.get("pool_context"),
                                    "split_version": p.get("split_version"),
                                    "cuda_matmul_allow_tf32": p.get("cuda_matmul_allow_tf32"),
                                    "cudnn_allow_tf32": p.get("cudnn_allow_tf32")})
                    except Exception as exc:  # inventory records unreadable, does not hide
                        rec["read_error"] = f"{type(exc).__name__}: {exc}"
                seeds[str(seed)] = rec
            if seeds:
                entry["roots"][os.path.join(root_rel, token)] = seeds
        seed_keys = set()
        for seeds in entry["roots"].values():
            seed_keys |= {int(k) for k in seeds}
        entry["seed_ids_found"] = sorted(seed_keys)
        complete = set(range(5)).issubset(seed_keys) and all(
            all(seeds.get(str(s), {}).get("metrics") and seeds.get(str(s), {}).get("checkpoint")
                for s in range(5))
            for seeds in entry["roots"].values() if len(seeds) >= 5)
        entry["complete_five_seed"] = bool(complete)
        found_flags = []
        for seeds in entry["roots"].values():
            for s in range(5):
                r = seeds.get(str(s)) or {}
                if r.get("metrics"):
                    found_flags.append(r.get("cuda_matmul_allow_tf32") is False
                                       and r.get("cudnn_allow_tf32") is False)
        entry["strict_fp32_all_found"] = bool(found_flags and all(found_flags))
        inv["tokens"][token] = entry
    return inv


def load_model_and_metrics(input_ws, spec, seed, device):
    from scripts import exp_thermal as ET
    ckpt_path, metrics_path = model_paths(input_ws, spec, seed)
    for p in (ckpt_path, metrics_path):
        _require(os.path.isfile(p), f"missing {spec['name']} seed{seed}: {p}")
    metrics = json.load(open(metrics_path, encoding="utf-8"))
    proto = metrics.get("protocol") or {}
    _require(int(metrics.get("seed", -1)) == seed, f"{spec['name']} seed{seed} metrics seed")
    _require(int(proto.get("model_params", -1)) == spec["expected_params"],
             f"{spec['name']} seed{seed} params != {spec['expected_params']}")
    _require(proto.get("model") == spec["expected_class"],
             f"{spec['name']} seed{seed} model class {proto.get('model')!r}")
    _require(proto.get("model_label") == spec["expected_label"],
             f"{spec['name']} seed{seed} model_label mismatch")
    _require(proto.get("cuda_matmul_allow_tf32") is False
             and proto.get("cudnn_allow_tf32") is False,
             f"{spec['name']} seed{seed} strictFP32 flags not false")
    _require(int(proto.get("epochs", -1)) == 500, f"{spec['name']} seed{seed} epochs != 500")
    _require(proto.get("split_version") == "splits_thermal_v1_protocolA",
             f"{spec['name']} seed{seed} split_version mismatch")
    _require(int(proto.get("split_seed", -1)) == 123,
             f"{spec['name']} seed{seed} split_seed != 123")
    _require((int(proto.get("n_train_geom", -1)), int(proto.get("n_val_geom", -1)),
              int(proto.get("n_test_geom", -1))) == (100, 15, 20),
             f"{spec['name']} seed{seed} split counts != 100/15/20")
    _require(proto.get("pool_context") == spec["pool_context"],
             f"{spec['name']} seed{seed} pool_context != {spec['pool_context']}")
    stats = {k: np.asarray(proto[k], np.float32)
             for k in ("x_mean", "x_std", "y_mean", "y_std")}
    model, label = ET.build_model(spec["token"], pool_context=spec["pool_context"])
    params = int(sum(p.numel() for p in model.parameters()))
    _require(params == spec["expected_params"], f"built params {params} != {spec['expected_params']}")
    _require(label == spec["expected_label"], f"built label {label!r} != expected")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    _require(isinstance(ckpt, dict) and "model_state_dict" in ckpt,
             f"checkpoint missing model_state_dict: {ckpt_path}")
    _require(int(ckpt.get("seed", -1)) == seed, f"{spec['name']} seed{seed} ckpt seed")
    vb = metrics.get("val_best") or {}
    _require(int(ckpt.get("epoch", -1)) == int(vb.get("epoch", -2)),
             f"{spec['name']} seed{seed} ckpt epoch != val_best")
    _require(abs(float(ckpt.get("val_score")) - float(vb.get("score"))) <= 1e-9,
             f"{spec['name']} seed{seed} ckpt val_score != val_best")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, metrics, stats, ckpt_path, metrics_path


def predict_geometry(model, x_phys, stats, device):
    from src.core import thermal_io
    xn = thermal_io.normalize(x_phys[None, ...], stats["x_mean"], stats["x_std"])
    xt = torch.tensor(xn, dtype=torch.float32, device=device).permute(0, 2, 1)
    with torch.no_grad():
        pn = model(xt)
    pn = pn.detach().to("cpu").numpy()
    pred = np.transpose(pn, (0, 2, 1))[0]
    return thermal_io.denormalize(pred, stats["y_mean"], stats["y_std"]).astype(np.float64)


# ---------------------------------------------------------------------------
# 主计算
# ---------------------------------------------------------------------------
def run(args):
    torch.set_num_threads(4)
    INPUT, DATA, OUT, splits_path = resolve_paths(args)
    if os.path.isdir(OUT):
        _require(not any(os.path.exists(os.path.join(OUT, f)) for f in OUT_FILES),
                 f"planned output exists in {OUT}; refusing overwrite")
    for f in OUT_FILES:
        _require(not os.path.exists(os.path.join(OUT, f)), f"target exists: {OUT}/{f}")
    _require(torch.cuda.is_available() if args.device == "cuda" else True,
             "device=cuda requested but CUDA unavailable")
    device = torch.device(args.device if args.device == "cuda" else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    from src.core import thermal_io
    X, Y = thermal_io.load_thermal(DATA)
    _require(X.shape == (135, N_POINTS, 2) and Y.shape == (135, N_POINTS, 4),
             f"unexpected data shapes X={X.shape} Y={Y.shape}")
    splits_path, test_ids = load_split(INPUT)

    input_hashes = {
        "data": {"path": DATA, "sha256": sha256_file(DATA)},
        "splits": {"path": splits_path, "sha256": sha256_file(splits_path)},
    }
    model_meta = {}
    for spec in MODEL_SPECS:
        model_meta[spec["name"]] = {"seeds": {},
                                    "root": os.path.join(INPUT, spec["root"])}
        for seed in SEEDS:
            ckpt, met = model_paths(INPUT, spec, seed)
            _require(os.path.isfile(ckpt) and os.path.isfile(met),
                     f"missing {spec['name']} seed{seed} artifacts")
            model_meta[spec["name"]]["seeds"][str(seed)] = {
                "checkpoint": os.path.relpath(ckpt, INPUT),
                "checkpoint_sha256": sha256_file(ckpt),
                "metrics": os.path.relpath(met, INPUT),
                "metrics_sha256": sha256_file(met),
            }
    source_hashes = {}
    for rel in ("scripts/exp_r2_pressure_gauge.py", "scripts/exp_thermal.py",
                "src/core/thermal_io.py",
                "src/models/featextractnet.py",
                "src/models/baselines/meshgraphnet_style.py"):
        p = os.path.join(WS, rel)
        if os.path.isfile(p):
            source_hashes[rel] = sha256_file(p)

    protocol = {
        "task": "pressure_constant_bias_sensitivity",
        "protocol": "R2-THERMAL-REFERENCE-02 A",
        "protocol_version": "v1",
        "r2_task": "R2-PRESSURE-GAUGE-02",
        "created_local": _now_local(),
        "input_workspace": INPUT,
        "out_dir": OUT,
        "data": DATA,
        "splits_config": splits_path,
        "test_ids": test_ids,
        "n_test_geom": len(test_ids),
        "models": [
            {"name": s["name"], "token": s["token"], "pool_context": s["pool_context"],
             "params": s["expected_params"], "class": s["expected_class"],
             "label": s["expected_label"]}
            for s in MODEL_SPECS
        ],
        "seeds": list(SEEDS),
        "input_sha256": input_hashes,
        "model_artifacts": model_meta,
        "source_sha256": source_hashes,
        "metric_definition": {
            "field": "p (physical units, stored scale)",
            "original": {"re": "||pred-ref||_2 / ||ref||_2", "mae": "mean|pred-ref|",
                         "max": "max|pred-ref|", "denominator": "||ref||_2"},
            "centered": {"re": "||Cpred-Cref||_2 / ||Cref||_2",
                         "mae": "mean|Cpred-Cref|", "max": "max|Cpred-Cref|",
                         "denominator": "||Cref||_2"},
            "Cp": "p - mean(p); Cpred = pred - mean(pred); independently per field/sample",
            "mean_rule": "N-point arithmetic equal-weight mean (NOT FEM area integral)",
            "zero_centered_norm": "undefined -> None; no epsilon substitution",
            "bias": "mean(pred) - mean(ref)",
            "identity": "||pred-ref||^2 = ||Cpred-Cref||^2 + N*bias^2",
            "no_improvement_delta": ("original and centered RE have different denominators; "
                                     "not report a direct difference as improvement"),
        },
        "statistics": {
            "per_geometry": "each geometry: 5-seed mean/sample std (per_geometry)",
            "per_seed": "each seed: mean/sample std over 20 geometries (per_seed)",
            "across": "mean/std over 20 geometry seed means",
            "paired_bootstrap": {"reps": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED,
                                 "ci": list(BOOTSTRAP_CI),
                                 "unit": "20 test geometries, paired FEN-graph by seed",
                                 "geometry_resample_shared_across_metrics": True,
                                 "conditional_on_fixed_training_seeds": True},
            "post_test_status": ("supplementary sensitivity analysis after seeing test; "
                                 "not independent confirmation"),
        },
        "included_baselines": ["fen", "graph"],
        "excluded_baselines": {
            "unif_therm": "no current same-protocol five-seed frozen weights verified",
            "mlp": "no current same-protocol five-seed frozen weights verified",
            "pointnet": "no current same-protocol five-seed frozen weights verified",
            "pointnetpp": "no current same-protocol five-seed frozen weights verified",
            "dgcnn": "no current same-protocol five-seed frozen weights verified",
            "pipn": "no current same-protocol five-seed frozen weights verified",
            "siren": "no current same-protocol five-seed frozen weights verified",
            "pointtransformer": "no current same-protocol five-seed frozen weights verified",
            "old_meshgraphnet_3seed": "historical 3seed not stitched into five-seed graph",
        },
        "excluded_baselines_note": ("no other current same-protocol five-seed weights were "
                                    "used; no historical 3seed stitching"),
        "expected_raw_records": len(MODEL_SPECS) * len(SEEDS) * len(test_ids),
        "training": False,
        "environment": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": str(device),
        },
        "inference": {"device": str(device), "strict_fp32": True,
                      "tf32_disabled": bool(device.type != "cuda" or
                                            (torch.backends.cuda.matmul.allow_tf32 is False
                                             and torch.backends.cudnn.allow_tf32 is False))},
        "registry_modified": False,
        "notes": ["no training/tuning; frozen weights only",
                  "graph is the fresh strict-FP32 five-seed run",
                  "original scores are reproduced first; failure stops"],
    }
    os.makedirs(OUT, exist_ok=True)
    _write_json(os.path.join(OUT, "protocol_snapshot.json"), protocol)
    print(f"[protocol] {OUT}/protocol_snapshot.json", flush=True)

    raw = []
    stats_by_model = {s["name"]: {} for s in MODEL_SPECS}
    for spec in MODEL_SPECS:
        for seed in SEEDS:
            model, metrics, stats, ckpt_path, metrics_path = load_model_and_metrics(
                INPUT, spec, seed, device)
            stats_by_model[spec["name"]][seed] = stats
            rows = {int(r["geom_id"]): r for r in metrics["test"]["rows"]}
            _require(set(rows) == set(test_ids),
                     f"{spec['name']} seed{seed} metrics rows != split test ids")
            for gid in test_ids:
                pred = predict_geometry(model, X[gid], stats, device)
                ref = Y[gid].astype(np.float64)
                fm = _field_metrics(pred, ref)
                # 原评分复算一致（逐通道 RE/MAE/max）
                max_rel = 0.0
                for field in FIELDS:
                    for stat in ("re", "mae", "max"):
                        got = fm[field][stat]
                        want = rows[gid].get(f"{stat}_{field}")
                        _require(got is not None and want is not None,
                                 f"{spec['name']} seed{seed} geom{gid} {field}.{stat} missing")
                        err = abs(float(got) - float(want))
                        tol = RE_TOL * max(1.0, abs(float(want)))
                        _require(err <= tol,
                                 f"{spec['name']} seed{seed} geom{gid} {field}.{stat} "
                                 f"reproduction error {err:.3e} > {tol:.3e}")
                        max_rel = max(max_rel, err)
                pm = p_metrics(pred[:, P_INDEX], ref[:, P_INDEX])
                raw.append({
                    "model": spec["name"], "seed": seed, "geom_id": int(gid),
                    "checkpoint": os.path.relpath(ckpt_path, INPUT),
                    "metrics": os.path.relpath(metrics_path, INPUT),
                    "original_reproduction_max_abs_error": max_rel,
                    "p": pm,
                })
            del model
            print(f"[{spec['name']} seed{seed}] 20 geometries done", flush=True)
        _cleanup()

    # 同一模型跨 seed 必须使用同一 train-only statistics
    for name, by_seed in stats_by_model.items():
        base = by_seed[0]
        for seed in SEEDS[1:]:
            for key in base:
                _require(np.array_equal(np.asarray(by_seed[seed][key]),
                                        np.asarray(base[key])),
                         f"{name} seed{seed} {key} differs from seed0")
    _require(len(raw) == len(MODEL_SPECS) * len(SEEDS) * len(test_ids),
             f"raw record count {len(raw)} != expected "
             f"{len(MODEL_SPECS) * len(SEEDS) * len(test_ids)}")

    _write_json(os.path.join(OUT, "raw_results.json"), raw)
    summary = build_summary(raw, test_ids)
    _write_json(os.path.join(OUT, "summary.json"), summary)
    write_report(OUT, summary, protocol)
    print(f"[done] {OUT}", flush=True)


def _cleanup():
    import gc
    gc.collect()


# ---------------------------------------------------------------------------
# 汇总 / bootstrap
# ---------------------------------------------------------------------------
def _seed_geometry_stats(raw, test_ids):
    out = {}
    for name in ("fen", "graph"):
        out[name] = {}
        for seed in SEEDS:
            out[name][seed] = {}
            for gid in test_ids:
                rec = next(r for r in raw
                           if r["model"] == name and r["seed"] == seed
                           and r["geom_id"] == gid)
                out[name][seed][gid] = rec["p"]
    return out


def build_summary(raw, test_ids):
    data = _seed_geometry_stats(raw, test_ids)
    summary = {}
    for name in ("fen", "graph"):
        summary[name] = {}
        for kind in ("original", "centered"):
            summary[name][kind] = {}
            for stat in ("re", "mae", "max"):
                per_geom = {}
                geom_means = []
                for gid in test_ids:
                    vals = []
                    for seed in SEEDS:
                        v = data[name][seed][gid][kind][stat]
                        vals.append(v)
                    valid = [v for v in vals if v is not None]
                    mean = float(np.mean(valid)) if valid else None
                    std = float(np.std(valid, ddof=1)) if len(valid) > 1 else None
                    per_geom[str(gid)] = {"seed_values": vals, "mean": mean,
                                          "std_sample": std,
                                          "n_valid_seeds": len(valid)}
                    if mean is not None:
                        geom_means.append(mean)
                summary[name][kind][stat] = {
                    "per_geometry": per_geom,
                    "per_seed": {
                        str(seed): {
                            "geometry_values": [
                                data[name][seed][gid][kind][stat] for gid in test_ids
                            ],
                            "mean": (float(np.mean([v for v in [
                                data[name][seed][gid][kind][stat] for gid in test_ids
                            ] if v is not None]))
                            if any(v is not None for v in [
                                data[name][seed][gid][kind][stat] for gid in test_ids
                            ]) else None),
                            "std_sample": (float(np.std([v for v in [
                                data[name][seed][gid][kind][stat] for gid in test_ids
                            ] if v is not None], ddof=1))
                            if len([v for v in [
                                data[name][seed][gid][kind][stat] for gid in test_ids
                            ] if v is not None]) > 1 else None),
                        }
                        for seed in SEEDS
                    },
                    "mean_over_geometries": (float(np.mean(geom_means))
                                             if geom_means else None),
                    "std_over_geometry_means": (float(np.std(geom_means, ddof=1))
                                                if len(geom_means) > 1 else None),
                    "n_geoms": len(geom_means),
                }
                seed_means = [summary[name][kind][stat]["per_seed"][str(s)]["mean"]
                              for s in SEEDS]
                valid_seed_means = [v for v in seed_means if v is not None]
                summary[name][kind][stat]["cross_seed"] = {
                    "seed_means": seed_means,
                    "mean": (float(np.mean(valid_seed_means))
                             if valid_seed_means else None),
                    "std_sample": (float(np.std(valid_seed_means, ddof=1))
                                   if len(valid_seed_means) > 1 else None),
                    "n_seeds": len(valid_seed_means),
                    "definition": ("primary: mean/sample std of the 5 seed means; "
                                   "each seed mean is over 20 geometries"),
                }
                summary[name][kind][stat]["geometry_std_aux"] = (
                    summary[name][kind][stat]["std_over_geometry_means"])
    # paired FEN-graph bootstrap per seed/metric；每个 seed 共用一组 geometry resample
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = {}
    for seed in SEEDS:
        idx = rng.integers(0, len(test_ids), size=(BOOTSTRAP_REPS, len(test_ids)))
        boot[str(seed)] = {}
        for kind in ("original", "centered"):
            for stat in ("re", "mae", "max"):
                a_vals = [data["fen"][seed][gid][kind][stat] for gid in test_ids]
                b_vals = [data["graph"][seed][gid][kind][stat] for gid in test_ids]
                if any(v is None for v in a_vals + b_vals):
                    boot[str(seed)][f"{kind}:{stat}"] = {
                        "n_pairs": len(test_ids), "mean_diff": None,
                        "ci95": None, "undefined": True,
                        "reason": "undefined centered/original value in paired geometry",
                        "reps": BOOTSTRAP_REPS, "analysis_seed": BOOTSTRAP_SEED,
                        "conditional_on_fixed_training_seeds": True,
                    }
                    continue
                arr = np.asarray(a_vals, dtype=np.float64) - np.asarray(
                    b_vals, dtype=np.float64)
                means = arr[idx].mean(axis=1)
                boot[str(seed)][f"{kind}:{stat}"] = {
                    "n_pairs": int(arr.size),
                    "mean_diff_fen_minus_graph": float(arr.mean()),
                    "ci95": [float(np.percentile(means, BOOTSTRAP_CI[0])),
                             float(np.percentile(means, BOOTSTRAP_CI[1]))],
                    "reps": BOOTSTRAP_REPS,
                    "analysis_seed": BOOTSTRAP_SEED,
                    "geometry_resample_shared_across_metrics": True,
                    "conditional_on_fixed_training_seeds": True,
                }
    summary["paired_bootstrap"] = boot
    # primary：每 geometry 先平均 5 seed，再 FEN-graph paired bootstrap；共享抽样索引
    rng_p = np.random.default_rng(BOOTSTRAP_SEED)
    idx_p = rng_p.integers(0, len(test_ids), size=(BOOTSTRAP_REPS, len(test_ids)))
    primary = {}
    for kind in ("original", "centered"):
        for stat in ("re", "mae", "max"):
            a_geom = []
            b_geom = []
            for gid in test_ids:
                av = [data["fen"][s][gid][kind][stat] for s in SEEDS]
                bv = [data["graph"][s][gid][kind][stat] for s in SEEDS]
                a_geom.append(None if any(v is None for v in av) else float(np.mean(av)))
                b_geom.append(None if any(v is None for v in bv) else float(np.mean(bv)))
            if any(v is None for v in a_geom + b_geom):
                primary[f"{kind}:{stat}"] = {
                    "n_geometries": len(test_ids), "mean_diff": None, "ci95": None,
                    "undefined": True, "reps": BOOTSTRAP_REPS,
                    "analysis_seed": BOOTSTRAP_SEED,
                    "geometry_resample_shared_across_metrics": True,
                    "conditional_on_fixed_training_seeds": True,
                    "method": "mean 5 seeds per geometry, then paired FEN-graph bootstrap",
                }
                continue
            arr = np.asarray(a_geom, dtype=np.float64) - np.asarray(
                b_geom, dtype=np.float64)
            means = arr[idx_p].mean(axis=1)
            primary[f"{kind}:{stat}"] = {
                "n_geometries": int(arr.size),
                "mean_diff_fen_minus_graph": float(arr.mean()),
                "ci95": [float(np.percentile(means, BOOTSTRAP_CI[0])),
                         float(np.percentile(means, BOOTSTRAP_CI[1]))],
                "reps": BOOTSTRAP_REPS,
                "analysis_seed": BOOTSTRAP_SEED,
                "geometry_resample_shared_across_metrics": True,
                "conditional_on_fixed_training_seeds": True,
                "method": "mean 5 seeds per geometry, then paired FEN-graph bootstrap",
                "not_100_independent_geometries": True,
            }
    summary["paired_bootstrap_primary"] = primary
    summary["primary_statistics_note"] = (
        "main mean±sample_std is across the five seed means (each seed mean over 20 "
        "geometries); geometry std is supplementary and is not the seed uncertainty. "
        "Primary paired interval: 5-seed mean per geometry then paired bootstrap; "
        "conditional on fixed training seeds, post-test sensitivity, not 100 independent geometries.")
    summary["note"] = ("original and centered RE have different denominators; "
                       "no direct delta is an improvement. Zero centered norm => undefined. "
                       "N-point arithmetic mean, not FEM area mean.")
    return summary


def write_report(out_dir, summary, protocol, filename="report.md", inventory=None):
    lines = [
        "# 压力常数偏置敏感性（R2-PRESSURE-GAUGE-02 / R2-THERMAL-REFERENCE-02 A）",
        "",
        "- 冻结权重推理，无训练/调参；FEN 与 fresh strict-FP32 graph 各 seeds0–4，"
        "同一 20 个 Protocol-A test 几何。",
        "- p 为存储尺度物理单位；mean 为 N=5000 点等权算术均值，**不是**有限元面积积分零均值。",
        "- 中心化逐场独立：Cp=p-mean(p)，Cp̂=p̂-mean(p̂)；零中心化参考 norm 严格 undefined。",
        "- **原 RE 与中心化 RE 分母不同，不把差值称为提升**；本表仅并列。",
        "- 已看 test 后的补充敏感性分析，条件于固定训练 seeds，不是独立确认。",
        "- **主统计**：每 geometry 先算 5seed 平均；跨 5 个 seed mean 报 mean ± sample std。"
        "geometry std 是 20 geometry 之间的变异，另列，不冒充 seed uncertainty。",
        "- summary.json 同时保留 per_geometry / per_seed / cross_seed。",
        "",
        "## 原评分复算",
        "",
        f"所有 model×seed×geometry 的 u/v/p/T RE/MAE/max 与 metrics JSON 复算一致"
        f"（容差 {RE_TOL}，raw_results 记录每场最大误差）。",
        "",
    ]
    for kind, label in (("original", "原始 p"), ("centered", "中心化 p（逐场独立去均值）")):
        lines += [f"## {label} — 主统计（跨 5 seed mean ± sample std）", "",
                  "| model | metric | primary mean ± sample_std | geometry std (auxiliary) |",
                  "|---|---|---|---|"]
        for name in ("fen", "graph"):
            for stat in ("re", "mae", "max"):
                cs = summary[name][kind][stat]["cross_seed"]
                gs = summary[name][kind][stat]["geometry_std_aux"]
                lines.append(f"| {name} | {stat} | {_fmt(cs['mean'])} ± "
                             f"{_fmt(cs['std_sample'])} | {_fmt(gs)} |")
    lines += ["", "## 逐 seed 20-geometry mean ± sample std（补充）", "",
              "| model | kind | metric | " +
              " | ".join(f"seed{s}" for s in SEEDS) + " |",
              "|---|---|---|" + "---|" * len(SEEDS)]
    for name in ("fen", "graph"):
        for kind in ("original", "centered"):
            for stat in ("re", "mae", "max"):
                cells = []
                for seed in SEEDS:
                    ps = summary[name][kind][stat]["per_seed"][str(seed)]
                    cells.append(f"{_fmt(ps['mean'])} ± {_fmt(ps['std_sample'])}")
                lines.append(f"| {name} | {kind} | {stat} | " +
                             " | ".join(cells) + " |")
    lines += ["", "## Primary FEN − graph paired bootstrap（每 geometry 先平均 5 seed）", "",
              "| metric | mean diff FEN−graph | 95% CI |", "|---|---|---|"]
    for kind in ("original", "centered"):
        for stat in ("re", "mae", "max"):
            b = summary["paired_bootstrap_primary"].get(f"{kind}:{stat}") or {}
            ci = b.get("ci95")
            lines.append(f"| {kind}:{stat} | "
                         f"{_fmt(b.get('mean_diff_fen_minus_graph'))} | "
                         f"{_fmt(ci[0]) if ci else 'undefined'} .. "
                         f"{_fmt(ci[1]) if ci else 'undefined'} |")
    lines += ["", "## 每 seed paired bootstrap（辅助，非主区间）", ""]
    for seed in SEEDS:
        lines += [f"### seed {seed}", "| metric | mean diff FEN−graph | 95% CI |",
                  "|---|---|---|"]
        for kind in ("original", "centered"):
            for stat in ("re", "mae", "max"):
                b = summary["paired_bootstrap"][str(seed)].get(f"{kind}:{stat}") or {}
                ci = b.get("ci95")
                lines.append(f"| {kind}:{stat} | "
                             f"{_fmt(b.get('mean_diff_fen_minus_graph'))} | "
                             f"{_fmt(ci[0]) if ci else 'undefined'} .. "
                             f"{_fmt(ci[1]) if ci else 'undefined'} |")
    if inventory:
        lines += ["", "## Baseline inventory（实际检查范围/存在性）", "",
                  "| token | found roots | seed ids | complete five | strict FP32 | included |",
                  "|---|---|---|---|---|---|"]
        for token in BASELINE_TOKENS:
            e = inventory["tokens"].get(token) or {}
            roots = list((e.get("roots") or {}).keys())
            lines.append(f"| {token} | {len(roots)}: {'; '.join(roots) if roots else 'none under checked roots'} "
                         f"| {e.get('seed_ids_found', [])} | {e.get('complete_five_seed')} "
                         f"| {e.get('strict_fp32_all_found')} | {e.get('included')} |")
        lines += ["", f"checked roots: {inventory.get('roots_checked')}", "",
                  "未找到仅指上述实际检查根中未发现；不是全局不存在断言。"]
    lines += [
        "",
        "## 限制",
        "",
        "- 中心化参考 norm=0 时 RE 为 undefined（不填 epsilon）；bias/denominator 均保存在 raw_results。",
        "- 原 RE 与中心化 RE 不能直接相减解释；中心化是预定逐场评价算子，不作超参/epoch/test 选路。",
        "- primary 区间条件于固定训练 seeds、5seed per geometry 平均；不是 100 个独立几何。",
        "- 基线 inventory 只记录实际检查根；未发现不等于全局不存在；未为覆盖全部额外训练。",
        "- 这不是独立数值确认，也不改变原数据/权重/split/既有 summary。",
        "",
    ]
    with open(os.path.join(out_dir, filename), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def _fmt(v):
    return "undefined" if v is None else f"{v:.6g}"


def run_export_array_probe(args):
    """SLURM small job: actual pred/ref arrays for FEN/graph seed0 first test geom."""
    INPUT, DATA, OUT, splits_path = resolve_paths(args)
    _require(args.device == "cuda" and torch.cuda.is_available(),
             "array export requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    from src.core import thermal_io
    X, Y = thermal_io.load_thermal(DATA)
    _, test_ids = load_split(INPUT)
    gid = int(test_ids[0])
    os.makedirs(OUT, exist_ok=True)
    inv_path = os.path.join(OUT, "baseline_inventory.json")
    _require(not os.path.exists(inv_path), f"inventory exists: {inv_path}")
    inv = scan_baseline_inventory(INPUT)
    _write_json(inv_path, inv)
    source_hashes = {}
    for rel in ("scripts/exp_r2_pressure_gauge.py", "scripts/exp_thermal.py",
                "src/core/thermal_io.py", "src/models/featextractnet.py",
                "src/models/baselines/meshgraphnet_style.py"):
        p = os.path.join(WS, rel)
        if os.path.isfile(p):
            source_hashes[rel] = sha256_file(p)
    files = []
    for spec in MODEL_SPECS:
        ckpt, met = model_paths(INPUT, spec, 0)
        target = os.path.join(OUT, f"array_probe_{spec['name']}_seed0_geom{gid}.npz")
        _require(not os.path.exists(target), f"array probe target exists: {target}")
        model, metrics, stats, ckpt_path, metrics_path = load_model_and_metrics(
            INPUT, spec, 0, device)
        pred = predict_geometry(model, X[gid], stats, device)
        ref = Y[gid].astype(np.float64)
        pm = p_metrics(pred[:, P_INDEX], ref[:, P_INDEX])
        np.savez_compressed(
            target,
            pred=pred.astype(np.float64), ref=ref,
            pred_p=pred[:, P_INDEX].astype(np.float64),
            ref_p=ref[:, P_INDEX].astype(np.float64),
            coords=X[gid].astype(np.float64),
            geom_id=np.int64(gid), seed=np.int64(0), model=np.array(spec["name"]),
            checkpoint=np.array(os.path.relpath(ckpt_path, INPUT)),
            checkpoint_sha256=np.array(sha256_file(ckpt_path)),
            metrics=np.array(os.path.relpath(metrics_path, INPUT)),
            metrics_sha256=np.array(sha256_file(metrics_path)),
            data_sha256=np.array(sha256_file(DATA)),
            p_metrics=np.array(json.dumps(pm, ensure_ascii=False)),
        )
        files.append({"path": os.path.relpath(target, OUT),
                      "sha256": sha256_file(target),
                      "checkpoint_sha256": sha256_file(ckpt_path),
                      "metrics_sha256": sha256_file(metrics_path),
                      "geometry_id": gid, "seed": 0})
        print(f"[array-probe] {spec['name']} seed0 geom{gid} -> {target}", flush=True)
        del model
        _cleanup()
    manifest = {
        "task": "pressure_gauge_array_probe",
        "protocol": "R2-THERMAL-REFERENCE-02 A",
        "created_local": _now_local(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "input_workspace": INPUT, "out_dir": OUT, "data": DATA,
        "data_sha256": sha256_file(DATA), "splits": splits_path,
        "splits_sha256": sha256_file(splits_path),
        "test_ids": test_ids, "geometry_id": gid, "seed": 0,
        "source_sha256": source_hashes,
        "inventory": {"path": os.path.relpath(inv_path, OUT),
                      "sha256": sha256_file(inv_path)},
        "files": files,
        "note": "actual pred/ref arrays for independent NumPy verification; no science re-run",
    }
    _write_json(os.path.join(OUT, "array_probe_manifest.json"), manifest)
    print("[array-probe] done", flush=True)


def run_postprocess(args):
    """Derive postreview summary/report from existing 50961 raw; no GPU rerun."""
    INPUT, DATA, OUT, splits_path = resolve_paths(args)
    raw_path = os.path.join(OUT, "raw_results.json")
    old_protocol_path = os.path.join(OUT, "protocol_snapshot.json")
    old_delivery_path = os.path.join(OUT, "delivery_50961.json")
    for p in (raw_path, old_protocol_path):
        _require(os.path.isfile(p), f"missing postprocess input: {p}")
    for f in POSTREVIEW_FILES:
        if f == "baseline_inventory.json":
            continue  # inventory may already exist from the SLURM array-probe job
        _require(not os.path.exists(os.path.join(OUT, f)), f"postreview target exists: {OUT}/{f}")
    raw = json.load(open(raw_path, encoding="utf-8"))
    old_protocol = json.load(open(old_protocol_path, encoding="utf-8"))
    _, test_ids = load_split(INPUT)
    summary = build_summary(raw, test_ids)
    inv_path = os.path.join(OUT, "baseline_inventory.json")
    if os.path.isfile(inv_path):
        inventory = json.load(open(inv_path, encoding="utf-8"))
        inventory_source = "server array-probe job inventory"
    else:
        inventory = scan_baseline_inventory(INPUT)
        inventory_source = "local postprocess read-only scan (not server inventory)"
        _write_json(inv_path, inventory)
    summary["baseline_inventory"] = inventory
    summary["postreview"] = {
        "created_local": _now_local(),
        "derived_from_raw": os.path.relpath(raw_path, INPUT),
        "raw_sha256": sha256_file(raw_path),
        "raw_records": len(raw),
        "old_protocol_path": os.path.relpath(old_protocol_path, INPUT),
        "old_protocol_sha256": sha256_file(old_protocol_path),
        "inventory_source": inventory_source,
        "no_gpu_rerun": True,
        "note": ("postreview statistics/report only; original 50961 raw/summary/report/"
                 "delivery_50961 remain byte-unchanged"),
    }
    _write_json(os.path.join(OUT, "summary_postreview.json"), summary)
    write_report(OUT, summary, old_protocol, filename="report_postreview.md",
                 inventory=inventory)
    new_sources = {}
    for rel in ("scripts/exp_r2_pressure_gauge.py", "scripts/exp_thermal.py",
                "src/core/thermal_io.py", "src/models/featextractnet.py",
                "src/models/baselines/meshgraphnet_style.py"):
        p = os.path.join(WS, rel)
        if os.path.isfile(p):
            new_sources[rel] = sha256_file(p)
    arrays = {}
    for name in sorted(os.listdir(OUT)):
        if name.startswith("array_probe_") and name.endswith(".npz"):
            p = os.path.join(OUT, name)
            arrays[name] = sha256_file(p)
    old_delivery = None
    if os.path.isfile(old_delivery_path):
        old_delivery = {"path": os.path.relpath(old_delivery_path, INPUT),
                        "sha256": sha256_file(old_delivery_path)}
    protocol = {
        "task": "pressure_gauge_postreview",
        "protocol": "R2-THERMAL-REFERENCE-02 A",
        "r2_task": "R2-PRESSURE-GAUGE-02",
        "version": "postreview",
        "created_local": _now_local(),
        "input_workspace": INPUT, "out_dir": OUT,
        "data": DATA, "data_sha256": sha256_file(DATA),
        "splits_config": splits_path, "splits_sha256": sha256_file(splits_path),
        "test_ids": test_ids,
        "models": [{"name": s["name"], "token": s["token"],
                    "pool_context": s["pool_context"],
                    "params": s["expected_params"]} for s in MODEL_SPECS],
        "seeds": list(SEEDS),
        "old_protocol_sha256": sha256_file(old_protocol_path),
        "old_raw_sha256": sha256_file(raw_path),
        "old_delivery": old_delivery,
        "new_source_sha256": new_sources,
        "inventory": {"path": os.path.relpath(inv_path, INPUT),
                      "sha256": sha256_file(inv_path),
                      "source": inventory_source},
        "array_probes": arrays,
        "statistics": {
            "primary_cross_seed": ("mean/sample std across the five seed means; each seed "
                                  "mean over 20 geometries"),
            "geometry_std_auxiliary": "std across 20 geometry seed means; not seed uncertainty",
            "primary_paired_bootstrap": {
                "definition": "per geometry mean of 5 seeds, then FEN-graph paired difference",
                "reps": BOOTSTRAP_REPS, "analysis_seed": BOOTSTRAP_SEED,
                "shared_geometry_resample_across_metrics": True,
                "conditional_on_fixed_training_seeds": True,
                "not_100_independent_geometries": True,
            },
            "per_seed_bootstrap": "kept as supplementary",
        },
        "provenance": {
            "original_job_id": old_protocol.get("environment", {}).get("slurm_job_id"),
            "original_protocol_sha256": sha256_file(old_protocol_path),
            "postprocess_no_gpu_rerun": True,
            "source_changed_since_original_50961": True,
            "note": ("original 50961 execution source is recorded in old protocol; "
                     "new postreview source hashes recorded separately, not backfilled"),
        },
        "notes": ["postreview derived from original raw; original files retained",
                  "baseline inventory records actual checked roots; not global absence",
                  "array probes are for independent NumPy centered/pressure verification"],
    }
    _write_json(os.path.join(OUT, "protocol_snapshot_postreview.json"), protocol)
    delivery = {
        "task": "R2-PRESSURE-GAUGE-02 postreview delivery",
        "dsn_status": "postreview delivery complete; Codex final review pending",
        "original_50961": {"job_id": 50961,
                           "old_files_byte_unchanged": True,
                           "old_delivery_sha256": old_delivery["sha256"] if old_delivery else None},
        "postreview_files": {
            f: sha256_file(os.path.join(OUT, f)) for f in POSTREVIEW_FILES
            if os.path.exists(os.path.join(OUT, f))},
        "array_probes": arrays,
        "inventory_source": inventory_source,
        "statistics": protocol["statistics"],
        "notes": ["no GPU rerun; derived from original raw_results.json",
                  "original delivery hash not claimed for new files",
                  "source/protocol change separately recorded"],
    }
    _write_json(os.path.join(OUT, "delivery_postreview.json"), delivery)
    print("[postreview] summary/report/protocol/delivery written", flush=True)


def main():
    args = parse_args()
    if args.postprocess_only:
        run_postprocess(args)
    elif args.export_array_probe:
        run_export_array_probe(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
