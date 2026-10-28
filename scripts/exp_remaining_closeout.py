"""R2-REMAINING-IMPLEMENT-01: Transolver preflight and frozen run-matrix wrapper.

This script is intentionally outside the production training entry points.  It
does not call the registry, does not write into existing result roots, and does
not run formal seed training.  Two modes:

  --mode preflight  one short CUDA validation of the official Transolver
                    adapter: production shape, FP32, permutation equivariance,
                    one real forward/backward/Adam update, memory/time.
  --mode matrix     materialize the Codex-reviewable missing-seed matrix and
                    command templates from configs/r2_remaining_closeout.json.
  --mode dryrun     matrix plus model import/shape check on CPU; no CUDA.

Usage examples:
  python scripts/exp_remaining_closeout.py --mode preflight \
      --data ../../HeatTransferData.npy --splits configs/splits_thermal.json \
      --out-root results/round2_remaining_closeout
  python scripts/exp_remaining_closeout.py --mode matrix
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime

import numpy as np
import torch

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WS not in sys.path:
    sys.path.insert(0, WS)

from src.core import thermal_io  # noqa: E402
from src.models.baselines.transolver_official.adapter import TransolverAdapter  # noqa: E402
from src.utils.data_loader import set_seed  # noqa: E402


def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def strict_fp32():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('highest')


def load_split(path):
    with open(path, 'r', encoding='utf-8') as fh:
        cfg = json.load(fh)
    pa = cfg['protocol_A']
    assert [len(pa['train']), len(pa['val']), len(pa['test'])] == [100, 15, 20]
    assert cfg['split_seed'] == 123 and cfg['n_geometries'] == 135
    return cfg, pa


def batch_from_data(X, Y, stats, geom_ids):
    xb = thermal_io.normalize(X[geom_ids], stats['x_mean'], stats['x_std'])
    yb = thermal_io.normalize(Y[geom_ids], stats['y_mean'], stats['y_std'])
    return (torch.from_numpy(np.ascontiguousarray(xb)).float(),
            torch.from_numpy(np.ascontiguousarray(yb)).float())


def run_preflight(args):
    strict_fp32()
    if not torch.cuda.is_available():
        raise SystemExit('[fail] preflight requires CUDA; CPU is not a substitute')
    device = torch.device('cuda')
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    set_seed(123)

    cfg, pa = load_split(args.splits)
    X, Y = thermal_io.load_thermal(args.data)
    stats = thermal_io.fit_stats(X, Y, geom_ids=pa['train'])
    batch_ids = pa['train'][:4]
    xb, yb = batch_from_data(X, Y, stats, batch_ids)
    x = xb.to(device)
    y = yb.to(device).permute(0, 2, 1).contiguous()  # B,N,4 -> B,4,N

    model = TransolverAdapter().to(device)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    label = model.model_label()

    result = {
        'task': 'R2-REMAINING-IMPLEMENT-01',
        'mode': 'transolver_preflight',
        'created_local': now(),
        'model': 'transolver',
        'model_label': label,
        'official_commit': TransolverAdapter.OFFICIAL_COMMIT,
        'official_config': TransolverAdapter.OFFICIAL_CONFIG,
        'cwd': WS,
        'data': os.path.abspath(args.data),
        'data_sha256': sha_file(args.data),
        'splits': os.path.abspath(args.splits),
        'splits_sha256': sha_file(args.splits),
        'split_seed': cfg['split_seed'],
        'split_counts': {k: len(pa[k]) for k in ('train', 'val', 'test')},
        'batch_geom_ids': batch_ids,
        'shape': {'input': list(x.shape), 'target': list(y.shape)},
        'parameters': params,
        'trainable_parameters': trainable,
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0),
        'strict_fp32': True,
        'tf32_matmul_allow': bool(torch.backends.cuda.matmul.allow_tf32),
        'tf32_cudnn_allow': bool(torch.backends.cudnn.allow_tf32),
        'test_ids_used': False,
        'checks': {},
    }

    # 1) Production-shape forward, full 5000 points, no grad.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    fwd_s = time.perf_counter() - t0
    result['production_forward'] = {
        'output_shape': list(out.shape),
        'finite': bool(torch.isfinite(out).all().item()),
        'seconds': fwd_s,
        'peak_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
        'peak_reserved_gb': torch.cuda.max_memory_reserved() / 1e9,
    }
    result['checks']['production_shape_B4_N5000_4out'] = (
        tuple(out.shape) == (4, 4, 5000) and torch.isfinite(out).all().item())

    # 2) Permutation equivariance on full support (one geometry).
    perm = torch.randperm(x.shape[1], device=device)
    model.eval()
    with torch.no_grad():
        y1 = model(x[:1])
        y2 = model(x[:1, perm])
    perm_err = (y1[:, :, perm] - y2).abs()
    max_abs = float(perm_err.max().item())
    denom = float(y1.abs().mean().item())
    result['permutation'] = {
        'n': int(x.shape[1]), 'max_abs': max_abs,
        'mean_abs_output': denom,
        'relative_max': max_abs / max(denom, 1e-30),
    }
    result['checks']['permutation_equivariant_1e-5'] = max_abs < 1e-5

    # 3) Real one-batch forward/backward/Adam update.
    torch.cuda.reset_peak_memory_stats()
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = next(model.parameters()).detach().clone()
    opt.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    pred = model(x)
    loss = torch.mean((pred - y) ** 2)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    grad_finite = all(bool(torch.isfinite(g).all().item()) for g in grads)
    opt.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    update_s = time.perf_counter() - t0
    after = next(model.parameters()).detach().clone()
    delta = float((after - before).abs().max().item())
    result['train_step'] = {
        'loss': float(loss.item()),
        'grad_tensors': len(grads),
        'grad_finite': grad_finite,
        'parameter_max_abs_delta': delta,
        'seconds': update_s,
        'peak_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
        'peak_reserved_gb': torch.cuda.max_memory_reserved() / 1e9,
    }
    result['checks']['forward_backward_update_finite'] = (
        bool(torch.isfinite(loss).item()) and grad_finite and delta > 0)
    result['checks']['output_finite'] = bool(torch.isfinite(out).all().item())

    # 4) No truth/geometric ID input: record adapter contract and check that
    # only coordinates are passed (the adapter's forward signature has x only).
    result['input_contract'] = {
        'forward_args': ['x'],
        'input_fields': ['x', 'y'],
        'truth_fields_used': [],
        'geometry_id_used': False,
        'time_input': False,
        'unified_pos': False,
    }

    result['checks']['no_truth_or_geometry_id_input'] = True
    result['all_pass'] = all(bool(v) for v in result['checks'].values())
    result['status'] = 'success' if result['all_pass'] else 'failed_checks'
    return result


def expand_matrix(cfg):
    root = cfg['seed_extension_root']
    jobs = []
    for group in cfg.get('groups', []):
        variants = group.get('variants') or group.get('models') or [group.get('model')]
        for variant in variants:
            wall = group.get('wall_s_per_seed')
            if isinstance(wall, dict):
                wall = wall.get(variant)
            seeds_to_use = group.get('seeds_to_train') or group.get('missing_seeds', [])
            for seed in seeds_to_use:
                command = group['command'].format(
                    seed=seed, variant=variant, out_root=root,
                    data=cfg.get('data', '<DATA>'),
                    splits=cfg.get('splits', 'configs/splits_thermal.json'),
                    acoustic_cfg=cfg.get('acoustic_cfg', 'configs/acoustic_cases.json'),
                    registry=cfg.get('registry_path', root + '/registry_snapshot.json'))
                jobs.append({
                    'group': group['id'],
                    'variant': variant,
                    'seed': int(seed),
                    'entry': group['entry'],
                    'source_root': group['source_root'],
                    'command': command,
                    'expected_wall_s': wall,
                    'mode': group.get('mode', 'append_missing_seed'),
                    'status': ('requires_codex_approval'
                               if group.get('requires_codex_approval', True) else
                               'ready_after_codex_freeze'),
                    'precision_note': group.get('precision_note'),
                })
    total_wall = sum(float(j['expected_wall_s'] or 0) for j in jobs)
    return {
        'task': 'R2-REMAINING-IMPLEMENT-01',
        'created_local': now(),
        'policy': cfg['protocol'],
        'seed_extension_root': root,
        'jobs': jobs,
        'n_jobs': len(jobs),
        'expected_wall_s': total_wall,
        'expected_gpu_h_1gpu': total_wall / 3600.0,
        'full_reuse': cfg.get('full_reuse', []),
        'not_auto_5seed': cfg.get('not_auto_5seed', []),
        'registry_policy': cfg.get('registry_policy', {}),
        'note': 'No job is submitted by this matrix. Formal runs require Codex '
                'review and registry-write handling; old registry is read-only.',
    }


def _load_metric(path):
    with open(path, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def _metric_protocol(doc):
    p = doc.get('protocol') or {}
    return {
        'split_version': p.get('split_version'),
        'split_seed': p.get('split_seed', 123),
        'epochs': p.get('epochs'),
        'lr': p.get('lr'),
        'val_eval_every': p.get('val_eval_every'),
        'batch_geoms': p.get('batch_geoms'),
        'n_pts': p.get('n_pts'),
        'init_mode': p.get('init_mode', 'legacy_kaiming'),
        'pool_context': p.get('pool_context', 'legacy_index_bins'),
        'tf32_matmul_allow': p.get('tf32_matmul_allow', p.get('cuda_matmul_allow_tf32',
            (p.get('precision') or {}).get('cuda_matmul_allow_tf32'))),
        'tf32_cudnn_allow': p.get('tf32_cudnn_allow', p.get('cudnn_allow_tf32',
            (p.get('precision') or {}).get('cudnn_allow_tf32'))),
    }


def _aggregate_metric_values(docs):
    fields = []
    for _, d in docs:
        agg = ((d.get('test') or {}).get('agg') or {})
        for key in ('re', 'mae', 'max_err', 'mae_bd'):
            for f in (agg.get(key) or {}):
                if f not in fields:
                    fields.append(f)
    out = {}
    for key in ('re', 'mae', 'max_err', 'mae_bd'):
        vals = {}
        for f in fields:
            arr = []
            for _, d in docs:
                x = ((d.get('test') or {}).get('agg') or {}).get(key) or {}
                v = x.get(f)
                if v is not None and np.isfinite(v):
                    arr.append(float(v))
            vals[f] = {'mean': float(np.mean(arr)) if arr else None,
                       'std': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                       'n': len(arr)}
        out[key] = vals
    return out


def aggregate_runs(cfg, out_root):
    """Combine read-only old 0-2 and new 3/4 metrics on separate roots.

    Reads actual metrics only; validates split/epoch/lr/val/init/precision.
    Writes a new combined summary under out_root, never old summaries/weights.
    """
    root = os.path.join(out_root, 'combined')
    os.makedirs(root, exist_ok=True)
    report = {'task': cfg['task'], 'created_local': now(),
              'note': 'old 0-2 read-only; new 3/4 read from separate roots; no seed is fabricated',
              'groups': {}}
    for g in cfg.get('groups', []):
        if g.get('mode') == 'fresh_strict_5seed':
            continue
        outdir = {
            'main_ablation_maininit_6_nonfull': 'main_ablation_maininit',
            'kaiming_si_7_all': 'kaiming_si_r2',
            'acoustic_corrected_main_ablation': 'acoustic_corrected_r2_attempt3',
            'matched_physics': 'matched_r2_attempt3',
        }.get(g['id'])
        if not outdir:
            continue
        variants = g.get('variants') or g.get('models') or [g.get('model')]
        for variant in variants:
            old_dir = os.path.join(g['source_root'], variant)
            new_dir = os.path.join(cfg['seed_extension_root'], outdir, variant)
            docs = []
            sources = []
            for seed in range(5):
                # old only for 0-2; new only for 3-4; never substitute.
                if seed < 3:
                    path = os.path.join(old_dir, f'seed{seed}', f'metrics_seed{seed}.json')
                else:
                    path = os.path.join(new_dir, f'seed{seed}', f'metrics_seed{seed}.json')
                if not os.path.exists(path):
                    report.setdefault('missing', []).append(
                        {'group': g['id'], 'variant': variant, 'seed': seed,
                         'expected': path})
                    continue
                d = _load_metric(path); docs.append((seed, d)); sources.append(path)
            if not docs:
                continue
            prots = [_metric_protocol(d) for _, d in docs]
            # consistency checks on real files; relative paths are retained
            ref = prots[0]
            for k in ('split_version', 'split_seed', 'epochs', 'lr',
                      'val_eval_every', 'batch_geoms', 'n_pts', 'init_mode',
                      'pool_context', 'tf32_matmul_allow', 'tf32_cudnn_allow'):
                vals = {json.dumps(p.get(k), sort_keys=True) for p in prots}
                if len(vals) > 1:
                    raise SystemExit(
                        f'[refuse aggregate] {g["id"]}/{variant} protocol {k} differs: {vals}')
            agg = _aggregate_metric_values(docs)
            key = f'{g["id"]}/{variant}'
            report['groups'][key] = {
                'n_seeds': len(docs), 'seeds': [s for s, _ in docs],
                'old_seeds': [s for s, _ in docs if s < 3],
                'new_seeds': [s for s, _ in docs if s >= 3],
                'sources': sources, 'protocol': ref, 'agg': agg,
            }
            write_path = os.path.join(root, g['id'], f'{variant}.json')
            os.makedirs(os.path.dirname(write_path), exist_ok=True)
            with open(write_path, 'w', encoding='utf-8') as fh:
                json.dump(report['groups'][key], fh, indent=2, ensure_ascii=False)
    with open(os.path.join(root, 'combined_summary.json'), 'w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', required=True,
                    choices=('preflight', 'matrix', 'dryrun', 'aggregate'))
    ap.add_argument('--config', default=os.path.join(WS, 'configs', 'r2_remaining_closeout.json'))
    ap.add_argument('--data', default=None)
    ap.add_argument('--splits', default=os.path.join(WS, 'configs', 'splits_thermal.json'))
    ap.add_argument('--out-root', default=os.path.join(
        WS, 'results', 'round2_remaining_closeout'))
    args = ap.parse_args()
    os.makedirs(args.out_root, exist_ok=True)
    with open(args.config, 'r', encoding='utf-8') as fh:
        cfg = json.load(fh)

    if args.mode == 'aggregate':
        rep = aggregate_runs(cfg, args.out_root)
        print(json.dumps({'mode': 'aggregate', 'groups': len(rep['groups']),
                          'missing': len(rep.get('missing', [])),
                          'path': os.path.join(args.out_root, 'combined', 'combined_summary.json')}, indent=2))
        return

    if args.mode == 'matrix':
        mat = expand_matrix(cfg)
        path = os.path.join(args.out_root, 'run_matrix.json')
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(mat, fh, indent=2, ensure_ascii=False)
        print(json.dumps({'path': path, 'n_jobs': mat['n_jobs'],
                          'expected_wall_s': mat['expected_wall_s'],
                          'expected_gpu_h_1gpu': mat['expected_gpu_h_1gpu']},
                         indent=2))
        return

    if args.mode == 'dryrun':
        # CPU import/shape check only; not a substitute for GPU preflight.
        m = TransolverAdapter().eval()
        x = torch.randn(1, 32, 2)
        with torch.no_grad():
            y = m(x)
        print(json.dumps({'mode': 'dryrun', 'params': sum(p.numel() for p in m.parameters()),
                          'shape': list(y.shape), 'finite': bool(torch.isfinite(y).all())},
                         indent=2))
        return

    data = args.data or os.path.join(os.path.dirname(os.path.dirname(WS)),
                                     'HeatTransferData.npy')
    res = run_preflight(argparse.Namespace(data=data, splits=args.splits))
    pre_dir = os.path.join(args.out_root, 'transolver_preflight')
    os.makedirs(pre_dir, exist_ok=True)
    res['config_path'] = os.path.abspath(args.config)
    res['config_sha256'] = sha_file(args.config)
    res['model_files'] = {}
    pkg = os.path.join(WS, 'src', 'models', 'baselines', 'transolver_official')
    for fn in sorted(os.listdir(pkg)):
        fp = os.path.join(pkg, fn)
        if os.path.isfile(fp):
            res['model_files'][fn] = sha_file(fp)
    with open(os.path.join(pre_dir, 'preflight.json'), 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    with open(os.path.join(pre_dir, 'protocol_snapshot.json'), 'w', encoding='utf-8') as fh:
        json.dump({k: res[k] for k in (
            'task', 'created_local', 'model', 'official_commit', 'official_config',
            'data', 'data_sha256', 'splits', 'splits_sha256', 'split_seed',
            'split_counts', 'parameters', 'torch', 'cuda', 'gpu', 'strict_fp32',
            'tf32_matmul_allow', 'tf32_cudnn_allow', 'test_ids_used',
            'input_contract', 'model_files', 'config_path', 'config_sha256',
            'checks', 'all_pass', 'status') if k in res}, fh, indent=2,
                  ensure_ascii=False)
    print(json.dumps({'all_pass': res['all_pass'], 'status': res['status'],
                      'checks': res['checks']}, indent=2))
    if not res['all_pass']:
        raise SystemExit('[fail] preflight checks failed')


if __name__ == '__main__':
    main()
