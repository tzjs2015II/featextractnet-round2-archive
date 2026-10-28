"""scripts/exp_r2_geometry.py - thin single-seed entry for R2 geometry protocols.

Trains feat_extract or meshgraphnet on one R2 protocol (version R2_geometry_v1),
selects the epoch on val only, and evaluates test exactly once after training.
Reuses scripts.exp_thermal.build_model, src.core.runner and src.core.thermal_io.

Output: <out-root>/<protocol>/<model>/seed<N>/
  protocol_snapshot.json (written before training), checkpoint_best.pth,
  metrics_seed<N>.json (both written by runner), report_seed<N>.md.
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WS not in sys.path:
    sys.path.insert(0, WS)

from src.core import runner, thermal_io  # noqa: E402
from src.utils.data_loader import set_seed  # noqa: E402
from scripts.exp_thermal import build_model  # noqa: E402

PROTOCOL_VERSION = 'R2_geometry_v1'
N_GEOM = 135
SPLITS = ('train', 'val', 'test', 'unused')
R2_MARKERS = ('round2', 'r2')
LEGACY_ROOTS = ('thermal_protocolA', 'ablation_thermal_protocolA')
HASH_FILES = (
    'scripts/exp_r2_geometry.py',
    'scripts/exp_thermal.py',
    'src/core/runner.py',
    'src/core/thermal_io.py',
)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol-file', required=True)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--model', required=True, choices=('feat_extract', 'meshgraphnet'))
    ap.add_argument('--seed', required=True, type=int)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out-root', required=True)
    ap.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    ap.add_argument('--epochs', type=int, default=500)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--batch-geoms', type=int, default=4)
    ap.add_argument('--val-every', type=int, default=25)
    return ap.parse_args()


def resolve_out_root(out_root):
    """Any path component containing round2 or r2 is accepted; legacy roots refused."""
    root = os.path.realpath(os.path.abspath(out_root))
    parts = [p for p in root.replace('\\', '/').split('/') if p]
    if not any(m in p.lower() for p in parts for m in R2_MARKERS):
        raise SystemExit(f'[refuse] --out-root has no round2/r2 path component: {root}')
    for legacy in LEGACY_ROOTS:
        bad = os.path.realpath(os.path.join(WS, 'results', legacy))
        if root == bad or root.startswith(bad + os.sep):
            raise SystemExit(f'[refuse] --out-root is inside legacy results {legacy}')
    return root


def sha256_file(path):
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load_protocol(path):
    with open(path, 'r', encoding='utf-8') as fh:
        doc = json.load(fh)
    if doc.get('protocol_version') != PROTOCOL_VERSION:
        raise SystemExit(f'[refuse] protocol_version != {PROTOCOL_VERSION}')
    return doc


def validate_split(protocol, split_name, meta):
    if split_name not in protocol:
        raise SystemExit(f'[refuse] unknown protocol {split_name!r}')
    split = protocol[split_name]
    ids = {r: [int(g) for g in split[r]] for r in SPLITS}
    for r in SPLITS:
        if len(ids[r]) != len(set(ids[r])):
            raise SystemExit(f'[refuse] duplicate ids in {split_name}/{r}')
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            if set(ids[a]) & set(ids[b]):
                raise SystemExit(f'[refuse] {a}/{b} overlap in {split_name}')
    union = set().union(*(set(v) for v in ids.values()))
    if union != set(range(N_GEOM)):
        raise SystemExit(f'[refuse] split union != 0..{N_GEOM - 1} for {split_name}')

    kind = split.get('kind')
    if kind == 'held_out_polygon_family':
        sides = int(split['held_out_sides'])
        for g in ids['test']:
            if int(meta[g]['polygon_sides']) != sides:
                raise SystemExit(f'[refuse] test geom {g} not in held-out family {sides}')
        for r in ('train', 'val'):
            for g in ids[r]:
                if int(meta[g]['polygon_sides']) == sides:
                    raise SystemExit(f'[refuse] {r} geom {g} in held-out family {sides}')
    elif kind == 'disjoint_canonical_rotation_ranges':
        intervals = split['phase_fraction_intervals']
        for r in ('train', 'val', 'test'):
            lo, hi = float(intervals[r][0]), float(intervals[r][1])
            for g in ids[r]:
                frac = float(meta[g]['canonical_phase_fraction'])
                if not (lo <= frac < hi):
                    raise SystemExit(f'[refuse] {r} geom {g} fraction {frac} outside [{lo},{hi})')
        used = set(ids['train']) | set(ids['val']) | set(ids['test'])
        if used | set(ids['unused']) != set(range(N_GEOM)):
            raise SystemExit('[refuse] unused is not the complement of used ids')
    else:
        raise SystemExit(f'[refuse] unknown protocol kind {kind!r}')
    return split, ids


def val_selector_factory(model, val_ids, X, Y, stats, device):
    def selector(m):
        agg = thermal_io.eval_geometry_wise(
            lambda x: m(x), val_ids, X, Y, stats, device, chunk_points=None)['agg']
        score = float(np.mean(list(agg['re'].values())))
        if not np.isfinite(score):
            raise FloatingPointError(f'non-finite validation score {score}')
        return score
    return selector


def test_eval_factory(test_ids, X, Y, stats, device):
    def evaluate(m):
        ev = thermal_io.eval_geometry_wise(
            lambda x: m(x), test_ids, X, Y, stats, device, chunk_points=None)
        return {'n_geom': ev['n_geom'], 'agg': ev['agg'], 'rows': ev['rows']}
    return evaluate


def write_snapshot(seed_dir, snapshot):
    path = os.path.join(seed_dir, 'protocol_snapshot.json')
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(snapshot, fh, indent=2)
        fh.write('\n')
    os.replace(tmp, path)


def write_report(seed_dir, seed, metrics):
    best = metrics['val_best']
    test = metrics['test']
    lines = [
        f'# R2 geometry report: seed {seed}',
        '',
        f'- best epoch: {best["epoch"]}',
        f'- best val mean RE: {best["score"]}',
        f'- test geometries: {test["n_geom"]}',
        '',
        '| field | RE | MAE | max_err | mae_bd |',
        '|---|---|---|---|---|',
    ]
    for key in test['agg']['re']:
        a = test['agg']
        lines.append(f'| {key} | {a["re"][key]:.6f} | {a["mae"][key]:.6e} | '
                     f'{a["max_err"][key]:.6e} | {a["mae_bd"][key]:.6e} |')
    with open(os.path.join(seed_dir, f'report_seed{seed}.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')


def main():
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('[refuse] --device cuda requested but CUDA is unavailable')
    device = torch.device(args.device)
    if args.epochs < 1 or args.batch_geoms < 1 or args.val_every < 1:
        raise SystemExit('[refuse] epochs, batch-geoms and val-every must be >= 1')

    out_root = resolve_out_root(args.out_root)
    data_path = os.path.realpath(os.path.abspath(args.data))
    protocol_path = os.path.realpath(os.path.abspath(args.protocol_file))
    doc = load_protocol(protocol_path)
    meta = {int(m['geom_id']): m for m in doc['metadata']}
    if set(meta) != set(range(N_GEOM)):
        raise SystemExit('[refuse] metadata does not cover geom ids 0..134')
    split, ids = validate_split(doc['protocols'], args.protocol, meta)

    X, Y = thermal_io.load_thermal(data_path)
    if X.shape[:2] != (N_GEOM, 5000) or Y.shape[:2] != (N_GEOM, 5000):
        raise SystemExit(f'[refuse] unexpected data shape X={X.shape} Y={Y.shape}')
    stats = thermal_io.fit_stats(X, Y, geom_ids=ids['train'])

    set_seed(args.seed)
    if args.model == 'feat_extract':
        model, label = build_model('feat_extract', pool_context='symmetric')
    else:
        model, label = build_model('meshgraphnet')
    n_params = sum(p.numel() for p in model.parameters())

    seed_dir = os.path.join(out_root, args.protocol, args.model, f'seed{args.seed}')
    protected = ('checkpoint_best.pth', f'metrics_seed{args.seed}.json',
                 'protocol_snapshot.json')
    for name in protected:
        if os.path.exists(os.path.join(seed_dir, name)):
            raise SystemExit(f'[refuse] {os.path.join(seed_dir, name)} already exists')
    os.makedirs(seed_dir, exist_ok=True)

    snapshot = {
        'protocol_version': doc['protocol_version'],
        'protocol': args.protocol,
        'protocol_kind': split.get('kind'),
        'model': args.model,
        'model_label': label,
        'model_params': int(n_params),
        'seed': args.seed,
        'epochs': args.epochs,
        'lr': args.lr,
        'batch_geoms': args.batch_geoms,
        'val_every': args.val_every,
        'device': str(device),
        'split_ids': {r: ids[r] for r in SPLITS},
        'split_counts': {r: len(ids[r]) for r in SPLITS},
        'norm_stats': {k: np.asarray(stats[k]).tolist()
                       for k in ('x_mean', 'x_std', 'y_mean', 'y_std')},
        'norm_fit': 'train-only',
        'fp32_flags': {
            'cuda_matmul_allow_tf32': bool(torch.backends.cuda.matmul.allow_tf32),
            'cudnn_allow_tf32': bool(torch.backends.cudnn.allow_tf32),
        },
        'optimizer': {'name': 'Adam', 'lr': args.lr},
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'config_sha256': sha256_file(protocol_path),
        'script_sha256': sha256_file(os.path.join(WS, 'scripts/exp_r2_geometry.py')),
        'file_sha256': {rel: sha256_file(os.path.join(WS, rel)) for rel in HASH_FILES},
        'data_path': data_path,
        'protocol_file': protocol_path,
    }
    write_snapshot(seed_dir, snapshot)

    provider = thermal_io.make_thermal_train_provider(
        X, Y, stats, ids['train'], batch_size=args.batch_geoms,
        n_pts=None, device=device, seed=args.seed)

    runner.train_supervised(
        model, task='thermal', data_provider=provider, epochs=args.epochs,
        lr=args.lr, seed=args.seed, device=device,
        val_eval_every=args.val_every,
        val_selector=val_selector_factory(model, ids['val'], X, Y, stats, device),
        test_eval_fn=test_eval_factory(ids['test'], X, Y, stats, device),
        out_dir=seed_dir, patience=None, protocol_extra=snapshot)

    metrics_path = os.path.join(seed_dir, f'metrics_seed{args.seed}.json')
    with open(metrics_path, 'r', encoding='utf-8') as fh:
        metrics = json.load(fh)
    write_report(seed_dir, args.seed, metrics)
    print(f'[done] {seed_dir} best_epoch={metrics["val_best"]["epoch"]}', flush=True)


if __name__ == '__main__':
    main()
