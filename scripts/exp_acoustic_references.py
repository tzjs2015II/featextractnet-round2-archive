#!/usr/bin/env python3
"""R2-IMPL-03: acoustic reference baselines (global scalar, train pointwise mean, zero).

Deterministic baselines only; actual eval_case_physical is the sole scorer.
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
if WS not in sys.path:
    sys.path.insert(0, WS)

from src.core.acoustic_io import (  # noqa: E402
    make_case_list,
    load_case,
    eval_case_physical,
    summarize_case_rows,
)
from scripts.exp_acoustic import (  # noqa: E402
    CASE_ROLES,
    PRIMARY_TEST_CASE_IDS,
    DIAGNOSTIC_TEST_CASE_IDS,
)

GROUPS = (
    ('historical_test10', 'all 10 historical test cases'),
    ('fixed_left_test9', 'primary test cases with source side left/fixed'),
    ('source_right_diagnostic1', 'single diagnostic case with source on right'),
)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-cfg', required=True, help='acoustic_cases.json path')
    ap.add_argument('--out-dir', required=True, help='output directory')
    args = ap.parse_args(argv)

    out_dir = os.path.abspath(args.out_dir)
    json_path = os.path.join(out_dir, 'reference_scores.json')
    md_path = os.path.join(out_dir, 'report.md')
    for p in (json_path, md_path):
        if os.path.exists(p):
            raise SystemExit(f'refusing to overwrite existing {p}')

    cases = make_case_list(args.data_cfg, roles=CASE_ROLES)
    by_role = {}
    for c in cases:
        role = 'unused' if c['role'] == 'unassigned' else c['role']
        by_role.setdefault(role, []).append(c)
    counts = {r: len(by_role.get(r, [])) for r in ('train', 'val', 'test', 'unused')}
    assert counts == {'train': 6, 'val': 1, 'test': 10, 'unused': 4}, counts
    assigned = {k: v for r in ('train', 'val', 'test') for k, v in CASE_ROLES.items() if v == r}
    assert len(CASE_ROLES) == 17 and len(assigned) == 17 and len(cases) == 21
    train_ids = sorted(c['case_id'] for c in by_role['train'])
    val_ids = sorted(c['case_id'] for c in by_role['val'])
    test_ids = sorted(c['case_id'] for c in by_role['test'])
    unused_ids = sorted(c['case_id'] for c in by_role['unused'])
    assert test_ids == sorted(set(PRIMARY_TEST_CASE_IDS) | set(DIAGNOSTIC_TEST_CASE_IDS)), test_ids
    assert len(test_ids) == 10, test_ids

    def _load(entry):
        x2d, y2d, p2d, meta = load_case(entry['file'])
        assert p2d.dtype == np.float64
        return x2d, y2d, p2d, meta

    train = {cid: _load(c) for cid, c in ((c['case_id'], c) for c in by_role['train'])}
    test = {cid: _load(c) for cid, c in ((c['case_id'], c) for c in by_role['test'])}

    # Grid checks required for pointwise mean.
    ref_id = train_ids[0]
    rx, ry = train[ref_id][0], train[ref_id][1]
    for cid, (x2d, y2d, _, _) in train.items():
        assert x2d.shape == rx.shape and y2d.shape == ry.shape, cid
        assert np.array_equal(x2d, rx) and np.array_equal(y2d, ry), cid
    for cid, (x2d, y2d, _, _) in test.items():
        assert x2d.shape == rx.shape and y2d.shape == ry.shape, cid
        assert np.array_equal(x2d, rx) and np.array_equal(y2d, ry), cid

    train_fields = [train[c][2] for c in train_ids]
    total_points = int(sum(p.size for p in train_fields))
    scalar_mean = float(sum(float(p.sum()) for p in train_fields) / total_points)
    pointwise_mean = np.mean(np.stack(train_fields, axis=0), axis=0)

    baselines = {
        'train_scalar_mean': lambda ref: np.full_like(ref, scalar_mean),
        'train_pointwise_mean': lambda ref: pointwise_mean,
        'zero': lambda ref: np.zeros_like(ref),
    }

    baseline_rows = []
    for cid in test_ids:
        _, _, truth, _ = test[cid]
        for name, fn in baselines.items():
            m = eval_case_physical(fn(truth), truth)
            baseline_rows.append({
                'case_id': cid,
                'baseline': name,
                're': float(m['re']),
                'mae': float(m['mae']),
                'max_err': float(m['max_err']),
            })

    test_entries = {c['case_id']: c for c in by_role['test']}
    test_meta = {cid: {'source_side': test_entries[cid].get('source_side')}
                 for cid in test_ids}
    diag = list(DIAGNOSTIC_TEST_CASE_IDS)
    diag_set = set(diag)
    assert len(diag) == 1, diag
    left9 = [cid for cid in test_ids if cid not in diag_set]
    assert len(left9) == 9, left9
    groups = {
        'historical_test10': test_ids,
        'fixed_left_test9': left9,
        'source_right_diagnostic1': [cid for cid in test_ids if cid in diag_set],
    }
    assert set(test_ids) == set().union(*groups.values()), groups
    assert len(groups['source_right_diagnostic1']) == 1, groups['source_right_diagnostic1']
    for g, ids in groups.items():
        count = sum(1 for cid in ids
                    if str(test_meta[cid].get('source_side')) == 'right')
        if g == 'fixed_left_test9':
            assert count == 0, (g, count)
        if g == 'source_right_diagnostic1':
            assert count == 1, (g, count)

    baseline_aggregates = {name: {} for name in baselines}
    for name in baselines:
        by_case = {cid: r for r in baseline_rows if r['baseline'] == name
                   for cid in [r['case_id']]}
        for g, ids in groups.items():
            agg = summarize_case_rows([by_case[cid] for cid in ids])
            baseline_aggregates[name][g] = agg

    report = {
        'protocol': 'R2-IMPL-03-acoustic-references',
        'target': 'p_total',
        'data_definition': 'real phase-locked full-field snapshot (FDTD p_total)',
        'train_case_ids': train_ids,
        'val_case_ids': val_ids,
        'test_case_ids': test_ids,
        'unused_case_ids': unused_ids,
        'role_counts': counts,
        'global_scalar_mean': scalar_mean,
        'train_total_points': total_points,
        'group_rule': {
            'historical_test10': GROUPS[0][1],
            'fixed_left_test9': 'primary test set excluding the source-right '
                                'diagnostic case (fixed left source)',
            'source_right_diagnostic1': 'primary test case(s) with '
                                        'metadata source_side == right',
        },
        'group_case_ids': groups,
        'data_cfg_schema': 'acoustic_cases.json',
        'data_cfg_sha256': _sha256(os.path.abspath(args.data_cfg)),
        'script_sha256': _sha256(os.path.abspath(__file__)),
        'baseline_rows': baseline_rows,
        'baseline_aggregates': baseline_aggregates,
        'seeds': 'not applicable (deterministic baselines, no seeded variance)',
    }

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, 'reference_scores.json')
    md_path = os.path.join(out_dir, 'report.md')
    with open(json_path, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
        fh.write('\n')

    names = list(baselines)
    lines = ['# Acoustic reference baselines', '',
             'Stored-scale RE / MAE / max error per baseline and case group.', '',
             '| baseline | ' + ' | '.join(g for g, _ in GROUPS) + ' |',
             '|---|' + '---|' * len(GROUPS)]
    for name in names:
        cells = []
        for g, _ in GROUPS:
            a = baseline_aggregates[name][g]
            cells.append('RE={:.6f} / MAE={:.6e} / max={:.6e}'.format(
                a['re']['p_total'], a['mae']['p_total'], a['max_err']['p_total']))
        lines.append('| {} | {} |'.format(name, ' | '.join(cells)))
    lines += ['', '- global scalar: mean of all train p_total values',
              '- pointwise: element-wise train mean; zero: exact zeros field',
              '- Scores computed with src.core.acoustic_io.eval_case_physical.',
              '- Deterministic references; no learned-model training.']
    with open(md_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')

    print('wrote', json_path, md_path)


if __name__ == '__main__':
    main()
