"""scripts/probe_round2_package.py —— R2 准备包的实现级自检（OS 临时目录，不写真实结果根）。

用途
----
在不发布、不训练、不联网、不安装、不写项目结果目录的前提下，验证
``scripts/package_round2.py`` 的实现契约：

1. 真实计划的固定白名单：src 全部 .py（排除 src/pipeline）、指定科学入口 +
   ``exp_pinn_thermal`` helper、明确包含 ``exp_r2_new_queries``、3 个共享配置、
   round2 结果目录（含 ``round2_new_queries`` 仅 *.json，排除内部
   ``registry_candidate.json``）内全部已完成 *.json、论文仍保留的历史有效 baseline
   精确白名单（thermal_protocolA 五个、baselines_ext 三个、acoustic_protocolA 四个、
   acoustic_wp5_ablation/ab_nofourier），以及仅表数据 source manifest 的 4 个
   round2_audit 文件；不含 data/权重/logs/其它 audit/registry/docs/paper/sbatch/
   旧 pipeline/已撤去旧 baseline；把 Codex 元数据当前缺失状态作为 pending 报告
   （不是失败）。请求但实际不存在的 ``results/acoustic_protocolA/ab_nofourier``
   必须被记录为 requested-missing 而不捏造。new_queries 已完成，但 PENDING_STATUS
   必须明确其参考是 interpolated-reference diagnostic、不是独立 FEM。
   必须包含 ``results/thermal_geometry_spec.json``（exp_r2_local_error /
   exp_r2_new_queries 的既有几何规格）与动态加载的 ``scripts/exp_mechanism.py``；
   缺失即拒绝。动态 importlib 加载（spec_from_file_location / run_path）必须由
   AST 扫描解析并校验其目标是否入包，不能仅靠静态白名单断言。
   三个冻结 split 快照 ``results/round2_execution_20261021_{v3,v7,v10}/configs/
   splits_thermal.json`` 必须按精确路径入包（配置类、byte-identical、不复制整个
   execution 目录）；probe 从真实 metric 的 ``protocol.splits_config`` 字段动态发现
   引用，并断言 snapshot 的 protocol_A train/val/test 与 ``configs/splits_thermal.json``
   实际一致，而不是只检查常量文件列表。
2. ``--list-only``：只读检查路径名单与缺失元数据，不创建输出目录；非法包名拒绝。
3. JSON release-copy：允许的 repo/workspace 私有 root 前缀改写为项目相对路径；
   相对路径前缀 ``返稿意见/返稿修订工作区/results/...`` 规范为 ``results/...``；
   数值/case_id/split ids/seed/统计/列表顺序完全不变（逐一断言）；未知私有绝对路径
   → 阻断且不输出内容。
4. 文本潜在凭据扫描：只给文件名/行号/规则，绝不输出匹配内容。
5. 小临时 fixture 全流程：构造 mock 工作区 → 真实 ``build_package`` → 归档成员/SHA/
   数量校验 → ``safe_extract`` 解包校验 → 数字不变性校验；含权重/数据/logs 的
   mock 文件必须被排除；``..``/绝对路径/symlink/hardlink 成员必须被拒绝；同名包
   拒绝覆盖；缺元数据/未知绝对路径/凭据命中均拒绝且不留半成品。
6. 模型 CPU 小 shape 冒烟：thermal ``FeatExtractNet`` 显式 ``pool_context=symmetric``；
   acoustic ``fourier_c`` 显式 ``fourier_layout=pointwise``；CPU 前向有限值。
7. core 数据分组路径只在项目内存在真实数据时检查（不自动下载）；torchgeo 缺失时
   明确 skip（不安装）；CUDA 不运行。

边界
----
* 不写项目内任何结果/包目录；fixture 与解包均在 OS 临时目录，结束后删除。
* 元数据（``zenodo/round2/`` 4 文件）尚缺时本 probe 仍可运行并通过：真实包构建
  属于 Codex 提供元数据后的步骤，probe 只报告 pending。
* 这不是科学结果、不是训练/收敛验证，也不构成发布。

用法
----
  python3 scripts/probe_round2_package.py
  python3 scripts/probe_round2_package.py --output <OS临时目录>/probe_round2_package.json
"""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import platform
import re
import shutil
import socket
import sys
import tarfile
import tempfile
import time
import traceback

# 线程上限：必须在 numpy/torch 初始化之前
for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_var] = '4'

import numpy as np  # noqa: E402

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WS not in sys.path:
    sys.path.insert(0, WS)

import scripts.package_round2 as pkg  # noqa: E402

PROBE_VERSION = 'R2-RELEASE-PACKAGE-PROBE-v1'
FAKE_THREADS = 4


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------
class ProbeBlocked(Exception):
    """A check cannot be concluded because required external sources are missing."""


class Probe:
    def __init__(self):
        self.checks = []
        self.failures = []
        self.blocked = []

    def record(self, name, ok, detail=None):
        entry = {'name': name, 'status': 'passed' if ok else 'failed',
                 'detail': detail}
        self.checks.append(entry)
        if not ok:
            self.failures.append(name)
        return ok

    def record_blocked(self, name, detail):
        self.checks.append({'name': name, 'status': 'blocked',
                            'detail': detail})
        self.blocked.append(name)
        return False

    def check(self, name, fn):
        try:
            detail = fn()
            return self.record(name, True, detail)
        except ProbeBlocked as exc:
            self.record_blocked(name, str(exc))
            return False
        except Exception as exc:            # noqa: BLE001 - report every failure
            self.record(name, False,
                        f'{type(exc).__name__}: {exc}\n'
                        + ''.join(traceback.format_exc().splitlines(True)[-4:]))
            return False


def expect_raises(exc_types, fn, what):
    try:
        fn()
    except exc_types as exc:
        return f'refused as expected ({type(exc).__name__}): {exc}'
    raise AssertionError(f'{what} did not raise {exc_types}')


def _sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _sha_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# mock workspace / archives
# ---------------------------------------------------------------------------
def _write(path, text, mode='w'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if mode == 'w':
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(text)
    else:
        with open(path, 'wb') as fh:
            fh.write(text)


def _write_dynamic_loader(ws, target_rel):
    """Overwrite the mock fig_r2_mechanism with an importlib dynamic source load."""
    code = (
        "import importlib.util\n"
        "import os\n"
        "WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\n"
        f"_TARGET = os.path.join(WS, *{target_rel.split('/')!r})\n"
        "_spec = importlib.util.spec_from_file_location('dynamic_target', _TARGET)\n"
    )
    _write(os.path.join(ws, 'scripts', 'fig_r2_mechanism.py'), code)


def _main_metric_doc(seed, v17_abs, v18_abs, epochs=500):
    """Synthetic five-seed main metrics fixture (500 epochs, six aggregate blocks)."""
    blocks = ('re', 'mae', 'max_err', 're_bd', 'mae_bd', 'max_bd')
    agg = {}
    for b, key in enumerate(blocks):
        fields = {}
        for f, field in enumerate(('u', 'v', 'p', 'T')):
            fields[field] = round(0.01 * (b + 1) + 0.001 * f + 0.0001 * seed, 8)
        agg[key] = fields
    return {
        'seed': seed,
        'task': 'thermal',
        'protocol': {
            'task': 'thermal', 'seed': seed, 'epochs': epochs,
            'split_version': 'splits_thermal_v1_protocolA',
            'splits_config': v18_abs if seed >= 3 else v17_abs,
        },
        'curves': {'epochs': list(range(epochs))},
        'test': {'agg': agg},
        'checkpoint_best': 'checkpoint_best.pth',
    }


def _main_summary_doc(metric_docs, epochs=500):
    """Summary consistent with five seed metrics (sample std, like the real file)."""
    blocks = ('re', 'mae', 'max_err', 're_bd', 'mae_bd', 'max_bd')
    agg = {}
    for key in blocks:
        block = {'mean': {}, 'std': {}}
        for stat in ('mean', 'std'):
            for field in ('u', 'v', 'p', 'T'):
                data = [metric_docs[s]['test']['agg'][key][field] for s in metric_docs]
                mean = sum(data) / len(data)
                if stat == 'mean':
                    block[stat][field] = mean
                else:
                    var = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
                    block[stat][field] = var ** 0.5
        agg[key] = block
    return {'n_seeds': len(metric_docs),
            'seeds': sorted(metric_docs),
            'epochs': epochs,
            'agg': agg}


def _write_main_graph_result(ws):
    """Create a complete five-seed main graph result fixture."""
    base = os.path.join(ws, *pkg.MAIN_GRAPH_RESULT_DIR.split('/'))
    v17_abs = os.path.join(ws, *pkg.EXECUTION_CONFIG_WHITELIST[3].split('/'))
    v18_abs = os.path.join(ws, *pkg.EXECUTION_CONFIG_WHITELIST[4].split('/'))
    docs = {}
    for seed in pkg.MAIN_GRAPH_SEEDS:
        seed_dir = os.path.join(base, f'seed{seed}')
        docs[seed] = _main_metric_doc(seed, v17_abs, v18_abs)
        _write(os.path.join(seed_dir, f'metrics_seed{seed}.json'),
               json.dumps(docs[seed], indent=2) + '\n')
        _write(os.path.join(seed_dir, 'checkpoint_best.pth'), 'WEIGHTS')
    _write(os.path.join(base, 'summary.json'),
           json.dumps(_main_summary_doc(docs), indent=2) + '\n')
    return base


def _make_mock_workspace(tmp, *, metadata_complete=True, unknown_path=False,
                         credential=False, mock_root_name='mock_proj'):
    """Create a realistic mock project tree (mock workspace = .../返稿意见/返稿修订工作区)."""
    project = os.path.join(tmp, mock_root_name)
    ws = os.path.join(project, '返稿意见', '返稿修订工作区')
    os.makedirs(ws)
    # Codex metadata (zenodo/round2/) + legacy zenodo files
    zen = os.path.join(ws, 'zenodo', 'round2')
    _write(os.path.join(zen, '.zenodo.json'),
           '{"title": "fixture preparation", "upload_type": "software", '
           '"license": "mit"}\n')
    _write(os.path.join(zen, 'CITATION.cff'),
           'cff-version: 1.2.0\nlicense: MIT\ntitle: fixture\n')
    _write(os.path.join(zen, 'DATA_SOURCES.md'),
           'thermal original data: download URL recorded; sha256 CHECKSUM\n')
    if metadata_complete:
        _write(os.path.join(zen, 'README.md'),
               '# fixture\n历史环境与实测版本差异说明（fixture）\n')
    _write(os.path.join(ws, 'zenodo', 'LICENSE'),
           'MIT License\n\nCopyright (c) fixture authors\n\n'
           'Permission is hereby granted, free of charge, to any person obtaining a '
           'copy\nof this software and associated documentation files (the '
           '"Software"), to deal\nin the Software without restriction.\n\n'
           'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.\n')
    _write(os.path.join(ws, 'zenodo', 'LICENSE-DATA'), 'CC-BY fixture (must not ship)\n')
    _write(os.path.join(ws, 'zenodo', 'requirements.txt'), 'numpy==1.0  # historical\n')
    _write(os.path.join(ws, 'zenodo', 'requirements-extended.txt'), 'torch==1.0\n')
    _write(os.path.join(ws, 'zenodo', 'environment.yml'), 'name: fixture\n')
    # src (all .py except src/pipeline)
    _write(os.path.join(ws, 'src', '__init__.py'), '')
    _write(os.path.join(ws, 'src', 'core', '__init__.py'), '')
    _write(os.path.join(ws, 'src', 'fixture_mod.py'), 'VALUE = 1\n')
    if credential:
        _write(os.path.join(ws, 'src', 'leaky_key.py'),
               "API_KEY = 'AKIA" + 'ABCDEFGHIJKLMNOP' + "'\n")
    _write(os.path.join(ws, 'src', 'pipeline', 'old_asset.py'), 'OLD = True\n')
    # scripts: every required whitelist name as a stub (no local imports)
    for name in pkg.ALL_SCRIPT_NAMES:
        _write(os.path.join(ws, 'scripts', name + '.py'), 'VALUE = 1\n')
    # configs (splits_thermal carries real-looking protocol_A/... for equality checks)
    split_fixture = {
        'version': 'fixture-1.0',
        'split_seed': 123,
        'n_geometries': 6,
        'protocol_A': {'train': [0, 1, 2, 3], 'val': [4], 'test': [5]},
        'protocol_B': {'train': [0, 1, 2], 'val': [3, 4], 'test': [5]},
    }
    for name in pkg.CONFIG_FILES:
        if name == 'splits_thermal.json':
            _write(os.path.join(ws, 'configs', name),
                   json.dumps(split_fixture, indent=2) + '\n')
        else:
            _write(os.path.join(ws, 'configs', name), '{}\n')
    # three frozen split snapshots referenced by metric protocol.splits_config
    for rel in pkg.EXECUTION_CONFIG_WHITELIST:
        _write(os.path.join(ws, *rel.split('/')),
               json.dumps(split_fixture, indent=2) + '\n')
    # existing geometry spec JSON read by exp_r2_local_error / exp_r2_new_queries
    _write(os.path.join(ws, *pkg.GEOMETRY_SPEC_PATH.split('/')), json.dumps({
        'geometry_ids': [17, 25, 47],
        'counts': {'n_geometry': 3, 'n_points': 5000},
        'source_path': os.path.join(ws, 'results', 'thermal_geometry_spec.json'),
        'numbers': [0.5, 2, 3.25],
    }, indent=2) + '\n')
    # dynamic loader in an included script (fig_r2_mechanism -> scripts/exp_mechanism.py)
    _write_dynamic_loader(ws, 'scripts/exp_mechanism.py')
    # complete five-seed main graph result fixture (gated; mutated by gate tests)
    _write_main_graph_result(ws)
    # results: one fixture JSON per whitelisted dir + excluded weights/logs/diagnostic
    for dir_name in pkg.RESULTS_DIR_WHITELIST:
        base = os.path.join(ws, 'results', dir_name)
        fixture = {
            'case_id': 'fixture_%s' % dir_name,
            'seed': 3,
            'split_ids': {'train': [1, 2, 3], 'test': [7, 8]},
            'metrics': {'re': 0.125, 'mae': 1.5e-3, 'max_err': 4.0,
                        'std': 0.25, 'count': 10},
            'stats': [0.1, 0.2, 0.3],
            'source_path': os.path.join(ws, 'results', dir_name,
                                        'fixture_result.json'),
        }
        if dir_name == 'round2_thermal_symmetric':
            fixture['protocol'] = {
                'splits_config': os.path.join(
                    ws, *pkg.EXECUTION_CONFIG_WHITELIST[0].split('/')),
            }
            fixture['snapshot_refs'] = [
                os.path.join(ws, *rel.split('/'))
                for rel in pkg.EXECUTION_CONFIG_WHITELIST]
        if unknown_path and dir_name == pkg.RESULTS_DIR_WHITELIST[0]:
            fixture['private_path'] = '/home/private/person.csv'
        _write(os.path.join(base, 'fixture_result.json'),
               json.dumps(fixture, indent=2) + '\n')
        _write(os.path.join(base, 'seed0', 'checkpoint_best.pth'), 'WEIGHTS')
        _write(os.path.join(base, 'logs', 'run.log'), 'log line\n')
        _write(os.path.join(base, 'seed0', 'diagnostics.npz'), 'NPZ')
    # historical valid manuscript baselines (exact whitelist)
    for rel_dir in pkg.HISTORICAL_RESULT_DIRS:
        base = os.path.join(ws, *rel_dir.split('/'))
        _write(os.path.join(base, 'summary.json'), json.dumps({
            'case_id': 'fixture_historical',
            'numbers': [0.5, 2, 3.25],
            'source_path': '返稿意见\\返稿修订工作区\\' + rel_dir,
        }, indent=2) + '\n')
        _write(os.path.join(base, 'seed0', 'metrics_seed0.json'),
               json.dumps({'seed': 0, 're': 0.25}, indent=2) + '\n')
        _write(os.path.join(base, 'seed0', 'checkpoint_best.pth'), 'WEIGHTS')
    # four manuscript table-source manifests (relative-path values get normalised)
    for rel in pkg.AUDIT_SOURCE_MANIFEST_WHITELIST:
        doc = {
            'source': '返稿意见\\返稿修订工作区\\' + rel,
            'rel_posix': '返稿意见/返稿修订工作区/' + rel,
            'numbers': [1, 2.5, -3.0],
            'nested': {'order': [3, 2, 1]},
        }
        _write(os.path.join(ws, *rel.split('/')), json.dumps(doc, indent=2) + '\n')
    # decoys that must stay excluded: other audit files, removed baselines/ablations
    _write(os.path.join(ws, 'results', 'round2_audit', 'other_audit.json'), '{}\n')
    # new_queries: completed JSON included; registry candidate + npz excluded
    new_q = os.path.join(ws, 'results', 'round2_new_queries')
    _write(os.path.join(new_q, 'diagnostics.json'), json.dumps({
        'abs_path': os.path.join(ws, 'results', 'round2_new_queries'),
        'numbers': [1, 2.5],
    }, indent=2) + '\n')
    _write(os.path.join(new_q, 'registry_candidate.json'),
           '{"appended": false, "note": "internal candidate; must not ship"}\n')
    _write(os.path.join(new_q, 'geom_fixture.npz'), 'NPZ')
    for rel_dir in ('results/thermal_protocolA/feat_extract',
                    'results/thermal_protocolA/unif_therm',
                    'results/acoustic_protocolA/fourier_c',
                    'results/acoustic_protocolA/fourier_nc',
                    'results/acoustic_protocolA/unif_acous',
                    'results/acoustic_wp5_ablation/ab_nores',
                    'results/joint'):
        _write(os.path.join(ws, *rel_dir.split('/'), 'summary.json'), '{}\n')
    # data files that must never be packaged
    _write(os.path.join(ws, 'data', 'acoustic', 'case_fixture.npz'), 'DATA')
    _write(os.path.join(project, 'HeatTransferData.npy'), 'THERMAL')
    _write(os.path.join(ws, 'paper', 'main.tex'), 'MANUSCRIPT')
    _write(os.path.join(ws, 'docs', 'internal.md'), 'INTERNAL')
    return ws


def _mock_roots(mock_ws):
    """Allowed roots for a mock project tree (probe fixtures only)."""
    project = os.path.dirname(os.path.dirname(mock_ws))
    return [{'label': 'fixture_workspace', 'root': mock_ws.replace('\\', '/')},
            {'label': 'fixture_project_root', 'root': project.replace('\\', '/')}]


def _write_tar(path, entries):
    """entries: list of (name, type, payload_or_linkname)."""
    with tarfile.open(path, 'w:gz') as tar:
        for name, kind, payload in entries:
            info = tarfile.TarInfo(name)
            if kind == 'file':
                data = payload.encode('utf-8') if isinstance(payload, str) else payload
                info.size = len(data)
                import io as _io
                tar.addfile(info, _io.BytesIO(data))
            elif kind == 'symlink':
                info.type = tarfile.SYMTYPE
                info.linkname = payload
                tar.addfile(info)
            elif kind == 'hardlink':
                info.type = tarfile.LNKTYPE
                info.linkname = payload
                tar.addfile(info)
            else:
                raise AssertionError(kind)


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def _count_json_under(rel_dir, exclude=frozenset()):
    parts = rel_dir.replace('\\', '/').split('/')
    base = os.path.join(WS, *parts)
    n = 0
    for root, dirs, files in os.walk(base):
        for f in files:
            if not f.endswith('.json'):
                continue
            rel = os.path.relpath(os.path.join(root, f), WS).replace(os.sep, '/')
            if rel in exclude:
                continue
            n += 1
    return n


def test_real_plan_whitelist():
    plan = pkg.collect_plan(WS)
    # Missing frozen-split sources are expected only until Codex recovers them;
    # anything else missing is a real failure. The exposure is reported as pending.
    missing_paths = {item['path'] for item in plan['missing_required']}
    pending_snapshots = sorted(missing_paths & set(pkg.EXECUTION_CONFIG_WHITELIST))
    assert missing_paths <= set(pkg.EXECUTION_CONFIG_WHITELIST), \
        sorted(missing_paths - set(pkg.EXECUTION_CONFIG_WHITELIST))
    assert plan['missing_helpers'] == [], plan['missing_helpers']
    # independent count of src *.py outside src/pipeline
    n_src = 0
    for root, dirs, files in os.walk(os.path.join(WS, 'src')):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        rel = os.path.relpath(root, WS).replace(os.sep, '/')
        if rel == 'src/pipeline' or rel.startswith('src/pipeline/'):
            continue
        n_src += sum(1 for f in files if f.endswith('.py'))
    assert plan['counts'].get('source', 0) == n_src, (plan['counts'], n_src)
    assert plan['counts'].get('script', 0) == len(pkg.ALL_SCRIPT_NAMES)
    assert plan['counts'].get('config', 0) == 3
    # dynamic independent counts: round2 (minus excluded internal candidates)
    # + historical manuscript baselines
    n_round2 = sum(_count_json_under(os.path.join('results', d),
                                     exclude=pkg.RESULTS_JSON_EXCLUDE)
                   for d in pkg.RESULTS_DIR_WHITELIST)
    n_historical = sum(_count_json_under(d) for d in pkg.HISTORICAL_RESULT_DIRS)
    assert n_round2 > 0 and n_historical > 0, (n_round2, n_historical)
    assert plan['counts'].get('result_json', 0) == n_round2 + n_historical, (
        plan['counts'], n_round2, n_historical)
    # new_queries: completed JSON included, internal registry candidate + npz excluded
    new_q_dir = os.path.join(WS, 'results', 'round2_new_queries')
    if os.path.isdir(new_q_dir):
        arcs = {m['arcname'] for m in plan['members']}
        assert 'results/round2_new_queries/registry_candidate.json' not in arcs
        assert not any(a.startswith('results/round2_new_queries/') and a.endswith('.npz')
                       for a in arcs)
        assert plan['counts'].get('result_json', 0) > 0
        assert any(a.startswith('results/round2_new_queries/') and a.endswith('.json')
                   for a in arcs), 'new_queries completed JSON missing from plan'
    assert 'scripts/exp_r2_new_queries.py' in {
        m['arcname'] for m in plan['members'] if m['kind'] == 'script'} or \
        not os.path.isfile(os.path.join(WS, 'scripts', 'exp_r2_new_queries.py'))
    # four-only audit source manifest whitelist; missing files must be recorded
    present_manifests = [rel for rel in pkg.AUDIT_SOURCE_MANIFEST_WHITELIST
                         if os.path.isfile(os.path.join(WS, *rel.split('/')))]
    missing_manifests = [rel for rel in pkg.AUDIT_SOURCE_MANIFEST_WHITELIST
                         if rel not in present_manifests]
    assert plan['source_manifests'] == present_manifests
    assert plan['missing_source_manifests'] == missing_manifests
    assert plan['counts'].get('source_manifest', 0) == len(present_manifests)
    # requested-but-missing acoustic ablation path is recorded, not fabricated
    requested = os.path.join(WS, 'results', 'acoustic_protocolA', 'ab_nofourier')
    actual = os.path.join(WS, 'results', 'acoustic_wp5_ablation', 'ab_nofourier')
    if not os.path.exists(requested) and os.path.exists(actual):
        assert any(item['requested'] == 'results/acoustic_protocolA/ab_nofourier'
                   for item in plan['requested_missing']), plan['requested_missing']
    # sensitive required inputs: geometry spec + dynamically loaded mechanism helper
    # + frozen split snapshots (a not-yet-recovered snapshot is reported as pending
    # here and blocks the closure check below, never silently accepted)
    arcs = {member['arcname'] for member in plan['members']}
    for arc in pkg.REQUIRED_PACKAGE_ARCS:
        if arc in pending_snapshots:
            continue
        assert arc in arcs, f'required sensitive input missing from plan: {arc}'
    assert set(plan['missing_required_arcs']) == set(pending_snapshots), \
        plan['missing_required_arcs']
    assert plan['counts'].get('geometry_spec', 0) == 1, plan['counts']
    assert 'scripts/exp_mechanism.py' in arcs
    assert 'results/thermal_geometry_spec.json' in arcs
    # dynamic-load closure: fig_r2_mechanism -> scripts/exp_mechanism.py (included)
    dynamic_refs = plan['dynamic_helper_references']
    resolved = {(item['loaded_by'], item['resolved_file'])
                for item in dynamic_refs if item['included']}
    assert ('scripts/fig_r2_mechanism.py', 'scripts/exp_mechanism.py') in resolved, \
        dynamic_refs
    assert plan['missing_dynamic_helpers'] == [], plan['missing_dynamic_helpers']
    assert plan['unresolved_dynamic_helpers'] == [], plan['unresolved_dynamic_helpers']
    assert plan['missing_metric_split_snapshots'] == [], \
        plan['missing_metric_split_snapshots']
    assert plan['unresolved_metric_split_references'] == [], \
        plan['unresolved_metric_split_references']
    assert set(plan['metric_split_references']) <= (
        {'configs/splits_thermal.json'} | set(pkg.EXECUTION_CONFIG_WHITELIST))
    assert plan['required_arcs'] == list(pkg.REQUIRED_PACKAGE_ARCS)
    exec_arcs = {a for a in arcs if 'round2_execution_2026102' in a}
    assert exec_arcs == set(pkg.EXECUTION_CONFIG_WHITELIST) - set(pending_snapshots), \
        exec_arcs
    assert plan['counts'].get('execution_config', 0) == \
        len(set(pkg.EXECUTION_CONFIG_WHITELIST) - set(pending_snapshots))
    # current main five-seed graph result is gated: partial -> zero members + pending
    main = plan['main_graph_result']
    main_arcs = sorted(a for a in arcs if a.startswith(main['dir'] + '/'))
    if main['status'] == 'complete':
        assert main_arcs == sorted(main['approved_files']), main_arcs
        assert plan['pending_main_result'] is None
        assert plan['counts'].get('main_result_json', 0) == len(main['approved_files'])
    else:
        assert main_arcs == [], f'partial main result leaked into plan: {main_arcs}'
        assert plan['pending_main_result'] is not None
        assert main['status'] in ('incomplete', 'absent')
    # legacy 3-seed graph main comparison (results/baselines) is never packaged
    assert not any(a.startswith('results/baselines/') for a in arcs)
    # current Table 5 (five-seed graph comparison) must be the packaged source
    table5 = 'results/round2_geometry/completed_comparison_graph_five_seed.json'
    if os.path.isfile(os.path.join(WS, *table5.split('/'))):
        assert table5 in arcs, 'current five-seed Table 5 source missing from plan'
        with open(os.path.join(WS, *table5.split('/')), 'r', encoding='utf-8') as fh:
            table5_doc = json.load(fh)
        assert table5_doc.get('geometry_job') == 50920
        for protocol, models in (table5_doc.get('protocols') or {}).items():
            for model, block in models.items():
                assert block.get('n_seeds') == 5 and list(block.get('seeds') or []) \
                    == [0, 1, 2, 3, 4], f'{protocol}/{model} is not five-seed'
    # forbidden package members
    allowed_audit = set(pkg.AUDIT_SOURCE_MANIFEST_WHITELIST)
    for member in plan['members']:
        arc = member['arcname']
        assert not arc.startswith(('data/', 'docs/', 'paper/', 'ccfa-',
                                   'sbatch/', 'src/pipeline', 'zenodo/', 'results/joint/')), arc
        assert not arc.endswith(('.pth', '.pt', '.npz', '.out', '.err')), arc
        assert arc not in pkg.RESULTS_JSON_EXCLUDE, f'excluded result JSON in plan: {arc}'
        if arc.startswith('results/round2_audit/'):
            assert arc in allowed_audit, f'excluded audit file in plan: {arc}'
        if member['kind'] == 'result_json':
            assert arc.endswith('.json') and arc.startswith('results/')
        for forbidden in ('results/thermal_protocolA/feat_extract/',
                          'results/thermal_protocolA/unif_therm/',
                          'results/acoustic_protocolA/fourier_c/',
                          'results/acoustic_protocolA/fourier_nc/',
                          'results/acoustic_protocolA/unif_acous/'):
            assert not arc.startswith(forbidden), f'excluded baseline in plan: {arc}'
    return {'n_members': len(plan['members']),
            'counts': plan['counts'],
            'results_counts': plan['results_counts'],
            'historical_counts': plan['historical_counts'],
            'source_manifests': plan['source_manifests'],
            'missing_source_manifests': plan['missing_source_manifests'],
            'pending_required_snapshots': pending_snapshots,
            'missing_metric_split_snapshots': plan['missing_metric_split_snapshots'],
            'metric_split_references': plan['metric_split_references'],
            'main_graph_status': plan['main_graph_result']['status'],
            'pending_main_result': bool(plan['pending_main_result']),
            'requested_missing': plan['requested_missing'],
            'missing_metadata': plan['missing_metadata'],
            'helper_additions': plan['helper_additions']}


def _iter_splits_config(doc, pointer=''):
    """Yield (json_pointer, value) for every protocol.splits_config field."""
    if isinstance(doc, dict):
        for key, value in doc.items():
            path = f'{pointer}.{key}' if pointer else str(key)
            if key == 'splits_config' and isinstance(value, str):
                yield path, value
            else:
                yield from _iter_splits_config(value, path)
    elif isinstance(doc, list):
        for index, value in enumerate(doc):
            yield from _iter_splits_config(value, f'{pointer}[{index}]')


def _iter_all_strings(doc):
    if isinstance(doc, dict):
        for value in doc.values():
            yield from _iter_all_strings(value)
    elif isinstance(doc, list):
        for value in doc:
            yield from _iter_all_strings(value)
    elif isinstance(doc, str):
        yield doc


def test_frozen_split_snapshots():
    """Real metrics must have their protocol.splits_config snapshots packaged and
    the snapshot protocol_A train/val/test must actually equal the canonical config."""
    plan = pkg.collect_plan(WS)
    arcs = {member['arcname']: member for member in plan['members']}
    roots = pkg.default_known_roots()
    missing_sources = [rel for rel in pkg.EXECUTION_CONFIG_WHITELIST
                       if not os.path.isfile(os.path.join(WS, *rel.split('/')))]
    if missing_sources:
        raise ProbeBlocked(
            'frozen split snapshot sources not yet recovered by Codex: '
            + ', '.join(missing_sources) + '; closure cannot be verified until '
            'these exact files exist (no fabrication from the canonical config)')
    # discover references from the real packaged metric JSONs (not a constant list)
    refs = {}
    for member in plan['members']:
        if member['kind'] not in ('result_json', 'main_result_json'):
            continue
        with open(member['src'], 'r', encoding='utf-8') as fh:
            doc = json.load(fh)
        for pointer, value in _iter_splits_config(doc):
            matched, rel = pkg._match_known_root(value, roots)
            assert matched is not None, \
                f'unknown private splits_config ref in {member["arcname"]}{pointer}'
            refs[rel] = refs.get(rel, 0) + 1
    allowed = {'configs/splits_thermal.json'} | set(pkg.EXECUTION_CONFIG_WHITELIST)
    assert set(refs) <= allowed, sorted(set(refs) - allowed)
    for rel in pkg.EXECUTION_CONFIG_WHITELIST:
        assert rel in arcs, f'required split snapshot missing from plan: {rel}'
    # v17/v18 are required dependencies; once the five-seed main metrics are in the
    # plan they must be actually referenced by those metrics as well.
    main = plan['main_graph_result']
    if main['status'] == 'complete':
        for rel in pkg.EXECUTION_CONFIG_WHITELIST[3:]:
            assert refs.get(rel, 0) > 0, \
                f'{rel} is not referenced by the packaged five-seed main metrics'
    canonical_path = os.path.join(WS, 'configs', 'splits_thermal.json')
    with open(canonical_path, 'r', encoding='utf-8') as fh:
        canonical = json.load(fh)
    canonical_sha = _sha_file(canonical_path)
    snapshot_stats = {}
    for rel in pkg.EXECUTION_CONFIG_WHITELIST:
        member = arcs[rel]
        path = os.path.join(WS, *rel.split('/'))
        with open(path, 'r', encoding='utf-8') as fh:
            snapshot = json.load(fh)
        # byte-identical copy: no path normalisation of the config file itself
        assert member['sanitized'] is False, rel
        for key in ('train', 'val', 'test'):
            assert snapshot['protocol_A'][key] == canonical['protocol_A'][key], \
                f'{rel} protocol_A.{key} differs from configs/splits_thermal.json'
        assert snapshot.get('split_seed') == canonical.get('split_seed'), rel
        assert snapshot.get('n_geometries') == canonical.get('n_geometries'), rel
        if rel in pkg.EXECUTION_CONFIG_WHITELIST[3:]:
            assert _sha_file(path) == canonical_sha, \
                f'{rel} bytes differ from the canonical splits_thermal.json'
        private = [value for value in _iter_all_strings(snapshot)
                   if value.startswith('/') or re.match(r'^[A-Za-z]:[\\/]', value)]
        assert not private, f'{rel} contains private absolute paths: {private[:3]}'
        snapshot_stats[rel] = {'references': refs.get(rel, 0),
                               'protocol_A_equal': True,
                               'byte_identical_copy': True,
                               'canonical_sha_equal': _sha_file(path) == canonical_sha}
    # only the exact execution-dir files, never the whole directories
    exec_members = {arc for arc in arcs if 'round2_execution_2026102' in arc}
    assert exec_members == set(pkg.EXECUTION_CONFIG_WHITELIST), exec_members
    assert 'configs/splits_thermal.json' in arcs
    return {'metric_splits_config_refs': refs,
            'snapshots': snapshot_stats,
            'canonical_member_included': True,
            'main_graph_status': main['status']}


def test_metric_derived_split_dependencies():
    """The metric's own protocol.splits_config fields decide the snapshot paths.

    This is read from the metrics (no hardcoded date assumption) and the source
    fields are never rewritten. If all five seeds are present, the union must be
    exactly the two recovered frozen snapshots (v17 for seeds 0-2, v18 for 3/4).
    """
    main_dir = os.path.join(WS, *pkg.MAIN_GRAPH_RESULT_DIR.split('/'))
    available = {}
    for seed in pkg.MAIN_GRAPH_SEEDS:
        metrics_path = os.path.join(main_dir, f'seed{seed}',
                                    f'metrics_seed{seed}.json')
        if not os.path.isfile(metrics_path):
            continue
        with open(metrics_path, 'r', encoding='utf-8') as fh:
            doc = json.load(fh)
        raw = (doc.get('protocol') or {}).get('splits_config')
        assert isinstance(raw, str) and raw, f'seed{seed} splits_config missing'
        # source fields stay absolute/verbatim (we never rewrite the source)
        assert raw.startswith('/') or re.match(r'^[A-Za-z]:[\\/]', raw), raw
        assert '20261021_v17' not in raw, f'seed{seed} still references 20261021_v17'
        rel = pkg._resolve_metric_split_path(raw, WS)[0]
        assert rel is not None, f'seed{seed} splits_config is not a usable path'
        assert rel in pkg.EXECUTION_CONFIG_WHITELIST, (seed, rel)
        available[seed] = rel
    assert available, 'no main seed metrics available to read splits_config'
    current_v17 = pkg.EXECUTION_CONFIG_WHITELIST[3]
    current_v18 = pkg.EXECUTION_CONFIG_WHITELIST[4]
    assert current_v17 == ('results/round2_execution_20261022_v17/configs/'
                           'splits_thermal.json')
    assert current_v18 == ('results/round2_execution_20261022_v18/configs/'
                           'splits_thermal.json')
    expected_by_seed = {seed: (current_v17 if seed <= 2 else current_v18)
                        for seed in pkg.MAIN_GRAPH_SEEDS}
    for seed, rel in available.items():
        assert rel == expected_by_seed[seed], (seed, rel)
    plan = pkg.collect_plan(WS)
    arcs = {m['arcname'] for m in plan['members']}
    assert current_v17 in arcs and current_v18 in arcs
    assert plan['missing_metric_split_snapshots'] == [], \
        plan['missing_metric_split_snapshots']
    assert plan['unresolved_metric_split_references'] == [], \
        plan['unresolved_metric_split_references']
    if len(available) == len(pkg.MAIN_GRAPH_SEEDS):
        assert set(available.values()) == {current_v17, current_v18}, available
        assert sum(1 for rel in available.values() if rel == current_v17) == 3
        assert sum(1 for rel in available.values() if rel == current_v18) == 2
        assert plan['main_graph_result']['status'] == 'complete'
        assert set(plan['metric_split_references']) >= {current_v17, current_v18}
    else:
        assert plan['main_graph_result']['status'] != 'complete'
        assert not any(a.startswith(pkg.MAIN_GRAPH_RESULT_DIR + '/')
                       for a in arcs)
    # authoritative-source negative test: wrong-but-whitelisted snapshot is refused
    with tempfile.TemporaryDirectory(prefix='r2_pkg_probe_metric_split_') as tmp:
        ws = _make_mock_workspace(os.path.join(tmp, 'wrong_snapshot'))
        metrics_path = os.path.join(ws, *pkg.MAIN_GRAPH_RESULT_DIR.split('/'),
                                    'seed0', 'metrics_seed0.json')
        doc = json.load(open(metrics_path, encoding='utf-8'))
        doc['protocol']['splits_config'] = os.path.join(
            ws, *pkg.EXECUTION_CONFIG_WHITELIST[0].split('/'))
        _write(metrics_path, json.dumps(doc, indent=2) + '\n')
        plan_bad = pkg.collect_plan(ws, roots=_mock_roots(ws))
        assert plan_bad['main_graph_result']['status'] == 'incomplete', \
            plan_bad['main_graph_result']
        out = os.path.join(tmp, 'wrong_snapshot', 'out')
        try:
            pkg.build_package(ws, out, 'wrong_split', roots=_mock_roots(ws))
            raise AssertionError('wrong whitelisted snapshot reference was accepted')
        except pkg.PackageRefused as exc:
            assert 'frozen split snapshot whitelist' in str(exc) \
                or 'partial main result' in str(exc), str(exc)
        assert not os.path.exists(os.path.join(out, 'wrong_split.tar.gz'))
        # unknown private path source -> refused, source value never echoed
        ws = _make_mock_workspace(os.path.join(tmp, 'unknown_snapshot'))
        metrics_path = os.path.join(ws, *pkg.MAIN_GRAPH_RESULT_DIR.split('/'),
                                    'seed4', 'metrics_seed4.json')
        doc = json.load(open(metrics_path, encoding='utf-8'))
        doc['protocol']['splits_config'] = '/home/private/secret_splits.json'
        _write(metrics_path, json.dumps(doc, indent=2) + '\n')
        plan_bad = pkg.collect_plan(ws, roots=_mock_roots(ws))
        assert plan_bad['main_graph_result']['status'] == 'incomplete'
        assert any('not a usable workspace path' in item
                   for item in plan_bad['main_graph_result']['pending_issues'])
        out = os.path.join(tmp, 'unknown_snapshot', 'out')
        try:
            pkg.build_package(ws, out, 'unknown_split', roots=_mock_roots(ws))
            raise AssertionError('unknown private splits_config was accepted')
        except pkg.PackageRefused as exc:
            assert 'secret_splits' not in str(exc), 'private path echoed'
    return {'metrics_read': {f'seed{k}': v for k, v in available.items()},
            'expected_v17': current_v17,
            'expected_v18': current_v18,
            'five_seed_complete': len(available) == len(pkg.MAIN_GRAPH_SEEDS),
            'wrong_snapshot_refused': True,
            'unknown_private_path_refused': True}


def test_license_alignment_preflight():
    """Zenodo `.zenodo.json`, CITATION.cff and the existing MIT LICENSE must agree.

    The old real mismatch (Zenodo cc-by-4.0 vs CFF MIT/LICENSE MIT) is reproduced
    as a fixture and must be refused; the current MIT metadata must pass. This is
    a consistency check only and grants no new licence or data rights.
    """
    plan = pkg.collect_plan(WS)
    checks = pkg.validate_metadata_inputs(plan)
    aligned = checks['license_alignment']
    assert aligned['aligned'] is True, aligned
    assert aligned['zenodo_license'] == 'mit', aligned
    assert aligned['cff_licenses'] and all(item == 'mit'
                                          for item in aligned['cff_licenses']), aligned
    assert aligned['mit_notice_present'] is True, aligned
    assert aligned['errors'] == [], aligned
    note = aligned['note'].lower()
    assert 'no new licence' in note and 'third-party data' in note, aligned['note']

    def _variant(tmp, name, *, zenodo_license=None, cff_license=None,
                 license_text=None):
        ws = _make_mock_workspace(os.path.join(tmp, name))
        zen = os.path.join(ws, 'zenodo', 'round2')
        if zenodo_license is not None:
            _write(os.path.join(zen, '.zenodo.json'), json.dumps({
                'title': 'fixture', 'upload_type': 'software',
                'license': zenodo_license}) + '\n')
        if cff_license is not None:
            _write(os.path.join(zen, 'CITATION.cff'),
                   f'cff-version: 1.2.0\nlicense: {cff_license}\ntitle: fixture\n')
        if license_text is not None:
            _write(os.path.join(ws, 'zenodo', 'LICENSE'), license_text)
        return ws

    variants = {
        'old_zenodo_ccby_mismatch': {
            'zenodo_license': 'cc-by-4.0'},
        'cff_ccby_mismatch': {
            'cff_license': 'CC-BY-4.0'},
        'missing_cff_license': {
            'cff_license': ''},
        'non_mit_license_notice': {
            'license_text': 'CC-BY-4.0\n\nThis fixture is not the MIT code licence.\n'},
    }
    details = {'real_aligned': True,
               'zenodo_license': aligned['zenodo_license'],
               'cff_license': aligned['cff_licenses']}
    with tempfile.TemporaryDirectory(prefix='r2_pkg_probe_license_') as tmp:
        for name, kwargs in variants.items():
            ws = _variant(tmp, name, **kwargs)
            plan_bad = pkg.collect_plan(ws, roots=_mock_roots(ws))
            checks_bad = pkg.validate_metadata_inputs(plan_bad)
            aligned_bad = checks_bad['license_alignment']
            assert aligned_bad['aligned'] is False, (name, aligned_bad)
            assert aligned_bad['errors'], name
            out = os.path.join(tmp, name, 'out')
            try:
                pkg.build_package(ws, out, name, roots=_mock_roots(ws))
                raise AssertionError(f'{name}: mismatched licence declarations were '
                                     f'not refused')
            except pkg.PackageRefused as exc:
                message = str(exc)
                assert ('licence preflight' in message
                        or 'MIT' in message), (name, message)
            assert not os.path.exists(os.path.join(out, name + '.tar.gz'))
            details[name] = aligned_bad['errors']
    return details


def test_list_only_no_write():
    with tempfile.TemporaryDirectory(prefix='r2_pkg_probe_list_') as tmp:
        out = os.path.join(tmp, 'would_be_package')
        result = pkg.list_only(WS, out, 'probe_list_only_fixture')
        assert not os.path.exists(out), '--list-only created the output dir'
        text = result['text']
        assert 'would refuse' in text
        assert 'historical manuscript baseline JSON by directory:' in text
        assert 'excluded result JSON (internal registry candidates):' in text
        for rel in pkg.RESULTS_JSON_EXCLUDE:
            assert rel in text, f'{rel} not reported as excluded result JSON'
        assert 'sensitive required inputs (missing => refuse):' in text
        for arc in pkg.REQUIRED_PACKAGE_ARCS:
            assert arc in text, f'{arc} not reported as required input'
        assert 'frozen split snapshots' in text
        for rel in pkg.EXECUTION_CONFIG_WHITELIST:
            assert rel in text, f'{rel} not reported as frozen split snapshot'
        assert 'main five-seed graph result gate:' in text
        assert pkg.MAIN_GRAPH_RESULT_DIR in text
        assert 'metadata licence preflight (consistency only; no new grant):' in text
        assert 'aligned            : True' in text
        assert 'metric protocol.splits_config references' in text
        assert 'dynamic source-load closure' in text
        assert 'scripts/exp_mechanism.py' in text
        assert 'manuscript table source manifests:' in text
        missing = result['plan']['missing_metadata']
        for name in missing:
            assert name in text, f'{name} not reported as missing'
        for rel in result['plan']['missing_source_manifests']:
            assert rel in text, f'{rel} not reported as missing source manifest'
        if result['plan']['requested_missing']:
            assert 'requested-but-missing paths' in text
        expect_raises(pkg.PackageRefused,
                      lambda: pkg.list_only(WS, out, '../evil'),
                      'invalid package name')
    return {'missing_metadata_reported': missing,
            'source_manifests': result['plan']['source_manifests'],
            'missing_source_manifests': result['plan']['missing_source_manifests'],
            'requested_missing': result['plan']['requested_missing'],
            'wrote_output_dir': False}


def test_sanitize_known_and_numeric():
    roots = pkg.default_known_roots(WS)
    with tempfile.TemporaryDirectory(prefix='r2_pkg_probe_sanitize_') as tmp:
        mock_project = os.path.join(tmp, 'proj')
        mock_ws = os.path.join(mock_project, '返稿意见', '返稿修订工作区')
        os.makedirs(mock_ws)
        doc = {
            'case_id': 'fixture_case_01',
            'seed': 4,
            'split_ids': {'train': [1, 2, 3], 'val': [4], 'test': [5, 6, 7]},
            'numbers': [0.125, 1.5e-3, 4.0, 10, True, None],
            'nested': {'stats': {'re': 0.25, 'std': 0.5}},
            'source_path': os.path.join(mock_ws, 'results', 'x.json'),
            'dataset': os.path.join(mock_project, 'HeatTransferData.npy'),
            'deployed': ('/deployed/PINN-AP/返稿意见/'
                         '返稿修订工作区/results/round2_mechanism/x.json'),
            'deployed_data': ('/deployed/PINN-AP/'
                              'HeatTransferData.npy'),
            'url': 'https://example.org/data/HeatTransferData.npy',
        }
        mock_roots = [
            {'label': 'workspace', 'root': mock_ws.replace('\\', '/')},
            {'label': 'project_root', 'root': mock_project.replace('\\', '/')},
        ] + [r for r in roots if r['label'] in ('workspace_linux', 'repo_root_linux')]
        sanitized, info = pkg.sanitize_json_document(doc, mock_roots)
        assert info['blocked'] == [], info['blocked']
        pkg.assert_json_equivalent(doc, sanitized, info['rewrites'])
        assert sanitized['source_path'] == 'results/x.json', sanitized['source_path']
        assert sanitized['dataset'] == 'HeatTransferData.npy'
        assert sanitized['deployed'] == 'results/round2_mechanism/x.json'
        assert sanitized['deployed_data'] == 'HeatTransferData.npy'
        assert sanitized['url'] == doc['url'], 'URL must not be treated as a path'
        # numbers / case ids / split ids / seeds / stats identical
        for key in ('case_id', 'seed', 'split_ids', 'numbers', 'nested'):
            assert sanitized[key] == doc[key], key
        assert json.dumps(sanitized['numbers']) == json.dumps(doc['numbers'])
        # only the documented repo/workspace private roots are accepted
        win_repo = pkg.ALLOWED_REPO_ROOTS[0][1]
        linux_repo = pkg.ALLOWED_REPO_ROOTS[1][1]
        allowed_doc = {
            'win_repo': win_repo + '/HeatTransferData.npy',
            'win_ws': win_repo + '/返稿意见/返稿修订工作区/results/x.json',
            'linux_repo': linux_repo + '/HeatTransferData.npy',
            'linux_ws': linux_repo + '/返稿意见/返稿修订工作区/results/x.json',
        }
        allowed, allowed_info = pkg.sanitize_json_document(allowed_doc)
        assert allowed_info['blocked'] == [], allowed_info['blocked']
        assert allowed['win_repo'] == 'HeatTransferData.npy'
        assert allowed['win_ws'] == 'results/x.json'
        assert allowed['linux_repo'] == 'HeatTransferData.npy'
        assert allowed['linux_ws'] == 'results/x.json'
        return {'rewrites': len(info['rewrites']),
                'deployed_roots_used': sorted({r['root'] for r in info['rewrites']}),
                'allowed_private_roots': sorted({r['root']
                                                 for r in allowed_info['rewrites']}),
                'blocked': 0}


def test_relative_prefix_normalization():
    doc = {
        'win_backslash': '返稿意见\\返稿修订工作区\\results\\thermal_protocolA\\'
                         'mlp\\summary.json',
        'posix': '返稿意见/返稿修订工作区/results/acoustic_protocolA/fno/summary.json',
        'with_dot': './返稿意见\\返稿修订工作区\\results\\x.json',
        'unrelated_relative': 'other/results/x.json',
        'numbers': [1, 2.5, -3],
        'nested': {'order': [3, 2, 1]},
        'case_id': 'fixture_case',
    }
    sanitized, info = pkg.sanitize_json_document(doc)
    assert info['blocked'] == [], info['blocked']
    pkg.assert_json_equivalent(doc, sanitized, info['rewrites'])
    assert sanitized['win_backslash'] == 'results/thermal_protocolA/mlp/summary.json'
    assert sanitized['posix'] == 'results/acoustic_protocolA/fno/summary.json'
    assert sanitized['with_dot'] == 'results/x.json'
    assert sanitized['unrelated_relative'] == 'other/results/x.json'
    assert sanitized['numbers'] == doc['numbers']
    assert sanitized['nested'] == doc['nested']
    assert json.dumps(sanitized['numbers']) == json.dumps(doc['numbers'])
    assert len(info['rewrites']) == 3, info['rewrites']
    return {'rewrites': [r['field'] for r in info['rewrites']],
            'unrelated_relative_unchanged': True, 'blocked': 0}


def test_real_json_release_copy():
    """Sanitise every real whitelisted result JSON in memory (no package written)."""
    plan = pkg.collect_plan(WS)
    roots = pkg.default_known_roots()
    n = 0
    relative_rewrites = 0
    absolute_rewrites = 0
    blocked = []
    for member in plan['members']:
        if member['kind'] not in ('result_json', 'source_manifest',
                                  'geometry_spec', 'main_result_json'):
            continue
        with open(member['src'], 'r', encoding='utf-8') as fh:
            original = json.load(fh)
        sanitized, info = pkg.sanitize_json_document(original, roots)
        if info['blocked']:
            blocked.append({'file': member['arcname'],
                            'blocked': info['blocked']})
            continue
        pkg.assert_json_equivalent(original, sanitized, info['rewrites'])
        n += 1
        for item in info['rewrites']:
            if item['root'] == 'workspace_relative_prefix':
                relative_rewrites += 1
            else:
                absolute_rewrites += 1
    assert not blocked, blocked[:3]
    expected_total = (plan['counts']['result_json']
                      + plan['counts'].get('source_manifest', 0)
                      + plan['counts'].get('geometry_spec', 0)
                      + plan['counts'].get('main_result_json', 0))
    assert n == expected_total, (n, plan['counts'])
    assert relative_rewrites >= 20, relative_rewrites
    assert absolute_rewrites > 400, absolute_rewrites
    return {'n_json_sanitised': n,
            'n_result_json': plan['counts']['result_json'],
            'n_source_manifest': plan['counts'].get('source_manifest', 0),
            'n_geometry_spec': plan['counts'].get('geometry_spec', 0),
            'relative_prefix_rewrites': relative_rewrites,
            'allowed_absolute_rewrites': absolute_rewrites,
            'blocked': 0,
            'numeric_and_ids_unchanged': True}


def test_dynamic_helper_detection():
    """Dynamic importlib loads must be detected beyond a static whitelist."""
    with tempfile.TemporaryDirectory(prefix='r2_pkg_probe_dyn_') as tmp:
        # positive: mock fig_r2_mechanism dynamically loads scripts/exp_mechanism.py
        ws = _make_mock_workspace(os.path.join(tmp, 'ok'))
        plan = pkg.collect_plan(ws, roots=_mock_roots(ws))
        assert plan['missing_dynamic_helpers'] == [], plan['missing_dynamic_helpers']
        assert plan['unresolved_dynamic_helpers'] == [], \
            plan['unresolved_dynamic_helpers']
        resolved = {(item['loaded_by'], item['resolved_file'])
                    for item in plan['dynamic_helper_references'] if item['included']}
        assert ('scripts/fig_r2_mechanism.py', 'scripts/exp_mechanism.py') in resolved

        # negative: dynamic target exists but is NOT in the package plan
        ws_bad = _make_mock_workspace(os.path.join(tmp, 'missing'))
        _write(os.path.join(ws_bad, 'scripts', 'hidden_helper.py'), 'VALUE = 1\n')
        _write_dynamic_loader(ws_bad, 'scripts/hidden_helper.py')
        plan_bad = pkg.collect_plan(ws_bad, roots=_mock_roots(ws_bad))
        needed = {item['needed_file'] for item in plan_bad['missing_dynamic_helpers']}
        assert 'scripts/hidden_helper.py' in needed, plan_bad['missing_dynamic_helpers']
        out = os.path.join(tmp, 'missing', 'out')
        try:
            pkg.build_package(ws_bad, out, 'dyn_missing', roots=_mock_roots(ws_bad))
            raise AssertionError('missing dynamic helper was not refused')
        except pkg.PackageRefused as exc:
            message = str(exc)
            assert 'dynamically loaded helper' in message and 'hidden_helper.py' in message
        assert not os.path.exists(os.path.join(out, 'dyn_missing.tar.gz'))

        # negative: dynamic target does not exist at all -> unresolved, refused
        ws_ghost = _make_mock_workspace(os.path.join(tmp, 'ghost'))
        _write_dynamic_loader(ws_ghost, 'scripts/does_not_exist.py')
        plan_ghost = pkg.collect_plan(ws_ghost, roots=_mock_roots(ws_ghost))
        assert plan_ghost['unresolved_dynamic_helpers'], plan_ghost
        out = os.path.join(tmp, 'ghost', 'out')
        try:
            pkg.build_package(ws_ghost, out, 'dyn_ghost', roots=_mock_roots(ws_ghost))
            raise AssertionError('unresolved dynamic loader was not refused')
        except pkg.PackageRefused as exc:
            assert 'could not be resolved' in str(exc)
        assert not os.path.exists(os.path.join(out, 'dyn_ghost.tar.gz'))
    return {'positive_resolved': 'fig_r2_mechanism -> scripts/exp_mechanism.py',
            'missing_non_whitelisted_detected': 'scripts/hidden_helper.py',
            'unresolved_detected': 'scripts/does_not_exist.py',
            'refused_without_partial_package': True}


def test_main_five_seed_gate():
    """Partial main graph results must never enter the package; complete five-seed
    data must be included after metrics/weights/summary consistency checks."""
    details = {}
    # real workspace (may still be running seeds 3/4): adaptive, not fabricated
    plan = pkg.collect_plan(WS)
    main = plan['main_graph_result']
    main_arcs = sorted(m['arcname'] for m in plan['members']
                       if m['arcname'].startswith(main['dir'] + '/'))
    arcs = {m['arcname'] for m in plan['members']}
    if main['status'] == 'complete':
        assert len(main['approved_files']) == 6, main['approved_files']
        assert main_arcs == sorted(main['approved_files'])
        assert plan['counts'].get('main_result_json', 0) == 6
        assert plan['pending_main_result'] is None
        details['real'] = 'complete and gated-in'
    else:
        assert main['status'] in ('incomplete', 'absent'), main['status']
        assert main_arcs == [], f'partial main result leaked into real plan: {main_arcs}'
        assert plan['pending_main_result'] is not None
        details['real'] = {'status': main['status'],
                           'missing': main['pending_missing'],
                           'issues': main['pending_issues']}
    with tempfile.TemporaryDirectory(prefix='r2_pkg_probe_main_') as tmp:
        # complete fixture: exactly 6 approved JSONs, weights excluded
        ws = _make_mock_workspace(os.path.join(tmp, 'complete'))
        plan = pkg.collect_plan(ws, roots=_mock_roots(ws))
        main = plan['main_graph_result']
        assert main['status'] == 'complete', main
        approved = main['approved_files']
        assert len(approved) == 6, approved
        assert plan['counts'].get('main_result_json', 0) == 6
        assert plan['pending_main_result'] is None
        out = os.path.join(tmp, 'complete', 'out')
        summary = pkg.build_package(ws, out, 'main_complete',
                                    roots=_mock_roots(ws))
        staging = summary['staging_dir']
        packaged = []
        for root, dirs, files in os.walk(staging):
            for fn in files:
                packaged.append(os.path.relpath(os.path.join(root, fn), staging)
                                .replace(os.sep, '/'))
        for rel in approved:
            assert rel in packaged, f'approved main result not packaged: {rel}'
        assert not any(a.endswith(('.pth', '.pt')) for a in packaged)
        with open(os.path.join(staging, approved[0]), 'r', encoding='utf-8') as fh:
            seed0 = json.load(fh)
        with open(os.path.join(staging, approved[4]), 'r', encoding='utf-8') as fh:
            seed4 = json.load(fh)
        assert seed0['protocol']['splits_config'] == pkg.EXECUTION_CONFIG_WHITELIST[3]
        assert seed4['protocol']['splits_config'] == pkg.EXECUTION_CONFIG_WHITELIST[4]
        details['complete_fixture'] = {'approved': approved,
                                       'weights_packaged': False}

        def fresh(name):
            return _make_mock_workspace(os.path.join(tmp, name))

        def assert_refused(ws, name, needle):
            plan = pkg.collect_plan(ws, roots=_mock_roots(ws))
            assert plan['pending_main_result'] is not None, (name, plan['main_graph_result'])
            assert not any(m['arcname'].startswith(pkg.MAIN_GRAPH_RESULT_DIR + '/')
                           for m in plan['members']), name
            out_dir = os.path.join(tmp, name, 'out')
            try:
                pkg.build_package(ws, out_dir, name, roots=_mock_roots(ws))
                raise AssertionError(f'{name}: partial main result was not refused')
            except pkg.PackageRefused as exc:
                assert needle in str(exc) or 'partial main result' in str(exc), str(exc)
            assert not os.path.exists(os.path.join(out_dir, name + '.tar.gz'))
            return plan['main_graph_result']

        # 1) missing seed4 metrics
        ws = fresh('missing_seed4')
        os.remove(os.path.join(ws, *pkg.MAIN_GRAPH_RESULT_DIR.split('/'),
                               'seed4', 'metrics_seed4.json'))
        res = assert_refused(ws, 'missing_seed4', 'metrics_seed4.json')
        assert any('metrics_seed4.json' in item for item in res['pending_missing'])
        details['missing_seed4'] = 'refused, zero main members'

        # 2) one metric not 500 epochs
        ws = fresh('epoch_short')
        metrics_path = os.path.join(ws, *pkg.MAIN_GRAPH_RESULT_DIR.split('/'),
                                    'seed2', 'metrics_seed2.json')
        doc = json.load(open(metrics_path, encoding='utf-8'))
        doc['protocol']['epochs'] = 499
        _write(metrics_path, json.dumps(doc, indent=2) + '\n')
        res = assert_refused(ws, 'epoch_short', 'epochs')
        assert any('epochs' in item for item in res['pending_issues'])
        details['epoch_short'] = 'refused'

        # 3) summary not consistent with the five seed metrics
        ws = fresh('summary_bad')
        summary_path = os.path.join(ws, *pkg.MAIN_GRAPH_RESULT_DIR.split('/'),
                                    'summary.json')
        summary_doc = json.load(open(summary_path, encoding='utf-8'))
        summary_doc['agg']['re']['mean']['u'] += 1.0
        _write(summary_path, json.dumps(summary_doc, indent=2) + '\n')
        res = assert_refused(ws, 'summary_bad', 'summary')
        assert res['summary_consistent'] is False
        details['summary_bad'] = 'refused'

        # 4) missing weight metadata file for seed3
        ws = fresh('missing_weights')
        os.remove(os.path.join(ws, *pkg.MAIN_GRAPH_RESULT_DIR.split('/'),
                               'seed3', 'checkpoint_best.pth'))
        res = assert_refused(ws, 'missing_weights', 'checkpoint_best.pth')
        assert any('checkpoint_best.pth' in item for item in res['pending_missing'])
        details['missing_weights'] = 'refused'
    return details


def test_sanitize_unknown_blocked():
    roots = pkg.default_known_roots(WS)
    doc = {'private': '/home/private/person.csv',
           'win_private': 'D:/Users/other/person.csv',
           'ok': 0.5}
    sanitized, info = pkg.sanitize_json_document(doc, roots)
    assert len(info['blocked']) == 2, info['blocked']
    assert sanitized['private'] is None and sanitized['win_private'] is None
    blocked_text = json.dumps(info['blocked'])
    assert '/home/private/person.csv' not in blocked_text, \
        'blocked report must not echo the unknown path content'
    assert 'D:/Users/other/person.csv' not in blocked_text
    assert all('field' in item and 'rule' in item for item in info['blocked'])
    return {'blocked_fields': [item['field'] for item in info['blocked']],
            'content_echoed': False}


def test_credential_scan():
    # construct the synthetic token strings dynamically so this probe source file
    # itself does not contain literal credential formats (it is packaged + scanned)
    fake_aws = 'AKIA' + 'ABCDEFGHIJKLMNOP'
    fake_gh = 'ghp_' + 'a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8'
    fake_value = 'A1b2C3d4E5f6G7h8'
    fake_assignment = 'api' + '_key' + " = '" + fake_value + "'"
    private_header = '-----BEGIN ' + 'RSA PRIVATE KEY-----'
    text = ('clean line\n'
            f'aws = {fake_aws}\n'
            f'github = {fake_gh}\n'
            f'{fake_assignment}\n'
            f'{private_header}\n')
    hits = pkg.scan_text_for_credentials(text)
    rules = {hit['rule'] for hit in hits}
    assert 'aws_access_key' in rules, rules
    assert 'github_token' in rules, rules
    assert 'secret_assignment' in rules, rules
    assert 'private_key_block' in rules, rules
    serialized = json.dumps(hits)
    for secret in (fake_aws, fake_gh, fake_value):
        assert secret not in serialized, 'secret content was echoed'
    clean = pkg.scan_text_for_credentials('VALUE = 1\n# no secrets here\n')
    assert clean == []
    return {'rules': sorted(rules), 'content_echoed': False}


def test_fixture_build_extract():
    with tempfile.TemporaryDirectory(prefix='r2_pkg_probe_build_') as tmp:
        mock_ws = _make_mock_workspace(tmp)
        out = os.path.join(tmp, 'out')
        summary = pkg.build_package(mock_ws, out, 'fixture_package',
                                 roots=_mock_roots(mock_ws))
        name = summary['name']
        archive = summary['archive']
        outer = json.load(open(summary['manifest_path'], encoding='utf-8'))
        assert os.path.exists(archive) and os.path.exists(summary['sha256_path'])
        assert os.path.exists(os.path.join(summary['staging_dir'], 'PACKAGE_CONTENTS.txt'))
        assert os.path.exists(os.path.join(summary['staging_dir'],
                                           'PACKAGE_MANIFEST.json'))
        assert outer['archive']['sha256'] == summary['archive_sha256']
        with open(summary['sha256_path'], encoding='utf-8') as fh:
            assert summary['archive_sha256'] in fh.read()
        # archive members: count, no forbidden files, sha matches manifest
        manifest = json.load(open(os.path.join(summary['staging_dir'],
                                               'PACKAGE_MANIFEST.json'),
                                  encoding='utf-8'))
        expected = {item['path']: item['sha256'] for item in manifest['files']}
        expected['PACKAGE_MANIFEST.json'] = sha256_file_path(
            os.path.join(summary['staging_dir'], 'PACKAGE_MANIFEST.json'))
        with tarfile.open(archive, 'r:gz') as tar:
            names = [m.name for m in tar.getmembers()]
            assert len(names) == len(set(names))
            assert len(names) == manifest['counts']['archive_members'], (
                len(names), manifest['counts'])
            for member in tar.getmembers():
                assert member.isfile()
                arc = pkg._safe_relpath(member.name)
                assert not arc.startswith(('data/acoustic/', 'docs/', 'paper/'))
                assert not arc.endswith(('.pth', '.pt', '.npz', '.log', '.out', '.err'))
                assert arc in expected, arc
                data = tar.extractfile(member).read()
                assert _sha_bytes(data) == expected[arc], arc
        # fixture results: numbers/ids unchanged, known paths normalised
        with tarfile.open(archive, 'r:gz') as tar:
            member = [m for m in tar.getmembers()
                      if m.name.endswith('results/round2_mechanism/fixture_result.json')][0]
            packaged = json.loads(tar.extractfile(member).read().decode('utf-8'))
        source_doc = json.load(open(os.path.join(
            mock_ws, 'results', 'round2_mechanism', 'fixture_result.json'),
            encoding='utf-8'))
        for key in ('case_id', 'seed', 'split_ids', 'metrics', 'stats'):
            assert packaged[key] == source_doc[key], key
        assert packaged['source_path'] == 'results/round2_mechanism/fixture_result.json'
        # generated files / exclusions
        with tarfile.open(archive, 'r:gz') as tar:
            member_names = {m.name for m in tar.getmembers()}
        assert 'PENDING_STATUS.json' in member_names
        assert 'data/README_DATA_PLACEMENT.md' in member_names
        assert 'ENVIRONMENT_PROVENANCE.txt' in member_names
        assert 'LICENSE' in member_names and 'LICENSE-DATA' not in member_names
        assert not any(n.endswith(('checkpoint_best.pth', 'diagnostics.npz',
                                   'data/acoustic/case_fixture.npz')) for n in member_names)
        # historical manuscript baselines + four-only audit manifests are included
        assert 'results/thermal_protocolA/mlp/summary.json' in member_names
        assert 'results/baselines_ext/thermal_protocolA/siren/summary.json' in member_names
        assert 'results/acoustic_protocolA/fno/summary.json' in member_names
        assert 'results/acoustic_wp5_ablation/ab_nofourier/summary.json' in member_names
        for rel in pkg.AUDIT_SOURCE_MANIFEST_WHITELIST:
            assert rel in member_names, rel
        assert 'results/round2_audit/other_audit.json' not in member_names
        # sensitive required inputs present
        assert 'scripts/exp_mechanism.py' in member_names
        assert 'results/thermal_geometry_spec.json' in member_names
        audit = manifest['dependency_audit']
        assert set(audit['required_arcs']) == set(pkg.REQUIRED_PACKAGE_ARCS)
        assert audit['missing_required_arcs'] == []
        assert audit['missing_dynamic_helpers'] == []
        assert audit['unresolved_dynamic_helpers'] == []
        assert any(item['loaded_by'] == 'scripts/fig_r2_mechanism.py'
                   and item['resolved_file'] == 'scripts/exp_mechanism.py'
                   and item['included'] is True
                   for item in audit['dynamic_helper_references'])
        # geometry spec release-copy: numeric/IDs unchanged, known path normalised
        with tarfile.open(archive, 'r:gz') as tar:
            geo_member = [m for m in tar.getmembers()
                          if m.name == 'results/thermal_geometry_spec.json'][0]
            geo_packaged = json.loads(tar.extractfile(geo_member).read().decode('utf-8'))
        geo_source = json.load(open(
            os.path.join(mock_ws, *pkg.GEOMETRY_SPEC_PATH.split('/')),
            encoding='utf-8'))
        for key in ('geometry_ids', 'counts', 'numbers'):
            assert geo_packaged[key] == geo_source[key], key
        assert geo_packaged['source_path'] == 'results/thermal_geometry_spec.json'
        # frozen split snapshots: byte-identical config class, exact files only
        assert all(rel in member_names for rel in pkg.EXECUTION_CONFIG_WHITELIST)
        assert manifest['counts']['by_kind'].get('execution_config') == \
            len(pkg.EXECUTION_CONFIG_WHITELIST)
        with tarfile.open(archive, 'r:gz') as tar:
            for rel in pkg.EXECUTION_CONFIG_WHITELIST:
                snap_member = [m for m in tar.getmembers() if m.name == rel][0]
                packed_bytes = tar.extractfile(snap_member).read()
                src_bytes = open(os.path.join(mock_ws, *rel.split('/')), 'rb').read()
                assert packed_bytes == src_bytes, f'{rel} is not byte-identical'
            metric_member = [m for m in tar.getmembers() if m.name ==
                             'results/round2_thermal_symmetric/fixture_result.json'][0]
            metric_doc = json.loads(tar.extractfile(metric_member).read().decode('utf-8'))
        assert metric_doc['protocol']['splits_config'] \
            == pkg.EXECUTION_CONFIG_WHITELIST[0]
        assert metric_doc['snapshot_refs'] == list(pkg.EXECUTION_CONFIG_WHITELIST)
        assert not any(n.startswith('results/round2_execution_20261021_')
                       and n not in pkg.EXECUTION_CONFIG_WHITELIST
                       for n in member_names)
        # complete main five-seed fixture: 6 approved JSONs, weights excluded
        assert manifest['counts']['by_kind'].get('main_result_json') == 6
        assert 'results/round2_graph_five_seed_main/meshgraphnet/summary.json' \
            in member_names
        for seed in pkg.MAIN_GRAPH_SEEDS:
            assert (f'results/round2_graph_five_seed_main/meshgraphnet/seed{seed}/'
                    f'metrics_seed{seed}.json') in member_names
        assert not any(n.endswith(('.pth', '.pt')) for n in member_names)
        assert not any(n.startswith('results/round2_graph_five_seed_main/') and
                       not n.endswith('.json') for n in member_names)
        # new_queries completed JSON included; internal registry candidate/npz excluded
        assert any(n.startswith('results/round2_new_queries/') and n.endswith('.json')
                   for n in member_names), 'new_queries JSON missing from fixture package'
        assert 'results/round2_new_queries/registry_candidate.json' not in member_names
        assert not any(n.endswith('.npz') for n in member_names)
        with tarfile.open(archive, 'r:gz') as tar:
            pend_member = [m for m in tar.getmembers()
                           if m.name == 'PENDING_STATUS.json'][0]
            pend = json.loads(tar.extractfile(pend_member).read().decode('utf-8'))
        assert [item['id'] for item in pend['pending_items']] \
            == ['R2-FDTD-CONV-01'], pend['pending_items']
        assert all(item['results_included'] is False
                   for item in pend['pending_items'])
        scope_notes = {item['id']: item
                       for item in pend.get('completed_items_scope_notes', [])}
        assert 'new_queries' in scope_notes, scope_notes
        nq = scope_notes['new_queries']
        assert nq['results_included'] is True
        assert 'interpolated' in nq['scope_note'].lower()
        assert 'not an independent fem' in nq['scope_note'].lower()
        assert all('new_queries' != item['id'] for item in pend['pending_items'])
        for removed in ('results/thermal_protocolA/feat_extract/',
                        'results/thermal_protocolA/unif_therm/',
                        'results/acoustic_protocolA/fourier_c/',
                        'results/acoustic_protocolA/fourier_nc/',
                        'results/acoustic_protocolA/unif_acous/',
                        'results/acoustic_wp5_ablation/ab_nores/',
                        'results/joint/'):
            assert not any(n.startswith(removed) for n in member_names), removed
        # relative-prefix path values are normalised in the packaged audit manifest
        with tarfile.open(archive, 'r:gz') as tar:
            member = [m for m in tar.getmembers() if m.name in
                      pkg.AUDIT_SOURCE_MANIFEST_WHITELIST][0]
            audit_doc = json.loads(tar.extractfile(member).read().decode('utf-8'))
        assert audit_doc['source'] == member.name, (audit_doc['source'], member.name)
        assert audit_doc['rel_posix'] == member.name
        assert audit_doc['numbers'] == [1, 2.5, -3.0]
        assert audit_doc['nested']['order'] == [3, 2, 1]
        # deterministic archive: rebuilding into a second out-dir gives same sha
        out2 = os.path.join(tmp, 'out2')
        summary2 = pkg.build_package(mock_ws, out2, 'fixture_package',
                                  roots=_mock_roots(mock_ws))
        assert summary2['archive_sha256'] == summary['archive_sha256'], \
            'archive hash is not reproducible for identical inputs'
        # overwrite refusal leaves the original untouched
        expect_raises(pkg.PackageRefused,
                      lambda: pkg.build_package(mock_ws, out, 'fixture_package',
                                            roots=_mock_roots(mock_ws)),
                      'same-name package overwrite')
        assert sha256_file_path(archive) == summary['archive_sha256']
        # safe extraction
        extract_dir = os.path.join(tmp, 'extract')
        pkg.safe_extract(archive, extract_dir)
        extracted = {}
        for root, dirs, files in os.walk(extract_dir):
            for fn in files:
                full = os.path.join(root, fn)
                extracted[os.path.relpath(full, extract_dir).replace(os.sep, '/')] = full
        assert set(extracted) == set(expected), 'extracted set differs'
        for arc, path in extracted.items():
            assert _sha_bytes(open(path, 'rb').read()) == expected[arc], arc
        return {'archive_sha256': summary['archive_sha256'],
                'archive_bytes': summary['archive_bytes'],
                'n_members': len(expected),
                'reproducible_sha': True,
                'safe_extract_ok': True,
                'numeric_invariance': True}


def sha256_file_path(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def test_malicious_archives_rejected():
    with tempfile.TemporaryDirectory(prefix='r2_pkg_probe_malicious_') as tmp:
        cases = {
            'absolute_member': [('/abs/evil.txt', 'file', 'x')],
            'dotdot_member': [('../escape.txt', 'file', 'x')],
            'symlink_member': [('pkg/link', 'symlink', '../outside')],
            'hardlink_member': [('pkg/hard', 'hardlink', 'pkg/file')],
        }
        for name, entries in cases.items():
            path = os.path.join(tmp, name + '.tar.gz')
            _write_tar(path, entries)
            expect_raises(pkg.PackageRefused,
                          lambda p=path, n=name: pkg.safe_extract(
                              p, os.path.join(tmp, 'extract_' + n)),
                          f'safe_extract for {name}')
        # a normal archive still extracts
        good = os.path.join(tmp, 'good.tar.gz')
        _write_tar(good, [('pkg/a.txt', 'file', 'hello')])
        dest = pkg.safe_extract(good, os.path.join(tmp, 'extract_good'))
        assert os.path.isfile(os.path.join(dest, 'pkg', 'a.txt'))
        return {'refused': sorted(cases), 'normal_extract_ok': True}


def test_build_refusals():
    details = {}
    with tempfile.TemporaryDirectory(prefix='r2_pkg_probe_refuse_') as tmp:
        # unknown absolute path -> blocked, no package left behind
        ws = _make_mock_workspace(os.path.join(tmp, 'blocked'), unknown_path=True)
        out = os.path.join(tmp, 'blocked', 'out')
        try:
            pkg.build_package(ws, out, 'blocked_package', roots=_mock_roots(ws))
            raise AssertionError('unknown absolute path was not refused')
        except pkg.PackageRefused as exc:
            assert 'unknown_absolute_path' in str(exc), str(exc)
            assert '/home/private/person.csv' not in str(exc), 'path content echoed'
            details['blocked'] = 'refused without echoing unknown path'
        assert not os.path.exists(os.path.join(out, 'blocked_package'))
        assert not os.path.exists(os.path.join(out, 'blocked_package.tar.gz'))

        # credential hit -> refused, secret not printed
        ws = _make_mock_workspace(os.path.join(tmp, 'cred'), credential=True)
        out = os.path.join(tmp, 'cred', 'out')
        try:
            pkg.build_package(ws, out, 'cred_package', roots=_mock_roots(ws))
            raise AssertionError('credential-like content was not refused')
        except pkg.PackageRefused as exc:
            message = str(exc)
            assert 'AKIA' + 'ABCDEFGHIJKLMNOP' not in message, 'secret echoed'
            assert 'leaky_key.py' in message and 'rule=' in message
            details['credential'] = 'refused with file/line/rule only'
        assert not os.path.exists(os.path.join(out, 'cred_package.tar.gz'))

        # missing Codex metadata -> refused, no package
        ws = _make_mock_workspace(os.path.join(tmp, 'metamissing'),
                                  metadata_complete=False)
        out = os.path.join(tmp, 'metamissing', 'out')
        try:
            pkg.build_package(ws, out, 'meta_package', roots=_mock_roots(ws))
            raise AssertionError('missing metadata was not refused')
        except pkg.PackageRefused as exc:
            assert 'README.md' in str(exc)
            details['missing_metadata'] = 'refused naming README.md'
        assert not os.path.exists(os.path.join(out, 'meta_package.tar.gz'))

        # missing manuscript table-source manifest -> refused, no package
        ws = _make_mock_workspace(os.path.join(tmp, 'auditmissing'))
        removed = os.path.join(ws, 'results', 'round2_audit',
                               'acoustic_ablation_manuscript_sources.json')
        os.remove(removed)
        out = os.path.join(tmp, 'auditmissing', 'out')
        try:
            pkg.build_package(ws, out, 'audit_package', roots=_mock_roots(ws))
            raise AssertionError('missing source manifest was not refused')
        except pkg.PackageRefused as exc:
            assert 'acoustic_ablation_manuscript_sources.json' in str(exc), str(exc)
            details['missing_source_manifest'] = 'refused naming the manifest'
        assert not os.path.exists(os.path.join(out, 'audit_package.tar.gz'))

        # sensitive required inputs removed -> refused, no package
        for missing_rel, label in (
                ('scripts/exp_mechanism.py', 'missing_exp_mechanism'),
                ('results/thermal_geometry_spec.json', 'missing_geometry_spec'),
                (pkg.EXECUTION_CONFIG_WHITELIST[1], 'missing_split_snapshot')):
            ws = _make_mock_workspace(os.path.join(tmp, label))
            os.remove(os.path.join(ws, *missing_rel.split('/')))
            out = os.path.join(tmp, label, 'out')
            try:
                pkg.build_package(ws, out, label, roots=_mock_roots(ws))
                raise AssertionError(f'missing required input not refused: {missing_rel}')
            except pkg.PackageRefused as exc:
                assert missing_rel in str(exc), str(exc)
                details[label] = 'refused naming ' + missing_rel
            assert not os.path.exists(os.path.join(out, label + '.tar.gz'))
    return details


def test_model_cpu_small_shape():
    import torch
    torch.set_num_threads(FAKE_THREADS)
    device = torch.device('cpu')
    x = torch.zeros(1, 2, 32, device=device)
    try:
        from scripts.exp_thermal import build_model as build_thermal
        from scripts.exp_acoustic import build_model as build_acoustic
    except ModuleNotFoundError as exc:
        if exc.name not in ('fcntl', 'pwd', 'grp', 'resource'):
            raise
        # Windows local probe: the CLI entry modules import POSIX-only fcntl for
        # file locking, so the entry builders cannot be imported here. Exercise the
        # same real model classes the builders construct, with explicit semantics,
        # and record that the entry-builder path is run on the Linux server/Codex.
        from src.models.featextractnet import FeatExtractNet
        from src.models.unified.cond_mlps import CondFourierMLP

        model = FeatExtractNet(input_channels=2, output_channels=4, scaling=1.0,
                               pool_mode='symmetric')
        assert getattr(model, 'pool_mode', None) == 'symmetric'
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert tuple(out.shape) == (1, 4, 32), out.shape
        assert torch.isfinite(out).all()
        acoustic = CondFourierMLP(input_channels=2, output_channels=1, cond_dim=2,
                                  hidden_dim=256, num_blocks=6, fourier_levels=16,
                                  fourier_layout='pointwise')
        assert getattr(acoustic, 'fourier_layout', None) == 'pointwise'
        acoustic.eval()
        cond = torch.tensor([[0.2, 0.3]], dtype=torch.float32)
        with torch.no_grad():
            out_ac = acoustic(x, cond)
        assert tuple(out_ac.shape) == (1, 1, 32), out_ac.shape
        assert torch.isfinite(out_ac).all()
        return {'builder_mode': 'direct_model_classes (entry modules need POSIX fcntl)',
                'thermal_pool_mode': model.pool_mode,
                'acoustic_layout': acoustic.fourier_layout,
                'shapes': [list(out.shape), list(out_ac.shape)],
                'device': 'cpu', 'cuda_used': False}

    model, label = build_thermal('feat_extract', pool_context='symmetric')
    assert getattr(model, 'pool_mode', None) == 'symmetric', label
    assert 'symmetric' in label
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert tuple(out.shape) == (1, 4, 32), out.shape
    assert torch.isfinite(out).all()
    legacy, legacy_label = build_thermal('feat_extract',
                                         pool_context='legacy_index_bins')
    assert getattr(legacy, 'pool_mode', None) == 'legacy_index_bins'
    assert 'legacy_index_bins' in legacy_label

    acoustic, acoustic_label = build_acoustic('fourier_c',
                                              fourier_layout='pointwise')
    assert 'pointwise' in acoustic_label, acoustic_label
    acoustic.set_cond(torch.tensor([[0.2, 0.3]], dtype=torch.float32))
    acoustic.eval()
    with torch.no_grad():
        out_ac = acoustic(x)
    assert tuple(out_ac.shape) == (1, 1, 32), out_ac.shape
    assert torch.isfinite(out_ac).all()
    acoustic_legacy, legacy_ac_label = build_acoustic('fourier_c',
                                                      fourier_layout='legacy')
    assert 'legacy' in legacy_ac_label and 'pointwise' not in legacy_ac_label
    return {'builder_mode': 'entry_builders',
            'thermal_pool_mode': model.pool_mode,
            'acoustic_layout': 'pointwise',
            'shapes': [list(out.shape), list(out_ac.shape)],
            'device': 'cpu', 'cuda_used': False}


def test_torchgeo_skip():
    spec = importlib.util.find_spec('torchgeo')
    return {'torchgeo_available': spec is not None,
            'skipped': spec is None,
            'installed_by_probe': False}


def test_core_data_paths():
    from src.core import acoustic_io
    details = {'real_data_checked': False, 'skipped': 'no real data present'}
    cfg = os.path.join(WS, 'configs', 'acoustic_cases.json')
    entries = acoustic_io.make_case_list(cfg)
    assert len(entries) == 21, len(entries)
    expected_root = os.path.realpath(os.path.join(WS, 'data', 'acoustic'))
    for entry in entries:
        assert os.path.dirname(os.path.realpath(entry['file'])) == expected_root, \
            entry['file']
        assert entry['case_id'] and entry['f'] is not None and entry['Ma'] is not None
    real_data = os.path.isdir(expected_root)
    if real_data:
        missing = [e['file'] for e in entries if not os.path.isfile(e['file'])]
        details = {'real_data_checked': True,
                   'n_cases': len(entries),
                   'missing_case_files': missing,
                   'all_case_files_present': not missing}
        assert not missing, missing
    thermal = os.path.join(os.path.dirname(os.path.dirname(WS)),
                           'HeatTransferData.npy')
    details['thermal_data_present'] = os.path.isfile(thermal)
    details['downloaded_by_probe'] = False
    return details


TEST_GROUPS = (
    ('real_plan_whitelist', test_real_plan_whitelist),
    ('license_alignment_preflight', test_license_alignment_preflight),
    ('frozen_split_snapshots', test_frozen_split_snapshots),
    ('metric_derived_split_dependencies', test_metric_derived_split_dependencies),
    ('main_five_seed_gate', test_main_five_seed_gate),
    ('list_only_no_write', test_list_only_no_write),
    ('json_sanitize_known_and_numeric', test_sanitize_known_and_numeric),
    ('json_relative_prefix_normalization', test_relative_prefix_normalization),
    ('json_real_release_copy_no_blocked', test_real_json_release_copy),
    ('dynamic_helper_detection', test_dynamic_helper_detection),
    ('json_sanitize_unknown_blocked', test_sanitize_unknown_blocked),
    ('credential_scan', test_credential_scan),
    ('fixture_build_verify_extract', test_fixture_build_extract),
    ('malicious_archive_rejected', test_malicious_archives_rejected),
    ('build_refusals', test_build_refusals),
    ('model_cpu_small_shape', test_model_cpu_small_shape),
    ('torchgeo_skip', test_torchgeo_skip),
    ('core_data_grouping_paths', test_core_data_paths),
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output', default=None,
                    help='JSON probe report path (default: OS temp dir, unique name)')
    ap.add_argument('--version', action='version', version=PROBE_VERSION)
    return ap.parse_args(argv)


def _default_output_path():
    stamp = time.strftime('%Y%m%d_%H%M%S')
    return os.path.join(tempfile.gettempdir(),
                        f'r2_release_package_probe_{stamp}.json')


def main(argv=None):
    args = parse_args(argv)
    probe = Probe()
    t0 = time.perf_counter()
    for name, fn in TEST_GROUPS:
        probe.check(name, fn)
    wall = time.perf_counter() - t0
    if probe.failures:
        status = 'failed'
    elif probe.blocked:
        status = 'blocked'
    else:
        status = 'passed'
    output = os.path.abspath(args.output or _default_output_path())
    os.makedirs(os.path.dirname(output), exist_ok=True)
    metadata_missing = [fname for fname in pkg.METADATA_FILES
                        if not os.path.isfile(os.path.join(pkg.ZENODO_ROUND2_DIR,
                                                           fname))]
    not_run = ['publication/DOI creation', 'CUDA runs', 'training',
               'network access / dependency installation']
    if metadata_missing:
        not_run.insert(0, 'real package build (Codex metadata still missing: '
                          + ', '.join(metadata_missing) + ')')
    else:
        not_run.insert(0, 'formal package build/rebuild (deferred until Codex '
                          'metadata/README final; this probe only builds OS-temp '
                          'fixtures and does not write dist/)')
    doc = {
        'probe': os.path.basename(__file__),
        'probe_version': PROBE_VERSION,
        'task_id': 'R2-RELEASE-PACKAGE-01',
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'status': status,
        'n_checks': len(probe.checks),
        'n_failed': len(probe.failures),
        'failed_checks': probe.failures,
        'n_blocked': len(probe.blocked),
        'blocked_checks': probe.blocked,
        'wall_s': wall,
        'environment': {
            'hostname': socket.gethostname(),
            'platform': platform.platform(),
            'python': platform.python_version(),
            'numpy': np.__version__,
            'torch': getattr(sys.modules.get('torch'), '__version__', None),
            'torchgeo': importlib.util.find_spec('torchgeo') is not None,
            'cuda_used': False,
        },
        'hashes': {
            'probe_sha256': pkg.sha256_file(os.path.abspath(__file__)),
            'package_script_sha256': pkg.sha256_file(
                os.path.join(WS, 'scripts', 'package_round2.py')),
        },
        'scope': 'implementation contract only: package planning/JSON sanitisation/'
                 'archive safety/CPU model import; no publication, no training, no '
                 'CUDA, no install, no network; fixture results are synthetic and '
                 'are not scientific evidence',
        'checks': probe.checks,
        'not_run': not_run,
    }
    pkg._write_json(output, doc)
    for entry in probe.checks:
        marker = {'passed': 'PASS', 'blocked': 'BLOCKED'}.get(entry['status'], 'FAIL')
        print(f'[{marker}] {entry["name"]}', flush=True)
        if marker in ('FAIL', 'BLOCKED'):
            print(f'       {entry["detail"]}', flush=True)
    print(f'[{status.upper()}] '
          f'{len(probe.checks) - len(probe.failures) - len(probe.blocked)}/'
          f'{len(probe.checks)} checks passed, {len(probe.blocked)} blocked in '
          f'{wall:.2f} s; report: {output}', flush=True)
    return 0 if status == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
