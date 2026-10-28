"""Lightweight unpack/dependency/fidelity probe for the R2 reference package.

This script is included in the package. It verifies bytes, JSON fidelity, NPZ
array preservation, text cleanliness, CLI importability, and a missing-dolfinx
negative without installing anything.
"""
import argparse
import hashlib
import json
import locale
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

FORBIDDEN_PARTS = ('/tools/', '/.git/', '/__pycache__/', '/install.log')
FORBIDDEN_SUFFIX = ('.pth', '.pt', '.npy')
PRIVATE_PATTERNS = (
    re.compile('C:' + r'[/\\]Users[/\\]'),
    re.compile('/slurm' + '/home'),
    re.compile('/home' + '/jql-' + 'wanggulin'),
)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_extract(archive, dest):
    with tarfile.open(archive, 'r:gz') as tf:
        names = tf.getnames()
        for name in names:
            norm = os.path.normpath(name)
            if os.path.isabs(name) or norm.startswith('..' + os.sep) or '..' in norm.split(os.sep):
                raise RuntimeError(f'unsafe archive member: {name}')
        tf.extractall(dest)
    return names


def text_clean(path):
    try:
        text = open(path, 'r', encoding='utf-8').read()
    except (UnicodeDecodeError, OSError):
        return True
    return not any(p.search(text) for p in PRIVATE_PATTERNS)


def science_equal(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(science_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(science_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, str):
        if a == b:
            return True
        # Package release-copy may normalise only path strings; scientific
        # labels/values must otherwise be identical.
        def pathish(x):
            return (x in ('', '.', './') or ('<' in x and '>' in x)
                    or '/' in x or '\\' in x or x.endswith(
                        ('.json', '.yaml', '.yml', '.txt', '.npy', '.npz')))
        return pathish(a) and pathish(b)
    return a == b


UTF8_CHILD_ENV = {'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'}


def child_env(utf8=True, package_root=None):
    env = dict(os.environ)
    if utf8:
        env.update(UTF8_CHILD_ENV)
    else:
        env.pop('PYTHONIOENCODING', None)
        env.pop('PYTHONUTF8', None)
    env['PYTHONPATH'] = (package_root or os.getcwd()) + os.pathsep + env.get('PYTHONPATH', '')
    return env


def decode_child(data):
    # Child is forced to UTF-8; decode explicitly rather than relying on the
    # Windows console code page.  Replacement only affects the report text; a
    # non-zero return code is still a failure and is never masked.
    return (data or b'').decode('utf-8', errors='replace')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive', required=True)
    ap.add_argument('--workspace', default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument('--out', default=None)
    ap.add_argument('--expect-missing-dolfinx', action='store_true', default=True)
    args = ap.parse_args(argv)
    checks = []
    archive = os.path.abspath(args.archive)
    checks.append({'name': 'archive_exists', 'ok': os.path.isfile(archive)})
    if not os.path.isfile(archive):
        return _finish(checks, args.out)
    work = tempfile.mkdtemp(prefix='r2_reference_probe_')
    try:
        names = safe_extract(archive, work)
        checks.append({'name': 'archive_member_count_nonempty', 'ok': len(names) > 300,
                       'detail': len(names)})
        checks.append({'name': 'no_forbidden_paths', 'ok': not any(
            any(part in ('/' + n.replace('\\', '/') + '/') for part in FORBIDDEN_PARTS)
            for n in names)})
        checks.append({'name': 'no_weights_or_raw_npy', 'ok': not any(
            n.lower().endswith(FORBIDDEN_SUFFIX) for n in names)})
        checks.append({'name': 'no_slurm_scripts', 'ok': not any(
            n.startswith('scripts/sbatch/') for n in names)})
        text_files = [os.path.join(work, n) for n in names
                      if os.path.isfile(os.path.join(work, n)) and n.lower().endswith(
                          ('.py', '.json', '.md', '.txt', '.yml', '.yaml', '.cff', '.log'))]
        bad_text = [n for n in names
                    if n.lower().endswith(('.py', '.json', '.md', '.txt', '.yml', '.yaml', '.cff', '.log'))
                    and not text_clean(os.path.join(work, n))]
        checks.append({'name': 'no_private_paths_in_text', 'ok': not bad_text,
                       'detail': bad_text[:10]})
        manifest_path = os.path.join(work, 'PACKAGE_MANIFEST.json')
        manifest = json.load(open(manifest_path, encoding='utf-8'))
        entries = manifest.get('files', manifest.get('entries', []))
        checks.append({'name': 'package_manifest_entries', 'ok': len(entries) > 300,
                       'detail': len(entries)})
        # Final merged reference evidence: exact pressure + FDTD aggregate/meta/protocol.
        exact_required = [
            'src/fem_reference/pressure_reference_audit.py',
            'scripts/exp_r2_exact_pressure_reference.py',
            'scripts/exp_r2_pressure_reference_audit.py',
            'scripts/probe_r2_pressure_reference_audit.py',
            'results/round2_pressure_reference_closure/exact/fields_geom17_h004.npz',
            'results/round2_pressure_reference_closure/exact/fields_geom46_h004.npz',
            'results/round2_pressure_reference_closure/exact/fields_geom96_h004.npz',
            'results/round2_audit/r2_pressure_closure_codex_acceptance.json',
        ]
        fdtd_required = ['results/round2_fdtd_convergence/aggregate.json'] + [
            f'results/round2_fdtd_convergence/{case}/{res}/{name}'
            for case in ('low100', 'high500')
            for res in ('coarse', 'mid', 'fine', 'mid_dt2', 'mid_dt4')
            for name in ('meta.json', 'protocol.json')]
        missing_new = [p for p in exact_required + fdtd_required
                       if p not in names]
        checks.append({'name': 'new_module_entry_and_fdtd_coverage',
                       'ok': not missing_new, 'detail': missing_new})
        fdtd_fields = [n for n in names if n.startswith(
            'results/round2_fdtd_convergence/') and n.endswith('.npz')]
        checks.append({'name': 'no_large_fdtd_fields', 'ok': not fdtd_fields,
                       'detail': fdtd_fields[:5]})
        try:
            agg = json.load(open(os.path.join(work, 'results/round2_fdtd_convergence/aggregate.json'),
                                 encoding='utf-8'))
            fdtd_ok = (agg.get('complete') is True and int(agg.get('n_solves', -1)) == 10)
            fdtd_detail = {'complete': agg.get('complete'), 'n_solves': agg.get('n_solves'),
                           'out_root': agg.get('out_root')}
        except Exception as exc:  # noqa: BLE001
            fdtd_ok = False
            fdtd_detail = repr(exc)
        checks.append({'name': 'fdtd_aggregate_10_of_10', 'ok': fdtd_ok,
                       'detail': fdtd_detail})
        audit_paths = ['results/round2_audit/r2_pressure_closure_codex_acceptance.json',
                       'results/round2_audit/r2_fdtd_complete_codex_acceptance.json']
        audit_bad = [p for p in audit_paths if p not in names]
        checks.append({'name': 'accepted_audits_included', 'ok': not audit_bad,
                       'detail': audit_bad})
        try:
            pa = json.load(open(os.path.join(work, audit_paths[0]), encoding='utf-8'))
            fa = json.load(open(os.path.join(work, audit_paths[1]), encoding='utf-8'))
            audit_ok = (
                str(pa.get('status', '')).startswith('accepted_exact_gauge')
                and int(pa.get('metric_cells', -1)) == 450
                and float(pa.get('max_abs_metric_difference', 1)) <= 1e-12
                and fa.get('status', '').startswith('accepted_complete_comparisons')
                and int(fa.get('solves', -1)) == 10
                and float(fa.get('max_absolute_difference', 1)) <= 1e-12)
            audit_detail = {'pressure_status': pa.get('status'),
                            'pressure_metric_cells': pa.get('metric_cells'),
                            'pressure_max_abs_metric_difference': pa.get('max_abs_metric_difference'),
                            'fdtd_status': fa.get('status'),
                            'fdtd_solves': fa.get('solves'),
                            'fdtd_max_absolute_difference': fa.get('max_absolute_difference')}
        except Exception as exc:  # noqa: BLE001
            audit_ok = False
            audit_detail = repr(exc)
        checks.append({'name': 'accepted_audit_claims_verified', 'ok': audit_ok,
                       'detail': audit_detail})
        pdfs = [n for n in names if n.lower().endswith('.pdf')]
        checks.append({'name': 'no_full_paper_pdfs', 'ok': not pdfs,
                       'detail': pdfs[:5]})
        try:
            lic = open(os.path.join(work, 'LICENSE'), encoding='utf-8').read()
            zen = json.load(open(os.path.join(work, '.zenodo.json'), encoding='utf-8'))
            cff = open(os.path.join(work, 'CITATION.cff'), encoding='utf-8').read()
            lic_ok = ('MIT' in lic and str(zen.get('license', '')).lower() == 'mit'
                      and 'mit' in cff.lower())
            lic_detail = {'zenodo_license': zen.get('license'),
                          'license_has_mit': 'MIT' in lic,
                          'cff_has_mit': 'mit' in cff.lower()}
        except Exception as exc:  # noqa: BLE001
            lic_ok = False
            lic_detail = repr(exc)
        checks.append({'name': 'mit_consistency', 'ok': lic_ok,
                       'detail': lic_detail})
        # NPZ byte/hash/array fidelity against the workspace source.
        npz_ok = True
        npz_detail = []
        for e in entries:
            path = e.get('path', '')
            if not path.endswith('.npz'):
                continue
            packaged = os.path.join(work, path)
            source = os.path.join(args.workspace, path)
            if not os.path.isfile(source):
                npz_ok = False
                npz_detail.append({'path': path, 'reason': 'source missing'})
                continue
            if sha(packaged) != sha(source):
                npz_ok = False
                npz_detail.append({'path': path, 'reason': 'sha mismatch'})
                continue
            import numpy as np
            a, b = np.load(packaged), np.load(source)
            keys = sorted(a.files)
            if keys != sorted(b.files):
                npz_ok = False
                npz_detail.append({'path': path, 'reason': 'keys mismatch'})
                continue
            for k in keys:
                if a[k].dtype != b[k].dtype or a[k].shape != b[k].shape or not np.array_equal(a[k], b[k]):
                    npz_ok = False
                    npz_detail.append({'path': path, 'key': k, 'reason': 'array mismatch'})
                    break
            a.close(); b.close()
        checks.append({'name': 'npz_byte_and_array_fidelity', 'ok': npz_ok,
                       'detail': npz_detail[:10]})
        # JSON scientific-value fidelity: packaged copy vs source with path strings allowed.
        json_ok = True
        json_detail = []
        for e in entries:
            if e.get('transformation', {}).get('type') != 'json_release_copy':
                continue
            path = e.get('path', '')
            packaged = os.path.join(work, path)
            source = os.path.join(args.workspace, path)
            if not os.path.isfile(source):
                json_ok = False
                json_detail.append({'path': path, 'reason': 'source missing'})
                continue
            try:
                a = json.load(open(source, encoding='utf-8'))
                b = json.load(open(packaged, encoding='utf-8'))
            except Exception as exc:  # noqa: BLE001
                json_ok = False
                json_detail.append({'path': path, 'reason': repr(exc)})
                continue
            if not science_equal(a, b):
                json_ok = False
                json_detail.append({'path': path, 'reason': 'scientific value mismatch'})
        checks.append({'name': 'json_scientific_value_fidelity', 'ok': json_ok,
                       'detail': json_detail[:10]})
        # Record the default Windows console condition and the explicit UTF-8
        # child environment needed to keep Unicode help text readable.
        default_env = child_env(utf8=False, package_root=work)
        default_probe = subprocess.run(
            [sys.executable, '-c',
             'import sys; print(sys.stdout.encoding); print("x\\u0302")'],
            cwd=work, env=default_env, capture_output=True)
        default_out = decode_child(default_probe.stdout).strip()
        default_err = decode_child(default_probe.stderr).strip()
        checks.append({'name': 'windows_default_console_observed',
                       'ok': True,
                       'detail': {'platform': sys.platform,
                                  'default_child_stdout': default_probe.stdout[:200].decode('ascii', 'replace'),
                                  'preferred_encoding': locale.getpreferredencoding(False),
                                  'default_child_ok': default_probe.returncode == 0,
                                  'default_stderr': default_err[-300:],
                                  'required_child_env': UTF8_CHILD_ENV}})
        # Syntax closure for all Python files; actual --help closure for the
        # reference modules.  Child stdout/stderr are forced to UTF-8 because the
        # Windows default console encoding is GBK and is not a code failure.
        env = child_env(utf8=True, package_root=work)
        py_bad = []
        for rel in [n for n in names if n.endswith('.py')]:
            proc = subprocess.run([sys.executable, '-m', 'py_compile', rel],
                                  cwd=work, env=env, capture_output=True, timeout=60)
            if proc.returncode != 0:
                py_bad.append({'script': rel,
                               'stderr': decode_child(proc.stderr)[-300:]})
        checks.append({'name': 'python_syntax_closure', 'ok': not py_bad,
                       'detail': py_bad[:10]})
        reference_clis = (
            'scripts/exp_r2_pressure_gauge.py', 'scripts/probe_r2_pressure_gauge.py',
            'scripts/exp_r2_thermal_fem.py', 'scripts/probe_r2_thermal_fem.py',
            'scripts/exp_r2_fem_model_eval.py', 'scripts/probe_r2_fem_model_eval.py',
            'scripts/postprocess_r2_thermal_fem_acceptance.py',
            'scripts/exp_r2_exact_pressure_reference.py',
            'scripts/exp_r2_pressure_reference_audit.py',
            'scripts/probe_r2_pressure_reference_audit.py',
            'scripts/probe_round2_reference_package.py',
        )
        cli_bad = []
        cli_ok = []
        for rel in reference_clis:
            if rel not in names:
                continue
            if rel.endswith('probe_round2_reference_package.py'):
                continue
            try:
                proc = subprocess.run([sys.executable, rel, '--help'], cwd=work,
                                      env=env, capture_output=True, timeout=60)
                if proc.returncode != 0:
                    cli_bad.append({'script': rel, 'returncode': proc.returncode,
                                    'stdout': decode_child(proc.stdout)[-300:],
                                    'stderr': decode_child(proc.stderr)[-800:]})
                else:
                    cli_ok.append(rel)
            except Exception as exc:  # noqa: BLE001
                cli_bad.append({'script': rel, 'error': repr(exc)})
        checks.append({'name': 'reference_cli_help_closure', 'ok': not cli_bad,
                       'detail': {'failed': cli_bad[:10], 'passed': cli_ok,
                                  'child_env': UTF8_CHILD_ENV,
                                  'real_fem_rerun': False,
                                  'real_fem_evidence_note': 'exact-pressure/FEM values are from server acceptance records; this probe only checks packaged entry syntax/help, not a new solve'}})
        # Missing dependency negative: explicit failure status, not a silent pass.
        env = child_env(utf8=True, package_root=work)
        neg_code = (
            'import sys; from src.fem_reference import fenics_solver; '
            's=fenics_solver.dependency_status(); '
            'print(\"MISSING\", [k for k,v in s.items() if k in (\"dolfinx\",\"ufl\",\"basix\",\"petsc4py\",\"mpi4py\",\"gmsh\") and not v]); '
            'sys.exit(0 if not s[\"all_present\"] else 2)')
        neg = subprocess.run([sys.executable, '-c', neg_code], cwd=work, env=env,
                             capture_output=True)
        checks.append({'name': 'missing_dolfinx_negative_explicit',
                       'ok': neg.returncode == 0 and 'MISSING' in decode_child(neg.stdout),
                       'detail': {'returncode': neg.returncode,
                                  'stdout': decode_child(neg.stdout).strip(),
                                  'stderr': decode_child(neg.stderr)[-300:]}})
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return _finish(checks, args.out)


def _finish(checks, out):
    ok = all(c['ok'] for c in checks)
    report = {'ok': ok, 'n_checks': len(checks), 'checks': checks}
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        json.dump(report, open(out, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False)[:12000])
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
