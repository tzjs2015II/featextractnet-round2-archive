"""scripts/package_round2.py —— 组装 R2 返稿准备包（代码 + 已完成结果 JSON，不含数据/权重）。

用途
----
准备当前**可审查、可复现**的 R2 修订材料包：源码（白名单）、固定科学入口脚本、
共享配置副本、已完成实验的结果 JSON（做 release-copy 路径规范化），以及由 Codex
提供的 4 个元数据文件。它**不是**正式发布、不写 DOI、不训练、不联网、不安装依赖。

严格边界（本脚本的实现契约）
----------------------------
* 只新增/读写包输出目录；不修改项目内任何源文件、配置、结果、zenodo 元数据。
* 不包含：原始数据（热场 .npy / 声学 .npz）、模型权重（*.pth/pt）、logs、
  diagnostics npz、整个 results/、除四个论文表格 source manifest 之外的
  round2_audit/、results/registry.json、内部 docs/、paper/、审稿/回复材料、
  ccfa-review-reports/、旧 zenodo 的 LICENSE-DATA。
* 结果 JSON 覆盖：既有的 10 个 round2 结果目录 + ``results/round2_new_queries``
  （仅 *.json，排除内部 ``registry_candidate.json``）+ 既有几何规格
  ``results/thermal_geometry_spec.json``（仅此一个 results/ 根级 JSON；由
  exp_r2_local_error / exp_r2_new_queries 读取）+ 5 个冻结 split 快照
  ``results/round2_execution_20261021/{v3,v7,v10}``、``20261022_v17`` 与
  ``20261022_v18`` 下的 ``configs/splits_thermal.json``
  （仅这 5 个小 JSON，不复制整个 execution 目录；被 metric 的
  ``protocol.splits_config`` 引用且 exp_mechanism 无 fallback；按配置类
  byte-identical 保留，无需路径规范化）+ 论文仍保留的历史有效 baseline
  精确白名单
  （见 ``HISTORICAL_RESULT_DIRS``；不含已撤去的 thermal feat_extract/unif_therm、
  旧 acoustic fourier_c/fourier_nc/unif_acous、joint 及旧 results/baselines 的
  3-seed graph 主比较——后者仅历史、由 5-seed 主结果与
  completed_comparison_graph_five_seed.json 取代），外加仅用于表格溯源的 4 个
  ``results/round2_audit/*_manuscript*_sources.json`` / 配对统计 JSON（见
  ``AUDIT_SOURCE_MANIFEST_WHITELIST``），与上方排除条目冲突时以上述精确名单为准。
* 新主留出图结果 ``results/round2_graph_five_seed_main/meshgraphnet`` 只在
  seed0-4 五份 500-epoch metrics 与权重 metadata 齐全、且 summary.json 的
  mean/sample-std 与五 seed metrics 实际一致时才按 6 个 JSON 入包；部分完成
  （如当前仅 seed0-2）时**不打包其任何文件并拒绝构建**，不得当作已完成主结果。
  旧 3seed 几何 summary 仅历史参考，当前 Table 5 以
  ``completed_comparison_graph_five_seed.json`` 为准。
* 不包含未完成实验（FDTD 压力收敛）的任何结果；``PENDING_STATUS.json`` 只把 FDTD 列为
  pending。new_queries 已完成并包含其 JSON，但其参考是**插值参考诊断（interpolated-
  reference diagnostic），不是独立 FEM 解**，不得当作独立 FEM 证据。
* 不复制旧 LICENSE-DATA 的笼统 CC-BY 声明作为第三方热数据授权：本包不重分发
  任何第三方热数据；热数据来源/下载/sha 由 Codex 提供的 ``DATA_SOURCES.md`` 说明。
* 元数据 licence preflight（只读一致性检查，**不构成新授权**）：``zenodo/LICENSE``
  必须仍是现有 MIT notice，``zenodo/round2/.zenodo.json`` 的 ``license`` 与
  ``CITATION.cff`` 的 ``license`` 必须同为 MIT；任何错配/缺失都会拒绝出包（保留
  LICENSE 原样，不重新许可任何第三方数据）。
* 不包含 scripts/sbatch（服务器私有硬编码入口）、旧 src/pipeline、DSH/sdk/probe
  凭据工具；不读取任何用户配置/凭据/DSH 数据。

包结构（固定名单，全部由白名单推导，不接受任意路径）
----------------------------------------------------
    <name>/
      README.md .zenodo.json CITATION.cff DATA_SOURCES.md   # zenodo/round2（Codex 提供）
      LICENSE                                               # zenodo/LICENSE（代码 MIT，原样）
      requirements.txt requirements-extended.txt environment.yml  # 旧 zenodo 历史环境快照
      ENVIRONMENT_PROVENANCE.txt                            # 由本脚本生成的来源说明
      PENDING_STATUS.json                                   # 仅状态/范围说明，无 pending 结果
      data/README_DATA_PLACEMENT.md                         # 用户放数据的目标路径（不含数据）
      src/**/*.py                                           # 全部 .py，排除 src/pipeline
      scripts/<白名单>.py + 明确列出的科学 helper
      configs/{splits_thermal,r2_geometry_protocols,acoustic_cases}.json
      results/<round2 结果白名单>/**/*.json                  # release-copy（路径规范化）
      results/thermal_protocolA/<5 baseline>/**/*.json       # 历史有效 baseline
      results/baselines_ext/thermal_protocolA/<3>/**/*.json
      results/acoustic_protocolA/<4>/**/*.json
      results/acoustic_wp5_ablation/ab_nofourier/**/*.json   # 实际存在的历史消融
      results/round2_audit/<4 个仅表 source manifest>.json   # 仅这 4 个 audit 文件
      PACKAGE_CONTENTS.txt PACKAGE_MANIFEST.json
    <out-dir>/<name>.tar.gz  <name>.tar.gz.sha256  <name>.manifest.json

JSON release-copy 规则
----------------------
只把**允许的私有 root 前缀**的机器绝对路径改写为项目相对路径，并把相对路径中常见的
``返稿意见/返稿修订工作区/results/...`` 前缀规范为 ``results/...``；数值、case_id、
split ids、seed、列表顺序与统计值保持完全不变（构建时逐一断言）。允许前缀仅限：
本 repo（Windows/Linux 私有 root 由运行环境提供；
仅规范化已知 workspace 相对路径前缀）及其 ``返稿意见/返稿修订工作区``
子目录。出现其它未知私有绝对路径时：不输出该内容，记录 JSON 字段位置并以非零状态
拒绝出包（不回显路径内容）。文本文件另做潜在凭据格式扫描，只报告文件名/行号/规则，
不打印匹配内容。

归档安全
--------
固定成员名单；拒绝空名、绝对路径、``..`` 组件、symlink/hardlink/device 成员；归档
元数据归一化（uid/gid/uname/gname/mtime=0）以便 archiveSHA 可复现；写出后重新读取并
逐文件校验 sha256 与数量。已存在的同名包一律拒绝覆盖。

用法
----
  python3 scripts/package_round2.py --list-only
  python3 scripts/package_round2.py \
      --out-dir dist/round2 --name featextractnet_round2_preparation

退出码：0 成功/仅清单；2 参数错误；3 拒绝（缺元数据/缺白名单文件/未知绝对路径/
凭据命中/同名包已存在等）。元数据尚缺时本脚本可先交付；Codex 提供
``zenodo/round2/{README.md,.zenodo.json,CITATION.cff,DATA_SOURCES.md}`` 后再运行。
"""

import argparse
import ast
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile

# ---------------------------------------------------------------------------
# fixed layout / whitelists
# ---------------------------------------------------------------------------
WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.realpath(os.path.join(WS, os.pardir, os.pardir))

DEFAULT_OUT_DIR = os.path.join(WS, 'dist', 'round2')
DEFAULT_NAME = 'featextractnet_round2_preparation'
ZENODO_ROUND2_DIR = os.path.join(WS, 'zenodo', 'round2')
OLD_ZENODO_DIR = os.path.join(WS, 'zenodo')

# Codex supplies these four files under zenodo/round2/ (do not author them here).
METADATA_FILES = ('README.md', '.zenodo.json', 'CITATION.cff', 'DATA_SOURCES.md')
LICENSE_FILE = 'LICENSE'
OLD_ENV_FILES = ('requirements.txt', 'requirements-extended.txt', 'environment.yml')

ADDITIONAL_CONFIG_FILES = ('r2_thermal_fem.json',)
CONFIG_FILES = ('splits_thermal.json', 'r2_geometry_protocols.json',
                'acoustic_cases.json')

# Scientific entry scripts explicitly approved for the preparation package.
SCIENTIFIC_SCRIPTS = (
    'exp_thermal', 'exp_acoustic', 'exp_ablation_thermal', 'exp_r2_geometry',
    'exp_r2_pinn_thermal', 'exp_r2_fdtd_convergence', 'exp_r2_cost_figures',
    'exp_r2_local_error', 'exp_r2_new_queries', 'fig_architecture_panels',
    'fig_r2_mechanism', 'generate_acoustic_cases', 'exp_acoustic_references',
)
# The two packaging scripts themselves are part of the reviewable package.
PACKAGING_SCRIPTS = ('package_round2', 'probe_round2_package',
                    'probe_round2_reference_package')
# Explicit scientific helpers required by whitelisted entries (listed in the manifest).
# exp_r2_pinn_thermal imports scripts.exp_pinn_thermal; fig_r2_mechanism loads
# scripts.exp_mechanism.py dynamically via importlib.util.spec_from_file_location
# (invisible to a plain static "from scripts import ..." scan).
SCRIPT_HELPER_ADDITIONS = {
    'exp_pinn_thermal': 'required by exp_r2_pinn_thermal (imports scripts.exp_pinn_thermal)',
    'exp_mechanism': 'dynamically loaded by fig_r2_mechanism via '
                     'importlib.util.spec_from_file_location; required to reproduce '
                     'the actual gate analysis',
}
ALL_SCRIPT_NAMES = (SCIENTIFIC_SCRIPTS + PACKAGING_SCRIPTS
                    + tuple(SCRIPT_HELPER_ADDITIONS))

# Existing geometry-spec JSON read by exp_r2_local_error / exp_r2_new_queries
# (INPUT/results/thermal_geometry_spec.json). Only this file from results/ root,
# never the whole results/ tree; release-copy sanitisation keeps values/IDs.
GEOMETRY_SPEC_PATH = 'results/thermal_geometry_spec.json'

# Frozen split snapshots referenced by accepted metric protocol.splits_config
# fields (exp_mechanism.compare_splits_snapshot requires the referenced file to
# exist; there is no fallback). Only these exact small JSONs from the frozen
# execution dirs are packaged - never the rest of those directories. They are
# byte-identical copies (no path normalisation needed; verified no private
# absolute paths, values/IDs unchanged).
EXECUTION_CONFIG_WHITELIST = (
    'results/round2_execution_20261021_v3/configs/splits_thermal.json',
    'results/round2_execution_20261021_v7/configs/splits_thermal.json',
    'results/round2_execution_20261021_v10/configs/splits_thermal.json',
    'results/round2_execution_20261022_v17/configs/splits_thermal.json',
    'results/round2_execution_20261022_v18/configs/splits_thermal.json',
)

# Current main held-out graph result (fresh FP32, graph seeds 0-4). This directory
# is packaged ONLY when all five seeds 0-4 have complete 500-epoch metrics and
# weight metadata and summary.json agrees with the five seed metrics; a partial
# directory must never be packaged as a completed main result.
MAIN_GRAPH_RESULT_DIR = 'results/round2_graph_five_seed_main/meshgraphnet'
MAIN_GRAPH_SEEDS = (0, 1, 2, 3, 4)
MAIN_GRAPH_EPOCHS = 500

# Sensitive real-plan requirement: these members must be present or packaging is
# refused (in addition to the dynamic-load detection below, so a silently missing
# dynamic helper cannot pass through a whitelist-only assertion).
REQUIRED_PACKAGE_ARCS = (
    'scripts/exp_mechanism.py',
    'results/thermal_geometry_spec.json',
) + EXECUTION_CONFIG_WHITELIST

# Completed-result JSON whitelist (only *.json, recursive; no weights/logs/npz).
RESULTS_DIR_WHITELIST = (
    'round2_thermal_symmetric',
    'round2_ablation_symmetric',
    'round2_ablation_maininit',
    'round2_acoustic_pointwise',
    'round2_acoustic_references',
    'round2_geometry',
    'round2_pinn_thermal_matched',
    'round2_mechanism',
    'round2_cost_figures',
    'round2_local_error',
    'round2_new_queries',
    'round2_remaining_closeout',
)

# Internal registry candidates are never packaged, even inside an included dir.
RESULTS_JSON_EXCLUDE = frozenset({
    'results/round2_new_queries/registry_candidate.json',
    # 51498 root-level failed-attempt artifacts, preserved but not part of 66/81
    'results/round2_remaining_closeout/seed_extension/acoustic_corrected_r2_attempt2/seed3/metrics_seed3.json',
    'results/round2_remaining_closeout/seed_extension/acoustic_corrected_r2_attempt2/summary.json',
    # intermediate BOM-bearing self-check copy; final_self_check.json is the clean record
    'results/round2_remaining_closeout/formal/final_self_check_remote.json',
    'results/round2_remaining_closeout/formal/recovery_inventory_remote.json',
    # internal matrix plans with server command paths; PACKAGE_PROTOCOL_INDEX is authoritative
    'results/round2_remaining_closeout/run_matrix.json',
    'results/round2_remaining_closeout/transolver_preflight/run_matrix.json',
    'results/round2_remaining_closeout/formal/package_unpack_check.json',
    'results/round2_remaining_closeout/formal/package_unpack_check_linux.json',
})

# Historically valid baselines that remain in the manuscript main tables
# (workspace-relative directories; only *.json recursively).
HISTORICAL_RESULT_DIRS = (
    'results/thermal_protocolA/mlp',
    'results/thermal_protocolA/pointnet',
    'results/thermal_protocolA/pointnetpp',
    'results/thermal_protocolA/dgcnn',
    'results/thermal_protocolA/pipn',
    'results/baselines_ext/thermal_protocolA/siren',
    'results/baselines_ext/thermal_protocolA/pointtransformer',
    'results/baselines_ext/thermal_protocolA/meshgraphnet',
    'results/acoustic_protocolA/mlp_c',
    'results/acoustic_protocolA/mlp_c_matched',
    'results/acoustic_protocolA/fno',
    'results/acoustic_protocolA/uno',
    'results/acoustic_wp5_ablation/ab_nofourier',
)

# The supplement review requested results/acoustic_protocolA/ab_nofourier; that
# path does not exist in this workspace and is NOT fabricated. The existing
# historical ablation directory below is used instead and the discrepancy is
# recorded in the plan/manifest (also confirmed by the audit source manifest).
REQUESTED_PATH_ALTERNATIVES = (
    {'requested': 'results/acoustic_protocolA/ab_nofourier',
     'actual': 'results/acoustic_wp5_ablation/ab_nofourier',
     'reason': 'requested path is absent in this workspace; the existing '
               'historical ablation directory is used (no fabrication)'},
)

# Only these four manuscript table-source manifests are taken from
# results/round2_audit/; every other audit file stays excluded.
AUDIT_SOURCE_MANIFEST_WHITELIST = (
    'results/round2_audit/thermal_manuscript_table_sources.json',
    'results/round2_audit/acoustic_manuscript_table_sources.json',
    'results/round2_audit/acoustic_ablation_manuscript_sources.json',
    'results/round2_audit/thermal_geometry_paired_r2_symmetric.json',
)


# --- R2-PACK-REFERENCE-03 incremental reference-checked additions -------------
# Scientific Python CLIs for the pressure-reference and independent-FEM modules.
ADDITIONAL_SCIENTIFIC_SCRIPTS = (
    'exp_remaining_closeout',
    'exp_r2_pressure_gauge', 'probe_r2_pressure_gauge',
    'exp_r2_thermal_fem', 'probe_r2_thermal_fem',
    'exp_r2_fem_model_eval', 'probe_r2_fem_model_eval',
    'postprocess_r2_thermal_fem_acceptance',
)
ADDITIONAL_CONFIG_FILES = ('r2_thermal_fem.json',)

# Exact JSON/Markdown records from the two accepted reference modules.  Old FEM v1
# files and pre-v3 diagnostics are deliberately absent.
ADDITIONAL_RESULT_JSON = (
    # pressure-reference sensitivity + independent pressure arrays manifest
    'results/round2_pressure_gauge/array_probe_manifest.json',
    'results/round2_pressure_gauge/array_probe_validation.json',
    'results/round2_pressure_gauge/baseline_inventory.json',
    'results/round2_pressure_gauge/delivery_50961.json',
    'results/round2_pressure_gauge/delivery_postreview.json',
    'results/round2_pressure_gauge/postreview_changes.md',
    'results/round2_pressure_gauge/probe_server_50961.json',
    'results/round2_pressure_gauge/protocol_snapshot.json',
    'results/round2_pressure_gauge/protocol_snapshot_postreview.json',
    'results/round2_pressure_gauge/raw_results.json',
    'results/round2_pressure_gauge/report.md',
    'results/round2_pressure_gauge/report_postreview.md',
    'results/round2_pressure_gauge/summary.json',
    'results/round2_pressure_gauge/summary_postreview.json',
    'results/round2_pressure_gauge/probe/probe_report.json',
    'results/round2_pressure_gauge/probe/fen_5seed20_cpu_validation.json',
    # independent FEM accepted v2/v3 records (v1 excluded)
    'results/round2_thermal_fem/FINAL_DELIVERY.md',
    'results/round2_thermal_fem/VERSION_CHAIN.md',
    'results/round2_thermal_fem/report.md',
    'results/round2_thermal_fem/protocol_snapshot.json',
    'results/round2_thermal_fem/v2_protocol_snapshot.json',
    'results/round2_thermal_fem/v3_protocol_snapshot.json',
    'results/round2_thermal_fem/v2_eps_sensitivity.log',
    'results/round2_thermal_fem/v3_fem_probe.json',
    'results/round2_thermal_fem/v3_postprocess/FINAL_DELIVERY_V3.md',
    'results/round2_thermal_fem/v3_postprocess/reacceptance.json',
    'results/round2_thermal_fem/v3_postprocess/reacceptance.md',
    'results/round2_thermal_fem/v3_eps/eps_differences.json',
    'results/round2_thermal_fem/v2/pilot/pilot_geom17_h0.08.json',
    'results/round2_thermal_fem/v2/levels/levels_geom17.json',
    'results/round2_thermal_fem/v2/representatives/representatives_46_96.json',
    'results/round2_thermal_fem/v2_model_eval/protocol_snapshot.json',
    'results/round2_thermal_fem/v2_model_eval/summary.json',
    'results/round2_thermal_fem/v2_model_eval/raw_metrics.json',
    'results/round2_thermal_fem/v2_model_eval/report.md',
    'results/round2_thermal_fem/job_ledger_v2.json',
    'results/round2_thermal_fem/job_ledger_v3.json',
    'results/round2_thermal_fem/hash_manifest_v2.json',
    'results/round2_thermal_fem/hash_manifest_v3.json',
    # accepted audit evidence for the two reference modules
    'results/round2_audit/r2_pressure_gauge_codex_acceptance.json',
    'results/round2_audit/r2_thermal_fem_codex_v2_acceptance.json',
)

# Binary scientific arrays; copied byte-for-byte and hash-verified at build and
# unpack time.  No raw data, weights, old FEM v1 fields, or full install logs.
ADDITIONAL_RESULT_ARRAYS = (
    'results/round2_pressure_gauge/array_probe_fen_seed0_geom17.npz',
    'results/round2_pressure_gauge/array_probe_graph_seed0_geom17.npz',
    'results/round2_thermal_fem/v2_reference/fem_reference_geom17_h0.08.npz',
    'results/round2_thermal_fem/v2_reference/fem_reference_geom17_h0.04.npz',
    'results/round2_thermal_fem/v2_reference/fem_reference_geom17_h0.02.npz',
    'results/round2_thermal_fem/v2_reference/fem_reference_geom46_h0.08.npz',
    'results/round2_thermal_fem/v2_reference/fem_reference_geom46_h0.04.npz',
    'results/round2_thermal_fem/v2_reference/fem_reference_geom46_h0.02.npz',
    'results/round2_thermal_fem/v2_reference/fem_reference_geom96_h0.08.npz',
    'results/round2_thermal_fem/v2_reference/fem_reference_geom96_h0.04.npz',
    'results/round2_thermal_fem/v2_reference/fem_reference_geom96_h0.02.npz',
    'results/round2_thermal_fem/v3_eps/fem_eps1e-09_geom17_h0.08.npz',
    'results/round2_thermal_fem/v3_eps/fem_eps1e-07_geom17_h0.08.npz',
    'results/round2_thermal_fem/model_eval/support_geom17.npz',
    'results/round2_thermal_fem/model_eval/support_geom46.npz',
    'results/round2_thermal_fem/model_eval/support_geom96.npz',
    'results/round2_thermal_fem/v2_model_eval/fem_reference_geom17.npz',
    'results/round2_thermal_fem/v2_model_eval/fem_reference_geom46.npz',
    'results/round2_thermal_fem/v2_model_eval/fem_reference_geom96.npz',
) + tuple(
    f'results/round2_thermal_fem/v2_model_eval/pred_{model}_seed{seed}_geom{geom}.npz'
    for model in ('FEN', 'graph') for seed in range(5) for geom in (46, 17, 96)
)

# Safe dependency evidence only (URLs/notes; no local prefix or install log).
ADDITIONAL_TEXT = (
    'results/round2_thermal_fem/environment/fem_conda_explicit_urls.txt',
    'results/round2_thermal_fem/environment/FEM_ENVIRONMENT.md',
    'src/models/baselines/transolver_official/LICENSE',
    'src/models/baselines/transolver_official/SOURCE.md',
    'results/round2_remaining_closeout/formal/acceptance_inputs/README.md',
    'results/round2_remaining_closeout/formal/FINAL_SELF_CHECK.md',
    'results/round2_remaining_closeout/formal/HISTORICAL_FAILURES.md',
    'results/round2_remaining_closeout/fno_uno_strict5/README.md',
)

# --- R2-FINAL-PACK-MERGE-01 accepted exact-pressure and FDTD evidence -------
ADDITIONAL_SCIENTIFIC_SCRIPTS += (
    'exp_r2_exact_pressure_reference',
    'exp_r2_pressure_reference_audit',
    'probe_r2_pressure_reference_audit',
)
ADDITIONAL_CONFIG_FILES += ('r2_pressure_reference_closure.json',)
ADDITIONAL_RESULT_JSON += (
    'results/round2_pressure_reference_closure/FINAL_DELIVERY.md',
    'results/round2_pressure_reference_closure/protocol_snapshot.json',
    'results/round2_pressure_reference_closure/artifact_manifest.json',
    'results/round2_pressure_reference_closure/exact_gauge_comparison.json',
    'results/round2_pressure_reference_closure/manufactured_solution.json',
    'results/round2_pressure_reference_closure/manufactured_polynomial_solution.json',
    'results/round2_pressure_reference_closure/manufactured_nontrivial_initial_solution.json',
    'results/round2_pressure_reference_closure/corrected_constraint_comparison.json',
    'results/round2_pressure_reference_closure/model_metric_sensitivity.json',
    'results/round2_pressure_reference_closure/all135_pressure_signatures.json',
    'results/round2_pressure_reference_closure/source_evidence_index.json',
    'results/round2_pressure_reference_closure/self_acceptance.json',
    'results/round2_pressure_reference_closure/probe_report.json',
    'results/round2_pressure_reference_closure/job_ledger.json',
    'results/round2_pressure_reference_closure/version_chain.json',
    'results/round2_pressure_reference_closure/exact/comparison_geom17_h004.json',
    'results/round2_pressure_reference_closure/exact/comparison_geom46_h004.json',
    'results/round2_pressure_reference_closure/exact/comparison_geom96_h004.json',
    'results/round2_pressure_reference_closure/exact/pilot_geom17_h004.json',
    'results/round2_audit/r2_pressure_closure_codex_acceptance.json',
    'results/round2_audit/r2_pressure_closure_codex_acceptance.md',
    # accepted FDTD aggregate + only 10 x meta/protocol/report (no fields/diagnostics)
    'results/round2_fdtd_convergence/aggregate.json',
    'results/round2_fdtd_convergence/aggregate.md',
    'results/round2_audit/r2_fdtd_complete_codex_acceptance.json',
) + tuple(
    f'results/round2_fdtd_convergence/{case}/{res}/{name}'
    for case in ('low100', 'high500')
    for res in ('coarse', 'mid', 'fine', 'mid_dt2', 'mid_dt4')
    for name in ('meta.json', 'protocol.json', 'report.md')
)
ADDITIONAL_RESULT_ARRAYS += (
    'results/round2_pressure_reference_closure/exact/fields_geom17_h004.npz',
    'results/round2_pressure_reference_closure/exact/fields_geom46_h004.npz',
    'results/round2_pressure_reference_closure/exact/fields_geom96_h004.npz',
)
ALL_SCRIPT_NAMES = tuple(ALL_SCRIPT_NAMES) + tuple(ADDITIONAL_SCIENTIFIC_SCRIPTS)
CONFIG_FILES = tuple(CONFIG_FILES) + tuple(ADDITIONAL_CONFIG_FILES)

SRC_EXCLUDED_DIRS = ('src/pipeline',)   # old asset, deliberately not shipped
# Explicitly not packaged until its own acceptance: unfinished exact-pressure module.
SRC_EXCLUDED_FILES = ()  # accepted exact-pressure module is now included

NAME_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,120}$')

# Only these two private root prefixes (and their 返稿意见/返稿修订工作区
# subdirectories) are eligible for absolute-path normalisation; anything else is
# reported as a blocked field without carrying its content.
ALLOWED_REPO_ROOTS = (
    ('repo_root_windows', PROJECT_ROOT.replace('\\', '/')),
    ('repo_root_linux', os.environ.get('R2_REPO_ROOT_LINUX', '')),
)
WORKSPACE_SUBDIR = ('返稿意见', '返稿修订工作区')
WORKSPACE_RELATIVE_PREFIX = '/'.join(WORKSPACE_SUBDIR)

EXCLUDED_CATEGORIES = (
    'raw data (HeatTransferData.npy, data/acoustic/*.npz; see DATA_SOURCES.md)',
    'model weights (*.pth, *.pt)',
    'logs/ and install logs (selected scientific arrays are included only via the explicit R2-PACK-REFERENCE-03 whitelist)',
    'results/ (only whitelisted JSON records and selected reference/prediction arrays)',
    'results/round2_audit/ except the four manuscript table-source manifests; '
    'results/registry.json (DSH/registry records); '
    'results/round2_new_queries/registry_candidate.json (internal registry candidate)',
    'removed/old baselines and architectures (thermal feat_extract/unif_therm; '
    'acoustic fourier_c/fourier_nc/unif_acous; joint results; legacy '
    'results/baselines 3-seed graph main comparison, historical only and superseded '
    'by results/round2_graph_five_seed_main + completed_comparison_graph_five_seed.json) '
    '-- only the manuscript-retained historical baseline list is included',
    'internal docs/, paper/, review/reply materials, ccfa-review-reports/',
    'scripts/sbatch/ (server-private hardcoded submissions)',
    'src/pipeline/ (old pipeline assets)',
    'zenodo/LICENSE-DATA (no third-party thermal data is redistributed here)',
)

# Credential formats: report only rule name + file/line, never the matched text.
CREDENTIAL_PATTERNS = (
    ('private_key_block', re.compile(r'-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----')),
    ('aws_access_key', re.compile(r'\bAKIA[0-9A-Z]{16}\b')),
    ('github_token', re.compile(r'\bgh[pousr]_[A-Za-z0-9]{20,}\b')),
    ('huggingface_token', re.compile(r'\bhf_[A-Za-z0-9]{20,}\b')),
    ('openai_style_key', re.compile(r'\bsk-[A-Za-z0-9]{20,}\b')),
    ('google_api_key', re.compile(r'\bAIza[0-9A-Za-z_-]{35}\b')),
    ('bearer_token', re.compile(r'(?i)\bbearer\s+[A-Za-z0-9._~+/-]{20,}')),
    ('secret_assignment', re.compile(
        r'(?i)\b(api[_-]?key|access[_-]?key|secret[_-]?key|password|passwd|'
        r'auth[_-]?token|access[_-]?token)\b\s*[:=]\s*["\']'
        r'(?=[A-Za-z0-9._-]{16,}["\'])[A-Za-z0-9._-]{16,}["\']')),
)

TEXT_SUFFIXES = ('.py', '.json', '.md', '.txt', '.yml', '.yaml', '.cff',
                 '.cfg', '.toml', '.sh', '.rst', '.in', '.sha256')
TEXT_NAMES = ('LICENSE', 'README', 'CITATION')


class PackageRefused(RuntimeError):
    """Controlled refusal: no package is written (or partial output is cleaned)."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _norm_path_text(path):
    return str(path).replace('\\', '/')


def _write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        fh.write(text)
    os.replace(tmp, path)


def _write_json(path, doc):
    _write_text(path, json.dumps(doc, indent=2, ensure_ascii=False) + '\n')


def _rel_ws(path, ws):
    return os.path.relpath(os.path.abspath(path), ws).replace(os.sep, '/')


def _safe_relpath(name):
    """Validate a package-internal member path (no absolute path / ``..`` / empty)."""
    norm = _norm_path_text(name)
    if not norm or norm.startswith('/') or re.match(r'^[A-Za-z]:', norm):
        raise PackageRefused(f'unsafe package member path (absolute/empty): {name!r}')
    parts = [p for p in norm.split('/') if p not in ('', '.')]
    if not parts or any(p == '..' for p in parts):
        raise PackageRefused(f'unsafe package member path (..): {name!r}')
    return '/'.join(parts)


# ---------------------------------------------------------------------------
# known project roots / JSON release-copy sanitisation
# ---------------------------------------------------------------------------
def default_known_roots(ws=None):
    """Allowed private root prefixes eligible for JSON-path normalisation.

    Only this repository (Windows and Linux deployment checkout) and its
    ``返稿意见/返稿修订工作区`` workspace are recognised; any other private absolute
    path is reported as a blocked field and never carried into the package.
    """
    roots = []
    for label, repo_root in ALLOWED_REPO_ROOTS:
        repo_root = _norm_path_text(repo_root).rstrip('/')
        workspace_label = ('workspace_windows' if label.endswith('windows')
                           else 'workspace_linux')
        roots.append({'label': label, 'root': repo_root})
        roots.append({'label': workspace_label,
                      'root': repo_root + '/' + WORKSPACE_RELATIVE_PREFIX})
    seen = set()
    unique = []
    for item in sorted(roots, key=lambda r: len(r['root']), reverse=True):
        key = item['root'].rstrip('/')
        if key in seen:
            continue
        seen.add(key)
        item['root'] = key
        unique.append(item)
    return unique


def _strip_workspace_relative_prefix(text):
    """Normalise ``返稿意见/返稿修订工作区/results/...`` to ``results/...``.

    Returns ``(new_value, changed)``; separators are normalised to ``/`` when the
    prefix matches. A ``..`` component in the remainder is rejected by the caller.
    """
    norm = _norm_path_text(text)
    while norm.startswith('./'):
        norm = norm[2:]
    prefix = WORKSPACE_RELATIVE_PREFIX
    if norm == prefix:
        return '.', True
    if not norm.startswith(prefix + '/'):
        return None, False
    remainder = norm[len(prefix) + 1:]
    parts = [p for p in remainder.split('/') if p not in ('', '.')]
    return '/'.join(parts), True


def _looks_absolute_path(text):
    if re.match(r'^[A-Za-z][A-Za-z0-9+.-]*://', text):
        return False                                    # URL
    if re.match(r'^[A-Za-z]:[\\/]', text):
        return True                                     # Windows drive path
    if text.startswith('\\\\'):
        return True                                     # UNC path
    return text.startswith('/')


def _match_known_root(text, roots):
    norm = _norm_path_text(text)
    for item in roots:
        root = item['root']
        if norm == root:
            return item, '.'
        if norm.startswith(root + '/'):
            rel = norm[len(root) + 1:]
            return item, rel
    return None, None


def _resolve_metric_split_path(value, ws, roots=None):
    """Resolve a metric ``protocol.splits_config`` value to workspace-relative.

    Source metric fields are authoritative: absolute paths under the allowed
    workspace/repo roots are used as-is; project-relative values are accepted.
    Returns ``(relative_path_or_None, absolute_or_None)``; unknown private
    absolute paths return ``(None, None)`` without echoing their content.
    """
    ws = os.path.realpath(ws)
    roots = roots if roots is not None else default_known_roots(ws)
    if _looks_absolute_path(value):
        matched, rel = _match_known_root(value, roots)
        if matched is None or rel is None:
            return None, None
        rel = rel.replace('\\', '/')
        if not rel or rel.startswith('../') or '/../' in rel:
            return None, None
        return rel, os.path.join(ws, *[p for p in rel.split('/') if p])
    norm = _norm_path_text(value)
    while norm.startswith('./'):
        norm = norm[2:]
    if not norm or norm.startswith('../') or '/../' in norm:
        return None, None
    return norm, os.path.join(ws, *[p for p in norm.split('/') if p])


def sanitize_json_document(doc, roots=None):
    """Return ``(new_doc, info)`` with allowed private paths rewritten relative.

    ``info`` = {'rewrites': [{'field','root','replacement'}],
                'blocked': [{'field','rule'}]}.

    Two normalisations are applied to string values, never to numbers/list order:
    * relative path strings prefixed with ``返稿意见/返稿修订工作区/`` (either
      separator style) become ``results/...`` etc.;
    * absolute paths under the allowed repository/workspace roots become
      project-relative paths.

    Unknown private absolute paths are blocked: their content is not carried over
    (value becomes ``None``) and the caller must refuse to write the package.
    Numeric values, case ids, split ids, seeds and statistics are untouched.
    """
    roots = roots if roots is not None else default_known_roots()
    rewrites = []
    blocked = []

    def walk(value, pointer):
        if isinstance(value, dict):
            return {str(k): walk(v, f'{pointer}.{k}' if pointer else str(k))
                    for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v, f'{pointer}[{i}]') for i, v in enumerate(value)]
        if not isinstance(value, str):
            return value
        stripped, changed = _strip_workspace_relative_prefix(value)
        if changed:
            parts = [p for p in stripped.split('/') if p not in ('', '.')]
            if any(p == '..' for p in parts):
                blocked.append({'field': pointer or '$',
                                'rule': 'path_escapes_workspace_prefix'})
                return None
            new_value = '/'.join(parts) if parts else '.'
            rewrites.append({'field': pointer or '$',
                             'root': 'workspace_relative_prefix',
                             'replacement': new_value})
            return new_value
        if not _looks_absolute_path(value):
            return value
        # Explicitly permitted environment field: the FEM runtime interpreter is
        # machine-specific; keep a package-relative recipe instead of a personal
        # conda prefix. No scientific value is changed.
        if (pointer in ('runtime.python',) or pointer.endswith('.runtime.python')
                and value.replace('\\', '/').startswith(
                    '/slurm' + '/home/' + 'jql-' + 'wanggulin' + '/miniconda3')):
            new_value = 'tools/envs/pinnap-fem/bin/python'
            rewrites.append({'field': pointer or '$',
                             'root': 'allowed_fem_runtime_field',
                             'replacement': new_value})
            return new_value
        matched, replacement = _match_known_root(value, roots)
        if matched is None:
            blocked.append({'field': pointer or '$',
                            'rule': 'unknown_absolute_path'})
            return None                                  # do not carry the content
        parts = [p for p in replacement.split('/') if p not in ('', '.')]
        if any(p == '..' for p in parts):
            blocked.append({'field': pointer or '$',
                            'rule': 'path_escapes_known_root'})
            return None
        new_value = '/'.join(parts) if parts else '.'
        rewrites.append({'field': pointer or '$', 'root': matched['label'],
                         'replacement': new_value})
        return new_value

    return walk(doc, ''), {'rewrites': rewrites, 'blocked': blocked}


def sanitize_text_document(text, roots):
    """Normalise only known project roots in non-scientific text.

    Scientific Python source files are copied byte-for-byte and never pass through
    this function.  Packaging/probe scripts and report/log text may contain the
    author's known checkout roots; those exact roots become ``<PROJECT_ROOT>``.
    Any remaining private absolute path is refused.
    """
    rewrites = []
    current = text
    for item in sorted(roots, key=lambda r: len(r.get('root', '')), reverse=True):
        root = item.get('root')
        if not root:
            continue
        variants = {root, root.replace('/', '\\'), root.replace('/', '\\\\')}
        for variant in sorted(variants, key=len, reverse=True):
            if variant and variant in current:
                count = current.count(variant)
                current = current.replace(variant, '<PROJECT_ROOT>')
                rewrites.append({'root': item.get('label', 'known_root'),
                                 'count': count})
    remaining = []
    for item in roots:
        root = item.get('root')
        if root and root in current:
            remaining.append(root)
    # Only actual author machine prefixes are refused; generic regex/fixture
    # strings used by the packaging self-tests are not machine paths.
    machine_markers = (
        'C:' + '/Users/',
        'C:' + '\\Users\\',
        '/slurm' + '/home/' + 'jql-' + 'wanggulin',
    )
    if remaining or any(marker in current for marker in machine_markers):
        raise PackageRefused('private absolute path remains in text member after normalisation')
    return current, {'rewrites': rewrites, 'blocked': []}


def assert_json_equivalent(original, packaged, rewrites):
    """Assert only the recorded path fields changed (numbers/ids/stats identical)."""
    allowed = {item['field'] for item in rewrites}

    def compare(a, b, pointer):
        location = pointer or '$'
        if isinstance(a, dict) or isinstance(b, dict):
            if not (isinstance(a, dict) and isinstance(b, dict)):
                raise PackageRefused(f'JSON type changed at {location}')
            if list(a.keys()) != list(b.keys()):
                raise PackageRefused(f'JSON keys/order changed at {location}')
            for key in a:
                compare(a[key], b[key], f'{pointer}.{key}' if pointer else str(key))
            return
        if isinstance(a, list) or isinstance(b, list):
            if not (isinstance(a, list) and isinstance(b, list)) or len(a) != len(b):
                raise PackageRefused(f'JSON list changed at {location}')
            for i, (x, y) in enumerate(zip(a, b)):
                compare(x, y, f'{pointer}[{i}]')
            return
        if isinstance(a, bool) or isinstance(b, bool):
            if a is not b:
                raise PackageRefused(f'JSON bool changed at {location}')
            return
        if isinstance(a, (int, float)) or isinstance(b, (int, float)):
            if type(a) is not type(b) or a != b:
                raise PackageRefused(f'JSON number changed at {location}')
            return
        if isinstance(a, str) or isinstance(b, str):
            if pointer not in allowed and a != b:
                raise PackageRefused(
                    f'JSON string changed outside approved path fields at {location}')
            return
        if a != b:
            raise PackageRefused(f'JSON value changed at {location}')

    compare(original, packaged, '')


# ---------------------------------------------------------------------------
# credential scan
# ---------------------------------------------------------------------------
def scan_text_for_credentials(text):
    """Return [{'line','rule'}] for potential credential formats; never the match."""
    hits = []
    for line_no, line in enumerate(text.splitlines(), 1):
        for rule, pattern in CREDENTIAL_PATTERNS:
            if pattern.search(line):
                hits.append({'line': line_no, 'rule': rule})
    return hits


def _is_text_file(path):
    base = os.path.basename(path)
    if base in TEXT_NAMES or base.startswith('README'):
        return True
    return path.lower().endswith(TEXT_SUFFIXES)


# ---------------------------------------------------------------------------
# package plan
# ---------------------------------------------------------------------------
def iter_src_py(ws):
    src_root = os.path.join(ws, 'src')
    out = []
    for root, dirs, files in os.walk(src_root):
        dirs[:] = sorted(d for d in dirs if d != '__pycache__')
        rel_dir = os.path.relpath(root, ws).replace(os.sep, '/')
        if any(rel_dir == exc or rel_dir.startswith(exc + '/')
               for exc in SRC_EXCLUDED_DIRS):
            dirs[:] = []
            continue
        for fn in sorted(files):
            if fn.endswith('.py'):
                rel = os.path.relpath(os.path.join(root, fn), ws).replace(os.sep, '/')
                if rel in SRC_EXCLUDED_FILES:
                    continue
                out.append(os.path.join(root, fn))
    return out


def _ast_imports(path):
    """Yield (module, level, names) for local-import dependency scanning."""
    try:
        tree = ast.parse(open(path, 'r', encoding='utf-8').read())
    except (SyntaxError, UnicodeDecodeError, OSError):
        return []
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            found.append((node.module or '', node.level,
                          [a.name for a in node.names]))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                found.append((alias.name, 0, []))
    return found


def _local_targets(module, level, names, source_path, ws):
    candidates = []
    base_dir = os.path.dirname(os.path.abspath(source_path))
    if level:
        prefix = base_dir
        for _ in range(level - 1):
            prefix = os.path.dirname(prefix)
        mod_path = module.replace('.', os.sep)
        stem = os.path.join(prefix, mod_path) if mod_path else prefix
        candidates += [stem + '.py', os.path.join(stem, '__init__.py')]
        for name in names:
            candidates.append(os.path.join(stem, name + '.py'))
        return candidates
    if not (module == 'scripts' or module.startswith('scripts.')
            or module == 'src' or module.startswith('src.')):
        return []
    stem = os.path.join(ws, module.replace('.', os.sep))
    candidates += [stem + '.py', os.path.join(stem, '__init__.py')]
    if module in ('scripts', 'src'):
        for name in names:
            candidates.append(os.path.join(stem, name + '.py'))
    return candidates


def scan_local_import_dependencies(members, ws):
    """Local project modules imported by included files that are not in the plan."""
    included = {m['arcname'] for m in members if m['kind'] in ('source', 'script')}
    missing = []
    for member in members:
        if member['kind'] not in ('source', 'script'):
            continue
        for module, level, names in _ast_imports(member['src']):
            for candidate in _local_targets(module, level, names,
                                            member['src'], ws):
                if not os.path.isfile(candidate):
                    continue
                rel = _rel_ws(candidate, ws)
                if rel not in included:
                    missing.append({'imported_module': module or '<relative>',
                                    'needed_file': rel,
                                    'imported_by': member['arcname']})
    # de-duplicate, keep deterministic order
    seen = set()
    unique = []
    for item in missing:
        key = (item['imported_module'], item['needed_file'], item['imported_by'])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


# ---------------------------------------------------------------------------
# dynamic source-load detection (importlib spec_from_file_location / run_path)
# ---------------------------------------------------------------------------
DYNAMIC_LOADER_NAMES = frozenset({
    'spec_from_file_location', 'SourceFileLoader', 'run_path',
})


def _collect_assignments(tree):
    """Map simple ``Name = <expr>`` assignments (module/function level) to expr."""
    assigns = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            assigns[str(node.targets[0].id)] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                and node.value is not None:
            assigns[str(node.target.id)] = node.value
    return assigns


def _string_literals(expr, assigns, depth=0):
    if expr is None or depth > 6:
        return []
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return [expr.value]
    if isinstance(expr, ast.Name) and expr.id in assigns:
        return _string_literals(assigns[expr.id], assigns, depth + 1)
    out = []
    for child in ast.iter_child_nodes(expr):
        out.extend(_string_literals(child, assigns, depth + 1))
    return out


def _dynamic_candidates(filename, source_path, ws):
    """Candidate workspace paths for a dynamic-loader filename reference."""
    norm = filename.replace('\\', '/')
    if norm.startswith('./'):
        norm = norm[2:]
    if os.path.isabs(filename) or re.match(r'^[A-Za-z]:[\\/]', filename):
        return [os.path.realpath(filename)]
    if '/' in norm:
        return [os.path.join(ws, *[p for p in norm.split('/') if p])]
    return [os.path.join(os.path.dirname(os.path.abspath(source_path)), filename),
            os.path.join(ws, 'scripts', filename),
            os.path.join(ws, filename)]


def scan_dynamic_source_dependencies(members, ws):
    """Detect dynamic source loads; report unresolved/missing helper scripts.

    A whitelist-only assertion cannot see ``importlib.util.spec_from_file_location``
    targets, so this AST scan resolves simple path expressions (including one-level
    variable assignments such as ``_EM_PATH = os.path.join(WS, "scripts", ...)``).
    Any dynamic loader call that cannot be resolved, or whose resolved ``.py`` file
    is outside the workspace / not included in the plan, is reported so packaging
    refuses instead of silently omitting the helper.
    """
    included = {m['arcname'] for m in members if m['kind'] in ('source', 'script')}
    references = []
    missing = []
    unresolved = []
    for member in members:
        if member['kind'] not in ('source', 'script'):
            continue
        try:
            text = open(member['src'], 'r', encoding='utf-8').read()
            tree = ast.parse(text)
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        assigns = _collect_assignments(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            loader = (func.attr if isinstance(func, ast.Attribute)
                      else func.id if isinstance(func, ast.Name) else '')
            if loader not in DYNAMIC_LOADER_NAMES:
                continue
            arg_expr = None
            if loader in ('spec_from_file_location', 'SourceFileLoader'):
                if len(node.args) >= 2:
                    arg_expr = node.args[1]
                else:
                    for keyword in node.keywords:
                        if keyword.arg in ('location', 'path'):
                            arg_expr = keyword.value
            else:  # run_path
                if node.args:
                    arg_expr = node.args[0]
                else:
                    for keyword in node.keywords:
                        if keyword.arg in ('path', 'path_name'):
                            arg_expr = keyword.value
            literals = _string_literals(arg_expr, assigns) if arg_expr is not None else []
            # also inspect the other call args for explicit .py path literals
            for other in node.args:
                if other is arg_expr:
                    continue
                literals.extend(_string_literals(other, assigns))
            py_literals = [item for item in literals if item.endswith('.py')]
            resolved = False
            for literal in py_literals:
                for candidate in _dynamic_candidates(literal, member['src'], ws):
                    if not os.path.isfile(candidate):
                        continue
                    if not _is_within(candidate, ws):
                        unresolved.append({
                            'loader': loader, 'loaded_by': member['arcname'],
                            'reference': literal,
                            'reason': 'dynamic source resolves outside the workspace'})
                        resolved = True
                        break
                    rel = _rel_ws(candidate, ws)
                    references.append({
                        'loader': loader, 'loaded_by': member['arcname'],
                        'reference': literal, 'resolved_file': rel,
                        'included': rel in included})
                    if rel not in included:
                        missing.append({
                            'loader': loader, 'loaded_by': member['arcname'],
                            'reference': literal, 'needed_file': rel})
                    resolved = True
                    break
                if resolved:
                    break
            if not resolved:
                unresolved.append({
                    'loader': loader, 'loaded_by': member['arcname'],
                    'argument_literals': py_literals or literals[:5],
                    'reason': 'dynamic source load could not be resolved to an '
                              'existing .py file inside the workspace'})
    for collection in (references, missing, unresolved):
        seen = set()
        unique = []
        for item in collection:
            key = tuple(sorted((str(k), str(v)) for k, v in item.items()))
            if key not in seen:
                seen.add(key)
                unique.append(item)
        collection[:] = unique
    return {'references': references, 'missing': missing,
            'unresolved': unresolved}


def _iter_splits_config_values(doc, pointer=''):
    """Yield (json_pointer, splits_config value) from a metric JSON document."""
    if isinstance(doc, dict):
        for key, value in doc.items():
            path = f'{pointer}.{key}' if pointer else str(key)
            if key == 'splits_config' and isinstance(value, str):
                yield path, value
            else:
                yield from _iter_splits_config_values(value, path)
    elif isinstance(doc, list):
        for index, value in enumerate(doc):
            yield from _iter_splits_config_values(value, f'{pointer}[{index}]')


def scan_metric_split_references(members, ws, roots=None):
    """Check every packaged metric's protocol.splits_config path is in the plan.

    The source metrics are authoritative: their splits_config fields decide which
    frozen split snapshots are required (no hardcoded date assumption). A
    referenced file that is not included in the plan (or an unresolvable private
    absolute path) makes packaging refuse instead of shipping a broken closure.
    """
    included = {member['arcname'] for member in members}
    references = {}
    missing = []
    unresolved = []
    for member in members:
        if member['kind'] not in ('result_json', 'main_result_json',
                                  'source_manifest'):
            continue
        try:
            with open(member['src'], 'r', encoding='utf-8') as fh:
                doc = json.load(fh)
        except (ValueError, OSError):
            continue
        for pointer, value in _iter_splits_config_values(doc):
            rel, _abs = _resolve_metric_split_path(value, ws, roots)
            if rel is None:
                unresolved.append({'referenced_by': member['arcname'],
                                   'field': pointer or '$',
                                   'reason': 'unknown private absolute path'})
                continue
            references[rel] = references.get(rel, 0) + 1
            if rel not in included:
                missing.append({'needed_file': rel,
                                'referenced_by': member['arcname'],
                                'field': pointer or '$'})
    for collection in (missing, unresolved):
        seen = set()
        unique = []
        for item in collection:
            key = tuple(sorted((str(k), str(v)) for k, v in item.items()))
            if key not in seen:
                seen.add(key)
                unique.append(item)
        collection[:] = unique
    return {'references': references, 'missing': missing,
            'unresolved': unresolved}


def validate_main_graph_result(ws, roots=None):
    """Gate the fresh FP32 five-seed graph main result before packaging.

    Complete means: seeds 0-4 each have ``metrics_seed{k}.json`` (protocol.epochs
    == 500, curves.epochs length == 500, seed k) plus an existing
    ``checkpoint_best.pth`` weight file, and ``summary.json`` reports n_seeds=5,
    seeds=[0..4], epochs=500 with aggregate mean/sample-std matching the five seed
    metrics. Partial/inconsistent data is never silently packaged.
    """
    rel_dir = MAIN_GRAPH_RESULT_DIR
    base = os.path.join(ws, *rel_dir.split('/'))
    result = {
        'dir': rel_dir,
        'status': 'absent',
        'approved_files': [],
        'pending_missing': [],
        'pending_issues': [],
        'summary_consistent': None,
        'seeds_found': [],
        'summary_seeds': None,
    }
    if not os.path.isdir(base):
        return result
    missing = []
    issues = []
    metric_docs = {}
    metric_splits = {}
    payload_rels = []
    for seed in MAIN_GRAPH_SEEDS:
        metrics_rel = f'{rel_dir}/seed{seed}/metrics_seed{seed}.json'
        ckpt_rel = f'{rel_dir}/seed{seed}/checkpoint_best.pth'
        metrics_path = os.path.join(ws, *metrics_rel.split('/'))
        ckpt_path = os.path.join(ws, *ckpt_rel.split('/'))
        if not os.path.isfile(metrics_path):
            missing.append(metrics_rel)
        if not os.path.isfile(ckpt_path):
            missing.append(ckpt_rel)
        if not os.path.isfile(metrics_path):
            continue
        try:
            with open(metrics_path, 'r', encoding='utf-8') as fh:
                doc = json.load(fh)
        except (ValueError, OSError) as exc:
            issues.append(f'{metrics_rel}: unreadable JSON ({type(exc).__name__})')
            continue
        protocol = doc.get('protocol') or {}
        curves = doc.get('curves') or {}
        epochs = curves.get('epochs')
        if doc.get('seed') != seed:
            issues.append(f'{metrics_rel}: seed field {doc.get("seed")!r} != {seed}')
        if protocol.get('epochs') != MAIN_GRAPH_EPOCHS:
            issues.append(f'{metrics_rel}: protocol.epochs != {MAIN_GRAPH_EPOCHS}')
        if not isinstance(epochs, list) or len(epochs) != MAIN_GRAPH_EPOCHS:
            issues.append(f'{metrics_rel}: curves.epochs length != {MAIN_GRAPH_EPOCHS}')
        if not isinstance((doc.get('test') or {}).get('agg'), dict):
            issues.append(f'{metrics_rel}: missing test.agg metrics')
        checkpoint_best = doc.get('checkpoint_best')
        if not isinstance(checkpoint_best, str) \
                or os.path.basename(checkpoint_best) != 'checkpoint_best.pth':
            issues.append(f'{metrics_rel}: checkpoint_best weight metadata missing/incorrect')
        # the metric's own protocol.splits_config field is authoritative (read it,
        # never rewrite it to hide a path/date difference)
        split_value = protocol.get('splits_config')
        if not isinstance(split_value, str) or not split_value:
            issues.append(f'{metrics_rel}: protocol.splits_config missing')
        else:
            split_rel, _split_abs = _resolve_metric_split_path(split_value, ws, roots)
            if split_rel is None:
                issues.append(f'{metrics_rel}: protocol.splits_config is not a '
                              f'usable workspace path')
            else:
                metric_splits[seed] = split_rel
                if split_rel not in EXECUTION_CONFIG_WHITELIST:
                    issues.append(
                        f'{metrics_rel}: protocol.splits_config {split_rel} is not '
                        f'in the frozen split snapshot whitelist')
        metric_docs[seed] = doc
        payload_rels.append(metrics_rel)
    summary_rel = f'{rel_dir}/summary.json'
    summary_path = os.path.join(ws, *summary_rel.split('/'))
    summary = None
    if not os.path.isfile(summary_path):
        missing.append(summary_rel)
    else:
        try:
            with open(summary_path, 'r', encoding='utf-8') as fh:
                summary = json.load(fh)
        except (ValueError, OSError) as exc:
            issues.append(f'{summary_rel}: unreadable JSON ({type(exc).__name__})')
    summary_consistent = False
    if summary is not None:
        result['summary_seeds'] = summary.get('seeds')
        if summary.get('n_seeds') != len(MAIN_GRAPH_SEEDS):
            issues.append(f'{summary_rel}: n_seeds != {len(MAIN_GRAPH_SEEDS)}')
        if list(summary.get('seeds') or []) != list(MAIN_GRAPH_SEEDS):
            issues.append(f'{summary_rel}: seeds != {list(MAIN_GRAPH_SEEDS)}')
        if summary.get('epochs') != MAIN_GRAPH_EPOCHS:
            issues.append(f'{summary_rel}: epochs != {MAIN_GRAPH_EPOCHS}')
        agg = summary.get('agg') or {}
        consistency_issues = []
        if len(metric_docs) == len(MAIN_GRAPH_SEEDS) and agg:
            for metric_key, block in agg.items():
                for stat in ('mean', 'std'):
                    values_by_field = block.get(stat) if isinstance(block, dict) else None
                    if not isinstance(values_by_field, dict):
                        consistency_issues.append(f'{metric_key}.{stat} missing')
                        continue
                    for field, target in values_by_field.items():
                        data = []
                        for seed in MAIN_GRAPH_SEEDS:
                            agg_doc = (metric_docs[seed].get('test') or {}).get('agg') \
                                or {}
                            metric_block = agg_doc.get(metric_key) or {}
                            value = metric_block.get(field)
                            if isinstance(value, (int, float)):
                                data.append(float(value))
                        if target is None:
                            if data:
                                consistency_issues.append(
                                    f'{metric_key}.{stat}.{field} is null but metrics exist')
                            continue
                        if len(data) != len(MAIN_GRAPH_SEEDS):
                            consistency_issues.append(
                                f'{metric_key}.{stat}.{field} lacks 5 numeric seed values')
                            continue
                        mean = sum(data) / len(data)
                        var = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
                        expected = mean if stat == 'mean' else var ** 0.5
                        if abs(expected - float(target)) > 1e-9:
                            consistency_issues.append(
                                f'{metric_key}.{stat}.{field} summary {target} != '
                                f'recomputed {expected}')
            summary_consistent = not consistency_issues
            issues.extend(f'{summary_rel}: {item}' for item in consistency_issues)
        else:
            summary_consistent = False
        result['summary_consistent'] = summary_consistent
    result['pending_missing'] = missing
    result['pending_issues'] = issues
    result['seeds_found'] = sorted(metric_docs)
    result['metric_split_config_paths'] = sorted(set(metric_splits.values()))
    # when all five metrics are present, their own splits_config fields must point
    # at both frozen snapshots actually recovered (v17 for seeds 0-2, v18 for 3-4);
    # this is read from the metrics, not from a hardcoded date assumption.
    if len(metric_docs) == len(MAIN_GRAPH_SEEDS):
        expected_splits = {EXECUTION_CONFIG_WHITELIST[3],
                           EXECUTION_CONFIG_WHITELIST[4]}
        actual_splits = set(metric_splits.values())
        if actual_splits != expected_splits:
            issues.append(
                'five main metrics reference frozen split snapshots '
                f'{sorted(actual_splits)}, expected {sorted(expected_splits)} '
                '(read from the metric protocol.splits_config fields)')
    if not missing and not issues and summary_consistent:
        result['status'] = 'complete'
        result['approved_files'] = payload_rels + [summary_rel]
    else:
        result['status'] = 'incomplete'
    return result


def collect_plan(ws=WS, zenodo_dir=None, old_zenodo_dir=None, roots=None):
    """Build the fixed package member list (no writes)."""
    ws = os.path.realpath(ws)
    roots = roots if roots is not None else default_known_roots(ws)
    zenodo_dir = os.path.realpath(zenodo_dir or os.path.join(ws, 'zenodo', 'round2'))
    old_zenodo = os.path.realpath(old_zenodo_dir or os.path.join(ws, 'zenodo'))
    members = []
    missing_metadata = []
    missing_required = []

    def add(kind, src, arcname, sanitized=False, note=''):
        src = os.path.abspath(src)
        if not os.path.isfile(src):
            missing_required.append({'kind': kind, 'path': _rel_ws(src, ws),
                                     'reason': 'missing or not a regular file'})
            return
        if os.path.islink(src):
            raise PackageRefused(f'symlink source is not allowed: {src}')
        members.append({'kind': kind, 'src': src,
                        'arcname': _safe_relpath(arcname),
                        'sanitized': sanitized, 'note': note})

    for name in METADATA_FILES:
        path = os.path.join(zenodo_dir, name)
        if not os.path.isfile(path):
            missing_metadata.append(name)
            continue
        add('metadata', path, name,
            note='provided by Codex under zenodo/round2/')
    for name in (LICENSE_FILE,) + OLD_ENV_FILES:
        kind = 'license' if name == LICENSE_FILE else 'environment'
        path = os.path.join(old_zenodo, name)
        if not os.path.isfile(path):
            missing_required.append({'kind': kind,
                                     'path': _rel_ws(path, ws),
                                     'reason': 'missing legacy zenodo file'})
            continue
        add(kind, path, name, note='copied from original zenodo/ release')
    for src in iter_src_py(ws):
        add('source', src, _rel_ws(src, ws))
    for name in ALL_SCRIPT_NAMES:
        path = os.path.join(ws, 'scripts', name + '.py')
        note = SCRIPT_HELPER_ADDITIONS.get(name, '')
        if not os.path.isfile(path):
            missing_required.append({'kind': 'script',
                                     'path': _rel_ws(path, ws),
                                     'reason': 'whitelisted entry/helper missing'})
            continue
        if name in PACKAGING_SCRIPTS:
            add('packaging_script', path, f'scripts/{name}.py', sanitized='text',
                note='packaging/probe tool; only known private roots normalised; '
                     'not a scientific solver source')
        else:
            add('script', path, f'scripts/{name}.py', note=note)
    for name in CONFIG_FILES:
        path = os.path.join(ws, 'configs', name)
        if not os.path.isfile(path):
            missing_required.append({'kind': 'config',
                                     'path': _rel_ws(path, ws),
                                     'reason': 'whitelisted config missing'})
            continue
        add('config', path, f'configs/{name}',
            sanitized=(name in ADDITIONAL_CONFIG_FILES),
            note='shared config read-only copy' +
                 ('; JSON paths normalised only, values/fields unchanged'
                  if name in ADDITIONAL_CONFIG_FILES else ''))
    geometry_spec = os.path.join(ws, *GEOMETRY_SPEC_PATH.split('/'))
    if not os.path.isfile(geometry_spec):
        missing_required.append(
            {'kind': 'geometry_spec', 'path': GEOMETRY_SPEC_PATH,
             'reason': 'existing geometry spec JSON is required by '
                       'exp_r2_local_error / exp_r2_new_queries'})
    else:
        add('geometry_spec', geometry_spec, GEOMETRY_SPEC_PATH, sanitized=True,
            note='existing geometry specification (read by exp_r2_local_error / '
                 'exp_r2_new_queries); release-copy JSON, values/IDs unchanged')
    # frozen split snapshots: exact files only, byte-identical config copies
    for rel in EXECUTION_CONFIG_WHITELIST:
        path = os.path.join(ws, *rel.split('/'))
        if not os.path.isfile(path):
            missing_required.append(
                {'kind': 'execution_config', 'path': rel,
                 'reason': 'frozen split snapshot referenced by metric '
                           'protocol.splits_config is missing (no fallback allowed)'})
            continue
        add('execution_config', path, rel, sanitized=False,
            note='frozen split snapshot referenced by metric protocol.splits_config; '
                 'byte-identical config copy (no path normalisation needed)')
    main_result = validate_main_graph_result(ws, roots)
    if main_result['status'] == 'complete':
        for rel in main_result['approved_files']:
            add('main_result_json', os.path.join(ws, *rel.split('/')), rel,
                sanitized=True,
                note='accepted main held-out graph result (five seeds, 500 epochs, '
                     'summary verified against seed metrics); release-copy JSON')
    # incomplete main result: include nothing from it, remember why for the refusal
    pending_main_result = (main_result if main_result['status'] == 'incomplete'
                           else None)
    results_counts = {}
    for dir_name in RESULTS_DIR_WHITELIST:
        base = os.path.join(ws, 'results', dir_name)
        if not os.path.isdir(base):
            missing_required.append({'kind': 'results',
                                     'path': _rel_ws(base, ws),
                                     'reason': 'whitelisted results dir missing'})
            continue
        n = 0
        for root, dirs, files in os.walk(base):
            dirs[:] = sorted(d for d in dirs if d != '__pycache__')
            for fn in sorted(files):
                if not fn.endswith('.json'):
                    continue
                path = os.path.join(root, fn)
                rel = _rel_ws(path, ws)
                if rel in RESULTS_JSON_EXCLUDE:
                    continue
                add('result_json', path, rel, sanitized=True,
                    note='release-copy: JSON paths normalised, values unchanged')
                n += 1
        results_counts[dir_name] = n
    historical_counts = {}
    for rel_dir in HISTORICAL_RESULT_DIRS:
        base = os.path.join(ws, rel_dir)
        if not os.path.isdir(base):
            missing_required.append(
                {'kind': 'historical_results', 'path': rel_dir,
                 'reason': 'whitelisted historical baseline dir missing'})
            continue
        n = 0
        for root, dirs, files in os.walk(base):
            dirs[:] = sorted(d for d in dirs if d != '__pycache__')
            for fn in sorted(files):
                if not fn.endswith('.json'):
                    continue
                path = os.path.join(root, fn)
                add('result_json', path, _rel_ws(path, ws), sanitized=True,
                    note='historical valid baseline (manuscript main table); '
                         'release-copy JSON, values unchanged')
                n += 1
        historical_counts[rel_dir] = n
    missing_source_manifests = []
    source_manifests = []
    for rel in AUDIT_SOURCE_MANIFEST_WHITELIST:
        path = os.path.join(ws, rel)
        if not os.path.isfile(path):
            missing_source_manifests.append(rel)
            continue
        add('source_manifest', path, rel, sanitized=True,
            note='manuscript table source manifest (only these audit files included)')
        source_manifests.append(rel)
    # R2-PACK-REFERENCE-03 exact incremental records and selected arrays.
    additional_json_count = 0
    for rel in ADDITIONAL_RESULT_JSON:
        path = os.path.join(ws, *rel.split('/'))
        if not os.path.isfile(path):
            missing_required.append({'kind': 'result_json', 'path': rel,
                                     'reason': 'incremental reference record missing'})
            continue
        if rel.endswith('.json'):
            add('result_json', path, rel, sanitized=True,
                note='incremental reference-checked result JSON; release-copy paths normalised, values unchanged')
        else:
            add('report_text', path, rel, sanitized='text',
                note='incremental reference-checked report/log; only known private roots normalised')
        additional_json_count += 1
    additional_array_count = 0
    for rel in ADDITIONAL_RESULT_ARRAYS:
        path = os.path.join(ws, *rel.split('/'))
        if not os.path.isfile(path):
            missing_required.append({'kind': 'result_array', 'path': rel,
                                     'reason': 'incremental reference array missing'})
            continue
        add('result_array', path, rel, sanitized=False,
            note='byte-identical scientific array; sha256 verified at build and unpack')
        additional_array_count += 1
    additional_text_count = 0
    for rel in ADDITIONAL_TEXT:
        path = os.path.join(ws, *rel.split('/'))
        if not os.path.isfile(path):
            missing_required.append({'kind': 'env_note', 'path': rel,
                                     'reason': 'safe FEM dependency evidence missing'})
            continue
        add('env_note', path, rel, sanitized='text',
            note='safe dependency URLs/note only; no local prefix or install log')
        additional_text_count += 1

    requested_missing = []
    for item in REQUESTED_PATH_ALTERNATIVES:
        requested = os.path.join(ws, item['requested'])
        actual = os.path.join(ws, item['actual'])
        if not os.path.exists(requested) and os.path.exists(actual):
            requested_missing.append(dict(item))
        elif not os.path.exists(requested) and not os.path.exists(actual):
            missing_required.append(
                {'kind': 'historical_results', 'path': item['requested'],
                 'reason': 'requested path absent and documented alternative also absent'})

    arcnames = [m['arcname'] for m in members]
    dupes = sorted({a for a in arcnames if arcnames.count(a) > 1})
    if dupes:
        raise PackageRefused(f'duplicate package members: {dupes[:10]}')
    # sensitive real-plan requirement: explicit presence of the geometry spec and
    # the dynamically loaded mechanism helper; missing means REFUSE, not warn.
    missing_required_arcs = [arc for arc in REQUIRED_PACKAGE_ARCS
                             if arc not in set(arcnames)]
    for arc in missing_required_arcs:
        missing_required.append(
            {'kind': 'required_sensitive', 'path': arc,
             'reason': 'sensitive required input missing from the real plan'})
    dynamic = scan_dynamic_source_dependencies(members, ws)
    # source metrics decide the frozen split dependencies; every referenced path
    # must already be in the plan (closure), otherwise packaging refuses.
    metric_splits = scan_metric_split_references(members, ws, roots)
    for item in metric_splits['missing']:
        missing_required.append(
            {'kind': 'metric_split_config', 'path': item['needed_file'],
             'reason': f"referenced by metric protocol.splits_config in "
                       f"{item['referenced_by']} but not included in the plan"})
    for item in metric_splits['unresolved']:
        missing_required.append(
            {'kind': 'metric_split_config', 'path': '(unresolved private path)',
             'reason': f"{item['referenced_by']} {item['field']}: "
                       f"{item['reason']} (source value not rewritten)"})
    counts = {}
    for member in members:
        counts[member['kind']] = counts.get(member['kind'], 0) + 1
    total_bytes = sum(os.path.getsize(m['src']) for m in members)
    return {
        'workspace': ws,
        'zenodo_dir': zenodo_dir,
        'old_zenodo_dir': old_zenodo,
        'members': members,
        'missing_metadata': missing_metadata,
        'missing_required': missing_required,
        'required_arcs': list(REQUIRED_PACKAGE_ARCS),
        'missing_required_arcs': missing_required_arcs,
        'dynamic_helper_references': dynamic['references'],
        'missing_dynamic_helpers': dynamic['missing'],
        'unresolved_dynamic_helpers': dynamic['unresolved'],
        'metric_split_references': metric_splits['references'],
        'missing_metric_split_snapshots': metric_splits['missing'],
        'unresolved_metric_split_references': metric_splits['unresolved'],
        'missing_source_manifests': missing_source_manifests,
        'source_manifests': source_manifests,
        'additional_json_count': additional_json_count,
        'additional_array_count': additional_array_count,
        'additional_text_count': additional_text_count,
        'requested_missing': requested_missing,
        'main_graph_result': main_result,
        'pending_main_result': pending_main_result,
        'historical_result_dirs': list(HISTORICAL_RESULT_DIRS),
        'helper_additions': SCRIPT_HELPER_ADDITIONS,
        'missing_helpers': scan_local_import_dependencies(members, ws),
        'results_counts': results_counts,
        'historical_counts': historical_counts,
        'counts': counts,
        'total_bytes': total_bytes,
    }


# ---------------------------------------------------------------------------
# metadata / policy validation
# ---------------------------------------------------------------------------
def _normalize_license_id(value):
    """Normalise a licence identifier/name for MIT comparison (no new grant)."""
    if not isinstance(value, str):
        return None
    norm = value.strip().strip('"\'').strip().lower()
    for prefix in ('spdx:', 'license:', 'licence:'):
        if norm.startswith(prefix):
            norm = norm[len(prefix):].strip()
    norm = norm.replace('_', '-')
    norm = ' '.join(norm.split())
    if norm in ('mit', 'mit license', 'mit licence', 'expat'):
        return 'mit'
    return norm or None


def _parse_cff_licenses(text):
    """Extract CFF ``license`` value(s) without a YAML dependency."""
    licenses = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith('license:'):
            continue
        inline = stripped[len('license:'):].strip()
        if inline:
            licenses.append(inline.strip('"\''))
            continue
        # YAML block list form: license:\n  - MIT
        for follow in lines[index + 1:]:
            follow = follow.strip()
            if not follow:
                continue
            if follow.startswith('-'):
                licenses.append(follow.lstrip('-').strip().strip('"\''))
                continue
            break
        break
    return licenses


def _is_mit_notice(text):
    lowered = text.lower()
    return ('permission is hereby granted, free of charge' in lowered
            and 'the software is provided "as is"' in lowered)


def license_alignment_checks(plan):
    """Preflight: Zenodo declaration, CFF declaration, existing MIT notice agree.

    This is a consistency check only. It never edits LICENSE and never grants a
    new licence or re-licenses third-party data (LICENSE-DATA is not shipped).
    """
    errors = []
    zenodo_license = None
    cff_licenses = []
    mit_notice = False
    zenodo_path = os.path.join(plan['zenodo_dir'], '.zenodo.json')
    if not os.path.isfile(zenodo_path):
        errors.append('.zenodo.json is missing')
    else:
        try:
            with open(zenodo_path, 'r', encoding='utf-8') as fh:
                doc = json.load(fh)
            zenodo_license = _normalize_license_id(doc.get('license'))
            if zenodo_license != 'mit':
                errors.append(
                    f'.zenodo.json license declaration {doc.get("license")!r} '
                    f'is not MIT')
        except (ValueError, OSError) as exc:
            errors.append(f'.zenodo.json cannot be parsed ({type(exc).__name__})')
    cff_path = os.path.join(plan['zenodo_dir'], 'CITATION.cff')
    if not os.path.isfile(cff_path):
        errors.append('CITATION.cff is missing')
    else:
        try:
            cff_text = open(cff_path, 'r', encoding='utf-8',
                            errors='replace').read()
        except OSError as exc:
            cff_text = ''
            errors.append(f'CITATION.cff cannot be read ({type(exc).__name__})')
        raw_licenses = _parse_cff_licenses(cff_text)
        cff_licenses = [_normalize_license_id(item) for item in raw_licenses]
        if not cff_licenses:
            errors.append('CITATION.cff has no license field')
        elif any(item != 'mit' for item in cff_licenses):
            errors.append(f'CITATION.cff license declarations {raw_licenses!r} '
                          f'are not MIT')
    license_path = os.path.join(plan['old_zenodo_dir'], LICENSE_FILE)
    if not os.path.isfile(license_path):
        errors.append('zenodo/LICENSE is missing')
    else:
        license_text = open(license_path, 'r', encoding='utf-8',
                            errors='replace').read()
        mit_notice = _is_mit_notice(license_text)
        if not mit_notice:
            errors.append('zenodo/LICENSE does not contain the expected MIT '
                          'permission notice')
    return {
        'aligned': not errors,
        'zenodo_license': zenodo_license,
        'cff_licenses': cff_licenses,
        'mit_notice_present': mit_notice,
        'license_sha256': checks_license_sha(license_path),
        'errors': errors,
        'note': 'Consistency/preflight only: verifies the existing MIT code '
                'licence declarations; it grants no new licence and does not '
                'relicense third-party data (LICENSE-DATA is not distributed).',
    }


def checks_license_sha(path):
    return sha256_file(path) if os.path.isfile(path) else None


def validate_metadata_inputs(plan, roots=None):
    """Check Codex metadata, MIT licence alignment and environment provenance."""
    checks = {'metadata': {}, 'warnings': []}
    for name in METADATA_FILES:
        path = os.path.join(plan['zenodo_dir'], name)
        entry = {'present': os.path.isfile(path),
                 'bytes': os.path.getsize(path) if os.path.isfile(path) else 0}
        checks['metadata'][name] = entry
    zenodo_path = os.path.join(plan['zenodo_dir'], '.zenodo.json')
    if os.path.isfile(zenodo_path):
        try:
            with open(zenodo_path, 'r', encoding='utf-8') as fh:
                json.load(fh)
            checks['metadata']['.zenodo.json']['valid_json'] = True
        except (ValueError, OSError) as exc:
            checks['metadata']['.zenodo.json']['valid_json'] = False
            checks['warnings'].append(f'.zenodo.json is not valid JSON: {exc!r}')
    readme_path = os.path.join(plan['zenodo_dir'], 'README.md')
    if os.path.isfile(readme_path):
        text = open(readme_path, 'r', encoding='utf-8', errors='replace').read()
        lowered = text.lower()
        checks['readme_mentions_history'] = any(
            token in lowered for token in
            ('历史', 'historical', 'inherited', 'previous release', '旧版'))
        checks['readme_mentions_measured'] = any(
            token in lowered for token in
            ('实测', 'measured', 'record python', 'recorded python',
             'server runs record', 'current version'))
        if not checks['readme_mentions_history'] or not checks['readme_mentions_measured']:
            checks['warnings'].append(
                'README.md does not clearly mention both the historical environment '
                'and the measured/current versions (Codex README must explain them).')
    data_src = os.path.join(plan['zenodo_dir'], 'DATA_SOURCES.md')
    if os.path.isfile(data_src):
        text = open(data_src, 'r', encoding='utf-8', errors='replace').read()
        checks['data_sources_mentions_sha'] = 'sha' in text.lower()
        if not checks['data_sources_mentions_sha']:
            checks['warnings'].append(
                'DATA_SOURCES.md does not mention a sha/sha256 for the original '
                'thermal data (Codex must document download + sha).')
    license_path = os.path.join(plan['old_zenodo_dir'], LICENSE_FILE)
    checks['license_mit'] = False
    if os.path.isfile(license_path):
        text = open(license_path, 'r', encoding='utf-8', errors='replace').read()
        checks['license_mit'] = 'mit license' in text.lower()
        checks['license_sha256'] = sha256_file(license_path)
    if not checks['license_mit']:
        checks['warnings'].append(
            'legacy zenodo/LICENSE does not look like the MIT licence; refusing to '
            'ship it as the code licence.')
    checks['license_alignment'] = license_alignment_checks(plan)
    if not checks['license_alignment']['aligned']:
        checks['warnings'].append(
            'metadata licence declarations are not aligned: '
            + '; '.join(checks['license_alignment']['errors']))
    return checks


# ---------------------------------------------------------------------------
# generated package files
# ---------------------------------------------------------------------------
def pending_status_document():
    return {
        'generated_by': 'scripts/package_round2.py',
        'note': ('Status only. Pending items have NO results in this package and no '
                 'scientific/complete claim is made; completed items retain their '
                 'scope notes. Do not cite pending items as completed.'),
        'pending_items': [
            {'id': 'R2-FDTD-CONV-01',
             'topic': 'pressure space/time convergence (Python-FDTD)',
             'status': 'pending',
             'results_included': False},
        ],
        'completed_items_scope_notes': [
            {'id': 'R2-PRESSURE-GAUGE',
             'topic': 'raw vs node-mean-centered pressure sensitivity',
             'status': 'completed sensitivity; full FDTD pressure-space/time convergence remains pending',
             'results_included': True,
             'scope_note': 'Supplemental two-model sensitivity and selected arrays; does not recover the historical FEM gauge.'},
            {'id': 'R2-THERMAL-FEM-02',
             'topic': 'independent thermal FEM reference + v3 Galerkin re-acceptance',
             'status': 'accepted bounded representative FEM study (v2/v3); v1 excluded',
             'results_included': True,
             'scope_note': 'Three representative geometries only; three-grid differences are discretization diagnostics, not true-error bounds; does not validate all historical labels.'},
            {'id': 'new_queries',
             'topic': 'additional thermal query experiments (SoBol / input-geometry '
                      'variants)',
             'status': 'completed (results JSON included)',
             'results_included': True,
             'scope_note': 'completed-result JSON is included, but its reference is '
                           'an interpolated-reference diagnostic (k-nearest '
                           'inverse-distance reference), NOT an independent FEM '
                           'solution; do not present it as independent FEM evidence. '
                           'Per-geometry metrics are supplementary to the '
                           'seed/geometry primary statistics.'},
        ],
        'excluded_from_this_package': list(EXCLUDED_CATEGORIES),
    }


def data_placement_markdown():
    return """# 数据放置说明（本包**不含任何数据**）

本准备包只包含代码、配置副本与已完成实验的 JSON 指标，不包含原始数据、模型权重、
日志或诊断 npz。复现前请自行获取数据并按下列目标路径放置；本包不会自动下载任何数据。
`README.md` 的 “Required data and layout” 一节与 `DATA_SOURCES.md` 是权威说明，如与
本文件不一致以它们为准。

## 热场数据（原作者数据，第三方，不随本包分发）

- 来源、下载地址与 sha256 见包内 `DATA_SOURCES.md`（由 Codex 提供；请核对下载文件的
  sha256 与其中记录一致）。
- 放置目标：包根目录下的 `data/thermal/HeatTransferData.npy`（或运行
  `scripts/exp_thermal.py` 时用 `--data <你的路径>/HeatTransferData.npy` 显式指定）。
  形状/列序约定见 `README.md` 与 `src/core/thermal_io.py`。

## 声学数据（作者单独的声学数据包，不随本包分发）

- 将作者声学数据包中的 `case_<id>.npz` 放到包根目录下的 `data/acoustic/`，共 21 个
  case，文件名必须与 `configs/acoustic_cases.json` 中每个 case 的 `file` 字段一致
  （`data/acoustic/case_{f}_{Ma}.npz`）；不要修改共享配置。
- 未放置数据时，本包只提供代码与 JSON 指标，不能声称已完成整场声学评估复现。

## 不包含的内容

- 模型权重（`*.pth`/`*.pt`）、训练/评测日志、诊断 npz 均不在本包。
- 未完成实验（FDTD 压力收敛）的结果不在本包；状态见 `PENDING_STATUS.json`。
- new_queries 的已完成 JSON 在本包内，但其参考是插值参考诊断（interpolated-reference
  diagnostic），不是独立 FEM 解；PENDING_STATUS.json 中有对应范围说明。
"""


def environment_provenance_text():
    return """环境文件来源说明
================

- `requirements.txt`、`requirements-extended.txt`、`environment.yml` 均**原样复制**自
  上一版 Zenodo 发布包（`zenodo/` 目录，featextractnet v1.1.0），属于历史环境快照。
- 历史记录版本（numpy/torch 等）与实际复现时实测版本可能不同；以 `README.md` 中
  Codex 写明的“历史版本 vs 实测版本”说明为准。
- 本准备包不安装依赖、不改共享环境；复现时请自行核对方括号内记录的实测版本。

说明由 `scripts/package_round2.py` 自动生成；不构成新的科学声明。
"""


# ---------------------------------------------------------------------------
# archive helpers
# ---------------------------------------------------------------------------
def _tar_info(arcname, size):
    info = tarfile.TarInfo(name=arcname)
    info.size = size
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ''
    info.gname = ''
    return info


def build_archive(staging_dir, archive_path, expected_sha):
    """Write a deterministic tar.gz from staging; verify members/sha before returning."""
    members = []
    for root, dirs, files in os.walk(staging_dir):
        dirs[:] = sorted(d for d in dirs if d != '__pycache__')
        for fn in sorted(files):
            full = os.path.join(root, fn)
            arc = _safe_relpath(os.path.relpath(full, staging_dir).replace(os.sep, '/'))
            members.append((arc, full))
    members.sort(key=lambda item: item[0])
    expected_names = {arc for arc, _ in members}
    os.makedirs(os.path.dirname(os.path.abspath(archive_path)), exist_ok=True)
    with open(archive_path, 'wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode='w') as tar:
                for arc, full in members:
                    with open(full, 'rb') as fh:
                        info = _tar_info(arc, os.path.getsize(full))
                        tar.addfile(info, fh)
    # re-read verification
    with tarfile.open(archive_path, 'r:gz') as tar:
        names = [m.name for m in tar.getmembers()]
        if len(names) != len(set(names)):
            raise PackageRefused('archive contains duplicate member names')
        if set(names) != expected_names:
            raise PackageRefused('archive member set differs from staging tree')
        for member in tar.getmembers():
            if not member.isfile():
                raise PackageRefused(f'non-file archive member: {member.name}')
            data = tar.extractfile(member).read()
            if sha256_bytes(data) != expected_sha[member.name]:
                raise PackageRefused(f'archive member sha mismatch: {member.name}')
    return members, sha256_file(archive_path), os.path.getsize(archive_path)


def safe_extract(archive_path, dest_dir):
    """Extract an archive, refusing absolute/.. paths, links and escaping members."""
    dest_real = os.path.realpath(dest_dir)
    os.makedirs(dest_real, exist_ok=True)
    with tarfile.open(archive_path, 'r:gz') as tar:
        for member in tar.getmembers():
            arc = _safe_relpath(member.name)
            if member.issym() or member.islnk():
                raise PackageRefused(f'link member is not allowed: {member.name}')
            if member.isdev():
                raise PackageRefused(f'device member is not allowed: {member.name}')
            if not (member.isfile() or member.isdir()):
                raise PackageRefused(f'unsupported member type: {member.name}')
            target = os.path.realpath(os.path.join(dest_real, arc))
            if target != dest_real and not target.startswith(dest_real + os.sep):
                raise PackageRefused(f'member escapes destination: {member.name}')
            if member.isdir():
                os.makedirs(target, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            data = tar.extractfile(member).read()
            with open(target, 'wb') as fh:
                fh.write(data)
    return dest_real


# ---------------------------------------------------------------------------
# list-only
# ---------------------------------------------------------------------------
def list_only(ws=WS, out_dir=DEFAULT_OUT_DIR, name=DEFAULT_NAME,
              zenodo_dir=None, old_zenodo_dir=None):
    """Report the planned package and missing metadata; writes nothing."""
    if not NAME_RE.match(name or ''):
        raise PackageRefused(f'invalid package name: {name!r}')
    ws = os.path.realpath(ws)
    out_dir = os.path.abspath(out_dir)
    staging = os.path.join(out_dir, name)
    archive = os.path.join(out_dir, name + '.tar.gz')
    roots = default_known_roots(ws)
    plan = collect_plan(ws, zenodo_dir, old_zenodo_dir, roots=roots)
    checks = validate_metadata_inputs(plan)
    lines = [
        f'package name : {name}',
        f'workspace    : {_rel_ws(ws, PROJECT_ROOT) if _is_within(ws, PROJECT_ROOT) else ws}',
        f'out-dir      : {out_dir}',
        f'would refuse : {os.path.exists(staging) or os.path.exists(archive)} '
        f'(same-name package already exists)',
        '',
        'metadata (zenodo/round2/):',
    ]
    for name_meta, entry in checks['metadata'].items():
        state = 'PRESENT' if entry['present'] else 'MISSING'
        lines.append(f'  {state:7s} {name_meta} ({entry["bytes"]} B)')
    lines += [
        '',
        'planned members:',
        f'  total files : {len(plan["members"])}',
        f'  total bytes : {plan["total_bytes"]}',
    ]
    for kind in sorted(plan['counts']):
        lines.append(f'  {kind:12s}: {plan["counts"][kind]}')
    lines += ['', 'completed-result JSON by round2 directory:']
    for dir_name, n in plan['results_counts'].items():
        lines.append(f'  {dir_name:34s}: {n} json')
    lines += ['', 'historical manuscript baseline JSON by directory:']
    for dir_name, n in plan['historical_counts'].items():
        lines.append(f'  {dir_name:58s}: {n} json')
    lines += ['', 'excluded result JSON (internal registry candidates):']
    lines += [f'  {rel}' for rel in sorted(RESULTS_JSON_EXCLUDE)]
    lines += ['', 'manuscript table source manifests:']
    for rel in plan['source_manifests']:
        lines.append(f'  PRESENT {rel}')
    lines += [f'  MISSING {rel}' for rel in plan['missing_source_manifests']]
    lines += ['', 'sensitive required inputs (missing => refuse):']
    arcs = {member['arcname'] for member in plan['members']}
    for arc in plan['required_arcs']:
        lines.append(f'  {"PRESENT" if arc in arcs else "MISSING"} {arc}')
    lines += ['', 'frozen split snapshots (byte-identical config class, no '
                  'path normalisation):']
    for rel in EXECUTION_CONFIG_WHITELIST:
        lines.append(f'  {"PRESENT" if rel in arcs else "MISSING"} {rel}')
    lines += ['', 'main five-seed graph result gate:']
    main = plan['main_graph_result']
    lines.append(f'  dir    : {main["dir"]}')
    lines.append(f'  status : {main["status"]} (contents packaged only when complete)')
    lines += [f'  APPROVED {rel}' for rel in main['approved_files']]
    lines += [f'  MISSING  {rel}' for rel in main['pending_missing']]
    lines += [f'  ISSUE    {item}' for item in main['pending_issues']]
    if plan['pending_main_result']:
        lines.append('  package build REFUSED until the five seeds are complete/'
                     'summary-consistent')
    lines += ['', 'metric protocol.splits_config references (source-authoritative):']
    for rel, count in sorted(plan['metric_split_references'].items()):
        lines.append(f'  {count:4d}  {rel}')
    lines += [f'  MISSING {item["needed_file"]} referenced by '
              f'{item["referenced_by"]}'
              for item in plan['missing_metric_split_snapshots']]
    lines += [f'  UNRESOLVED {item["referenced_by"]} {item["field"]}: '
              f'{item["reason"]}'
              for item in plan['unresolved_metric_split_references']]
    lines += ['', 'dynamic source-load closure (importlib spec_from_file_location etc.):']
    for item in plan['dynamic_helper_references']:
        lines.append(f'  {item["loaded_by"]} -> {item["resolved_file"]} '
                     f'({"included" if item["included"] else "MISSING"})')
    lines += [f'  MISSING {item["needed_file"]} loaded by {item["loaded_by"]}'
              for item in plan['missing_dynamic_helpers']]
    lines += [f'  UNRESOLVED {item["loaded_by"]} via {item["loader"]}: {item["reason"]}'
              for item in plan['unresolved_dynamic_helpers']]
    lines += ['', 'requested-but-missing paths (recorded, not fabricated):']
    lines += [f'  {item["requested"]} -> actual {item["actual"]} ({item["reason"]})'
              for item in plan['requested_missing']] or ['  (none)']
    lines += ['', 'scientific scripts:']
    for member in plan['members']:
        if member['kind'] == 'script':
            note = f'  [{member["note"]}]' if member['note'] else ''
            lines.append(f'  {member["arcname"]}{note}')
    lines += ['', 'missing metadata (build will refuse until provided):']
    lines += [f'  {item}' for item in plan['missing_metadata']] or ['  (none)']
    lines += ['', 'missing required files:']
    lines += [f'  {item["path"]} ({item["reason"]})'
              for item in plan['missing_required']] or ['  (none)']
    lines += ['', 'missing local scientific helpers:']
    lines += [f'  {item["needed_file"]} imported by {item["imported_by"]}'
              for item in plan['missing_helpers']] or ['  (none)']
    lines += ['', 'excluded categories:']
    lines += [f'  - {item}' for item in EXCLUDED_CATEGORIES]
    lines += ['', 'metadata licence preflight (consistency only; no new grant):']
    alignment = checks['license_alignment']
    lines.append(f'  aligned            : {alignment["aligned"]}')
    lines.append(f'  .zenodo.json       : {alignment["zenodo_license"]}')
    lines.append(f'  CITATION.cff       : {alignment["cff_licenses"]}')
    lines.append(f'  MIT notice present : {alignment["mit_notice_present"]}')
    lines += [f'  ERROR {item}' for item in alignment['errors']]
    lines.append(f'  scope note         : {alignment["note"]}')
    lines += ['', 'metadata warnings:']
    lines += [f'  - {item}' for item in checks['warnings']] or ['  (none)']
    text = '\n'.join(lines) + '\n'
    print(text, end='')
    return {'name': name, 'plan': plan, 'checks': checks, 'text': text,
            'would_refuse_existing': os.path.exists(staging) or os.path.exists(archive)}


def _is_within(path, root):
    p = os.path.realpath(path)
    r = os.path.realpath(root)
    return p == r or p.startswith(r + os.sep)


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def build_package(ws=WS, out_dir=DEFAULT_OUT_DIR, name=DEFAULT_NAME,
                  zenodo_dir=None, old_zenodo_dir=None, roots=None):
    """Build staging tree + tar.gz + sha256 + manifests. No overwrite, no partials."""
    if not NAME_RE.match(name or ''):
        raise PackageRefused(f'invalid package name: {name!r}')
    ws = os.path.realpath(ws)
    out_dir = os.path.abspath(out_dir)
    staging = os.path.join(out_dir, name)
    archive = os.path.join(out_dir, name + '.tar.gz')
    sha_path = archive + '.sha256'
    outer_manifest_path = os.path.join(out_dir, name + '.manifest.json')
    existing = [p for p in (staging, archive, sha_path, outer_manifest_path)
                if os.path.exists(p)]
    if existing:
        raise PackageRefused(
            'refusing to overwrite an existing same-name package/artifacts: '
            + ', '.join(existing))

    roots = roots if roots is not None else default_known_roots(ws)
    plan = collect_plan(ws, zenodo_dir, old_zenodo_dir, roots=roots)
    if plan['missing_required']:
        raise PackageRefused(
            'missing required package inputs: '
            + '; '.join(f'{item["path"]} ({item["reason"]})'
                        for item in plan['missing_required']))
    if plan['missing_dynamic_helpers']:
        raise PackageRefused(
            'dynamically loaded helper script(s) are not included in the plan: '
            + '; '.join(f'{item["needed_file"]} loaded by {item["loaded_by"]} '
                        f'via {item["loader"]}' for item in plan['missing_dynamic_helpers']))
    if plan['unresolved_dynamic_helpers']:
        raise PackageRefused(
            'dynamic source-load calls could not be resolved to existing workspace '
            '.py files (refusing to ship an unverifiable helper closure): '
            + '; '.join(f'{item["loaded_by"]} via {item["loader"]}: '
                        f'{item["reason"]}' for item in plan['unresolved_dynamic_helpers']))
    if plan['pending_main_result']:
        pending = plan['pending_main_result']
        raise PackageRefused(
            'main five-seed graph result is not complete/accepted; refusing to '
            'package a partial main result (need seeds 0-4, 500-epoch metrics and '
            'weights, summary consistent): missing='
            + (', '.join(pending['pending_missing']) or '(none)')
            + '; issues=' + (', '.join(pending['pending_issues']) or '(none)'))
    if plan['missing_source_manifests']:
        raise PackageRefused(
            'manuscript table source manifests are missing (refusing to ship the '
            'table result JSONs without provenance): '
            + ', '.join(plan['missing_source_manifests']))
    if plan['missing_helpers']:
        raise PackageRefused(
            'included scripts import local scientific helpers that are not in the '
            'plan: '
            + '; '.join(f'{item["needed_file"]} (imported by {item["imported_by"]})'
                        for item in plan['missing_helpers']))
    if plan['missing_metadata']:
        raise PackageRefused(
            'Codex metadata is incomplete under zenodo/round2/: missing '
            + ', '.join(plan['missing_metadata'])
            + ' (this script does not author these files; provide them first)')
    checks = validate_metadata_inputs(plan, roots)
    if not checks['license_mit']:
        raise PackageRefused(
            'legacy zenodo/LICENSE is not the MIT code licence; refusing to ship it')
    if not checks['license_alignment']['aligned']:
        raise PackageRefused(
            'metadata licence preflight failed (Zenodo/CFF declarations must match '
            'the existing MIT LICENSE; no new licence is granted): '
            + '; '.join(checks['license_alignment']['errors']))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(staging, exist_ok=False)
    try:
        entries = []          # manifest entries for payload files
        blocked = []
        rewrites_total = 0
        for member in plan['members']:
            dst = os.path.join(staging, member['arcname'])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            source_sha = sha256_file(member['src'])
            if member['sanitized'] == 'text':
                with open(member['src'], 'r', encoding='utf-8') as fh:
                    text = fh.read()
                packaged_text, text_info = sanitize_text_document(text, roots)
                _write_text(dst, packaged_text)
                entries.append({
                    'path': member['arcname'], 'kind': member['kind'],
                    'bytes': os.path.getsize(dst),
                    'sha256': sha256_file(dst),
                    'source_sha256': source_sha,
                    'transformation': {
                        'type': 'text_path_normalise',
                        'rewrites': sum(x['count'] for x in text_info['rewrites']),
                        'approved_roots': sorted({x['root'] for x in text_info['rewrites']}) or None,
                    },
                })
            elif member['sanitized']:
                try:
                    with open(member['src'], 'r', encoding='utf-8') as fh:
                        original = json.load(fh)
                except (ValueError, OSError) as exc:
                    raise PackageRefused(
                        f'cannot read JSON result {member["arcname"]}: {exc!r}')
                packaged, info = sanitize_json_document(original, roots)
                if info['blocked']:
                    blocked.extend([{'file': member['arcname'], **item}
                                    for item in info['blocked']])
                    continue
                assert_json_equivalent(original, packaged, info['rewrites'])
                _write_json(dst, packaged)
                with open(dst, 'r', encoding='utf-8') as fh:
                    reloaded = json.load(fh)
                assert_json_equivalent(original, reloaded, info['rewrites'])
                rewrites_total += len(info['rewrites'])
                entries.append({
                    'path': member['arcname'], 'kind': member['kind'],
                    'bytes': os.path.getsize(dst),
                    'sha256': sha256_file(dst),
                    'source_sha256': source_sha,
                    'transformation': {
                        'type': 'json_release_copy',
                        'rewrites': len(info['rewrites']),
                        'approved_roots': sorted({r['root']
                                                  for r in info['rewrites']}) or None,
                        'rewrite_fields': [r['field'] for r in info['rewrites']],
                    },
                })
            else:
                shutil.copyfile(member['src'], dst)
                if sha256_file(dst) != source_sha:
                    raise PackageRefused(f'copy changed bytes: {member["arcname"]}')
                entries.append({
                    'path': member['arcname'], 'kind': member['kind'],
                    'bytes': os.path.getsize(dst),
                    'sha256': source_sha, 'source_sha256': source_sha,
                    'transformation': {'type': 'copy'},
                })
        if blocked:
            raise PackageRefused(
                'unknown absolute path(s) in result JSON: content withheld. '
                'Blocked JSON fields: '
                + '; '.join(f'{item["file"]}::{item["field"]} ({item["rule"]})'
                            for item in blocked))

        generated = {
            'PENDING_STATUS.json': json.dumps(pending_status_document(),
                                              indent=2, ensure_ascii=False) + '\n',
            'data/README_DATA_PLACEMENT.md': data_placement_markdown(),
            'ENVIRONMENT_PROVENANCE.txt': environment_provenance_text(),
        }
        for arc, text in sorted(generated.items()):
            dst = os.path.join(staging, arc)
            _write_text(dst, text)
            entries.append({
                'path': arc, 'kind': 'generated', 'bytes': os.path.getsize(dst),
                'sha256': sha256_file(dst), 'source_sha256': None,
                'transformation': {'type': 'generated'},
            })
        entries.sort(key=lambda item: item['path'])

        # credential scan over every staged file (report file/line/rule only)
        credential_hits = []
        scanned = 0
        for root, dirs, files in os.walk(staging):
            dirs[:] = sorted(d for d in dirs if d != '__pycache__')
            for fn in sorted(files):
                path = os.path.join(root, fn)
                if not _is_text_file(path):
                    continue
                text = open(path, 'r', encoding='utf-8', errors='replace').read()
                scanned += 1
                for hit in scan_text_for_credentials(text):
                    credential_hits.append({'file': _rel_ws(path, staging), **hit})
        if credential_hits:
            raise PackageRefused(
                'potential credential formats found (matched content is never '
                'printed): '
                + '; '.join(f'{h["file"]}:{h["line"]} rule={h["rule"]}'
                            for h in credential_hits))

        # PACKAGE_CONTENTS + PACKAGE_MANIFEST
        contents_lines = [
            f'{name}/  files: {len(entries) + 2}',
            f'generator: scripts/package_round2.py',
            'policy: code + completed-result JSON only; no raw data, no weights, '
            'no logs, no internal docs; FDTD remains pending in PENDING_STATUS.json; '
            'new_queries JSON included with interpolated-reference scope note',
            'results: round2 completed JSON + historical valid manuscript baselines '
            '(exact whitelist) + four manuscript table-source manifests + three frozen '
            'split snapshots referenced by metric protocol.splits_config',
            'metadata: zenodo/round2/{README.md,.zenodo.json,CITATION.cff,DATA_SOURCES.md}',
            'license: zenodo/LICENSE (MIT, unchanged) aligned with .zenodo.json and CITATION.cff at preflight; preflight grants no new licence; LICENSE-DATA not redistributed',
            'environment: requirements*/environment.yml copied from the previous '
            'zenodo release (historical snapshot; see README/ENVIRONMENT_PROVENANCE)',
            *[f'requested-but-missing path: {item["requested"]} -> actual: '
              f'{item["actual"]} ({item["reason"]})'
              for item in plan['requested_missing']],
            '',
        ]
        contents_lines += [entry['path'] for entry in entries]
        contents_lines += ['PACKAGE_CONTENTS.txt', 'PACKAGE_MANIFEST.json']
        contents_path = os.path.join(staging, 'PACKAGE_CONTENTS.txt')
        _write_text(contents_path, '\n'.join(contents_lines) + '\n')
        contents_sha = sha256_file(contents_path)
        entries.append({
            'path': 'PACKAGE_CONTENTS.txt', 'kind': 'manifest',
            'bytes': os.path.getsize(contents_path), 'sha256': contents_sha,
            'source_sha256': None, 'transformation': {'type': 'generated'},
        })
        entries.sort(key=lambda item: item['path'])
        manifest = {
            'package': name,
            'generator': 'scripts/package_round2.py',
            'policy': {
                'included': 'source code (src all .py except src/pipeline), '
                            'whitelisted scientific/packaging scripts, shared config '
                            'copies, completed-result JSON, Codex metadata',
                'excluded': list(EXCLUDED_CATEGORIES),
                'no_data': True, 'no_weights': True, 'no_doi': True,
                'unfinished_items': 'FDTD pressure convergence remains pending in '
                                    'PENDING_STATUS.json; new_queries completed JSON is '
                                    'included with an interpolated-reference scope note',
            },
            'counts': {
                'files_listed': len(entries),        # payload+generated+contents (no self)
                'archive_members': len(entries) + 1,  # self manifest is the +1
                'by_kind': _count_kinds(entries),
            },
            'metadata': {'inputs': checks['metadata'],
                         'warnings': checks['warnings'],
                         'license_alignment': checks['license_alignment']},
            'results_scope': {
                'round2_result_dirs': list(RESULTS_DIR_WHITELIST),
                'historical_baseline_dirs': list(HISTORICAL_RESULT_DIRS),
                'source_manifests': list(plan['source_manifests']),
                'requested_missing': list(plan['requested_missing']),
                'excluded_result_json': sorted(RESULTS_JSON_EXCLUDE),
                'geometry_spec': GEOMETRY_SPEC_PATH,
                'execution_configs': list(EXECUTION_CONFIG_WHITELIST),
                'main_graph_result': {
                    'dir': plan['main_graph_result']['dir'],
                    'status': plan['main_graph_result']['status'],
                    'approved_files': list(plan['main_graph_result']['approved_files']),
                    'summary_consistent': plan['main_graph_result']['summary_consistent'],
                    'metric_split_config_paths':
                        list(plan['main_graph_result']['metric_split_config_paths']),
                },
                'metric_split_config_references':
                    dict(plan['metric_split_references']),
            },
            'dependency_audit': {
                'required_arcs': list(plan['required_arcs']),
                'missing_required_arcs': list(plan['missing_required_arcs']),
                'dynamic_helper_references': list(plan['dynamic_helper_references']),
                'missing_dynamic_helpers': list(plan['missing_dynamic_helpers']),
                'unresolved_dynamic_helpers': list(plan['unresolved_dynamic_helpers']),
            },
            'helper_additions': SCRIPT_HELPER_ADDITIONS,
            'json_release_copy': {
                'known_roots': [r['label'] for r in roots],
                'rewrites_total': rewrites_total,
                'blocked': [],
            },
            'credential_scan': {'files_scanned': scanned, 'hits': []},
            'files': entries,
            'archive': None,   # filled in the sibling manifest (self-hash impossible)
        }
        manifest_path = os.path.join(staging, 'PACKAGE_MANIFEST.json')
        _write_json(manifest_path, manifest)
        manifest_sha = sha256_file(manifest_path)
        entries.append({
            'path': 'PACKAGE_MANIFEST.json', 'kind': 'manifest',
            'bytes': os.path.getsize(manifest_path), 'sha256': manifest_sha,
            'source_sha256': None, 'transformation': {'type': 'generated'},
        })
        expected_sha = {item['path']: item['sha256'] for item in entries}
        members, archive_sha, archive_bytes = build_archive(
            staging, archive, expected_sha)
        with open(sha_path, 'w', encoding='utf-8') as fh:
            fh.write(f'{archive_sha}  {os.path.basename(archive)}\n')
        outer = {
            'package': name,
            'generator': 'scripts/package_round2.py',
            'staging_dir': os.path.basename(staging),
            'archive': {'file': os.path.basename(archive),
                        'sha256': archive_sha, 'bytes': archive_bytes},
            'archive_members': len(members),
            'archive_members_verified': True,
            'counts': {'archive_members': len(members),
                       'by_kind': _count_kinds(entries)},
            'metadata': {'inputs': checks['metadata'],
                         'warnings': checks['warnings'],
                         'license_alignment': checks['license_alignment']},
            'results_scope': {
                'round2_result_dirs': list(RESULTS_DIR_WHITELIST),
                'historical_baseline_dirs': list(HISTORICAL_RESULT_DIRS),
                'source_manifests': list(plan['source_manifests']),
                'requested_missing': list(plan['requested_missing']),
                'excluded_result_json': sorted(RESULTS_JSON_EXCLUDE),
                'geometry_spec': GEOMETRY_SPEC_PATH,
                'execution_configs': list(EXECUTION_CONFIG_WHITELIST),
                'main_graph_result': {
                    'dir': plan['main_graph_result']['dir'],
                    'status': plan['main_graph_result']['status'],
                    'approved_files': list(plan['main_graph_result']['approved_files']),
                    'summary_consistent': plan['main_graph_result']['summary_consistent'],
                    'metric_split_config_paths':
                        list(plan['main_graph_result']['metric_split_config_paths']),
                },
                'metric_split_config_references':
                    dict(plan['metric_split_references']),
            },
            'json_release_copy': {'known_roots': [r['label'] for r in roots],
                                  'rewrites_total': rewrites_total,
                                  'blocked': []},
            'credential_scan': {'files_scanned': scanned, 'hits': []},
            'dependency_audit': {
                'required_arcs': list(plan['required_arcs']),
                'missing_required_arcs': list(plan['missing_required_arcs']),
                'dynamic_helper_references': list(plan['dynamic_helper_references']),
                'missing_dynamic_helpers': list(plan['missing_dynamic_helpers']),
                'unresolved_dynamic_helpers': list(plan['unresolved_dynamic_helpers']),
            },
            'pending_status_file': 'PENDING_STATUS.json',
            'manifest_sha256': manifest_sha,
            'contents_sha256': contents_sha,
            'no_data': True, 'no_weights': True, 'no_doi': True,
        }
        _write_json(outer_manifest_path, outer)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        for path in (archive, sha_path, outer_manifest_path):
            if os.path.exists(path):
                os.remove(path)
        raise

    print(f'[package] {name}: {len(members)} files, '
          f'{archive_bytes} B archive, sha256 {archive_sha}')
    print(f'[package] staging {staging}')
    print(f'[package] archive {archive}')
    print(f'[package] manifest {outer_manifest_path}')
    if checks['warnings']:
        for warning in checks['warnings']:
            print(f'[package][warn] {warning}')
    return {'name': name, 'staging_dir': staging, 'archive': archive,
            'archive_sha256': archive_sha, 'archive_bytes': archive_bytes,
            'sha256_path': sha_path, 'manifest_path': outer_manifest_path,
            'plan': plan, 'entries': entries, 'checks': checks}


def _count_kinds(entries):
    counts = {}
    for entry in entries:
        counts[entry['kind']] = counts.get(entry['kind'], 0) + 1
    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out-dir', default=DEFAULT_OUT_DIR,
                    help=f'package output directory (default {DEFAULT_OUT_DIR})')
    ap.add_argument('--name', default=DEFAULT_NAME,
                    help=f'package name (default {DEFAULT_NAME})')
    ap.add_argument('--list-only', action='store_true',
                    help='only list the planned members and missing metadata; no writes')
    ap.add_argument('--version', action='version', version='R2-RELEASE-PACKAGE-01')
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = args.out_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(WS, out_dir)
    try:
        if args.list_only:
            list_only(WS, out_dir, args.name)
            return 0
        build_package(WS, out_dir, args.name)
        return 0
    except PackageRefused as exc:
        print(f'[refuse] {exc}', file=sys.stderr)
        return 3


if __name__ == '__main__':
    sys.exit(main())
