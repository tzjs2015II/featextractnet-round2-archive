# R2-THERMAL-FEM-02 v3 — acceptance correction + eps quantitative closure

**DSH 收尾交付完成；Codex 集中终审仍待做。** v1 结果已失效并归档；v2 数值结果保留，但其旧 acceptance 文字被本 v3 后处理取代。没有重跑 9 个有效求解，没有改论文/registry/50910。

## 1. 验收判据修正（不再把 cell projection 当弱残差）

- 真正的离散 Galerkin 残差：`weak_residual.blocks.full.free`，即完整非线性 residual 向量把所有 velocity/T Dirichlet 行清零后的 free norm；`full.full` 是含 BC 行的 raw norm。
- 验收 gate：`free_absolute <= 1e-12`（solver absolute update tolerance）且 `free/full <= 1e-8`（原 weak-residual tolerance），再加 Newton 收敛、壁面速度、温度 BC、质量流和 finite；**Newton reason 不能覆盖 free residual**。
- 原 momentum/energy cell projection 保留并明确命名为 `cell_strong_against_test_projection`（cell strong residual against test，含内界面项缺失），**不再称为 weak residual**。
- 对已保存 9 条诊断重新验收（无重解）：全部通过；最大 free absolute = **2.875e-16**，free/full 远小于 1e-8。故意构造的 `full.free=1.0` 在 Newton 收敛且 projection 很小时仍被拒绝。

| geom | h | free_abs | free/full | pass |
|---|---|---|---|---|
| 17 | 0.08 | 1.631e-16 | 1.071e-15 | True |
| 17 | 0.04 | 1.931e-16 | 1.789e-15 | True |
| 17 | 0.02 | 2.864e-16 | 3.747e-15 | True |
| 46 | 0.08 | 1.571e-16 | 1.052e-15 | True |
| 46 | 0.04 | 1.898e-16 | 1.796e-15 | True |
| 46 | 0.02 | 2.826e-16 | 3.778e-15 | True |
| 96 | 0.08 | 1.607e-16 | 1.063e-15 | True |
| 96 | 0.04 | 1.957e-16 | 1.831e-15 | True |
| 96 | 0.02 | 2.875e-16 | 3.797e-15 | True |

- 旧 v2 acceptance 声明/JSON 已存 `archive_v2_prepost/`；postprocess 源为 `scripts/postprocess_r2_thermal_fem_acceptance.py`，不改旧 job 代码。

## 2. eps 定量证据（geom17 h=.08，固定 support，eps vs 1e-8）

- 真实约束：FE 弱式中的 `div(u) = -eps p`（不是点态恒等式；pilot div L2 ~0.166，远大于 eps p）。

| eps | field | RE | MAE | max |
|---|---|---|---|---|
| 1e-09 | u | 1.017e-08 | 4.624e-10 | 2.309e-09 |
| 1e-09 | v | 1.125e-08 | 9.176e-10 | 3.365e-09 |
| 1e-09 | T | 4.572e-09 | 1.461e-09 | 4.977e-09 |
| 1e-09 | p_centered | 9.417e-09 | 1.730e-09 | 2.972e-09 |
| 1e-07 | u | 1.017e-07 | 4.624e-09 | 2.309e-08 |
| 1e-07 | v | 1.125e-07 | 9.176e-09 | 3.365e-08 |
| 1e-07 | T | 4.572e-08 | 1.461e-08 | 4.977e-08 |
| 1e-07 | p_centered | 9.417e-08 | 1.730e-08 | 2.972e-08 |

eps 从 1e-8 偏离到 1e-9/1e-7 时场差约 1e-8/1e-7 量级，随 eps 线性缩小/增大；证明 penalty 是近似 gauge，不是精确零空间。数组在 `v3_eps/`，job **51027**。

## 3. 模型评价：raw p 与 centered p 分开

- 参考评价已使用实际 mesh/`Function.eval`，fixed 5000 support，outside/nonfinite=0；非全局 CloughTocher 重建。
- FEN 副本 hash `9f8d2...` 与验收模型一致；模型 probe 143/143 checks（含 full-support prediction shape/finite）。
- 三网格差是**离散差**，不是数学误差界；只覆盖 46/17/96，不是 135 几何验证。

| geom | FEN 5-seed | graph 5-seed |
|---|---|---|
| 46 | u 0.1104 / v 0.0844 / p_raw 0.0600 / p_c 0.0591 / T 0.0412 | u 0.0764 / v 0.0602 / p_raw 0.0643 / p_c 0.0641 / T 0.0384 |
| 17 | u 0.0920 / v 0.0681 / p_raw 0.0477 / p_c 0.0472 / T 0.0344 | u 0.0645 / v 0.0394 / p_raw 0.0385 / p_c 0.0378 / T 0.0230 |
| 96 | u 0.0863 / v 0.0674 / p_raw 0.0525 / p_c 0.0521 / T 0.0379 | u 0.0639 / v 0.0423 / p_raw 0.0493 / p_c 0.0488 / T 0.0292 |

- raw p RE 与 centered p RE 分母不同，不能把二者差当改善量；两者均在此表分开给出。

## 4. 作业/版本

| jobid | job | state | exit | elapsed | purpose |
|---|---|---|---|---|---|
| 51016 | r2_fem_v2levels | COMPLETED | 0:0 | 00:01:27 | geom17 3h + actual support |
| 51020 | r2_fem_v2reps2 | COMPLETED | 0:0 | 00:03:18 | 46/96 3h + actual support |
| 51021 | r2_fem_v2me2 | COMPLETED | 0:0 | 00:00:20 | model eval actual FEM refs |
| 51022 | r2_fem_v2pilot2 | COMPLETED | 0:0 | 00:00:07 | geom17 pilot final |
| 51023 | r2_fem_eps | COMPLETED | 0:0 | 00:00:12 | eps summary 1e-9/1e-7 |
| 51027 | r2_fem_v3eps | COMPLETED | 0:0 | 00:00:09 | eps 1e-9/1e-7 support arrays |
| 51029 | r2_fem_v3verify2 | COMPLETED | 0:0 | 00:00:08 | final v3 probe 37 checks |
| 51030 | r2_fem_v3modeleval | COMPLETED | 0:0 | 00:00:22 | model eval regenerated with current source hashes |
| 51028 | r2_fem_v3verify | FAILED | 1:0 | 00:00:08 | pre-final probe, retained |
| 51026 | r2_fem_v2verify4 | COMPLETED | 0:0 | 00:00:08 | v2 final probe 37 checks |
| 50910 | pinnap_r2fdtdconv | RUNNING | n/a | running | pre-existing FDTD untouched |

- **v1 invalid**：旧 `results/round2_thermal_fem`/旧 pilot/levels/reps/model_eval 因错误对流算子和错误参考评价失效；源/报告存 `archive_v1/`。
- **v2**：修正算子/实际 Function.eval 后 9 个有效求解；旧 acceptance 文字已被本 v3 后处理取代，原始 v2 声明存 `archive_v2_prepost/`。
- **v3**：本次 postprocess/eps 定量/顶层入口修正；数值解没有重跑。

## 5. 顶层入口修正

- `FINAL_DELIVERY.md` 与 `report.md` 改为简短入口：当前权威为 `v2/FINAL_DELIVERY_V2.md` + `v3_postprocess/`；明确 v1 已失效。
- `VERSION_CHAIN.md` 已更新。

## 6. 源码/归档 SHA256

- `src/fem_reference/__init__.py`: `32affee2eda9846f578235d127ba82135ac629db0469332d668bdaff578f0fc7`
- `src/fem_reference/geometry.py`: `5c694a0aac1cd4c50be88d8c7f4f73a1a0f214afa9f1f9a7440ac14653d0ab96`
- `src/fem_reference/fenics_solver.py`: `02d54782aa52b7aecc41f06d7bf2b82cd15d23f9fcc24add6fe5aa85f3bbb194`
- `src/fem_reference/featextractnet_symmetric.py`: `9f8d2acbd5a6441dbed2102f6d7a6d27d37669fe5ff1a4884f946a6ea44c02c7`
- `scripts/exp_r2_thermal_fem.py`: `5e82f85e86608f854f692a598ac35d77ec1bc2cfa30f21fa98aeb32c5bc40c64`
- `scripts/probe_r2_thermal_fem.py`: `2798c5fcad5343a151d12bc96d84a783af964fab0c44c322a34bac7de1c64b95`
- `scripts/exp_r2_fem_model_eval.py`: `2973caffdd5bea29806ffad700026a6609b134a43e69b322077ea4a263e50ddc`
- `scripts/probe_r2_fem_model_eval.py`: `f84fa42894b125a50ad85ec0b641fb07f133e83aeccfbc334568b94fbe1ca340`
- `scripts/postprocess_r2_thermal_fem_acceptance.py`: `27d4c4e4f1b42b953c1c951219ed674708fa80d31b97f272353d920270ccce81`
- `scripts/sbatch/r2_thermal_fem.sbatch`: `802aa5f0d82231f1eb488b117f345e14591f8bdafade0a5b15406cfb04d238ef`
- `configs/r2_thermal_fem.json`: `536329883ff523e1c0e66c6c7d4b06d2dd4350fbea1cdb7182bcc43f439e4978`

- `results/round2_thermal_fem/archive_v1/FINAL_DELIVERY_v1.md`: `3d8da5d10ee39a67ec4454c7165cba512da9e22316d962d138af2883b70e5b80`
- `results/round2_thermal_fem/archive_v2_prepost/FINAL_DELIVERY_V2_prepost.md`: `4d3db4d287d73a6cf561436e9efaaa271ed2cb8833767c384c88024eb9016c33`
- `results/round2_thermal_fem/archive_v2_prepost/report_prepost.md`: `3f711419b7c7440ba4ab6a13a55ae97681da6fad7c121ee904f95883b1ed377d`
- `results/round2_thermal_fem/archive_v2_prepost/levels_geom17_prepost.json`: `1fdb603bd2baef1bac27020a73917096c58f76559adada30264ba75e50909bf1`
- `results/round2_thermal_fem/archive_v2_prepost/representatives_46_96_prepost.json`: `afd14094d297361028afa98db59336098fec8cfd7f666a10a490de2ce7d87969`

## 7. 关键产物

- `v3_postprocess/reacceptance.json` / `.md`（9 条重新验收 + 大 free 拒绝）
- `v3_eps/eps_differences.json` + `fem_eps1e-09/1e-07_geom17_h0.08.npz`
- `v3_fem_probe.json`（37 checks；Stokes+Boussinesq manufactured）
- `v2_model_eval/`（actual FEM refs + 30 predictions；summary/raw/protocol）
- `archive_v1/`, `archive_v2_prepost/`, `VERSION_CHAIN.md`, `job_ledger_v3.json`, `hash_manifest_v3.json`
