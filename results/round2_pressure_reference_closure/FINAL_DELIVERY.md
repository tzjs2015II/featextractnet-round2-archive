# R3.16 压力基准来源追溯与规范不变性闭合 — DSH 交付

**协议**：R2-PRESSURE-REFERENCE-CLOSURE-03 v1  
**状态**：DSH 实现、自查与真实 SLURM 执行完成；Codex 独立终审与 reviewer 复核尚未做；**R3.16 不标 pass**。  
**日期**：2026-10-23（香港时间）

## 1. 结论分级

| 项目 | 状态 |
|---|---|
| 历史压力参考已识别 | **否**；公开生成器/参考节点/gauge 记录不可得 |
| 数据相容候选 | 是；历史零锚不确认，p≈1e-3 候选不称共同零锚 |
| 当前规范不变性数学检查 | 是；常数平移/恒等式/零分母独立检查 |
| 精确 FEM gauge 三代表比较 | **是**；46/17/96 h=.04，eps=0，自由残差 ~2e-16 |
| exact-gauge 30 预测敏感性 | **是**；复用冻结预测，不重训/不重跑 GPU |
| 原始数据/权重未改 | 是；SHA 见下 |

**严格区分**：`historical_reference_identified=false`、`compatible_only=true`、`current_gauge_invariant_evaluation_closed=true`。历史不可识别与当前规范闭合是两件事，不能互相替代。

## 2. 路线 A：来源与数据法证

- [arXiv:2202.05476v2](https://arxiv.org/abs/2202.05476v2) §4.2 明确：自然对流 inverse loss **没有 pressure boundary condition**；§4.1.2 的制造解压力 BC 是另一案例，未移植。135 几何（108/27），点云为有限元网格顶点子集，数据由作者所述 in-house FEM 家族生成（refs [34,29,28,30,31]）。
- 官方 [Ali-Stanford/PhysicsInformedPointNet](https://github.com/Ali-Stanford/PhysicsInformedPointNet) commit `dcce412d20f5bf44891c96b8d2415560be95718d`：仅有 PIPN 实现，无数据生成器/FEM solver/参考节点/gauge 记录。
- 公开 [HeatTransferData.npy](https://web.stanford.edu/~kashefi/data/HeatTransferData.npy) 下载 32,400,128 bytes，SHA `e9bb20c51105607f2e20b6003fc387e8eaed5ad8906a8f11d8f83bc1cf535e26`，与本项目 canonical 数组完全一致；**数据分发可追溯，生成 gauge 不可追溯**。
- 135 场法证：每场 492 公共外壁坐标；p 节点均值范围 `[-0.003436192208, 0.008112745450]`；tau=`1e-12*max(1,max|p|)` 及 1e-10/1e-8；train100 选候选、val15 冻结确认、test20 冻结后检查。候选 `(-1.0,0.1869918699)` 的 p 约 1e-3，**只称 compatible candidate**。
- 证据：[source_evidence_index.json](source_evidence_index.json)、[all135_pressure_signatures.json](all135_pressure_signatures.json)。

## 3. 路线 B：精确 gauge 与模型敏感性

### 实现
- 新模块 `src/fem_reference/pressure_reference_audit.py`：生产 Taylor–Hood 弱形式，**eps=0**，PETSc 全局 Lagrange 乘子增广：
  `[J b; bᵀ 0]`，`b_i=∫q_i dx`，约束 `∫p dx=0`。
- Newton 残差按 dolfinx 0.8 正确形式：`apply_lifting(R,[J],bcs=[bcs],x0=[w],scale=-1)`，`set_bc(R,bcs,w,-1)`；b 行/列与约束 RHS 一致缩放；乘子对角为 0，不放松精确约束。
- 线性残差用独立保留的 A_copy 计算，超过 1e-8 立即失败；不使用 KSP reason 代替真实残差。
- 点 gauge 由同一积分规范解作**精确常数平移**得到；不称两次独立求解。

### 三代表真实结果（修正约束 RHS 后）
| geom | h | 自由残差 | 积分约束 | point-vs-integral center 差 | exact-vs-eps center RE |
|---|---|---|---|---|---|
| 17 | .04 | 1.97e-16 | -2.78e-17 | 7.94e-17 | 1.046e-8 |
| 46 | .04 | 1.92e-16 | -2.78e-17 | 9.29e-17 | 1.040e-8 |
| 96 | .04 | 1.96e-16 | -8.33e-17 | 8.40e-17 | 1.042e-8 |

**约束 RHS 修正**：Newton 未知量是 `delta`，增广底部 RHS 必须为 `-bscale*C`（旧版写 0 只约束 `bᵀdelta=0`，会保留初值积分）。修正后每步做精确压力常数投影；最终同时检查实际积分约束和自由 Galerkin 残差，不只看 `update_rel`。
- 旧生产快照保留在 `audit_snapshots/uncorrected_production_51065/`；旧积分残留 `-3.92e-9/2.47e-9/-1.09e-10`，修正后 `~1e-17`。
- 旧/新逐场对照：u/v/T 最大差 **0**；p_integral 差异为纯常数；p_point 差 `~1e-16`；30 预测 centered RE 最大差 `3.47e-17`，raw RE 最大差 `1.76e-9`。因此旧结果**仅压力常数项失效**，不需要重算 u/v/T 或 GPU；旧文件未删除。详见 `corrected_constraint_comparison.json`。
- 新增可判别反例：MMS 初始 p 加 **+1**、relax=**0.6**；初始积分 `3.469806640162439`，16 次后 `2.78e-17`；free 残差 `2.06e-7`，linear 残差 `9.49e-14`。

- 三例均 `Function.eval` 在冻结 5000 支持上：valid 5000/5000；u/v/T 在两种精确 gauge 下相同；阈值 1e-6 通过。
- 旧 v2 eps h=.04 参考只作单独近似对照，不称 exact。
- 30 个冻结预测（FEN/graph × seeds0–4 × geom17/46/96）用精确积分规范重新计算原/中心化 p 指标；无重训、无重复全 GPU。

### 新增广路径的 MMS 补验

**旧三角函数 MMS（兼容性诊断，不作精确 F 门禁）**：保留在 `manufactured_solution.json`，并显式标注 `compatibility_only=true`、`weak_residual_status=not_applicable_for_MMS_default_diagnostics_no_source`；其默认 `weak_residual` 未传制造源，不能作为真实 MMS 残差。

**新增严格相容多项式 MMS（生产路径）**：
- 三次 streamfunction `psi=x^3-3xy^2+0.5x^2y+0.2y^3`，`u=(psi_y,-psi_x)` 为二次速度（P2 可精确表示、连续 div=0、straight edge 总通量相容）；T 二次（P2）、p 线性（P1），源项由独立分量与非零对流构造。
- 同一 `solve_integral_gauge` 增广/非齐次 BC/eps=0 路径，三级 h=.20/.14/.10：
  - 带源 free 向量残差：`1.242e-15 / 1.305e-15 / 1.223e-15`（均 <1e-12）；
  - free/source relative：`4.06e-17 / 4.26e-17 / 3.99e-17`（均 <1e-8）；
  - 逐 Dirichlet DOF 最大误差：速度 `0 / 0 / 0`（224/336/448 DOFs），温度 `0 / 0 / 0`（112/168/224 DOFs）；边界点采样误差也 ~1e-15；
  - closed-boundary flux：`-1.22e-15/-5.55e-16/-1.67e-16`；q=1 continuity：`3.55e-16/-1.74e-16/1.05e-15`；
  - 积分约束：`-3.38e-17/-1.39e-17/-3.82e-17`；线性残差 `~1e-15`；
  - block L2 residual：continuity `~9e-14`、momentum `~4-5e-13`、energy `~6-11e-14`；
  - 场误差已在机器精度（~1e-15），不再以细化单调性作为门禁。
- p+1/relax=0.6 反例（同生产路径）：初始 integral `3.469806640162439`，2 次后 `1.39e-17`；free `1.68e-15`、linear `4.07e-15`。
- `manufactured_polynomial_solution.json` 的 `all_pass=true`；旧三角结果仅兼容性诊断。

**非平凡初值多步测试（R5）**：同一生产路径 h=.14，确定性扰动内部 u/T DOF（0.05 尺度，Dirichlet 保持精确制造 BC），叠加 p+1、relax=0.6：
- 初始扰动范数 `19.7402`；初始带源 free `1.982826`（显著>门槛）；
- **31 次真实 Newton 更新**，update 范数非零（首步 24.98，末步 2.65e-11）；
- final free `9.410e-13`（<1e-12），free/source relative `3.07e-14`（<1e-8）；
- final integral `1.39e-17`、linear `~1e-15`、逐 DOF BC `0`、boundary flux `-5.55e-16`、q=1 continuity `3.62e-16`、场误差 `~1e-13`；
- `manufactured_nontrivial_initial_solution.json` 的 `all_pass=true`；停止判据要求实际 free 残差，不只看 update_rel。

## 4. 必交产物

| 产物 | 说明 |
|---|---|
| [protocol_snapshot.json](protocol_snapshot.json) | 协议/源/输入/输出哈希/状态 |
| [source_evidence_index.json](source_evidence_index.json) | 论文/仓库/数据 URL、commit、段落、SHA |
| [all135_pressure_signatures.json](all135_pressure_signatures.json) | 135 场 p/外壁/tau/candidate 法证 |
| [exact_gauge_comparison.json](exact_gauge_comparison.json) | 三代表精确 gauge 比较，`all_pass=true` |
| [manufactured_solution.json](manufactured_solution.json) | 旧三角 MMS：兼容性诊断，非 exact-F 门禁 |
| [manufactured_polynomial_solution.json](manufactured_polynomial_solution.json) | 严格相容多项式 MMS + p+1/relax 反例，`all_pass=true` |
| [manufactured_nontrivial_initial_solution.json](manufactured_nontrivial_initial_solution.json) | 非平凡 u/T 初值多步收敛，31 步，final free 9.41e-13，`all_pass=true` |
| [corrected_constraint_comparison.json](corrected_constraint_comparison.json) | 旧/新约束 RHS 与压力常数差异、u/v/T 不变证据 |
| [model_metric_sensitivity.json](model_metric_sensitivity.json) | exact-gauge 30 预测 p 指标 |
| [job_ledger.json](job_ledger.json) | pilot/full/失败作业与原因 |
| [self_acceptance.json](self_acceptance.json) | R3.16 逐项自查 |
| [probe_report.json](probe_report.json) | 解析不变性/恒等式 probe |
| [FINAL_DELIVERY.md](FINAL_DELIVERY.md) | 本文件 |

## 5. 完整性与边界
- Data SHA256：`e9bb20c51105607f2e20b6003fc387e8eaed5ad8906a8f11d8f83bc1cf535e26`。
- 未修改原数据/权重/原 production solver/论文/registry/打包文件；依赖仅既有 `tools/envs/pinnap-fem`；未安装新依赖。
- 50910 保持 RUNNING，未取消/重跑；本项目并发未超 3。
- 失败版本/日志保留在 `audit_snapshots/`、`logs/`。

## 6. 未决（Codex 终审）
- 历史生成器/gauge/参考节点仍不可得；这部分只能以 unavailable provenance + compatible candidate 表述。
- Codex 需独立复核 exact solver、三例原始数组、30 预测和来源索引；通过前 R3.16 不标 pass。
