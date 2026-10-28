# 压力常数偏置敏感性（R2-PRESSURE-GAUGE-02 / R2-THERMAL-REFERENCE-02 A）

- 冻结权重推理，无训练/调参；FEN 与 fresh strict-FP32 graph 各 seeds0–4，同一 20 个 Protocol-A test 几何。
- p 为存储尺度物理单位；mean 为 N=5000 点等权算术均值，**不是**有限元面积积分零均值。
- 中心化逐场独立：Cp=p-mean(p)，Cp̂=p̂-mean(p̂)；零中心化参考 norm 严格 undefined。
- **原 RE 与中心化 RE 分母不同，不把差值称为提升**；本表仅并列。
- 已看 test 后的补充敏感性分析，条件于固定训练 seeds，不是独立确认。
- **主统计**：每 geometry 先算 5seed 平均；跨 5 个 seed mean 报 mean ± sample std。geometry std 是 20 geometry 之间的变异，另列，不冒充 seed uncertainty。
- summary.json 同时保留 per_geometry / per_seed / cross_seed。

## 原评分复算

所有 model×seed×geometry 的 u/v/p/T RE/MAE/max 与 metrics JSON 复算一致（容差 1e-06，raw_results 记录每场最大误差）。

## 原始 p — 主统计（跨 5 seed mean ± sample std）

| model | metric | primary mean ± sample_std | geometry std (auxiliary) |
|---|---|---|---|
| fen | re | 0.0206296 ± 0.00168366 | 0.00330601 |
| fen | mae | 0.00248496 ± 0.000542203 | 0.000338074 |
| fen | max | 0.0382009 ± 0.000814971 | 0.00661563 |
| graph | re | 0.0146644 ± 0.000999717 | 0.00330339 |
| graph | mae | 0.00242167 ± 0.000142763 | 0.000608813 |
| graph | max | 0.0182277 ± 0.00475759 | 0.00359498 |
## 中心化 p（逐场独立去均值） — 主统计（跨 5 seed mean ± sample std）

| model | metric | primary mean ± sample_std | geometry std (auxiliary) |
|---|---|---|---|
| fen | re | 0.0196344 ± 0.00132411 | 0.00319604 |
| fen | mae | 0.00207503 ± 0.000440427 | 0.00023673 |
| fen | max | 0.0379769 ± 0.00156712 | 0.00651679 |
| graph | re | 0.0133772 ± 0.00161227 | 0.0032554 |
| graph | mae | 0.00220514 ± 0.000274687 | 0.000651806 |
| graph | max | 0.0185553 ± 0.00469878 | 0.00364639 |

## 逐 seed 20-geometry mean ± sample std（补充）

| model | kind | metric | seed0 | seed1 | seed2 | seed3 | seed4 |
|---|---|---|---|---|---|---|---|
| fen | original | re | 0.0201654 ± 0.00357525 | 0.0234253 ± 0.00343886 | 0.0188657 ± 0.00313648 | 0.0202132 ± 0.00333164 | 0.0204784 ± 0.00348858 |
| fen | original | mae | 0.0024035 ± 0.000551607 | 0.00332189 ± 0.000457505 | 0.00180319 ± 0.000243056 | 0.00242232 ± 0.000463526 | 0.00247389 ± 0.000439602 |
| fen | original | max | 0.0378603 ± 0.0067048 | 0.0390353 ± 0.00649504 | 0.0373607 ± 0.00644485 | 0.0391085 ± 0.00642171 | 0.0376398 ± 0.00747504 |
| fen | centered | re | 0.019033 ± 0.00314667 | 0.0217939 ± 0.00341895 | 0.018564 ± 0.00320019 | 0.0187892 ± 0.00316291 | 0.0199919 ± 0.00311859 |
| fen | centered | mae | 0.00186788 ± 0.000196207 | 0.00273602 ± 0.000373105 | 0.00166035 ± 0.000218971 | 0.00180711 ± 0.000195353 | 0.00230381 ± 0.000278959 |
| fen | centered | max | 0.0366475 ± 0.0065442 | 0.0406763 ± 0.00660699 | 0.0376486 ± 0.00635064 | 0.0376999 ± 0.00645368 | 0.0372122 ± 0.00709055 |
| graph | original | re | 0.0151257 ± 0.00469676 | 0.0153141 ± 0.00314141 | 0.0137381 ± 0.00361386 | 0.0156874 ± 0.00390455 | 0.0134568 ± 0.00300321 |
| graph | original | mae | 0.0024818 ± 0.000848054 | 0.00248673 ± 0.000619837 | 0.00228362 ± 0.000610041 | 0.00259324 ± 0.000790766 | 0.00226298 ± 0.000520995 |
| graph | original | max | 0.0231628 ± 0.00662776 | 0.0235339 ± 0.0062943 | 0.0134499 ± 0.00330309 | 0.0151247 ± 0.00377437 | 0.0158673 ± 0.00441927 |
| graph | centered | re | 0.014107 ± 0.00426057 | 0.0147042 ± 0.00282765 | 0.0117203 ± 0.00275168 | 0.014802 ± 0.00422173 | 0.0115527 ± 0.00306237 |
| graph | centered | mae | 0.00231593 ± 0.000843633 | 0.00237628 ± 0.00059383 | 0.0019415 ± 0.000545517 | 0.00250469 ± 0.000824657 | 0.00188731 ± 0.000622737 |
| graph | centered | max | 0.0234812 ± 0.00701772 | 0.0235829 ± 0.00653962 | 0.0134945 ± 0.0035541 | 0.0153367 ± 0.00397141 | 0.0168815 ± 0.00450627 |

## Primary FEN − graph paired bootstrap（每 geometry 先平均 5 seed）

| metric | mean diff FEN−graph | 95% CI |
|---|---|---|
| original:re | 0.00596516 | 0.00564637 .. 0.00626595 |
| original:mae | 6.32857e-05 | -9.44273e-05 .. 0.000205953 |
| original:max | 0.0199732 | 0.0177408 .. 0.0223739 |
| centered:re | 0.00625713 | 0.00611719 .. 0.00638779 |
| centered:mae | -0.00013011 | -0.000315146 .. 4.737e-05 |
| centered:max | 0.0194216 | 0.0172961 .. 0.0217385 |

## 每 seed paired bootstrap（辅助，非主区间）

### seed 0
| metric | mean diff FEN−graph | 95% CI |
|---|---|---|
| original:re | 0.00503964 | 0.00395522 .. 0.00603454 |
| original:mae | -7.83001e-05 | -0.000403921 .. 0.000229298 |
| original:max | 0.0146975 | 0.0118071 .. 0.0177233 |
| centered:re | 0.00492599 | 0.00434319 .. 0.00549998 |
| centered:mae | -0.000448059 | -0.00073911 .. -0.000165484 |
| centered:max | 0.0131663 | 0.0104497 .. 0.0159565 |
### seed 1
| metric | mean diff FEN−graph | 95% CI |
|---|---|---|
| original:re | 0.00811117 | 0.00775451 .. 0.00846465 |
| original:mae | 0.000835156 | 0.000693122 .. 0.000980343 |
| original:max | 0.0155014 | 0.0121225 .. 0.0192429 |
| centered:re | 0.00708963 | 0.00655255 .. 0.00758883 |
| centered:mae | 0.000359741 | 0.000212739 .. 0.000501046 |
| centered:max | 0.0170934 | 0.0138423 .. 0.0207511 |
### seed 2
| metric | mean diff FEN−graph | 95% CI |
|---|---|---|
| original:re | 0.00512757 | 0.00440453 .. 0.00582264 |
| original:mae | -0.000480427 | -0.000670591 .. -0.000299256 |
| original:max | 0.0239108 | 0.0217024 .. 0.0261042 |
| centered:re | 0.00684361 | 0.00644449 .. 0.00723419 |
| centered:mae | -0.000281153 | -0.000430243 .. -0.000134543 |
| centered:max | 0.0241542 | 0.0219975 .. 0.0262582 |
### seed 3
| metric | mean diff FEN−graph | 95% CI |
|---|---|---|
| original:re | 0.0045258 | 0.00390611 .. 0.00514534 |
| original:mae | -0.000170916 | -0.000457835 .. 9.27728e-05 |
| original:max | 0.0239838 | 0.0219034 .. 0.0262618 |
| centered:re | 0.00398722 | 0.00326542 .. 0.00477669 |
| centered:mae | -0.000697581 | -0.00098659 .. -0.000421632 |
| centered:max | 0.0223632 | 0.0200421 .. 0.0249313 |
### seed 4
| metric | mean diff FEN−graph | 95% CI |
|---|---|---|
| original:re | 0.00702161 | 0.00583903 .. 0.00812499 |
| original:mae | 0.000210917 | 2.10265e-05 .. 0.000387779 |
| original:max | 0.0217725 | 0.018921 .. 0.0247918 |
| centered:re | 0.00843921 | 0.00774559 .. 0.00903189 |
| centered:mae | 0.0004165 | 0.00023447 .. 0.000586965 |
| centered:max | 0.0203307 | 0.0173785 .. 0.0235091 |

## Baseline inventory（实际检查范围/存在性）

| token | found roots | seed ids | complete five | strict FP32 | included |
|---|---|---|---|---|---|
| unif_therm | 1: results/thermal_protocolA/unif_therm | [0, 1, 2, 3, 4] | True | False | False |
| mlp | 1: results/thermal_protocolA/mlp | [0, 1, 2, 3, 4] | True | False | False |
| pointnet | 1: results/thermal_protocolA/pointnet | [0, 1, 2, 3, 4] | True | False | False |
| pointnetpp | 1: results/thermal_protocolA/pointnetpp | [0, 1, 2, 3, 4] | True | False | False |
| dgcnn | 1: results/thermal_protocolA/dgcnn | [0, 1, 2, 3, 4] | True | False | False |
| pipn | 1: results/thermal_protocolA/pipn | [0, 1, 2, 3, 4] | True | False | False |
| siren | 1: results/baselines_ext/thermal_protocolA/siren | [0, 1, 2] | False | False | False |
| pointtransformer | 1: results/baselines_ext/thermal_protocolA/pointtransformer | [0, 1, 2] | False | False | False |
| meshgraphnet | 2: results/baselines_ext/thermal_protocolA/meshgraphnet; results/round2_graph_five_seed_main/meshgraphnet | [0, 1, 2, 3, 4] | True | False | True |
| feat_extract | 2: results/thermal_protocolA/feat_extract; results/round2_thermal_symmetric/feat_extract | [0, 1, 2, 3, 4] | True | False | True |

checked roots: ['<PROJECT_ROOT>/results/baselines_ext/thermal_protocolA', '<PROJECT_ROOT>/results/thermal_protocolA', '<PROJECT_ROOT>/results/thermal', '<PROJECT_ROOT>/results/round2_thermal_symmetric', '<PROJECT_ROOT>/results/round2_graph_five_seed_main']

未找到仅指上述实际检查根中未发现；不是全局不存在断言。

## 限制

- 中心化参考 norm=0 时 RE 为 undefined（不填 epsilon）；bias/denominator 均保存在 raw_results。
- 原 RE 与中心化 RE 不能直接相减解释；中心化是预定逐场评价算子，不作超参/epoch/test 选路。
- primary 区间条件于固定训练 seeds、5seed per geometry 平均；不是 100 个独立几何。
- 基线 inventory 只记录实际检查根；未发现不等于全局不存在；未为覆盖全部额外训练。
- 这不是独立数值确认，也不改变原数据/权重/split/既有 summary。
