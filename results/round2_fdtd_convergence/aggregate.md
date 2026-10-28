# R2-FDTD-CONV-01：压力解空间/时间收敛聚合报告

- task: R2-FDTD-CONV-01 / protocol R2_FDTD_CONV_v1
- created (UTC): 2026-10-23T06:52:24+00:00
- solves: 10/10 complete & ok
- out-root: `../round2_fdtd_convergence`
- solver: `src/fdtd/python_fdtd.py` (sha256 020cf4858cf8d65c...)
- script sha256: d6d6684381886777..., config sha256: 69a4b4bc047cca0b...

## 约定（聚合前已核）

- 主字段: `p_total`（真实终时压力快照）；补充字段: `p_scattered`
- 空间 3 点：coarse/mid/fine 全部在 **coarse 气域节点**上直接按整数索引抽取，无插值、无坐标平移。
- 时间 3 点：mid/mid_dt2/mid_dt4 固定 **mid 气域节点**，同物理终时 `Ncycle/f`、整数个源周期（终相 sin(2π f t)=0），无插值、无任意相位平移。
- 指标：RE=||num-ref||₂/||ref||₂，MAE=mean|num-ref|，max=max|num-ref|；参考 = 相邻更细一档（细网格只是数值参考，不是解析真解）。
- observed order 使用**同节点绝对 L2 差范数**之比：log₂(‖p_coarse−p_mid‖₂ / ‖p_mid−p_fine‖₂) / log₂(2)（不是 RE 之比，因为两 pair 的参考场范数不同）；仅当差范数严格单调下降且分母非零时报，否则显式 `undefined` 并给原因。RE 仍是主计分，不受此影响。

## 协议身份与坐标核查

| case | solves | identity 一致 | t_final 实测一致 | 空间 coords 嵌套精确 | 时间 coords 精确 |
|---|---|---|---|---|---|
| low100 | 5 | True | True | True | True |
| high500 | 5 | True | True | True | True |

## 空间收敛（共同节点抽取）

### low100

共同节点：241×241（coarse air nodes (direct node extraction, no interpolation)）

| 字段 | pair | RE | MAE | max | ||num-ref|| | ||ref|| |
|---|---|---|---|---|---|---|
| p_total | coarse_vs_mid | 4.591431e-02 | 2.158741e-02 | 1.169055e-01 | 6.787702e+00 | 1.478341e+02 |
| p_total | mid_vs_fine | 1.068735e-02 | 5.219589e-03 | 2.366878e-02 | 1.583181e+00 | 1.481360e+02 |
| p_total | coarse_vs_fine | 5.588622e-02 | 2.661574e-02 | 1.404809e-01 | 8.278761e+00 | 1.481360e+02 |
| p_scattered | coarse_vs_mid | 6.718941e-02 | 2.509972e-02 | 1.756904e-01 | 9.680576e+00 | 1.440789e+02 |
| p_scattered | mid_vs_fine | 1.650549e-02 | 6.276842e-03 | 4.135268e-02 | 2.384627e+00 | 1.444748e+02 |
| p_scattered | coarse_vs_fine | 8.316984e-02 | 3.116965e-02 | 2.170431e-01 | 1.201594e+01 | 1.444748e+02 |

| 字段 | ‖p_coarse−p_mid‖₂ | ‖p_mid−p_fine‖₂ | observed order |
|---|---|---|---|
| p_total | 6.787702e+00 | 1.583181e+00 | 2.1001 |
| p_scattered | 9.680576e+00 | 2.384627e+00 | 2.0213 |

### high500

共同节点：241×241（coarse air nodes (direct node extraction, no interpolation)）

| 字段 | pair | RE | MAE | max | ||num-ref|| | ||ref|| |
|---|---|---|---|---|---|---|
| p_total | coarse_vs_mid | 1.365820e+00 | 6.833291e-01 | 8.498014e+00 | 2.076551e+02 | 1.520369e+02 |
| p_total | mid_vs_fine | 1.298351e+00 | 6.735813e-01 | 6.594233e+00 | 2.037071e+02 | 1.568967e+02 |
| p_total | coarse_vs_fine | 1.401043e+00 | 7.290795e-01 | 7.348064e+00 | 2.198191e+02 | 1.568967e+02 |
| p_scattered | coarse_vs_mid | 1.469683e+00 | 8.928587e-01 | 8.477124e+00 | 2.940246e+02 | 2.000599e+02 |
| p_scattered | mid_vs_fine | 1.418549e+00 | 8.608528e-01 | 7.107442e+00 | 2.764543e+02 | 1.948852e+02 |
| p_scattered | coarse_vs_fine | 1.548406e+00 | 9.361109e-01 | 7.764478e+00 | 3.017614e+02 | 1.948852e+02 |

| 字段 | ‖p_coarse−p_mid‖₂ | ‖p_mid−p_fine‖₂ | observed order |
|---|---|---|---|
| p_total | 2.076551e+02 | 2.037071e+02 | 0.0277 |
| p_scattered | 2.940246e+02 | 2.764543e+02 | 0.0889 |

## 时间收敛（固定 mid 网格）

### low100

固定节点：481×481（fixed mid air nodes (no interpolation, no phase shift)）

| 字段 | pair | RE | MAE | max | ||num-ref|| | ||ref|| |
|---|---|---|---|---|---|---|
| p_total | mid_vs_mid_dt2 | 6.042122e-02 | 3.402801e-02 | 1.085068e-01 | 1.900307e+01 | 3.145098e+02 |
| p_total | mid_dt2_vs_mid_dt4 | 3.211507e-02 | 1.869710e-02 | 5.920693e-02 | 1.043499e+01 | 3.249251e+02 |
| p_total | mid_vs_mid_dt4 | 9.059760e-02 | 5.272436e-02 | 1.677137e-01 | 2.943743e+01 | 3.249251e+02 |
| p_scattered | mid_vs_mid_dt2 | 5.991013e-02 | 2.637058e-02 | 1.386224e-01 | 1.833583e+01 | 3.060556e+02 |
| p_scattered | mid_dt2_vs_mid_dt4 | 3.180531e-02 | 1.447551e-02 | 7.555877e-02 | 1.005339e+01 | 3.160916e+02 |
| p_scattered | mid_vs_mid_dt4 | 8.981144e-02 | 4.084526e-02 | 2.140973e-01 | 2.838864e+01 | 3.160916e+02 |

| 字段 | ‖p_mid−p_mid_dt2‖₂ | ‖p_mid_dt2−p_mid_dt4‖₂ | observed order |
|---|---|---|---|
| p_total | 1.900307e+01 | 1.043499e+01 | 0.8648 |
| p_scattered | 1.833583e+01 | 1.005339e+01 | 0.8670 |

### high500

固定节点：481×481（fixed mid air nodes (no interpolation, no phase shift)）

| 字段 | pair | RE | MAE | max | ||num-ref|| | ||ref|| |
|---|---|---|---|---|---|---|
| p_total | mid_vs_mid_dt2 | 6.161319e-02 | 3.505794e-02 | 3.139418e-01 | 1.990625e+01 | 3.230842e+02 |
| p_total | mid_dt2_vs_mid_dt4 | 3.242993e-02 | 1.905827e-02 | 1.751551e-01 | 1.082055e+01 | 3.336592e+02 |
| p_total | mid_vs_mid_dt4 | 9.201408e-02 | 5.407882e-02 | 4.890969e-01 | 3.070134e+01 | 3.336592e+02 |
| p_scattered | mid_vs_mid_dt2 | 6.245032e-02 | 4.372853e-02 | 3.328435e-01 | 2.658590e+01 | 4.257127e+02 |
| p_scattered | mid_dt2_vs_mid_dt4 | 3.251169e-02 | 2.352471e-02 | 1.845117e-01 | 1.429909e+01 | 4.398138e+02 |
| p_scattered | mid_vs_mid_dt4 | 9.289688e-02 | 6.720285e-02 | 5.170428e-01 | 4.085733e+01 | 4.398138e+02 |

| 字段 | ‖p_mid−p_mid_dt2‖₂ | ‖p_mid_dt2−p_mid_dt4‖₂ | observed order |
|---|---|---|---|
| p_total | 1.990625e+01 | 1.082055e+01 | 0.8794 |
| p_scattered | 2.658590e+01 | 1.429909e+01 | 0.8947 |

## 复相量（补充）

未计算：all runs were saved without --save-phasor; supplementary phasor comparison skipped

## 求解器诊断（原样记录）

| case/res | ppw | under-resolved | steady_ok | rel_growth | max\|p\| seen | broke_nan | diverged |
|---|---|---|---|---|---|---|---|
| low100/coarse | 27.20 | False | True | 0.0030315544829827823 | 1.756676e+00 | False | False |
| low100/mid | 54.40 | False | True | 0.0024610602237799635 | 1.763598e+00 | False | False |
| low100/fine | 108.80 | False | True | 0.0023206051767070274 | 1.771738e+00 | False | False |
| low100/mid_dt2 | 54.40 | False | True | 0.0026165961660150187 | 1.873791e+00 | False | False |
| low100/mid_dt4 | 54.40 | False | True | 0.002700667366200154 | 1.933942e+00 | False | False |
| high500/coarse | 5.44 | True | True | 0.00038038558641608503 | 5.847807e+00 | False | False |
| high500/mid | 10.88 | False | True | 3.198136222045135e-05 | 6.069618e+00 | False | False |
| high500/fine | 21.76 | False | True | 6.838615520091376e-05 | 6.657144e+00 | False | False |
| high500/mid_dt2 | 10.88 | False | True | 3.322848665504818e-05 | 6.451418e+00 | False | False |
| high500/mid_dt4 | 10.88 | False | True | 3.31021551557609e-05 | 6.660112e+00 | False | False |

## 边界与注意

- 完整 10/10 次真实求解均 status=ok，协议身份、代码/配置哈希、coords 嵌套与终时相位均已核验后才聚合。
- 细网格（fine）只是数值参考，不是解析真解；本实验只比较同一物理终时快照。
- 稳态/有限值/blow-up 诊断按原样记录，不作为压力收敛的证据；single final-time snapshot at >=4 air crossings 不等同稳态测量。
- observed order 由同节点绝对 L2 差范数之比计算（log₂(‖p_coarse−p_mid‖₂/‖p_mid−p_fine‖₂)/log₂(2)），不使用 RE 之比；仅对差范数严格单调下降且分母非零报告，非单调/零分母显式 undefined。
- 未使用插值或任意相位平移；空间抽取为整数节点索引，时间三点固定同一 mid 网格。
- 不设收敛通过阈值：不因出现正收敛或负收敛改变协议或挑选参数。
- ppw<10 的欠分辨诊断点仍保留在比较中，不剔除：high500/coarse
