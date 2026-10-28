# R2-REMAINING-FINALIZE-03 最终自查报告

**Status: 66/66 residual trainings + Transolver 5seed + FNO/UNO 5seed (10) complete; total 81 trainings; final self-check all_pass=true. Codex review pending.**

## 1. Counts and checks

| root | metrics | checkpoints | seeds |
|---|---:|---:|---|
| transolver | 5 | 5 | [0, 1, 2, 3, 4] |
| maininit | 12 | 12 | [3, 4] |
| kaiming | 14 | 14 | [3, 4] |
| acoustic | 14 | 14 | [3, 4] |
| matched | 6 | 6 | [3, 4] |
| siren | 5 | 5 | [0, 1, 2, 3, 4] |
| pt | 5 | 5 | [0, 1, 2, 3, 4] |
| mlp | 10 | 10 | [0, 1, 2, 3, 4] |

- Training total (excluding Transolver): **66/66**
- Transolver: **5/5** seeds; checkpoints on server.
- Fresh summaries: SIREN, PT, MLP-c, MLP-c-matched all **n=5**.
- Combined: groups **23**, missing **0**; no fabricated seeds.
- Registry snapshot entries: **5**; canonical contains transolver: **False**.
- Self-check errors: **[]**

## 2. Completed aggregate values

### corrected acoustic (old 0-2 + new 3/4, n=5)

| model | p_total RE mean | std |
|---|---:|---:|
| fourier_c | 1.200538 | 0.019785 |
| fourier_nc | 1.069533 | 0.003407 |
| unif_acous | 1.166632 | 0.024691 |
| ab_L8 | 1.228006 | 0.029527 |
| ab_L32 | 1.169858 | 0.010579 |
| ab_nores | 1.151513 | 0.048255 |
| ab_nonorm | 1.185998 | 0.022483 |

### matched physics (old 0-2 + new 3/4, n=5)

| objective | u RE mean |
|---|---:|
| supervised | 0.136254 |
| bc | 0.136644 |
| bc_pde | 0.268385 |

## 3. Fresh strict-FP32 5-seed

| row | RE |
|---|---|
| SIREN | u/v/p/T 0.08551, 0.07087, 0.02474, 0.02837 |
| PointTransformer | u/v/p/T 0.17920, 0.13686, 0.06429, 0.06021 |
| acoustic MLP-c | p_total 1.002335 |
| acoustic MLP-c matched | p_total 1.000938 |

## 4. Transolver

- RE mean: u 0.076928, v 0.057056, p 0.020470, T 0.018936
- RE std: u 0.001166, v 0.001355, p 0.000993, T 0.000857
- Checkpoints retained on server; SHA in checkpoint_manifest.json

## 5. Job ledger

- 51492 COMPLETED 32:19 (Transolver 5seed)
- 51497 COMPLETED 31:12 (Kaiming 14)
- 51508 COMPLETED 6:55:30 (acoustic14 + matched6 + SIREN5 + PT5 + MLP10)
- 51495/51496/51498/51507 failed attempts preserved; successful seeds not rerun

## 6. Guards
- No canonical results/registry.json write; isolated registry snapshot only
- No paper/data/weights overwrite; no upload/publish/FDTD/dependency install
- FNO/UNO new strict5 runs are independent; old 3-seed results are historical and read-only (see fno_uno_audit.json).

## 7. R2-REMAINING-FINALIZE-03 additions (10 FNO/UNO + acceptance)

- **51529 FNO COMPLETED 6:41**, **51530 UNO COMPLETED 30:48**; 5 seeds each.
- FNO params 1,052,049; UNO params 1,181,601; both strict FP32, TF32 matmul/cudnn false, no AMP.
- Protocol: 6 train cases, train-only norm 1,388,166 points, fullgrid 6 steps/epoch, 1000 epochs, Adam lr 1e-3, val every 50, 9 left test + srcR diagnostic, no test selection.
- FNO/UNO checkpoint validation: 10/10 loaded, epoch/val match, state finite; all_pass=true.
- Test primary9 p_total RE: FNO **1.175185±0.016156**, UNO **1.139454±0.059089**; 400_0.5_srcR is diagnostic only.
- Actual formal logs contain `epoch=1000/1000` five times per model; steps=6 in every metric; no AMP; TF32 false.
- Acceptance inputs: 66 new seed metrics/protocols, 115 sources in 23 groups with SHA/val/params/version, 71/71 checkpoint load validation, fresh 4x n=5, Transolver preflight-vs-final adapter SHA audit, shared-source impact audit.
- Final counts: 66 residual matrix + 5 Transolver + 10 FNO/UNO = **81 trainings**; combined missing=0; canonical registry untouched.
