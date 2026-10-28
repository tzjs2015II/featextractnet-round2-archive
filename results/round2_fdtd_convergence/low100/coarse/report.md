# R2-FDTD-CONV-01 / low100 / coarse

- status: **ok**
- written: 2026-10-21T08:35:38+00:00
- wall time: 577.72 s

## Fixed physical protocol

- f = 100 Hz, Ma = 0.3, source side = left at x = -15 m
- c0 = 340 m/s, rho0 = 1, L = 1, alpha = 1.25643, gamma_coef = 2.8, sigma = 100
- air domain = [-15.0, 15.0] m, PML thickness = 10 m/side

## Numerical plan (fixed, written before the solve)

- dx = dy = 0.125 m -> 401 nodes total, 241 air nodes
- n_pml = 80 cells, n_buffer = 120 cells
- spp = 385 steps/period, n_steps = 13860
- dt = 2.597402597403e-05 s, t_final = 0.360000000000 s = Ncycle/f (Ncycle = 36)
- courant_factor (back-solved) = 0.099913270 (target CFL <= 0.1)
- points per wavelength = 27.20

## Realized solver parameters and interface checks

| check | ok | actual | expected |
|---|---|---|---|
| result_schema | True | null | null |
| dt_matches_plan | True | 2.597402597402598e-05 | 2.5974025974025975e-05 |
| t_final_matches_identity | True | 0.36000000000000004 | 0.36 |
| final_phase_cycles | True | 36.00000000000001 | 36.0 |
| final_source_phase_sin | True | 1.9604252476543064e-14 | 0.0 |
| courant_factor_matches_plan | True | 0.09991326986116414 | 0.09991326986116414 |
| effective_cfl_within_target | True | 0.09991326986116414 | 0.1 |
| grid_cells | True | [400, 400] | [400, 400] |
| grid_dx | True | [0.125, 0.125] | [0.125, 0.125] |
| pml_buffer_cells | True | [80, 120] | [80, 120] |
| air_node_range | True | [80, 321, 80, 321] | [80, 321, 80, 321] |
| n_steps_done | True | 13860 | 13860 |
| source_side | True | "left" | "left" |
| source_physical_position | True | -15.0 | -15.0 |
| air_coords_x_exact | True | 0.0 | 0.0 |
| air_coords_y_exact | True | 0.0 | 0.0 |
| field_shapes | True | {"p_total": [241, 241], "p_incident": [241, 241], "p_scattered": [241, 241]} | [241, 241] |
| fields_finite | True | {"p_total": true, "p_incident": true, "p_scattered": true} | true |
| diag_not_broke_nan | True | false | false |
| diag_not_diverged | True | false | false |
| physical_constants_match_identity | True | {"c0": 340.0, "rho0": 1.0, "L": 1.0, "alpha": 1.256431, "gamma_coef": 2.8, "sigma_pml": 100.0, "f_hz": 100.0, "Ma": 0.3} | {"c0": 340.0, "rho0": 1.0, "L": 1.0, "alpha": 1.256431, "gamma_coef": 2.8, "sigma_pml": 100.0, "f_hz": 100.0, "Ma": 0.3} |

## Solver diagnostics (recorded, not convergence evidence)

- n_steps_done = 13860
- broke_nan = False, diverged = False
- max |p| seen = 1.7566763629516706, rms p_air final = 0.6115804361717376
- steady = {"method": "phase-locked integer-period mean", "n_samples": 1540, "steps_per_period": 385, "n_period": 4, "steady_ok": true, "rel_growth": 0.0030315544829827823, "phase_err": 1.4764524379430656e-16, "tol": 0.02, "energy_per_period": [0.3744512777705806, 0.37558644722036466, 0.3757335559079252, 0.37576403201020797], "rel_per_period": [0.0030315544829827823, 0.0003916773053160877, 8.111094099421645e-05], "mean_energy_last": 0.37576403201020797}

These finite / steady / blow-up diagnostics are model-side evidence only. The convergence comparison of this experiment uses the real ``p_total`` final-time snapshots, not the steady flag.

## Outputs

- protocol (written pre-solve): ../round2_fdtd_convergence/low100/coarse/protocol.json (sha256 28779bfd6cdb3ee6...)
- fields: ../round2_fdtd_convergence/low100/coarse/fields.npz (sha256 2c2749c9f963bda3...)

- environment: {"hostname": "gn1", "platform": "Linux-6.8.0-57-generic-x86_64-with-glibc2.35", "python": "3.13.9", "numpy": "2.4.2", "cpu_count": 96, "slurm": {"SLURM_JOB_ID": "50907", "SLURM_JOBID": "50907", "SLURM_JOB_NODELIST": "gn1", "SLURM_JOB_PARTITION": "compute", "SLURM_CPUS_PER_TASK": "4"}}
