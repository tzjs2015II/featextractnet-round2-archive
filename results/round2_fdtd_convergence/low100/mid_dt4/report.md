# R2-FDTD-CONV-01 / low100 / mid_dt4

- status: **ok**
- written: 2026-10-22T09:00:45+00:00
- wall time: 20509.56 s

## Fixed physical protocol

- f = 100 Hz, Ma = 0.3, source side = left at x = -15 m
- c0 = 340 m/s, rho0 = 1, L = 1, alpha = 1.25643, gamma_coef = 2.8, sigma = 100
- air domain = [-15.0, 15.0] m, PML thickness = 10 m/side

## Numerical plan (fixed, written before the solve)

- dx = dy = 0.0625 m -> 801 nodes total, 481 air nodes
- n_pml = 160 cells, n_buffer = 240 cells
- spp = 3080 steps/period, n_steps = 110880
- dt = 3.246753246753e-06 s, t_final = 0.360000000000 s = Ncycle/f (Ncycle = 36)
- courant_factor (back-solved) = 0.024978317 (target CFL <= 0.1)
- points per wavelength = 54.40

## Realized solver parameters and interface checks

| check | ok | actual | expected |
|---|---|---|---|
| result_schema | True | null | null |
| dt_matches_plan | True | 3.2467532467532473e-06 | 3.246753246753247e-06 |
| t_final_matches_identity | True | 0.36000000000000004 | 0.36 |
| final_phase_cycles | True | 36.00000000000001 | 36.0 |
| final_source_phase_sin | True | 1.9604252476543064e-14 | 0.0 |
| courant_factor_matches_plan | True | 0.024978317465291035 | 0.024978317465291035 |
| effective_cfl_within_target | True | 0.024978317465291035 | 0.1 |
| grid_cells | True | [800, 800] | [800, 800] |
| grid_dx | True | [0.0625, 0.0625] | [0.0625, 0.0625] |
| pml_buffer_cells | True | [160, 240] | [160, 240] |
| air_node_range | True | [160, 641, 160, 641] | [160, 641, 160, 641] |
| n_steps_done | True | 110880 | 110880 |
| source_side | True | "left" | "left" |
| source_physical_position | True | -15.0 | -15.0 |
| air_coords_x_exact | True | 0.0 | 0.0 |
| air_coords_y_exact | True | 0.0 | 0.0 |
| field_shapes | True | {"p_total": [481, 481], "p_incident": [481, 481], "p_scattered": [481, 481]} | [481, 481] |
| fields_finite | True | {"p_total": true, "p_incident": true, "p_scattered": true} | true |
| diag_not_broke_nan | True | false | false |
| diag_not_diverged | True | false | false |
| physical_constants_match_identity | True | {"c0": 340.0, "rho0": 1.0, "L": 1.0, "alpha": 1.256431, "gamma_coef": 2.8, "sigma_pml": 100.0, "f_hz": 100.0, "Ma": 0.3} | {"c0": 340.0, "rho0": 1.0, "L": 1.0, "alpha": 1.256431, "gamma_coef": 2.8, "sigma_pml": 100.0, "f_hz": 100.0, "Ma": 0.3} |

## Solver diagnostics (recorded, not convergence evidence)

- n_steps_done = 110880
- broke_nan = False, diverged = False
- max |p| seen = 1.9339417574471525, rms p_air final = 0.6755199171224758
- steady = {"method": "phase-locked integer-period mean", "n_samples": 12320, "steps_per_period": 3080, "n_period": 4, "steady_ok": true, "rel_growth": 0.002700667366200154, "phase_err": 1.4764524379430656e-16, "tol": 0.02, "energy_per_period": [0.45581310493422017, 0.4570441045118024, 0.4571970931489361, 0.45721600826787456], "rel_per_period": [0.002700667366200154, 0.00033473495363673165, 4.1371914261748476e-05], "mean_energy_last": 0.45721600826787456}

These finite / steady / blow-up diagnostics are model-side evidence only. The convergence comparison of this experiment uses the real ``p_total`` final-time snapshots, not the steady flag.

## Outputs

- protocol (written pre-solve): ../round2_fdtd_convergence/low100/mid_dt4/protocol.json (sha256 3edce91b25258f99...)
- fields: ../round2_fdtd_convergence/low100/mid_dt4/fields.npz (sha256 4c2af971a424b3b2...)

- environment: {"hostname": "gn1", "platform": "Linux-6.8.0-57-generic-x86_64-with-glibc2.35", "python": "3.13.9", "numpy": "2.4.2", "cpu_count": 96, "slurm": {"SLURM_JOB_ID": "50910", "SLURM_JOBID": "50910", "SLURM_JOB_NODELIST": "gn1", "SLURM_JOB_PARTITION": "compute", "SLURM_CPUS_PER_TASK": "4"}}
