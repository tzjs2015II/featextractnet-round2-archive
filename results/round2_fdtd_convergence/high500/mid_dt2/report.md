# R2-FDTD-CONV-01 / high500 / mid_dt2

- status: **ok**
- written: 2026-10-23T01:16:11+00:00
- wall time: 9993.80 s

## Fixed physical protocol

- f = 500 Hz, Ma = 0.7, source side = left at x = -15 m
- c0 = 340 m/s, rho0 = 1, L = 1, alpha = 1.25643, gamma_coef = 2.8, sigma = 100
- air domain = [-15.0, 15.0] m, PML thickness = 10 m/side

## Numerical plan (fixed, written before the solve)

- dx = dy = 0.0625 m -> 801 nodes total, 481 air nodes
- n_pml = 160 cells, n_buffer = 240 cells
- spp = 308 steps/period, n_steps = 54516
- dt = 6.493506493506e-06 s, t_final = 0.354000000000 s = Ncycle/f (Ncycle = 177)
- courant_factor (back-solved) = 0.049956635 (target CFL <= 0.1)
- points per wavelength = 10.88

## Realized solver parameters and interface checks

| check | ok | actual | expected |
|---|---|---|---|
| result_schema | True | null | null |
| dt_matches_plan | True | 6.493506493506495e-06 | 6.493506493506494e-06 |
| t_final_matches_identity | True | 0.35400000000000004 | 0.354 |
| final_phase_cycles | True | 177.00000000000003 | 177.0 |
| final_source_phase_sin | True | 4.901805895899672e-14 | 0.0 |
| courant_factor_matches_plan | True | 0.04995663493058207 | 0.04995663493058207 |
| effective_cfl_within_target | True | 0.04995663493058207 | 0.1 |
| grid_cells | True | [800, 800] | [800, 800] |
| grid_dx | True | [0.0625, 0.0625] | [0.0625, 0.0625] |
| pml_buffer_cells | True | [160, 240] | [160, 240] |
| air_node_range | True | [160, 641, 160, 641] | [160, 641, 160, 641] |
| n_steps_done | True | 54516 | 54516 |
| source_side | True | "left" | "left" |
| source_physical_position | True | -15.0 | -15.0 |
| air_coords_x_exact | True | 0.0 | 0.0 |
| air_coords_y_exact | True | 0.0 | 0.0 |
| field_shapes | True | {"p_total": [481, 481], "p_incident": [481, 481], "p_scattered": [481, 481]} | [481, 481] |
| fields_finite | True | {"p_total": true, "p_incident": true, "p_scattered": true} | true |
| diag_not_broke_nan | True | false | false |
| diag_not_diverged | True | false | false |
| physical_constants_match_identity | True | {"c0": 340.0, "rho0": 1.0, "L": 1.0, "alpha": 1.256431, "gamma_coef": 2.8, "sigma_pml": 100.0, "f_hz": 500.0, "Ma": 0.7} | {"c0": 340.0, "rho0": 1.0, "L": 1.0, "alpha": 1.256431, "gamma_coef": 2.8, "sigma_pml": 100.0, "f_hz": 500.0, "Ma": 0.7} |

## Solver diagnostics (recorded, not convergence evidence)

- n_steps_done = 54516
- broke_nan = False, diverged = False
- max |p| seen = 6.451418134896154, rms p_air final = 0.6716927904580767
- steady = {"method": "phase-locked integer-period mean", "n_samples": 1232, "steps_per_period": 308, "n_period": 4, "steady_ok": true, "rel_growth": 3.322848665504818e-05, "phase_err": 1.845565547428832e-16, "tol": 0.02, "energy_per_period": [0.4558606973449488, 0.4558737238022617, 0.45588887179620946, 0.4558904378452177], "rel_per_period": [2.857552183990923e-05, 3.322848665504818e-05, 3.4351551554220767e-06], "mean_energy_last": 0.4558904378452177}

These finite / steady / blow-up diagnostics are model-side evidence only. The convergence comparison of this experiment uses the real ``p_total`` final-time snapshots, not the steady flag.

## Outputs

- protocol (written pre-solve): ../round2_fdtd_convergence/high500/mid_dt2/protocol.json (sha256 d5ac2a41ca5661b2...)
- fields: ../round2_fdtd_convergence/high500/mid_dt2/fields.npz (sha256 cb70470da9fd52e9...)

- environment: {"hostname": "gn1", "platform": "Linux-6.8.0-57-generic-x86_64-with-glibc2.35", "python": "3.13.9", "numpy": "2.4.2", "cpu_count": 96, "slurm": {"SLURM_JOB_ID": "50910", "SLURM_JOBID": "50910", "SLURM_JOB_NODELIST": "gn1", "SLURM_JOB_PARTITION": "compute", "SLURM_CPUS_PER_TASK": "4"}}
