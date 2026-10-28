# R2-FDTD-CONV-01 / high500 / mid_dt4

- status: **ok**
- written: 2026-10-23T06:52:22+00:00
- wall time: 20169.64 s

## Fixed physical protocol

- f = 500 Hz, Ma = 0.7, source side = left at x = -15 m
- c0 = 340 m/s, rho0 = 1, L = 1, alpha = 1.25643, gamma_coef = 2.8, sigma = 100
- air domain = [-15.0, 15.0] m, PML thickness = 10 m/side

## Numerical plan (fixed, written before the solve)

- dx = dy = 0.0625 m -> 801 nodes total, 481 air nodes
- n_pml = 160 cells, n_buffer = 240 cells
- spp = 616 steps/period, n_steps = 109032
- dt = 3.246753246753e-06 s, t_final = 0.354000000000 s = Ncycle/f (Ncycle = 177)
- courant_factor (back-solved) = 0.024978317 (target CFL <= 0.1)
- points per wavelength = 10.88

## Realized solver parameters and interface checks

| check | ok | actual | expected |
|---|---|---|---|
| result_schema | True | null | null |
| dt_matches_plan | True | 3.2467532467532473e-06 | 3.246753246753247e-06 |
| t_final_matches_identity | True | 0.35400000000000004 | 0.354 |
| final_phase_cycles | True | 177.00000000000003 | 177.0 |
| final_source_phase_sin | True | 4.901805895899672e-14 | 0.0 |
| courant_factor_matches_plan | True | 0.024978317465291035 | 0.024978317465291035 |
| effective_cfl_within_target | True | 0.024978317465291035 | 0.1 |
| grid_cells | True | [800, 800] | [800, 800] |
| grid_dx | True | [0.0625, 0.0625] | [0.0625, 0.0625] |
| pml_buffer_cells | True | [160, 240] | [160, 240] |
| air_node_range | True | [160, 641, 160, 641] | [160, 641, 160, 641] |
| n_steps_done | True | 109032 | 109032 |
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

- n_steps_done = 109032
- broke_nan = False, diverged = False
- max |p| seen = 6.660111835838661, rms p_air final = 0.6936781423971297
- steady = {"method": "phase-locked integer-period mean", "n_samples": 2464, "steps_per_period": 616, "n_period": 4, "steady_ok": true, "rel_growth": 3.31021551557609e-05, "phase_err": 1.845565547428832e-16, "tol": 0.02, "energy_per_period": [0.486312986724499, 0.4863270344266613, 0.48634313289961134, 0.4863444699076298], "rel_per_period": [2.8886134127131477e-05, 3.31021551557609e-05, 2.749104342223546e-06], "mean_energy_last": 0.4863444699076298}

These finite / steady / blow-up diagnostics are model-side evidence only. The convergence comparison of this experiment uses the real ``p_total`` final-time snapshots, not the steady flag.

## Outputs

- protocol (written pre-solve): ../round2_fdtd_convergence/high500/mid_dt4/protocol.json (sha256 a51b0b56a31b8b2d...)
- fields: ../round2_fdtd_convergence/high500/mid_dt4/fields.npz (sha256 1c866e6092faf382...)

- environment: {"hostname": "gn1", "platform": "Linux-6.8.0-57-generic-x86_64-with-glibc2.35", "python": "3.13.9", "numpy": "2.4.2", "cpu_count": 96, "slurm": {"SLURM_JOB_ID": "50910", "SLURM_JOBID": "50910", "SLURM_JOB_NODELIST": "gn1", "SLURM_JOB_PARTITION": "compute", "SLURM_CPUS_PER_TASK": "4"}}
