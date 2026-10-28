# R2-FDTD-CONV-01 / low100 / fine

- status: **ok**
- written: 2026-10-22T00:26:15+00:00
- wall time: 43975.46 s

## Fixed physical protocol

- f = 100 Hz, Ma = 0.3, source side = left at x = -15 m
- c0 = 340 m/s, rho0 = 1, L = 1, alpha = 1.25643, gamma_coef = 2.8, sigma = 100
- air domain = [-15.0, 15.0] m, PML thickness = 10 m/side

## Numerical plan (fixed, written before the solve)

- dx = dy = 0.03125 m -> 1601 nodes total, 961 air nodes
- n_pml = 320 cells, n_buffer = 480 cells
- spp = 1540 steps/period, n_steps = 55440
- dt = 6.493506493506e-06 s, t_final = 0.360000000000 s = Ncycle/f (Ncycle = 36)
- courant_factor (back-solved) = 0.099913270 (target CFL <= 0.1)
- points per wavelength = 108.80

## Realized solver parameters and interface checks

| check | ok | actual | expected |
|---|---|---|---|
| result_schema | True | null | null |
| dt_matches_plan | True | 6.493506493506495e-06 | 6.493506493506494e-06 |
| t_final_matches_identity | True | 0.36000000000000004 | 0.36 |
| final_phase_cycles | True | 36.00000000000001 | 36.0 |
| final_source_phase_sin | True | 1.9604252476543064e-14 | 0.0 |
| courant_factor_matches_plan | True | 0.09991326986116414 | 0.09991326986116414 |
| effective_cfl_within_target | True | 0.09991326986116414 | 0.1 |
| grid_cells | True | [1600, 1600] | [1600, 1600] |
| grid_dx | True | [0.03125, 0.03125] | [0.03125, 0.03125] |
| pml_buffer_cells | True | [320, 480] | [320, 480] |
| air_node_range | True | [320, 1281, 320, 1281] | [320, 1281, 320, 1281] |
| n_steps_done | True | 55440 | 55440 |
| source_side | True | "left" | "left" |
| source_physical_position | True | -15.0 | -15.0 |
| air_coords_x_exact | True | 0.0 | 0.0 |
| air_coords_y_exact | True | 0.0 | 0.0 |
| field_shapes | True | {"p_total": [961, 961], "p_incident": [961, 961], "p_scattered": [961, 961]} | [961, 961] |
| fields_finite | True | {"p_total": true, "p_incident": true, "p_scattered": true} | true |
| diag_not_broke_nan | True | false | false |
| diag_not_diverged | True | false | false |
| physical_constants_match_identity | True | {"c0": 340.0, "rho0": 1.0, "L": 1.0, "alpha": 1.256431, "gamma_coef": 2.8, "sigma_pml": 100.0, "f_hz": 100.0, "Ma": 0.3} | {"c0": 340.0, "rho0": 1.0, "L": 1.0, "alpha": 1.256431, "gamma_coef": 2.8, "sigma_pml": 100.0, "f_hz": 100.0, "Ma": 0.3} |

## Solver diagnostics (recorded, not convergence evidence)

- n_steps_done = 55440
- broke_nan = False, diverged = False
- max |p| seen = 1.7717377054163113, rms p_air final = 0.6161826107501716
- steady = {"method": "phase-locked integer-period mean", "n_samples": 6160, "steps_per_period": 1540, "n_period": 4, "steady_ok": true, "rel_growth": 0.0023206051767070274, "phase_err": 1.4764524379430656e-16, "tol": 0.02, "energy_per_period": [0.3792890504494843, 0.38016923058342567, 0.3802851136601109, 0.3803063361980719], "rel_per_period": [0.0023206051767070274, 0.00030481971543939236, 5.5806912231554375e-05], "mean_energy_last": 0.3803063361980719}

These finite / steady / blow-up diagnostics are model-side evidence only. The convergence comparison of this experiment uses the real ``p_total`` final-time snapshots, not the steady flag.

## Outputs

- protocol (written pre-solve): ../round2_fdtd_convergence/low100/fine/protocol.json (sha256 d21b3cbc927f6f45...)
- fields: ../round2_fdtd_convergence/low100/fine/fields.npz (sha256 a2df9c020c0ffd5d...)

- environment: {"hostname": "gn1", "platform": "Linux-6.8.0-57-generic-x86_64-with-glibc2.35", "python": "3.13.9", "numpy": "2.4.2", "cpu_count": 96, "slurm": {"SLURM_JOB_ID": "50910", "SLURM_JOBID": "50910", "SLURM_JOB_NODELIST": "gn1", "SLURM_JOB_PARTITION": "compute", "SLURM_CPUS_PER_TASK": "4"}}
