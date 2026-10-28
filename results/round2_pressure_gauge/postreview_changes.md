# R2-PRESSURE-GAUGE-02 postreview changes (2026-10-23)

Original 50961 outputs are byte-unchanged. This change set derives new summary/report/protocol/delivery from the original raw_results, and adds SLURM-51017 array probes.

1. Report primary statistics now use **mean/sample std across the five seed means** (each seed mean over 20 geometries), not the std of 20 geometry means. Geometry std is a separate auxiliary column. MAE/max included.
2. Primary paired bootstrap: for each geometry average the five seeds first, then FEN-graph paired difference; 20,000 resamples, analysis seed 20261021, one shared geometry resample index for all metrics. Per-seed bootstrap retained as supplementary.
3. Baseline inventory: actual checked roots/seed existence recorded in `baseline_inventory.json` and report; no unverified blanket "missing" claim.
4. Array reproducibility: SLURM job 51017 exported FEN/graph seed0 geom17 actual `pred`, `ref`, `pred_p`, `ref_p` (5000 each) plus checkpoint/metrics hashes. Independent NumPy recomputation matches the exported arrays; raw comparison max abs diff 2.63e-10 (separate GPU runs), tolerance 1e-8 documented.

No GPU full rerun; no training; original protocol/raw/summary/report/delivery_50961 not overwritten.
