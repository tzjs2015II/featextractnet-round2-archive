# Historical failures and version separation

This index is for provenance only. It does not replace the per-job ledger
`job_ledger.json`.

## Current completed matrix

- 66/66 residual seed-extension trainings (maininit 12, Kaiming 14, acoustic
  14, matched 6, fresh SIREN 5, PT 5, MLP 10) completed.
- Transolver 5 seeds (preflight 51482, formal job 51492) completed.
- FNO/UNO independent strict5: 10 trainings (jobs 51529/51530) completed.

## Preserved failed attempts

- 51490: Transolver CLI lacked `--registry-path`; failed before training.
- 51491: adapter rejected production `[B,2,N]`; fixed to accept both layouts.
- 51493: remote `featextractnet_variants.py` stale (missing init/context);
  fixed and replaced by 51495.
- 51494: remote `exp_acoustic.py` `--seeds` parsed but not applied; fixed.
- 51495: maininit 12 succeeded; Kaiming failed before metrics because output
  directory lacked `r2`; Kaiming was completed by 51497.
- 51496: stale `CondFourierMLP` lacked `fourier_layout`; no metrics.
- 51498: `fourier_c seed3` succeeded; root-level acoustic out-dir caused
  collision at `fourier_nc`; preserved and corrected by later attempts.
- 51507: `fourier_c`/`fourier_nc` seed3 succeeded; `unif_acous` needed
  `--eval-chunk 0`; preserved and corrected by 51508.

## Old FNO/UNO records

Old FNO/UNO `results/acoustic_protocolA/{fno,uno}` 3-seed records have
`steps_per_epoch=3` and a norm string saying 3 train cases, while their
`n_train_points=1388166` corresponds to six 481x481 fields. Those records are
historical only and are not appended to the new FNO/UNO strict5 results.
The new independent protocol uses six train cases, steps=6, fullgrid,
1000 epochs, val every 50 and primary9+srcR diagnostic.

## Old packages

The previous canonical package
`dist/round2/featextractnet_round2_final_checked.*` is retained under
`dist/round2/history/` before the new code package is written. The old
`dist/submission/current_source_review.*` source ZIP is untouched pending the
Codex source freeze.
