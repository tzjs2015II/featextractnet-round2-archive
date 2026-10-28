# FNO/UNO independent strict5 (R2-REMAINING-FINALIZE-03)

Frozen protocol: 6 train cases from `configs/acoustic_cases.json`; train-only
normalization 1,388,166 points; fullgrid with six case steps per epoch
(`steps_per_epoch=6`); 1000 epochs; Adam lr 1e-3; val every 50; validation-only
checkpoint selection; test aggregate over 9 fixed-left cases with
`400_0.5_srcR` as a separate diagnostic row; strict FP32, TF32 matmul/cudnn
disabled, no AMP; seeds 0-4.

Results:

- FNO job 51529 COMPLETED 6:41; 1,052,049 parameters; primary9 p_total RE
  1.175185 +/- 0.016156.
- UNO job 51530 COMPLETED 30:48; 1,181,601 parameters; primary9 p_total RE
  1.139454 +/- 0.059089.

Old 3-seed FNO/UNO results are read-only and are not mixed into these means.
Checkpoints are retained on the server; actual load/epoch/val/hash checks are in
`checkpoint_validation.json` and the package acceptance manifest. Raw acoustic
arrays and weights are excluded from the code package; see DATA_SOURCES.md.
