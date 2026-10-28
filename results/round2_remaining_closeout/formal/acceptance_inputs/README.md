# R2-REMAINING-FINALIZE-03 acceptance inputs

Status: 66/66 residual seed-extension trainings, Transolver 5seed and
FNO/UNO 5seed (10) are complete. FNO job 51529 and UNO job 51530 both
COMPLETED; FNO/UNO verification all_pass=true. No paper/DOI/data edits.

## What is in this directory

- `new_seed_manifest.json` — all **66 new seed metrics** (old paths retained), with
  metric SHA-256, seed, val-best epoch/score, checkpoint path/SHA/load/epoch/val
  status. Metrics files were pulled back under their original
  `results/round2_remaining_closeout/seed_extension/...` paths.
- `group_manifest.json` — **23 combined groups**, 115 old+new source metric
  files; each entry lists path, seed, val-best, params, implementation/protocol
  signature, metrics SHA, checkpoint path/SHA/load/epoch/val status;
  `version_match` and `param_match`.
- `checkpoint_validation.json` — **71/71** new metrics + Transolver checkpoint
  files actually loaded with `torch.load(map_location='cpu')`; epoch, val score,
  state_dict finite and parameter count checked; all_pass=true, errors=[].
- `fresh_manifest.json` — fresh SIREN/PT/MLP-c/MLP-c-matched **20 metrics**, all
  summaries n=5.
- `transolver_source_audit.json` — preflight vs final adapter SHA, B2N change,
  preflight/formal shapes, five Transolver checkpoint loads. Metrics do not embed
  a training-time source SHA; the final adapter SHA is inferred from job
  chronology, explicitly not cryptographic binding.
- `shared_source_impact.json` — current source hashes and impact/limitations for
  already-saved weights and old read-only metrics.
- `transolver_final_adapter.py` — byte copy of the final adapter source.
- `r2_metrics_protocols.tgz` — original archive of pulled metrics/summaries.

## FNO/UNO strict5 (new independent protocol)

Frozen: 6 train cases; train-only normalization 1,388,166 points; fullgrid
6 steps/epoch; 1000 epochs; Adam lr 1e-3; val every 50 with val-only selection;
9 left test cases plus `400_0.5_srcR` diagnostic; strict FP32, TF32 off, no AMP;
seeds 0–4; no test-based selection/early stopping/tuning. Old FNO/UNO results
are read-only.

- **51529 (FNO) COMPLETED 6:41**; **51530 (UNO) COMPLETED 30:48**.
- FNO params 1,052,049; UNO params 1,181,601.
- `fno_uno_manifest.json` lists 10 metric files, val-best, params, protocol
  signatures, checkpoint SHA/load and old-result hashes.
- GPU preflight passed; see `../fno_uno_strict5/preflight.json`.
- `../fno_uno_strict5/checkpoint_validation.json`: 10/10 loaded, epoch/val
  match, state finite; all_pass=true.

## Important limitation

- `combined_summary.json` is derived by reading old 0–2 and new 3/4; no seed is
  fabricated. The old 0–2 metrics remain read-only.
- Current source code is not claimed to reproduce old test values bitwise; saved
  checkpoints were not rewritten. See `shared_source_impact.json`.
