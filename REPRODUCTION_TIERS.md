# Reproduction tiers

1. **Table reconstruction (complete):** read the retained per-seed/per-case
   metric/protocol JSON and the four `results/round2_audit/*table_sources.json`
   manifests. Displayed means/sample std and n are reconstructed without
   checkpoints. The retained metric JSON is authoritative.
2. **Direct evaluation (complete current Table 1):** all 85 current Table-1
   checkpoints are included (`weights/WEIGHTS_MANIFEST.json`): 50 thermal
   (10 models x 5 seeds) and 35 acoustic (7 models x 5 seeds).
3. **Full retraining (protocols and 22 raw inputs retained):** commands and
   fixed protocols are in the README and metric protocol JSON. Both thermal
   and acoustic raw inputs are present. No training is performed by this
   package.

Checkpoint presence is not required for Tier 1. Tier 2 covers all 85 current table-model/seed checkpoints.
