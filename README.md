# FeatExtractNet round-two standalone package

**Single new revision-code DOI placeholder:** `10.5281/zenodo.XXXXXXX`

This is a local standalone preparation archive, not a published release. The
old DOI `10.5281/zenodo.22876119` is historical provenance only; no file in
this package is fetched from or depends on that old upload. The package starts
from the Codex-accepted 678-file round-two archive. It includes the attributed
original thermal array, the 21 author-generated acoustic arrays, and all 85
current Table-1 checkpoints.

## Quick start from an empty directory

```sh
mkdir featextractnet
cd featextractnet
tar -xzf ../featextractnet_round2_standalone.tar.gz
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements-extended.txt` and `environment.yml` are retained as inherited
dependency snapshots. No dependency installation is performed by the package;
use a POSIX Python environment for the Linux-targeted CLIs. CUDA is not needed
for the supplied CPU/import checks.

## Data layout

- Acoustic: **included** at `data/acoustic/case_<id>.npz` (21 files; original
  bytes and SHA-256 in `data/acoustic/ACOUSTIC_DATA_MANIFEST.json`). The case
  roles are in `configs/acoustic_cases.json`.
- Thermal: **included** at `data/thermal/HeatTransferData.npy` as third-party
  data with attribution. Original authors Ali Kashefi and Tapan Mukerji,
  J. Comput. Phys. 468 (2022) 111510, https://doi.org/10.1016/j.jcp.2022.111510;
  original URL `https://web.stanford.edu/~kashefi/data/HeatTransferData.npy`; SHA-256 `e9bb20c51105607f2e20b6003fc387e8eaed5ad8906a8f11d8f83bc1cf535e26`; shape
  `(135,5000,6)`. No separate data licence was found; original rights remain
  with the original authors; the file is not relicensed as MIT or CC BY.

## Weights and reproduction tiers

- **Table reconstruction:** use retained metric/protocol JSON; no checkpoint is
  required. Table source manifests map every displayed value to its source.
- **Direct evaluation (complete current Table 1):** `weights/WEIGHTS_MANIFEST.json`
  contains all **85 current Table-1 checkpoints**: 50 thermal (10 models x 5 seeds) and 35 acoustic (7 models x 5 seeds). All are placed at their original relative checkpoint paths. No historical/3-seed/6-case-budget checkpoints are used.
- **Full direct evaluation:** all 85 current Table-1 checkpoints are included;
  all current table weights are included; table reconstruction
  still uses the metric/protocol JSON and does not require checkpoints.
- **Retraining:** protocols and commands are retained; no training is performed.

**Every current table comparison uses five training seeds.** If you rerun, use
the seed/epoch/budget recorded in each metric protocol; do not substitute old
3-seed or old six-case-budget examples.

See `REPRODUCTION_TIERS.md` for the exact hierarchy.

## Existing commands

Thermal (included third-party input; GPU through the user's scheduler):

```sh
python scripts/exp_thermal.py --model feat_extract --seed 0 --epochs 500 --lr 0.001 --batch-geoms 4 --val-every 25 --device cuda --data data/thermal/HeatTransferData.npy --splits configs/splits_thermal.json --pool-context symmetric --strict-fp32 --out-dir results/round2_thermal_reproduction
```

Acoustic (the 21 included arrays; repeat the current five seeds 0-4 and use the metric protocol for any steps/budget argument):

```sh
python scripts/exp_acoustic.py --model fourier_c --seed 0 --epochs 1000 --lr 0.001 --val-every 50 --device cuda --data-cfg configs/acoustic_cases.json --point-mode fullgrid --eval-chunk 0 --fourier-layout pointwise --strict-fp32 --out-dir results/round2_acoustic_reproduction/fourier_c
```

Lightweight checks (no GPU, no training):

```sh
python scripts/exp_thermal.py --help
python scripts/exp_acoustic.py --help
python scripts/exp_remaining_closeout.py --help
python -c "from src.models.baselines.transolver_official.adapter import TransolverAdapter; from src.models.baselines.fno import FNO2d; print('import ok')"
```

## Tables

The displayed paper tables are mapped by the four manifests under
`results/round2_audit/`:

- `thermal_manuscript_table_sources.json` — **Thermal Table 1**.
- `acoustic_manuscript_table_sources.json` — **Acoustic Table 2**.
- `acoustic_ablation_manuscript_sources.json` — acoustic component/ablation table.
- `thermal_geometry_paired_r2_symmetric.json` — paired thermal comparison.

S6 is the `coordinate_mlp_reference` row (n=5, params 814501, CondCoordMLP450/6,
RE 1.0009381961 +/- 0.0022950496), not old `ab_nofourier`/797185. S10 keeps the
original CI for unchanged mlp/pointnet/dgcnn/graph; SIREN/PT/Transolver are
`added_not_displayed`. The retained metric JSON and the four table source
manifests are authoritative for displayed values.

## Licences

See `LICENSES/README.md` and `LICENSES/THIRD-PARTY-NOTICES.md`. Code is MIT;
Transolver is MIT; author-generated acoustic arrays carry a separate CC BY 4.0
notice; thermal data is not redistributed or relicensed.
