# FeatExtractNet standalone reproducible archive

This repository is the **standalone, self-contained reproducibility archive for
FeatExtractNet**. It freezes, in one public place, everything
needed to understand and reproduce the results reported in the manuscript:

- the model source code and experiment scripts;
- the configuration and geometry/split files;
- the attributed input data (135 thermal geometries and the 21 author-generated acoustic cases);
- the retained metric/protocol JSON and result arrays;
- the 85 current Table-1 checkpoints placed at their original relative paths.

It is intended to be cited via a **Zenodo DOI** produced through the Zenodo–GitHub
integration (a release tag on this repository is archived by Zenodo as an immutable
snapshot). The DOI is filled in after the first Zenodo-archived release:

- **Archive DOI:** `10.5281/zenodo.XXXXXXX` (to be updated after the Zenodo release)

The historical identifier `10.5281/zenodo.22876119` is provenance of an earlier local
package only; nothing here is fetched from or depends on that older upload.

## Repository layout (path guide)

| Path | What it contains |
|---|---|
| `src/` | Model source code: network architectures (`src/models/`, including baselines such as FNO, Transolver, graph/PointNet family), physics losses (`src/physics/`), and shared utilities (`src/utils/`). |
| `scripts/` | Experiment and packaging CLIs: `exp_thermal.py`, `exp_acoustic.py`, `exp_remaining_closeout.py`, etc. |
| `configs/` | Geometry/split and acoustic-case definitions, including `splits_thermal.json` and `acoustic_cases.json`. |
| `data/thermal/HeatTransferData.npy` | Third-party thermal input array, shape `(135,5000,6)`, with attribution (see below). |
| `data/acoustic/case_<id>.npz` | The 21 author-generated acoustic arrays; bytes and SHA-256 in `data/acoustic/ACOUSTIC_DATA_MANIFEST.json`. |
| `weights/` | The 85 current Table-1 checkpoints (50 thermal + 35 acoustic) and `WEIGHTS_MANIFEST.json`; checkpoints sit at their original relative paths. |
| `results/` | Retained metric/protocol JSON, result arrays, and the four manuscript table-source manifests under `results/round2_audit/`. |
| `REPRODUCTION_TIERS.md` | Hierarchy of reproduction options (table reconstruction / direct evaluation / retraining). |
| `requirements.txt`, `requirements-extended.txt`, `environment.yml` | Pinned dependency snapshots. |
| `.zenodo.json`, `CITATION.cff` | Zenodo/GitHub citation metadata (title, authors, licence). |
| `LICENSE`, `LICENSES/` | Code licence (MIT) and third-party notices. |
| `PENDING_STATUS.json` | Items still pending (e.g. FDTD pressure space/time convergence); do not cite these as completed. |
| `DATA_SOURCES.md`, `PACKAGE_CONTENTS.md`, `STANDALONE_MANIFEST.json`, `BASE_PACKAGE_MANIFEST.json` | Data provenance, package contents and build manifests. |

## Quick start

```sh
python -m venv .venv
# POSIX:
. .venv/bin/activate
python -m pip install -r requirements.txt
```

No dependency installation is performed automatically, and CUDA is not required for the
supplied CPU/import checks. Retraining uses the seed/epoch/budget recorded in each
metric protocol; every current table comparison uses five seeds (0–4).

## Data attribution

- Thermal data `data/thermal/HeatTransferData.npy` is third-party: Ali Kashefi and
  Tapan Mukerji, *J. Comput. Phys.* 468 (2022) 111510,
  https://doi.org/10.1016/j.jcp.2022.111510; original URL
  `https://web.stanford.edu/~kashefi/data/HeatTransferData.npy`;
  SHA-256 `e9bb20c51105607f2e20b6003fc387e8eaed5ad8906a8f11d8f83bc1cf535e26`.
  It is not relicensed; original rights remain with the original authors.
- The acoustic `.npz` arrays are author-generated and carry a CC BY 4.0 notice.

## Licences

Code is MIT; Transolver is MIT. See `LICENSES/README.md` and
`LICENSES/THIRD-PARTY-NOTICES.md` for full details.
