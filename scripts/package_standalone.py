#!/usr/bin/env python3
"""Build the independent R2-SINGLE-DOI-01 standalone archive.

The archived base is the Codex-accepted 678-file round-two package.  This
builder adds the authors' 21 raw acoustic NPZ files, the attributed original
thermal HeatTransferData.npy, all 85 current Table-1 checkpoints, layered
licence notices, a single-DOI placeholder and a standalone README.  It does not
edit paper, results, model code or raw arrays.

Thermal data is third-party; it is included byte-identically with attribution
and is not relicensed as MIT or CC BY.  No source ZIP, reviewer reply, response
letter or cover letter is embedded in the public archive.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
BASE = WS / "dist/round2/featextractnet_round2_final_checked.tar.gz"
ACOUSTIC = WS / "dist/round2/featextractnet_acoustic_author_data.tar"
ACOUSTIC_MANIFEST = WS / "dist/round2/featextractnet_acoustic_author_data.manifest.json"
OUT = WS / "dist/round2/featextractnet_round2_standalone.tar.gz"
HISTORY = WS / "dist/round2/history"
DOI = "10.5281/zenodo.XXXXXXX"
THERMAL_URL = "https://web.stanford.edu/~kashefi/data/HeatTransferData.npy"
THERMAL_SHA = "e9bb20c51105607f2e20b6003fc387e8eaed5ad8906a8f11d8f83bc1cf535e26"
PIKAN_REPO = "https://github.com/Ali-Stanford/Physics_Informed_KAN_PointNet"
PIKAN_MIT = "https://raw.githubusercontent.com/Ali-Stanford/Physics_Informed_KAN_PointNet/main/LICENSE"
PIKAN_README = "https://raw.githubusercontent.com/Ali-Stanford/Physics_Informed_KAN_PointNet/main/README.md"
CCBY = "https://creativecommons.org/licenses/by/4.0/"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_extract(archive: Path, dest: Path) -> None:
    dest = dest.resolve()
    with tarfile.open(archive, "r:*") as tf:
        for member in tf.getmembers():
            p = Path(member.name.replace("\\", "/"))
            if p.is_absolute() or ".." in p.parts:
                raise SystemExit(f"unsafe member: {member.name}")
            target = (dest / p).resolve()
            if target != dest and dest not in target.parents:
                raise SystemExit(f"escape: {member.name}")
            if member.issym() or member.islnk() or member.isdev():
                raise SystemExit(f"link/device refused: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(member) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    if not BASE.is_file() or not ACOUSTIC.is_file():
        raise SystemExit("accepted base/acoustic archive missing")
    author = json.loads(ACOUSTIC_MANIFEST.read_text(encoding="utf-8"))
    if author.get("npz_files") != 21 or author.get("payload_hashes") != "passed":
        raise SystemExit("acoustic manifest does not certify 21 files")

    history = HISTORY / "featextractnet_round2_standalone_before_rebuild"
    if OUT.exists():
        history.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUT, history / (OUT.name + ".previous"))

    cache = WS / 'dist/round2/cache'
    cache.mkdir(parents=True, exist_ok=True)
    staging = cache / 'standalone_stage'
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        safe_extract(BASE, staging)
        # The accepted-base manifest is retained only as historical provenance;
        # STANDALONE_MANIFEST.json is the current manifest.
        base_manifest = staging / "PACKAGE_MANIFEST.json"
        if base_manifest.is_file():
            bm = json.loads(base_manifest.read_text(encoding="utf-8"))
            bm["historical_base_only"] = True
            bm["current_manifest"] = "STANDALONE_MANIFEST.json"
            bm["note"] = "Historical manifest of the accepted final_checked base. Its excluded raw-data/weights policy is superseded by STANDALONE_MANIFEST.json; do not use this file as the current archive policy."
            write(staging / "BASE_PACKAGE_MANIFEST.json", json.dumps(bm, indent=2, ensure_ascii=False) + "\n")
            base_manifest.unlink()
        # The public archive must contain only scientific/code/data material;
        # packaging/review audit reports are intentionally not published here.
        for rem in [
            "results/round2_remaining_closeout/formal/package_final_state.json",
            "results/round2_remaining_closeout/formal/registry_update.json",
            "results/round2_remaining_closeout/formal/source_review_whitelist_check.json",
            "results/round2_remaining_closeout/transolver_preflight/source_manifest.json",
        ]:
            fp = staging / rem
            if fp.is_file():
                fp.unlink()
        # Acoustic data and its provenance files.
        acoustic_stage = staging / "data/acoustic"
        acoustic_stage.mkdir(parents=True, exist_ok=True)
        with tarfile.open(ACOUSTIC, "r:*") as tf:
            for member in tf.getmembers():
                name = member.name.replace("\\", "/")
                if not member.isfile():
                    continue
                if "/data/acoustic/" in "/" + name:
                    rel = name.split("data/acoustic/", 1)[1]
                    target = acoustic_stage / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tf.extractfile(member) as src, open(target, "wb") as out:
                        shutil.copyfileobj(src, out)
        # keep the authoritative acoustic manifest/readme
        shutil.copy2(ACOUSTIC_MANIFEST, acoustic_stage / "ACOUSTIC_DATA_MANIFEST.json")
        # verify all 21 bytes
        nps = sorted(acoustic_stage.glob("*.npz"))
        if len(nps) != 21:
            raise SystemExit(f"expected 21 acoustic npz, got {len(nps)}")
        manifest_cases = {x["path"].split("/")[-1]: x for x in author["manifest"]["files"]}
        for p in nps:
            item = manifest_cases.get(p.name)
            if item is None or sha(p) != item["sha256"]:
                raise SystemExit(f"acoustic hash mismatch: {p.name}")

        # Current Table-1 weights: 50 thermal (10 models x 5 seeds) + 35
        # acoustic (7 current table models x 5 seeds) = 85.
        # The locally present 35 and server-recovered 50 are both copied to the
        # original relative checkpoint paths.
        recovered_manifest = WS / "results/round2_audit/standalone_recovered_weights.json"
        recovered = {x["path"]: x for x in json.loads(recovered_manifest.read_text(encoding="utf-8"))}
        recovered_root = WS / "dist/round2/cache/standalone_weights_extract"
        sources = []
        for seed in range(5):
            sources.append((f"results/round2_thermal_symmetric/feat_extract/seed{seed}/checkpoint_best.pth", "FeatExtractNet", seed, "local_main"))
            sources.append((f"results/round2_graph_five_seed_main/meshgraphnet/seed{seed}/checkpoint_best.pth", "MeshGraphNet", seed, "local_main"))
            for model, label in [("mlp", "CoordMLP"), ("pointnet", "PointNet"), ("pointnetpp", "PointNet++"), ("dgcnn", "DGCNN"), ("pipn", "PIPN-style")]:
                sources.append((f"results/thermal_protocolA/{model}/seed{seed}/checkpoint_best.pth", label, seed, "local_table1"))
        for rel, info in recovered.items():
            model = "Transolver" if "transolver" in rel else ("SIREN" if "/siren/" in rel else ("PointTransformer" if "pointtransformer" in rel else ("FNO" if "/fno/" in rel else ("UNO" if "/uno/" in rel else "acoustic"))))
            sources.append((rel, model, None, "server_recovered"))
        weight_records = []
        for rel, model, seed, source_kind in sources:
            src = WS / rel if source_kind.startswith("local") else recovered_root / rel
            if not src.is_file():
                raise SystemExit(f"required weight missing: {rel}")
            if source_kind == "server_recovered":
                expected = recovered[rel]["sha256"]
                if sha(src) != expected:
                    raise SystemExit(f"recovered weight hash mismatch: {rel}")
            dst = staging / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            weight_records.append({
                "path": rel, "model": model, "seed": seed,
                "bytes": dst.stat().st_size, "sha256": sha(dst),
                "source": source_kind, "tier": "direct_evaluation_current_table1",
            })
        if len(weight_records) != 85:
            raise SystemExit(f"expected 85 current Table-1 weights, got {len(weight_records)}")
        write(staging / "weights/WEIGHTS_MANIFEST.json", json.dumps({
            "note": "Complete current Table-1 direct-evaluation weights: 50 thermal (10 models x 5 seeds) + 35 acoustic (7 models x 5 seeds). No historical/old checkpoints are included.",
            "n_files": 85,
            "coverage": {"thermal_table1": 50, "acoustic_table1": 35},
            "weights": weight_records,
        }, indent=2, ensure_ascii=False) + "\n")
        write(staging / "weights/REMOTE_RECOVERY_MANIFEST.json", json.dumps({
            "note": "Recovered from the existing SSH server workspace cache; all 85 current Table-1 weights are now included; no new training.",
            "n_recovered": len(recovered), "recovered": list(recovered.values()),
        }, indent=2, ensure_ascii=False) + "\n")

        # Thermal data is included as a third-party input with attribution.
        thermal_dir = staging / "data/thermal"
        thermal_dir.mkdir(parents=True, exist_ok=True)
        thermal_cache = WS / "dist/round2/cache/thermal/HeatTransferData.npy"
        if not thermal_cache.is_file():
            raise SystemExit("thermal cache missing")
        th_sha = sha(thermal_cache)
        if th_sha != THERMAL_SHA:
            raise SystemExit(f"thermal cache SHA mismatch: {th_sha}")
        shutil.copy2(thermal_cache, thermal_dir / "HeatTransferData.npy")
        write(thermal_dir / "THERMAL_DATA_PROVENANCE.json", json.dumps({
            "path": "data/thermal/HeatTransferData.npy",
            "original_authors": ["Ali Kashefi", "Tapan Mukerji"],
            "source_paper": "Physics-informed PointNet: A deep learning solver for steady-state incompressible flows and thermal fields on multiple sets of irregular geometries",
            "paper_doi": "10.1016/j.jcp.2022.111510",
            "journal": "Journal of Computational Physics 468 (2022) 111510",
            "software_repository": PIKAN_REPO,
            "software_license": "MIT (software only)",
            "software_license_url": PIKAN_MIT,
            "author_readme_url": PIKAN_README,
            "download_url": THERMAL_URL,
            "bytes": thermal_cache.stat().st_size,
            "shape": [135, 5000, 6],
            "columns": ["x", "y", "u", "v", "p", "T"],
            "sha256": th_sha,
            "license_status": "Third-party data. No separate data-redistribution licence text was found at the source; the software MIT notice is not applied to the data and the old Zenodo LICENSE-DATA is not used as a rights source. Included with attribution in this archive; original rights remain with the original authors; no new MIT/CC-BY grant is asserted for this file.",
            "included": True,
        }, indent=2, ensure_ascii=False) + "\n")
        write(thermal_dir / "README.md", f"""# Thermal input (third-party, included with attribution)

`HeatTransferData.npy` is the original third-party thermal file, included
unchanged in this archive for reproducibility.

- Original authors: Ali Kashefi and Tapan Mukerji
- Paper: J. Comput. Phys. 468 (2022) 111510, https://doi.org/10.1016/j.jcp.2022.111510
- Software repository: {PIKAN_REPO}
- Software MIT notice: {PIKAN_MIT} (software only)
- Author README download line: {PIKAN_README}
- Original URL: {THERMAL_URL}
- Bytes `{thermal_cache.stat().st_size}`, shape `(135,5000,6)`, SHA-256 `{th_sha}`
- Columns: `x,y,u,v,p,T`

This is third-party data. No separate data-distribution licence text was found;
the MIT notice covers the software and is not applied to the file. The original
authors retain their rights. This archive includes the file with attribution
and does not relicense it as MIT or CC BY. The provenance JSON is
`THERMAL_DATA_PROVENANCE.json`.
""")

        # Layered licences / notices.
        lic = staging / "LICENSES"
        lic.mkdir(exist_ok=True)
        shutil.copy2(staging / "LICENSE", lic / "CODE-MIT.txt")
        transolver = staging / "src/models/baselines/transolver_official/LICENSE"
        if transolver.is_file():
            shutil.copy2(transolver, lic / "TRANSOLVER-MIT.txt")
        write(lic / "ACOUSTIC-DATA-CC-BY-4.0.md", f"""# Author-generated acoustic data

The twenty-one acoustic NPZ files in `data/acoustic/` were generated by the
FeatExtractNet authors with the included FDTD generator. To the extent the
authors hold rights in these generated arrays, they are offered under
Creative Commons Attribution 4.0 International (CC BY 4.0):
{CCBY}

Attribution: cite the FeatExtractNet manuscript and retain the source paper /
data provenance. This licence applies only to the author-generated acoustic
arrays, not to third-party thermal data.

The archive preserves the original NPZ bytes and their SHA-256 values; see
`data/acoustic/ACOUSTIC_DATA_MANIFEST.json`.
""")
        write(lic / "THIRD-PARTY-NOTICES.md", f"""# Third-party notices

## Thermal data (included with attribution)

- Original authors: Ali Kashefi and Tapan Mukerji
- Paper: https://doi.org/10.1016/j.jcp.2022.111510
- PI-KAN software: {PIKAN_REPO}
- Software licence (MIT, software only): {PIKAN_MIT}
- Author README: {PIKAN_README}
- Original download URL: {THERMAL_URL}
- Included file: `data/thermal/HeatTransferData.npy`
- SHA-256: `{THERMAL_SHA}`, bytes 32400128, shape `(135,5000,6)`

No separate data-distribution licence text was found at the source. The
software MIT notice is not applied to this data, and the old Zenodo
LICENSE-DATA text is not used as a rights source. The file is included with
attribution in this archive; original rights remain with the original authors;
no MIT or CC BY grant is asserted for this file.

## Transolver

Official Transolver source and its MIT notice are included under
`src/models/baselines/transolver_official/`; the original upstream is
https://github.com/thuml/Transolver. Its MIT notice is copied to
`LICENSES/TRANSOLVER-MIT.txt`.

## Dependencies

The package records the inherited dependency specifications (`requirements.txt`,
`requirements-extended.txt`, `environment.yml`). Dependencies are not vendored;
their own upstream licences apply. PyTorch is BSD-style; NumPy/SciPy/
Matplotlib retain their respective upstream licences. No global installation is
performed by this package.
""")
        write(staging / "LICENSES/README.md", """# Licence layers

- `CODE-MIT.txt`: FeatExtractNet code, MIT.
- `TRANSOLVER-MIT.txt`: official Transolver, MIT.
- `ACOUSTIC-DATA-CC-BY-4.0.md`: author-generated acoustic arrays only.
- `THIRD-PARTY-NOTICES.md`: third-party thermal data attribution and dependency notices.

These layers are intentionally separate. Thermal data is included with
attribution, is not relicensed, and is not covered by the software MIT or
author-acoustic CC BY notices.
""")

        # Single DOI and metadata.  Only one placeholder is used.
        zenodo_path = staging / ".zenodo.json"
        if zenodo_path.is_file():
            zd = json.loads(zenodo_path.read_text(encoding="utf-8"))
        else:
            zd = {}
        zd["doi"] = DOI
        zd["version"] = "round2-standalone"
        zd["license"] = "mit"
        zd["notes"] = (
            "Single new revision-code DOI placeholder; old DOI is historical provenance only, not an input dependency. "
            "Software is MIT (code and Transolver). Author-generated acoustic arrays carry a separate CC BY 4.0 notice. The included third-party thermal HeatTransferData.npy retains original rights and has no separate data licence asserted; it is not relicensed as MIT or CC BY. See LICENSES/THIRD-PARTY-NOTICES.md."
        )
        zd["description"] = (
            "Standalone code, metric records, 21 author-generated acoustic arrays, the attributed third-party thermal file, and all 85 current Table-1 weights "
            "for the FeatExtractNet round-two revision. Thermal HeatTransferData.npy is included with attribution; it retains original rights and is not relicensed. "
            "see DATA_SOURCES.md and LICENSES/THIRD-PARTY-NOTICES.md."
        )
        write(zenodo_path, json.dumps(zd, indent=2, ensure_ascii=False) + "\n")
        cff_path = staging / "CITATION.cff"
        if cff_path.is_file():
            text = cff_path.read_text(encoding="utf-8")
            text = text.replace("No new DOI is assigned.", f"Single new DOI placeholder: {DOI}.")
            text = text.replace("version: round2-preparation", "version: round2-standalone")
            text = text.replace("doi: \"10.1016/j.jcp.2022.111510\"", "doi: \"10.1016/j.jcp.2022.111510\"\ndoi: \"" + DOI + "\"") if "doi: \"" + DOI + "\"" not in text else text
            # Avoid duplicate doi keys if the base already has a package DOI.
            if "doi: \"" + DOI + "\"" not in text:
                text += f"\ndoi: \"{DOI}\"\n"
            write(cff_path, text)
        # overwrite the base README with a standalone-specific one
        write(staging / "README.md", f"""# FeatExtractNet round-two standalone package

**Single new revision-code DOI placeholder:** `{DOI}`

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
  original URL `{THERMAL_URL}`; SHA-256 `{THERMAL_SHA}`; shape
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
""")
        write(staging / "DATA_SOURCES.md", f"""# Data sources and provenance (standalone)

## Thermal HeatTransferData.npy — INCLUDED WITH ATTRIBUTION

- Source paper: Ali Kashefi and Tapan Mukerji, *Physics-informed PointNet*,
  Journal of Computational Physics 468 (2022) 111510,
  https://doi.org/10.1016/j.jcp.2022.111510
- Software repository: {PIKAN_REPO}
- Software MIT licence: {PIKAN_MIT} (software only)
- Author README: {PIKAN_README}
- Original file URL: {THERMAL_URL}
- Included path: `data/thermal/HeatTransferData.npy`
- Bytes `32400128`; shape `(135,5000,6)`; columns x,y,u,v,p,T;
  SHA-256 `{THERMAL_SHA}`.

No separate data-distribution licence text was found. The MIT notice covers
software, not this data; the old Zenodo `LICENSE-DATA` is not used as a rights
source. The file is included with attribution and original rights remain with
the original authors; no MIT/CC BY grant is asserted for this third-party file.

## Acoustic 21 cases — INCLUDED

The 21 NPZ files are author-generated by the included FDTD code. Original
bytes are retained and SHA-256 checked in
`data/acoustic/ACOUSTIC_DATA_MANIFEST.json`. Case roles and six train / one
validation / ten test / four unused identities are in
`configs/acoustic_cases.json`. The target is the real final-time `p_total`
snapshot. No phase correction is applied. Author-generated arrays are offered
under CC BY 4.0 as a separate layer; this does not relicense thermal data.

## Old DOI

`https://doi.org/10.5281/zenodo.22876119` is mentioned only as historical
provenance. It is not used as an input, download dependency or package member.
""")
        write(staging / "REPRODUCTION_TIERS.md", """# Reproduction tiers

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
""")
        # Include this builder for reproducibility.
        (staging / "scripts").mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__), staging / "scripts/package_standalone.py")
        # Freeze payload/contents first, then compute manifest. Contents excludes
        # itself and manifest; manifest includes all other files and only itself
        # is excluded.
        contents = staging / "PACKAGE_CONTENTS.txt"
        if contents.exists():
            contents.unlink()
        payload = [p for p in sorted(staging.rglob("*")) if p.is_file() and p != staging / "STANDALONE_MANIFEST.json"]
        write(contents, "\n".join(p.relative_to(staging).as_posix() for p in payload) + "\n")
        files = []
        for p in sorted(staging.rglob("*")):
            if p.is_file() and p != staging / "STANDALONE_MANIFEST.json":
                files.append({"path": p.relative_to(staging).as_posix(),
                              "bytes": p.stat().st_size, "sha256": sha(p)})
        write(staging / "STANDALONE_MANIFEST.json", json.dumps({
            "task": "R2-SINGLE-DOI-01",
            "single_new_doi_placeholder": DOI,
            "current_manifest": "STANDALONE_MANIFEST.json",
            "base_accepted_tar": "dist/round2/featextractnet_round2_final_checked.tar.gz",
            "base_accepted_sha256": sha(BASE),
            "base_manifest_historical": "BASE_PACKAGE_MANIFEST.json",
            "acoustic_author_archive_sha256": sha(ACOUSTIC),
            "acoustic_npz_count": 21,
            "thermal_data_included": True,
            "thermal_data_sha256": THERMAL_SHA,
            "thermal_original_url": THERMAL_URL,
            "license_layers": ["CODE-MIT", "TRANSOLVER-MIT", "ACOUSTIC-DATA-CC-BY-4.0", "THIRD-PARTY-NOTICES"],
            "weights_included": len(weight_records),
            "tiers": ["table_reconstruction", "complete_current_table1_direct_evaluation", "retraining_protocols"],
            "files": files,
        }, indent=2, ensure_ascii=False) + "\n")

        OUT.parent.mkdir(parents=True, exist_ok=True)
        tmp = OUT.with_suffix(".tmp")
        with tarfile.open(tmp, "w:gz", compresslevel=6, format=tarfile.PAX_FORMAT) as tf:
            for p in sorted(staging.rglob("*")):
                if not p.is_file():
                    continue
                info = tf.gettarinfo(str(p), arcname=p.relative_to(staging).as_posix())
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                with open(p, "rb") as fh:
                    tf.addfile(info, fh)
        os.replace(tmp, OUT)
        archive_sha = sha(OUT)
        write(WS / "dist/round2/featextractnet_round2_standalone.manifest.json", json.dumps({
            "task": "R2-SINGLE-DOI-01",
            "archive": OUT.name,
            "sha256": archive_sha,
            "bytes": OUT.stat().st_size,
            "members": len(tarfile.open(OUT).getnames()),
            "single_new_doi_placeholder": DOI,
            "base_tar_sha256": sha(BASE),
            "acoustic_archive_sha256": sha(ACOUSTIC),
            "acoustic_npz_count": 21,
            "thermal_data_included": True,
            "thermal_data_sha256": THERMAL_SHA,
            "thermal_data_license_status": "third_party_attributed_not_relicensed",
            "weights_included": len(weight_records),
            "licenses": ["CODE-MIT", "TRANSOLVER-MIT", "ACOUSTIC-DATA-CC-BY-4.0", "THIRD-PARTY-NOTICES"],
            "standalone_manifest": "STANDALONE_MANIFEST.json",
        }, indent=2, ensure_ascii=False) + "\n")
        write(WS / "dist/round2/featextractnet_round2_standalone.tar.gz.sha256",
              archive_sha + "  " + OUT.name + "\n")
        print(f"[standalone] {OUT.name}: {OUT.stat().st_size} bytes, members {len(tarfile.open(OUT).getnames())}, sha256 {archive_sha}")
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
