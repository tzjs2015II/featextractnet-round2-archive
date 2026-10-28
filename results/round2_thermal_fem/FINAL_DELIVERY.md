# R2-THERMAL-FEM-02 — current entry

**Current authoritative closing report: `v3_postprocess/FINAL_DELIVERY_V3.md`.** The numerical v2 delivery is `v2/FINAL_DELIVERY_V2.md`; its acceptance wording is superseded by the v3 post-process.

- **v1 invalid / superseded**: old top-level v1 results and earlier pilot/levels/representatives/model_eval used the wrong convection operator and interpolation-based reference. Full v1 docs are archived in `archive_v1/`; do not use them as current results.
- **v2**: corrected convection and actual dolfinx `Function.eval`; 9 valid solves retained (no rerun in v3). Pre-v3 acceptance statements are archived in `archive_v2_prepost/`.
- **v3**: corrected acceptance to true assembled Galerkin free residual, added quantitative eps evidence, and fixed this entry point.

Codex unified final review remains pending; DSH delivery is not a scientific acceptance.
