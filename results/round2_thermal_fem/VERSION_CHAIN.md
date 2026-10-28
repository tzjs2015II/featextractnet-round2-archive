# R2-THERMAL-FEM-02 version chain

- **v1 (invalid)**: wrong UFL convection (`dot(u, grad(u))` = 0.5 grad(|u|^2)) and global interpolation reference; old top-level documents/JSON/fields are failure history only. Archived: `archive_v1/`, old job IDs 50986/50987/50996/51000/51001.
- **v2 (numeric valid, acceptance wording corrected)**: explicit component convection; actual dolfinx Function.eval reference; 9 valid solves (51016/51020) and model eval (51021). Pre-v3 acceptance docs archived in `archive_v2_prepost/`.
- **v3 (current closure)**: post-process re-acceptance with true assembled Galerkin free residual (`v3_postprocess/`), quantitative eps evidence (`v3_eps/`, job 51027), corrected top-level entry, final probe job 51029.
- Codex unified final review remains pending; no DSH scientific acceptance claim.
