# Independent thermal FEM environment (reference-checked package)

The FEM results were produced in an isolated conda prefix inside the author workspace:
`tools/envs/pinnap-fem`, created from conda-forge with the package cache
`tools/conda-pkgs-fem`. That prefix and cache are intentionally excluded from this
package (machine-specific, large, and may contain local paths).

Observed versions during the accepted runs:
- Python 3.12.15
- dolfinx 0.8.0; UFL 2024.1.0; Basix 0.8.0
- PETSc 3.21.6 / petsc4py 3.21.6; mpi4py 4.1.2
- gmsh 4.15.0; NumPy 2.5.4; SciPy 1.18.1

The explicit conda package URLs are in `environment/fem_conda_explicit_urls.txt`
(URLs only; no local prefix or installation log). Reproduce with conda-forge and
this exact lock/recipe metadata, or use an equivalent FEniCSx 0.8 environment.
The FEM solvers do not require GPU. Model prediction evaluation uses a separate
existing PyTorch environment (see `requirements.txt` / `requirements-extended.txt`);
do not install the FEniCSx stack into that environment.
