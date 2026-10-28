"""Independent thermal FEM reference package (R2-THERMAL-REFERENCE-02).

Modules:
    geometry       independent geometry reconstruction + polar probe mesh
    fenics_solver  FEniCSx Taylor-Hood Boussinesq solver (lazy imports)

The laboratory environment had no dolfinx/gmsh installation when this review
implementation was prepared; see ``fenics_solver.dependency_status()`` for the
exact dependency state and the isolated-environment plan.
"""

__all__ = ["geometry", "fenics_solver"]
