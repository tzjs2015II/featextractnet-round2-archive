"""Acoustic FDTD data-generation package (Python port of MATLAB codematlab).

Modules
-------
python_fdtd : 2-D acoustic FDTD solver (RK4 + PML + Lamb-Oseen vortex
              + incident-plane-wave source), a port of the MATLAB
              *discretisation* in the read-only dir ``codematlab/``: same PML
              coefficients, 4-stage RK4, 2nd-order central differences and hard
              line source. It is **not** a pointwise-validated reproduction (see
              the module docstring for the list of deliberate deviations, and
              docs/审计_协议合规核对.md P1-4).
"""

from .python_fdtd import FDTDAcoustic2D, run_case

__all__ = ["FDTDAcoustic2D", "run_case"]
