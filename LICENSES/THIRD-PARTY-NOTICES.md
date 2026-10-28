# Third-party notices

## Thermal data (included with attribution)

- Original authors: Ali Kashefi and Tapan Mukerji
- Paper: https://doi.org/10.1016/j.jcp.2022.111510
- PI-KAN software: https://github.com/Ali-Stanford/Physics_Informed_KAN_PointNet
- Software licence (MIT, software only): https://raw.githubusercontent.com/Ali-Stanford/Physics_Informed_KAN_PointNet/main/LICENSE
- Author README: https://raw.githubusercontent.com/Ali-Stanford/Physics_Informed_KAN_PointNet/main/README.md
- Original download URL: https://web.stanford.edu/~kashefi/data/HeatTransferData.npy
- Included file: `data/thermal/HeatTransferData.npy`
- SHA-256: `e9bb20c51105607f2e20b6003fc387e8eaed5ad8906a8f11d8f83bc1cf535e26`, bytes 32400128, shape `(135,5000,6)`

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
