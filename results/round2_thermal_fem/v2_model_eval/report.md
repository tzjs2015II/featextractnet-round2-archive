# R2-THERMAL-FEM-02 model evaluation on FEM support

Reference: independent FEM h=0.04, transferred to fixed 5000-point support; not label interpolation.

## Summary (5 seeds)

### FEN
- geom 46: u RE=0.1104±0.0016, v RE=0.0844±0.0007, p_raw RE=0.0600±0.0043, p_centered RE=0.0591±0.0047, T RE=0.0412±0.0026
- geom 17: u RE=0.0920±0.0027, v RE=0.0681±0.0040, p_raw RE=0.0477±0.0038, p_centered RE=0.0472±0.0036, T RE=0.0344±0.0036
- geom 96: u RE=0.0863±0.0051, v RE=0.0674±0.0054, p_raw RE=0.0525±0.0041, p_centered RE=0.0521±0.0040, T RE=0.0379±0.0034

### graph
- geom 46: u RE=0.0764±0.0044, v RE=0.0602±0.0058, p_raw RE=0.0643±0.0036, p_centered RE=0.0641±0.0037, T RE=0.0384±0.0031
- geom 17: u RE=0.0645±0.0088, v RE=0.0394±0.0055, p_raw RE=0.0385±0.0034, p_centered RE=0.0378±0.0031, T RE=0.0230±0.0028
- geom 96: u RE=0.0639±0.0037, v RE=0.0423±0.0041, p_raw RE=0.0493±0.0034, p_centered RE=0.0488±0.0033, T RE=0.0292±0.0042

## Limitations

- FEM transfer interpolation is not the historical solver; it does not validate historical labels.
- Only three representative geometries have independent FEM support.
- No retraining/model selection; pressure raw and centered metrics use different denominators.
