"""Official Transolver irregular-mesh components, isolated for benchmark use.

Upstream: https://github.com/thuml/Transolver
Commit: 75e0f67643806a81cd1d3f6adc88dd8c02416fe7 (2026-02-26)
License: MIT (see LICENSE in this directory).

This package keeps the official Physics-Attention irregular-mesh block,
placeholder token, LayerNorm/FFN structure and preprocess MLP. Only import
paths and the following compatibility edits are made (documented in
SOURCE.md):
  * timm trunc_normal_ -> torch.nn.init.trunc_normal_
  * einops rearrange -> equivalent torch permute/reshape
No architecture simplification is applied.
"""
