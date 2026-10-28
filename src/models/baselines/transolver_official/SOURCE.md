# Transolver official source and compatibility record

- Upstream: https://github.com/thuml/Transolver
- Pinned commit: `75e0f67643806a81cd1d3f6adc88dd8c02416fe7`
- Source branch: `main` (commit date 2026-02-26)
- Upstream license: MIT (Copyright 2024 THUML @ Tsinghua University), copied verbatim in `LICENSE`.
- Copied files:
  - `PDE-Solving-StandardBenchmark/model/Transolver_Irregular_Mesh.py` -> `Transolver_Irregular_Mesh.py`
  - `PDE-Solving-StandardBenchmark/model/Physics_Attention.py` -> `Physics_Attention.py`
  - `PDE-Solving-StandardBenchmark/model/Embedding.py` -> `Embedding.py`
- Added: `adapter.py`, `__init__.py`.

## Exact configuration

`Adapter.OFFICIAL_CONFIG`:

```python
space_dim=2, fun_dim=0, out_dim=4,
n_layers=5, n_hidden=256, n_head=8, slice_num=32,
mlp_ratio=1, dropout=0.0, act='gelu',
Time_Input=False, unified_pos=False, ref=8
```

Only coordinates `[B,N,2]` enter the adapter. Truth fields, geometry IDs and
auxiliary features are not inputs. `forward` returns `[B,4,N]` after the
official `[B,N,4]` output is transposed.

## Compatibility changes (complete list)

1. Import paths changed from upstream `model.*` to this local package
   (`.Embedding`, `.Physics_Attention`).
2. `from timm.models.layers import trunc_normal_` replaced by
   `from torch.nn.init import trunc_normal_`. This is the same `functools.partial`
   initialization used by timm; no distribution or initialization scale change.
3. `einops` is not installed in the isolated environment. The only official
   irregular-mesh use, `rearrange(out_x, 'b h n d -> b n (h d)')`, is replaced
   by an equivalent `permute(0,2,1,3).reshape(B,N,H*D)` helper. `Embedding.py`
   likewise replaces the unused rotary `rearrange` with an equivalent reshape.
   No attention/projection/normalization/placeholder/FFN operation is changed.
4. `timm` and `einops` are not added as dependencies.

There is no architecture simplification: all five `Transolver_block`s retain
Physics-Attention (slice, slice attention, deslice), LayerNorm and FFN; the
placeholder parameter and preprocess MLP are retained; last-layer output is
the official linear head with `out_dim=4`.
