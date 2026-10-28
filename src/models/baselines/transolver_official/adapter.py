"""Thermal four-output adapter for the official Transolver irregular-mesh model.

The adapter only reshapes coordinates into/out of the official model.  It does
not accept truth fields, geometry IDs, or auxiliary features as input.
Input:  x [B, N, 2] coordinates (normalized by the production thermal_io).
Output: [B, 4, N] physical-scale predictions (u, v, p, T).
"""

import torch
import torch.nn as nn

from .Transolver_Irregular_Mesh import Model as OfficialTransolver


class TransolverAdapter(nn.Module):
    """Official Transolver with the frozen benchmark configuration."""

    OFFICIAL_COMMIT = '75e0f67643806a81cd1d3f6adc88dd8c02416fe7'
    OFFICIAL_CONFIG = {
        'space_dim': 2,
        'fun_dim': 0,
        'out_dim': 4,
        'n_layers': 5,
        'n_hidden': 256,
        'n_head': 8,
        'slice_num': 32,
        'mlp_ratio': 1,
        'dropout': 0.0,
        'act': 'gelu',
        'Time_Input': False,
        'unified_pos': False,
        'ref': 8,
    }

    def __init__(self):
        super().__init__()
        self.net = OfficialTransolver(**self.OFFICIAL_CONFIG)

    def forward(self, x):
        # The production thermal runner uses [B,2,N]; the direct preflight used
        # [B,N,2].  Accept both without changing the official model.
        if x.ndim != 3:
            raise ValueError(f'Transolver expects a 3-D tensor, got {tuple(x.shape)}')
        if x.shape[1] == 2 and x.shape[2] != 2:
            x = x.transpose(1, 2).contiguous()  # B,2,N -> B,N,2
        if x.shape[-1] != 2:
            raise ValueError(f'Transolver expects 2 coordinate channels, got {tuple(x.shape)}')
        # fx=None and T=None: no truth/fun features/time input.
        out = self.net(x, fx=None, T=None)  # B,N,4
        return out.transpose(1, 2).contiguous()  # B,4,N

    def model_label(self):
        return ('Transolver official irregular-mesh '
                f'commit={self.OFFICIAL_COMMIT}, '
                'layers=5 hidden=256 heads=8 slices=32 mlp_ratio=1 '
                'dropout=0 gelu, Time_Input=False unified_pos=False')
