"""SIREN baseline: sinusoidal representation network for coordinate-to-field regression.

Reference
---------
Sitzmann, Martel, Bergman, Lindell, Wetzstein, "Implicit Neural Representations
with Periodic Activation Functions", NeurIPS 2020. No official package is
distributed, so the implementation follows the paper directly (the paper's
architecture is a plain MLP whose activation is ``sin`` and whose initialisation
is scaled by ``omega_0``).

Why it is a relevant baseline here
----------------------------------
B R1 asks for head-to-head positioning against "recent strong surrogates
(... SIREN)". SIREN is the canonical coordinate-MLP variant with periodic
activations, so it is the natural counterpart of the CoordMLP / Fourier-MLP rows
already in the tables: same input (raw coordinates), same output, different
activation and initialisation. That makes it a controlled comparison of the
activation/initialisation choice rather than of capacity or architecture family.

Faithfulness notes (what is taken from the paper, verbatim in effect)
-------------------------------------------------------------------
* Activation ``sin(omega_0 * h)``; ``omega_0 = 30`` on the **first** layer,
  ``omega_0 = 1`` on the hidden layers (paper §3, "we choose omega_0 = 30").
* First-layer weights ``~ U(-sqrt(6/fan_in)/omega_0, +sqrt(6/fan_in)/omega_0)``;
  hidden weights ``~ U(-sqrt(6/fan_in)/omega, +sqrt(6/fan_in)/omega)`` with
  ``omega = omega_0`` for the first layer and ``1`` afterwards (this keeps the
  pre-activations of the first sine layer close to unit variance).
* Biases of the first layer ``~ U(-1/sqrt(fan_in), +1/sqrt(fan_in))``, hidden
  biases zero.
* The output layer is a plain linear map (no activation), matching the paper's
  final layer.

Operates on ``[B, C_in, N]`` (the layout used throughout this repo) and returns
``[B, C_out, N]``.
"""

import numpy as np
import torch
import torch.nn as nn


class SIRENNet(nn.Module):
    """SIREN MLP: ``(x, y) -> (u, v, p, T)`` (or 1 channel for acoustics).

    Args:
        input_channels: coordinate channels (2 for 2-D).
        output_channels: field channels (4 thermal, 1 acoustic).
        hidden_dim: width of the hidden layers.
        num_layers: total number of linear layers (>= 2).
        omega_0: first-layer frequency scale (paper default 30; the hidden
            layers use 1 regardless).
    """

    def __init__(self, input_channels=2, output_channels=4, hidden_dim=256,
                 num_layers=6, omega_0=30.0):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.omega_0 = float(omega_0)

        dims = [self.input_channels] + [self.hidden_dim] * (num_layers - 1) \
            + [self.output_channels]
        self.linears = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self._siren_init()

    def _siren_init(self):
        """Paper-faithful initialisation (see module docstring)."""
        with torch.no_grad():
            for i, lin in enumerate(self.linears):
                fan_in = lin.in_features
                if i == 0:
                    bound = np.sqrt(6.0 / fan_in) / self.omega_0
                    lin.weight.uniform_(-bound, bound)
                    b = 1.0 / np.sqrt(fan_in)
                    lin.bias.uniform_(-b, b)
                elif i == len(self.linears) - 1:
                    # output layer: paper uses the standard uniform bound
                    bound = np.sqrt(6.0 / fan_in)
                    lin.weight.uniform_(-bound, bound)
                    lin.bias.zero_()
                else:
                    bound = np.sqrt(6.0 / fan_in) / 1.0
                    lin.weight.uniform_(-bound, bound)
                    lin.bias.zero_()

    def forward(self, x):
        """[B, C_in, N] -> [B, C_out, N]."""
        h = x
        for i, lin in enumerate(self.linears):
            h = torch.einsum('bcn,dc->bdn', h, lin.weight)
            h = h + lin.bias.view(1, -1, 1)
            if i == 0:
                h = torch.sin(self.omega_0 * h)
            elif i < len(self.linears) - 1:
                h = torch.sin(h)
        return h
