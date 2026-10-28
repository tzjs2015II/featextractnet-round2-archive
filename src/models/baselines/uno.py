"""UNO baseline (U-Net Neural Operator) for the regular-grid acoustic task.

Reference
---------
Rahman, Ross, Azizzadenesheli, "U-NO: U-shaped Neural Operators", TMLR 2023.
Implemented through the **official** ``neuraloperator`` package (import name
``neuralop``), not reimplemented here, so the operator follows the authors' code
(B R1/R2 both flag the risk of self-implemented baselines).

Why UNO is a relevant baseline
-----------------------------
B R1 asks for positioning against "FNO/Geo-FNO/GNO/UNO". UNO generalises FNO by
combining multi-resolution U-Net down/up-sampling with spectral convolutions, so
it is the natural "stronger spectral operator" row next to the existing FNO row on
the acoustic grid task.

Interface
---------
Mirrors ``src/models/baselines/fno.py``: ``forward_grid([B, Cin, H, W]) -> [B, Cout, H, W]``
with ``Cin >= 2`` (channel 0/1 = normalised grid coordinates, further channels =
broadcast conditioning such as f and Ma). The acoustic data pipeline already
assembles exactly this tensor for the FNO row, so UNO reuses it unchanged.

Grid-size note
--------------
The acoustic crop is 481x481 (odd). UNO's horizontal skip connections and its
U-shaped scalings were verified to run on that size in the smoke test; the
scalings used here are the package example's pattern scaled to this resolution.
"""

import torch
import torch.nn as nn

try:  # the official package; imported under a guard so a failed install is
    # reported as unavailable rather than crashing every other experiment
    from neuralop.models import UNO as _UNO

    _NEURALOP_AVAILABLE = True
    _IMPORT_ERROR = None
except Exception as _exc:  # noqa: BLE001
    _UNO = None
    _NEURALOP_AVAILABLE = False
    _IMPORT_ERROR = repr(_exc)


class UNO2d(nn.Module):
    """U-Net Neural Operator on a regular grid (official ``neuralop`` UNO).

    Args:
        in_channels: grid input channels (2 coordinates + conditioning).
        out_channels: output field channels (1 for ``p_total``).
        hidden_channels: width of the lifting/projection MLPs.
        n_layers: number of operator blocks.
        uno_out_channels: per-level widths of the U-shape.
        uno_n_modes: per-level Fourier modes.
        uno_scalings: per-level down/up-sampling factors.
    """

    def __init__(self, in_channels=4, out_channels=1, hidden_channels=32,
                 n_layers=5,
                 uno_out_channels=(32, 64, 64, 64, 32),
                 uno_n_modes=None,
                 uno_scalings=((1.0, 1.0), (0.5, 0.5), (1, 1), (1, 1), (2, 2)),
                 horizontal_skips_map=None,
                 domain_padding=0.2):
        super().__init__()
        if not _NEURALOP_AVAILABLE:
            raise RuntimeError(
                "neuraloperator is not importable in this environment: "
                f"{_IMPORT_ERROR}. Install it (see requirements_ext.txt) or "
                "report this row as unavailable -- do NOT substitute a "
                "hand-written UNO under this name.")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        # This configuration is derived from the package's own test
        # (neuralop/models/tests/test_uno.py) rather than invented here:
        #   * symmetric U-shape [32, 64, 64, 64, 32] with an explicit
        #     horizontal_skips_map {4:0, 3:1}, which is what makes the paired
        #     encoder/decoder widths match -- a mismatched pairing raises
        #     "must be the same for soft-gating";
        #   * channel_mlp_skip="linear" (the package default 'soft-gating'
        #     additionally requires *every* level to share one width);
        #   * domain_padding=0.2, which the test relies on for non-power-of-two
        #     and odd spatial sizes -- the acoustic crop is 481x481 (odd).
        # Only channels and mode counts are adapted to this task.
        if uno_n_modes is None:
            uno_n_modes = ((10, 10),) * int(n_layers)
        if horizontal_skips_map is None:
            horizontal_skips_map = {4: 0, 3: 1} if int(n_layers) == 5 else None
        if not (len(uno_out_channels) == len(uno_n_modes) == len(uno_scalings)
                == int(n_layers)):
            raise ValueError("uno_out_channels / uno_n_modes / uno_scalings must "
                             "each have n_layers entries")
        self.net = _UNO(in_channels=int(in_channels),
                        out_channels=int(out_channels),
                        hidden_channels=int(hidden_channels),
                        n_layers=int(n_layers),
                        uno_out_channels=list(uno_out_channels),
                        uno_n_modes=[list(m) for m in uno_n_modes],
                        uno_scalings=[list(s) for s in uno_scalings],
                        horizontal_skips_map=horizontal_skips_map,
                        domain_padding=domain_padding,
                        channel_mlp_skip="linear")

    def forward_grid(self, grid):
        """[B, Cin, H, W] -> [B, Cout, H, W]."""
        out = self.net(grid)
        # the official UNO returns [B, Cout, H, W]; guard against a [B, H, W, Cout]
        # layout so a future version change surfaces as an explicit error
        if out.dim() != 4 or out.shape[1] != self.out_channels:
            raise RuntimeError(
                f"unexpected UNO output shape {tuple(out.shape)}; expected "
                f"[B, {self.out_channels}, H, W]")
        return out


def availability():
    """Report whether the official UNO can be constructed here (for the audit)."""
    return {"neuraloperator_available": _NEURALOP_AVAILABLE,
            "import_error": _IMPORT_ERROR,
            "package": "neuraloperator (import neuralop)",
            "class": "neuralop.models.UNO"}
