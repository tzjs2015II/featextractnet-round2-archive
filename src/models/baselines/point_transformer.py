"""Point Transformer baseline for the thermal point-cloud task.

Reference
---------
Zhao, Jiang, Jia, Torr, Koltun, "Point Transformer", ICCV 2021. Built on the
**official** ``torch_geometric.nn.PointTransformerConv`` operator (package
``torch_geometric`` 2.8.0.post1), not on a hand-written attention layer, so the
vector-attention/positional-encoding details follow the library implementation
(B R1/R2 both flag self-implemented baselines as a fairness risk).

Why it is a relevant baseline
-----------------------------
B R1 asks to position against "DGCNN/PointTransformer": these are the
neighbourhood-aware point models. ``results/thermal_protocolA`` already contains a
DGCNN row; PointTransformer adds the *attention*-based neighbourhood model, so the
table then covers both neighbourhood aggregation styles.

Interface
---------
``forward(x: [B, 2, N]) -> [B, 4, N]``, i.e. the same layout as the other
point-cloud baselines (PointNet / PointNet++ / DGCNN / FeatExtractNet), so it
drops into the existing Protocol-A pipeline and evaluation unchanged.

Graph construction
------------------
A k-nearest-neighbour graph in raw (train-normalised) coordinate space, built with
``torch.cdist`` + ``topk`` per sample. ``torch_cluster`` is not installed here, so
the neighbour search is done in plain PyTorch; the per-sample loop keeps the peak
distance-matrix memory at one N x N block instead of B x N x N.
"""

import torch
import torch.nn as nn

try:
    from torch_geometric.nn import PointTransformerConv
    _PYG_AVAILABLE = True
    _IMPORT_ERROR = None
except Exception as _exc:  # noqa: BLE001
    PointTransformerConv = None
    _PYG_AVAILABLE = False
    _IMPORT_ERROR = repr(_exc)


class PointTransformerBaseline(nn.Module):
    """Stack of ``PointTransformerConv`` blocks with residual + LayerNorm.

    Args:
        input_channels: input feature channels (the coordinate channels).
        output_channels: field channels (4 for thermal).
        hidden_dim: embedding width (default 320).
        num_layers: number of attention blocks (default 3).
        k: neighbours per node in the kNN graph.

    Capacity: the defaults ``hidden_dim=320, num_layers=3`` give **928,644**
    parameters, the closest configuration found to the ~1.05M rows already in the
    table (``CoordMLP(512,6)`` / ``PointNet`` = 1,054,212; ``h=320, L=4`` overshoots
    to 1,237,444 and ``h=384, L=2`` undershoots to 891,268). It is still 11.9 %
    smaller, so any conclusion drawn from this row must carry that capacity
    qualification (design doc §2).

    Note on attention heads: ``torch_geometric.nn.PointTransformerConv`` runs a
    single attention head by default (its ``attn_nn`` is a plain ``Linear`` to
    ``out_channels``); it does **not** accept a ``heads`` argument (passing one
    raises ``TypeError`` from ``MessagePassing.__init__``). Multi-head attention
    would require supplying a custom ``attn_nn``, which would deviate from the
    library's operator -- so the defaults are used and the row is described as
    the library's PointTransformerConv block, without claiming multi-head.
    """

    def __init__(self, input_channels=2, output_channels=4, hidden_dim=320,
                 num_layers=3, k=16):
        super().__init__()
        if not _PYG_AVAILABLE:
            raise RuntimeError(
                "torch_geometric is not importable in this environment: "
                f"{_IMPORT_ERROR}. Install it (see requirements_ext.txt) or "
                "report this row as unavailable -- do NOT substitute a "
                "hand-written Point Transformer under this name.")
        self.k = int(k)
        self.pos_dim = 2          # the data is 2-D; see the pos_nn note below
        self.input_proj = nn.Linear(int(input_channels), int(hidden_dim))
        # torch_geometric's PointTransformerConv defaults its positional encoder
        # to Linear(3, out_channels), i.e. it assumes 3-D coordinates; on 2-D
        # data that raises
        #   RuntimeError: mat1 and mat2 shapes cannot be multiplied (17000x2 and 3x128)
        # so the encoder is created explicitly for the actual coordinate
        # dimensionality instead of silently padding the coordinates to 3-D.
        self.convs = nn.ModuleList([
            PointTransformerConv(int(hidden_dim), int(hidden_dim),
                                 pos_nn=nn.Linear(self.pos_dim, int(hidden_dim)))
            for _ in range(int(num_layers))])
        self.norms = nn.ModuleList([
            nn.LayerNorm(int(hidden_dim)) for _ in range(int(num_layers))])
        self.output_proj = nn.Linear(int(hidden_dim), int(output_channels))

    def _knn_edges(self, pos):
        """kNN graph per sample; returns a single edge_index over the flattened
        batch. ``pos``: [B, N, C_pos]."""
        b, n, _ = pos.shape
        src, dst = [], []
        for i in range(b):
            pi = pos[i]
            d = torch.cdist(pi, pi)                       # [N, N]
            d.fill_diagonal_(float("inf"))                # exclude self
            idx = d.topk(self.k, dim=-1, largest=False).indices   # [N, k]
            s = torch.arange(n, device=pos.device).view(n, 1).expand(n, self.k)
            src.append(s.reshape(-1) + i * n)
            dst.append(idx.reshape(-1) + i * n)
        return torch.stack([torch.cat(src), torch.cat(dst)], dim=0)

    def forward(self, x):
        """[B, C_in, N] -> [B, C_out, N]."""
        b, _, n = x.shape
        h = x.permute(0, 2, 1)                            # [B, N, C_in]
        pos = h[..., :2]                                  # positions = coordinates
        h = self.input_proj(h)
        edge_index = self._knn_edges(pos)
        hf = h.reshape(b * n, -1)
        pf = pos.reshape(b * n, pos.shape[-1])
        for conv, norm in zip(self.convs, self.norms):
            hf = norm(hf + conv(hf, pf, edge_index))
        out = self.output_proj(hf).reshape(b, n, -1)       # [B, N, C_out]
        return out.permute(0, 2, 1)


def availability():
    """Report whether the official operator is importable (for the audit)."""
    return {"torch_geometric_available": _PYG_AVAILABLE,
            "import_error": _IMPORT_ERROR,
            "package": "torch_geometric",
            "class": "torch_geometric.nn.PointTransformerConv"}
