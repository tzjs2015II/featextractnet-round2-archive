"""MeshGraphNet-style GNN baseline for the thermal point-cloud task.

Reference and naming honesty
----------------------------
Pfaff, Fortunato, Sanchez-Gonzalez, Battaglia, "Learning Mesh-Based Simulation
with Graph Neural Networks", ICLR 2021. The scheme is the "encode - process -
decode" graph network: a per-node encoder, L message-passing blocks that update
**edges first and then nodes** (each with a residual connection), and a per-node
decoder.

``torch_geometric`` ships **no** ``MeshGraphNet`` class (verified:
``hasattr(torch_geometric.nn, 'MeshGraphNet') is False``), so this module
*assembles* the architecture from its stated components rather than wrapping an
official implementation. Following the design doc's no-substitution rule, the row
is therefore named **"MeshGraphNet-style encode-process-decode GNN (assembled)"**
in every table and in the text -- it must not be presented as the authors'
reference implementation.

What follows the paper and what is a local choice
-------------------------------------------------
Following: the encode/process/decode structure; edge-then-node updates; residual
connections in the processor; relative node positions as edge features; MLP
update functions.
Local choices (recorded here because the paper leaves them open): kNN graph with
``k`` neighbours instead of a provided mesh, LayerNorm instead of the paper's
unspecified normalisation, and the hidden width chosen for capacity alignment.

Interface
---------
``forward(x: [B, 2, N]) -> [B, 4, N]``, identical to the other point-cloud
baselines, so it runs in the existing Protocol-A pipeline unchanged.
"""

import torch
import torch.nn as nn

try:
    from torch_geometric.nn import knn_graph  # noqa: F401  (import check only)
    _PYG_AVAILABLE = True
    _IMPORT_ERROR = None
except Exception as _exc:  # noqa: BLE001
    _PYG_AVAILABLE = False
    _IMPORT_ERROR = repr(_exc)


def _mlp(dims):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class MeshGraphNetStyleBaseline(nn.Module):
    """Encode-process-decode GNN (assembled; see the module docstring).

    Args:
        input_channels: node input features (the coordinate channels).
        output_channels: per-node output channels (4 for thermal).
        hidden_dim: width of the encoder/processor/decoder MLPs.
        num_layers: number of processor (message-passing) blocks.
        k: neighbours per node in the kNN graph.

    Capacity: the defaults ``hidden_dim=128, num_layers=8`` give **976,516**
    parameters, the closest configuration searched to the ~1.05M rows already in
    the table (``h=192, L=4`` overshoots to 1,151,428 = +9.2 %; ``h=128, L=6``
    undershoots to 745,092 = −29.3 %). It is still 7.4 % smaller, so conclusions
    from this row carry that capacity qualification (design doc §2).
    """

    def __init__(self, input_channels=2, output_channels=4, hidden_dim=128,
                 num_layers=8, k=16):
        super().__init__()
        if not _PYG_AVAILABLE:
            raise RuntimeError(
                "torch_geometric is not importable here: "
                f"{_IMPORT_ERROR}. Install it (see requirements_ext.txt) or "
                "report the row as unavailable.")
        self.k = int(k)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)

        # encoder: node features -> latent (edge features are the relative
        # positions, projected separately as in the paper)
        self.node_encoder = _mlp([int(input_channels), hidden_dim, hidden_dim])
        self.edge_encoder = _mlp([int(input_channels), hidden_dim, hidden_dim])

        # processor: L blocks, each updating edges then nodes with residuals
        self.edge_mlps = nn.ModuleList([
            _mlp([3 * hidden_dim, hidden_dim, hidden_dim])
            for _ in range(self.num_layers)])
        self.node_mlps = nn.ModuleList([
            _mlp([2 * hidden_dim, hidden_dim, hidden_dim])
            for _ in range(self.num_layers)])
        self.edge_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(self.num_layers)])
        self.node_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(self.num_layers)])

        # decoder: latent -> output channels
        self.decoder = _mlp([hidden_dim, hidden_dim, int(output_channels)])

    def _edges(self, pos):
        """kNN edges per sample, returned as a single flattened index pair."""
        b, n, _ = pos.shape
        src, dst, eattr = [], [], []
        for i in range(b):
            pi = pos[i]
            d = torch.cdist(pi, pi)
            d.fill_diagonal_(float("inf"))
            idx = d.topk(self.k, dim=-1, largest=False).indices     # [N, k]
            s = torch.arange(n, device=pos.device).view(n, 1).expand(n, self.k)
            s = s.reshape(-1) + i * n
            t = idx.reshape(-1) + i * n
            src.append(s)
            dst.append(t)
            # edge feature = relative position of the neighbour (paper: relative
            # positions of the two nodes)
            eattr.append((pos[i].reshape(-1, pos.shape[-1])[idx.reshape(-1)]
                          - pos[i].reshape(-1, pos.shape[-1])[s - i * n]))
        return torch.cat(src), torch.cat(dst), torch.cat(eattr)

    def forward(self, x):
        """[B, C_in, N] -> [B, C_out, N]."""
        b, _, n = x.shape
        h = x.permute(0, 2, 1)                                  # [B, N, C_in]
        pos = h[..., :2]
        hf = h.reshape(b * n, -1)
        pf = pos.reshape(b * n, pos.shape[-1])

        hn = self.node_encoder(hf)
        src, dst, eattr = self._edges(pos)
        he = self.edge_encoder(eattr)

        for l in range(self.num_layers):
            # --- edge update (residual) ---
            e_in = torch.cat([hn[src], hn[dst], he], dim=-1)
            he = self.edge_norms[l](he + self.edge_mlps[l](e_in))
            # --- node update (residual), aggregating incoming edge latents ---
            agg = torch.zeros_like(hn).index_add_(0, dst, he)
            hn = self.node_norms[l](hn + self.node_mlps[l](
                torch.cat([hn, agg], dim=-1)))

        out = self.decoder(hn).reshape(b, n, -1)                 # [B, N, C_out]
        return out.permute(0, 2, 1)


def availability():
    """Report whether the required primitives are importable (for the audit)."""
    return {"torch_geometric_available": _PYG_AVAILABLE,
            "import_error": _IMPORT_ERROR,
            "official_class": "none in torch_geometric; assembled here",
            "row_name": "MeshGraphNet-style encode-process-decode GNN (assembled)"}
