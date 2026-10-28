import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatExtractNet(nn.Module):
    """
    FeatExtractNet v2: PINN with optional Fourier Feature Encoding and
    Spatial Grid Context, plus Cross-Layer Gated Fusion (CGF).

    Strategy selection:
      - Thermal (small N, irregular geometries): use_fourier=False,
        use_spatial_grid=False → global context (historical index bins by default).
      - Acoustic (large N, regular grid, high-freq): use_fourier=True,
        use_spatial_grid=True → Fourier encoding + spatial grid context
        (gradient-vanishing benefit has not been established).

    Architecture:
      Input:  [B, 2, N]  — point coordinates (x, y)
      Output: [B, Cout, N] — Cout=4 (u,v,p,T) for thermal, Cout=1 (p_total) for acoustic
    """

    POOL_K = 32
    POOL_MODES = ('legacy_index_bins', 'symmetric')
    FOURIER_LAYOUTS = ('legacy', 'pointwise')

    def __init__(self, input_channels=2, output_channels=4, scaling=1.0,
                 fourier_levels=12, spatial_grid=16,
                 use_fourier=False, use_spatial_grid=False,
                 pool_mode='legacy_index_bins', fourier_layout='legacy'):
        super().__init__()
        if pool_mode not in self.POOL_MODES:
            raise ValueError(f"Unsupported pool_mode: {pool_mode!r}")
        if fourier_layout not in self.FOURIER_LAYOUTS:
            raise ValueError(f"Unsupported fourier_layout: {fourier_layout!r}")
        self.pool_mode = pool_mode
        self.fourier_layout = fourier_layout
        s = lambda x: int(x * scaling)
        self.s64 = s(64)
        self.s128 = s(128)
        self.s256 = s(256)
        self.s512 = s(512)
        self.output_channels = output_channels

        self.use_fourier = use_fourier
        self.use_spatial_grid = use_spatial_grid
        self.fourier_levels = fourier_levels
        self.spatial_grid = spatial_grid

        if use_fourier:
            input_dim = input_channels + input_channels * 2 * fourier_levels
            self.register_buffer('_freqs',
                                 2.0 ** torch.arange(fourier_levels, dtype=torch.float32) * torch.pi,
                                 persistent=False)
        else:
            input_dim = input_channels

        self.input_fc = nn.Conv1d(input_dim, self.s64, kernel_size=1)
        self.conv_mid = nn.Conv1d(self.s64, self.s128, kernel_size=1)
        self.conv_high = nn.Conv1d(self.s128, self.s256, kernel_size=1)

        if use_spatial_grid:
            self.spatial_conv1 = nn.Conv1d(self.s256, self.s256, kernel_size=1)
            self.spatial_conv2 = nn.Conv1d(self.s256, self.s256, kernel_size=1)
        else:
            self.global_conv1 = nn.Conv1d(3 * self.s256, self.s256, kernel_size=1)
            self.global_conv2 = nn.Conv1d(self.s256, self.s256, kernel_size=1)
            self.global_lnorm = nn.LayerNorm(self.s256)

        self.gate_low = nn.Conv1d(self.s64 + self.s256, self.s64, kernel_size=1)
        self.gate_mid = nn.Conv1d(self.s128 + self.s256, self.s128, kernel_size=1)
        self.gate_high = nn.Conv1d(self.s256 + self.s256, self.s256, kernel_size=1)

        self.compress_low = nn.Conv1d(self.s256, self.s64, kernel_size=1)
        self.compress_mid = nn.Conv1d(self.s256, self.s128, kernel_size=1)
        self.compress_high = nn.Conv1d(self.s256, self.s256, kernel_size=1)

        fused_dim = self.s64 + self.s128 + self.s256 + self.s256
        self.final_fc1 = nn.Conv1d(fused_dim, self.s512, kernel_size=1)
        self.final_fc2 = nn.Conv1d(self.s512, self.s256, kernel_size=1)
        self.final_fc3 = nn.Conv1d(self.s256, self.s128, kernel_size=1)

        self.shortcut = nn.Conv1d(fused_dim, output_channels, kernel_size=1)
        self.output_conv = nn.Conv1d(self.s128, output_channels, kernel_size=1)

    def _fourier_encode(self, x):
        B, C, N = x.shape
        freqs = self._freqs.to(x.device).view(1, 1, 1, -1)
        x_exp = x.unsqueeze(-1)
        angles = x_exp * freqs
        fourier = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if self.fourier_layout == 'pointwise':
            fourier = fourier.permute(0, 1, 3, 2).reshape(
                B, C * 2 * self.fourier_levels, N)
        else:
            fourier = fourier.reshape(B, C * 2 * self.fourier_levels, N)
        return torch.cat([x, fourier], dim=1)

    def _global_pool_context(self, f_high):
        B, C, N = f_high.shape
        if self.pool_mode == 'symmetric':
            K = 1
        else:
            K = min(self.POOL_K, N)
        g_max = F.adaptive_max_pool1d(f_high, K)
        g_avg = F.adaptive_avg_pool1d(f_high, K)
        g_var_full = torch.var(f_high, dim=2, keepdim=True)
        g_var = g_var_full.expand(-1, -1, K)
        g_cat = torch.cat([g_max, g_avg, g_var], dim=1)

        g_spatial = F.relu(self.global_conv1(g_cat))
        g_spatial = F.relu(self.global_conv2(g_spatial))

        f_global_spatial = F.adaptive_avg_pool1d(g_spatial, 1)
        f_global_flat = f_global_spatial.squeeze(-1)
        f_global_normed = self.global_lnorm(f_global_flat)
        f_global = f_global_normed.unsqueeze(-1)
        return f_global.expand(-1, -1, N)

    def _spatial_grid_context(self, f_high, coords):
        B, C_high, N = f_high.shape
        G = min(self.spatial_grid, int(N ** 0.25))
        G = max(G, 2)
        num_cells = G * G

        x, y = coords[:, 0:1, :], coords[:, 1:2, :]
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        eps = 1e-6
        xi = ((x - x_min) / (x_max - x_min + eps) * (G - 1)).long().clamp(0, G - 1)
        yi = ((y - y_min) / (y_max - y_min + eps) * (G - 1)).long().clamp(0, G - 1)
        cell_idx = yi * G + xi

        context_list = []
        for b in range(B):
            idx_b = cell_idx[b, 0].long()

            f_b = f_high[b]
            one_hot = F.one_hot(idx_b, num_classes=num_cells).float()
            cell_sum = f_b @ one_hot
            counts = one_hot.sum(dim=0, keepdim=True)
            cell_mean = cell_sum / (counts + 1e-8)

            cell_mean = self.spatial_conv1(cell_mean.unsqueeze(0))
            cell_mean = self.spatial_conv2(F.relu(cell_mean))

            context_b = cell_mean.squeeze(0)[:, idx_b]
            context_list.append(context_b)

        return torch.stack(context_list, dim=0)

    def forward(self, x):
        B, Cin, N = x.shape

        if self.use_fourier:
            x_enc = self._fourier_encode(x)
        else:
            x_enc = x

        f_low = torch.tanh(self.input_fc(x_enc))
        f_mid = F.relu(self.conv_mid(f_low))
        f_high = F.relu(self.conv_high(f_mid))

        if self.use_spatial_grid:
            f_context = self._spatial_grid_context(f_high, x)
        else:
            f_context = self._global_pool_context(f_high)

        cat_low = torch.cat([f_low, f_context], dim=1)
        gate_low = torch.sigmoid(self.gate_low(cat_low))
        f_low_fused = gate_low * f_low + (1 - gate_low) * self.compress_low(f_context)

        cat_mid = torch.cat([f_mid, f_context], dim=1)
        gate_mid = torch.sigmoid(self.gate_mid(cat_mid))
        f_mid_fused = gate_mid * f_mid + (1 - gate_mid) * self.compress_mid(f_context)

        cat_high = torch.cat([f_high, f_context], dim=1)
        gate_high = torch.sigmoid(self.gate_high(cat_high))
        f_high_fused = gate_high * f_high + (1 - gate_high) * self.compress_high(f_context)

        f_fused = torch.cat([f_low_fused, f_mid_fused, f_high_fused, f_context], dim=1)

        out = torch.tanh(self.final_fc1(f_fused))
        out = torch.tanh(self.final_fc2(out))
        out = torch.tanh(self.final_fc3(out))
        output = self.output_conv(out)

        residual = self.shortcut(f_fused)
        output = output + residual

        return output


class FourierMLP(nn.Module):
    """
    Fourier Feature Coordinate MLP with residual connections.

    Architecture:
      (x,y) → Fourier encode → ResidualBlock×N → Linear → p

    Residual blocks with LayerNorm help gradient flow in deep networks.
    """

    def __init__(self, input_channels=2, output_channels=1,
                 hidden_dim=256, num_blocks=4, fourier_levels=16,
                 fourier_layout='legacy'):
        super().__init__()
        if fourier_layout not in FeatExtractNet.FOURIER_LAYOUTS:
            raise ValueError(f"Unsupported fourier_layout: {fourier_layout!r}")
        self.fourier_layout = fourier_layout
        self.fourier_levels = fourier_levels
        fourier_dim = input_channels * 2 * fourier_levels
        input_dim = input_channels + fourier_dim

        self.register_buffer('_freqs',
                             2.0 ** torch.arange(fourier_levels, dtype=torch.float32) * torch.pi,
                             persistent=False)

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])

        self.output_proj = nn.Linear(hidden_dim, output_channels)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _fourier_encode(self, x):
        B, C, N = x.shape
        freqs = self._freqs.to(x.device).view(1, 1, 1, -1)
        x_exp = x.unsqueeze(-1)
        angles = x_exp * freqs
        fourier = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if self.fourier_layout == 'pointwise':
            fourier = fourier.permute(0, 1, 3, 2).reshape(
                B, C * 2 * self.fourier_levels, N)
        else:
            fourier = fourier.reshape(B, C * 2 * self.fourier_levels, N)
        return torch.cat([x, fourier], dim=1)

    def forward(self, x):
        B, C, N = x.shape
        x_enc = self._fourier_encode(x)
        x_t = x_enc.permute(0, 2, 1).reshape(B * N, -1)

        h = F.relu(self.input_proj(x_t))
        for block in self.blocks:
            h = block(h)
        y = self.output_proj(h)

        return y.reshape(B, N, -1).permute(0, 2, 1)


class ResidualBlock(nn.Module):
    """Residual block with LayerNorm + Linear + ReLU."""
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        residual = x
        h = self.norm1(x)
        h = F.relu(self.linear1(h))
        h = self.norm2(h)
        h = F.relu(self.linear2(h))
        return residual + h
