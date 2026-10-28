"""cond_mlps.py —— 声学条件化逐点 MLP 基线（src.models.unified.cond_mlps）。

用途
----
为声学整场留出对比实验（scripts/exp_acoustic.py，Track A2）提供两个"逐点、
坐标→声压"监督基线，输入均为仿射归一化坐标，输出在归一化 p 域：
  CondFourierMLP  Fourier 特征编码 + 条件标量拼接 + ResidualBlock 残差 MLP，
                  与 FeatExtractNet.FourierMLP（无条件化）同规模级。
  CondCoordMLP    纯坐标逐点 MLP + 条件标量按点复制为常数通道，
                  与 baseline CoordMLP 同规模级（~1.05M）。

条件注入方式（本模块的选择，供报告注明）
----------------------------------------
统一采用"拼接（concat）"：cond 标量 [B, Nc] 沿 N 维复制为常数通道
[B, Nc, N]，与 Fourier 特征 / 坐标一起作为逐点 MLP 的输入通道。理由：
1) 常数通道语义透明、实现简单，不引入额外的 cond 投影层；
2) cond 与 Fourier 频率特征在进入 MLP 前即可交互（深层可习得按 f/Ma
   调节频带的功能），比"每样本加性偏置"（cond 经 Linear 后广播加）给条件
   更直接的建模通道；代价是输入通道多 Nc（本协议 Nc=2，参数量几乎不变）。
两种注入方式在消融里可互换，本模块固定用 concat 并在 docstring 注明。

接口约定
--------
forward(x: [B, Cin, N], cond: [B, Nc]) -> [B, Cout, N]
  x/cond 均应为 torch.float32；cond 若与 x 不同设备会自动搬移。输出为
  归一化 p 域（物理单位指标由 acoustic_io 评测层反归一化后计算）。
  与 UnifiedNet 的约定一致：坐标在通道维、点在最后维；权重 Kaiming
  (fan_in, relu) + bias 置零；模块内不设随机种子，由外层训练脚本统一 set_seed。

ResidualBlock 从同包 unified_net.py 导入（该块为 FeatExtractNet 的拷贝，
统合包保持自包含，见 unified_net.py docstring）。

声学一致消融（Track A3）的消融开关
----------------------------------
`CondFourierMLP` 增加两个开关，供消融实验在**同一基础架构**上只改单一组件
（token 见 scripts/exp_acoustic.py 的 MODEL_TOKENS）：
  fourier_levels=0   去 Fourier 编码（只留坐标 + cond）
  block_kind='nores' 去残差 skip；'nonorm' 去 LayerNorm
`unified_net.ResidualBlock` 本身**不改**——它被 UnifiedNet（FeatExtractNet 主体）
使用，改动会污染 A2 主对比与热场结果；消融变体以 cond_mlps 内的局部块实现。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unified_net import ResidualBlock  # noqa: E402（同包拷贝块，见模块 docstring）


def _fourier_sin_cos(coords, freqs, layout='legacy'):
    """sin/cos Fourier 特征：coords [B, C, N] -> [B, C*2*L, N]。

    频率 freqs 由各模块构造为 [L] buffer；不含原始坐标（需要原始坐标时由
    调用方 concat）。与 unified_net._fourier_sin_cos 同一套计算。

    layout：通道内 sin/cos 与频率的排布方式
      'legacy'    （默认，历史行为）直接 reshape [B,C,N,2L]，混合点与频率轴，
                  仅用于重现旧权重对应的历史实现。
      'pointwise' 按 [B, C, 2L, N] 排布：每个 (坐标通道, sin|cos) 的频率维
                  在通道维连续，便于逐点（pointwise）读取频带。
    两种布局通道数相同（C*2*L），参数量与 state_dict 不变。
    """
    if layout not in ('legacy', 'pointwise'):
        raise ValueError(f"unknown fourier layout {layout!r}")
    B, C, N = coords.shape
    L = freqs.numel()
    f = freqs.view(1, 1, 1, -1)             # [1,1,1,L]
    angles = coords.unsqueeze(-1) * f        # [B,C,N,L]
    fc = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [B,C,N,2L]
    if layout == 'pointwise':
        fc = fc.permute(0, 1, 3, 2)          # [B,C,2L,N]
    # 显式给出 reshape 维度（L=0 时 C*2*L=0，不可用推断 -1）
    return fc.reshape(B, C * 2 * L, N)


def _cond_channels(cond, x):
    """把 cond [B, Nc] 复制为逐点常数通道 [B, Nc, N]（与 x 同设备）。"""
    B, _, N = x.shape
    if cond is None:
        raise ValueError('cond_mlps 需要 cond [B, Nc]（声学协议 Nc=2）')
    if cond.device != x.device:
        cond = cond.to(x.device)
    if cond.ndim == 1:                      # 容错：[Nc] -> [1, Nc]
        cond = cond.unsqueeze(0)
    if cond.shape[0] != B:
        raise ValueError(f'cond batch {cond.shape[0]} != x batch {B}')
    return cond.unsqueeze(-1).expand(B, cond.shape[1], N)   # [B, Nc, N]


class _BlockNoResidual(nn.Module):
    """ResidualBlock 的消融变体：去掉残差 skip（`residual + h` -> `h`）。

    与 ResidualBlock 完全相同的层配置（LayerNorm + Linear + ReLU 两层），仅前向
    少一次加和，因此**参数量与 ResidualBlock 相同**，可直接归因于"残差连接"。
    在 cond_mlps 内局部定义而不改 unified_net.ResidualBlock：后者被 UnifiedNet
    （=FeatExtractNet 主体）使用，改它会污染 A2 主对比与热场结果。
    """

    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.norm1(x)
        h = F.relu(self.linear1(h))
        h = self.norm2(h)
        h = F.relu(self.linear2(h))
        return h


class _BlockNoNorm(nn.Module):
    """ResidualBlock 的消融变体：去掉两处 LayerNorm（保留残差 skip）。

    去掉 LayerNorm 会移除其可学习参数，故参数量略少于 ResidualBlock
    （差额 = 2*2*dim，记录在消融表的参数量列）。
    """

    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = F.relu(self.linear1(x))
        h = F.relu(self.linear2(h))
        return x + h


class CondFourierMLP(nn.Module):
    """条件化 Fourier 特征 MLP（与 FourierMLP 同规模级，仅多 Nc 个条件通道）。

    结构：
      x 归一化坐标 [B,2,N]
        --Fourier: cat([x(原始坐标), sin/cos(freqs*x)], dim=1)--> [B, 2+4L, N]
        --concat cond 常数通道-------------------------------> [B, 2+4L+Nc, N]
        --Linear(in_dim -> hidden) + ReLU--> 逐点 tokens [B*N, hidden]
        --ResidualBlock(hidden) x num_blocks-->
        --Linear(hidden -> 1)--> reshape 回 [B, 1, N]

    Fourier 频率 2^[0..L-1]*pi 与 FeatExtractNet.FourierMLP / UnifiedNet
    分支 F 完全一致；默认 hidden=256、num_blocks=6、levels=16（与分支 F 同款）。
    cond 用 concat 注入（选择理由见模块 docstring）。

    消融开关（供 Track A3 声学一致消融；除下列开关外一切保持与 A2 的
    `fourier_c` 同款，保证处理组与对照只差单一可归因变量）：
      fourier_levels=0  关闭 Fourier 编码，只留原始坐标 + cond 通道
                        （输入维 2+4L+Nc -> 2+Nc）。原实现以 `>=1` 硬拦，
                        这里放开为 `>=0`；L=0 时 `_freqs` 为空 buffer，
                        `_fourier_sin_cos` 自然返回 [B,0,N]，无需分支。
      block_kind        逐点 MLP 的 block 类型：
                        'full'（默认，ResidualBlock）
                        'nores'（去残差）/ 'nonorm'（去 LayerNorm）
    """

    def __init__(self, input_channels=2, output_channels=1, cond_dim=2,
                 hidden_dim=256, num_blocks=6, fourier_levels=16,
                 block_kind='full', fourier_layout='legacy'):
        super().__init__()
        if fourier_levels < 0:
            raise ValueError('fourier_levels must be >= 0')
        if block_kind not in ('full', 'nores', 'nonorm'):
            raise ValueError(f"unknown block_kind {block_kind!r}")
        if fourier_layout not in ('legacy', 'pointwise'):
            raise ValueError(f"unknown fourier_layout {fourier_layout!r}")
        if cond_dim < 1:
            raise ValueError('cond_dim must be >= 1')
        self.block_kind = block_kind
        self.fourier_layout = fourier_layout
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.fourier_levels = fourier_levels

        self.register_buffer(
            '_freqs',
            2.0 ** torch.arange(fourier_levels, dtype=torch.float32) * math.pi,
            persistent=False)

        # 输入通道 = 原始坐标 + sin/cos(2L/坐标) + cond(Nc)
        self.fourier_dim = input_channels * 2 * fourier_levels
        self.input_dim = input_channels + self.fourier_dim + cond_dim
        self.input_proj = nn.Linear(self.input_dim, hidden_dim)
        block_cls = {'full': ResidualBlock,
                     'nores': _BlockNoResidual,
                     'nonorm': _BlockNoNorm}[block_kind]
        self.blocks = nn.ModuleList(
            [block_cls(hidden_dim) for _ in range(num_blocks)])
        self.output_proj = nn.Linear(hidden_dim, output_channels)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, cond):
        """x: [B, Cin, N] 归一化坐标；cond: [B, Nc] -> [B, Cout, N]。"""
        B, Cin, N = x.shape
        if Cin != self.input_channels:
            raise ValueError(
                f'expected input_channels={self.input_channels}, got {Cin}')

        # Fourier 特征：原始坐标 + sin/cos（与 FourierMLP 一致）
        freqs = self._freqs.to(x.device)
        xf = _fourier_sin_cos(x, freqs, layout=self.fourier_layout)  # [B, Cin*2L, N]
        feats = torch.cat([x, xf], dim=1)                # [B, Cin+2L*Cin, N]

        # cond concat 注入：常数通道 [B, Nc, N]
        cc = _cond_channels(cond, x)                     # [B, Nc, N]
        x_enc = torch.cat([feats, cc], dim=1)            # [B, input_dim, N]

        # 逐点 MLP（tokens 化，同 FourierMLP 布局）
        xt = x_enc.permute(0, 2, 1).reshape(B * N, self.input_dim)
        h = F.relu(self.input_proj(xt))                  # [B*N, hidden]
        for blk in self.blocks:
            h = blk(h)
        y = self.output_proj(h)                          # [B*N, Cout]
        return y.reshape(B, N, -1).permute(0, 2, 1)      # [B, Cout, N]


class CondCoordMLP(nn.Module):
    """条件化纯坐标逐点 MLP（与 baseline CoordMLP 同规模级，~1.05M）。

    结构：
      x [B,2,N] --concat cond 常数通道 [B,Nc,N]--> [B, 2+Nc, N]
        --Conv1d(k=1) 逐点 MLP（Cin -> hidden；hidden x (num_layers-2)；-> 1）

    用 nn.Conv1d(kernel_size=1) 实现逐点共享全连接（与 CoordMLP 同形态）；
    hidden=512、num_layers=6 -> 输入 2+Nc 时 ~1.05M（含 Nc=2，与 CoordMLP
    默认规模一致）。cond concat 注入，理由见模块 docstring。
    """

    def __init__(self, input_channels=2, output_channels=1, cond_dim=2,
                 hidden_dim=512, num_layers=6):
        super().__init__()
        if num_layers < 2:
            raise ValueError('num_layers must be >= 2')
        if cond_dim < 1:
            raise ValueError('cond_dim must be >= 1')
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim

        layers = []
        layers.append(nn.Conv1d(input_channels + cond_dim, hidden_dim, 1))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_layers - 2):
            layers.append(nn.Conv1d(hidden_dim, hidden_dim, 1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv1d(hidden_dim, output_channels, 1))
        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, cond):
        """x: [B, Cin, N] 归一化坐标；cond: [B, Nc] -> [B, Cout, N]。"""
        cc = _cond_channels(cond, x)                     # [B, Nc, N]
        x_cat = torch.cat([x, cc], dim=1)                # [B, Cin+Nc, N]
        return self.net(x_cat)                           # [B, Cout, N]
