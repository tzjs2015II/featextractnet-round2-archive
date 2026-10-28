"""FNO2d —— 面向规则网格声场的 2D 谱方法（Fourier Neural Operator）基线。

用途
----
规则网格（如图像式声压场）上的监督回归基线：以 FFT 谱卷积学习频域算子，
与点云类基线（逐点/邻域）形成对照。网格化声场时输出单通道 p_total。

接口
----
forward_grid(grid: [B, Cin, H, W]) -> [B, Cout, H, W]
    Cin >= 2：前两通道为归一化网格坐标 (x,y)，可附加物理参数条件通道
    （如声源频率 f、马赫数 Ma 等标量广播通道，由数据管线在调用方拼装）；
    Cout 默认 1（p_total）。要求 H、W 均可被 2 整除且 >= 2*modes。

结构（L 层谱卷积，默认 L=4, modes=16, width=32）
----
1. fc0：Conv2d(1x1, Cin -> width) 提升通道。
2. 每层 = 逐点卷积分支(Conv2d 1x1, width->width) + 谱卷积分支：
       FFT2 -> 截取低频/负频模态块（前 modes 行、后 modes 行、前 modes 列，
       共 2 个复权块）-> IFFT2 -> 残差相加 -> ReLU。
   谱卷积参数为复权（存储为实部/虚部两组实数参数）。
3. 输出：Conv2d(1x1) 逐点 MLP(width -> 128 -> Cout)，最终投影到 Cout 通道。

实现要点
--------
- 作者自实现（无第三方算子）：torch.fft.rfft2/irfft2 + einsum 复乘。
- 复权以 nn.Parameter(...,2) 实数形式存储（便于 Adam 与 kaiming 风格一致），
  前向用 torch.view_as_complex 构造成复张量。
- 参考标准 FNO（Li et al.）残差结构：逐点卷积分支与谱卷积分支相加。
- 注意与 [B,2,N] 点云接口不同：本类只提供网格前向 forward_grid。
- export_fno_point_forward() 仅为点云入口的说明（未实现，见其 docstring）。
"""

import torch
import torch.nn as nn


def _compl_mul2d(x_ft, w):
    """复乘: x_ft:[B, i, X, Y], w:[i, o, X, Y] -> [B, o, X, Y]。"""
    return torch.einsum('bixy,ioxy->boxy', x_ft, w)


class SpectralConv2d(nn.Module):
    """2D 谱卷积（FNO）：低频块 + 负频行块两路复权线性。"""

    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        # 复权以实数形式存为 (in, out, modes, modes, 2)（实部/虚部）
        self.w1 = nn.Parameter(scale * torch.rand(
            in_channels, out_channels, modes, modes, 2))
        self.w2 = nn.Parameter(scale * torch.rand(
            in_channels, out_channels, modes, modes, 2))

    def forward(self, x):
        # x: [B, in_channels, H, W]（实值）-> [B, out_channels, H, W]
        B, C, H, W = x.shape
        m = self.modes
        w1 = torch.view_as_complex(self.w1.contiguous())  # [i,o,m,m] cfloat
        w2 = torch.view_as_complex(self.w2.contiguous())

        x_ft = torch.fft.rfft2(x)                         # [B,C,H,W//2+1]
        out_ft = torch.zeros(
            B, C, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        # 低频块：前 m 行、前 m 列
        out_ft[:, :, :m, :m] = _compl_mul2d(x_ft[:, :, :m, :m], w1)
        # 负频行块：后 m 行（rfft2 最后一维已隐含共轭对称，故只取前 m 列）
        out_ft[:, :, -m:, :m] = _compl_mul2d(x_ft[:, :, -m:, :m], w2)
        return torch.fft.irfft2(out_ft, s=(H, W))         # [B,o,H,W] 实值


class FNO2d(nn.Module):
    """Fourier Neural Operator (2D) for regular-grid acoustic fields."""

    def __init__(self, in_channels=2, out_channels=1,
                 modes=16, width=32, n_layers=4):
        super().__init__()
        self.modes = modes
        self.width = width
        self.n_layers = n_layers

        self.fc0 = nn.Conv2d(in_channels, width, 1)          # 通道提升

        self.spectral_convs = nn.ModuleList(
            [SpectralConv2d(width, width, modes) for _ in range(n_layers)])
        self.convs = nn.ModuleList(                          # 逐点卷积分支
            [nn.Conv2d(width, width, 1) for _ in range(n_layers)])

        # 输出投影 MLP：width -> 128 -> out_channels
        self.fc1 = nn.Conv2d(width, 128, 1)
        self.fc2 = nn.Conv2d(128, out_channels, 1)

        self._init_weights()

    def forward_grid(self, grid):
        # grid: [B, Cin, H, W] -> [B, Cout, H, W]
        B, Cin, H, W = grid.shape
        if H < 2 * self.modes or W < 2 * self.modes:
            raise RuntimeError(
                f"FNO2d requires H,W >= 2*modes={2 * self.modes}; got "
                f"{H}x{W}.")
        x = self.fc0(grid)                                   # [B,width,H,W]
        for l in range(self.n_layers):
            x1 = self.convs[l](x)                            # 逐点分支
            x2 = self.spectral_convs[l](x)                   # 谱分支
            x = torch.relu(x1 + x2)                          # 残差求和
        x = torch.relu(self.fc1(x))
        return self.fc2(x)                                   # [B,Cout,H,W]

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def export_fno_point_forward():
    """点云接口 [B,2,N] -> [B,Cout,N] 的接入说明（未实现）。

    若需在"点云式声场评测"里使用 FNO2d，可采用的接入方式（在数据管线层
    实现，不进入本模块）：
      1) 训练/评测几何均为规则网格，先将点坐标 (x,y) 映射回规则网格
         （反归一化 + 双线性/最近邻散射），拼上物理条件通道得到
         [B,Cin,H,W]；跑 forward_grid 得 [B,1,H,W]；
      2) 用原坐标在网格上做最近邻/双线性插值回 [B,1,N] 完成逐点评测，
         或直接以网格整场做像素级监督（此时评测对象是网格场本身）。
    实现取舍：FNO 依赖规则网格 FFT，无法直接作用于任意稀疏点集；
    是否启用取决于"声场主实验"最终裁决的数据形态（网格场 vs 点云）。
    """
    raise NotImplementedError(
        "FNO2d 只提供网格前向 forward_grid；点云接入见本函数 docstring。")
