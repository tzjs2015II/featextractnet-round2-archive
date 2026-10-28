"""UnifiedNet —— 共享语义空间 + 动态路由的统合网络（作者 D1 决策核心新架构）。

用途
----
单一网络同时承担热场（u,v,p,T，Cout=4）与声场（p_total，Cout=1）两类任务：
两任务共享一个坐标编码 stem 与一个采样级软路由，在共享语义空间
（Dsem 维逐点特征）上做分支级加权混合，再分别经两个浅任务头输出。
对应设计文档：``返稿意见/返稿修订工作区/docs/U1_统合架构设计.md``
（v1, 2026-09-09；本文件为 U1 §2/§3/§6 的实现）。

设计依据（对照 U1）
-------------------
- 张量约定 [B, C, N]（坐标在通道维、点在最后维）；主干全部为
  ``Conv1d(kernel_size=1)`` 逐点卷积；条件 ``cond: [B, Nc]`` 经线性层后
  按样本广播加到 stem 输出（无 cond 时该项为 0）。
- 分支 P = MLFE + tri-stat 全局池化 + CGF 门控融合（几何感知，见 U1 §2.3）；
- 分支 F = 坐标 Fourier 特征 + ResidualBlock 残差 MLP（高频网格，见 U1 §2.4）；
- 采样级软路由 alpha = sigmoid(logit / T)，T 由外层退火（见 U1 §2.2/§3）。

=== 关键约束：路由输入禁止任务名 / 数据集 id ===
路由的输入 ``sig`` 只由**几何结构统计量**（log N、NN 距离统计、包围盒、
可选条件标量 cond）构成。任务名/数据集 id **不得**作为路由输入，原因：
1) 若允许把 task-id 喂给路由，alpha 会退化成"记任务标签的查表 oracle"，
   等于把"该选哪个分支"这一判断外包给标签，违背 U1"路由不得以 task-id
   为 oracle"的反批评目标（审稿 A#2）；
2) 几何签名是跨几何泛化实验（几何级留出、整场留出）唯一允许出现在路由
   输入里的信息，task-id 会在留出测试中造成信息泄漏；
3) 同一几何在不同任务下不应改变路由行为——路由应学习"网格规整程度、
   点云密度"这类与物理域无关的结构特征，才可能在热/声两域间迁移。
因此本类 forward 只接收坐标与 cond，task 仅决定取哪个任务头与损失，
**不进入路由**。回归/冒烟脚本也不得改此签名。

=== STE 与硬路由实现要点（本文件选择的梯度处理方式）===
U1 §3 的训练策略是"软路由热身 -> 硬路由退火"；U1 §7 提供两种硬路由实现
（直通 STE 旁路 / 只算被选分支+外层 router loss）。本实现选择后者，理由
与具体处理如下：

1. ``hard=False``（默认；软路由阶段，对应 U1 §3"前 10% epoch 用软 alpha"）：
   - 两个分支都前向，``h_sem = alpha*projP(F_p) + (1-alpha)*projF(F_f)``；
   - 任务损失经 alpha 的凸组合**直接回传到 router logit**（alpha 由
     sigmoid(logit/T) 得到，可微），router 借此学到"哪个分支对当前样本
     更有效"。这是 router 最主要的任务驱动学习信号。
2. ``hard=True``（硬路由阶段，外层在 epoch 比例 > HARD_ROUTER_WARMUP_EPOCH_FRAC
   后传入；也可用 hard 参数直接控制单次调用）：
   - 按 ``alpha_hard = (alpha > 0.5)`` 对**每个样本**选择分支；前向**只计算
     被选分支**（batch 内按 mask 切子集计算，未选分支完全不前向），节省算力、
     让分支各自专业化。此时 ``h_sem`` 经布尔索引/切子集得到，**不依赖
     router 参数**——二元选择本身不可微，本实现刻意不让任务损失穿过
     硬掩码回传（若强行回传需要未选分支的 (p-f) 差值，会抵消省算力收益；
     直通 STE 的完整形态见下）。
   - router 的可学习信号来自两条路径：(a) 软路由热身阶段积累的任务梯度；
     (b) 外层基于 ``diag`` 计算的 router loss：``entropy_reg * H(alpha)`` 与
     ``load_balance_reg * (mean(alpha)-0.5)^2``。本模块**不内置**该 loss，
     只把 ``entropy_reg`` / ``load_balance_reg`` 存为属性常量供外层取用
     （U1 §2.6/§3）；``diag`` 已暴露 alpha/alpha_hard/entropy/logit/sig，
     外层据此回传即实现对 router logit 的"保留可回传路径"。
   - 若外层确实需要 hard 阶段的任务损失驱动 router（直通 STE 形态：
     前向取二元选择值、反向按 ``alpha_st = alpha + (alpha_hard-alpha).detach()``
     的凸组合走梯度），请改为 ``hard=False`` + 低 ``router_temperature``
     的近似（soft-argmax），或自行同时前向两分支并构造 STE 凸组合；
     本模块 forward 不做该默认行为，已在 docstring 明示取舍。
3. 评测（eval）：外层用 ``hard=False``，``h_sem`` 为确定性软混合；diag 同时
   返回 alpha_soft（=alpha）与 alpha_hard 供 U4 路由诊断存 npz。

=== 消融 branch_mode（U1 §2.6 / U3d）===
- 'unified'         默认：router 决定 alpha。
- 'branchP_only'    固定 alpha=1（只前向分支 P，router 不参与融合）。
- 'branchF_only'    固定 alpha=0（只前向分支 F）。
- 'concat_no_router'去掉路由，改为拼接 projP(F_p) 与 projF(F_f)，头输入维度
                    变为 2*Dsem（Head 以 in_dim 参数支持）。
以上四模式 forward 接口与 diag 字段保持一致。

=== 输入输出与诊断 ===
forward(x, cond=None, task=None, hard=False) -> (out, diag)
- x:    [B, Cin, N] 归一化坐标（Cin=2 为 (x,y)）
- cond: [B, Nc] 或 None（声场为归一化 f、Ma；热场 None/零）
- task: 'thermal' -> out=[B, 4, N]（取 HeadT）；'acoustic' -> out=[B, 1, N]
  （取 HeadA）；None 默认 'thermal'。task 不影响路由输入。
- diag: dict{alpha:[B]（软）, alpha_soft:[B]（=alpha）, alpha_hard:[B],
  sig:[B, Dr]（Dr=5+Nc）, entropy:标量, logit:[B],
  out_thermal:[B,4,N], out_acoustic:[B,1,N]}；后两键在
  compute_both_heads=False 时只含 task 对应头。

权重初始化：Kaiming(fan_in, relu) + bias 置零（同 FeatExtractNet/CoordMLP
风格）；无 BN/Dropout。LayerNorm 保持默认。模块内不设随机种子，由外层训练
脚本统一 set_seed。

文件内辅助类：ResidualBlock（从 featextractnet.py 拷贝，出处见类 docstring）、
BranchP、BranchF、Router、TaskHead——均以独立子模块属性挂在 UnifiedNet 下，
便于按组件统计参数量（见 ``component_param_counts``）。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _fourier_sin_cos(coords, freqs, levels, layout='legacy'):
    """Fourier 特征（sin/cos，不含原始坐标）。

    coords: [B, C, N] -> 返回 [B, C*2*L, N]。
    用于分支 F（与 U1 §2.4 一致：分支 F 用自身 Fourier，不与 stem_fourier 叠加）。

    layout：'legacy'（默认，历史排布：[B,C,N,2L] 展平）或 'pointwise'
    （[B,C,2L,N] 排布，频率维在通道维连续）。通道数相同，参数量不变。
    """
    if layout not in ('legacy', 'pointwise'):
        raise ValueError(f"unknown fourier layout {layout!r}")
    B, C, N = coords.shape
    f = freqs.view(1, 1, 1, -1)          # [1, 1, 1, L]
    angles = coords.unsqueeze(-1) * f     # [B, C, N, L]
    fc = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [B,C,N,2L]
    if layout == 'pointwise':
        fc = fc.permute(0, 1, 3, 2)       # [B, C, 2L, N]
    # 显式给出 reshape 维度（L=0 时 C*2*L=0，不可用推断 -1）
    return fc.reshape(B, C * 2 * levels, N)


def _fourier_append(coords, freqs, levels, layout='legacy'):
    """原始坐标 + sin/cos（FeatExtractNet._fourier_encode 风格，供 stem_fourier 用）。

    coords: [B, C, N] -> [B, C*(1+2L), N]。layout 透传给 _fourier_sin_cos。
    """
    return torch.cat(
        [coords, _fourier_sin_cos(coords, freqs, levels, layout=layout)], dim=1)


class ResidualBlock(nn.Module):
    """残差块（LayerNorm + Linear + ReLU x2），从 featextractnet.py 原样拷贝。

    输入输出均为 [*, dim]（逐点：外层把 [B,C,N] 压成 [B*N, C] 再喂入）。
    拷贝而非 import：featextractnet.py 为只读资产，统合包保持自包含。
    """

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
        # x: [*, dim]
        residual = x
        h = self.norm1(x)
        h = F.relu(self.linear1(h))
        h = self.norm2(h)
        h = F.relu(self.linear2(h))
        return residual + h


class BranchP(nn.Module):
    """分支 P：MLFE + tri-stat 全局池化 + CGF 门控融合（U1 §2.3，几何感知路径）。

    结构（默认宽度，可被 width_scale 缩放）：
      h0 [B,D0,N] --conv_mid(+ReLU)--> f_mid [B,D1,N]
                 --conv_high(+ReLU)--> f_high [B,D2,N]
      tri-stat 全局池化: cat([max, avg, var] over N, dim=1) = [B,3*D2,1]
                 --pool_conv(+ReLU)--> [B,D2,1] --LayerNorm--> global [B,D2,1]
      global 广播展开到 N: g_exp [B,D2,N]
      CGF（只对 mid/high 两层做，h0 本身已是"低层"64-d，见 U1 §2.3）:
        gate_mid    = sigmoid(Conv1d(cat(f_mid,  g_exp)) [D1+D2 -> D1])
        f_mid_fused = gate_mid * f_mid + (1-gate_mid) * compress_mid(g_exp)
        gate_high   = sigmoid(Conv1d(cat(f_high, g_exp)) [D2+D2 -> D2])
        f_high_fused= gate_high * f_high + (1-gate_high) * compress_high(g_exp)
      输出 F_p = cat([f_mid_fused(128); f_high_fused(256); g_exp(256)], dim=1)
              = [B, D1+2*D2, N]（默认 640-d，见 U1 §2.3）。
    """

    def __init__(self, in_dim=64, mid_dim=128, high_dim=256):
        super().__init__()
        self.in_dim = in_dim
        self.mid_dim = mid_dim
        self.high_dim = high_dim
        self.fused_out = mid_dim + 2 * high_dim   # 640 @ 默认宽度

        # MLFE 层级（U1 §2.3，同 FeatExtractNet 的 conv_mid/conv_high）
        self.conv_mid = nn.Conv1d(in_dim, mid_dim, kernel_size=1)
        self.conv_high = nn.Conv1d(mid_dim, high_dim, kernel_size=1)

        # tri-stat 全局池化：cat([max; avg; var]) -> Conv1d(3*D2 -> D2) -> ReLU -> LayerNorm
        self.pool_conv = nn.Conv1d(3 * high_dim, high_dim, kernel_size=1)
        self.pool_ln = nn.LayerNorm(high_dim)

        # CGF 门控：对 f_mid 与 f_high 逐层融合全局上下文
        self.gate_mid = nn.Conv1d(mid_dim + high_dim, mid_dim, kernel_size=1)
        self.compress_mid = nn.Conv1d(high_dim, mid_dim, kernel_size=1)
        self.gate_high = nn.Conv1d(high_dim + high_dim, high_dim, kernel_size=1)
        self.compress_high = nn.Conv1d(high_dim, high_dim, kernel_size=1)

    def forward(self, h0):
        # h0: [B, D0, N] 逐点共享语义嵌入
        f_mid = F.relu(self.conv_mid(h0))       # [B, D1, N]
        f_high = F.relu(self.conv_high(f_mid))  # [B, D2, N]
        N = f_high.shape[-1]

        # --- tri-stat 全局池化（逐样本在 N 维上聚合，尺度不变） ---
        g_max = F.adaptive_max_pool1d(f_high, 1)                 # [B, D2, 1]
        g_avg = F.adaptive_avg_pool1d(f_high, 1)                 # [B, D2, 1]
        g_var = torch.var(f_high, dim=2, keepdim=True)           # [B, D2, 1]
        g_cat = torch.cat([g_max, g_avg, g_var], dim=1)          # [B, 3*D2, 1]
        g = F.relu(self.pool_conv(g_cat))                        # [B, D2, 1]
        g = self.pool_ln(g.squeeze(-1)).unsqueeze(-1)            # [B, D2, 1]
        g_exp = g.expand(-1, -1, N)                              # [B, D2, N]

        # --- CGF 逐层融合 ---
        gate_mid = torch.sigmoid(self.gate_mid(torch.cat([f_mid, g_exp], dim=1)))
        f_mid_fused = gate_mid * f_mid + (1 - gate_mid) * self.compress_mid(g_exp)

        gate_high = torch.sigmoid(self.gate_high(torch.cat([f_high, g_exp], dim=1)))
        f_high_fused = gate_high * f_high + (1 - gate_high) * self.compress_high(g_exp)

        # F_p: [B, D1+D2+D2, N]（默认 [B, 640, N]，U1 §2.3）
        return torch.cat([f_mid_fused, f_high_fused, g_exp], dim=1)


class BranchF(nn.Module):
    """分支 F：Fourier 残差 MLP 路径（U1 §2.4，高频规整网格）。

    输入：cat([h0, Fourier(coords, L)], dim=1)，其中
      - h0: [B, D0, N]（stem 输出，共享）
      - Fourier(coords, L): sin/cos 特征 [B, Cin*2*L, N]（分支 F 自身编码，
        与 stem_fourier 不叠加，U1 §7）
      -> 分支 F 输入通道 = D0 + Cin*2*L（默认 64 + 64 = 128）
    结构：Linear(D0+Cin*2*L -> D2=256) + ReLU -> num_fourier_blocks 个
      ResidualBlock(256)（逐点，[B,N] 拍平后过 LayerNorm+Linear+ReLU）
      -> F_f [B, D2, N]（256-d）。
    """

    def __init__(self, in_dim, hidden_dim=256, num_blocks=6, levels=16,
                 fourier_layout='legacy'):
        super().__init__()
        if fourier_layout not in ('legacy', 'pointwise'):
            raise ValueError(f"unknown fourier_layout {fourier_layout!r}")
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.levels = levels
        self.fourier_layout = fourier_layout
        # Fourier 频率：2^[0..L-1] * pi（与 FeatExtractNet/FourierMLP 一致）
        self.register_buffer(
            '_freqs',
            2.0 ** torch.arange(levels, dtype=torch.float32) * math.pi,
            persistent=False)

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim) for _ in range(num_blocks)])

    def forward(self, h0, coords):
        # h0: [B, D0, N]；coords: [B, Cin, N]（原始归一化坐标，供 Fourier）
        B, _, N = h0.shape
        fe = _fourier_sin_cos(coords, self._freqs, self.levels,
                              layout=self.fourier_layout)          # [B, Cin*2L, N]
        x_in = torch.cat([h0, fe], dim=1)          # [B, D0+Cin*2L, N]（默认 128-d）

        xt = x_in.permute(0, 2, 1).reshape(-1, self.in_dim)  # [B*N, in_dim]
        h = F.relu(self.input_proj(xt))                      # [B*N, hidden_dim]
        for block in self.blocks:                            # 逐点残差 MLP
            h = block(h)
        # 还原回点云布局：[B, N, hidden] -> [B, hidden, N]
        return h.reshape(B, N, self.hidden_dim).permute(0, 2, 1)


class Router(nn.Module):
    """采样级软路由：几何签名 -> 两层 MLP -> 标量 logit（U1 §2.2）。

    输入 sig: [B, Dr]，Dr = 5 + Nc（5 个几何统计量 + cond 维度，见
    UnifiedNet._compute_signature 的逐字段说明）。
    归一化：LayerNorm（对每个样本在特征维归一化，无 running state，
    训练/评测行为一致）-> Linear(Dr -> hidden) + ReLU -> Linear(hidden -> 1)。
    alpha = sigmoid(logit / T) 在 UnifiedNet 内计算（需要温度与 branch_mode
    的固定覆盖逻辑），本模块只出 logit。
    """

    def __init__(self, sig_dim, hidden_dim=128):
        super().__init__()
        self.sig_dim = sig_dim
        self.sig_norm = nn.LayerNorm(sig_dim)
        self.fc1 = nn.Linear(sig_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, sig):
        # sig: [B, Dr] -> logit: [B]
        h = F.relu(self.fc1(self.sig_norm(sig)))
        return self.fc2(h).squeeze(-1)


class TaskHead(nn.Module):
    """浅任务头（Conv1d K=1 逐点 MLP，U1 §2.5）。

    HeadT 默认：256 -> 512 -> 256 -> 128 -> 4，中间激活由 act 控制
      （建议 'tanh'，与 FeatExtractNet 末段 tanh 风格一致），带 in_dim->4
      的 shortcut 残差加和。
    HeadA 默认：256 -> 128 -> 1（线性收尾，无 shortcut）。
    hidden_widths 与 in_dim 均可用 width_scale / concat_no_router 调整。
    """

    def __init__(self, in_dim, hidden_widths, out_dim, act='tanh', shortcut=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        widths = (in_dim,) + tuple(hidden_widths) + (out_dim,)

        layers = []
        for i in range(len(widths) - 1):
            layers.append(nn.Conv1d(widths[i], widths[i + 1], kernel_size=1))
            if i < len(widths) - 2:  # 中间层激活；最后一层线性收尾
                layers.append(nn.Tanh() if act == 'tanh' else nn.ReLU())
        self.net = nn.Sequential(*layers)

        # HeadT 的 4-d shortcut（in_dim -> out_dim 直连加和）
        self.shortcut = nn.Conv1d(in_dim, out_dim, kernel_size=1) if shortcut else None

    def forward(self, h):
        # h: [B, in_dim, N] -> [B, out_dim, N]
        out = self.net(h)
        if self.shortcut is not None:
            out = out + self.shortcut(h)
        return out


class UnifiedNet(nn.Module):
    """共享语义空间 + 动态路由统合网络（详见本模块 docstring 与 U1 文档）。"""

    # U1 §3：前 10% epoch 用软 alpha，之后切硬路由。外层据此决定 hard 传参。
    HARD_ROUTER_WARMUP_EPOCH_FRAC = 0.1
    # 外层 router loss 权重（U1 §2.6/§3）：本模块只存常量，不内置 loss。
    ENTROPY_REG_DEFAULT = 1e-3
    LOAD_BALANCE_REG_DEFAULT = 1e-3
    # 路由签名子采样点数上限（U1 §2.2）
    MAX_SIG_POINTS_DEFAULT = 1024

    def __init__(self, input_channels=2, cond_dim=0, thermal_out=4, acoustic_out=1,
                 branch_mode='unified', stem_fourier=False, fourier_levels=16,
                 num_fourier_blocks=6, width_scale=1.0, router_temperature=1.0,
                 entropy_reg=ENTROPY_REG_DEFAULT, load_balance_reg=LOAD_BALANCE_REG_DEFAULT,
                 max_sig_points=MAX_SIG_POINTS_DEFAULT, head_act='tanh',
                 compute_both_heads=True, fourier_layout='legacy'):
        super().__init__()
        if branch_mode not in ('unified', 'branchP_only', 'branchF_only', 'concat_no_router'):
            raise ValueError(
                f"branch_mode must be one of unified/branchP_only/branchF_only/"
                f"concat_no_router, got {branch_mode!r}")
        if fourier_levels < 1:
            raise ValueError('fourier_levels must be >= 1')
        if fourier_layout not in ('legacy', 'pointwise'):
            raise ValueError(f"unknown fourier_layout {fourier_layout!r}")
        if width_scale <= 0:
            raise ValueError('width_scale must be > 0')

        self.input_channels = input_channels
        self.cond_dim = cond_dim
        self.thermal_out = thermal_out
        self.acoustic_out = acoustic_out
        self.branch_mode = branch_mode
        self.stem_fourier = stem_fourier
        self.fourier_levels = fourier_levels
        self.num_fourier_blocks = num_fourier_blocks
        self.width_scale = width_scale
        self.max_sig_points = max_sig_points
        self.head_act = head_act
        self.compute_both_heads = compute_both_heads
        self.fourier_layout = fourier_layout

        # 仅作属性常量暴露给外层（U1 §2.6/§3）；模块内部不计算这些 loss
        self.entropy_reg = float(entropy_reg)
        self.load_balance_reg = float(load_balance_reg)
        self.router_temperature = float(router_temperature)

        # --- 宽度（width_scale 缩放 stem/分支/头；最小 1，保证 >0） ---
        s = lambda v: max(1, int(round(v * width_scale)))
        self.s64 = s(64)
        self.s128 = s(128)
        self.s256 = s(256)
        D0, D1, D2, Dsem = self.s64, self.s128, self.s256, self.s256
        self.dsem = Dsem

        # --- 共享坐标编码 stem（U1 §2.1） ---
        stem_in = input_channels
        if stem_fourier:
            stem_in = input_channels * (1 + 2 * fourier_levels)
            self.register_buffer(
                '_stem_freqs',
                2.0 ** torch.arange(fourier_levels, dtype=torch.float32) * math.pi,
                persistent=False)
        self.stem = nn.Conv1d(stem_in, D0, kernel_size=1)  # stem_in -> 64
        # cond 注入：Linear(Nc -> D0)，无 cond 时为 0
        if cond_dim > 0:
            self.cond_proj = nn.Linear(cond_dim, D0)
        else:
            self.cond_proj = None

        # --- 分支 P（点云路径）与分支 F（Fourier 残差 MLP） ---
        self.branch_p = BranchP(in_dim=D0, mid_dim=D1, high_dim=D2)
        f_branch_in = D0 + input_channels * 2 * fourier_levels  # 默认 64+64=128
        self.branch_f = BranchF(in_dim=f_branch_in, hidden_dim=D2,
                                num_blocks=num_fourier_blocks, levels=fourier_levels,
                                fourier_layout=fourier_layout)

        # --- 融合投影到共享语义空间（U1 §2.5） ---
        self.proj_p = nn.Conv1d(self.branch_p.fused_out, Dsem, kernel_size=1)  # 640->256
        self.proj_f = nn.Conv1d(D2, Dsem, kernel_size=1)                        # 256->256

        # --- 路由（采样级，U1 §2.2） ---
        # 签名 Dr = 5 个几何统计量 + cond_dim（cond 缺失时补零，保持维度一致）
        self.sig_dim = 5 + cond_dim
        self.router = Router(sig_dim=self.sig_dim, hidden_dim=s(128))

        # --- 任务头（in_dim 支持 concat 模式 2*Dsem=512） ---
        head_in = Dsem if branch_mode != 'concat_no_router' else 2 * Dsem
        self.head_t = TaskHead(head_in, (s(512), s(256), s(128)), thermal_out,
                               act=head_act, shortcut=True)
        self.head_a = TaskHead(head_in, (s(128),), acoustic_out,
                               act=head_act, shortcut=False)

        self._init_weights()

    # ------------------------------------------------------------------ #
    # 工具
    # ------------------------------------------------------------------ #
    def _init_weights(self):
        """Kaiming(fan_in, relu) + bias 置零（同 FeatExtractNet 风格）；LayerNorm 保持默认。"""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_router_temperature(self, temperature):
        """外层退火 router 温度 T：alpha = sigmoid(logit / T)。"""
        if temperature <= 0:
            raise ValueError('router_temperature must be > 0')
        self.router_temperature = float(temperature)

    def component_param_counts(self):
        """按组件统计参数量，返回 dict（供冒烟/报告使用）。

        分组：stem（stem 卷积 + cond_proj）、branch_p、branch_f、router、
        fusion_proj（proj_p + proj_f）、head_thermal、head_acoustic、total。
        """
        stem_mods = [self.stem]
        if self.cond_proj is not None:
            stem_mods.append(self.cond_proj)
        groups = {
            'stem': stem_mods,
            'branch_p': [self.branch_p],
            'branch_f': [self.branch_f],
            'router': [self.router],
            'fusion_proj': [self.proj_p, self.proj_f],
            'head_thermal': [self.head_t],
            'head_acoustic': [self.head_a],
        }
        counts = {k: sum(p.numel() for m in mods for p in m.parameters())
                  for k, mods in groups.items()}
        counts['total'] = sum(counts.values())
        return counts

    # ------------------------------------------------------------------ #
    # 路由签名（U1 §2.2）
    # ------------------------------------------------------------------ #
    def _compute_signature(self, x, cond):
        """样本级几何签名 sig: [B, Dr=5+Nc]。

        每个样本：从 x 中均匀子采样 <= max_sig_points 个点（训练随机、
        评测确定性等距），做 chunked 最近邻距离统计避免爆显存/内存
        （CPU/GPU 均可），字段依次为：
          0. log N
          1. log mean_nn   （近邻距离均值对数）
          2. log std_nn    （近邻距离标准差对数）
          3. std_nn / mean_nn  （规则网格 -> 比值小；不规则几何 -> 比值大）
          4. log bbox 对角线  （包围盒（全点 min/max 估计）对角线对数）
          5..5+Nc-1: cond（若存在原样拼接；缺失补零）
        约束：禁止任务名/数据集 id 进入签名（理由见模块 docstring）。
        """
        B = x.shape[0]
        eps = 1e-6
        sig_list = []
        for b in range(B):
            xb = x[b]                              # [Cin, N_b]
            nb = xb.shape[1]

            # 包围盒（用全部点做 min/max，O(N) 便宜且稳定）
            pmin = xb.amin(dim=1)                  # [Cin]
            pmax = xb.amax(dim=1)                  # [Cin]
            bbox_diag = (pmax - pmin).square().sum().sqrt().clamp_min(eps)

            # 子采样索引：训练随机（randperm 无放回）；评测确定性等距
            kb = min(nb, self.max_sig_points)
            if kb < nb:
                if self.training:
                    idx = torch.randperm(nb, device=x.device)[:kb]
                else:
                    step = (nb + kb - 1) // kb
                    idx = torch.arange(0, nb, step, device=x.device)[:kb]
                pts = xb[:, idx].t().contiguous()  # [kb, Cin]
            else:
                pts = xb.t().contiguous()          # [nb, Cin]

            mean_nn, std_nn = self._nn_stats(pts)  # 标量 tensor

            log_n = torch.log(torch.as_tensor(float(nb), device=x.device) + eps)
            scalars = torch.stack([
                log_n,
                torch.log(mean_nn.clamp_min(eps)),
                torch.log(std_nn.clamp_min(eps)),
                std_nn / mean_nn.clamp_min(eps),
                torch.log(bbox_diag),
            ])                                     # [5]

            if self.cond_dim > 0:
                cond_part = cond[b] if cond is not None else torch.zeros(
                    self.cond_dim, device=x.device)
                sig_b = torch.cat([scalars, cond_part])   # [5+Nc]
            else:
                sig_b = scalars                          # [5]
            sig_list.append(sig_b)

        return torch.stack(sig_list)                     # [B, Dr]

    @staticmethod
    def _nn_stats(pts):
        """chunked 最近邻统计（排除自距离）。

        pts: [K, Cin]，K <= max_sig_points（默认 1024）。
        按行分块（chunk=256）算 cdist，避免一次性 [K,K] 全矩阵爆内存；
        返回 (mean_nn, std_nn) 两个标量 tensor。
        """
        K = pts.shape[0]
        if K < 2:
            zero = pts.new_zeros(())
            return zero, zero
        chunk = 256
        nn_min = pts.new_full((K,), float('inf'))
        for s in range(0, K, chunk):
            e = min(s + chunk, K)
            d2 = torch.cdist(pts[s:e], pts)              # [e-s, K]
            rows = torch.arange(e - s, device=pts.device)
            d2[rows, s + rows] = float('inf')            # 排除自身
            nn_min[s:e] = d2.min(dim=1).values
        nn_dist = nn_min.clamp_min(0.0).sqrt()
        return nn_dist.mean(), nn_dist.std()

    # ------------------------------------------------------------------ #
    # 分支前向（返回投影到共享语义空间的逐点特征 [B, Dsem, N]）
    # ------------------------------------------------------------------ #
    def _branch_p_proj(self, h0):
        # h0: [B, D0, N] -> F_p [B, 640, N] -> projP -> [B, Dsem, N]
        return self.proj_p(self.branch_p(h0))

    def _branch_f_proj(self, h0, x):
        # x: [B, Cin, N]（原始坐标给分支 F 的 Fourier）-> F_f -> projF -> [B, Dsem, N]
        return self.proj_f(self.branch_f(h0, x))

    # ------------------------------------------------------------------ #
    # 主前向
    # ------------------------------------------------------------------ #
    def forward(self, x, cond=None, task=None, hard=False):
        """前向契约（U1 §6），详见模块 docstring。

        返回 (out, diag)：
        - out: [B, Cout, N]，task='thermal'(或 None) -> HeadT；'acoustic' -> HeadA；
        - diag: dict（alpha/alpha_soft/alpha_hard/sig/entropy/logit，
          以及 compute_both_heads 时的 out_thermal/out_acoustic）。
        """
        B, Cin, N = x.shape
        if Cin != self.input_channels:
            raise ValueError(
                f'expected input_channels={self.input_channels}, got {Cin}')

        # ---- 1. 共享 stem + cond 注入（U1 §2.1） ----
        if self.stem_fourier:
            x_enc = _fourier_append(x, self._stem_freqs, self.fourier_levels,
                                    layout=self.fourier_layout)
        else:
            x_enc = x                                  # [B, Cin, N]
        h0 = torch.tanh(self.stem(x_enc))              # [B, D0, N]
        if cond is not None:
            if self.cond_proj is None:
                raise ValueError('cond provided but cond_dim=0 in constructor')
            # cond: [B, Nc] -> Linear -> [B, D0, 1] 按样本广播加到 N 维
            h0 = h0 + self.cond_proj(cond).unsqueeze(-1)

        # ---- 2. 几何签名 + 路由（U1 §2.2；输入不含任务名/数据集 id） ----
        sig = self._compute_signature(x, cond)         # [B, Dr]
        logit = self.router(sig)                       # [B]
        alpha_soft = torch.sigmoid(logit / self.router_temperature)  # [B]

        # ---- 3. 分支融合（U1 §2.5 / §2.6 branch_mode） ----
        mode = self.branch_mode
        alpha = alpha_soft
        if mode == 'branchP_only':
            # 固定 alpha=1：只前向分支 P
            h_sem = self._branch_p_proj(h0)            # [B, Dsem, N]
            alpha = torch.ones_like(alpha_soft)
        elif mode == 'branchF_only':
            # 固定 alpha=0：只前向分支 F
            h_sem = self._branch_f_proj(h0, x)         # [B, Dsem, N]
            alpha = torch.zeros_like(alpha_soft)
        elif mode == 'concat_no_router':
            # 去掉路由：拼接 projP 与 projF，头输入 2*Dsem
            p_sem = self._branch_p_proj(h0)            # [B, Dsem, N]
            f_sem = self._branch_f_proj(h0, x)         # [B, Dsem, N]
            h_sem = torch.cat([p_sem, f_sem], dim=1)   # [B, 2*Dsem, N]
            # alpha 不参与融合，仅保留作诊断（固定模式下无梯度）
        else:  # unified
            if hard:
                # 硬路由：逐样本二值选择，只前向被选分支（U1 §3 + 模块 docstring STE）
                alpha_hard = (alpha_soft > 0.5).float()
                p_sel = alpha_hard > 0.5               # [B] bool，True -> 分支 P
                if bool(p_sel.all()):
                    h_sem = self._branch_p_proj(h0)
                elif bool(not p_sel.any()):
                    h_sem = self._branch_f_proj(h0, x)
                else:
                    h_sem = x.new_empty(B, self.dsem, N)
                    h_sem[p_sel] = self._branch_p_proj(h0[p_sel])
                    h_sem[~p_sel] = self._branch_f_proj(h0[~p_sel], x[~p_sel])
                alpha = alpha_soft                    # diag 仍给软值
            else:
                # 软路由：两分支均前向，alpha 凸组合（router 借此接收任务梯度）
                p_sem = self._branch_p_proj(h0)        # [B, Dsem, N]
                f_sem = self._branch_f_proj(h0, x)     # [B, Dsem, N]
                a = alpha_soft.view(B, 1, 1)           # [B,1,1] 广播
                h_sem = a * p_sem + (1 - a) * f_sem    # [B, Dsem, N]

        # ---- 4. 任务头（U1 §2.5；两任务头均计算，compute_both_heads=False
        #        时只算 task 对应头以省大 N 声场的算力） ----
        task = task or 'thermal'
        if self.compute_both_heads or task == 'thermal':
            out_thermal = self.head_t(h_sem)           # [B, thermal_out, N]
        if self.compute_both_heads or task == 'acoustic':
            out_acoustic = self.head_a(h_sem)          # [B, acoustic_out, N]

        if task == 'acoustic':
            out = out_acoustic
        else:
            out = out_thermal

        # ---- 5. 诊断（U1 §6） ----
        alpha_hard = (alpha > 0.5).float()             # [B]（固定模式对应常数）
        eps = 1e-6
        a = alpha.clamp(eps, 1.0 - eps)
        entropy = -(a * a.log() + (1 - a) * (1 - a).log()).mean()  # 标量 tensor
        # 二阶 Bernoulli 熵 H(a)=-a ln a -(1-a)ln(1-a) 的样本均值

        diag = {
            'alpha': alpha,                            # 软路由权重 [B]
            'alpha_soft': alpha,                       # = alpha（U1 §6 别名）
            'alpha_hard': alpha_hard,                  # [B]
            'sig': sig,                                # 路由输入签名 [B, Dr]
            'entropy': entropy,                        # 标量
            'logit': logit,                            # [B]（外层 router loss 用）
            'mode': mode,
        }
        if self.compute_both_heads:
            diag['out_thermal'] = out_thermal
            diag['out_acoustic'] = out_acoustic

        return out, diag
