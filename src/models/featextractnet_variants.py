"""featextractnet_variants.py —— 热场组件消融变体模型（src.models.featextractnet_variants）。

用途与背景
----------
为"真消融表"（实验计划.md §5 Track T3；回复口径 A#3 / B R3）提供基于
FeatExtractNet 结构的受控变体。全部消融/基线均为独立真实训练，不再出现
历史稿件中的 projected/♯ 推算行。本模块只定义模型，训练与评测见
scripts/exp_ablation_thermal.py。

与 FeatExtractNet 的关系
------------------------
默认开关（mode='mlfe', pool_mode='tri', use_cgf=True, use_shortcut=True）下，
本类逐层复刻 src/models/featextractnet.py 的 FeatExtractNet 热场结构
（use_fourier=False, use_spatial_grid=False 路径）：

  输入 [B,Cin,N] -> Conv1d(Cin,64)@tanh -> Conv1d(64,128)@relu
                  -> Conv1d(128,256)@relu
                  -> tri-stat(adaptive max/avg/var@K=32) 全局池化
                     -> Conv1d(3*256,256)@relu -> Conv1d(256,256)@relu
                     -> adaptive-avg-pool(1) -> LayerNorm(256) -> expand 到 N
                  -> 三层 sigmoid CGF 门控逐层融合（gate_i*f_i + (1-gate_i)*compress_i(f_ctx)）
                  -> concat[f_low_fused, f_mid_fused, f_high_fused, f_ctx]
                  -> tanh FC head(704->512->256->128) -> Conv1d(128,Cout)
                  -> + shortcut(704->Cout)

因此 mode='mlfe' 默认开关全开时，模块命名与通道数均与 FeatExtractNet 一致，
state_dict 键名相同（可直接互换 checkpoint），参数量应相等（1,149,512）。

开关语义与对应消融（scripts/exp_ablation_thermal.py 的 --variant）：
  * pool_mode='tri' ：max/avg/var 三统计全局池化（FeatExtractNet 默认）。
  * pool_mode='max' ：去掉 avg/var 统计通道，仅保留 max -> "三统计池化消融"
                      （对应变体 fen_poolmax）。注意这会同时缩减 global_conv1
                      输入通道 3*256 -> 256，参数量减少 131,072。
  * use_cgf=True     ：保留 sigmoid 门控 + compress 逐层融合（FeatExtractNet 默认）。
  * use_cgf=False    ：去掉 gate_*/compress_* 模块，融合特征改为各层局部特征
                       (f_low/f_mid/f_high) 与 global(expand 到 N 的 256-d)
                       直接 concat -> "MLFE-only" 消融（对应变体 fen_nocgf）。
  * use_shortcut=True ：head 输出 + shortcut(f_fused) 残差（FeatExtractNet 默认）。
  * use_shortcut=False：去掉 shortcut 卷积，"head residual shortcut" 消融
                       （对应变体 fen_noshortcut）。
  * mode='mlfe'       ：多尺度层次 64/128/256（FeatExtractNet 主结构）。
  * mode='single128'  ：CGF-only 单尺度参数匹配臂：三层均匀 128-d Conv1d +
                        CGF 门控 + 与历史 CGF†(421,960 参数，见
                        修订工作区/run_three_experiments.py 的 FeatExtractNet_CGF
                        类) 一致的单尺度 head（对应变体 fen_single128）。
                        本分支默认（tri + CGF + shortcut）参数量 = 421,960，
                        与历史 CGF† 完全一致，用于"控制参数量下门控 vs 池化
                        通道"的对照。

初始化约定
----------
所有 Conv1d/Linear 统一 kaiming_normal_(fan_in, relu) + bias 置零（与
baselines/coordinate_mlp.py、历史 FeatExtractNet_CGF 一致），保证 5 个变体
共享同一初始化协议、消融内部干净可比。注意：FeatExtractNet 原实现未显式调
用 init（走 torch 默认 kaiming_uniform），因此本类与 T1 主表运行的权重初值
协议不同（架构/参数量一致），跨运行对比时需在结果表注明这一点。

init_mode（R2-IMPL-01E）：
  * init_mode='legacy_kaiming'（默认，历史行为）：构造后调用 _reset_weights，
    全部 Conv1d/Linear 使用 kaiming_normal_(fan_in, relu) + bias 置零；模块创建
    顺序保持历史不变（mlfe 分支 output_conv 先于 shortcut）。历史结果保持不变。
  * init_mode='main_default'：不调用 _reset_weights，保留 nn.Conv1d/Linear/
    LayerNorm 的 PyTorch 默认初始化——即真实 FeatExtractNet 主模型实际使用的
    初值协议。为使同一 torch seed 下初始 state_dict 与 FeatExtractNet 逐位一致，
    mlfe 分支在 use_shortcut=True 时把 shortcut 的创建提前到 output_conv 之前，
    以复刻 FeatExtractNet 的模块（及 RNG 消耗）创建顺序。single128 分支沿用其
    自身既有模块顺序，仅跳过显式 reset。
  两种模式均不改变参数形状、键名、前向计算路径与参数量。

forward(x: [B, Cin, N]) -> [B, Cout, N]，与 FeatExtractNet 输入/输出约定一致。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatExtractNetVariants(nn.Module):
    """基于 FeatExtractNet 的热场组件消融变体（开关重写的 forward）。

    Args:
        input_channels: 输入坐标通道数（热场 2）。
        output_channels: 输出物理量通道数（热场 4：u,v,p,T）。
        scaling: 宽度缩放（同 FeatExtractNet，scale=1.0 时为名义宽度）。
        pool_mode: 'tri'=max/avg/var 三统计池化；'max'=仅 max。
        use_cgf: True=保留 sigmoid 门控+compress 融合；False=局部特征直接 concat。
        use_shortcut: True=head 残差 shortcut；False=去掉 shortcut 卷积。
        mode: 'mlfe'=64/128/256 多尺度层次；'single128'=128 三层均匀单尺度。
        init_mode: 'legacy_kaiming'=历史 kaiming_normal 显式初始化（默认）；
            'main_default'=PyTorch 默认初始化（=FeatExtractNet 主模型实际初值协议）。
    """

    POOL_K = 32
    INIT_MODES = ('legacy_kaiming', 'main_default')

    def __init__(self, input_channels=2, output_channels=4, scaling=1.0,
                 pool_mode='tri', use_cgf=True, use_shortcut=True,
                 mode='mlfe', context_mode='legacy_index_bins',
                 init_mode='legacy_kaiming'):
        super().__init__()
        if mode not in ('mlfe', 'single128'):
            raise ValueError(f'mode must be mlfe|single128, got {mode!r}')
        if pool_mode not in ('tri', 'max'):
            raise ValueError(f'pool_mode must be tri|max, got {pool_mode!r}')
        if context_mode not in ('legacy_index_bins', 'symmetric'):
            raise ValueError(
                f'context_mode must be legacy_index_bins|symmetric, '
                f'got {context_mode!r}')
        if init_mode not in self.INIT_MODES:
            raise ValueError(
                f'init_mode must be legacy_kaiming|main_default, '
                f'got {init_mode!r}')
        if mode == 'single128' and not use_cgf:
            # CGF-only 臂的语义前提是"单尺度 + 门控融合"；允许置 False 但
            # 会退化为单尺度局部特征 + global concat（非登记消融臂）。
            pass
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.scaling = float(scaling)
        self.pool_mode = pool_mode
        self.context_mode = context_mode
        self.use_cgf = bool(use_cgf)
        self.use_shortcut = bool(use_shortcut)
        self.mode = mode
        self.init_mode = init_mode

        s = lambda x: int(x * scaling)
        self.s64 = s(64)
        self.s128 = s(128)
        self.s256 = s(256)
        self.s512 = s(512)

        if mode == 'mlfe':
            self._build_mlfe(s)
        else:
            self._build_single128(s)
        if init_mode == 'legacy_kaiming':
            self._reset_weights()

    # ------------------------------------------------------------------ #
    # 构建
    # ------------------------------------------------------------------ #
    def _build_mlfe(self, s):
        """64/128/256 三尺度 + 全局上下文 + 可选 CGF（FeatExtractNet 结构）。"""
        self.input_fc = nn.Conv1d(self.input_channels, self.s64, 1)
        self.conv_mid = nn.Conv1d(self.s64, self.s128, 1)
        self.conv_high = nn.Conv1d(self.s128, self.s256, 1)

        pool_ch = (3 if self.pool_mode == 'tri' else 1) * self.s256
        self.global_conv1 = nn.Conv1d(pool_ch, self.s256, 1)
        self.global_conv2 = nn.Conv1d(self.s256, self.s256, 1)
        self.global_lnorm = nn.LayerNorm(self.s256)

        if self.use_cgf:
            self.gate_low = nn.Conv1d(self.s64 + self.s256, self.s64, 1)
            self.gate_mid = nn.Conv1d(self.s128 + self.s256, self.s128, 1)
            self.gate_high = nn.Conv1d(self.s256 + self.s256, self.s256, 1)
            self.compress_low = nn.Conv1d(self.s256, self.s64, 1)
            self.compress_mid = nn.Conv1d(self.s256, self.s128, 1)
            self.compress_high = nn.Conv1d(self.s256, self.s256, 1)

        self.fused_dim = self.s64 + self.s128 + self.s256 + self.s256
        self.final_fc1 = nn.Conv1d(self.fused_dim, self.s512, 1)
        self.final_fc2 = nn.Conv1d(self.s512, self.s256, 1)
        self.final_fc3 = nn.Conv1d(self.s256, self.s128, 1)
        if self.use_shortcut and self.init_mode == 'main_default':
            # main_default：shortcut 先于 output_conv，复刻 FeatExtractNet 的
            # 模块创建顺序（同一 torch seed 下初始 state_dict 逐位一致）。
            self.shortcut = nn.Conv1d(self.fused_dim, self.output_channels, 1)
            self.output_conv = nn.Conv1d(self.s128, self.output_channels, 1)
        else:
            # legacy_kaiming（历史顺序）与无 shortcut 情形：output_conv 在前。
            self.output_conv = nn.Conv1d(self.s128, self.output_channels, 1)
            if self.use_shortcut:
                self.shortcut = nn.Conv1d(self.fused_dim, self.output_channels, 1)

    def _build_single128(self, s):
        """单尺度 128 三层均匀 + 全局上下文 + CGF（历史 CGF† 移植，421,960 参数）。"""
        self.conv1 = nn.Conv1d(self.input_channels, self.s128, 1)
        self.conv2 = nn.Conv1d(self.s128, self.s128, 1)
        self.conv3 = nn.Conv1d(self.s128, self.s128, 1)

        pool_ch = (3 if self.pool_mode == 'tri' else 1) * self.s128
        self.gconv1 = nn.Conv1d(pool_ch, self.s256, 1)
        self.gconv2 = nn.Conv1d(self.s256, self.s256, 1)
        self.glnorm = nn.LayerNorm(self.s256)

        if self.use_cgf:
            self.gate_layer = nn.Conv1d(self.s128 + self.s256, self.s128, 1)
            self.compress = nn.Conv1d(self.s256, self.s128, 1)

        self.fused_dim = self.s128 + self.s256
        self.fc1 = nn.Conv1d(self.fused_dim, self.s256, 1)
        self.fc2 = nn.Conv1d(self.s256, self.s128, 1)
        self.fc3 = nn.Conv1d(self.s128, self.s64, 1)
        self.output = nn.Conv1d(self.s64, self.output_channels, 1)
        if self.use_shortcut:
            self.shortcut = nn.Conv1d(self.fused_dim, self.output_channels, 1)

    def _reset_weights(self):
        """Conv1d/Linear: kaiming_normal_(fan_in, relu) + bias 置零（5 变体统一）。"""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ #
    # 全局池化上下文（tri/max 两档，K=min(32, N)）
    # ------------------------------------------------------------------ #
    def _global_pool_context(self, f_high, g_conv1, g_conv2, g_lnorm):
        """max(/avg/var) 全局统计 -> 两层 Conv1d -> 1x1 向量 -> LayerNorm -> expand(N)。

        与 FeatExtractNet._global_pool_context 一致；pool_mode='max' 时 g_cat
        仅含 max 通道（无 avg/var），global_conv1 输入通道相应为 256 而非 768。
        """
        B, C, N = f_high.shape
        K = 1 if self.context_mode == 'symmetric' else min(self.POOL_K, N)
        g_max = F.adaptive_max_pool1d(f_high, K)
        if self.pool_mode == 'tri':
            g_avg = F.adaptive_avg_pool1d(f_high, K)
            g_var = torch.var(f_high, dim=2, keepdim=True).expand(-1, -1, K)
            g_cat = torch.cat([g_max, g_avg, g_var], dim=1)
        else:
            g_cat = g_max

        g = F.relu(g_conv1(g_cat))
        g = F.relu(g_conv2(g))
        f_global = F.adaptive_avg_pool1d(g, 1).squeeze(-1)
        f_global_normed = g_lnorm(f_global)
        return f_global_normed.unsqueeze(-1).expand(-1, -1, N)

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(self, x):
        if self.mode == 'single128':
            return self._forward_single128(x)
        return self._forward_mlfe(x)

    def _forward_mlfe(self, x):
        """64/128/256 多尺度 + 全局上下文 + 可选 CGF（FeatExtractNet 默认路径）。"""
        f_low = torch.tanh(self.input_fc(x))
        f_mid = F.relu(self.conv_mid(f_low))
        f_high = F.relu(self.conv_high(f_mid))

        f_ctx = self._global_pool_context(
            f_high, self.global_conv1, self.global_conv2, self.global_lnorm)

        if self.use_cgf:
            cat_low = torch.cat([f_low, f_ctx], dim=1)
            gate_low = torch.sigmoid(self.gate_low(cat_low))
            f_low_fused = gate_low * f_low + (1 - gate_low) * self.compress_low(f_ctx)

            cat_mid = torch.cat([f_mid, f_ctx], dim=1)
            gate_mid = torch.sigmoid(self.gate_mid(cat_mid))
            f_mid_fused = gate_mid * f_mid + (1 - gate_mid) * self.compress_mid(f_ctx)

            cat_high = torch.cat([f_high, f_ctx], dim=1)
            gate_high = torch.sigmoid(self.gate_high(cat_high))
            f_high_fused = gate_high * f_high + (1 - gate_high) * self.compress_high(f_ctx)
        else:
            # MLFE-only：局部特征不经门控，直接与 global 拼接
            f_low_fused = f_low
            f_mid_fused = f_mid
            f_high_fused = f_high

        f_fused = torch.cat([f_low_fused, f_mid_fused, f_high_fused, f_ctx], dim=1)

        out = torch.tanh(self.final_fc1(f_fused))
        out = torch.tanh(self.final_fc2(out))
        out = torch.tanh(self.final_fc3(out))
        output = self.output_conv(out)

        if self.use_shortcut:
            output = output + self.shortcut(f_fused)
        return output

    def _forward_single128(self, x):
        """单尺度 128 三层 + 全局上下文 + CGF（历史 CGF† 结构移植）。"""
        f1 = torch.tanh(self.conv1(x))
        f2 = F.relu(self.conv2(f1))
        f3 = F.relu(self.conv3(f2))

        f_ctx = self._global_pool_context(f3, self.gconv1, self.gconv2, self.glnorm)

        if self.use_cgf:
            cat = torch.cat([f3, f_ctx], dim=1)
            gate = torch.sigmoid(self.gate_layer(cat))
            fused = gate * f3 + (1 - gate) * self.compress(f_ctx)
        else:
            fused = f3
        f_all = torch.cat([fused, f_ctx], dim=1)

        out = torch.tanh(self.fc1(f_all))
        out = torch.tanh(self.fc2(out))
        out = torch.tanh(self.fc3(out))
        output = self.output(out)

        if self.use_shortcut:
            output = output + self.shortcut(f_all)
        return output

    # ------------------------------------------------------------------ #
    # 描述
    # ------------------------------------------------------------------ #
    @property
    def arch_label(self):
        """结构描述串（供日志 / 结果表 label 使用）。"""
        scale = ('64/128/256' if self.mode == 'mlfe' else '128x3 (CGF-only)')
        pool = 'tri(max/avg/var)' if self.pool_mode == 'tri' else 'max-only'
        return (f'FeatExtractNetVariants(mode={self.mode}, {scale}, '
                f'pool={pool}, cgf={self.use_cgf}, '
                f'shortcut={self.use_shortcut}, scaling={self.scaling})')

    def __repr__(self):
        n = sum(p.numel() for p in self.parameters())
        return (f'{self.__class__.__name__}({self.arch_label}, '
                f'params={n:,})')
