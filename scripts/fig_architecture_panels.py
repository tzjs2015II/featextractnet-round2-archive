# R2: corrected symmetric context; explicit thermal four-output head; existing source layout retained.
"""fig_architecture_panels.py —— Fig. 1：拆成四张可读子图 (a)-(d) + 共用图例。

用途
----
审稿附加项要求把 Fig. 1 拆成更小、更可读的子图（原文："Fig. 1 could be broken down
into smaller sub-figures for better visibility. It is difficult to see some of the text
in its current form."）。原图是一整张 1975 px 宽的长图，缩到版心后文字只有 2-4 pt。
本脚本按正文 §3.1-§3.5 重画为 2x2 四面板 + 共用图例，各面板独立可读。

内容口径（严格对应 paper/review_bc.tex）
------------------------------------
(a) MLFE      ：X∈R^(N×2) → Conv1d(2→64)+Tanh → Conv1d(64→128)+ReLU → Conv1d(128→256)+ReLU
                输出 f_low∈R^(64×N)、f_mid∈R^(128×N)、f_high∈R^(256×N)          (§3.2)
(b) 全局聚合   ：f_high → MaxPool1d/AvgPool1d/Var → [f_max;f_avg;f_var]∈R^(768×1)
                → ReLU(768→256) → ReLU(256→256) → LayerNorm → g∈R^256                       (§3.3)
                注：正文 §3.3 明确"条件注入不用于热场实例"，故不画条件分支。
(c) CGF       ：[f_l ; f_global^(expand)] → σ(Conv1d c_l+256 → c_l) = g_l；
                f_l^fused = g_l ⊙ f_l + (1-g_l) ⊙ W_c f_global^(expand)
                逐层 c_l = 64 / 128 / 256（对应 320→64 / 384→128 / 512→256）    (§3.4)
(d) 预测头     ：f_fused∈R^(704×N) → FC 512 → 256 → 128 → C，并联 Shortcut
                Conv1d(704→4)；热场输出(u,v,p,T)，声学另见(e)                           (§3.5)
(e) 声学实例   ：整幅宽度单行主链（热场实例之外的第二个域自适应实例）
                (x,y) → Fourier 编码 sin/cos(2^0π..2^15π) → 与原始坐标拼接(66 通道)
                → Linear 68→256 → 6 × ResidualBlock(256) → Linear 256→1 → p_total；
                条件向量 [f/500, Ma] 沿点数复制为 2 个常量通道后拼接进 Concat      (§3.5)
                注：正文明确条件注入是 concatenation（常量通道拼接），不是 broadcast
                相加；残差块为 LayerNorm→Linear→ReLU 两遍 + skip。

样式
----
沿用原图（修订工作区/架构图.drawio）的实测样式 token：模块色 #e6f3ff / #cce3ff /
#e6ffe6(描边 #82b366) / #fff2e6，描边 #333333、线宽 1.5，算子用椭圆，红色 #FF0000
虚线标注 "Innovation N"，实心箭头，四列图例表。标签一律单行（双行标签会互相叠）。

输出
----
    <out>/fig_architecture.png / .pdf / .svg   （2x2 四面板 + 声学面板 (e) + 图例）

用法
----
    python3 scripts/fig_architecture_panels.py --out ../../paper/bc_img

自检
----
逐面板检查：越出画布、文字互叠、箭头穿字、文字骑框边线。任一非零即返回非零退出码。
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse

matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'stix'

C_LOCAL = '#e6f3ff'
C_GLOBAL = '#cce3ff'
C_FUSE = '#e6ffe6'
C_GATE = '#fff2e6'
S_FUSE = '#82b366'
EDGE = '#333333'
WARN = '#FF0000'
GREY = '#555555'

SYM_FS = 9.0       # 框内单行标签
ANNO_FS = 8.0      # 框外注释
TITLE_FS = 11.0    # 面板标题
INNO_FS = 8.0      # 红色 Innovation 副标
EQ_FS = 8.5        # 等式条
LEG_FS = 8.0
SHRINK = 0.3       # 让箭尖真正贴住目标框（此前悬空 2-4.8 pt）
HEAD_PX = 7.0      # 自检时扣除箭头头部包围盒的虚胖（mutation_scale/2）

# 面板 (e) 为整幅宽度（横向约两倍），y 轴范围按行高比例缩小，以保持单位长度一致
E_YMAX = 26.0      # 面板 (e) 的 y 上限
E_YTOP = 24.5      # (e) 标题基线
E_CHAIN_Y = 12.5   # (e) 主链所在行
E_COND_Y = 2.5     # (e) 条件注入支路所在行


def box(ax, x, y, w, h, label, fc='#ffffff', sc=EDGE, fs=SYM_FS, dashed=False,
        zorder=3):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.25,rounding_size=0.9',
        linewidth=1.5, edgecolor=sc, facecolor=fc,
        linestyle=(0, (4, 2)) if dashed else 'solid', zorder=zorder))
    ax.text(x + w / 2, y + h / 2, label, ha='center', va='center',
            fontsize=fs, zorder=zorder + 1, color='#111111')


def arrow(ax, p0, p1, dashed=False, color=EDGE, lw=1.5, shrink=SHRINK):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle='-|>', mutation_scale=11, linewidth=lw,
        color=color, zorder=6, shrinkA=shrink, shrinkB=shrink,
        linestyle=(0, (4, 2)) if dashed else 'solid'))


def seg(ax, p0, p1, dashed=True, color=EDGE, lw=1.4):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle='-', mutation_scale=1, linewidth=lw,
        color=color, zorder=6, shrinkA=0, shrinkB=0,
        linestyle=(0, (4, 2)) if dashed else 'solid'))


def panel_head(ax, title, innovation=None, ytop=60.0):
    ax.text(0, ytop, title, fontsize=TITLE_FS, fontweight='bold',
            ha='left', va='center', color='#111111')
    if innovation:
        ax.text(0, ytop - 4.6, innovation, fontsize=INNO_FS, ha='left',
                va='center', color=WARN, fontweight='bold')


def callout(ax, x, y, w, h):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.3,rounding_size=1.4',
        linewidth=1.6, edgecolor=WARN, facecolor='none',
        linestyle=(0, (5, 3)), zorder=1))


def panel_a(ax):
    panel_head(ax, '(a) Multi-level local features (MLFE)',
               'Shared pointwise feature hierarchy')
    callout(ax, 0.5, 8.5, 47.5, 36)
    box(ax, 2, 45, 45, 7.5, r'Input $X \in \mathbb{R}^{N \times 2}$')
    box(ax, 2, 33, 45, 7.5, r'Conv1d $2 \rightarrow 64$')
    box(ax, 2, 21, 45, 7.5, r'Conv1d $64 \rightarrow 128$')
    box(ax, 2, 9, 45, 7.5, r'Conv1d $128 \rightarrow 256$')
    for y0, y1 in ((45, 40.8), (33, 28.8), (21, 16.8)):
        arrow(ax, (24.5, y0 - 0.2), (24.5, y1 + 0.2))
    for y, sym in ((33, r'$f_{\mathrm{low}}$ ($64 \times N$)'),
                   (21, r'$f_{\mathrm{mid}}$ ($128 \times N$)'),
                   (9, r'$f_{\mathrm{high}}$ ($256 \times N$)')):
        box(ax, 60, y, 40, 7.5, sym, C_LOCAL)
        arrow(ax, (47.5, y + 3.75), (59.5, y + 3.75), dashed=True)
    ax.text(2, 3.5, 'activations: Tanh $\\rightarrow$ ReLU $\\rightarrow$ ReLU',
            fontsize=ANNO_FS, ha='left', va='center', color=GREY)


def panel_b(ax):
    panel_head(ax, '(b) Symmetric global context',
               'Statistics over all support points')
    box(ax, 2, 46, 96, 7.5, r'$f_{\mathrm{high}}$ ($256\times N$)', C_LOCAL)
    for x, label in ((2, 'Max over N'), (35, 'Mean over N'), (68, 'Variance')):
        box(ax, x, 34, 30, 7.5, label, C_GLOBAL)
        arrow(ax, (x + 15, 45.8), (x + 15, 41.8))
    box(ax, 2, 23, 96, 7.5,
        r'$[m;a;v]\in\mathbb{R}^{768\times1}$', C_GLOBAL)
    for x in (17, 50, 83):
        arrow(ax, (x, 33.8), (x, 30.8))
    box(ax, 2, 12, 46, 7.5, r'$768\to256$ + ReLU', C_GLOBAL, fs=8.5)
    box(ax, 52, 12, 46, 7.5, r'$256\to256$ + ReLU', C_GLOBAL, fs=8.5)
    arrow(ax, (25, 22.8), (25, 19.8))
    arrow(ax, (48.3, 15.75), (51.7, 15.75))
    box(ax, 2, 1, 96, 7.5, r'LayerNorm $\to g\in\mathbb{R}^{256}$', C_GLOBAL)
    arrow(ax, (75, 11.8), (75, 8.8))


def panel_c(ax):
    panel_head(ax, '(c) Cross-layer gated fusion (CGF)',
               'Learned pointwise feature weights')
    callout(ax, 18.5, 36.0, 81.5, 14)
    box(ax, 0, 40, 18, 7.5, r'$f_l$', C_LOCAL)
    box(ax, 24, 40, 40, 7.5, r'Concat $\rightarrow$ map', C_FUSE, S_FUSE)
    ax.add_patch(Ellipse((72.5, 43.75), 7, 10, linewidth=1.5, edgecolor=EDGE,
                         facecolor=C_GATE, zorder=3))
    ax.text(72.5, 43.75, 'σ', ha='center', va='center', fontsize=11.5, zorder=4,
            color='#111111')
    box(ax, 84, 40, 16, 7.5, r'$g_l$', C_GATE)
    arrow(ax, (18.5, 43.75), (23.5, 43.75))
    arrow(ax, (64.5, 43.75), (68.6, 43.75))
    arrow(ax, (76.4, 43.75), (83.5, 43.75))
    box(ax, 24, 23.5, 44, 8.5, r'$f_{\mathrm{global}}^{\mathrm{expand}}$'
                            r' ($256 \times N$)', C_GLOBAL)
    arrow(ax, (46, 32.2), (46, 39.7))
    ax.add_patch(FancyBboxPatch(
        (0, 1), 100, 18, boxstyle='round,pad=0.3,rounding_size=0.8',
        linewidth=1.0, edgecolor='#999999', facecolor='#f5f5f5',
        linestyle=(0, (4, 2)), zorder=1))
    ax.text(50, 13.3, r'$g_l = \sigma(\mathrm{Conv1d}(\,[\,f_l\,;\,'
                      r'f_{\mathrm{global}}^{\mathrm{expand}}\,]))$',
            ha='center', va='center', fontsize=EQ_FS, color='#111111', zorder=2)
    ax.text(50, 4.7, r'$f_l^{\mathrm{fused}} = g_l \odot f_l + (1-g_l) \odot'
                     r' W_c\, f_{\mathrm{global}}^{\mathrm{expand}}$',
            ha='center', va='center', fontsize=EQ_FS, color='#111111', zorder=2)


def panel_d(ax):
    panel_head(ax, '(d) Thermal prediction head')
    box(ax, 2, 45, 96, 7.5, r'$F$ ($704\times N$)', C_FUSE, S_FUSE)
    box(ax, 2, 30, 96, 7.5, r'MLP: $704\to512\to256\to128\to4$')
    arrow(ax, (50, 44.8), (50, 37.8))
    ax.text(2, 23, 'tanh after the three hidden layers', fontsize=ANNO_FS,
            ha='left', va='center', color=GREY)
    box(ax, 2, 5, 60, 7.5, r'Linear shortcut: $704\to4$', C_GATE, dashed=True)
    seg(ax, (1.7, 48.75), (0.4, 48.75))
    seg(ax, (0.4, 48.75), (0.4, 8.75))
    arrow(ax, (0.4, 8.75), (1.7, 8.75), dashed=True)
    ax.add_patch(Ellipse((80, 8.75), 7, 9, linewidth=1.5, edgecolor=EDGE,
                         facecolor=C_GATE, zorder=3))
    ax.text(80, 8.75, r'$\oplus$', ha='center', va='center', fontsize=11,
            zorder=4, color='#111111')
    arrow(ax, (80, 29.8), (80, 13.5))
    arrow(ax, (62.5, 8.75), (76.5, 8.75), dashed=True)
    box(ax, 90, 5, 10, 7.5, r'$\widehat{Y}$', C_GATE)
    arrow(ax, (83.5, 8.75), (89.5, 8.75))
    ax.text(2, 16, r'Output: $(u,v,p,T)$, shape $4\times N$',
            fontsize=ANNO_FS, ha='left', va='center', color=GREY)


def panel_e(ax):
    """(e) 声学实例：条件化 Fourier 特征残差 MLP（对应正文 §3.5 末段）。

    与正文逐句对应的四个要点：
      * 6 个残差块、256 隐维、16 个 Fourier 频带 $2^0\\pi..2^{15}\\pi$；
      * 原始 $(x,y)$ 与 sin/cos 展开**拼接**（共 66 通道），故 stem 输入为
        $66+2=68$ 通道；
      * 条件向量 $[f/500,\\mathrm{Ma}]$ 沿点数复制为 2 个**常量通道**后拼接
        （实现是 concatenation，不是 broadcast 相加）；
      * 每个残差块为 LayerNorm→Linear→ReLU 两遍 + 一条 skip。
    """
    panel_head(ax, '(e) Acoustic instance: conditioned Fourier feature MLP',
               ytop=E_YTOP)
    chain = [(0, 16, r'$(x,y)$', None),
             (19, 30, 'Fourier enc.', None),
             (52, 22, 'Concat', None),
             (77, 34, r'Linear $68\to256$', None),
             (114, 30, r'$6\times$ Residual', None),
             (147, 30, r'Linear $256\to1$', None),
             (180, 18, r'$p_{\mathrm{total}}$', C_GATE)]
    for x, w, label, fc in chain:
        box(ax, x, E_CHAIN_Y, w, 7.5, label, fc if fc else '#ffffff')
    for (x, w, *_), (x2, *__) in zip(chain, chain[1:]):
        arrow(ax, (x + w + 0.5, E_CHAIN_Y + 3.75), (x2 - 0.5, E_CHAIN_Y + 3.75))
    # 条件注入支路：拼接为常量通道后进入 Concat 框
    box(ax, 24, E_COND_Y, 30, 7, r'$[f/500,\mathrm{Ma}]$')
    box(ax, 59, E_COND_Y, 46, 7, r'replicate along $N$')
    arrow(ax, (54.5, E_COND_Y + 3.5), (58.5, E_COND_Y + 3.5))
    arrow(ax, (62, E_COND_Y + 7.2), (62, E_CHAIN_Y - 0.2))
    ax.text(106, E_COND_Y + 3.5, r'each block: LN $\rightarrow$ Linear '
                                 r'$\rightarrow$ ReLU ($\times2$) $+$ skip',
            fontsize=ANNO_FS, ha='left', va='center', color=GREY)


def legend_axis(fig, spec):
    ax = fig.add_subplot(spec)
    ax.axis('off')
    rows = [
        ['Symbol', 'Meaning', 'Colour', 'Feature type'],
        [r'$[\,\cdot\,;\,\cdot\,]$', 'Tensor concatenation', 'Light blue',
         'local features'],
        [r'$\sigma$', 'Sigmoid activation', 'Blue', 'global descriptor'],
        [r'$\odot$', 'Element-wise product', 'Light green',
         'gated-fusion features'],
        [r'$\oplus$', 'Element-wise addition', 'Light orange',
         'gate / output'],
    ]
    tab = ax.table(cellText=rows, loc='center', cellLoc='left',
                   bbox=[0.0, 0.14, 1.0, 0.82])
    tab.auto_set_font_size(False)
    tab.set_fontsize(LEG_FS)
    swatch = {1: C_LOCAL, 2: C_GLOBAL, 3: C_FUSE, 4: C_GATE}
    ax.text(0.0, 0.055, 'Dashed arrows: shared or residual paths. '
            'Red dashed boxes: changes relative to PointNet.',
            transform=ax.transAxes, fontsize=LEG_FS, ha='left', va='center',
            color=GREY)
    for (r, c), cell in tab.get_celld().items():
        cell.set_linewidth(0.6)
        cell.set_edgecolor('#cccccc')
        if r == 0:
            cell.set_facecolor('#f2f2f2')
            cell.set_text_props(fontweight='bold')
        elif c == 2 and r in swatch:
            cell.set_facecolor(swatch[r])
        else:
            cell.set_facecolor('#ffffff')


def audit(fig, axes):
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    fb = fig.bbox
    out, issues = [], []

    def near(a, b, pad=1.0):
        return not (a.x1 < b.x0 + pad or b.x1 < a.x0 + pad or
                    a.y1 < b.y0 + pad or b.y1 < a.y0 + pad)

    for tag, ax in axes:
        for t in ax.texts:
            b = t.get_window_extent(rend)
            if b.x0 < fb.x0 - 0.5 or b.x1 > fb.x1 + 0.5 or \
                    b.y0 < fb.y0 - 0.5 or b.y1 > fb.y1 + 0.5:
                out.append(f'{tag}: text "{t.get_text()[:40]}"')
        texts = [(t.get_text()[:36], t.get_window_extent(rend))
                 for t in ax.texts]
        arrows = [p.get_window_extent(rend) for p in ax.patches
                  if isinstance(p, FancyArrowPatch)]
        shapes = [p.get_window_extent(rend) for p in ax.patches
                  if not isinstance(p, FancyArrowPatch)]
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if near(texts[i][1], texts[j][1]):
                    issues.append(f'{tag}: text/text overlap '
                                  f'"{texts[i][0]}" <-> "{texts[j][0]}"')
        for label, tb in texts:
            for ab in arrows:
                # 箭头包围盒把三角头也算进去，会虚胖出框外；先按头长的 1/2 内缩再判重叠
                sh = type(ab).from_extents(ab.x0 + HEAD_PX, ab.y0 + HEAD_PX,
                                           ab.x1 - HEAD_PX, ab.y1 - HEAD_PX)
                if near(tb, sh):
                    issues.append(f'{tag}: arrow crosses text "{label}"')
            for sb in shapes:
                if near(tb, sb) and not (
                        tb.x0 >= sb.x0 and tb.x1 <= sb.x1 and
                        tb.y0 >= sb.y0 and tb.y1 <= sb.y1):
                    issues.append(f'{tag}: text "{label}" straddles a shape')
    # 箭头端点必须落在某个图形上（悬空箭头是最隐蔽的一类错：画在框外还看不出来）
    shapes_all = []
    for tag, ax in axes:
        for p in ax.patches:
            if not isinstance(p, FancyArrowPatch):
                shapes_all.append(p.get_window_extent(rend).expanded(1.10, 1.25))
    ends = []
    for tag, ax in axes:
        for p in ax.patches:
            if isinstance(p, FancyArrowPatch):
                pa, pb = p._posA_posB            # 存的是数据坐标
                tr = p.get_transform()
                ends.append((tag, tr.transform(pa), tr.transform(pb)))
    for tag, pa, pb in ends:
        for which, pt in (('source', pa), ('target', pb)):
            ok = any(sb.contains(pt[0], pt[1]) for sb in shapes_all)
            if not ok:
                for tag2, pa2, pb2 in ends:
                    if pa2 is pa and pb2 is pb:
                        continue
                    if ((abs(pa2[0] - pt[0]) < 6 and abs(pa2[1] - pt[1]) < 6) or
                            (abs(pb2[0] - pt[0]) < 6 and
                             abs(pb2[1] - pt[1]) < 6)):
                        ok = True
                        break
            if not ok:
                issues.append(f'{tag}: floating arrow {which} end at '
                              f'({pt[0]:.1f}, {pt[1]:.1f})')
    # 跨面板：一个面板的文字溢出到相邻面板（版面上最刺眼的一类错）
    alltexts = []
    for tag, ax in axes:
        for t in ax.texts:
            alltexts.append((tag, t.get_text()[:32], t.get_window_extent(rend)))
    for i in range(len(alltexts)):
        for j in range(i + 1, len(alltexts)):
            if alltexts[i][0] != alltexts[j][0] and near(alltexts[i][2],
                                                       alltexts[j][2]):
                issues.append(f'cross-panel: {alltexts[i][0]}:"{alltexts[i][1]}"'
                              f' <-> {alltexts[j][0]}:"{alltexts[j][1]}"')
    print(f'[audit] canvas {fb.width:.0f}x{fb.height:.0f} px | '
          f'out-of-canvas {len(out)} | overlap/cross/straddle {len(issues)}')
    for s in out + issues:
        print('   [WARN]', s)
    return len(out) + len(issues)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', 'figures', 'round2_architecture'))
    ap.add_argument('--dpi', type=int, default=300)
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    # 6.3 in = 目标期刊单栏版心宽；正文用 \includegraphics[width=\textwidth] 1:1 排版，
    # 因此这里的字号就是印刷字号（8-8.5 pt），不会再被缩小。
    # 高度按行高比例放大，使各面板的"单位长度"与四面板版本保持一致。
    fig = plt.figure(figsize=(6.3, 6.12))
    spec = fig.add_gridspec(4, 2, height_ratios=[1.0, 1.0, 0.42, 0.40],
                            hspace=0.14, wspace=0.10)
    axes = []
    for (fun, r, c, name) in ((panel_a, 0, 0, 'a'), (panel_b, 0, 1, 'b'),
                              (panel_c, 1, 0, 'c'), (panel_d, 1, 1, 'd')):
        ax = fig.add_subplot(spec[r, c])
        ax.set_xlim(-1, 101)
        ax.set_ylim(0, 62)
        ax.axis('off')
        fun(ax)
        axes.append((name, ax))
    ax_e = fig.add_subplot(spec[2, :])          # (e) 整幅宽度，横向约两倍
    ax_e.set_xlim(-1, 201)
    ax_e.set_ylim(0, E_YMAX)
    ax_e.axis('off')
    panel_e(ax_e)
    axes.append(('e', ax_e))
    legend_axis(fig, spec[3, :])
    fig.subplots_adjust(left=0.008, right=0.995, top=0.995, bottom=0.01)

    nbad = audit(fig, axes)
    for ext, kw in (('png', dict(dpi=args.dpi)), ('pdf', {}), ('svg', {})):
        fp = os.path.join(out, f'fig_architecture.{ext}')
        fig.savefig(fp, **kw)
        print('saved', fp)
    plt.close(fig)
    return 1 if nbad else 0


if __name__ == '__main__':
    raise SystemExit(main())
