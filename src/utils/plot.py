import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

FIELD_LABELS = {
    'u': '$u$ (m/s)',
    'v': '$v$ (m/s)',
    'p': '$p$ (Pa)',
    'T': '$T$ (K)',
    'p_total': '$p_{\\mathrm{total}}$ (Pa)',
    'p_scat': '$p_{\\mathrm{scat}}$ (Pa)',
}
FIELD_LATEX = {
    'u': '$u$',
    'v': '$v$',
    'p': '$p$',
    'T': '$T$',
    'p_total': '$p_{\\mathrm{total}}$',
    'p_scat': '$p_{\\mathrm{scat}}$',
}
FIELD_CMAP = {
    'u': 'jet', 'v': 'jet', 'p': 'jet', 'T': 'jet',
    'p_total': 'jet', 'p_scat': 'jet',
}


def _get_label(name):
    return FIELD_LABELS.get(name, name)


def _get_latex(name):
    return FIELD_LATEX.get(name, name)


def _get_cmap(name):
    return FIELD_CMAP.get(name, 'jet')


def plot_field_2d(x, y, values, save_path, title='', cmap='jet',
                   xlabel='$x$ (m)', ylabel='$y$ (m)', dpi=300,
                   vmin=None, vmax=None):
    """Plot a 2D scalar field with colorbar for publication."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sc = ax.scatter(x, y, c=values, cmap=cmap, s=2.0, linewidths=0,
                     vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(labelsize=10)
    plt.tight_layout(pad=0.5)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_acoustic_grid(x, y, truth, pred, save_path, angles=None,
                        gt_amplitudes=None, pred_amplitudes=None,
                        title1='$p_{\\mathrm{total}}$',
                        title2='Scattered Sound Pressure Directivity',
                        cmap='jet', dpi=300, re_value=None):
    """
    Generate the 2-row acoustic figure for the paper.
    Top: GT | Prediction | Absolute Error for p_total.
    Bottom: Directivity pattern overlay.

    Args:
        x, y: [N] coordinates
        truth, pred: [N] total sound pressure
        save_path: output path
        angles: [-180, 180] degrees for directivity
        gt_amplitudes, pred_amplitudes: directivity amplitudes
        title1, title2: subplot titles
        cmap: colormap
        dpi: resolution
        re_value: optional RE annotation
    """
    nrows = 2 if angles is not None else 1
    height = 5.2 * nrows
    fig = plt.figure(figsize=(18, height))

    gs_top = fig.add_gridspec(nrows, 1, hspace=0.35)
    gs_field = gs_top[0].subgridspec(1, 3, wspace=0.3)

    vmin = min(truth.min(), pred.min())
    vmax = max(truth.max(), pred.max())

    ax0 = fig.add_subplot(gs_field[0])
    sc0 = ax0.scatter(x, y, c=truth, cmap=cmap, s=3.0, linewidths=0,
                       vmin=vmin, vmax=vmax)
    ax0.set_title(f'{title1} — Ground Truth', fontsize=13)
    ax0.set_xlabel('$x$ (m)')
    ax0.set_ylabel('$y$ (m)')
    ax0.set_aspect('equal', adjustable='box')
    ax0.tick_params(labelsize=9)
    cb0 = plt.colorbar(sc0, ax=ax0, fraction=0.046, pad=0.04)
    cb0.ax.tick_params(labelsize=9)
    cb0.set_label('$p_{\\mathrm{total}}$ (Pa)', fontsize=10)

    ax1 = fig.add_subplot(gs_field[1])
    sc1 = ax1.scatter(x, y, c=pred, cmap=cmap, s=3.0, linewidths=0,
                       vmin=vmin, vmax=vmax)
    ax1.set_title(f'{title1} — Prediction', fontsize=13)
    ax1.set_xlabel('$x$ (m)')
    ax1.set_aspect('equal', adjustable='box')
    ax1.tick_params(labelsize=9)
    cb1 = plt.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04)
    cb1.ax.tick_params(labelsize=9)
    cb1.set_label('$p_{\\mathrm{total}}$ (Pa)', fontsize=10)
    if re_value is not None:
        ax1.text(0.5, -0.18, f'RE = {re_value:.4f}',
                 transform=ax1.transAxes, ha='center',
                 fontsize=11, style='italic')

    ax2 = fig.add_subplot(gs_field[2])
    abs_err = np.abs(truth - pred)
    sc2 = ax2.scatter(x, y, c=abs_err, cmap='hot', s=3.0, linewidths=0)
    ax2.set_title(f'{title1} — |Error|', fontsize=13)
    ax2.set_xlabel('$x$ (m)')
    ax2.set_aspect('equal', adjustable='box')
    ax2.tick_params(labelsize=9)
    cb2 = plt.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.ax.tick_params(labelsize=9)
    cb2.set_label('$|p_{\\mathrm{total}}$ error| (Pa)', fontsize=10)

    if angles is not None:
        gs_dir = gs_top[1].subgridspec(1, 1)
        ax_dir = fig.add_subplot(gs_dir[0])
        if gt_amplitudes is not None:
            ax_dir.plot(angles, gt_amplitudes, 'k-', linewidth=2.5,
                         label='Ground Truth')
            ax_dir.plot(angles, pred_amplitudes, 'r--', linewidth=2.0,
                         label='Prediction')
            ax_dir.legend(loc='upper right')
        else:
            ax_dir.plot(angles, pred_amplitudes, 'b-', linewidth=2.0)
        ax_dir.set_xlim([-180, 180])
        ax_dir.set_ylim([0, 1.15])
        ax_dir.set_xlabel('Angle (°)')
        ax_dir.set_ylabel('Normalized Amplitude')
        ax_dir.set_title(title2, fontsize=13)
        ax_dir.grid(True, alpha=0.3, linewidth=0.5)
        ax_dir.tick_params(labelsize=11)

    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def generate_thermal_figures(results_dir, output_dir, prefix='thermal'):
    """
    Batch-generate all thermal field publication figures from experiment results.

    Reads prediction NPZ files from results_dir and generates:
      - 4 individual comparison figures: {u,v,p,T}_{truth,pred,error}.png
      - 1 4×3 grid figure: thermal_grid.png
      - 1 loss curve: loss_curve.png

    Args:
        results_dir: path containing .npz files (preds, truth, X, loss_history)
        output_dir: where to save figures
        prefix: filename prefix for loaded NPZ files
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    pred_path = os.path.join(results_dir, f'{prefix}_preds.npz')
    if not os.path.exists(pred_path):
        pred_path = os.path.join(results_dir, 'fast_test.npz')

    data = np.load(pred_path, allow_pickle=True)
    X = data['X'] if 'X' in data else data.get('coords', None)
    preds = data['preds'] if 'preds' in data else data.get('predictions', None)
    truth = data['truth'] if 'truth' in data else data.get('ground_truth', None)
    loss_history = data.get('loss_history', None)

    if X is None or preds is None or truth is None:
        raise ValueError(f'Missing data in {pred_path}. Found keys: {list(data.keys())}')

    field_names = ['u', 'v', 'p', 'T']
    re_dict = {}
    for i, name in enumerate(field_names):
        re = np.linalg.norm(preds[i] - truth[i]) / np.linalg.norm(truth[i])
        re_dict[name] = re
        plot_comparison(X[:, 0], X[:, 1], truth[i], preds[i],
                         os.path.join(output_dir, f'{name}_comparison.png'),
                         field_name=name, re_value=re)

    plot_thermal_grid(X, preds, truth, field_names,
                       os.path.join(output_dir, 'thermal_grid.png'),
                       re_dict=re_dict, dpi=300)

    if loss_history is not None:
        plot_loss_curve(loss_history,
                         os.path.join(output_dir, 'loss_curve.png'),
                         title='Thermal Training Loss')

    return re_dict


def generate_acoustic_figures(results_dir, output_dir, prefix='acoustic'):
    """
    Batch-generate all acoustic field publication figures from experiment results.

    Reads prediction NPZ files from results_dir and generates:
      - 1 comparison figure: p_total_comparison.png (GT|Pred|Error)
      - 1 acoustic grid figure: acoustic_grid.png (p_total + directivity)
      - 1 directivity plot: directivity.png
      - 1 loss curve: loss_curve.png

    Args:
        results_dir: path containing .npz files
        output_dir: where to save figures
        prefix: filename prefix
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    pred_path = os.path.join(results_dir, f'{prefix}_preds.npz')
    if not os.path.exists(pred_path):
        pred_path = os.path.join(results_dir, 'fast_test.npz')

    data = np.load(pred_path, allow_pickle=True)
    X = data['X'] if 'X' in data else data.get('coords', None)
    preds = data['preds'] if 'preds' in data else data.get('predictions', None)
    truth = data['truth'] if 'truth' in data else data.get('ground_truth', None)
    loss_history = data.get('loss_history', None)

    if X is None or preds is None or truth is None:
        raise ValueError(f'Missing data in {pred_path}. Found keys: {list(data.keys())}')

    re = np.linalg.norm(preds - truth) / np.linalg.norm(truth)

    plot_comparison(X[:, 0], X[:, 1], truth, preds,
                     os.path.join(output_dir, 'p_total_comparison.png'),
                     field_name='p_total', re_value=re)

    angles = data.get('angles', None)
    gt_amp = data.get('gt_amplitudes', None)
    pred_amp = data.get('pred_amplitudes', None)

    if angles is not None and pred_amp is not None:
        plot_directivity(angles, pred_amp,
                          os.path.join(output_dir, 'directivity.png'),
                          gt_amplitudes=gt_amp)
        plot_acoustic_grid(X[:, 0], X[:, 1], truth, preds,
                            os.path.join(output_dir, 'acoustic_grid.png'),
                            angles=angles, gt_amplitudes=gt_amp,
                            pred_amplitudes=pred_amp, re_value=re)

    if loss_history is not None:
        plot_loss_curve(loss_history,
                         os.path.join(output_dir, 'loss_curve.png'),
                         title='Acoustic Training Loss')

    return re


def plot_loss_curve(loss_history, save_path, title='Training Loss',
                     dpi=300, test_losses=None):
    """Plot loss curve with optional test loss in dual y-axis."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(loss_history, linewidth=1.2, label='Train', color='#1f77b4')
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.tick_params(labelsize=10)

    if test_losses is not None and len(test_losses) > 0:
        ax_test = ax.twinx()
        ax_test.set_yscale('log')
        ax_test.plot(test_losses, linewidth=1.2, alpha=0.5,
                      label='Test', color='#ff7f0e', linestyle='--')
        ax_test.set_ylabel('Test MSE (log scale)')
        ax_test.tick_params(labelsize=10)
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax_test.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, loc='upper right')

    plt.tight_layout(pad=0.5)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_comparison(x, y, truth, pred, save_path, field_name='',
                    title_truth='Ground Truth',
                    title_pred='Prediction',
                    title_error='Absolute Error',
                    cmap='jet', dpi=300, re_value=None):
    """
    Side-by-side comparison: GT | Prediction | Absolute Error.

    All three panels share the same color scale for GT and Prediction.
    The error panel uses a 'hot' colormap.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    vmin = min(truth.min(), pred.min())
    vmax = max(truth.max(), pred.max())
    cbar_label = _get_label(field_name) if field_name else ''

    sc0 = axes[0].scatter(x, y, c=truth, cmap=cmap, s=3.0, linewidths=0,
                           vmin=vmin, vmax=vmax)
    axes[0].set_title(title_truth, fontsize=13)
    axes[0].set_xlabel('$x$ (m)')
    axes[0].set_ylabel('$y$ (m)')
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].tick_params(labelsize=9)
    cb0 = plt.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.04)
    cb0.ax.tick_params(labelsize=9)
    if cbar_label:
        cb0.set_label(cbar_label, fontsize=10)

    sc1 = axes[1].scatter(x, y, c=pred, cmap=cmap, s=3.0, linewidths=0,
                           vmin=vmin, vmax=vmax)
    axes[1].set_title(title_pred, fontsize=13)
    axes[1].set_xlabel('$x$ (m)')
    axes[1].set_aspect('equal', adjustable='box')
    axes[1].tick_params(labelsize=9)
    cb1 = plt.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.04)
    cb1.ax.tick_params(labelsize=9)
    if cbar_label:
        cb1.set_label(cbar_label, fontsize=10)
    if re_value is not None:
        axes[1].text(0.5, -0.18, f'RE = {re_value:.4f}',
                     transform=axes[1].transAxes, ha='center',
                     fontsize=11, style='italic')

    abs_err = np.abs(truth - pred)
    sc2 = axes[2].scatter(x, y, c=abs_err, cmap='hot', s=3.0, linewidths=0)
    axes[2].set_title(title_error, fontsize=13)
    axes[2].set_xlabel('$x$ (m)')
    axes[2].set_aspect('equal', adjustable='box')
    axes[2].tick_params(labelsize=9)
    cb2 = plt.colorbar(sc2, ax=axes[2], fraction=0.046, pad=0.04)
    cb2.ax.tick_params(labelsize=9)
    if cbar_label:
        cb2.set_label(f'|{_get_latex(field_name)} error| {_get_label(field_name).split("(")[-1].rstrip(")") if "(" in _get_label(field_name) else ""}', fontsize=10)

    plt.tight_layout(pad=1.0, w_pad=2.0)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_thermal_grid(X, preds, truth, field_names, save_path, re_dict=None, dpi=300):
    """
    Generate the 4×3 grid figure for the thermal field paper figure.
    Rows: u, v, p, T. Columns: GT, Prediction, Absolute Error.

    Args:
        X: [N, 2] coordinates
        preds: [4, N] predictions
        truth: [4, N] ground truth
        field_names: ['u', 'v', 'p', 'T']
        save_path: output path
        re_dict: {name: re_value} for RE annotation
        dpi: resolution
    """
    n_fields = len(field_names)
    fig, axes = plt.subplots(n_fields, 3, figsize=(18, 5 * n_fields))

    for row, name in enumerate(field_names):
        vmin = min(truth[row].min(), preds[row].min())
        vmax = max(truth[row].max(), preds[row].max())
        cmap = _get_cmap(name)
        label = _get_label(name)

        sc0 = axes[row, 0].scatter(X[:, 0], X[:, 1], c=truth[row],
                                    cmap=cmap, s=2.0, linewidths=0,
                                    vmin=vmin, vmax=vmax)
        axes[row, 0].set_title(f'{_get_latex(name)} Ground Truth', fontsize=13)
        axes[row, 0].set_ylabel('$y$ (m)')
        axes[row, 0].set_aspect('equal', adjustable='box')
        axes[row, 0].tick_params(labelsize=9)
        cb0 = plt.colorbar(sc0, ax=axes[row, 0], fraction=0.046, pad=0.04)
        cb0.ax.tick_params(labelsize=9)
        cb0.set_label(label, fontsize=10)

        sc1 = axes[row, 1].scatter(X[:, 0], X[:, 1], c=preds[row],
                                    cmap=cmap, s=2.0, linewidths=0,
                                    vmin=vmin, vmax=vmax)
        axes[row, 1].set_title(f'{_get_latex(name)} Prediction', fontsize=13)
        axes[row, 1].set_aspect('equal', adjustable='box')
        axes[row, 1].tick_params(labelsize=9)
        cb1 = plt.colorbar(sc1, ax=axes[row, 1], fraction=0.046, pad=0.04)
        cb1.ax.tick_params(labelsize=9)
        cb1.set_label(label, fontsize=10)
        if re_dict is not None and name in re_dict:
            axes[row, 1].text(0.5, -0.16, f'RE = {re_dict[name]:.4f}',
                              transform=axes[row, 1].transAxes, ha='center',
                              fontsize=11, style='italic')

        abs_err = np.abs(truth[row] - preds[row])
        sc2 = axes[row, 2].scatter(X[:, 0], X[:, 1], c=abs_err,
                                    cmap='hot', s=2.0, linewidths=0)
        axes[row, 2].set_title(f'{_get_latex(name)} |Error|', fontsize=13)
        axes[row, 2].set_aspect('equal', adjustable='box')
        axes[row, 2].tick_params(labelsize=9)
        cb2 = plt.colorbar(sc2, ax=axes[row, 2], fraction=0.046, pad=0.04)
        cb2.ax.tick_params(labelsize=9)

    axes[-1, 0].set_xlabel('$x$ (m)')
    axes[-1, 1].set_xlabel('$x$ (m)')
    axes[-1, 2].set_xlabel('$x$ (m)')

    plt.tight_layout(pad=1.0, w_pad=2.0, h_pad=2.0)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_re_bar(re_dict, save_path, title='Relative L2 Error',
                 dpi=300, field_names=None):
    """
    RE bar chart with Mean±Std error bars.
    re_dict: {method_name: (mean, std)} or {method_name: [re1, re2, re3, re4]}
    """
    if field_names is None:
        field_names = ['$u$', '$v$', '$p$', '$T$']

    fig, ax = plt.subplots(figsize=(10, 5))
    methods = list(re_dict.keys())
    num_methods = len(methods)
    num_fields = len(field_names)
    x = np.arange(num_fields)
    width = 0.8 / num_methods

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, method in enumerate(methods):
        val = re_dict[method]
        if isinstance(val, tuple):
            mean, std = val
            ax.bar(x + i * width, mean, width, yerr=std, capsize=3,
                    label=method, color=colors[i % len(colors)])
        elif isinstance(val, list):
            ax.bar(x + i * width, val, width,
                    label=method, color=colors[i % len(colors)])
        else:
            ax.bar(x + i * width, [val], width,
                    label=method, color=colors[i % len(colors)])

    ax.set_xlabel('Physical Quantity')
    ax.set_ylabel('Relative L2 Error')
    ax.set_title(title)
    ax.set_xticks(x + width * (num_methods - 1) / 2)
    ax.set_xticklabels(field_names)
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', alpha=0.3, linewidth=0.5)
    ax.tick_params(labelsize=11)
    plt.tight_layout(pad=0.5)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_directivity(angles, amplitudes, save_path,
                      title=r'Scattered Sound Pressure $p_{\mathrm{scat}}$ Directivity',
                      dpi=300, gt_amplitudes=None):
    """
    Scattered sound pressure directivity plot.

    Args:
        angles: [-180, 180] degrees
        amplitudes: predicted normalized amplitudes
        save_path: output path
        title: plot title
        dpi: resolution
        gt_amplitudes: optional GT amplitudes for overlay
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    if gt_amplitudes is not None:
        ax.plot(angles, gt_amplitudes, 'k-', linewidth=2.5, label='Ground Truth')
        ax.plot(angles, amplitudes, 'r--', linewidth=2.0, label='Prediction')
        ax.legend(loc='upper right')
    else:
        ax.plot(angles, amplitudes, 'b-', linewidth=2.0)
    ax.set_xlim([-180, 180])
    ax.set_ylim([0, 1.15])
    ax.set_xlabel('Angle (°)')
    ax.set_ylabel('Normalized Amplitude')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.tick_params(labelsize=11)
    plt.tight_layout(pad=0.5)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)