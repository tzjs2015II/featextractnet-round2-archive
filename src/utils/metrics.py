import numpy as np
import torch


def relative_l2_error(pred, truth):
    """
    Relative L2 error: ||pred - truth||_2 / ||truth||_2

    Args:
        pred: np.ndarray or torch.Tensor
        truth: np.ndarray or torch.Tensor

    Returns:
        float scalar
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(truth, torch.Tensor):
        truth = truth.detach().cpu().numpy()

    diff_norm = np.linalg.norm(pred - truth)
    truth_norm = np.linalg.norm(truth)

    if truth_norm < 1e-12:
        return float('inf')

    return diff_norm / truth_norm


def compute_all_re(predictions, ground_truth, field_names=None):
    """
    Compute relative L2 error for each output channel.

    Args:
        predictions: [num_samples, num_channels, num_points] np.ndarray
        ground_truth: [num_samples, num_channels, num_points] np.ndarray
        field_names: list of field names, e.g. ['u', 'v', 'p', 'T']

    Returns:
        dict: {field_name: [re_sample1, re_sample2, ...]}
    """
    if field_names is None:
        num_channels = predictions.shape[1]
        field_names = [f'ch{i}' for i in range(num_channels)]

    results = {}
    for ch, name in enumerate(field_names):
        re_list = []
        for i in range(predictions.shape[0]):
            re = relative_l2_error(predictions[i, ch, :], ground_truth[i, ch, :])
            re_list.append(re)
        results[name] = re_list

    return results


def summarize_re(re_dict):
    """
    Compute mean ± std for each field from a dict of RE lists.

    Returns:
        dict: {field_name: (mean, std)}
    """
    summary = {}
    for name, re_list in re_dict.items():
        arr = np.array(re_list)
        summary[name] = (float(np.mean(arr)),
                         float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0)
    return summary


def format_re_table(summary, precision=4):
    """
    Format RE summary as a LaTeX-ready table string.

    Args:
        summary: dict from summarize_re()
        precision: decimal places

    Returns:
        str: formatted table row
    """
    parts = []
    for name, (mean, std) in summary.items():
        parts.append(f'{mean:.{precision}f}$\\pm${std:.{precision}f}')
    return ' & '.join(parts)


def compute_all_re_acoustic(predictions, ground_truth):
    """
    Compute RE for acoustic field (single-channel: p_total).

    Args:
        predictions: [num_samples, 1, num_points]
        ground_truth: [num_samples, 1, num_points]

    Returns:
        list of RE values
    """
    re_list = []
    for i in range(predictions.shape[0]):
        re = relative_l2_error(predictions[i, 0, :], ground_truth[i, 0, :])
        re_list.append(re)
    return re_list