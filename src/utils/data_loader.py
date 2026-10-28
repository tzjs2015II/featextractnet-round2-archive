import numpy as np
import torch
from scipy.io import loadmat


def load_thermal_data(data_path, problem_dimension=2, variable_number=4):
    """
    Load thermal field data from .npy file.

    Expected format: [num_geometries, num_points, dim+variables]

    Args:
        data_path: path to .npy file (e.g., HeatTransferData.npy)
        problem_dimension: input coordinate dim (default 2: x, y)
        variable_number: output variable count (default 4: u, v, p, T)

    Returns:
        X_train: [num_geometries, num_points, problem_dimension]  — coordinates
        Y_train: [num_geometries, num_points, variable_number]    — CFD solution
        num_geometries, num_points
    """
    data = np.load(data_path)
    num_geometries, num_points, _ = data.shape

    X_train = data[:, :, :problem_dimension].astype(np.float32)
    Y_train = data[:, :, problem_dimension:problem_dimension + variable_number].astype(np.float32)

    return X_train, Y_train, num_geometries, num_points


def load_acoustic_data(mat_path):
    """
    Load acoustic FDTD data from .mat file.

    Expected MAT-file variables:
      - p:  total sound pressure field [nx, ny]
      - P:  incident pressure field [nx, ny] (optional)
      - Various FDTD parameters

    The function extracts a sub-region (avoiding PML boundaries) and
    converts grid data to (x, y, p_total) point cloud format.

    Args:
        mat_path: path to .mat file

    Returns:
        X: point coordinates [N, 2]
        Y: total sound pressure [N, 1]
    """
    mat = loadmat(mat_path)

    p_total = mat.get('p')
    if p_total is None:
        raise KeyError(f"Variable 'p' not found in {mat_path}. Available keys: {list(mat.keys())}")

    nx, ny = p_total.shape

    # Reconstruct coordinates from known FDTD parameters
    # Grid: dx=dy=0.125, domain from -50 to +50 (800 cells)
    dx = 0.125
    dy = 0.125
    x_min = -nx // 2 * dx
    y_min = -ny // 2 * dy

    x_coords = np.linspace(x_min, x_min + (nx - 1) * dx, nx)
    y_coords = np.linspace(y_min, y_min + (ny - 1) * dy, ny)

    X_grid, Y_grid = np.meshgrid(x_coords, y_coords, indexing='ij')

    X = np.stack([X_grid.flatten(), Y_grid.flatten()], axis=-1).astype(np.float32)
    Y = p_total.flatten().reshape(-1, 1).astype(np.float32)

    return X, Y


def load_acoustic_data_custom(mat_path, x_range=None, y_range=None):
    """
    Load acoustic data with optional spatial cropping.

    Args:
        mat_path: path to .mat file
        x_range: (x_min, x_max) or None for full range
        y_range: (y_min, y_max) or None for full range

    Returns:
        X: point coordinates [N, 2]
        Y: total sound pressure [N, 1]
    """
    mat = loadmat(mat_path)
    p_total = mat['p']
    nx, ny = p_total.shape

    dx = 0.125
    dy = 0.125
    x_min = -nx // 2 * dx
    y_min = -ny // 2 * dy

    x_coords = np.linspace(x_min, x_min + (nx - 1) * dx, nx)
    y_coords = np.linspace(y_min, y_min + (ny - 1) * dy, ny)

    if x_range is not None:
        x_mask = (x_coords >= x_range[0]) & (x_coords <= x_range[1])
        x_coords = x_coords[x_mask]
        p_total = p_total[x_mask, :]

    if y_range is not None:
        y_mask = (y_coords >= y_range[0]) & (y_coords <= y_range[1])
        y_coords = y_coords[y_mask]
        p_total = p_total[:, y_mask]

    X_grid, Y_grid = np.meshgrid(x_coords, y_coords, indexing='ij')
    X = np.stack([X_grid.flatten(), Y_grid.flatten()], axis=-1).astype(np.float32)
    Y = p_total.flatten().reshape(-1, 1).astype(np.float32)

    return X, Y


def to_torch(X, Y, device='cpu', requires_grad=False):
    """
    Convert numpy arrays to PyTorch tensors in [B, C, N] format.

    Args:
        X: [B, N, C_in] or [N, C_in]
        Y: [B, N, C_out] or [N, C_out]
        device: torch device
        requires_grad: whether X requires gradient

    Returns:
        X_t: [B, C_in, N]
        Y_t: [B, C_out, N]
    """
    if X.ndim == 2:
        X = X[np.newaxis, ...]
        Y = Y[np.newaxis, ...]

    X_t = torch.tensor(X, dtype=torch.float32, device=device).permute(0, 2, 1)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device).permute(0, 2, 1)

    if requires_grad:
        X_t.requires_grad_(True)

    return X_t, Y_t


def split_boundary_interior(X, Y, boundary_indices):
    """
    Split data into boundary and interior points.

    Args:
        X: [B, C, N] tensor
        Y: [B, Cout, N] tensor
        boundary_indices: list of boundary point indices

    Returns:
        X_boundary, Y_boundary, X_interior, Y_interior
    """
    interior_mask = torch.ones(X.shape[-1], dtype=torch.bool, device=X.device)
    interior_mask[boundary_indices] = False

    X_boundary = X[:, :, boundary_indices]
    Y_boundary = Y[:, :, boundary_indices]
    X_interior = X[:, :, interior_mask]
    Y_interior = Y[:, :, interior_mask]

    return X_boundary, Y_boundary, X_interior, Y_interior


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False