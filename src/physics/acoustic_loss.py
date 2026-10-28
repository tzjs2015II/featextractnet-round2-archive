import torch


def acoustic_obs_loss(model, X, p_true):
    """
    Supervised observation loss for acoustic field.

    The acoustic prediction uses FDTD-generated data in a supervised manner.
    The FDTD simulation solves the linearized Euler equations with a background
    vortex mean flow (Ma=0.5), which the standard Helmholtz equation does not
    capture. Therefore, we use pure supervised learning (L_obs only) rather
    than adding a Helmholtz PDE loss that would conflict with the convection
    effects present in the data.

    Incorporating acoustic PDE constraints (e.g., convective Helmholtz equation)
    is noted as future work.

    Args:
        model: FeatExtractNet that takes [B, 2, N] → [B, 1, N] (p_total)
        X: point coordinates [B, 2, N]
        p_true: ground truth total sound pressure [B, 1, N]

    Returns:
        Observation MSE loss scalar
    """
    Y = model(X)
    p_pred = Y[:, 0:1, :]
    loss = torch.mean((p_pred - p_true) ** 2)
    return loss


def acoustic_total_loss(model, X, p_true, lambda_obs=1.0):
    """
    Total acoustic loss = λ_obs * L_obs

    Currently purely supervised. Future work may add:
      L_PDE = || ∂²p/∂x² + ∂²p/∂y² + k²p ||²  (Helmholtz equation)
    with appropriate handling of mean flow convection effects.

    Args:
        model: FeatExtractNet
        X: point coordinates [B, 2, N]
        p_true: ground truth total sound pressure [B, 1, N]
        lambda_obs: observation loss weight

    Returns:
        total loss, loss components dict
    """
    loss_obs = acoustic_obs_loss(model, X, p_true)
    total = lambda_obs * loss_obs
    return total, {'obs': loss_obs.item()}


def helmholtz_pde_loss(model, X, k):
    """
    Helmholtz PDE residual (for future use).

    R_helmholtz = ∂²p/∂x² + ∂²p/∂y² + k²·p

    Note: This equation assumes a quiescent medium. When mean flow effects
    are significant (as in the current FDTD data with Ma=0.5), the convective
    Helmholtz equation or linearized Euler equations should be used instead.

    Args:
        model: FeatExtractNet
        X: interior points [B, 2, N], requires_grad
        k: wavenumber = 2π·f / c0

    Returns:
        Helmholtz PDE loss scalar
    """
    X.requires_grad_(True)
    Y = model(X)
    p = Y[:, 0:1, :]

    dp = torch.autograd.grad(p, X, grad_outputs=torch.ones_like(p),
                             create_graph=True, retain_graph=True)[0]
    p_x = dp[:, 0:1, :]
    p_y = dp[:, 1:2, :]

    dp_x = torch.autograd.grad(p_x, X, grad_outputs=torch.ones_like(p_x),
                               create_graph=True, retain_graph=True)[0]
    dp_y = torch.autograd.grad(p_y, X, grad_outputs=torch.ones_like(p_y),
                               create_graph=True, retain_graph=True)[0]

    p_xx = dp_x[:, 0:1, :]
    p_yy = dp_y[:, 1:2, :]

    R_helmholtz = p_xx + p_yy + (k ** 2) * p
    loss = torch.mean(R_helmholtz ** 2)
    return loss