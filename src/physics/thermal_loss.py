import torch


def thermal_pde_loss(model, X_interior, density, viscosity, kappa, cp=1.0, g_beta=1.0):
    """
    Navier-Stokes PDE residual loss for 2D steady incompressible thermal flow.

    Governing equations (Boussinesq approximation):
      R_cont:   ∂u/∂x + ∂v/∂y = 0
      R_mom_x:  ρ(u·∂u/∂x + v·∂u/∂y) + ∂p/∂x - μ(∂²u/∂x² + ∂²u/∂y²) = 0
      R_mom_y:  ρ(u·∂v/∂x + v·∂v/∂y) + ∂p/∂y - μ(∂²v/∂x² + ∂²v/∂y²) - ρ·g·β·(T - T_ref) = 0
      R_energy: ρ·cp(u·∂T/∂x + v·∂T/∂y) - κ(∂²T/∂x² + ∂²T/∂y²) = 0

    Args:
        model: FeatExtractNet that takes [B, 2, N] → [B, 4, N] (u, v, p, T)
        X_interior: interior collocation points [B, 2, N_interior], requires_grad
        density (ρ): fluid density
        viscosity (μ): dynamic viscosity
        kappa (κ): thermal conductivity
        cp: specific heat capacity (default 1.0)
        g_beta: g·β product for buoyancy (default 1.0)

    Returns:
        PDE loss scalar
    """
    X = X_interior
    X.requires_grad_(True)

    Y = model(X)
    u = Y[:, 0:1, :]
    v = Y[:, 1:2, :]
    p = Y[:, 2:3, :]
    T = Y[:, 3:4, :]

    # First-order derivatives
    du = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u),
                             create_graph=True, retain_graph=True)[0]
    dv = torch.autograd.grad(v, X, grad_outputs=torch.ones_like(v),
                             create_graph=True, retain_graph=True)[0]
    dp = torch.autograd.grad(p, X, grad_outputs=torch.ones_like(p),
                             create_graph=True, retain_graph=True)[0]
    dT = torch.autograd.grad(T, X, grad_outputs=torch.ones_like(T),
                             create_graph=True, retain_graph=True)[0]

    u_x = du[:, 0:1, :]
    u_y = du[:, 1:2, :]
    v_x = dv[:, 0:1, :]
    v_y = dv[:, 1:2, :]
    p_x = dp[:, 0:1, :]
    p_y = dp[:, 1:2, :]
    T_x = dT[:, 0:1, :]
    T_y = dT[:, 1:2, :]

    # Second-order derivatives
    du_x = torch.autograd.grad(u_x, X, grad_outputs=torch.ones_like(u_x),
                               create_graph=True, retain_graph=True)[0]
    du_y = torch.autograd.grad(u_y, X, grad_outputs=torch.ones_like(u_y),
                               create_graph=True, retain_graph=True)[0]
    dv_x = torch.autograd.grad(v_x, X, grad_outputs=torch.ones_like(v_x),
                               create_graph=True, retain_graph=True)[0]
    dv_y = torch.autograd.grad(v_y, X, grad_outputs=torch.ones_like(v_y),
                               create_graph=True, retain_graph=True)[0]
    dT_x = torch.autograd.grad(T_x, X, grad_outputs=torch.ones_like(T_x),
                               create_graph=True, retain_graph=True)[0]
    dT_y = torch.autograd.grad(T_y, X, grad_outputs=torch.ones_like(T_y),
                               create_graph=True, retain_graph=True)[0]

    u_xx = du_x[:, 0:1, :]
    u_yy = du_y[:, 1:2, :]
    v_xx = dv_x[:, 0:1, :]
    v_yy = dv_y[:, 1:2, :]
    T_xx = dT_x[:, 0:1, :]
    T_yy = dT_y[:, 1:2, :]

    T_ref = 0.0

    R_cont = u_x + v_y
    R_mom_x = density * (u * u_x + v * u_y) + p_x - viscosity * (u_xx + u_yy)
    R_mom_y = density * (u * v_x + v * v_y) + p_y - viscosity * (v_xx + v_yy) - density * g_beta * (T - T_ref)
    R_energy = density * cp * (u * T_x + v * T_y) - kappa * (T_xx + T_yy)

    loss = torch.mean(R_cont ** 2 + R_mom_x ** 2 + R_mom_y ** 2 + R_energy ** 2)
    return loss


def thermal_bc_loss(model, X_boundary, u_bc=0.0, v_bc=0.0, T_bc=0.0):
    """
    Dirichlet BC loss for thermal field.

    Args:
        model: FeatExtractNet
        X_boundary: boundary points [B, 2, N_bc]
        u_bc, v_bc, T_bc: prescribed boundary values (default 0 for no-slip, cold wall)

    Returns:
        BC loss scalar
    """
    Y = model(X_boundary)
    u_pred = Y[:, 0:1, :]
    v_pred = Y[:, 1:2, :]
    T_pred = Y[:, 3:4, :]

    loss = torch.mean((u_pred - u_bc) ** 2 + (v_pred - v_bc) ** 2 + (T_pred - T_bc) ** 2)
    return loss


def thermal_obs_loss(model, X_sparse, u_true, v_true, p_true, T_true):
    """
    Observation loss at sensor points.

    Args:
        model: FeatExtractNet
        X_sparse: sensor point coordinates [B, 2, N_obs]
        u_true, v_true, p_true, T_true: ground truth values at sensor locations [B, 1, N_obs]

    Returns:
        Observation loss scalar
    """
    Y = model(X_sparse)
    u_pred = Y[:, 0:1, :]
    v_pred = Y[:, 1:2, :]
    p_pred = Y[:, 2:3, :]
    T_pred = Y[:, 3:4, :]

    loss = torch.mean((u_pred - u_true) ** 2 + (v_pred - v_true) ** 2
                      + (p_pred - p_true) ** 2 + (T_pred - T_true) ** 2)
    return loss


def thermal_total_loss(model, X_interior, X_boundary, X_sparse,
                        u_true, v_true, p_true, T_true,
                        density, viscosity, kappa,
                        lambda_pde=1.0, lambda_bc=1.0, lambda_obs=10.0):
    """
    Total thermal loss = λ_pde * L_PDE + λ_bc * L_BC + λ_obs * L_obs

    Default weights: λ_pde=1.0, λ_bc=1.0, λ_obs=10.0 (as per paper)
    """
    loss_pde = thermal_pde_loss(model, X_interior, density, viscosity, kappa)
    loss_bc = thermal_bc_loss(model, X_boundary)
    loss_obs = thermal_obs_loss(model, X_sparse, u_true, v_true, p_true, T_true)

    total = lambda_pde * loss_pde + lambda_bc * loss_bc + lambda_obs * loss_obs
    return total, {'pde': loss_pde.item(), 'bc': loss_bc.item(), 'obs': loss_obs.item()}