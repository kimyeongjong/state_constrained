import math
import os
from typing import Callable, Optional

import numpy as np
import torch

from pinn_hj.cylinder_domain import cylindrical_params, distance_to_cylindrical_domain


def _batched_predict(
    model: torch.nn.Module,
    x: torch.Tensor,
    batch_size: int = 8192,
) -> torch.Tensor:
    u_chunks = []
    n = x.shape[0]
    for i in range(0, n, batch_size):
        xi = x[i : i + batch_size].detach().clone()
        with torch.no_grad():
            u = model(xi)
        u_chunks.append(u.detach())
    return torch.cat(u_chunks, dim=0)


def _orthonormal_plane_basis(axis: torch.Tensor) -> torch.Tensor:
    """Return two orthonormal vectors spanning a 2D subspace perpendicular to axis."""
    n_dim = axis.numel()
    if n_dim < 3:
        raise ValueError("Need n>=3 for a 2D slice perpendicular to the axis")
    axis = axis / torch.linalg.norm(axis)
    eye = torch.eye(n_dim, device=axis.device, dtype=axis.dtype)
    proj = eye - axis.view(-1, 1) @ axis.view(1, -1)
    q, r = torch.linalg.qr(proj)
    diag = torch.abs(torch.diagonal(r))
    idx = torch.nonzero(diag > 1e-8, as_tuple=False).view(-1)
    if idx.numel() < 2:
        raise ValueError("Failed to construct orthonormal basis for the slice plane")
    return q[:, idx[:2]].t()


def _plot_heatmap(
    data: np.ndarray,
    extent,
    title: str,
    out_path: Optional[str],
    cmap: str = "coolwarm",
    symmetric: bool = False,
    show: bool = False,
):
    import matplotlib.pyplot as plt

    if symmetric:
        vmax = np.nanmax(np.abs(data))
        vmin = -vmax
    else:
        vmin = np.nanmin(data)
        vmax = np.nanmax(data)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        data,
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(im)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        plt.savefig(out_path, dpi=200)
        print(f"Saved plot: {out_path}")
    if show:
        try:
            plt.ion()
            plt.show(block=False)
            plt.pause(0.001)
        except Exception:
            plt.show()
    else:
        plt.close()


def _ensure_dir(path: Optional[str], save: bool) -> Optional[str]:
    if not save:
        return None
    os.makedirs(path, exist_ok=True)
    return path


def _mask_to_cylinder(data: torch.Tensor, x_flat: torch.Tensor, k: float) -> torch.Tensor:
    d = distance_to_cylindrical_domain(x_flat, k).view(data.shape)
    mask = d <= 1e-6
    return torch.where(mask, data, torch.tensor(float("nan"), device=data.device))


def _exact_solution_axis(x: torch.Tensor, k: float) -> torch.Tensor:
    n_dim = x.shape[1]
    a = x.sum(dim=1, keepdim=True) / math.sqrt(float(n_dim))
    return torch.exp(a - k / math.sqrt(float(n_dim)))


def plot_after_epsilon(
    *,
    n_dim: int,
    k: float,
    eps: float,
    model: torch.nn.Module,
    save: bool,
    save_dir: Optional[str],
    show: bool,
    eval_points: int,
    device: torch.device,
    dtype: torch.dtype,
    exact_solution_1d: Optional[Callable[[torch.Tensor, float], torch.Tensor]] = None,
    grid: int = 201,
    margin_plot_ratio: float = 0.1,
) -> None:
    if n_dim == 1:
        if exact_solution_1d is None:
            raise ValueError("exact_solution_1d is required for 1D plotting.")
        import matplotlib.pyplot as plt

        x_plot = torch.linspace(-k, k, int(eval_points), device=device, dtype=dtype).view(-1, 1)
        with torch.no_grad():
            u_pred = model(x_plot)
            u_true = exact_solution_1d(x_plot, k)

        x_cpu = x_plot.squeeze().cpu().numpy()
        u_pred_cpu = u_pred.squeeze().cpu().numpy()
        u_true_cpu = u_true.squeeze().cpu().numpy()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x_cpu, u_true_cpu, "k--", label="Exact: exp(x-k)")
        ax.plot(x_cpu, u_pred_cpu, label=f"PINN (eps={eps:g})")
        ax.set_title("HJ solution in [-k,k]")
        ax.set_xlabel("x")
        ax.set_ylabel("u(x)")
        ax.legend()
        fig.tight_layout()

        save_root = _ensure_dir(save_dir, save)
        if save_root:
            fname = f"hj_pinn_k{int(k) if float(k).is_integer() else k}_eps_{eps:g}.png"
            out_path = os.path.join(save_root, fname)
            fig.savefig(out_path, dpi=150)
            print(f"Saved plot to: {out_path}")

        if show:
            try:
                plt.ion()
                try:
                    fig.show()
                except Exception:
                    pass
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.001)
            except Exception:
                pass
        else:
            plt.close(fig)
        return

    margin_plot = margin_plot_ratio * k
    tag = f"n{n_dim}_k{k:g}_eps{eps:g}"
    save_root = _ensure_dir(save_dir, save)

    if n_dim == 2:
        lim = k + margin_plot
        xs = torch.linspace(-lim, lim, grid, device=device, dtype=dtype)
        x1, x2 = torch.meshgrid(xs, xs, indexing="ij")
        x_flat = torch.stack([x1.reshape(-1), x2.reshape(-1)], dim=1)
        u_pred = _batched_predict(model, x_flat)
        u_true = _exact_solution_axis(x_flat, k)
        diff = (u_pred - u_true).view(grid, grid)
        diff = _mask_to_cylinder(diff, x_flat, k)
        diff_np = diff.detach().cpu().numpy()
        extent = [-lim, lim, -lim, lim]

        diff_path = os.path.join(save_root, f"heatmap_diff_{tag}.png") if save_root else None
        _plot_heatmap(diff_np, extent, f"diff heatmap ({tag})", diff_path, symmetric=True, show=show)
        return

    # n >= 3: axis line (u_true vs u_pred) + 2D slice diff heatmap
    ones = torch.ones(n_dim, device=device, dtype=dtype)
    axis = ones / torch.linalg.norm(ones)
    basis = _orthonormal_plane_basis(axis)
    v1 = basis[0].view(1, n_dim)
    v2 = basis[1].view(1, n_dim)

    # Axis line plot (u_true vs u_pred)
    s_min = -k - margin_plot
    s_max = k + margin_plot
    s_vals = torch.linspace(s_min, s_max, grid, device=device, dtype=dtype).view(-1, 1)
    x_line = (s_vals / float(n_dim)) * torch.ones(s_vals.shape[0], n_dim, device=device, dtype=dtype)
    u_pred_line = _batched_predict(model, x_line)
    u_true_line = _exact_solution_axis(x_line, k)
    s_cpu = s_vals.squeeze().cpu().numpy()
    u_pred_cpu = u_pred_line.squeeze().cpu().numpy()
    u_true_cpu = u_true_line.squeeze().cpu().numpy()

    save_root = _ensure_dir(save_dir, save)
    line_path = os.path.join(save_root, f"axis_line_{tag}.png") if save_root else None

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(s_cpu, u_true_cpu, "k--", label="u_true")
    ax.plot(s_cpu, u_pred_cpu, label="u_pred")
    ax.set_xlabel("s (sum x_i)")
    ax.set_ylabel("u")
    ax.set_title(f"Axis line (u_true vs u_pred, {tag})")
    ax.legend()
    fig.tight_layout()
    if line_path:
        fig.savefig(line_path, dpi=200)
        print(f"Saved plot: {line_path}")
    if show:
        plt.show(block=False)
    plt.close(fig)

    # 2D slice heatmap
    _, r_k = cylindrical_params(n_dim, k)
    lim = r_k + margin_plot
    xs = torch.linspace(-lim, lim, grid, device=device, dtype=dtype)
    x1, x2 = torch.meshgrid(xs, xs, indexing="ij")
    m = torch.zeros(n_dim, device=device, dtype=dtype)
    x_flat = m + x1.reshape(-1, 1) * v1 + x2.reshape(-1, 1) * v2
    u_pred = _batched_predict(model, x_flat)
    u_true = _exact_solution_axis(x_flat, k)
    diff = (u_pred - u_true).view(grid, grid)
    diff = _mask_to_cylinder(diff, x_flat, k)
    diff_np = diff.detach().cpu().numpy()
    extent = [-lim, lim, -lim, lim]

    diff_path = os.path.join(save_root, f"slice_diff_{tag}.png") if save_root else None
    _plot_heatmap(diff_np, extent, f"slice diff ({tag})", diff_path, symmetric=True, show=show)
