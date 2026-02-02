import argparse
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

# Make top-level package importable when running as a script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pinn_hj.networks import MLP
from pinn_hj.cylinder_domain import cylindrical_params, distance_to_cylindrical_domain


def parse_args():
    p = argparse.ArgumentParser(description="Axis line + 2D slice plots for n-D checkpoints")
    p.add_argument("--ckpt", required=True, help="Checkpoint path (.pt)")
    p.add_argument("--k", type=float, default=None, help="Override k (default: read from checkpoint)")
    p.add_argument("--n", type=int, default=None, help="Override dimension n (default: read from checkpoint)")
    p.add_argument("--grid", type=int, default=201, help="Grid resolution per axis for 2D slice")
    p.add_argument("--line-points", type=int, default=400, help="Number of points for axis line")
    p.add_argument(
        "--axis-s",
        type=float,
        default=0.0,
        help="Sum coordinate for the orthogonal slice plane: sum(x_i) = axis-s",
    )
    p.add_argument(
        "--margin",
        type=float,
        default=None,
        help="Extra margin for plot ranges (default: 0.1*k)",
    )
    p.add_argument(
        "--truth",
        choices=["sum-exp", "axis-exp", "zero"],
        default="sum-exp",
        help="Ground truth function",
    )
    p.add_argument(
        "--metric",
        choices=["diff", "rel"],
        default="diff",
        help="Error metric: diff = u_pred - u_true, rel = (u_pred - u_true)/|u_true|",
    )
    p.add_argument("--device", type=str, default="auto", help="cpu|cuda|auto")
    p.add_argument("--out-dir", type=str, default="plots", help="Output directory")
    p.add_argument("--show", action="store_true", help="Show plots interactively")
    p.add_argument("--cmap", type=str, default="coolwarm", help="Matplotlib colormap")
    p.add_argument("--vmax", type=float, default=None, help="Fix symmetric color scale (vmin=-vmax)")
    p.add_argument("--width", type=int, default=None, help="Fallback width if arch missing in ckpt")
    p.add_argument("--layers", type=int, default=None, help="Fallback layers if arch missing in ckpt")
    p.add_argument("--act", type=str, default=None, help="Fallback activation if arch missing in ckpt")
    return p.parse_args()


def build_model(ckpt, args):
    arch = ckpt.get("arch") or {}
    in_dim = arch.get("in_dim") or ckpt.get("n") or args.n
    if in_dim is None:
        raise ValueError("Checkpoint missing in_dim/n; please pass --n (and width/layers/act if needed)")

    widths = arch.get("widths")
    act = arch.get("act")
    if widths is None or act is None:
        if args.width is None or args.layers is None:
            raise ValueError("Checkpoint lacks arch; provide --width and --layers (and optionally --act)")
        widths = tuple([args.width] * int(args.layers))
        act = args.act or "tanh"

    model = MLP(in_dim=int(in_dim), out_dim=1, widths=tuple(widths), act=act)
    model.load_state_dict(ckpt["model_state"])
    return model, int(in_dim)


def ground_truth(x: torch.Tensor, k: float, mode: str) -> torch.Tensor:
    if mode == "sum-exp":
        s = x.sum(dim=1, keepdim=True)
        return torch.exp(s - k)
    if mode == "axis-exp":
        n = x.shape[1]
        a = x.sum(dim=1, keepdim=True) / math.sqrt(float(n))
        return torch.exp(a - k / math.sqrt(float(n)))
    if mode == "zero":
        return torch.zeros_like(x[:, 0:1])
    raise ValueError(f"Unknown truth mode: {mode}")


def compute_error(u_pred: torch.Tensor, u_true: torch.Tensor, metric: str) -> torch.Tensor:
    diff = u_pred - u_true
    if metric == "rel":
        denom = torch.clamp(torch.abs(u_true), min=1e-12)
        return diff / denom
    return diff


def format_tag(val: float) -> str:
    if float(val).is_integer():
        return str(int(val))
    return f"{val:g}"


def orthonormal_plane_basis(n_dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return two orthonormal vectors spanning a 2D subspace orthogonal to (1,...,1)."""
    if n_dim < 3:
        raise ValueError("Need n>=3 for a 2D slice orthogonal to (1,...,1)")
    ones = torch.ones(n_dim, device=device, dtype=dtype)
    u0 = ones / math.sqrt(float(n_dim))
    basis = []
    for i in range(n_dim):
        v = torch.zeros(n_dim, device=device, dtype=dtype)
        v[i] = 1.0
        v = v - torch.dot(v, u0) * u0
        for b in basis:
            v = v - torch.dot(v, b) * b
        norm = torch.linalg.norm(v)
        if norm > 1e-8:
            basis.append(v / norm)
        if len(basis) == 2:
            break
    if len(basis) < 2:
        raise ValueError("Failed to construct orthonormal basis for the slice plane")
    return torch.stack(basis, dim=0)


def main():
    args = parse_args()

    map_location = None
    if args.device == "auto":
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        map_location = args.device

    ckpt = torch.load(args.ckpt, map_location=map_location, weights_only=False)
    model, n_dim = build_model(ckpt, args)

    k = float(args.k) if args.k is not None else float(ckpt.get("k"))
    if args.margin is None:
        margin = 0.1 * k
    else:
        margin = float(args.margin)

    eps = ckpt.get("eps")
    eps_tag = f"{eps:g}" if eps is not None else "na"

    device = torch.device(map_location)
    model.to(device=device, dtype=torch.float32)
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)

    # Axis line plot: x = (s/n, ..., s/n), s in [-k,k] (+ margin)
    s_min = -k - margin
    s_max = k + margin
    s_vals = torch.linspace(s_min, s_max, int(args.line_points), device=device).view(-1, 1)
    x_line = (s_vals / float(n_dim)) * torch.ones(s_vals.shape[0], n_dim, device=device)

    with torch.no_grad():
        u_pred_line = model(x_line)
        u_true_line = ground_truth(x_line, k, args.truth)
        err_line = compute_error(u_pred_line, u_true_line, args.metric)

    s_cpu = s_vals.squeeze().cpu().numpy()
    u_pred_cpu = u_pred_line.squeeze().cpu().numpy()
    u_true_cpu = u_true_line.squeeze().cpu().numpy()
    err_cpu = err_line.squeeze().cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax1.plot(s_cpu, u_true_cpu, "k--", label="u_true")
    ax1.plot(s_cpu, u_pred_cpu, label="u_pred")
    ax1.set_ylabel("u")
    ax1.legend()

    ax2.plot(s_cpu, err_cpu, color="tab:red")
    ax2.axhline(0.0, color="k", linewidth=0.5)
    ax2.set_xlabel("s (sum x_i)")
    ax2.set_ylabel("error")
    ax2.set_title(f"Axis line error ({args.metric}, truth={args.truth})")
    fig.tight_layout()

    line_name = f"axis_line_{args.metric}_n{n_dim}_k{format_tag(k)}_eps{eps_tag}.png"
    line_path = os.path.join(args.out_dir, line_name)
    fig.savefig(line_path, dpi=200)
    print(f"Saved axis line plot: {line_path}")

    if args.show:
        plt.show(block=False)
    plt.close(fig)

    # 2D slice heatmap
    g = int(args.grid)
    _, r_k = cylindrical_params(n_dim, k)
    lim = r_k + margin
    xs = torch.linspace(-lim, lim, g, device=device)
    x1, x2 = torch.meshgrid(xs, xs, indexing="ij")
    basis = orthonormal_plane_basis(n_dim, device=device, dtype=torch.float32)
    v1 = basis[0].view(1, n_dim)
    v2 = basis[1].view(1, n_dim)
    s0 = float(args.axis_s)
    m = (s0 / float(n_dim)) * torch.ones(n_dim, device=device, dtype=torch.float32)
    x_flat = m + x1.reshape(-1, 1) * v1 + x2.reshape(-1, 1) * v2

    with torch.no_grad():
        u_pred = model(x_flat)
        u_true = ground_truth(x_flat, k, args.truth)
        diff = compute_error(u_pred, u_true, args.metric).view(g, g)
        d = distance_to_cylindrical_domain(x_flat, k).view(g, g)

    diff_np = diff.cpu().numpy()
    d_np = d.cpu().numpy()
    diff_np = np.where(d_np <= 1e-6, diff_np, np.nan)

    vmax = args.vmax
    if vmax is None:
        vmax = np.nanmax(np.abs(diff_np))
    vmin = -vmax

    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        diff_np,
        origin="lower",
        extent=[-lim, lim, -lim, lim],
        cmap=args.cmap,
        vmin=vmin,
        vmax=vmax,
    )
    label = "u_pred - u_true" if args.metric == "diff" else "(u_pred - u_true)/|u_true|"
    plt.colorbar(im, label=label)
    plt.xlabel("b1")
    plt.ylabel("b2")
    plt.title(f"2D orthogonal slice (s={args.axis_s:g}, {args.metric}, truth={args.truth})")
    plt.tight_layout()

    s_tag = format_tag(args.axis_s)
    slice_name = f"slice2d_orth_{args.metric}_n{n_dim}_k{format_tag(k)}_s{s_tag}_eps{eps_tag}.png"
    slice_path = os.path.join(args.out_dir, slice_name)
    plt.savefig(slice_path, dpi=200)
    print(f"Saved 2D slice heatmap: {slice_path}")

    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
