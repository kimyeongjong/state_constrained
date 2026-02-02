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
from pinn_hj.cylinder_domain import distance_to_cylindrical_domain


def parse_args():
    p = argparse.ArgumentParser(description="Heatmap of checkpoint vs ground truth for 2D Ω_k")
    p.add_argument("--ckpt", required=True, help="Checkpoint path (.pt)")
    p.add_argument("--k", type=float, default=None, help="Override k (default: read from checkpoint)")
    p.add_argument("--grid", type=int, default=201, help="Grid resolution per axis (odd recommended)")
    p.add_argument(
        "--margin",
        type=float,
        default=None,
        help="Extra margin around [-k,k] for plotting (default: 0.1*k)",
    )
    p.add_argument("--truth", choices=["axis-exp", "zero"], default="axis-exp", help="Ground truth function")
    p.add_argument(
        "--metric",
        choices=["diff", "rel"],
        default="diff",
        help="Error metric: diff = u_pred - u_true, rel = (u_pred - u_true)/|u_true|",
    )
    p.add_argument("--device", type=str, default="auto", help="cpu|cuda|auto")
    p.add_argument("--out", type=str, default=None, help="Output image path (default auto-named with n,k,eps)")
    p.add_argument("--show", action="store_true", help="Show plot interactively")
    p.add_argument("--cmap", type=str, default="coolwarm", help="Matplotlib colormap")
    p.add_argument("--vmax", type=float, default=None, help="Fix symmetric color scale (vmin=-vmax)")
    p.add_argument("--width", type=int, default=None, help="Fallback width if arch missing in ckpt")
    p.add_argument("--layers", type=int, default=None, help="Fallback layers if arch missing in ckpt")
    p.add_argument("--act", type=str, default=None, help="Fallback activation if arch missing in ckpt")
    return p.parse_args()


def build_model(ckpt, args):
    arch = ckpt.get("arch") or {}
    in_dim = arch.get("in_dim") or ckpt.get("n")
    if in_dim is None:
        raise ValueError("Checkpoint missing in_dim/n; please pass --width/--layers/--act manually with in_dim=2")
    if in_dim != 2:
        raise ValueError(f"Heatmap script expects 2D checkpoint, got in_dim={in_dim}")

    widths = arch.get("widths")
    act = arch.get("act")
    if widths is None or act is None:
        if args.width is None or args.layers is None:
            raise ValueError("Checkpoint lacks arch; provide --width and --layers (and optionally --act)")
        widths = tuple([args.width] * int(args.layers))
        act = args.act or "tanh"

    model = MLP(in_dim=2, out_dim=1, widths=tuple(widths), act=act)
    model.load_state_dict(ckpt["model_state"])
    return model


def ground_truth(x: torch.Tensor, k: float, mode: str) -> torch.Tensor:
    if mode == "axis-exp":
        a = (x[:, 0:1] + x[:, 1:2]) / math.sqrt(2.0)
        return torch.exp(a - k / math.sqrt(2.0))
    if mode == "zero":
        return torch.zeros_like(x[:, 0:1])
    raise ValueError(f"Unknown truth mode: {mode}")


def main():
    args = parse_args()

    map_location = None
    if args.device == "auto":
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        map_location = args.device

    ckpt = torch.load(args.ckpt, map_location=map_location, weights_only=False)
    n_dim = ckpt.get("n") or ckpt.get("arch", {}).get("in_dim") or 2
    eps = ckpt.get("eps")
    k = float(args.k) if args.k is not None else float(ckpt.get("k"))
    if args.margin is None:
        margin = 0.1 * k
    else:
        margin = float(args.margin)

    model = build_model(ckpt, args)
    device = torch.device(map_location)
    model.to(device=device, dtype=torch.float32)
    model.eval()

    g = int(args.grid)
    lim = k + margin
    xs = torch.linspace(-lim, lim, g, device=device)
    x1, x2 = torch.meshgrid(xs, xs, indexing="ij")
    x_flat = torch.stack([x1.reshape(-1), x2.reshape(-1)], dim=1)

    with torch.no_grad():
        u_pred = model(x_flat)
        u_true = ground_truth(x_flat, k, args.truth)
        raw_diff = u_pred - u_true
        if args.metric == "rel":
            denom = torch.clamp(torch.abs(u_true), min=1e-12)
            diff = (raw_diff / denom).view(g, g)
        else:
            diff = raw_diff.view(g, g)
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
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f"Heatmap ({args.metric}, k={k:g}, truth={args.truth})")

    if args.out:
        out_path = args.out
    else:
        eps_tag = f"{eps:g}" if eps is not None else "na"
        fname = f"heatmap_{args.metric}_n{int(n_dim)}_k{k:g}_eps{eps_tag}.png"
        out_path = os.path.join("plots", fname)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved heatmap: {out_path}")

    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
