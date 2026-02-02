import argparse
import os
import sys

import numpy as np
import torch

# Make top-level package importable when running as a script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pinn_hj.networks import MLP
from pinn_hj.hamiltonians import ShiftedAbsHamiltonianQuadTail, ShiftedL1Hamiltonian
from pinn_hj.trainers import TrainConfig, compute_residual, train_for_epsilon
from pinn_hj.cylinder_domain import distance_to_cylindrical_domain, sample_box_domain


def parse_args():
    p = argparse.ArgumentParser(description="State-constrained HJ via PINNs on cylindrical Ω_k in R^n")
    p.add_argument("--n", type=int, default=3, help="Dimension n (>=2)")
    p.add_argument("--k", type=float, default=2.0, help="Domain scale k for Ω_k")
    p.add_argument("--eps", type=float, nargs="+", default=[0.5, 0.2, 0.1, 0.05], help="Epsilon schedule")
    p.add_argument("--steps", type=int, default=4000, help="Training steps per epsilon")
    p.add_argument("--batch", type=int, default=2048, help="Collocation batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--width", type=int, default=128, help="Hidden layer width")
    p.add_argument("--layers", type=int, default=3, help="Number of hidden layers")
    p.add_argument("--act", type=str, default="tanh", help="Activation (tanh|silu|gelu|relu)")
    p.add_argument("--margin", type=float, default=None, help="Sampling margin outside cube [-k,k]^n (default 0)")
    p.add_argument("--alpha", type=float, default=0.5, help="Quadratic tail curvature for H")
    p.add_argument("--device", type=str, default="auto", help="cpu|cuda|auto")
    p.add_argument("--seed", type=int, default=1234, help="Random seed")
    p.add_argument("--save-ckpt", action="store_true", help="Save model checkpoint after each epsilon")
    p.add_argument("--ckpt-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    dtype = torch.float32

    n_dim = int(args.n)
    k = float(args.k)
    margin = args.margin if args.margin is not None else 0.0

    widths = tuple([args.width] * int(args.layers))
    model = MLP(in_dim=n_dim, out_dim=1, widths=widths, act=args.act).to(device=device, dtype=dtype)

    if n_dim >= 2:
        H = ShiftedL1Hamiltonian()
    else:
        H = ShiftedAbsHamiltonianQuadTail(quad_alpha=args.alpha)

    cfg = TrainConfig(
        steps=int(args.steps),
        batch_size=int(args.batch),
        lr=float(args.lr),
        outer_margin=float(margin),
        device=args.device,
    )

    # Placeholders (not used by the sampler) to keep trainer API consistent
    low = torch.full((n_dim,), -k, device=device, dtype=dtype)
    high = torch.full((n_dim,), +k, device=device, dtype=dtype)

    sample_fn = lambda n, low, high, m, dev, dt: sample_box_domain(n, n_dim, k, margin=m, device=dev, dtype=dt)
    distance_fn = lambda x: distance_to_cylindrical_domain(x, k)

    for eps in args.eps:
        train_for_epsilon(
            model,
            H,
            eps,
            low,
            high,
            cfg,
            sample_fn=sample_fn,
            distance_fn=distance_fn,
        )

        # Evaluate residual on Ω_k to monitor progress (no closed-form solution assumed)
        x_eval = sample_box_domain(cfg.eval_points, n_dim, k, margin=0.0, device=device, dtype=dtype)
        r_eval, _ = compute_residual(model, H, x_eval, low, high, eps, distance_fn=distance_fn)
        res_mse = torch.mean(r_eval ** 2).item()
        print(f"[eps={eps:g}] residual MSE on Ω_k: {res_mse:.6e}")

        if args.save_ckpt:
            os.makedirs(args.ckpt_dir, exist_ok=True)
            fname = f"pinn_cylinder_n{n_dim}_k{k:g}_eps{eps:g}.pt"
            path = os.path.join(args.ckpt_dir, fname)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "arch": {
                        "in_dim": n_dim,
                        "out_dim": 1,
                        "widths": widths,
                        "act": args.act,
                    },
                    "hamiltonian": {"type": "ShiftedAbsHamiltonianQuadTail", "quad_alpha": args.alpha},
                    "eps": eps,
                    "n": n_dim,
                    "k": k,
                    # Store config as plain dict to avoid pickling custom classes on load
                    "config": cfg.__dict__,
                    "seed": args.seed,
                },
                path,
            )
            print(f"Saved checkpoint: {path}")


if __name__ == "__main__":
    main()
