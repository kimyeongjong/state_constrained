import argparse
import os
import sys
from typing import List

import torch
import numpy as np

# Make top-level package importable when running as a script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pinn_hj.networks import MLP
from pinn_hj.hamiltonians import ShiftedAbsHamiltonianQuadTail
from pinn_hj.trainers import TrainConfig, train_for_epsilon
from pinn_hj.distance import distance_to_box
from pinn_hj.config import load_yaml_config


def exact_solution(x: torch.Tensor, k: float) -> torch.Tensor:
    # u(x) = exp(x - k), inside [-k, k]
    return torch.exp(x - float(k))


def _build_parser(defaults=None):
    defaults = defaults or {}
    p = argparse.ArgumentParser(description="State-constrained HJ via PINNs (1D)")
    p.add_argument("--config", type=str, default=defaults.get("config"), help="YAML config path")
    p.add_argument("--k", type=float, default=defaults.get("k", 2.0), help="Domain half-width: Ω = [-k, k]")
    p.add_argument("--eps", type=float, nargs="+", default=defaults.get("eps", [0.5, 0.2, 0.1, 0.05]), help="Epsilon schedule")
    p.add_argument("--steps", type=int, default=defaults.get("steps", 4000), help="Training steps per epsilon")
    p.add_argument("--batch", type=int, default=defaults.get("batch", 2048), help="Collocation batch size")
    p.add_argument("--lr", type=float, default=defaults.get("lr", 1e-3), help="Learning rate")
    p.add_argument("--width", type=int, default=defaults.get("width", 128), help="Hidden layer width")
    p.add_argument("--layers", type=int, default=defaults.get("layers", 3), help="Number of hidden layers")
    p.add_argument("--act", type=str, default=defaults.get("act", "tanh"), help="Activation (tanh|silu|gelu|relu)")
    p.add_argument("--margin", type=float, default=defaults.get("margin", None), help="Sampling margin outside Ω (default 0.5*k)")
    p.add_argument("--alpha", type=float, default=defaults.get("alpha", 0.5), help="Quadratic tail curvature for H")
    p.add_argument("--device", type=str, default=defaults.get("device", "auto"), help="cpu|cuda|auto")
    p.add_argument("--seed", type=int, default=defaults.get("seed", 1234), help="Random seed")
    p.add_argument("--plot", action="store_true", default=defaults.get("plot", False), help="Plot predictions after each epsilon")
    p.add_argument("--save", action="store_true", default=defaults.get("save", False), help="Save plots after each epsilon")
    p.add_argument("--save-dir", type=str, default=defaults.get("save_dir", "plots"), help="Directory to save plots")
    return p


def parse_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    cfg_args, _ = pre.parse_known_args(argv)
    cfg = load_yaml_config(cfg_args.config) if cfg_args.config else {}
    parser = _build_parser(cfg)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device and dtype
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    dtype = torch.float32

    # Domain Ω = [-k, k] in 1D, and sampling box expansion margin
    k = float(args.k)
    low = torch.tensor([-k], dtype=dtype)
    high = torch.tensor([+k], dtype=dtype)
    margin = args.margin if args.margin is not None else 1.0 * k

    # PINN model
    widths = tuple([args.width] * int(args.layers))
    model = MLP(in_dim=1, out_dim=1, widths=widths, act=args.act).to(device=device, dtype=dtype)

    # Hamiltonian with smooth convex quadratic tails
    H = ShiftedAbsHamiltonianQuadTail(quad_alpha=args.alpha)

    # Training config
    cfg = TrainConfig(
        steps=int(args.steps),
        batch_size=int(args.batch),
        lr=float(args.lr),
        outer_margin=float(margin),
        device=args.device,
    )

    # Outer loop over epsilons, warm-starting the same model
    for eps in args.eps:
        train_for_epsilon(model, H, eps, low, high, cfg)

        # Evaluate MSE against exact solution on a fine grid inside Ω
        n_eval = 400
        x_eval = torch.linspace(-k, k, n_eval, device=device, dtype=dtype).view(-1, 1)
        with torch.no_grad():
            u_pred = model(x_eval)
            u_true = exact_solution(x_eval, k)
            mse = torch.mean((u_pred - u_true) ** 2).item()
        print(f"[eps={eps:g}] MSE on [-k,k]: {mse:.6e}")

        if args.plot or args.save:
            try:
                import matplotlib.pyplot as plt
                x_cpu = x_eval.squeeze().cpu().numpy()
                u_pred_cpu = u_pred.squeeze().cpu().numpy()
                u_true_cpu = u_true.squeeze().cpu().numpy()

                # Non-blocking plotting
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(x_cpu, u_true_cpu, 'k--', label='Exact: exp(x-k)')
                ax.plot(x_cpu, u_pred_cpu, label=f'PINN (eps={eps:g})')
                ax.set_title('HJ solution in [-k,k]')
                ax.set_xlabel('x')
                ax.set_ylabel('u(x)')
                ax.legend()
                fig.tight_layout()

                if args.save:
                    os.makedirs(args.save_dir, exist_ok=True)
                    fname = f"hj_pinn_k{int(k) if float(k).is_integer() else k}_eps_{eps:g}.png"
                    out_path = os.path.join(args.save_dir, fname)
                    fig.savefig(out_path, dpi=150)
                    print(f"Saved plot to: {out_path}")

                if args.plot:
                    try:
                        plt.ion()
                        # Show and refresh without blocking
                        try:
                            fig.show()
                        except Exception:
                            pass
                        fig.canvas.draw()
                        fig.canvas.flush_events()
                        plt.pause(0.001)
                    except Exception:
                        # Fallback: ensure we don't block training
                        pass
                else:
                    plt.close(fig)
            except Exception as e:
                print(f"Plotting failed: {e}")


if __name__ == "__main__":
    main()
