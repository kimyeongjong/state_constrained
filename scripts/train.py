import argparse
import os
import shutil
import sys
from datetime import datetime
from typing import Callable, Optional

import torch
import yaml

# Make top-level package importable when running as a script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pinn_hj.networks import MLP
from pinn_hj.hamiltonians import ShiftedL1Hamiltonian
from pinn_hj.trainers import TrainConfig, compute_residual, train_for_epsilon
from pinn_hj.cylinder_domain import distance_to_cylindrical_domain, sample_box_domain
from pinn_hj.config import parse_args_with_config
from pinn_hj.utils import resolve_device, set_seed
from scripts.plot import plot_after_epsilon


def exact_solution_1d(x: torch.Tensor, k: float) -> torch.Tensor:
    # u(x) = exp(x - k), inside [-k, k]
    return torch.exp(x - float(k))


def _build_parser(defaults=None):
    defaults = defaults or {}
    p = argparse.ArgumentParser(description="State-constrained HJ via PINNs (unified)")
    p.add_argument("--config", type=str, default=defaults.get("config"), help="YAML config path")
    p.add_argument("--n", type=int, default=defaults.get("n", 1), help="Dimension n")
    p.add_argument(
        "--domain",
        choices=["auto", "box", "cylinder"],
        default=defaults.get("domain", "auto"),
        help="Domain type (auto: box for n=1, cylinder for n>=2)",
    )
    p.add_argument(
        "--eval",
        choices=["auto", "exact", "residual"],
        default=defaults.get("eval", "auto"),
        help="Evaluation mode (auto: exact for 1D box, else residual)",
    )
    p.add_argument("--k", type=float, default=defaults.get("k", 2.0), help="Domain scale k")
    p.add_argument("--eps", type=float, nargs="+", default=defaults.get("eps", [0.5, 0.2, 0.1, 0.05]), help="Epsilon schedule")
    p.add_argument("--steps", type=int, default=defaults.get("steps", 4000), help="Training steps per epsilon")
    p.add_argument("--batch", type=int, default=defaults.get("batch", 2048), help="Collocation batch size")
    p.add_argument("--lr", type=float, default=defaults.get("lr", 1e-3), help="Learning rate")
    p.add_argument("--width", type=int, default=defaults.get("width", 128), help="Hidden layer width")
    p.add_argument("--layers", type=int, default=defaults.get("layers", 3), help="Number of hidden layers")
    p.add_argument("--act", type=str, default=defaults.get("act", "tanh"), help="Activation (tanh|silu|gelu|relu)")
    p.add_argument(
        "--margin",
        type=float,
        default=defaults.get("margin"),
        help="Sampling margin outside domain (default: 0.5*k for box, 0 for cylinder)",
    )
    p.add_argument("--alpha", type=float, default=defaults.get("alpha", 0.5), help="Quadratic tail curvature for phi(t)")
    p.add_argument("--device", type=str, default=defaults.get("device", "auto"), help="cpu|cuda|auto")
    p.add_argument("--seed", type=int, default=defaults.get("seed", 1234), help="Random seed")
    p.add_argument("--eval-points", type=int, default=defaults.get("eval_points", 400), help="Evaluation points")
    p.add_argument("--log-every", type=int, default=defaults.get("log_every", 200), help="Log interval (steps)")
    return p


def parse_args(argv=None):
    return parse_args_with_config(argv, _build_parser)


def _resolve_domain(name: str, n_dim: int) -> str:
    if name == "auto":
        return "box" if n_dim == 1 else "cylinder"
    if name == "cylinder" and n_dim < 2:
        raise ValueError("cylinder domain requires n >= 2")
    return name


def _resolve_eval(name: str, n_dim: int, domain: str) -> str:
    if name == "auto":
        return "exact" if (n_dim == 1 and domain == "box") else "residual"
    return name


def _save_config_snapshot(run_dir: str, args: argparse.Namespace) -> None:
    config_dir = os.path.join(run_dir, "config")
    os.makedirs(config_dir, exist_ok=True)

    if args.config:
        src = args.config
        dst = os.path.join(config_dir, "config_original.yaml")
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            print(f"Warning: failed to copy config file: {e}")

    resolved = {k: v for k, v in vars(args).items() if k != "config"}
    out_path = os.path.join(config_dir, "config_resolved.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved, f, sort_keys=False)




def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)

    device = resolve_device(args.device)
    dtype = torch.float32

    n_dim = int(args.n)
    k = float(args.k)
    domain = _resolve_domain(args.domain, n_dim)
    eval_mode = _resolve_eval(args.eval, n_dim, domain)

    if args.margin is None:
        margin = 0.5 * k if domain == "box" else 0.0
    else:
        margin = float(args.margin)

    plot_enabled = False
    save_enabled = True
    save_ckpt_enabled = True

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    k_tag = f"{int(k)}" if float(k).is_integer() else f"{k:g}"
    run_dir = os.path.join("results", f"{ts}_n{n_dim}_k{k_tag}")
    save_dir = os.path.join(run_dir, "plots")
    ckpt_dir = os.path.join(run_dir, "checkpoints")

    _save_config_snapshot(run_dir, args)

    widths = tuple([args.width] * int(args.layers))
    model = MLP(in_dim=n_dim, out_dim=1, widths=widths, act=args.act).to(device=device, dtype=dtype)
    H = ShiftedL1Hamiltonian(quad_alpha=float(args.alpha))

    cfg = TrainConfig(
        steps=int(args.steps),
        batch_size=int(args.batch),
        lr=float(args.lr),
        outer_margin=float(margin),
        eval_points=int(args.eval_points),
        log_every=int(args.log_every),
        device=args.device,
    )

    # Placeholders used by trainer API
    low = torch.full((n_dim,), -k, device=device, dtype=dtype)
    high = torch.full((n_dim,), +k, device=device, dtype=dtype)

    if domain == "box":
        sample_fn = lambda n, low, high, m, dev, dt: sample_box_domain(n, n_dim, k, margin=m, device=dev, dtype=dt)
        distance_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    else:
        # Train on a superset box; use cylindrical distance in the residual.
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

        if eval_mode == "exact":
            if n_dim != 1:
                raise ValueError("exact evaluation is only supported for 1D")
            x_eval = torch.linspace(-k, k, int(cfg.eval_points), device=device, dtype=dtype).view(-1, 1)
            with torch.no_grad():
                u_pred = model(x_eval)
                u_true = exact_solution_1d(x_eval, k)
                mse = torch.mean((u_pred - u_true) ** 2).item()
            print(f"[eps={eps:g}] MSE on [-k,k]: {mse:.6e}")
        else:
            x_eval = sample_box_domain(cfg.eval_points, n_dim, k, margin=0.0, device=device, dtype=dtype)
            r_eval, _ = compute_residual(
                model,
                H,
                x_eval,
                low,
                high,
                eps,
                distance_fn=distance_fn,
                create_graph=False,
            )
            res_mse = torch.mean(r_eval ** 2).item()
            print(f"[eps={eps:g}] residual MSE on eval set: {res_mse:.6e}")

        if plot_enabled or save_enabled:
            try:
                plot_after_epsilon(
                    n_dim=n_dim,
                    k=k,
                    eps=eps,
                    domain=domain,
                    model=model,
                    hamiltonian=H,
                    low=low,
                    high=high,
                    distance_fn=distance_fn,
                    compute_residual=compute_residual,
                    save=save_enabled,
                    save_dir=save_dir,
                    show=plot_enabled,
                    eval_points=int(cfg.eval_points),
                    device=device,
                    dtype=dtype,
                    exact_solution_1d=exact_solution_1d,
                )
            except Exception as e:
                print(f"Plotting failed: {e}")

        if save_ckpt_enabled:
            os.makedirs(ckpt_dir, exist_ok=True)
            fname = f"pinn_n{n_dim}_k{k:g}_eps{eps:g}.pt"
            path = os.path.join(ckpt_dir, fname)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "arch": {
                        "in_dim": n_dim,
                        "out_dim": 1,
                        "widths": widths,
                        "act": args.act,
                    },
                    "hamiltonian": {"type": "l1-quad-tail", "quad_alpha": float(args.alpha)},
                    "eps": eps,
                    "n": n_dim,
                    "k": k,
                    "domain": domain,
                    "config": cfg.__dict__,
                    "seed": args.seed,
                },
                path,
            )
            print(f"Saved checkpoint: {path}")


if __name__ == "__main__":
    main()
