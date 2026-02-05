from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
from torch import Tensor
from torch.optim import Adam
from tqdm import trange

from .distance import distance_to_box
from .utils import resolve_device


@dataclass
class TrainConfig:
    steps: int = 4000
    batch_size: int = 2048
    lr: float = 1e-3
    outer_margin: float = 1.0  # sampling margin outside the box
    eval_points: int = 200
    log_every: int = 200
    device: str = "auto"  # "cpu" | "cuda" | "auto" | torch.device


def compute_residual(
    model,
    hamiltonian,
    x: Tensor,
    low: Tensor,
    high: Tensor,
    eps: float,
    distance_fn: Optional[Callable[[Tensor], Tensor]] = None,
    create_graph: bool = True,
) -> Tuple[Tensor, Tensor]:
    """
    Compute residual r(x) = u + H(x, Du) - d(x,Ω)/ε and return (r, u).
    """
    x.requires_grad_(True)
    u = model(x)  # (N,1)
    grad_u = torch.autograd.grad(u.sum(), x, create_graph=create_graph)[0]  # (N,D)

    # For D=1 pass the scalar gradient; for D>1 pass Du unless the Hamiltonian requests ||Du||.
    if grad_u.shape[1] == 1:
        p = grad_u
    elif getattr(hamiltonian, "use_grad_norm", False):
        p = torch.linalg.norm(grad_u, dim=1, keepdim=True)
    else:
        p = grad_u

    H = hamiltonian(x, p)  # (N,1)
    if distance_fn is None:
        d = distance_to_box(x, low, high)  # (N,1)
    else:
        d = distance_fn(x)
    r = u + H - d / float(eps)
    return r, u


def sample_collocation(
    n: int,
    low: Tensor,
    high: Tensor,
    margin: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """
    Uniformly sample in the expanded box [low - margin, high + margin].
    """
    low = low.to(device=device, dtype=dtype)
    high = high.to(device=device, dtype=dtype)
    lo = low - margin
    hi = high + margin
    x = lo + (hi - lo) * torch.rand(n, low.shape[0], device=device, dtype=dtype)
    return x


def train_for_epsilon(
    model,
    hamiltonian,
    eps: float,
    low: Tensor,
    high: Tensor,
    cfg: TrainConfig,
    sample_fn: Callable[[int, Tensor, Tensor, float, torch.device, torch.dtype], Tensor] = sample_collocation,
    distance_fn: Optional[Callable[[Tensor], Tensor]] = None,
) -> List[Tuple[int, float]]:
    """
    Train the PINN for a fixed epsilon. Returns sparse (step, loss) logs.
    """
    device = resolve_device(cfg.device)
    model.to(device)
    low = low.to(device)
    high = high.to(device)

    opt = Adam(model.parameters(), lr=cfg.lr)
    losses: List[Tuple[int, float]] = []

    bar = trange(cfg.steps, desc=f"eps={eps:g}", leave=False)
    for t in bar:
        x = sample_fn(cfg.batch_size, low, high, cfg.outer_margin, device, next(model.parameters()).dtype)
        r, _ = compute_residual(
            model,
            hamiltonian,
            x,
            low,
            high,
            eps,
            distance_fn=distance_fn,
            create_graph=True,
        )
        loss = (r ** 2).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (t + 1) % cfg.log_every == 0:
            l = loss.item()
            losses.append((t + 1, l))
            bar.set_postfix(loss=f"{l:.3e}")
    return losses
