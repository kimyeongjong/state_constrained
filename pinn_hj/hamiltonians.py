import torch
from torch import Tensor


class BaseHamiltonian:
    """
    Base class for Hamiltonians H(x, p). Subclass and implement __call__.
    """
    # If True, compute_residual will pass ||Du|| instead of full Du for D>1.
    use_grad_norm: bool = False

    def __call__(self, x: Tensor, p: Tensor) -> Tensor:
        raise NotImplementedError


class ShiftedL1Hamiltonian(BaseHamiltonian):
    """
    H(x, p) = (1/n) * sum_i phi(p_i) - 1, where

      phi(t) = |t-1|                     for t in [0, 2]
             = a(t-2)^2 + (t-2)          for t > 2
             = a t^2 - t                 for t < 0

    This is C^1, convex, and coercive for any a > 0.
    """
    def __init__(self, quad_alpha: float = 0.5):
        self.quad_alpha = float(quad_alpha)

    def __call__(self, x: Tensor, p: Tensor) -> Tensor:
        if p.ndim == 1:
            p = p.view(-1, 1)
        a = self.quad_alpha

        inside = (p >= 0.0) & (p <= 2.0)
        left = p < 0.0
        right = p > 2.0

        h_inside = torch.abs(p - 1.0)
        h_left = a * (p ** 2) - p
        h_right = a * ((p - 2.0) ** 2) + (p - 2.0)

        h = torch.where(inside, h_inside, torch.zeros_like(p))
        h = torch.where(left, h_left, h)
        h = torch.where(right, h_right, h)

        n = p.shape[1]
        return h.sum(dim=1, keepdim=True) / float(n) - 1.0
