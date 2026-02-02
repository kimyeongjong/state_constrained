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


class ShiftedAbsHamiltonianQuadTail(BaseHamiltonian):
    """
    H(x, p) = |p - 1| - 1 for p ∈ [0, 2], extended smoothly and convexly outside that interval.

    Quadratic tails:
      - For p < 0:   H(p) = a p^2 - p,     with H(0) = 0 and H'(0) = -1  (matches inside)
      - For p > 2:   H(p) = a (p-2)^2 + (p-2), with H(2) = 0 and H'(2) = +1 (matches inside)

    For any a > 0, H is convex and C^1 at p=0 and p=2.
    """
    use_grad_norm: bool = True

    def __init__(self, quad_alpha: float = 0.5):
        self.quad_alpha = float(quad_alpha)

    def __call__(self, x: Tensor, p: Tensor) -> Tensor:
        # p: (N, 1) or (N,)
        p_flat = p.view(-1)
        a = self.quad_alpha

        inside_mask = (p_flat >= 0.0) & (p_flat <= 2.0)
        left_mask   = p_flat < 0.0
        right_mask  = p_flat > 2.0

        h_inside = torch.abs(p_flat - 1.0) - 1.0
        h_left   = a * (p_flat ** 2) - p_flat
        h_right  = a * ((p_flat - 2.0) ** 2) + (p_flat - 2.0)

        h = torch.where(inside_mask, h_inside, torch.zeros_like(p_flat))
        h = torch.where(left_mask,   h_left,   h)
        h = torch.where(right_mask,  h_right,  h)

        return h.view(-1, 1)


class ShiftedL1Hamiltonian(BaseHamiltonian):
    """
    H(x, p) = ||p - 1||_1 / n - 1 for p in R^n (no tail smoothing).
    """
    def __call__(self, x: Tensor, p: Tensor) -> Tensor:
        if p.ndim == 1:
            p = p.view(-1, 1)
        n = p.shape[1]
        h = torch.abs(p - 1.0).sum(dim=1, keepdim=True) / float(n) - 1.0
        return h
