import torch
from torch import Tensor


def _ensure_nd(b: Tensor, n: int) -> Tensor:
    if b.ndim == 1:
        return b.unsqueeze(0).expand(n, -1)
    if b.ndim == 2 and b.shape[0] == 1:
        return b.expand(n, -1)
    return b


def distance_to_box(x: Tensor, low: Tensor, high: Tensor) -> Tensor:
    """
    Euclidean distance from each x_i to an axis-aligned box [low, high] in R^d.

    Args:
      x:    (N, D) points
      low:  (D,) lower bounds
      high: (D,) upper bounds

    Returns:
      d: (N, 1) distances
    """
    assert x.ndim == 2
    N, D = x.shape

    low = torch.as_tensor(low, dtype=x.dtype, device=x.device).view(-1)
    high = torch.as_tensor(high, dtype=x.dtype, device=x.device).view(-1)
    assert low.shape[0] == D and high.shape[0] == D

    lowN = _ensure_nd(low, N)
    highN = _ensure_nd(high, N)

    below = torch.clamp(lowN - x, min=0.0)
    above = torch.clamp(x - highN, min=0.0)
    delta = below + above  # (N, D); zero inside the box
    d = torch.linalg.norm(delta, dim=1, keepdim=True)
    return d

