import math
from typing import Tuple

import torch
from torch import Tensor


def cylindrical_params(n_dim: int, k: float) -> Tuple[float, float]:
    """
    Return (a_max, r_k) for the enlarged cylindrical domain Ω_k.

    a_max = k / sqrt(n)  (axis half-length in the 1/√n direction)
    r_k   = k / sqrt(n (n-1))  (cross-section radius)
    """
    if n_dim < 2:
        raise ValueError("Cylindrical domain requires n_dim >= 2")
    n = float(n_dim)
    a_max = k / math.sqrt(n)
    r_k = k / math.sqrt(n * (n - 1.0))
    return a_max, r_k


def _axis_and_radial_norm(x: Tensor) -> Tuple[Tensor, Tensor]:
    """Decompose x into axis coordinate a and radial norm ||v||."""
    n_dim = x.shape[1]
    s = torch.sum(x, dim=1, keepdim=True)
    a = s / math.sqrt(float(n_dim))
    norm2 = torch.sum(x * x, dim=1, keepdim=True)
    v_norm_sq = torch.clamp(norm2 - a * a, min=0.0)
    v_norm = torch.sqrt(v_norm_sq)
    return a, v_norm


def distance_to_cylindrical_domain(x: Tensor, k: float) -> Tensor:
    """
    Distance to Ω_k defined by |sum x_i| <= k and Q(x) <= k^2 / (n(n-1)).

    Args:
      x: (N, n_dim) points
      k: scalar controlling domain size

    Returns:
      (N, 1) tensor of Euclidean distances
    """
    n_dim = x.shape[1]
    a_max, r_k = cylindrical_params(n_dim, k)
    a, v_norm = _axis_and_radial_norm(x)

    a_max_t = torch.as_tensor(a_max, device=x.device, dtype=x.dtype)
    r_k_t = torch.as_tensor(r_k, device=x.device, dtype=x.dtype)

    d_parallel = torch.clamp(torch.abs(a) - a_max_t, min=0.0)
    d_perp = torch.clamp(v_norm - r_k_t, min=0.0)
    d = torch.sqrt(d_parallel ** 2 + d_perp ** 2)
    return d


def sample_cylindrical_domain(
    n_samples: int,
    n_dim: int,
    k: float,
    margin: float = 0.0,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> Tensor:
    """
    Uniform sampler for Ω_k (and an expanded margin around it).

    Uses the procedure from guide.md: sample axis coordinate s ∈ [-k,k],
    project Gaussian noise to the sum=0 hyperplane for a random direction,
    and sample radius with the correct (n-1)-ball density. If margin>0,
    both axis range and cross-section radius are enlarged by margin to expose
    points outside the domain to the penalization term.
    """
    if n_dim < 2:
        raise ValueError("n_dim must be >= 2 for the cylindrical domain")
    device = device or torch.device("cpu")
    dtype = dtype or torch.float32

    a_max, r_k = cylindrical_params(n_dim, k)
    axis_half = float(k) + float(margin)
    r_max = float(r_k) + float(margin)

    s = torch.empty(n_samples, 1, device=device, dtype=dtype).uniform_(-axis_half, axis_half)

    w = torch.randn(n_samples, n_dim, device=device, dtype=dtype)
    mean_w = w.mean(dim=1, keepdim=True)
    v = w - mean_w  # project to {sum x_i = 0}
    v_norm = torch.linalg.norm(v, dim=1, keepdim=True)
    v_norm = torch.clamp(v_norm, min=1e-12)
    v_hat = v / v_norm

    U = torch.rand(n_samples, 1, device=device, dtype=dtype)
    rho = r_max * U.pow(1.0 / (n_dim - 1.0))
    u = rho * v_hat

    m = (s / float(n_dim)) * torch.ones_like(u)
    x = m + u
    return x


def sample_box_domain(
    n_samples: int,
    n_dim: int,
    k: float,
    margin: float = 0.0,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> Tensor:
    """Uniformly sample in the hypercube [-k-margin, k+margin]^n."""
    device = device or torch.device("cpu")
    dtype = dtype or torch.float32
    half = float(k) + float(margin)
    low = -half
    high = half
    x = torch.empty(n_samples, n_dim, device=device, dtype=dtype).uniform_(low, high)
    return x
