from __future__ import annotations

from typing import Optional, Union

import torch


def resolve_device(device: Union[str, torch.device]) -> torch.device:
    """
    Resolve a device string or torch.device into a torch.device instance.
    """
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def set_seed(seed: Optional[int]) -> None:
    """
    Set torch (and numpy if available) RNG seeds for reproducibility.
    """
    if seed is None:
        return
    torch.manual_seed(int(seed))
    try:
        import numpy as np

        np.random.seed(int(seed))
    except Exception:
        # NumPy is optional in some environments.
        pass
