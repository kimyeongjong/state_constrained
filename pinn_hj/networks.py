import torch
import torch.nn as nn
from typing import Sequence


def _act(name: str):
    name = (name or "tanh").lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unknown activation: {name}")


class MLP(nn.Module):
    """
    Simple MLP for PINNs.
    """
    def __init__(self, in_dim: int, out_dim: int = 1, widths: Sequence[int] = (128, 128, 128), act: str = "tanh"):
        super().__init__()
        layers = []
        last = in_dim
        for w in widths:
            layers += [nn.Linear(last, w), _act(act)]
            last = w
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

