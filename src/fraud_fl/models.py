"""MLP classifier used as the federated client / global model."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class FraudMLP(nn.Module):
    """Feed-forward classifier for the 29-d fraud-detection features."""

    def __init__(self, input_dim: int = 29,
                 hidden: Iterable[int] = (64, 32, 16),
                 dropout: float = 0.3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.GroupNorm(1, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)

    def count_params(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())


def to_tensor(df: pd.DataFrame, features: list[str],
              target: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a pandas frame to (X, y) float32 tensors."""
    X = torch.tensor(df[features].values, dtype=torch.float32)
    y = torch.tensor(df[target].values.astype(np.float32),
                     dtype=torch.float32).unsqueeze(1)
    return X, y
