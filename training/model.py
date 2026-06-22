"""PyTorch model definition for 1D dynamics learning."""

from __future__ import annotations

import torch
from torch import nn

from training.dataset import INPUT_DIM


class DynamicsMLP(nn.Module):
    """Small MLP that predicts acceleration from a flattened history window."""

    def __init__(self, input_dim: int = INPUT_DIM, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return acceleration predictions."""

        return self.network(inputs).squeeze(-1)
