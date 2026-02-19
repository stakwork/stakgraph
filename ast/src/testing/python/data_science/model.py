
import torch
import torch.nn as nn
from torch.nn import functional as F

class FraudDetector(nn.Module):
    """
    A PyTorch model for detecting fraudulent transactions.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super(FraudDetector, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self._threshold = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc2(x))

    @property
    def threshold(self) -> float:
        """Get the decision threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float):
        if not 0 <= value <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        self._threshold = value
