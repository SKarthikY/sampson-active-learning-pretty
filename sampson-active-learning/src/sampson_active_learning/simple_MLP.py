import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleFluxMLP(nn.Module):
    """
    Simple MLP mapping physical parameters -> flux spectrum.

    Args:
        n_physical_param: input dimension (number of physical parameters)
        n_wavelength: output dimension (number of wavelength points)
        d_model: hidden dimension
        num_layers: total number of linear layers (>= 2)
        nhead, learnedPE: kept for API compatibility, not used
    """
    def __init__(self,
                 n_physical_param=10,
                 n_wavelength=602,
                 d_model=128,
                 nhead=8,
                 num_layers=4,
                 learnedPE=True):
        super().__init__()
        assert num_layers >= 2, "num_layers must be at least 2 for an input and output layer."

        layers = []

        # Input layer
        layers.append(nn.Linear(n_physical_param, d_model))
        layers.append(nn.GELU())

        # Hidden layers: num_layers - 2 internal linear layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(d_model, d_model))
            layers.append(nn.GELU())

        # Output layer
        layers.append(nn.Linear(d_model, n_wavelength))

        self.net = nn.Sequential(*layers)

    def forward(self, physical_param):
        """
        physical_param: [B, n_physical_param]
        Returns: [B, n_wavelength]
        """
        return self.net(physical_param)