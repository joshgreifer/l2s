from __future__ import annotations

from typing import Dict, List

import joblib
import torch
from torch import nn
import torch.nn.functional as F

from pkg.config import Config
from pkg.gaze_model import GazeModel




class GazePCA(GazeModel):
    def __init__(self, config, *, logger=None, filename=None):
        super().__init__(config, logger, filename=filename)

        assert (config.version == 300), "GazePCA is only compatible with config version 300"
        try:
            self.pca = joblib.load(config.pca_path)
            print(f"PCA model loaded from {config.pca_path}")

        except FileNotFoundError:
            raise FileNotFoundError(f"PCA model not found at {config.pca_path}. Please ensure the PCA model is created and saved before using GazePCA.")

        except Exception as e:
            raise RuntimeError(f"Error loading PCA model from {config.pca_path}: {e}")

        self.pca.
        hidden_channels = config.hidden_channels
        self.points_of_interest_indices = config.points_of_interest_indices
        n_points = len(self.points_of_interest_indices)
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, padding_mode="reflect", bias=False)
            for dilation in range(2, 4)  # Dilation sizes from 1 to 5
        ])
        self.input_conv = nn.Conv1d(3, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False)  # Initial conv layer
        self.bn = nn.BatchNorm1d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)
        self.last_convs =  nn.Sequential(
            nn.Conv1d(hidden_channels, 4, kernel_size=3, stride=1, padding=0, bias=False),
            nn.Conv1d(4, 2, kernel_size=3, stride=1, padding=0, bias=False),
            nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=0, bias=False)
        )
        self.last_act = torch.nn.Tanh()
        self.config = config

        if filename is not None:
            self.load(filename)

    def forward(self, x):
        batch_size, num_nodes, in_channels = x.shape
        assert num_nodes == 478, f"Expected 478 nodes, got {num_nodes}"
        assert in_channels == 3, f"Expected 3 channels, got {in_channels}"

        points_of_interest = [x[:, i, :] for i in self.points_of_interest_indices]
        x = torch.stack(points_of_interest, dim=1)  # Shape: [B, 8, 3]
        x = x.permute(0, 2, 1)  # Change shape to [B, 3, 8] for Conv1d

        x = self.input_conv(x)  # Initial conv layer
        # x = self.bn(x)
        x = self.relu(x)

        outputs = [conv(x) for conv in self.convs]  # Apply convolutions with different dilations
        x = sum(outputs)  # Sum all outputs element-wise
        x = self.last_convs(x)  # Fully connected layer, shape: [B, 2]
        x = self.last_act(x)
        return torch.squeeze(x)



if __name__ == '__main__':
    x = torch.randn(478, 3)
    x = torch.unsqueeze(x, 0)
    model = GazePCA(Config())
    model.eval()
    y = model(x)
    print(y.shape)



