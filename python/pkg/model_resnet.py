from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn
import torch.nn.functional as F

from pkg.config import Config
from pkg.gaze_model import GazeModel


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               padding_mode='reflect',
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               padding_mode='reflect',
                               bias=False)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=stride,
                          padding=0,
                          bias=False))

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = F.relu(self.bn1(x), inplace=True)
        y = self.conv1(x)
        y = F.relu(self.bn2(y), inplace=True)
        y = self.conv2(y)
        y += self.shortcut(x)
        return y


class GazeResNet(GazeModel):
    """
    This model predicts the screen location (normalized to range [-1..1]
    of a gaze.
    It takes a 2d 3-channel input "image",
    where the channels are x,y,z coordinates of face landmarks, which have
    been obtained by Mediapipe face-landmarker.
    The Mediapipe landmarker identifies 478 landmarks, which ar padded with six more  3D points
    are then arranged into a 3-channel 22  X 22 Tensor (see pack_input()).
    """
    def __init__(self, config, *, logger=None, filename=None):
        super().__init__(config, logger, filename=filename)


        if config.version == 400:
            n_internal = 3
            self.mlp = nn.Sequential(
                # Input: [3, 22, 22]

                ResBlock(3, n_internal, 1), # [16, 22, 22]
                ResBlock(n_internal, n_internal * 2 , 2), # [32, 11, 11]
                ResBlock(n_internal * 2, n_internal * 4, 2), # [64, 6, 6]
                ResBlock(n_internal * 4, n_internal * 2, 2), # [32, 3, 3]
                ResBlock(n_internal * 2, n_internal, 2), # [16, 2, 2]
                nn.Conv2d(n_internal, 2, kernel_size=2, stride=2)

            )
        else:
            raise RuntimeError(f'Unsupported model version {config.version}')


        self.last_act = torch.nn.Tanh()

        self.config = config

        if filename is not None:
            self.load(filename)


    def forward(self, x):
        batch_size = x.size(0)  # Get the batch size
        zeros = torch.zeros((batch_size, 6, 3), device=x.device)  # Create zeros for concatenation
        x = torch.cat((x, zeros), dim=1)  # Concatenate along the second dimension (landmarks)
        x = x.view(batch_size, 3, 22, 22)  # Reshape to [batch_size, 3, 22, 22]

        for layer in self.mlp:
            x = layer(x)
        # x = self.mlp(x)
        x = self.last_act(x)
        return torch.squeeze(x)




if __name__ == '__main__':
    x = torch.randn(478, 3)
    x = torch.unsqueeze(x, 0)
    model = GazeResNet(Config())
    model.eval()
    y = model(x)
    print(y.shape)



