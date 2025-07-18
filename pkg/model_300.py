import torch
import torch.nn as nn
import joblib

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

        # Main MLP
        mlp_layers = [nn.Linear(self.pca.n_components_, config.hidden_channels), nn.ReLU()]
        for _ in range(config.num_mlp_layers):
            mlp_layers.append(nn.Linear(config.hidden_channels, config.hidden_channels))
            mlp_layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp_layers)

        # Calibration layers (initialized to identity)
        calibration_layers = []
        for _ in range(config.num_calibration_layers):
            fc = nn.Linear(config.hidden_channels, config.hidden_channels)
            # with torch.no_grad():
            #     fc.weight.copy_(torch.eye(config.hidden_channels))
            #     fc.bias.zero_()
            calibration_layers.append(fc)
            calibration_layers.append(nn.ReLU())
        self.calibration_layers = nn.Sequential(*calibration_layers)

        # Output projection
        self.last_fc = nn.Linear(config.hidden_channels, 2)

        # Gating layer for learned skip connection (per-output blending)
        self.gate_layer = nn.Linear(config.hidden_channels, 2)  # Output shape: [B, 2]

        self.last_act = nn.Tanh()
        self.config = config

        if filename is not None:
            self.load(filename)

    def set_calibration_mode(self, mode: bool):
        """
        Set the calibration mode for the model. No-op now: all parameters train in all phases.
        """
        pass

    def forward(self, x):
        batch_size, num_nodes, in_channels = x.shape
        assert num_nodes == 478, f"Expected 478 nodes, got {num_nodes}"
        assert in_channels == 3, f"Expected 3 channels, got {in_channels}"
        x = x.view(batch_size, -1)  # [B, 478*3]
        x_pca = self.pca.transform(x.cpu().numpy())  # [B, n_components]
        x_pca = torch.tensor(x_pca, dtype=torch.float32, device=x.device)

        # Main MLP feature extraction
        h = self.mlp(x_pca)  # [B, hidden_channels]
        # Calibration path
        h_calib = self.calibration_layers(h)  # [B, hidden_channels]

        # Project both to output
        y_main = self.last_fc(h)         # [B, 2]
        y_calib = self.last_fc(h_calib)  # [B, 2]

        # Compute learnable gate for blending
        gate = torch.sigmoid(self.gate_layer(h))  # [B, 2], each in (0, 1)

        # Blend outputs
        y = gate * y_calib + (1 - gate) * y_main
        y = self.last_act(y)  # [-1, 1] screen coords

        return torch.squeeze(y)


if __name__ == '__main__':
    landmarks = torch.randn(478, 3)
    landmarks = torch.unsqueeze(landmarks, 0)
    model = GazePCA(Config())
    model.eval()
    pred = model(landmarks)
    print(pred.shape)



