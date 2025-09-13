import torch
import torch.nn as nn
import joblib

from pkg.config import Config
from pkg.gaze_model import GazeModel
from pkg.pca import LandMarkPCA


class GazePCAMLP(GazeModel):
    def __init__(self, config):
        super().__init__(config)


        self.pca = LandMarkPCA(config).pca


        # Main MLP
        mlp_layers = [nn.Linear(self.pca.n_components_, config.model.hidden_channels), nn.ReLU()]
        for _ in range(config.model.num_mlp_layers):
            mlp_layers.append(nn.Linear(config.model.hidden_channels, config.model.hidden_channels))
            mlp_layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp_layers)

        # Calibration layers (initialized to identity)
        calibration_layers = []
        for _ in range(config.model.num_calibration_layers):
            fc = nn.Linear(config.model.hidden_channels, config.model.hidden_channels)
            # with torch.no_grad():
            #     fc.weight.copy_(torch.eye(config.model.hidden_channels))
            #     fc.bias.zero_()
            calibration_layers.append(fc)
            calibration_layers.append(nn.ReLU())
        self.calibration_layers = nn.Sequential(*calibration_layers)

        # Output projection
        self.last_fc = nn.Linear(config.model.hidden_channels, 2)

        # Gating layer for learned skip connection (per-output blending)
        self.gate_layer = nn.Linear(config.model.hidden_channels, 2)  # Output shape: [B, 2]

        self.last_act = nn.Tanh()
        self.config = config


        self.load(config.checkpoint)

    def set_calibration_mode(self, mode: bool):
        """
        Set the calibration mode for the model. No-op now: all parameters train in all phases.
        """
        pass

    def forward_features(self, feats: torch.Tensor):
        """Run the MLP stack given PCA features."""
        h = self.mlp(feats)
        h_calib = self.calibration_layers(h)

        y_main = self.last_fc(h)
        y_calib = self.last_fc(h_calib)

        gate = torch.sigmoid(self.gate_layer(h))
        y = gate * y_calib + (1 - gate) * y_main
        y = self.last_act(y)

        if torch.onnx.is_in_onnx_export():
            return y
        return torch.squeeze(y)

    # pkg/model_pca_mlp.py
    def forward(self, x, streaming=False):
        batch_size, num_nodes, in_channels = x.shape

        # Avoid Python booleans during ONNX export
        if not torch.onnx.is_in_onnx_export():
            assert num_nodes == 478, f"Expected 478 nodes, got {num_nodes}"
            assert in_channels == 3, f"Expected 3 channels, got {in_channels}"

        x = x.view(batch_size, -1)  # [B, 478*3]

        feats_np = self.pca.transform(x.cpu().numpy())
        feats = torch.tensor(feats_np, dtype=torch.float32, device=x.device)

        return self.forward_features(feats)


if __name__ == '__main__':
    landmarks = torch.randn(478, 3)
    landmarks = torch.unsqueeze(landmarks, 0)
    model = GazePCAMLP(Config())
    model.eval()
    pred = model(landmarks)
    print(pred.shape)



