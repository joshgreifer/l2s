import joblib
import torch
from torch import nn

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


        mlp_layers = [nn.Linear(self.pca.n_components_, config.hidden_channels), nn.ReLU()]
        for _ in  range(config.num_mlp_layers):
            mlp_layers.append(nn.Linear(config.hidden_channels, config.hidden_channels))
            mlp_layers.append(nn.ReLU())


        self.mlp =  nn.Sequential(*mlp_layers)

        calibration_layers = []
        for _ in range(config.num_calibration_layers):
            # initialize calibration layers to identity
            fc = nn.Linear(config.hidden_channels, config.hidden_channels)
            with torch.no_grad():
                fc.weight.copy_(torch.eye(config.hidden_channels))
                fc.bias.zero_()
            calibration_layers.append(fc)
            calibration_layers.append(nn.ReLU())
        self.calibration_layers = nn.Sequential(*calibration_layers)

        self.last_fc = nn.Linear(config.hidden_channels, 2)
        self.last_act = nn.Tanh()
        self.config = config

        if filename is not None:
            self.load(filename)

    def set_calibration_mode(self, mode: bool):
        """
        Set the calibration mode for the model.
        :param mode: True for calibration mode, False for full training mode.
        """
        # In calibration mode, we freeze all layers except the last one
        # This allows the model to adapt only the final layer during calibration
        if mode:
            for param in self.mlp.parameters():
                param.requires_grad = False
        else:
            for param in self.calibration_layers.parameters():
                param.requires_grad = False


    def forward(self, x):
        batch_size, num_nodes, in_channels = x.shape
        # Flatten the input to match PCA input shape
        assert num_nodes == 478, f"Expected 478 nodes, got {num_nodes}"
        assert in_channels == 3, f"Expected 3 channels, got {in_channels}"
        x = x.view(batch_size, -1)  # Flatten to shape [B, 478 * 3]

        # Apply PCA transformation
        x_pca = self.pca.transform(x.cpu().numpy())  # PCA works on numpy arrays

        # Convert the transformed data back to a PyTorch tensor
        x_pca = torch.tensor(x_pca, dtype=torch.float32, device=x.device)

        # Pass through the MLP
        x = self.mlp(x_pca)
        # Pass through the calibration stage
        x = self.calibration_layers(x)

        # reduce to x, y
        x = self.last_fc(x)
        # Apply the final activation function
        x = self.last_act(x)

        return torch.squeeze(x)



if __name__ == '__main__':
    landmarks = torch.randn(478, 3)
    landmarks = torch.unsqueeze(landmarks, 0)
    model = GazePCA(Config())
    model.eval()
    pred = model(landmarks)
    print(pred.shape)



