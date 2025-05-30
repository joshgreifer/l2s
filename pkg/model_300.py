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


        self.mlp =  nn.Sequential(
            nn.Linear(self.pca.n_components_, config.hidden_channels),
            nn.ReLU(),
            nn.Linear(config.hidden_channels, config.hidden_channels),
            nn.ReLU(),
            nn.Linear(config.hidden_channels, config.hidden_channels),
            nn.ReLU(),
            nn.Linear(config.hidden_channels, 2),
        )
        self.last_act = torch.nn.Tanh()
        self.config = config

        if filename is not None:
            self.load(filename)

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



