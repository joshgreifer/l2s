import joblib
import torch
import torch.nn as nn

from pkg.gaze_model import GazeModel
from pkg.pca import LandMarkPCA
from pkg.util import log


class GazePCALSTM(GazeModel):
    def __init__(self, config):
        super().__init__(config)
        self.hidden_dim = config.model.hidden_dim
        self.num_layers = config.model.num_layers
        self.lstm = nn.LSTM(config.model.pca_num, self.hidden_dim, self.num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh()
        )
        # Register hidden and cell state as persistent buffers
        self.register_buffer("h_buffer", None, persistent=False)
        self.register_buffer("c_buffer", None, persistent=False)
        self.h_buffer = torch.zeros(self.num_layers, 1, self.hidden_dim)
        self.c_buffer = torch.zeros(self.num_layers, 1, self.hidden_dim)
        self.pca = LandMarkPCA(config).pca
        self.load(config.checkpoint)


    def forward(self, x, streaming=False):
        # x: [batch_size, seq_len, 478, 3] (or [batch_size, 1, 478, 3])
        batch_size, seq_len, num_landmarks, num_channels = x.shape
        # Flatten landmarks for PCA: [batch_size, seq_len, 478*3]
        x_flat = x.view(batch_size, seq_len, num_landmarks * num_channels)
        # Apply PCA on each frame (works for both batch_size>1 and seq_len>1)
        # Flatten batch+seq for PCA (sklearn expects 2D: [batch_size*seq_len, 478*3])
        x_pca_in = x_flat.reshape(-1, num_landmarks * num_channels).cpu().numpy()
        x_pca = self.pca.transform(x_pca_in)  # [batch_size*seq_len, 32]
        x_pca = torch.tensor(x_pca, dtype=torch.float32, device=x.device)
        # Reshape back to [batch_size, seq_len, 32]
        x_pca = x_pca.view(batch_size, seq_len, -1)

        if streaming:
            # Streaming mode: maintain state across calls
            # If state not initialized, do so
            assert seq_len == 1, "Streaming mode requires seq_len=1"
            assert batch_size == 1, "Streaming mode requires batch_size=1"

            output, (h_n, c_n) = self.lstm(x_pca, (self.h_buffer, self.c_buffer))
            self.h_buffer = h_n.detach()
            self.c_buffer = c_n.detach()
        else:
            # Batch mode: new state per batch
            batch_size = x.size(0)
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
            output, (h_n, c_n) = self.lstm(x_pca, (h0, c0))
        # Output for last step in sequence
        y = self.head(output[:, -1, :])
        return y

# Example usage:
# model = StreamingGazeLSTM()
# y_batch = model(x_batch, streaming=False)       # x_batch: [B, SEQ, 32]
# y_stream = model(x_frame, streaming=True)       # x_frame: [B, 1, 32]
# model.reset_streaming_state(batch_size=1)       # To reset streaming state, e.g., on new sequence
