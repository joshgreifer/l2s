import logging
import math
import sys

import numpy as np
import torch
import torch.utils.data
from torch import TensorType
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from pkg.config import Config
from pkg.dataset import SequenceDataset, SimpleDataset
from pkg.model_pca_lstm import GazePCALSTM

from pkg.model_pca_mlp import GazePCAMLP
from pkg.util import log

"""
export interface LandmarkFeatures {
    face_oval: Number[][];
    left_eye: number[][];
    right_eye: number[][];
    left_iris: number[][];
    right_iris: number[][];
    nose: number[][];
    eye_blendshapes: number[];
}
"""


class L2S:

    def __init__(self, config):
        try:
            self.config = config
            self.mode = 'eval'
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.target = np.ndarray((2,))  # x, y coords, -1..1, origin center of the screen

            if self.config.model_type == 'GazePCAMLP':
                model = GazePCAMLP
            elif self.config.model_type == 'GazePCALSTM':
                model = GazePCALSTM
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_type}")

            self.model = model(self.config).to(self.device)

            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config.train.lr, betas=self.config.train.betas, weight_decay=self.config.train.weight_decay
            )
            self.scheduler = StepLR(self.optimizer, step_size=self.config.train.step_size, gamma=self.config.train.gamma)

        except (RuntimeError, FileNotFoundError, ValueError) as e:
            print(f"Couldn't initialize model: {e}")
            self.model = None

        self.dataset = SimpleDataset(capacity=self.config.train.dataset_capacity)
        self.dataset.load(self.config.dataset_path, expand_to_fit=False)

        self.fine_tuning_dataset = SimpleDataset(capacity=self.config.train.fine_tuning_dataset_capacity)

        self.losses = {"h_loss": 1., "v_loss": 1., "loss": math.sqrt(2)}



    def save(self, epoch):
        if self.model:
            save_filename = f"{self.config.checkpoint}"
            self.model.save(save_filename)
            log().info(f"\nModel saved to {save_filename}.")

    def train(self, epochs, streaming_mode=False):


        self.model.set_calibration_mode(streaming_mode)


        if streaming_mode:
            dataset = self.fine_tuning_dataset
        else:
            dataset = self.dataset
        if len(dataset) == 0:
            log().warn("Dataset is empty, cannot train.")
            return self.losses

        if self.config.model_type == "GazePCALSTM":
            dataset = SequenceDataset(dataset, window_size=1 if streaming_mode else self.config.train.sequence_length)

        log().info(f"Training with dataset size: {len(dataset)}")

        if len(dataset) >= self.config.train.dataset_min_size:
            self.model.train()

            loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1 if streaming_mode else self.config.train.batch_size, shuffle=not streaming_mode)


            if epochs <= 0:
                epochs = 1 + len(self.dataset) // 100

            with tqdm(total=epochs, desc="Training Progress") as pbar:
                for epoch in range(epochs):
                    losses = {"h_loss": 0., "v_loss": 0., "loss": 0.}
                    n_batches = 0

                    x: TensorType
                    y: TensorType
                    for n_batches, (x, y) in enumerate(loader):
                        x = x.to(self.device)
                        y = y.to(self.device)

                        pred = self.model(x, streaming=streaming_mode)


                        dists = torch.norm(pred - y, dim=1)
                        diff = pred - y
                        h_dist = diff[:, 0].abs().mean()
                        v_dist = diff[:, 1].abs().mean()
                        loss = dists.mean()

                        losses["loss"] += loss.cpu().item()
                        losses["h_loss"] += h_dist.cpu().item()
                        losses["v_loss"] += v_dist.cpu().item()
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()


                    # Compute average losses for the epoch
                    losses["loss"] /= n_batches
                    losses["h_loss"] /= n_batches
                    losses["v_loss"] /= n_batches
                    self.losses = losses

                    # Update the progress bar at the end of the epoch
                    pbar.set_postfix(h_loss=self.losses["h_loss"], v_loss=self.losses["v_loss"], loss=self.losses["loss"])
                    pbar.update(1)

                    self.scheduler.step()
                    # logging.getLogger('app').info(f'Epoch {epoch + 1}: lr {self.scheduler.get_last_lr()}  h_loss: {self.losses["h_loss"]: .4f} v_loss: {self.losses["v_loss"]: .4f}')

                    if (epoch % self.config.train.model_checkpoint_frequency) == 0:
                        self.save(epoch)
        self.model.eval()

        return self.losses

    def predict(self, landmarks, label):

        # print("-------", landmarks, "-------")
        # print("-------", label, "-------")

        # Return some of the landmarks back to the client.
        # This is not really necessary, as the client itself found the landmarks in the first place.
        # This is a legacy from the old app, where the client posted video frames
        # and the server found the landmarks.

        # face_oval_landmarks = [[landmark[0], landmark[1]] for landmark in landmarks["face_oval"]]
        # nose_landmarks = [[landmark[0], landmark[1]] for landmark in landmarks["nose"]]
        # landmarks_for_display = face_oval_landmarks + nose_landmarks

        # Convert the landmarks into a tensor.  This tensor is what's saved in the dataset
        # as well as being passed to the model.

        assert isinstance(landmarks, list)
        landmarks = torch.Tensor(landmarks).to(torch.float32)
        # If we're gathering training data, add the landmarks (x) and label (y)
        # In the training dataset.
        if label is not None:
            label_as_tensor = torch.Tensor(label).to(torch.float32)
            self.dataset.add_item(landmarks, label_as_tensor)
            self.fine_tuning_dataset.add_item(landmarks, label_as_tensor)
            # Save dataset periodically
            if (self.dataset.idx % self.config.train.dataset_checkpoint_frequency) == 0:
                self.dataset.save(self.config.dataset_path)
                log().info(f"Saved dataset to {self.config.dataset_path}. Dataset size {len(self.dataset)}")

        # Predict the gaze coordinates

        if self.model is not None:
            with torch.no_grad():
                self.model.eval()
                if self.config.model_type == "GazePCALSTM":
                    #  unsqueeze once for sequence dimension, and again for batch dimension
                    pred = self.model(landmarks.view(1, 1, *landmarks.shape).to(self.device), streaming=True)
                else:
                    pred = self.model(torch.unsqueeze(landmarks, 0).to(self.device))

                pred = torch.squeeze(pred)
                gaze_location = pred.cpu().detach().numpy()
        else:
            gaze_location = [0,0]
        # print(label, features, gaze_location)
        return {
            'data_index': self.dataset.idx,
            'faces': 1,
            'eyes': 2,
            'gaze': {
                'x': float(gaze_location[0]),
                'y': float(gaze_location[1])
            },
            'landmarks': [],  # landmarks_for_display,
            'losses': self.losses
        }


if __name__ == '__main__':
    def main():
         #get the number of epochs and calibration mode from the arguments without argparse
        import sys
        config_file = sys.argv[1] if len(sys.argv) > 1 else "cache/config.json"
        if len(sys.argv) > 2:
            epochs = int(sys.argv[2])
        else:
            epochs = 1000
        if len(sys.argv) > 3:
            calibration_mode = sys.argv[3].lower() == 'true'
        else:
            calibration_mode = False

        config = Config(config_file)
        l2s = L2S(config)

        log().info(f"Running model on device: {next(l2s.model.parameters()).device}")

        log().info(f"Training for {epochs} epochs, calibration mode: {calibration_mode}")

        l2s.train(epochs, calibration_mode)
    try:

        main()
    except Exception as e:
        log().exception(f"Error in main: {e}")
