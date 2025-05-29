import logging



import numpy as np
import torch
import torch.utils.data
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from pkg.config import Config
from pkg.model_300 import GazeResNetSimple
from pkg.model_600 import GazeGAT
from pkg.simple_dataset import SimpleDataset
from pkg.model_400 import GazeResNet
from pkg.model_500 import  GazeGCN
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


class Landmarks2ScreenCoords:

    def __init__(self, logger):

        self.config = Config()
        self.logger = logger
        self.mode = 'eval'
        self.device = self.config.device
        self.target = np.ndarray((2,))  # x, y coords, -1..1, origin center of the screen
        if self.config.version == 300:
            model = GazeResNetSimple
        elif self.config.version == 400:
            model = GazeResNet
        elif self.config.version == 500:
            model = GazeGCN
        elif self.config.version == 600:
            model = GazeGAT
        else:
            raise ValueError(f"Unsupported model version: {self.config.version}")

        self.model = model(self.config, logger=logger, filename=self.config.checkpoint).to(self.device)
        # self.optimizer = torch.optim.SGD(self.model.parameters(),
        #                                  lr=self.config.lr,
        #                                  momentum=self.config.momentum,
        #                                  nesterov=self.config.nesterov)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.config.lr, betas=self.config.betas)

        self.scheduler = StepLR(self.optimizer, step_size=self.config.step_size, gamma=self.config.gamma)

        self.dataset = SimpleDataset(capacity=self.config.dataset_capacity, logger=logger)
        self.dataset.load(self.config.dataset_path)
        self.losses = {"h_loss": 0., "v_loss": 0., "loss": 0.}

        # self.model.eval()


    def save(self):
        self.model.save(self.config.checkpoint)

    def train(self, epochs):
        if len(self.dataset) >= self.config.dataset_min_size:
            self.model.train()

            loader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=self.config.batch_size, shuffle=True)
            logging.getLogger('app').info(f"Training with dataset size: {len(self.dataset)}")

            if epochs <= 0:
                epochs = 1 + len(self.dataset) // 100

            with tqdm(total=epochs, desc="Training Progress") as pbar:
                for epoch in range(epochs):
                    losses = {"h_loss": 0., "v_loss": 0., "loss": 0.}
                    n_batches = 0

                    for n_batches, (idx, x, y) in enumerate(loader):
                        x = x.to(self.device)
                        y = y.to(self.device)

                        pred = self.model(x)

                        try:
                            dists = torch.mean(abs(pred - y), dim=0)
                            h_dist = dists[0]
                            v_dist = dists[1]
                            loss = torch.pow(v_dist, 3) + torch.pow(h_dist, 3)

                            losses["loss"] += loss.cpu().item()
                            losses["h_loss"] += h_dist.cpu().item()
                            losses["v_loss"] += v_dist.cpu().item()
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()
                        except IndexError:
                            continue

                    # Compute average losses for the epoch
                    losses["loss"] /= n_batches + 1
                    losses["h_loss"] /= n_batches + 1
                    losses["v_loss"] /= n_batches + 1
                    self.losses = losses

                    # Update the progress bar at the end of the epoch
                    pbar.set_postfix(h_loss=self.losses["h_loss"], v_loss=self.losses["v_loss"], loss=self.losses["loss"])
                    pbar.update(1)

                    self.scheduler.step()
                    # logging.getLogger('app').info(f'Epoch {epoch + 1}: lr {self.scheduler.get_last_lr()}  h_loss: {self.losses["h_loss"]: .4f} v_loss: {self.losses["v_loss"]: .4f}')

                    if (epoch % self.config.model_checkpoint_frequency) == 0:
                        self.save()

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

            self.dataset.add_item(landmarks, torch.Tensor(label).to(torch.float32))

            # Save dataset periodically
            if (self.dataset.idx % self.config.dataset_checkpoint_frequency) == 0:
                self.dataset.save(self.config.dataset_path)
                logging.getLogger('app').info(
                    f"Saved dataset to {self.config.dataset_path}. Dataset size {len(self.dataset)}")

        # Predict the gaze coordinates
        with torch.no_grad():
            self.model.eval()
            pred = self.model(torch.unsqueeze(landmarks, 0).to(self.device))

            pred = torch.squeeze(pred)
            gaze_location = pred.cpu().detach().numpy()

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
        l2s = Landmarks2ScreenCoords(logging.getLogger())

        print(f"Running model on device: {next(l2s.model.parameters()).device}")
        l2s.train(1000)
    try:
        main()
    except Exception as e:
        logging.getLogger('app').exception(e)
