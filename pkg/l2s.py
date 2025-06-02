import logging
import math
from math import gamma

import numpy as np
import torch
import torch.utils.data
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from pkg.config import Config
from pkg.model_300 import GazePCA
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
        try:
            self.config = Config()
            self.logger = logger
            self.mode = 'eval'
            self.device = self.config.device
            self.target = np.ndarray((2,))  # x, y coords, -1..1, origin center of the screen
            if self.config.version == 300:
                model = GazePCA
            elif self.config.version == 400:
                model = GazeResNet
            elif self.config.version == 500:
                model = GazeGCN
            elif self.config.version == 600:
                model = GazeGAT
            else:
                raise ValueError(f"Unsupported model version: {self.config.version}")

            self.model = model(self.config, logger=logger, filename=self.config.checkpoint).to(self.device)

            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config.lr, betas=self.config.betas, weight_decay=self.config.weight_decay
            )
            self.scheduler = StepLR(self.optimizer, step_size=self.config.step_size, gamma=self.config.gamma)

        except (RuntimeError, FileNotFoundError, ValueError) as e:
            print(f"Couldn't initialize model: {e}")
            self.model = None

        self.dataset = SimpleDataset(capacity=self.config.dataset_capacity, logger=logger)
        self.dataset.load(self.config.dataset_path, expand_to_fit=False)

        self.fine_tuning_dataset = SimpleDataset(capacity=self.config.fine_tuning_dataset_capacity, logger=logger)

        self.losses = {"h_loss": 1., "v_loss": 1., "loss": math.sqrt(2)}



    def save(self):
        if self.model:
            self.model.save(self.config.checkpoint)

    def train(self, epochs, calibration_mode=False):
        if self.model:

            self.model.set_calibration_mode(calibration_mode)

            if calibration_mode:
                dataset =  self.fine_tuning_dataset
                print(f"Fine-tuning with dataset size: {len(dataset)}")
            else:
                dataset = self.dataset
                print(f"Training with dataset size: {len(dataset)}")

            if len(dataset) >= self.config.dataset_min_size:
                self.model.train()

                loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.config.batch_size, shuffle=True)


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
            if (self.dataset.idx % self.config.dataset_checkpoint_frequency) == 0:
                self.dataset.save(self.config.dataset_path)
                logging.getLogger('app').info(
                    f"Saved dataset to {self.config.dataset_path}. Dataset size {len(self.dataset)}")

        # Predict the gaze coordinates

        if self.model is not None:
            with torch.no_grad():
                self.model.eval()
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

    def do_pca(self):
        """
        Do PCA on the dataset and save the model.
        :return: PCA model
        """
        from pkg.pca import do_pca as do_pca_
        try:
            pca = do_pca_()
            logging.getLogger('app').info(f"PCA model saved with {pca.n_components_} components.")
        except Exception as e:
            logging.getLogger('app').error(f"Error during PCA: {e}")
            return {'status': 'failed', 'error': str(e)}

        return {'status': 'success', 'pca_num_components': pca.n_components_}


if __name__ == '__main__':
    def main():
        l2s = Landmarks2ScreenCoords(logging.getLogger())

        print(f"Running model on device: {next(l2s.model.parameters()).device}")
        #get the number of epochs and calibration mode from the arguments without argparse
        import sys
        if len(sys.argv) > 1:
            epochs = int(sys.argv[1])
        else:
            epochs = 1000
        if len(sys.argv) > 2:
            calibration_mode = sys.argv[2].lower() == 'true'
        else:
            calibration_mode = False
        print(f"Training for {epochs} epochs, calibration mode: {calibration_mode}")

        l2s.train(epochs, calibration_mode)
    try:
        main()
    except Exception as e:
        logging.getLogger('app').exception(e)
