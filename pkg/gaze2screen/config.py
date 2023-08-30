import torch


class Config:
    class _G2S:
        def __init__(self):
            self.version = 101

            # Checkpoint to load l2s model from
            self.checkpoint = f'cache/gaze2screen/checkpoints/l2s_v{self.version}.pth'
            self.dataset_path = f'cache/gaze2screen/checkpoints/l2s_v{self.version}_db.pth'

            # Max capacity of dataset.
            # The dataset is a ring buffer, so data can be continuously added (loses the oldest data).
            # The capacity should be  kept small so that training can track changing poses/lighting etc.
            self.dataset_capacity = 20000

            # Minimum size of dataset before training can begin
            self.dataset_min_size = 1000

            # After adding this many frames to dataset, do a training run
            self.training_frequency_frames = 10

            # Lower this number if too slow
            self.number_of_epochs_per_run = 1

            # Raise this number if too slow (but degrades training)
            self.batch_size = 10

            # Optimizer learning rate
            self.lr = 1e-3

            # Optimizer (SGD)
            self.momentum = 0.5
            self.nesterov = True

            # Optimizer (Adam)
            self.betas = [0.9, 0.999]

            #

    def __init__(self):
        # Device to run models on

        self.device = 'cpu' if torch.cuda.device_count() == 0 else 'cuda'

        # Gaze2Screen config
        self.g2s = self._G2S()