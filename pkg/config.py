import torch


class Config:

    def __init__(self):
        # Device to run models on

        self.device = 'cpu' if torch.cuda.device_count() == 0 else 'cuda'

        self.db_version = 200
        self.version = 300

        # Model hyperparameters

        # Number of resblocks.  Only used if version >= 102, earlier versions hard-code it to 4
        self.num_resblocks = 5

        # Checkpoint to load l2s model from
        self.checkpoint = f'cache/checkpoints/l2s_v{self.version}.pth'
        self.dataset_path = f'cache/checkpoints/l2s_v{self.db_version}_db.pth'

        # Max capacity of dataset.
        # The dataset is a ring buffer, so data can be continuously added (loses the oldest data).
        # The capacity should be  kept small so that training can track changing poses/lighting etc.
        self.dataset_capacity = 65536

        # Minimum size of dataset before training can begin
        self.dataset_min_size = 512

        # Save the dataset to disk each time this many data items are added
        self.dataset_checkpoint_frequency = 2048

        # Save the model each time this many epochs are completed
        self.model_checkpoint_frequency = 5

        self.batch_size = 32

        # Scheduler initial learning rate
        self.lr = 1e-3
        self.step_size = 50
        self.gamma = .5

        # Optimizer (SGD)
        self.momentum = 0.5
        self.nesterov = True

        # Optimizer (Adam)
        self.betas = [0.9, 0.999]
