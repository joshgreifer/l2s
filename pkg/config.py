import json

class Config:

    def __init__(self):
        # load config from cache/config.json
        config = None
        try:
            with open('cache/config.json', 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            print("No cache/config.json found, using default config.")

        except json.JSONDecodeError as e:
            print(f"Error parsing cache/config.json: {e} U sing default config.")
        except Exception as e:
            print(f"Error loading cache/config.json: {e} using default config.")

        if config is not None:
            # Update the config with the loaded values
            self.__dict__.update(config)
        else:
            # Device to run models on
            self.device = 'cpu'

            self.db_version = 200
            self.version = 400

            # Model hyperparameters

            # Number of resblocks.  Only used if version >= 102, earlier versions hard-code it to 4
            self.num_resblocks = 5

            # Max capacity of dataset.
            # The dataset is a ring buffer, so data can be continuously added (loses the oldest data).
            # The capacity should be  kept small so that training can track changing poses/lighting etc.
            self.dataset_capacity = 8192

            # Minimum size of dataset before training can begin
            self.dataset_min_size = 512

            # Save the dataset to disk each time this many data items are added
            self.dataset_checkpoint_frequency = 1024

            # Save the model each time this many epochs are completed
            self.model_checkpoint_frequency = 5

            # Large batch size for version 400 - input size is small - [B, 8, 3]
            self.batch_size = 64

            # Scheduler initial learning rate
            self.lr = 5e-3
            self.step_size = 50
            self.gamma = .9

            # Optimizer (SGD)
            self.momentum = 0.5
            self.nesterov = True

            # Optimizer (Adam)
            self.betas = [0.9, 0.999]

        # Checkpoint to load l2s model from
        self.checkpoint = f'cache/checkpoints/l2s_v{self.version}.pth'
        self.dataset_path = f'cache/checkpoints/l2s_v{self.db_version}_db.pth'