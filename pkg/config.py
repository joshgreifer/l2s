import json

import torch

from pkg.util import AttrDict


class Config(AttrDict):

    def __init__(self, filename):
        # load config from cache/config.json
        super().__init__(json.load(open(filename)))

        # Checkpoint to load model from
        self.checkpoint = f'cache/checkpoints/model_{self.model_type}.pt'
        self.dataset_path = f'cache/landmarks_db.pth'

        if hasattr(self.model, 'pca_num'):
            self.pca_path = f'cache/checkpoints/pca_{self.model.pca_num}.joblib'