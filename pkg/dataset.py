from typing import Tuple, List

import numpy as np
import torch.utils.data.dataset

from pkg.util import log


class SimpleDataset(torch.utils.data.Dataset):

    def __init__(self, *, capacity=2048,):
        self.capacity = capacity

        self.idx = 0

        # Dataset items are a dict
        self._db: List[Tuple[any, any]] = [(None, None)] * capacity
        self.full = False

    def __getitem__(self, idx: int) -> Tuple[int, Tuple[any, any]]:
        item: Tuple = self._db[idx]

        assert item is not None, f"Item at {idx} is None"
        (landmark, target) = item
        assert isinstance(landmark, torch.FloatTensor), f"Item at {idx} is not a Tensor"
        assert landmark.shape == (478, 3), f"Unexpected shape: {landmark.shape}"
        assert isinstance(target, torch.FloatTensor), f"Item at {idx} target is not a Tensor"
        assert target.shape == (2,), f"Unexpected target shape: {target.shape}"
        return (idx, *item)

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add_item(self, item: any, target: any):

        self._db[self.idx] = (item, target)
        self.idx = self.idx + 1
        if self.idx == self.capacity:
            self.full = True
            self.idx = 0

    def clear(self):
        self.__init__()

    def save(self, filename):
        torch.save({
            "_db": self._db,
            "capacity": self.capacity,
            "idx": self.idx,
            "full": self.full,
        }, filename)

    def load(self, filename, expand_to_fit=True):
        """
        Load the dataset from a file.
        :param filename: The file to load the dataset from.
        :param expand_to_fit: If True, expand the dataset to fit the loaded data, otherwise fill up to the current capacity.
        """
        try:
            db = torch.load(filename)
            loaded_capacity = db["capacity"]
            loaded_db = db["_db"]

            if self.capacity <= loaded_capacity:
                # Set the capacity to that of the loaded dataset
                if expand_to_fit:
                    self._db = loaded_db
                    self.capacity = loaded_capacity
                    self.full = db["full"]
                    self.idx = db["idx"]
                else:
                    self._db = loaded_db[:self.capacity]
                    self.full = True if self.capacity < loaded_capacity else db["full"]
                    self.idx = db["idx"] % self.capacity if self.capacity < loaded_capacity else db["idx"]

            else:
                self._db[:loaded_capacity] = loaded_db
                self.full = False
                self.idx = loaded_capacity
            log().info(f'Loaded database: Capacity {self.capacity}, full: {self.full}, idx : {self.idx}, len() = {len(self)}')
        except RuntimeError as er:
            log().warning(er)
        except FileNotFoundError:
            log().warning(f'{filename} not found')

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, window_size):
        self.base = base_dataset  # SimpleDataset of (landmarks, target)
        self.window_size = window_size

    def __len__(self):
        return len(self.base) - self.window_size + 1

    def __getitem__(self, idx):
        # Gather T consecutive landmarks
        X_seq = [self.base._db[idx + j][0] for j in range(self.window_size)]
        X_seq = torch.stack([torch.tensor(x, dtype=torch.float32) for x in X_seq])  # [T, 32] (if PCA used)
        # Target is the gaze for the last frame in window
        y = self.base._db[idx + self.window_size - 1][1]
        return X_seq, torch.tensor(y, dtype=torch.float32)
