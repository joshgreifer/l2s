from typing import Tuple, List, Any

import numpy as np
import torch.utils.data.dataset

from pkg.util import log


class SimpleDataset(torch.utils.data.Dataset):

    def __init__(self, *, capacity=2048,):
        self.capacity = capacity

        self.idx = 0

        # Dataset
        self._db: list[tuple[any, any]] = [(None, None)] * capacity  # type: ignore
        self.full = False

    def __getitem__(self, idx: int) -> Tuple[any, any]:

        (landmark, target) = self._db[idx]  # type: ignore
        # print type of landmark, target

        return landmark, target

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add_item(self, item: any, target: any):

        self._db[self.idx] = (item, target)  # type: ignore
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
            log().info(f'Loading database from {filename}...')

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
            log().info(f'...Loaded.  Capacity {self.capacity}, full: {self.full},  len() = {len(self)}')
        except RuntimeError as er:
            log().warning(er)
        except FileNotFoundError:
            log().warning(f'{filename} not found')

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, window_size):
        assert isinstance(base_dataset, SimpleDataset), "base_dataset must be an instance of SimpleDataset"
        self.base: SimpleDataset = base_dataset  # SimpleDataset
        self.window_size = window_size

    def __len__(self):
        return len(self.base) - self.window_size + 1

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get window_size consecutive samples from the base dataset
        landmark_seq = []
        for j in range(self.window_size):
            landmark, _ = self.base[idx + j]  # base.__getitem__ returns (landmark, target)
            landmark_seq.append(landmark)
        landmark_seq = torch.stack(landmark_seq)  # shape: [window_size, 478, 3]
        # Target is gaze for the last frame in window
        _, target = self.base[idx + self.window_size - 1]

        assert isinstance(landmark_seq, torch.Tensor), "landmark_seq must be a torch.Tensor"
        assert isinstance(target, torch.Tensor), "target must be a torch.Tensor"
        assert landmark_seq.shape == (self.window_size, 478, 3), f"landmark_seq shape mismatch: {landmark_seq.shape}"
        assert target.shape == (2,), f"target shape mismatch: {target.shape}"
        return landmark_seq, target