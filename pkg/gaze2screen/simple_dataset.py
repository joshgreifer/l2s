from typing import Tuple, List

import numpy as np
import torch.utils.data.dataset


class SimpleDataset(torch.utils.data.Dataset):

    def __init__(self, *, capacity=2048, logger=None):
        self.logger = logger
        self.capacity = capacity

        self.idx = 0

        # Dataset items are a dict
        self._db: List[Tuple[any, any]] = [(None, None)] * capacity
        self.full = False

    def __getitem__(self, idx: int) -> Tuple[int, Tuple[any, any]]:
        item: Tuple = self._db[idx]

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

    def load(self, filename):
        try:
            db = torch.load(filename)
            loaded_capacity = db["capacity"]
            loaded_db = db["_db"]
            loaded_size = loaded_capacity if db["full"] else db["idx"]
            if self.capacity <= loaded_capacity:
                self._db = loaded_db
                self.capacity = loaded_capacity
                self.full = db["full"]
                self.idx = db["idx"]

            else:
                self._db[:loaded_capacity] = loaded_db
                self.full = False
                self.idx = loaded_capacity

        except RuntimeError as er:
            self.logger.warning(er)
        except FileNotFoundError:
            self.logger.warning(f'{filename} not found')

