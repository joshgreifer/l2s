import logging
import sys

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__()
        for k, v in dict(*args, **kwargs).items():
            if isinstance(v, dict):
                v = AttrDict(v)
            self[k] = v
        self.__dict__ = self


def log():
    logger = logging.getLogger("default")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        # No formatter, just raw output
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
