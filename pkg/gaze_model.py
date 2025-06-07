import torch.nn

from pkg.config import Config
from pkg.util import log


class GazeModel(torch.nn.Module):
    def __init__(self, config: Config, logger=None, filename=None):
        super(GazeModel, self).__init__()
        self.config = config

        self.filename = filename

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        try:
            self.load_state_dict(torch.load(filename, map_location=self.config.device, weights_only=True))
        except FileNotFoundError:
            log().warning(f'{filename} not found, using random weights.')
        except RuntimeError as err:
            log().warning(f'{filename} model is incompatible with this version, using random weights. {err}')

    def set_calibration_mode(self, mode: bool):
        """
        Set the calibration mode for the model.
        :param mode: True for calibration mode, False for full-training mode.
        """
        pass

