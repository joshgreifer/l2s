import torch.nn

from pkg.config import Config


class GazeModel(torch.nn.Module):
    def __init__(self, config: Config, logger=None, filename=None):
        super(GazeModel, self).__init__()
        self.config = config
        self.logger = logger
        self.filename = filename

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        try:
            self.load_state_dict(torch.load(filename, map_location=self.config.device, weights_only=True))
        except FileNotFoundError:
            self.logger.warning(f'{filename} not found, using random weights.')
        except RuntimeError as err:
            self.logger.warning(f'{filename} model is incompatible with this version, using random weights. {err}')