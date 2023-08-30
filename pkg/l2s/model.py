from typing import Dict


import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               padding_mode='reflect',
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               padding_mode='reflect',
                               bias=False)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=stride,
                          padding=0,
                          bias=False))

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = F.relu(self.bn1(x), inplace=True)
        y = self.conv1(x)
        y = F.relu(self.bn2(y), inplace=True)
        y = self.conv2(y)
        y += self.shortcut(x)
        return y


class LandMarks2ScreenModel(torch.nn.Module):
    """
    This model predicts the screen location (normalized to range [-1..1]
    of a gaze.
    It takes a 2d 3-channel input "image",
    where the channels are x,y,z coordinates of face landmarks, which have
    been obtained by Mediapipe face-landmarker.
    The Mediapipe landmarker identifies 478 landmarks, a subset of which
    are then arranged into a 3-channel 32 X 4  Tensor (see pack_input()).
    """
    def __init__(self, *, logger=None, filename=None):
        super().__init__()
        self.logger = logger

        # Resnet
        self.resnet = nn.Sequential()
        ch = 3

        n_blocks = 4
        for b in range(n_blocks):
            self.resnet.add_module(f'resblock_u{b}', ResBlock(ch, 2 * ch, 1))
            ch *= 2
        for b in range(n_blocks):
            self.resnet.add_module(f'resblock_d{b}', ResBlock(ch, ch // 2, 1))
            ch //= 2
        assert ch == 3

        # Reduce to single channel, single row
        self.dim_reduce = nn.Sequential()
        self.dim_reduce.add_module(f'reduce_1', nn.Conv2d(3, 3, (3, 3), (1, 1), (1, 0), padding_mode='reflect'))
        self.dim_reduce.add_module(f'reduce_2', nn.Conv2d(3, 2, (3, 2), (1, 1), (1, 0), padding_mode='reflect'))
        self.dim_reduce.add_module(f'reduce_3', nn.Conv2d(2, 1, (3, 1), (1, 1), (1, 0), padding_mode='reflect'))
        self.fc = nn.Linear(32, 2)

        self.last_act = torch.nn.Tanh()

        if filename is not None:
            self.load(filename)

    @staticmethod
    def pack_input(feature_dict: Dict[str, any]) -> torch.Tensor:
        """

        :param feature_dict:
        :return: 3 channel 2D tensor of face landmark features, where the channels are
        X,Y,Z coordinates of the landmarks
        """

        face_oval = torch.Tensor(feature_dict["face_oval"]).to(torch.float32)
        left_eye = torch.Tensor(feature_dict["left_eye"]).to(torch.float32)
        right_eye = torch.Tensor(feature_dict["right_eye"]).to(torch.float32)
        left_iris = torch.Tensor(feature_dict["left_iris"]).to(torch.float32)
        right_iris = torch.Tensor(feature_dict["right_iris"]).to(torch.float32)
        eye_blendshapes = torch.Tensor(feature_dict["eye_blendshapes"]).to(torch.float32)

        # print("face_oval:", face_oval.shape)
        # print("left_eye:", left_eye.shape)
        # print("right_eye:", right_eye.shape)
        # print("left_iris:", left_iris.shape)
        # print("right_iris:", right_iris.shape)
        # print("eye_blendshapes:", eye_blendshapes.shape)

        face_row = face_oval[2:-2, :]
        eyes_row = torch.cat((left_eye, right_eye), dim=-2)
        iris_row = torch.cat((left_iris.repeat((4, 1)), right_iris.repeat((4, 1))), dim=-2)

        # Blendshapes for eyes are  L R L R L R...
        # Change to L L L L ... R R R R
        eye_blendshapes = eye_blendshapes.reshape(-1, 2).transpose(0, 1).flatten()
        # 14 shapes, change to 16
        eye_blendshapes = F.pad(eye_blendshapes, (1, 1))
        # Add height dim
        eye_blendshapes = torch.unsqueeze(eye_blendshapes, -1)
        # Tile to same size as other rows
        eye_blendshapes_row = eye_blendshapes.repeat(2, 3)

        # print("face_row:", face_row.shape)
        # print("eyes_row:", eyes_row.shape)
        # print("iris_row:", iris_row.shape)
        # print("eye_blendshapes_row:", eye_blendshapes_row.shape)

        rows = torch.stack((face_row, eyes_row, iris_row, eye_blendshapes_row), dim=-1)
        # print("Stacked shape:", rows.shape)

        # Put channel dim first
        rows = rows.transpose(0, 1)
        # print("After transpose:", rows.shape)

        # Do NOT add batch dim
        # rows = torch.unsqueeze(rows, 0)

        # print("Final shape:", rows.shape)
        return rows

    def forward(self, x):
        # print(x.shape)
        x = self.resnet(x)
        # print(x.shape)
        x = self.dim_reduce(x)
        # print(x.shape)

        x = self.fc(torch.transpose(x, -1, -2))
        # print(x.shape)
        x = self.last_act(x)

        return torch.squeeze(x)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        try:
            self.load_state_dict(torch.load(filename))
        except FileNotFoundError:
            self.logger.warning(f'{filename} not found, using random weights.')
        except RuntimeError as err:
            self.logger.warning(f'{filename} model is incompatible with this version, using random weights.')


if __name__ == '__main__':
    pass


