from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn
import torch.nn.functional as F

from pkg.config import Config


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=stride,
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

        config = Config()
        if config.version == 101:
            n_blocks = 4
        elif config.version == 300:
            n_blocks = 3
        else:
            n_blocks = config.num_resblocks

        stride = 1

        for b in range(n_blocks):
            self.resnet.add_module(f'resblock_u{b}', ResBlock(ch, 2 * ch, stride))
            ch *= 2
        for b in range(n_blocks):
            self.resnet.add_module(f'resblock_d{b}', ResBlock(ch, ch // 2, stride))
            ch //= 2
        assert ch == 3

        self.dim_reduce1 = nn.Sequential()
        self.dim_reduce2 = nn.Sequential()
        if config.version == 300:
            stride = 2
            # Reduce (1,24,3,3) to (1,1,1,1)
            self.dim_reduce1.add_module(f'reduce1_resblock_1', ResBlock(ch, 2 * ch, stride))
            self.dim_reduce1.add_module(f'reduce1_resblock_2', ResBlock(2 * ch, 4 * ch, stride))
            self.dim_reduce1.add_module(f'reduce1_resblock_3', ResBlock(4 * ch, 8 * ch, stride))
            self.dim_reduce1.add_module(f'reduce1_conv', nn.Conv2d(24, 1, (3, 3)))

            self.dim_reduce2.add_module(f'reduce2_resblock_1', ResBlock(ch, 2 * ch, stride))
            self.dim_reduce2.add_module(f'reduce2_resblock_2', ResBlock(2 * ch, 4 * ch, stride))
            self.dim_reduce2.add_module(f'reduce2_resblock_3', ResBlock(4 * ch, 8 * ch, stride))
            self.dim_reduce2.add_module(f'reduce2_conv', nn.Conv2d(24, 1, (3, 3)))



        elif config.version == 200:
            # Reduce to single channel, single row of 15
            self.dim_reduce1.add_module(f'reduce_1', nn.Conv2d(3, 1, (3, 5), (1, 1), (1, 0), padding_mode='reflect'))
            self.fc = nn.Linear(15, 2)
        elif config.version < 102:
            # Reduce to single channel, single row of 32
            self.dim_reduce1.add_module(f'reduce_1', nn.Conv2d(3, 3, (3, 3), (1, 1), (1, 0), padding_mode='reflect'))
            self.dim_reduce1.add_module(f'reduce_2', nn.Conv2d(3, 2, (3, 2), (1, 1), (1, 0), padding_mode='reflect'))
            self.dim_reduce1.add_module(f'reduce_3', nn.Conv2d(2, 1, (3, 1), (1, 1), (1, 0), padding_mode='reflect'))
            self.fc = nn.Linear(32, 2)
        else:
            # Reduce to single channel, single row of 32
            self.dim_reduce1.add_module(f'reduce_1', nn.Conv2d(3, 3, (3, 3), (1, 1), (1, 0), padding_mode='reflect'))
            self.dim_reduce1.add_module(f'reduce_2', nn.Conv2d(3, 2, (3, 2), (1, 1), (1, 0), padding_mode='reflect'))
            self.dim_reduce1.add_module(f'reduce_3', nn.Conv2d(2, 1, (3, 1), (1, 1), (1, 0), padding_mode='reflect'))
            self.fc = nn.Linear(32, 2)

        self.last_act = torch.nn.Tanh()

        self.config = config

        if filename is not None:
            self.load(filename)

    def pack_input(self, landmarks: Dict[str, any] | List[any]) -> torch.Tensor:
        """

        :param landmarks:
        :return: 3 channel 2D tensor of face landmark features, where the channels are
        X,Y,Z coordinates of the landmarks
        """
        if self.config.version < 200:
            assert isinstance(landmarks, dict)
            face_oval = torch.Tensor(landmarks["face_oval"]).to(torch.float32)
            left_eye = torch.Tensor(landmarks["left_eye"]).to(torch.float32)
            right_eye = torch.Tensor(landmarks["right_eye"]).to(torch.float32)
            left_iris = torch.Tensor(landmarks["left_iris"]).to(torch.float32)
            right_iris = torch.Tensor(landmarks["right_iris"]).to(torch.float32)
            nose = torch.Tensor(landmarks["nose"]).to(torch.float32)
            eye_blendshapes = torch.Tensor(landmarks["eye_blendshapes"]).to(torch.float32)

            # print("face_oval:", face_oval.shape)
            # print("left_eye:", left_eye.shape)
            # print("right_eye:", right_eye.shape)
            # print("left_iris:", left_iris.shape)
            # print("right_iris:", right_iris.shape)
            # print("nose:", nose.shape)
            # print("eye_blendshapes:", eye_blendshapes.shape)

            face_row = face_oval[2:-2, :]
            eyes_row = torch.cat((left_eye, right_eye), dim=-2)
            iris_row = torch.cat((left_iris.repeat((4, 1)), right_iris.repeat((4, 1))), dim=-2)
            face_row[16-2:16+2, :] = nose
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
            # Put channel dim first
            rows = rows.transpose(0, 1)
            # print("After transpose:", rows.shape)

        elif self.config.version == 200:
            assert isinstance(landmarks, list)
            landmarks = torch.Tensor(landmarks).to(torch.float32)
            row1 = torch.index_select(landmarks, 0, torch.tensor([127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356]))
            row2 = torch.index_select(landmarks, 0, torch.tensor([246, 161, 160, 159, 158, 157, 173, 8, 398, 384, 385, 386, 387, 388, 466]))
            row3 = torch.index_select(landmarks, 0, torch.tensor([33, 470, 471, 468, 469, 472, 133, 168, 362, 477, 476, 473, 474, 475, 263]))
            row4 = torch.index_select(landmarks, 0, torch.tensor( [7, 163, 144, 145, 153, 154, 155, 6, 382, 381, 380, 374, 373, 390, 249]))
            row5 = torch.index_select(landmarks, 0, torch.tensor( [132, 147, 187, 207, 206, 165, 167, 164, 393, 391, 426, 427, 411, 376, 361,]))

            rows = torch.stack((row1, row2, row3, row4, row5), dim=-1)
            # Put channel dim first
            rows = rows.transpose(0, 1)
            # print("After transpose:", rows.shape)
        elif self.config.version == 300:
            assert isinstance(landmarks, list)
            landmarks = torch.Tensor(landmarks).to(torch.float32)
            landmarks = torch.cat((landmarks, torch.zeros((6, 3))), dim=0)
            rows = torch.reshape(landmarks, (22, 22, 3))
            # Put channel dim first
            rows = rows.transpose(0, 2)
            # print("After transpose:", rows.shape)
        # print("Stacked shape:", rows.shape)
        else:
            raise ValueError(f'Unknown version {self.config.version}')



        # Do NOT add batch dim
        # rows = torch.unsqueeze(rows, 0)

        # print("Final shape:", rows.shape)
        return rows
    def forward(self, x):
        if self.config.version == 300:
            x = self.resnet(x)
            x1 = self.dim_reduce1(x)
            x2 = self.dim_reduce2(x)
            x = torch.cat((x1, x2), dim=-1)
            x = self.last_act(x)
        else:
            for i in range(0, len(self.resnet)):
                x = self.resnet[i](x)
            # x = self.resnet(x)
            # For testing
            # for i in range(0, len(self.dim_reduce1)):
            #     x1 = self.dim_reduce1[i](x)
            x = self.dim_reduce1(x)


            if self.config.version >= 107:
                x = self.last_act(x)
                x = self.fc(torch.transpose(x, -1, -2))
            else:
                x = self.fc(torch.transpose(x, -1, -2))
                x = self.last_act(x)

        return torch.squeeze(x)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        try:
            self.load_state_dict(torch.load(filename, map_location=self.config.device))
        except FileNotFoundError:
            self.logger.warning(f'{filename} not found, using random weights.')
        except RuntimeError as err:
            self.logger.warning(f'{filename} model is incompatible with this version, using random weights. {err}')


if __name__ == '__main__':
    pass


