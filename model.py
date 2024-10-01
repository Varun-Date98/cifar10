import torch
from torch import nn as nn


class TinyVGG(nn.Module):
    def __init__(self, input_shape: int, num_classes: int):
        super(TinyVGG, self).__init__()
        self.conv_block = nn.Sequential(
            #=================================#
            #             BLOCK 1             #
            #=================================#
            nn.Conv2d(in_channels=input_shape,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            #=================================#
            #           Max Pool 1            #
            #=================================#
            nn.MaxPool2d(kernel_size=2),
            #=================================#
            #             BLOCK 2             #
            #=================================#
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            #=================================#
            #           Max Pool 2            #
            #=================================#
            nn.MaxPool2d(kernel_size=2),
            #=================================#
            #             BLOCK 3             #
            #=================================#
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            #=================================#
            #           Max Pool 3            #
            #=================================#
            nn.MaxPool2d(kernel_size=2),
            #=================================#
            #             BLOCK 4             #
            #=================================#
            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            #=================================#
            #           Max Pool 4            #
            #=================================#
            nn.MaxPool2d(kernel_size=2)
        )
        self.fcs = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=2048),
            nn.Linear(in_features=2048, out_features=2048),
            nn.Linear(in_features=2048, out_features=num_classes)
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.fcs(self.conv_block(x))
