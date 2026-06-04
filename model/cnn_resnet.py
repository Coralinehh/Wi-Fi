import torch.nn as nn
import torch.nn.functional as F


class CNNResNet(nn.Module):
    def __init__(self, input_shape, channels=64, num_classes=22):
        super().__init__()
        _, steps, _, _, channels_in = input_shape
        self.initial_conv = nn.Sequential(
            nn.Conv2d(channels_in, channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.res_block1 = self._make_res_block(channels)
        self.res_block2 = self._make_res_block(channels)
        self.feature_reduction = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * steps, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def _make_res_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        batch, steps, height, width, channels = x.size()
        x = x.permute(0, 1, 4, 2, 3).reshape(batch * steps, channels, height, width)
        x = self.initial_conv(x)

        identity = x
        x = F.relu(self.res_block1(x) + identity)

        identity = x
        x = F.relu(self.res_block2(x) + identity)

        x = self.feature_reduction(x)
        x = x.reshape(batch, -1)
        return self.classifier(x)

