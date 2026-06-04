import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, input_shape, channels=96, num_classes=22):
        super().__init__()
        _, steps, _, _, channels_in = input_shape
        self.initial_conv = nn.Sequential(
            nn.Conv2d(channels_in, channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.cnn_block1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.cnn_block2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.feature_reduction = nn.Sequential(
            nn.Conv2d(channels, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(128 * steps, num_classes)

    def forward(self, x):
        batch, steps, height, width, channels = x.size()
        x = x.permute(0, 1, 4, 2, 3).reshape(batch * steps, channels, height, width)
        x = self.initial_conv(x)
        x = self.cnn_block1(x)
        x = self.cnn_block2(x)
        x = self.feature_reduction(x)
        x = x.reshape(batch, steps * 128)
        return self.classifier(x)

