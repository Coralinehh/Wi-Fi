import torch.nn as nn


class CNNLSTM(nn.Module):
    def __init__(self, channels=32, lstm_hidden_size=64, num_classes=22):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(channels * 2 * 10 * 10, 64),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=lstm_hidden_size, batch_first=True)
        self.classifier = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        batch, steps, height, width, channels = x.size()
        x = x.permute(0, 1, 4, 2, 3).reshape(batch * steps, channels, height, width)
        x = self.cnn(x)
        x = x.reshape(batch, steps, -1)
        _, (hidden, _) = self.lstm(x)
        return self.classifier(hidden.squeeze(0))

