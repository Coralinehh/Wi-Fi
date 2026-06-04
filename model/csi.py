import torch.nn as nn

try:
    from spikingjelly.activation_based import functional, layer, neuron, surrogate
except ImportError as exc:
    functional = layer = neuron = surrogate = None
    SPIKINGJELLY_IMPORT_ERROR = exc
else:
    SPIKINGJELLY_IMPORT_ERROR = None


class CSIConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(x)))


class CSICNN(nn.Module):
    def __init__(self, channels=64, num_classes=14):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=5, padding=2, stride=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4)),
        )
        self.conv_block1 = CSIConvBlock(channels, channels * 2)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))
        self.conv_block2 = CSIConvBlock(channels * 2, channels * 4)
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.conv_block1(x)
        x = self.pool1(x)
        x = self.conv_block2(x)
        x = self.pool2(x)
        return self.classifier(x)


class CSIBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(identity)
        return self.relu(out + identity)


class CSICNNResNet(nn.Module):
    def __init__(self, channels=52, num_classes=14):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=5, padding=2, stride=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4)),
        )
        self.res_block1 = nn.Sequential(
            CSIBasicBlock(channels, channels * 2),
            CSIBasicBlock(channels * 2, channels * 2),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))
        self.res_block2 = nn.Sequential(
            CSIBasicBlock(channels * 2, channels * 4),
            CSIBasicBlock(channels * 4, channels * 4),
        )
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.res_block1(x)
        x = self.pool1(x)
        x = self.res_block2(x)
        x = self.pool2(x)
        return self.classifier(x)


class CSILSTM(nn.Module):
    def __init__(self, num_classes=14, lstm_hidden=128, lstm_layers=3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.lstm = nn.LSTM(
            input_size=128 * 28,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.4,
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.size(0), x.size(1), -1)
        x, _ = self.lstm(x)
        return self.classifier(x[:, -1, :])


class CSITransformer(nn.Module):
    def __init__(self, num_classes=14, embed_dim=144, num_heads=8, num_layers=4, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(96, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        self.input_proj = nn.Linear(192 * 28, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)
        x = x.flatten(2)
        x = self.input_proj(x)
        x = self.transformer(x)
        return self.classifier(x[:, -1, :])


class SNN(nn.Module):
    def __init__(self, time_steps=10, channels=32, num_classes=14, v_threshold=0.5, input_width=500):
        if SPIKINGJELLY_IMPORT_ERROR is not None:
            raise ImportError("CSI SNN requires spikingjelly. Install it with `pip install spikingjelly`.") from SPIKINGJELLY_IMPORT_ERROR

        super().__init__()
        self.time_steps = time_steps
        segment_len = input_width // time_steps
        width_final = segment_len // 4
        height_final = 28
        self.net = nn.Sequential(
            layer.Conv2d(3, channels, kernel_size=5, padding=2, bias=False),
            layer.BatchNorm2d(channels),
            neuron.IFNode(v_threshold=v_threshold, surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.IFNode(v_threshold=v_threshold, surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            layer.Flatten(),
            layer.Linear(channels * height_final * width_final, 148, bias=False),
            neuron.IFNode(v_threshold=v_threshold, surrogate_function=surrogate.ATan()),
            layer.Linear(148, num_classes, bias=False),
        )

    def forward(self, x):
        _, _, _, total_steps = x.shape
        if total_steps % self.time_steps != 0:
            raise ValueError(f"Input time length {total_steps} must be divisible by time_steps={self.time_steps}.")

        segment_len = total_steps // self.time_steps
        x = x.unfold(dimension=3, size=segment_len, step=segment_len)
        x = x.permute(3, 0, 1, 2, 4)
        functional.reset_net(self)
        out = 0
        for step in range(self.time_steps):
            out = out + self.net(x[step])
        return out / self.time_steps


CSI_MODEL_REGISTRY = {
    "cnn": CSICNN,
    "cnnres": CSICNNResNet,
    "lstm": CSILSTM,
    "snn": SNN,
    "transformer": CSITransformer,
}


CSISNN = SNN


def build_csi_model(name, num_classes=14, time_steps=10, channels=None, v_threshold=0.5):
    try:
        model_cls = CSI_MODEL_REGISTRY[name]
    except KeyError as exc:
        choices = ", ".join(sorted(CSI_MODEL_REGISTRY))
        raise ValueError(f"Unknown CSI model '{name}'. Available models: {choices}") from exc

    if name == "snn":
        kwargs = {"time_steps": time_steps, "num_classes": num_classes, "v_threshold": v_threshold}
        if channels is not None:
            kwargs["channels"] = channels
        return model_cls(**kwargs)
    if channels is not None and name in {"cnn", "cnnres"}:
        return model_cls(channels=channels, num_classes=num_classes)
    return model_cls(num_classes=num_classes)
