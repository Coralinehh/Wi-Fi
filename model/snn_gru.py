import torch
import torch.nn as nn

try:
    from spikingjelly.activation_based import functional, layer, neuron, surrogate
except ImportError as exc:
    functional = layer = neuron = surrogate = None
    SPIKINGJELLY_IMPORT_ERROR = exc
else:
    SPIKINGJELLY_IMPORT_ERROR = None


class SNNGRU(nn.Module):
    def __init__(self, channels=32, gru_hidden_size=196, num_classes=22, v_threshold=0.5, time_groups=None):
        if SPIKINGJELLY_IMPORT_ERROR is not None:
            raise ImportError("SNNGRU requires spikingjelly. Install it with `pip install spikingjelly`.") from SPIKINGJELLY_IMPORT_ERROR

        super().__init__()
        self.snn = layer.SeqToANNContainer(
            layer.Conv2d(1, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.IFNode(v_threshold=v_threshold, surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            layer.Flatten(),
            layer.Linear(channels * 10 * 10, 64, bias=False),
            neuron.IFNode(v_threshold=v_threshold, surrogate_function=surrogate.ATan()),
        )
        self.gru = nn.GRU(64, gru_hidden_size, batch_first=True)
        self.classifier = nn.Linear(gru_hidden_size, num_classes)
        self.time_groups = time_groups

    def forward(self, x, time_groups=None):
        batch, steps, height, width, channels = x.size()
        time_groups = self.time_groups if time_groups is None else time_groups

        if time_groups is None:
            x = x.permute(1, 0, 4, 2, 3)
            functional.reset_net(self)
            x = self.snn(x)
            x = x.permute(1, 0, 2)
        else:
            group_len = steps // time_groups
            steps_used = group_len * time_groups
            x = x[:, :steps_used].view(batch, time_groups, group_len, height, width, channels)
            outputs = []
            functional.reset_net(self)
            for group_index in range(time_groups):
                x_group = x[:, group_index].permute(1, 0, 4, 2, 3)
                out_group = self.snn(x_group).permute(1, 0, 2)
                outputs.append(out_group)
            x = torch.cat(outputs, dim=1)

        _, hidden = self.gru(x)
        return self.classifier(hidden.squeeze(0))
