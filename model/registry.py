from .cnn import CNN
from .cnn_lstm import CNNLSTM
from .cnn_resnet import CNNResNet
from .cnn_transformer import CNNTransformer
from .snn_gru import SNNGRU


def _build_cnn(input_shape, num_classes):
    return CNN(input_shape=input_shape, num_classes=num_classes)


def _build_cnnres(input_shape, num_classes):
    return CNNResNet(input_shape=input_shape, num_classes=num_classes)


def _build_lstm(input_shape, num_classes):
    return CNNLSTM(num_classes=num_classes)


def _build_snngru(input_shape, num_classes, v_threshold=0.5, time_groups=None):
    return SNNGRU(num_classes=num_classes, v_threshold=v_threshold, time_groups=time_groups)


def _build_transformer(input_shape, num_classes):
    return CNNTransformer(num_classes=num_classes)


BVP_MODEL_REGISTRY = {
    "cnn": _build_cnn,
    "cnnres": _build_cnnres,
    "lstm": _build_lstm,
    "snngru": _build_snngru,
    "transformer": _build_transformer,
}


def build_bvp_model(name, input_shape, num_classes=22, **kwargs):
    try:
        builder = BVP_MODEL_REGISTRY[name]
    except KeyError as exc:
        choices = ", ".join(sorted(BVP_MODEL_REGISTRY))
        raise ValueError(f"Unknown BVP model '{name}'. Available models: {choices}") from exc
    if name == "snngru":
        return builder(input_shape, num_classes, **kwargs)
    return builder(input_shape, num_classes)


MODEL_REGISTRY = BVP_MODEL_REGISTRY
build_model = build_bvp_model
