import argparse
from pathlib import Path

import torch

from data import build_csi_loaders
from model import CSI_MODEL_REGISTRY, build_csi_model
from train.trainer import fit


DATASET_DEFAULTS = {
    "humanid": {
        "data_root": "dataset/NTU-Fi-HumanID",
        "num_classes": 14,
    },
    "har": {
        "data_root": "dataset/NTU-Fi_HAR",
        "num_classes": 6,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train NTU-Fi CSI models for HumanID or HAR.")
    parser.add_argument("--dataset", default="humanid", choices=sorted(DATASET_DEFAULTS))
    parser.add_argument("--model", default="cnn", choices=sorted(CSI_MODEL_REGISTRY))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adam", "rmsprop"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--scheduler", default="cosine", choices=["cosine", "step", "none"])
    parser.add_argument("--time-steps", type=int, default=10)
    parser.add_argument("--time-steps-list", default="", help="Comma-separated SNN time steps, e.g. 5,10,15.")
    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--v-threshold", type=float, default=0.5)
    parser.add_argument(
        "--v-thresholds",
        default="",
        help="Comma-separated thresholds for SNN sweeps, e.g. 0.25,0.5,0.75.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--modal", default="CSIamp")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--early-stop-patience", type=int, default=20)
    return parser.parse_args()


def parse_thresholds(args):
    if args.v_thresholds:
        return [float(item.strip()) for item in args.v_thresholds.split(",") if item.strip()]
    return [args.v_threshold]


def parse_time_steps(args):
    if args.time_steps_list:
        return [int(item.strip()) for item in args.time_steps_list.split(",") if item.strip()]
    return [args.time_steps]


def main():
    args = parse_args()
    defaults = DATASET_DEFAULTS[args.dataset]
    data_root = args.data_root or defaults["data_root"]
    num_classes = args.num_classes or defaults["num_classes"]

    torch.manual_seed(args.seed)
    loaders, detected_classes = build_csi_loaders(
        data_root=data_root,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        modal=args.modal,
        pin_memory=torch.cuda.is_available(),
    )
    if args.num_classes is None:
        num_classes = detected_classes

    device = torch.device(args.device)
    thresholds = parse_thresholds(args)
    time_steps_values = parse_time_steps(args)
    for time_steps in time_steps_values:
        for v_threshold in thresholds:
            model = build_csi_model(
                args.model,
                num_classes=num_classes,
                time_steps=time_steps,
                channels=args.channels,
                v_threshold=v_threshold,
            )

            model = model.to(device)
            print(model)

            suffix = ""
            if args.model == "snn":
                suffix = f"_T{time_steps}_vth{v_threshold:g}"
            checkpoint_path = Path(args.checkpoint_dir) / f"{args.dataset}_{args.model}{suffix}_best.pt"
            fit(
                model=model,
                loaders=loaders,
                device=device,
                epochs=args.epochs,
                learning_rate=args.lr,
                num_classes=num_classes,
                checkpoint_path=checkpoint_path,
                early_stop_patience=args.early_stop_patience,
                optimizer_name=args.optimizer,
                momentum=args.momentum,
                scheduler_name=args.scheduler,
                amp_enabled=args.amp,
            )


if __name__ == "__main__":
    main()
