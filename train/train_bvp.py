import argparse
from pathlib import Path

import torch

from data import build_bvp_loaders
from model import BVP_MODEL_REGISTRY, build_bvp_model
from train.trainer import fit


def parse_args():
    parser = argparse.ArgumentParser(description="Train BVP 22-gesture models.")
    parser.add_argument("--model", default="cnn", choices=sorted(BVP_MODEL_REGISTRY))
    parser.add_argument("--data-root", default="dataset/BVP_22guesture_all")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num-classes", type=int, default=22)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--v-threshold", type=float, default=0.5)
    parser.add_argument(
        "--v-thresholds",
        default="",
        help="Comma-separated thresholds for SNN-GRU sweeps, e.g. 0.25,0.5,0.75.",
    )
    parser.add_argument(
        "--time-groups",
        default=None,
        help="Only for snngru. Use 'none' for full-time baseline or an integer K for grouped time.",
    )
    parser.add_argument(
        "--time-groups-list",
        default="",
        help="Comma-separated BVP SNN-GRU K values. Use none for full-time baseline, e.g. none,4,8.",
    )
    return parser.parse_args()


def parse_time_groups(value):
    if value is None or str(value).lower() == "none":
        return None
    return int(value)


def parse_thresholds(args):
    if args.v_thresholds:
        return [float(item.strip()) for item in args.v_thresholds.split(",") if item.strip()]
    return [args.v_threshold]


def parse_time_group_list(args):
    if args.time_groups_list:
        return [parse_time_groups(item.strip()) for item in args.time_groups_list.split(",") if item.strip()]
    return [parse_time_groups(args.time_groups)]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    loaders, input_shape = build_bvp_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    device = torch.device(args.device)
    thresholds = parse_thresholds(args)
    time_group_values = parse_time_group_list(args)

    for time_groups in time_group_values:
        for v_threshold in thresholds:
            model_kwargs = {}
            if args.model == "snngru":
                model_kwargs = {"v_threshold": v_threshold, "time_groups": time_groups}
            model = build_bvp_model(args.model, input_shape=input_shape, num_classes=args.num_classes, **model_kwargs)

            model = model.to(device)
            print(model)

            suffix = ""
            if args.model == "snngru":
                group_name = "full" if time_groups is None else f"k{time_groups}"
                suffix = f"_vth{v_threshold:g}_{group_name}"
            checkpoint_path = Path(args.checkpoint_dir) / f"bvp_{args.model}{suffix}_best.pt"
            fit(
                model=model,
                loaders=loaders,
                device=device,
                epochs=args.epochs,
                learning_rate=args.lr,
                num_classes=args.num_classes,
                checkpoint_path=checkpoint_path,
                early_stop_patience=args.early_stop_patience,
            )


if __name__ == "__main__":
    main()
