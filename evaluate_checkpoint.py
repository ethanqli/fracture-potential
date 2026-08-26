import argparse

import torch
from torch.utils.data import DataLoader

from load_data import load_raw_data, normalize_predictors_with_stats
from model import UNet
from spatial_split import make_spatial_splits
from train import (
    boundary_tolerant_threshold_metrics,
    make_dataset,
    run_epoch,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate an existing fracture U-Net checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-per-class", type=int, default=1000)
    parser.add_argument("--test-per-class", type=int, default=1000)
    parser.add_argument("--evaluation-min-positive-pixels", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[
            0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
            0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
        ],
    )
    parser.add_argument(
        "--boundary-tolerances", type=int, nargs="+", default=[1, 2, 3]
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.thresholds or any(not 0 <= value <= 1 for value in args.thresholds):
        raise ValueError("Thresholds must be between 0 and 1")
    if not args.boundary_tolerances or any(
        radius < 0 for radius in args.boundary_tolerances
    ):
        raise ValueError("Boundary tolerances must be non-negative")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=True
    )
    patch_size = checkpoint["patch_size"]
    base_channels = checkpoint["base_channels"]

    X_raw, y = load_raw_data()
    _, height, width = X_raw.shape
    _, val_region, test_region = make_spatial_splits(
        height, width, patch_size
    )
    X = normalize_predictors_with_stats(
        X_raw, checkpoint["normalization_stats"], copy=False
    )

    val_dataset = make_dataset(
        X,
        y,
        val_region,
        patch_size,
        args.val_per_class,
        args.val_per_class,
        args.seed + 1,
        min_positive_pixels=args.evaluation_min_positive_pixels,
    )
    test_dataset = make_dataset(
        X,
        y,
        test_region,
        patch_size,
        args.test_per_class,
        args.test_per_class,
        args.seed + 2,
        min_positive_pixels=args.evaluation_min_positive_pixels,
    )
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

    model = UNet(
        in_channels=checkpoint["input_channels"],
        out_channels=1,
        base_channels=base_channels,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    exact_threshold = checkpoint["inference_threshold"]
    test_metrics = run_epoch(
        model,
        test_loader,
        device,
        checkpoint.get("bce_weight", 0.5),
        checkpoint.get("dice_weight", 0.5),
        checkpoint.get("tversky_weight", 0.0),
        checkpoint.get("tversky_alpha", 0.3),
        checkpoint.get("tversky_beta", 0.7),
        threshold=exact_threshold,
    )
    print(
        f"Exact test at threshold {exact_threshold:.2f}: "
        f"dice={test_metrics['dice']:.4f} "
        f"precision={test_metrics['precision']:.4f} "
        f"recall={test_metrics['recall']:.4f}"
    )

    validation_results = boundary_tolerant_threshold_metrics(
        model,
        val_loader,
        device,
        args.thresholds,
        args.boundary_tolerances,
    )
    selected_thresholds = {
        radius: max(
            args.thresholds,
            key=lambda threshold: results[threshold]["dice"],
        )
        for radius, results in validation_results.items()
    }
    test_results = boundary_tolerant_threshold_metrics(
        model,
        test_loader,
        device,
        selected_thresholds.values(),
        args.boundary_tolerances,
    )

    for radius, threshold in selected_thresholds.items():
        validation = validation_results[radius][threshold]
        test = test_results[radius][threshold]
        print(
            f"Tolerance {radius} px selected threshold {threshold:.2f}: "
            f"validation dice={validation['dice']:.4f} "
            f"precision={validation['precision']:.4f} "
            f"recall={validation['recall']:.4f}"
        )
        print(
            f"Tolerance {radius} px test: dice={test['dice']:.4f} "
            f"precision={test['precision']:.4f} "
            f"recall={test['recall']:.4f}"
        )


if __name__ == "__main__":
    main()
