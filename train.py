import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import FracturePatchDataset
from load_data import load_raw_data, normalize_predictors
from model import UNet
from spatial_split import make_spatial_splits


def masked_bce_loss(logits, targets, ice_mask):
    valid = ice_mask > 0
    if not valid.any():
        raise ValueError("Batch contains no valid ice pixels")
    pixel_loss = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    return pixel_loss[valid].mean()


def masked_dice_loss(logits, targets, ice_mask, smooth=1.0):
    """Soft Dice loss, averaged per patch over valid ice pixels only."""
    valid = ice_mask > 0
    probabilities = torch.sigmoid(logits) * valid
    masked_targets = targets * valid

    spatial_dims = tuple(range(1, logits.ndim))
    intersection = (probabilities * masked_targets).sum(dim=spatial_dims)
    denominator = (
        probabilities.sum(dim=spatial_dims)
        + masked_targets.sum(dim=spatial_dims)
    )
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


def combined_loss(logits, targets, ice_mask, bce_weight, dice_weight):
    loss = logits.new_zeros(())
    if bce_weight:
        loss = loss + bce_weight * masked_bce_loss(logits, targets, ice_mask)
    if dice_weight:
        loss = loss + dice_weight * masked_dice_loss(logits, targets, ice_mask)
    return loss / (bce_weight + dice_weight)


def segmentation_metrics(true_positives, false_positives, false_negatives):
    """Compute dataset-level binary segmentation metrics from pixel counts."""
    denominator = 2 * true_positives + false_positives + false_negatives
    dice = 1.0 if denominator == 0 else 2 * true_positives / denominator
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = (
        1.0
        if precision_denominator == 0
        else true_positives / precision_denominator
    )
    recall = (
        1.0 if recall_denominator == 0 else true_positives / recall_denominator
    )
    return {"dice": dice, "precision": precision, "recall": recall}


def run_epoch(
    model,
    loader,
    device,
    bce_weight,
    dice_weight,
    optimizer=None,
    threshold=0.5,
):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    with torch.set_grad_enabled(training):
        for X_batch, y_batch, ice_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            ice_batch = ice_batch.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad()

            logits = model(X_batch)
            loss = combined_loss(
                logits,
                y_batch,
                ice_batch,
                bce_weight,
                dice_weight,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Loss became non-finite")

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            with torch.no_grad():
                valid = ice_batch > 0
                predictions = torch.sigmoid(logits) >= threshold
                targets = y_batch > 0.5
                true_positives += (
                    predictions & targets & valid
                ).sum().item()
                false_positives += (
                    predictions & ~targets & valid
                ).sum().item()
                false_negatives += (
                    ~predictions & targets & valid
                ).sum().item()

    metrics = segmentation_metrics(
        true_positives, false_positives, false_negatives
    )
    metrics["loss"] = total_loss / len(loader)
    return metrics


def make_dataset(X, y, region, patch_size, positives, negatives, seed):
    return FracturePatchDataset(
        X_full=X,
        y_full=y,
        ice_mask=X[7],
        allowed_mask=region,
        patch_size=patch_size,
        positive_patches=positives,
        negative_patches=negatives,
        seed=seed,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train the fracture U-Net.")
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-per-class", type=int, default=50)
    parser.add_argument("--val-per-class", type=int, default=20)
    parser.add_argument("--test-per-class", type=int, default=20)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--bce-weight", type=float, default=0.5)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default="unet_smoke_test.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.bce_weight < 0 or args.dice_weight < 0:
        raise ValueError("Loss weights must be non-negative")
    if args.bce_weight + args.dice_weight == 0:
        raise ValueError("At least one loss weight must be positive")
    if not 0 <= args.threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    X_raw, y = load_raw_data()
    _, height, width = X_raw.shape
    train_region, val_region, test_region = make_spatial_splits(
        height, width, args.patch_size
    )

    # Only training-region pixels contribute to normalization statistics.
    # Normalize in place to avoid a second multi-gigabyte raster-stack copy.
    X, normalization_stats = normalize_predictors(
        X_raw, train_region, copy=False
    )
    print("X shape:", X.shape, "y shape:", y.shape)

    train_dataset = make_dataset(
        X, y, train_region, args.patch_size,
        args.train_per_class, args.train_per_class, args.seed,
    )
    val_dataset = make_dataset(
        X, y, val_region, args.patch_size,
        args.val_per_class, args.val_per_class, args.seed + 1,
    )
    test_dataset = make_dataset(
        X, y, test_region, args.patch_size,
        args.test_per_class, args.test_per_class, args.seed + 2,
    )

    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

    model = UNet(
        in_channels=X.shape[0],
        out_channels=1,
        base_channels=args.base_channels,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate
    )

    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            args.bce_weight,
            args.dice_weight,
            optimizer,
            args.threshold,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            args.bce_weight,
            args.dice_weight,
            threshold=args.threshold,
        )
        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_dice={train_metrics['dice']:.4f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_dice={val_metrics['dice']:.4f} "
            f"val_precision={val_metrics['precision']:.4f} "
            f"val_recall={val_metrics['recall']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "normalization_stats": normalization_stats,
                    "input_channels": X.shape[0],
                    "patch_size": args.patch_size,
                    "base_channels": args.base_channels,
                    "bce_weight": args.bce_weight,
                    "dice_weight": args.dice_weight,
                    "threshold": args.threshold,
                },
                args.checkpoint,
            )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = run_epoch(
        model,
        test_loader,
        device,
        args.bce_weight,
        args.dice_weight,
        threshold=args.threshold,
    )
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Test loss: {test_metrics['loss']:.6f}")
    print(f"Test Dice: {test_metrics['dice']:.4f}")
    print(f"Test precision: {test_metrics['precision']:.4f}")
    print(f"Test recall: {test_metrics['recall']:.4f}")
    print(f"Saved best checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
