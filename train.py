import argparse

import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from torch.utils.data import DataLoader

from dataset import FracturePatchDataset
from load_data import load_raw_data, normalize_predictors
from model import UNet
from spatial_split import make_spatial_splits
from visualize_results import plot_training_history, visualize_test_predictions


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


def dilate_binary_mask(mask, radius):
    """Dilate a BCHW Boolean mask by an integer pixel radius."""
    if radius < 0:
        raise ValueError("Dilation radius must be non-negative")
    if radius == 0:
        return mask
    kernel_size = 2 * radius + 1
    return F.max_pool2d(
        mask.float(),
        kernel_size=kernel_size,
        stride=1,
        padding=radius,
    ) > 0


def boundary_tolerant_metrics(model, loader, device, threshold, radii=(1, 2)):
    """Evaluate segmentation allowing predictions within each pixel radius.

    Precision matches each predicted pixel against a dilated target, while
    recall matches each target pixel against a dilated prediction. Counts are
    accumulated over the full dataset before computing the metrics.
    """
    radii = tuple(dict.fromkeys(radii))
    counts = {
        radius: {
            "matched_predictions": 0,
            "predicted_pixels": 0,
            "matched_targets": 0,
            "target_pixels": 0,
        }
        for radius in radii
    }

    model.eval()
    with torch.inference_mode():
        for X_batch, y_batch, ice_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            ice_batch = ice_batch.to(device, non_blocking=True)

            valid = ice_batch > 0
            predictions = (torch.sigmoid(model(X_batch)) >= threshold) & valid
            targets = (y_batch > 0.5) & valid

            predicted_pixels = predictions.sum().item()
            target_pixels = targets.sum().item()
            for radius in radii:
                expanded_targets = dilate_binary_mask(targets, radius) & valid
                expanded_predictions = (
                    dilate_binary_mask(predictions, radius) & valid
                )
                radius_counts = counts[radius]
                radius_counts["matched_predictions"] += (
                    predictions & expanded_targets
                ).sum().item()
                radius_counts["predicted_pixels"] += predicted_pixels
                radius_counts["matched_targets"] += (
                    targets & expanded_predictions
                ).sum().item()
                radius_counts["target_pixels"] += target_pixels

    results = {}
    for radius, radius_counts in counts.items():
        predicted_pixels = radius_counts["predicted_pixels"]
        target_pixels = radius_counts["target_pixels"]
        if predicted_pixels == 0:
            precision = 1.0 if target_pixels == 0 else 0.0
        else:
            precision = (
                radius_counts["matched_predictions"] / predicted_pixels
            )
        if target_pixels == 0:
            recall = 1.0 if predicted_pixels == 0 else 0.0
        else:
            recall = radius_counts["matched_targets"] / target_pixels
        dice = (
            0.0
            if precision + recall == 0
            else 2.0 * precision * recall / (precision + recall)
        )
        results[radius] = {
            "dice": dice,
            "precision": precision,
            "recall": recall,
        }
    return results


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


def validate_epoch(
    model,
    loader,
    device,
    bce_weight,
    dice_weight,
    thresholds,
    reporting_threshold,
):
    """Compute validation loss and threshold metrics in one inference pass."""
    model.eval()
    evaluated_thresholds = list(
        dict.fromkeys([reporting_threshold, *thresholds])
    )
    counts = {
        threshold: {"tp": 0, "fp": 0, "fn": 0}
        for threshold in evaluated_thresholds
    }
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch, ice_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            ice_batch = ice_batch.to(device, non_blocking=True)

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
            total_loss += loss.item()

            probabilities = torch.sigmoid(logits)
            targets = y_batch > 0.5
            valid = ice_batch > 0

            for threshold in evaluated_thresholds:
                predictions = probabilities >= threshold
                counts[threshold]["tp"] += (
                    predictions & targets & valid
                ).sum().item()
                counts[threshold]["fp"] += (
                    predictions & ~targets & valid
                ).sum().item()
                counts[threshold]["fn"] += (
                    ~predictions & targets & valid
                ).sum().item()

    threshold_metrics = {
        threshold: segmentation_metrics(
            count["tp"], count["fp"], count["fn"]
        )
        for threshold, count in counts.items()
    }
    reporting_metrics = threshold_metrics[reporting_threshold].copy()
    reporting_metrics["loss"] = total_loss / len(loader)
    return reporting_metrics, threshold_metrics


def evaluate_by_fracture_density(model, loader, device, threshold):
    """Return pixel metrics for sparse, medium, and dense positive patches."""
    bins = {
        "sparse (1-31 px)": {"tp": 0, "fp": 0, "fn": 0, "patches": 0},
        "medium (32-127 px)": {"tp": 0, "fp": 0, "fn": 0, "patches": 0},
        "dense (128+ px)": {"tp": 0, "fp": 0, "fn": 0, "patches": 0},
    }
    names = list(bins)
    model.eval()
    with torch.inference_mode():
        for X_batch, y_batch, ice_batch in loader:
            probabilities = torch.sigmoid(model(X_batch.to(device))).cpu()
            for item in range(len(X_batch)):
                target = y_batch[item, 0] > 0.5
                valid = ice_batch[item, 0] > 0
                positive_count = int((target & valid).sum().item())
                if positive_count == 0:
                    continue
                bin_index = 0 if positive_count < 32 else (1 if positive_count < 128 else 2)
                counts = bins[names[bin_index]]
                prediction = probabilities[item, 0] >= threshold
                counts["tp"] += int((prediction & target & valid).sum().item())
                counts["fp"] += int((prediction & ~target & valid).sum().item())
                counts["fn"] += int((~prediction & target & valid).sum().item())
                counts["patches"] += 1

    results = {}
    for name, counts in bins.items():
        metrics = segmentation_metrics(counts["tp"], counts["fp"], counts["fn"])
        metrics["patches"] = counts["patches"]
        results[name] = metrics
    return results


def make_dataset(
    X,
    y,
    region,
    patch_size,
    positives,
    negatives,
    seed,
    augment=False,
    normalization_stats=None,
    positive_density_upper_bounds=None,
    boundary_distance=None,
    min_positive_pixels=8,
):
    return FracturePatchDataset(
        X_full=X,
        y_full=y,
        ice_mask=X[7],
        allowed_mask=region,
        patch_size=patch_size,
        positive_patches=positives,
        negative_patches=negatives,
        seed=seed,
        augment=augment,
        normalization_stats=normalization_stats,
        positive_density_upper_bounds=positive_density_upper_bounds,
        boundary_distance=boundary_distance,
        min_positive_pixels=min_positive_pixels,
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
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
        help="Validation thresholds used to select the best inference threshold.",
    )
    parser.add_argument(
        "--boundary-tolerances",
        type=int,
        nargs="+",
        default=[1, 2],
        help="Pixel radii used for boundary-tolerant test metrics.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default="unet_smoke_test.pt")
    parser.add_argument(
        "--stratified-positive-sampling",
        action="store_true",
        help="Sample equal counts of 1-31, 32-127, and 128+ pixel positive patches.",
    )
    parser.add_argument(
        "--boundary-matched-negatives",
        action="store_true",
        help="Match negative patches to positive-patch ice-boundary distances.",
    )
    parser.add_argument(
        "--resample-each-epoch",
        action="store_true",
        help="Regenerate training coordinates each epoch instead of keeping them fixed.",
    )
    parser.add_argument("--training-min-positive-pixels", type=int, default=1)
    parser.add_argument(
        "--evaluation-min-positive-pixels",
        type=int,
        default=8,
        help="Keep at 8 to preserve comparability with earlier validation/test sets.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.bce_weight < 0 or args.dice_weight < 0:
        raise ValueError("Loss weights must be non-negative")
    if args.bce_weight + args.dice_weight == 0:
        raise ValueError("At least one loss weight must be positive")
    if not 0 <= args.threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    if not args.thresholds or any(
        threshold < 0 or threshold > 1 for threshold in args.thresholds
    ):
        raise ValueError("All validation thresholds must be between 0 and 1")
    if not args.boundary_tolerances or any(
        radius < 0 for radius in args.boundary_tolerances
    ):
        raise ValueError("Boundary tolerances must be non-negative integers")
    if (
        args.training_min_positive_pixels <= 0
        or args.evaluation_min_positive_pixels <= 0
    ):
        raise ValueError("minimum positive-pixel counts must be positive")
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

    boundary_distance = None
    if args.boundary_matched_negatives:
        print("Computing distance to the nearest non-ice pixel...")
        # Distances are only used for integer bins; uint16 halves memory versus
        # float32 and easily covers this raster's dimensions.
        boundary_distance = distance_transform_edt(X[7] > 0).astype("uint16")
    density_bounds = (32, 128) if args.stratified_positive_sampling else None

    train_dataset = make_dataset(
        X, y, train_region, args.patch_size,
        args.train_per_class, args.train_per_class, args.seed,
        augment=False, normalization_stats=normalization_stats,
        positive_density_upper_bounds=density_bounds,
        boundary_distance=boundary_distance,
        min_positive_pixels=args.training_min_positive_pixels,
    )
    val_dataset = make_dataset(
        X, y, val_region, args.patch_size,
        args.val_per_class, args.val_per_class, args.seed + 1,
        min_positive_pixels=args.evaluation_min_positive_pixels,
    )
    test_dataset = make_dataset(
        X, y, test_region, args.patch_size,
        args.test_per_class, args.test_per_class, args.seed + 2,
        min_positive_pixels=args.evaluation_min_positive_pixels,
    )
    if args.stratified_positive_sampling:
        print("Training positive density-bin counts:", train_dataset.positive_density_counts)
    if args.boundary_matched_negatives:
        print("Matched boundary-distance bin counts:", train_dataset.boundary_bin_counts)

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
    best_val_dice = -1.0
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_dice": [],
        "val_dice": [],
    }
    for epoch in range(args.epochs):
        # Epoch zero uses the coordinates created during dataset construction;
        # subsequent epochs receive new deterministic samples.
        if args.resample_each_epoch and epoch > 0:
            train_dataset.resample(args.seed + epoch)
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            args.bce_weight,
            args.dice_weight,
            optimizer,
            args.threshold,
        )
        val_metrics, validation_results = validate_epoch(
            model,
            val_loader,
            device,
            args.bce_weight,
            args.dice_weight,
            args.thresholds,
            args.threshold,
        )
        epoch_threshold = max(
            args.thresholds,
            key=lambda threshold: validation_results[threshold]["dice"],
        )
        epoch_threshold_metrics = validation_results[epoch_threshold]
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_dice"].append(train_metrics["dice"])
        history["val_dice"].append(epoch_threshold_metrics["dice"])
        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_dice={train_metrics['dice']:.4f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_dice={val_metrics['dice']:.4f} "
            f"val_precision={val_metrics['precision']:.4f} "
            f"val_recall={val_metrics['recall']:.4f} "
            f"best_threshold={epoch_threshold:.2f} "
            f"tuned_val_dice={epoch_threshold_metrics['dice']:.4f}"
        )

        best_val_loss = min(best_val_loss, val_metrics["loss"])
        if epoch_threshold_metrics["dice"] > best_val_dice:
            best_val_dice = epoch_threshold_metrics["dice"]
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "normalization_stats": normalization_stats,
                    "input_channels": X.shape[0],
                    "patch_size": args.patch_size,
                    "base_channels": args.base_channels,
                    "bce_weight": args.bce_weight,
                    "dice_weight": args.dice_weight,
                    "threshold": args.threshold,
                    "inference_threshold": epoch_threshold,
                    "validation_dice": best_val_dice,
                    "validation_loss": val_metrics["loss"],
                    "stratified_positive_sampling": args.stratified_positive_sampling,
                    "boundary_matched_negatives": args.boundary_matched_negatives,
                    "training_min_positive_pixels": args.training_min_positive_pixels,
                    "evaluation_min_positive_pixels": args.evaluation_min_positive_pixels,
                },
                args.checkpoint,
            )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_threshold = checkpoint["inference_threshold"]

    test_metrics = run_epoch(
        model,
        test_loader,
        device,
        args.bce_weight,
        args.dice_weight,
        threshold=best_threshold,
    )
    print(f"Lowest observed validation loss: {best_val_loss:.6f}")
    print(
        f"Selected checkpoint: epoch {checkpoint['epoch']} "
        f"with threshold {best_threshold:.2f} "
        f"validation Dice {checkpoint['validation_dice']:.4f} "
        f"and validation loss {checkpoint['validation_loss']:.6f}"
    )
    print(f"Test loss: {test_metrics['loss']:.6f}")
    print(f"Test Dice: {test_metrics['dice']:.4f}")
    print(f"Test precision: {test_metrics['precision']:.4f}")
    print(f"Test recall: {test_metrics['recall']:.4f}")
    tolerant_metrics = boundary_tolerant_metrics(
        model,
        test_loader,
        device,
        best_threshold,
        args.boundary_tolerances,
    )
    for radius, metrics in tolerant_metrics.items():
        print(
            f"Test tolerance {radius} px: dice={metrics['dice']:.4f} "
            f"precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f}"
        )
    density_metrics = evaluate_by_fracture_density(
        model, test_loader, device, best_threshold
    )
    for name, metrics in density_metrics.items():
        print(
            f"Test {name}: patches={metrics['patches']} "
            f"dice={metrics['dice']:.4f} precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f}"
        )
    print(f"Saved best checkpoint to {args.checkpoint}")

    plot_training_history(history, output="training_curves.png")
    visualize_test_predictions(
        model,
        test_dataset,
        device,
        best_threshold,
        output="test_predictions.png",
    )


if __name__ == "__main__":
    main()
