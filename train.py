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


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0

    with torch.set_grad_enabled(training):
        for X_batch, y_batch, ice_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            ice_batch = ice_batch.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad()

            logits = model(X_batch)
            loss = masked_bce_loss(logits, y_batch, ice_batch)
            if not torch.isfinite(loss):
                raise RuntimeError("Loss became non-finite")

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

    return total_loss / len(loader)


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default="unet_smoke_test.pt")
    return parser.parse_args()


def main():
    args = parse_args()
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
        train_loss = run_epoch(model, train_loader, device, optimizer)
        val_loss = run_epoch(model, val_loader, device)
        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "normalization_stats": normalization_stats,
                    "input_channels": X.shape[0],
                    "patch_size": args.patch_size,
                    "base_channels": args.base_channels,
                },
                args.checkpoint,
            )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss = run_epoch(model, test_loader, device)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Test loss: {test_loss:.6f}")
    print(f"Saved best checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
