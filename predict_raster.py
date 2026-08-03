import argparse

import torch

from full_raster_inference import predict_full_raster, save_prediction_geotiffs
from load_data import (
    PREDICTOR_PATHS,
    load_raw_predictors,
    normalize_predictors_with_stats,
)
from model import UNet


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a trained fracture U-Net over the complete raster."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--potential-output", default="fracture_potential.tif")
    parser.add_argument(
        "--binary-output",
        default=None,
        help="Optional thresholded uint8 GeoTIFF.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the validation-selected checkpoint threshold.",
    )
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)

    model = UNet(
        in_channels=checkpoint["input_channels"],
        out_channels=1,
        base_channels=checkpoint["base_channels"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    X_raw = load_raw_predictors()
    X = normalize_predictors_with_stats(
        X_raw, checkpoint["normalization_stats"], copy=False
    )
    potential = predict_full_raster(
        model,
        X,
        device,
        patch_size=checkpoint["patch_size"],
        batch_size=args.batch_size,
        stride=args.stride,
    )
    threshold = (
        checkpoint["inference_threshold"]
        if args.threshold is None
        else args.threshold
    )
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    save_prediction_geotiffs(
        potential,
        X[7],
        PREDICTOR_PATHS[0],
        args.potential_output,
        threshold,
        args.binary_output,
    )


if __name__ == "__main__":
    main()
