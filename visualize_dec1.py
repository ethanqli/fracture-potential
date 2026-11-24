"""Visualize final-decoder activations for a trained fracture U-Net.

The script creates three figures for one held-out test patch:

1. Context, truth, probability, positive evidence, negative evidence, and logit.
2. Every raw dec1 activation map.
3. Every dec1 activation multiplied by its final 1x1 output weight.

The weighted maps are an exact decomposition of the output logit:

    logit = output_bias + sum(dec1_feature * output_weight)
"""

import argparse
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="matplotlib-"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader

from dataset import FracturePatchDataset
from load_data import PREDICTOR_NAMES, load_raw_data, normalize_predictors_with_stats
from model import UNet
from spatial_split import make_spatial_splits


SELECTIONS = (
    "lowest-recall-positive",
    "highest-recall-positive",
    "highest-potential-negative",
    "random-positive",
    "random-negative",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize dec1 activations from a fracture U-Net checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="dec1_visualization")
    parser.add_argument(
        "--selection",
        choices=SELECTIONS,
        default="lowest-recall-positive",
        help="How to select a diagnostic patch when --sample-index is omitted.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Use this exact test-dataset index instead of ranking patches.",
    )
    parser.add_argument("--samples-per-class", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-positive-pixels",
        type=int,
        default=None,
        help="Defaults to the evaluation value stored in the checkpoint, or 8.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the validation-selected checkpoint threshold.",
    )
    return parser.parse_args()


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # Compatibility with older PyTorch releases.
        return torch.load(path, map_location=device)


def make_test_dataset(checkpoint, samples_per_class, seed, min_positive_pixels):
    required = ("normalization_stats", "patch_size", "input_channels")
    missing = [name for name in required if name not in checkpoint]
    if missing:
        raise ValueError(
            "Checkpoint is missing required metadata: " + ", ".join(missing)
        )

    X_raw, target = load_raw_data()
    patch_size = int(checkpoint["patch_size"])
    _, _, test_region = make_spatial_splits(
        X_raw.shape[1], X_raw.shape[2], patch_size
    )
    X = normalize_predictors_with_stats(
        X_raw, checkpoint["normalization_stats"], copy=False
    )

    saved_input_indices = checkpoint.get("input_channel_indices")
    input_indices = tuple(
        range(int(checkpoint["input_channels"]))
        if saved_input_indices is None
        else saved_input_indices
    )
    dataset = FracturePatchDataset(
        X_full=X,
        y_full=target,
        ice_mask=X[7],
        allowed_mask=test_region,
        patch_size=patch_size,
        positive_patches=samples_per_class,
        negative_patches=samples_per_class,
        seed=seed + 2,
        min_positive_pixels=min_positive_pixels,
        input_channel_indices=input_indices,
    )
    return dataset, input_indices


def select_sample(model, dataset, device, threshold, args):
    if args.sample_index is not None:
        if not 0 <= args.sample_index < len(dataset):
            raise ValueError(
                f"sample-index must be between 0 and {len(dataset) - 1}"
            )
        return args.sample_index, "explicit-index"

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    positive_scores = []
    negative_scores = []
    offset = 0
    model.eval()
    with torch.inference_mode():
        for inputs, targets, ice_masks in loader:
            probabilities = torch.sigmoid(
                model(inputs.to(device, non_blocking=True))
            ).cpu()
            for item in range(len(inputs)):
                index = offset + item
                target = targets[item, 0] > 0.5
                valid = ice_masks[item, 0] > 0
                positives = target & valid
                probability = probabilities[item, 0]
                if positives.any():
                    recall = float(
                        ((probability >= threshold) & positives).sum().item()
                        / positives.sum().item()
                    )
                    positive_scores.append((index, recall))
                else:
                    potential = (
                        float(probability[valid].max().item())
                        if valid.any()
                        else 0.0
                    )
                    negative_scores.append((index, potential))
            offset += len(inputs)

    rng = np.random.default_rng(args.seed)
    if args.selection == "lowest-recall-positive":
        chosen = min(positive_scores, key=lambda pair: pair[1])
    elif args.selection == "highest-recall-positive":
        chosen = max(positive_scores, key=lambda pair: pair[1])
    elif args.selection == "highest-potential-negative":
        chosen = max(negative_scores, key=lambda pair: pair[1])
    elif args.selection == "random-positive":
        chosen = positive_scores[int(rng.integers(len(positive_scores)))]
    else:
        chosen = negative_scores[int(rng.integers(len(negative_scores)))]
    return chosen[0], args.selection


def capture_dec1(model, input_patch, device):
    captured = {}

    def hook(_module, _inputs, output):
        captured["dec1"] = output.detach()

    handle = model.dec1.register_forward_hook(hook)
    model.eval()
    try:
        with torch.inference_mode():
            logits = model(input_patch.unsqueeze(0).to(device))
    finally:
        handle.remove()

    dec1 = captured["dec1"][0]
    weights = model.output.weight.detach()[0, :, 0, 0]
    contributions = dec1 * weights[:, None, None]
    bias = model.output.bias.detach()[0] if model.output.bias is not None else 0.0
    reconstructed = contributions.sum(dim=0) + bias
    maximum_error = float((reconstructed - logits[0, 0]).abs().max().item())
    return logits[0, 0], dec1, contributions, weights, maximum_error


def add_ice_boundary(ax, valid):
    if valid.any() and not valid.all():
        ax.contour(valid.astype(float), levels=[0.5], colors="cyan", linewidths=0.45)


def finite_quantile(array, quantile, fallback=1.0):
    values = np.asarray(array)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return fallback
    result = float(np.quantile(values, quantile))
    return result if result > 0 else fallback


def plot_overview(
    path,
    input_patch,
    target,
    valid,
    probability,
    logits,
    contributions,
    weights,
    input_indices,
    threshold,
    title,
):
    positive = contributions[weights > 0].sum(dim=0).cpu().numpy()
    negative = contributions[weights < 0].sum(dim=0).cpu().numpy()
    logit = logits.cpu().numpy()
    probability = probability.cpu().numpy()
    target = target.cpu().numpy() > 0.5
    valid = valid.cpu().numpy() > 0

    if 0 in input_indices:
        channel = input_indices.index(0)
        context = input_patch[channel].cpu().numpy()
        context_title = "Normalized thickness"
    else:
        context = valid.astype(float)
        context_title = "Valid ice"

    prediction = (probability >= threshold) & valid
    results = np.zeros(target.shape, dtype=np.uint8)
    results[prediction & target] = 1
    results[prediction & ~target] = 2
    results[~prediction & target & valid] = 3
    result_cmap = ListedColormap(["#e6e6e6", "#1b9e77", "#e66101", "#5e3c99"])

    evidence_limit = finite_quantile(
        np.abs(np.concatenate([positive.ravel(), negative.ravel()])), 0.99
    )
    logit_limit = finite_quantile(np.abs(logit), 0.99)
    panels = (
        (context, context_title, "gray", None, None),
        (np.ma.masked_where(~valid, target), "Ground truth", "gray", 0, 1),
        (
            np.ma.masked_where(~valid, probability),
            "Fracture probability",
            "magma",
            0,
            1,
        ),
        (
            np.ma.masked_where(~valid, results),
            "TP green / FP orange / FN purple",
            result_cmap,
            0,
            3,
        ),
        (positive, "Summed positive evidence", "Reds", 0, evidence_limit),
        (negative, "Summed negative evidence", "Blues_r", -evidence_limit, 0),
        (logit, "Reconstructed net logit", "coolwarm", -logit_limit, logit_limit),
    )

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for ax, panel in zip(axes.flat, panels):
        image, panel_title, cmap, vmin, vmax = panel
        shown = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        add_ice_boundary(ax, valid)
        ax.set_title(panel_title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(shown, ax=ax, fraction=0.046, pad=0.04)

    axes.flat[-1].axis("off")
    fig.suptitle(title)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_feature_grid(path, maps, valid, titles, figure_title, cmap, symmetric):
    arrays = maps.cpu().numpy()
    valid = valid.cpu().numpy() > 0
    count = arrays.shape[0]
    columns = min(4, count)
    rows = math.ceil(count / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.4 * columns, 3.1 * rows),
        squeeze=False,
        constrained_layout=True,
    )

    if symmetric:
        limit = finite_quantile(np.abs(arrays), 0.99)
        vmin, vmax = -limit, limit
    else:
        vmin, vmax = 0, finite_quantile(arrays, 0.99)

    for index, ax in enumerate(axes.flat):
        if index >= count:
            ax.axis("off")
            continue
        shown = ax.imshow(arrays[index], cmap=cmap, vmin=vmin, vmax=vmax)
        add_ice_boundary(ax, valid)
        ax.set_title(titles[index], fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(shown, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(figure_title)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    if args.samples_per_class <= 0 or args.batch_size <= 0:
        raise ValueError("samples-per-class and batch-size must be positive")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    checkpoint = load_checkpoint(args.checkpoint, device)
    saved_threshold = checkpoint.get("inference_threshold")
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else 0.5 if saved_threshold is None else float(saved_threshold)
    )
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    min_positive_pixels = (
        args.min_positive_pixels
        if args.min_positive_pixels is not None
        else int(checkpoint.get("evaluation_min_positive_pixels", 8))
    )

    dataset, input_indices = make_test_dataset(
        checkpoint,
        args.samples_per_class,
        args.seed,
        min_positive_pixels,
    )
    model = UNet(
        in_channels=int(checkpoint["input_channels"]),
        out_channels=1,
        base_channels=int(checkpoint["base_channels"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    sample_index, selection_label = select_sample(
        model, dataset, device, threshold, args
    )
    input_patch, target, ice_mask = dataset[sample_index]
    logits, dec1, contributions, weights, reconstruction_error = capture_dec1(
        model, input_patch, device
    )
    probability = torch.sigmoid(logits)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{selection_label}_{sample_index}"
    patch_row, patch_col = dataset.patch_coords[sample_index]
    title = (
        f"dec1 analysis: {selection_label}, sample {sample_index}, "
        f"raster origin ({patch_row}, {patch_col})"
    )

    plot_overview(
        output_dir / f"{stem}_overview.png",
        input_patch,
        target[0],
        ice_mask[0],
        probability,
        logits,
        contributions,
        weights,
        input_indices,
        threshold,
        title,
    )
    activation_titles = [
        f"Feature {index}: max={dec1[index].max().item():.3g}"
        for index in range(dec1.shape[0])
    ]
    plot_feature_grid(
        output_dir / f"{stem}_activations.png",
        dec1,
        ice_mask[0],
        activation_titles,
        f"Raw dec1 activations — {selection_label}, sample {sample_index}",
        "inferno",
        symmetric=False,
    )
    contribution_titles = [
        f"Feature {index}: w={weights[index].item():+.3g}"
        for index in range(contributions.shape[0])
    ]
    plot_feature_grid(
        output_dir / f"{stem}_contributions.png",
        contributions,
        ice_mask[0],
        contribution_titles,
        f"Weighted dec1 contributions — {selection_label}, sample {sample_index}",
        "coolwarm",
        symmetric=True,
    )

    input_names = [PREDICTOR_NAMES[index] for index in input_indices]
    print("Model inputs:", ", ".join(input_names))
    print("Selected sample:", sample_index, selection_label)
    print("Patch raster origin:", (patch_row, patch_col))
    print("Inference threshold:", threshold)
    print("Maximum logit reconstruction error:", reconstruction_error)
    print("Saved figures to:", output_dir.resolve())


if __name__ == "__main__":
    main()
