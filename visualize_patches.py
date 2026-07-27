import argparse
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="matplotlib-"))

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from image_patching import make_negative_coords, make_positive_coords
from load_data import load_data


def robust_image(arr):
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype="float32")

    lo, hi = np.nanpercentile(arr, [2, 98])
    if hi <= lo:
        return np.zeros_like(arr, dtype="float32")

    return np.clip((arr - lo) / (hi - lo), 0, 1)


def add_fracture_overlay(ax, base, target):
    rgb = np.dstack([base, base, base])
    fractures = target > 0
    rgb[fractures] = [1.0, 0.05, 0.05]
    ax.imshow(rgb)


def plot_patch_row(axes, X_patch, y_patch, label, coord):
    thickness = robust_image(X_patch[0])
    speed = robust_image(np.sqrt(X_patch[1] ** 2 + X_patch[2] ** 2))
    surface = robust_image(X_patch[3])
    missing = np.isnan(X_patch).any(axis=0)

    row, col = coord
    axes[0].set_ylabel(f"{label}\nr={row}, c={col}", fontsize=8)

    add_fracture_overlay(axes[0], speed, y_patch)
    axes[0].set_title("speed + fractures")

    add_fracture_overlay(axes[1], thickness, y_patch)
    axes[1].set_title("thickness + fractures")

    add_fracture_overlay(axes[2], surface, y_patch)
    axes[2].set_title("surface + fractures")

    axes[3].imshow(missing, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("any NaN")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])


def main():
    parser = argparse.ArgumentParser(
        description="Save a quick visual check of sampled fracture patches."
    )
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--n", type=int, default=4, help="Positive and negative patches each.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="patch_preview.png")
    args = parser.parse_args()

    X, y = load_data()
    with rasterio.open("mask_500m.tif") as src:
        ice_mask = src.read(1).astype("float32")

    rng = np.random.default_rng(args.seed)
    positive_coords = make_positive_coords(
        y, X, args.patch_size, args.n, rng=rng
    )
    negative_coords = make_negative_coords(
        y, X, ice_mask, args.patch_size, args.n, rng=rng
    )

    rows = [(coord, "positive") for coord in positive_coords]
    rows += [(coord, "negative") for coord in negative_coords]

    fig, axes = plt.subplots(
        len(rows),
        4,
        figsize=(10, max(2.2 * len(rows), 3)),
        constrained_layout=True,
    )
    if len(rows) == 1:
        axes = axes[None, :]

    for row_axes, (coord, label) in zip(axes, rows):
        row, col = coord
        p = args.patch_size
        X_patch = X[:, row:row + p, col:col + p]
        y_patch = y[row:row + p, col:col + p]
        plot_patch_row(row_axes, X_patch, y_patch, label, coord)

    fig.savefig(args.output, dpi=180)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
    
