import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="matplotlib-"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader

from full_raster_inference import predict_full_raster


def plot_auc_curves(curves, output="auc_curves.png"):
    """Save precision-recall and ROC curves from pixel-level scores."""
    required = {
        "precision", "recall", "fpr", "tpr", "pr_auc", "roc_auc",
        "prevalence",
    }
    missing = required.difference(curves)
    if missing:
        raise ValueError(f"Curve data is missing: {', '.join(sorted(missing))}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].step(
        curves["recall"], curves["precision"], where="post",
        label=f"Model (AP = {curves['pr_auc']:.3f})",
    )
    axes[0].axhline(
        curves["prevalence"], color="gray", linestyle="--",
        label=f"Random baseline = {curves['prevalence']:.3f}",
    )
    axes[0].set(
        title="Precision–recall curve", xlabel="Recall", ylabel="Precision",
        xlim=(0, 1), ylim=(0, 1),
    )

    axes[1].plot(
        curves["fpr"], curves["tpr"],
        label=f"Model (AUC = {curves['roc_auc']:.3f})",
    )
    axes[1].plot(
        [0, 1], [0, 1], color="gray", linestyle="--",
        label="Random baseline",
    )
    axes[1].set(
        title="ROC curve", xlabel="False-positive rate",
        ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1),
    )
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Sampled test valid-ice pixels")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved AUC curves to {output}")


def plot_training_history(history, output="training_curves.png"):
    """Save train/validation loss and Dice curves from a training run."""
    required = {"train_loss", "val_loss", "train_dice", "val_dice"}
    missing = required.difference(history)
    if missing:
        raise ValueError(f"History is missing: {', '.join(sorted(missing))}")

    lengths = {len(history[name]) for name in required}
    if len(lengths) != 1 or not lengths or lengths == {0}:
        raise ValueError("All history series must have the same non-zero length")

    epochs = np.arange(1, lengths.pop() + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.25))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set(
        title="Loss curves",
        xlabel="Epoch",
        ylabel="Combined BCE + Dice loss",
    )

    axes[1].plot(epochs, history["train_dice"], label="Train")
    axes[1].plot(epochs, history["val_dice"], label="Validation")
    axes[1].set(
        title="Dice curves",
        xlabel="Epoch",
        ylabel="Dice score",
        ylim=(0, 1),
    )

    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xticks(epochs if len(epochs) <= 15 else ax.get_xticks())

    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training curves to {output}")


def _max_pool_for_display(array, max_display_size):
    """Reduce a binary raster while preserving isolated positive pixels."""
    height, width = array.shape
    factor = max(1, int(np.ceil(max(height, width) / max_display_size)))
    if factor == 1:
        return array

    padded_height = int(np.ceil(height / factor) * factor)
    padded_width = int(np.ceil(width / factor) * factor)
    padded = np.zeros((padded_height, padded_width), dtype=bool)
    padded[:height, :width] = array
    return padded.reshape(
        padded_height // factor, factor, padded_width // factor, factor
    ).max(axis=(1, 3))


def visualize_test_region_maps(
    model,
    X,
    target,
    test_region,
    device,
    patch_size,
    threshold,
    output="test_region_maps.png",
    batch_size=32,
    stride=None,
    max_display_size=1200,
):
    """Save four overview maps covering the complete held-out test region."""
    if X.ndim != 3 or target.shape != X.shape[1:]:
        raise ValueError("X and target have incompatible shapes")
    if test_region.shape != target.shape or not test_region.any():
        raise ValueError("test_region must be a non-empty mask matching target")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if max_display_size <= 0:
        raise ValueError("max_display_size must be positive")

    rows, cols = np.where(test_region)
    row_start, row_stop = int(rows.min()), int(rows.max()) + 1
    col_start, col_stop = int(cols.min()), int(cols.max()) + 1

    # Include surrounding context so predictions at the test-region boundary do
    # not suffer from artificial crop edges. Only the held-out area is plotted.
    context = patch_size
    input_row_start = max(0, row_start - context)
    input_row_stop = min(X.shape[1], row_stop + context)
    input_col_start = max(0, col_start - context)
    input_col_stop = min(X.shape[2], col_stop + context)
    X_crop = X[
        :,
        input_row_start:input_row_stop,
        input_col_start:input_col_stop,
    ]
    probability_crop = predict_full_raster(
        model,
        X_crop,
        device,
        patch_size=patch_size,
        batch_size=batch_size,
        stride=stride,
    )
    probability = probability_crop[
        row_start - input_row_start:row_stop - input_row_start,
        col_start - input_col_start:col_stop - input_col_start,
    ]

    valid = test_region[row_start:row_stop, col_start:col_stop]
    valid &= X[7, row_start:row_stop, col_start:col_stop] > 0
    truth = (target[row_start:row_stop, col_start:col_stop] > 0) & valid
    prediction = (probability >= threshold) & valid
    maps = (
        (truth, "Ground truth", "Reds"),
        (prediction, f"Prediction (p >= {threshold:.2f})", "Blues"),
        (prediction & ~truth, "False positives", "Oranges"),
        (~prediction & truth & valid, "False negatives", "Purples"),
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    extent = (col_start, col_stop, row_stop, row_start)
    for ax, (values, title, cmap) in zip(axes.flat, maps):
        display = _max_pool_for_display(values, max_display_size)
        ax.imshow(display, cmap=cmap, vmin=0, vmax=1, extent=extent, aspect="auto")
        ax.set_title(f"{title} ({int(values.sum()):,} pixels)")
        ax.set_xlabel("Raster column")
        ax.set_ylabel("Raster row")

    fig.suptitle("Held-out test-region overview")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved test-region overview to {output}")


def visualize_test_predictions(
    model,
    dataset,
    device,
    threshold,
    output="test_predictions.png",
    samples_per_group=2,
    batch_size=32,
    seed=42,
):
    """Visualize random and performance-ranked test-patch groups."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if samples_per_group <= 0 or batch_size <= 0:
        raise ValueError("samples_per_group and batch_size must be positive")
    if len(dataset) == 0:
        raise ValueError("Cannot visualize an empty dataset")

    result_cmap = ListedColormap(
        ["#e6e6e6", "#1b9e77", "#e66101", "#5e3c99"]
    )

    was_training = model.training
    model.eval()

    # Score every test patch first. Fracture-free potential is ranked by the
    # maximum valid-pixel score; positive patches are ranked by fracture recall.
    positive_scores = []
    negative_scores = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    offset = 0
    with torch.inference_mode():
        for X_batch, target_batch, ice_batch in loader:
            probabilities = torch.sigmoid(model(X_batch.to(device))).cpu()
            for item in range(len(X_batch)):
                index = offset + item
                target = target_batch[item, 0] > 0.5
                valid = ice_batch[item, 0] > 0
                probability = probabilities[item, 0]
                positives = target & valid
                if positives.any():
                    recall = (
                        ((probability >= threshold) & positives).sum().item()
                        / positives.sum().item()
                    )
                    positive_scores.append((index, recall))
                else:
                    potential = probability[valid].max().item() if valid.any() else 0.0
                    negative_scores.append((index, potential))
            offset += len(X_batch)

    if not positive_scores or not negative_scores:
        raise ValueError(
            "Visualization requires both fracture-positive and fracture-free patches"
        )

    rng = np.random.default_rng(seed)

    def random_indices(scores):
        count = min(samples_per_group, len(scores))
        chosen = rng.choice(len(scores), size=count, replace=False)
        return [scores[int(i)][0] for i in chosen]

    count_positive = min(samples_per_group, len(positive_scores))
    count_negative = min(samples_per_group, len(negative_scores))
    groups = [
        ("Random positive", random_indices(positive_scores)),
        ("Random fracture-free", random_indices(negative_scores)),
        (
            "Highest-potential fracture-free",
            [index for index, _ in sorted(
                negative_scores, key=lambda item: item[1], reverse=True
            )[:count_negative]],
        ),
        (
            "Lowest-recall positive",
            [index for index, _ in sorted(
                positive_scores, key=lambda item: item[1]
            )[:count_positive]],
        ),
        (
            "Highest-recall positive",
            [index for index, _ in sorted(
                positive_scores, key=lambda item: item[1], reverse=True
            )[:count_positive]],
        ),
    ]
    selected = [
        (group, index) for group, indices in groups for index in indices
    ]
    n_rows = len(selected)
    fig, axes = plt.subplots(
        n_rows, 5, figsize=(14, 2.7 * n_rows), squeeze=False
    )

    with torch.inference_mode():
        for row, (group, index) in enumerate(selected):
            X_patch, target_tensor, ice_tensor = dataset[int(index)]
            logits = model(X_patch.unsqueeze(0).to(device))
            probability = torch.sigmoid(logits)[0, 0].cpu().numpy()

            target = target_tensor[0].numpy() > 0.5
            valid = ice_tensor[0].numpy() > 0
            prediction = (probability >= threshold) & valid
            probability = np.where(valid, probability, np.nan)

            # Channel zero is normalized ice thickness.
            thickness = np.ma.masked_where(~valid, X_patch[0].numpy())
            results = np.zeros(target.shape, dtype=np.uint8)
            results[prediction & target] = 1
            results[prediction & ~target] = 2
            results[~prediction & target & valid] = 3
            results = np.ma.masked_where(~valid, results)

            panels = (
                (thickness, "Thickness", "gray", None, None),
                (np.ma.masked_where(~valid, target), "Ground truth", "gray", 0, 1),
                (probability, "Fracture probability", "magma", 0, 1),
                (np.ma.masked_where(~valid, prediction), f"Prediction >= {threshold:.2f}", "gray", 0, 1),
                (
                    results,
                    "Results: TP=green, FP=orange, FN=purple",
                    result_cmap,
                    0,
                    3,
                ),
            )

            for column, (image, title, cmap, vmin, vmax) in enumerate(panels):
                ax = axes[row, column]
                shown = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
                if row == 0:
                    ax.set_title(title)
                ax.set_xticks([])
                ax.set_yticks([])
                if column == 0:
                    ax.set_ylabel(f"{group}\nsample {int(index)}", fontsize=8)
                if column == 2:
                    fig.colorbar(shown, ax=ax, fraction=0.046, pad=0.04)

    model.train(was_training)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved test predictions to {output}")
