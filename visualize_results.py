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
