import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="matplotlib-"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap


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
    n=6,
):
    """Save input, target, probability, prediction, and error test panels."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if n <= 0:
        raise ValueError("n must be positive")
    if len(dataset) == 0:
        raise ValueError("Cannot visualize an empty dataset")

    n = min(n, len(dataset))
    # The dataset coordinates are already seeded and shuffled; evenly spaced
    # indices avoid showing only one portion of that ordering.
    indices = np.linspace(0, len(dataset) - 1, n, dtype=int)
    error_cmap = ListedColormap(["#e6e6e6", "#e66101", "#5e3c99"])

    was_training = model.training
    model.eval()
    fig, axes = plt.subplots(n, 5, figsize=(14, 2.7 * n), squeeze=False)

    with torch.no_grad():
        for row, index in enumerate(indices):
            X_patch, target_tensor, ice_tensor = dataset[int(index)]
            logits = model(X_patch.unsqueeze(0).to(device))
            probability = torch.sigmoid(logits)[0, 0].cpu().numpy()

            target = target_tensor[0].numpy() > 0.5
            valid = ice_tensor[0].numpy() > 0
            prediction = (probability >= threshold) & valid
            probability = np.where(valid, probability, np.nan)

            # Channel zero is normalized ice thickness.
            thickness = np.ma.masked_where(~valid, X_patch[0].numpy())
            errors = np.zeros(target.shape, dtype=np.uint8)
            errors[prediction & ~target] = 1
            errors[~prediction & target & valid] = 2
            errors = np.ma.masked_where(~valid, errors)

            panels = (
                (thickness, "Thickness", "gray", None, None),
                (np.ma.masked_where(~valid, target), "Ground truth", "gray", 0, 1),
                (probability, "Fracture probability", "magma", 0, 1),
                (np.ma.masked_where(~valid, prediction), f"Prediction >= {threshold:.2f}", "gray", 0, 1),
                (errors, "Errors: FP=orange, FN=purple", error_cmap, 0, 2),
            )

            for column, (image, title, cmap, vmin, vmax) in enumerate(panels):
                ax = axes[row, column]
                shown = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
                if row == 0:
                    ax.set_title(title)
                ax.set_xticks([])
                ax.set_yticks([])
                if column == 0:
                    ax.set_ylabel(f"Test sample {int(index)}")
                if column == 2:
                    fig.colorbar(shown, ax=ax, fraction=0.046, pad=0.04)

    model.train(was_training)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved test predictions to {output}")
