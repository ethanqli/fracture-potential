import numpy as np


def make_spatial_splits(height, width, patch_size):
    """Return spatial train/validation/test masks separated by buffer zones."""
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")

    train_end = int(width * 0.70)
    val_end = int(width * 0.85)
    gap = patch_size

    if train_end <= gap or val_end - train_end <= 2 * gap or width - val_end <= gap:
        raise ValueError(
            "Raster is too narrow for 70/15/15 splits with the requested buffer"
        )

    train_mask = np.zeros((height, width), dtype=bool)
    val_mask = np.zeros((height, width), dtype=bool)
    test_mask = np.zeros((height, width), dtype=bool)

    train_mask[:, : train_end - gap] = True
    val_mask[:, train_end + gap : val_end - gap] = True
    test_mask[:, val_end + gap :] = True

    return train_mask, val_mask, test_mask
