import numpy as np

def make_positive_coords(y_full, X_full, patch_size, n_patches):
    C, H, W = X_full.shape
    frac_rows, frac_cols = np.where(y_full == 1)

    coords = []

    while len(coords) < n_patches:
        i = np.random.randint(len(frac_rows))

        frac_row = frac_rows[i]
        frac_col = frac_cols[i]

        row = frac_row - np.random.randint(0, patch_size)
        col = frac_col - np.random.randint(0, patch_size)

        row = np.clip(row, 0, H - patch_size)
        col = np.clip(col, 0, W - patch_size)

        X_patch = X_full[:, row:row+patch_size, col:col+patch_size]
        y_patch = y_full[row:row+patch_size, col:col+patch_size]

        if np.isnan(X_patch).mean() > 0.25:
            continue

        if y_patch.sum() == 0:
            continue

        coords.append((row, col))

    return coords

def make_negative_coords(y_full, X_full, ice_mask, patch_size, n_patches):
    C, H, W = X_full.shape

    coords = []

    while len(coords) < n_patches:
        row = np.random.randint(0, H - patch_size + 1)
        col = np.random.randint(0, W - patch_size + 1)

        X_patch = X_full[:, row:row+patch_size, col:col+patch_size]
        y_patch = y_full[row:row+patch_size, col:col+patch_size]
        ice_patch = ice_mask[row:row+patch_size, col:col+patch_size]

        if np.isnan(X_patch).mean() > 0.25:
            continue

        if y_patch.sum() > 0:
            continue

        if ice_patch.mean() < 0.25:
            continue

        coords.append((row, col))

    return coords