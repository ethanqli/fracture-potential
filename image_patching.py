import numpy as np


def _validate_inputs(y_full, X_full, patch_size, n_patches):
    C, H, W = X_full.shape
    if y_full.shape != (H, W):
        raise ValueError(
            f"y_full shape {y_full.shape} must match X_full spatial shape {(H, W)}"
        )
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if patch_size > H or patch_size > W:
        raise ValueError(
            f"patch_size {patch_size} cannot exceed image shape {(H, W)}"
        )
    if n_patches < 0:
        raise ValueError("n_patches must be non-negative")
    return C, H, W


def _invalid_velocity_fraction(X_patch, velocity_mask_channel=8):
    velocity_mask = X_patch[velocity_mask_channel]
    return np.mean(velocity_mask <= 0)


def _patch_is_allowed(allowed_mask, row, col, patch_size):
    if allowed_mask is None:
        return True
    patch = allowed_mask[row:row+patch_size, col:col+patch_size]
    return patch.shape == (patch_size, patch_size) and patch.all()


def make_positive_coords(
    y_full,
    X_full,
    patch_size,
    n_patches,
    max_attempts_per_pixel=20,
    max_nan_fraction=0.25,
    rng=None,
    allowed_mask=None,
):
    _, H, W = _validate_inputs(y_full, X_full, patch_size, n_patches)
    if max_attempts_per_pixel <= 0:
        raise ValueError("max_attempts_per_pixel must be positive")

    if allowed_mask is not None and allowed_mask.shape != (H, W):
        raise ValueError(
            f"allowed_mask shape {allowed_mask.shape} must match {(H, W)}"
        )

    positive_pixels = y_full > 0
    if allowed_mask is not None:
        positive_pixels &= allowed_mask
    frac_rows, frac_cols = np.where(positive_pixels)

    if len(frac_rows) == 0:
        raise ValueError("y_full does not contain any positive fracture pixels")

    coords = []
    seen = set()
    rng = rng or np.random.default_rng()
    fracture_indices = rng.permutation(len(frac_rows))

    for i in fracture_indices:
        frac_row = frac_rows[i]
        frac_col = frac_cols[i]

        for _ in range(max_attempts_per_pixel):
            row = frac_row - rng.integers(0, patch_size)
            col = frac_col - rng.integers(0, patch_size)

            row = int(np.clip(row, 0, H - patch_size))
            col = int(np.clip(col, 0, W - patch_size))

            if (row, col) in seen:
                continue

            if not _patch_is_allowed(allowed_mask, row, col, patch_size):
                continue

            X_patch = X_full[:, row:row+patch_size, col:col+patch_size]
            y_patch = y_full[row:row+patch_size, col:col+patch_size]

            if _invalid_velocity_fraction(X_patch) > max_nan_fraction:
                continue

            if np.nansum(y_patch > 0) == 0:
                continue

            seen.add((row, col))
            coords.append((row, col))
            break

        if len(coords) == n_patches:
            break

    if len(coords) < n_patches:
        raise RuntimeError(
            "Could not sample enough positive patches: "
            f"requested {n_patches}, found {len(coords)} after trying "
            f"{len(fracture_indices)} fracture pixels with up to "
            f"{max_attempts_per_pixel} patch placements each. Try reducing "
            "n_patches, increasing max_attempts_per_pixel, using a smaller "
            "patch_size, or relaxing max_nan_fraction."
        )

    return coords


def make_negative_coords(
    y_full,
    X_full,
    ice_mask,
    patch_size,
    n_patches,
    max_attempts=None,
    max_nan_fraction=0.25,
    min_ice_fraction=0.25,
    rng=None,
    allowed_mask=None,
):
    _, H, W = _validate_inputs(y_full, X_full, patch_size, n_patches)
    if ice_mask.shape != (H, W):
        raise ValueError(
            f"ice_mask shape {ice_mask.shape} must match X_full spatial shape {(H, W)}"
        )
    if allowed_mask is not None and allowed_mask.shape != (H, W):
        raise ValueError(
            f"allowed_mask shape {allowed_mask.shape} must match {(H, W)}"
        )

    coords = []
    seen = set()
    attempts = 0
    max_attempts = max_attempts or max(1000, n_patches * 200)
    rng = rng or np.random.default_rng()

    while len(coords) < n_patches and attempts < max_attempts:
        attempts += 1
        row = int(rng.integers(0, H - patch_size + 1))
        col = int(rng.integers(0, W - patch_size + 1))

        if (row, col) in seen:
            continue

        if not _patch_is_allowed(allowed_mask, row, col, patch_size):
            continue

        X_patch = X_full[:, row:row+patch_size, col:col+patch_size]
        y_patch = y_full[row:row+patch_size, col:col+patch_size]
        ice_patch = ice_mask[row:row+patch_size, col:col+patch_size]

        if _invalid_velocity_fraction(X_patch) > max_nan_fraction:
            continue

        if np.nansum(y_patch > 0) > 0:
            continue

        if np.nanmean(ice_patch > 0) < min_ice_fraction:
            continue

        seen.add((row, col))
        coords.append((row, col))

    if len(coords) < n_patches:
        raise RuntimeError(
            "Could not sample enough negative patches: "
            f"requested {n_patches}, found {len(coords)} after {attempts} attempts. "
            "Try reducing n_patches, increasing max_attempts, using a smaller "
            "patch_size, relaxing max_nan_fraction, or lowering min_ice_fraction."
        )

    return coords
