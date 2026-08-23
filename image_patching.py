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


def _bin_index(value, upper_bounds):
    return int(np.searchsorted(upper_bounds, value, side="right"))


def patch_positive_count(y_full, ice_mask, row, col, patch_size):
    region = np.s_[row:row + patch_size, col:col + patch_size]
    return int(((y_full[region] > 0) & (ice_mask[region] > 0)).sum())


def make_positive_coords(
    y_full,
    X_full,
    patch_size,
    n_patches,
    max_attempts_per_pixel=20,
    max_nan_fraction=0.25,
    min_positive_pixels=8,
    rng=None,
    allowed_mask=None,
    positive_pixel_pool=None,
    density_upper_bounds=None,
):
    _, H, W = _validate_inputs(y_full, X_full, patch_size, n_patches)
    if max_attempts_per_pixel <= 0:
        raise ValueError("max_attempts_per_pixel must be positive")

    if allowed_mask is not None and allowed_mask.shape != (H, W):
        raise ValueError(
            f"allowed_mask shape {allowed_mask.shape} must match {(H, W)}"
        )

    if positive_pixel_pool is None:
        positive_pixels = y_full > 0
        if allowed_mask is not None:
            positive_pixels &= allowed_mask
        frac_rows, frac_cols = np.where(positive_pixels)
    else:
        frac_rows, frac_cols = positive_pixel_pool
        if len(frac_rows) != len(frac_cols):
            raise ValueError("positive_pixel_pool row and column lengths differ")

    if len(frac_rows) == 0:
        raise ValueError("y_full does not contain any positive fracture pixels")

    coords = []
    seen = set()
    rng = rng or np.random.default_rng()
    density_upper_bounds = (
        tuple(density_upper_bounds) if density_upper_bounds is not None else None
    )
    if density_upper_bounds is not None:
        if tuple(sorted(density_upper_bounds)) != density_upper_bounds:
            raise ValueError("density_upper_bounds must be sorted")
        n_bins = len(density_upper_bounds) + 1
        base, remainder = divmod(n_patches, n_bins)
        quotas = [base + (i < remainder) for i in range(n_bins)]
        counts = [0] * n_bins
        # Repeated draws allow sparse fracture pixels to yield several distinct
        # placements, while `seen` still prevents duplicate patches.
        fracture_indices = rng.integers(
            0, len(frac_rows), size=max(len(frac_rows), n_patches * 500)
        )
    else:
        quotas = counts = None
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

            valid_positive_pixels = (
                (y_patch > 0)
                & (X_patch[7] > 0)
            )

            positive_count = int(valid_positive_pixels.sum())
            if positive_count < min_positive_pixels:
                continue

            if density_upper_bounds is not None:
                density_bin = _bin_index(positive_count, density_upper_bounds)
                if counts[density_bin] >= quotas[density_bin]:
                    continue

            if _invalid_velocity_fraction(X_patch) > max_nan_fraction:
                continue

            if np.nansum(y_patch > 0) == 0:
                continue

            seen.add((row, col))
            coords.append((row, col))
            if density_upper_bounds is not None:
                counts[density_bin] += 1
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
            "patch_size, relaxing max_nan_fraction, or changing the density "
            f"bins. Density-bin counts: {counts}."
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
    boundary_distance=None,
    boundary_upper_bounds=(8, 32, 128),
    target_boundary_counts=None,
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
    attempts_per_patch = 500 if target_boundary_counts is not None else 200
    max_attempts = max_attempts or max(1000, n_patches * attempts_per_patch)
    rng = rng or np.random.default_rng()
    if target_boundary_counts is not None:
        if boundary_distance is None or boundary_distance.shape != (H, W):
            raise ValueError(
                "A full-size boundary_distance raster is required for matching"
            )
        expected_bins = len(boundary_upper_bounds) + 1
        if len(target_boundary_counts) != expected_bins:
            raise ValueError("target_boundary_counts has the wrong number of bins")
        if sum(target_boundary_counts) != n_patches:
            raise ValueError("target_boundary_counts must sum to n_patches")
        boundary_counts = [0] * expected_bins
    else:
        boundary_counts = None

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

        if boundary_counts is not None:
            center_row = row + patch_size // 2
            center_col = col + patch_size // 2
            boundary_bin = _bin_index(
                boundary_distance[center_row, center_col], boundary_upper_bounds
            )
            if boundary_counts[boundary_bin] >= target_boundary_counts[boundary_bin]:
                continue

        seen.add((row, col))
        coords.append((row, col))
        if boundary_counts is not None:
            boundary_counts[boundary_bin] += 1

    if len(coords) < n_patches:
        raise RuntimeError(
            "Could not sample enough negative patches: "
            f"requested {n_patches}, found {len(coords)} after {attempts} attempts. "
            "Try reducing n_patches, increasing max_attempts, using a smaller "
            "patch_size, relaxing max_nan_fraction, lowering min_ice_fraction, "
            f"or widening boundary bins. Boundary-bin counts: {boundary_counts}."
        )

    return coords
