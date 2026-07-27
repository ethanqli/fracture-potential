import numpy as np
import rasterio

PREDICTOR_PATHS = [
    "thickness_500m.tif",
    "vx_nan_500m.tif",
    "vy_nan_500m.tif",
    "surface_500m.tif",
    "bed_500m.tif",
    "effective_strain_rate_500.tif",
    "effective_stress_500.tif",
    "mask_500m.tif",
    "velocity_mask_500.tif",
]
TARGET_PATH = "frac_map_500m_on_ice.tif"


def normalize_channel(arr, statistics_valid, output_valid=None):
    statistics_valid = statistics_valid & np.isfinite(arr)
    if not np.any(statistics_valid):
        raise ValueError("No valid training pixels available for normalization")

    mean = np.mean(arr[statistics_valid], dtype=np.float64)
    std = np.std(arr[statistics_valid], dtype=np.float64)
    if not np.isfinite(mean) or not np.isfinite(std):
        raise ValueError("Normalization statistics are not finite")

    normalized = np.zeros(arr.shape, dtype="float32")
    if output_valid is None:
        output_valid = np.isfinite(arr)
    else:
        output_valid = output_valid & np.isfinite(arr)
    normalized[output_valid] = ((arr[output_valid] - mean) / max(std, 1e-6)).astype(
        "float32"
    )

    return normalized, {"mean": float(mean), "std": float(std)}


def load_raw_data():
    """Load aligned rasters, clean masks, and return unnormalized predictors."""
    predictor_arrays = []
    reference_grid = None

    for path in PREDICTOR_PATHS:
        with rasterio.open(path) as src:
            grid = (src.shape, src.transform, src.crs)
            if reference_grid is None:
                reference_grid = grid
            elif grid != reference_grid:
                raise ValueError(f"{path} does not match the predictor grid")

            arr = src.read(1).astype("float32")
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
            predictor_arrays.append(arr)

    X = np.stack(predictor_arrays, axis=0)
    with rasterio.open(TARGET_PATH) as src:
        if (src.shape, src.transform, src.crs) != reference_grid:
            raise ValueError(f"{TARGET_PATH} does not match the predictor grid")
        y = src.read(1).astype("float32")
        if src.nodata is not None:
            y[y == src.nodata] = np.nan

    # BedMachine mask classes: 2 = grounded ice, 3 = floating ice.
    ice = np.isin(X[7], [2, 3])
    velocity_valid = (
        ice
        & np.isfinite(X[8])
        & (X[8] > 0)
        & np.isfinite(X[1])
        & np.isfinite(X[2])
    )

    X[7] = ice.astype("float32")
    X[8] = velocity_valid.astype("float32")
    y = (np.isfinite(y) & (y > 0) & ice).astype("float32")

    return X, y


def normalize_predictors(X_raw, train_region, copy=True):
    """Normalize continuous channels using statistics from train_region only."""
    if train_region.shape != X_raw.shape[1:]:
        raise ValueError(
            f"train_region shape {train_region.shape} must match {X_raw.shape[1:]}"
        )

    X = X_raw.copy() if copy else X_raw
    ice = X[7] > 0

    velocity_valid = ice & (X[8] > 0) & np.isfinite(X[1]) & np.isfinite(X[2])
    valid_masks = {
        0: ice & np.isfinite(X[0]),
        1: velocity_valid,
        2: velocity_valid,
        3: ice & np.isfinite(X[3]),
        4: ice & np.isfinite(X[4]),
        5: velocity_valid & np.isfinite(X[5]),
        6: velocity_valid & np.isfinite(X[6]),
    }

    stats = {}
    for channel, valid in valid_masks.items():
        X[channel], stats[channel] = normalize_channel(
            X[channel], valid & train_region, output_valid=valid
        )

    if not np.isfinite(X).all():
        raise ValueError("X still contains NaN or infinite values")

    return X, stats


def load_data(train_region=None, return_stats=False):
    """Compatibility loader; pass train_region to prevent normalization leakage."""
    X_raw, y = load_raw_data()
    if train_region is None:
        train_region = np.ones(y.shape, dtype=bool)

    X, stats = normalize_predictors(X_raw, train_region)

    if not np.isfinite(y).all():
        raise ValueError("y still contains NaN or infinite values")

    if return_stats:
        return X, y, stats
    return X, y
