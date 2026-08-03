import numpy as np
import rasterio
import torch


def _start_positions(length, patch_size, stride):
    if length < patch_size:
        raise ValueError(
            f"Raster dimension {length} is smaller than patch size {patch_size}"
        )
    positions = list(range(0, length - patch_size + 1, stride))
    final = length - patch_size
    if positions[-1] != final:
        positions.append(final)
    return positions


def _blend_window(patch_size):
    """Return a positive 2-D Hann window for seam-resistant blending."""
    window = np.hanning(patch_size).astype("float32")
    window = np.maximum(window, 1e-3)
    return np.outer(window, window).astype("float32")


def predict_full_raster(
    model,
    X,
    device,
    patch_size,
    batch_size=32,
    stride=None,
):
    """Predict a raster with overlapping patches and blend their probabilities."""
    if patch_size <= 0 or batch_size <= 0:
        raise ValueError("patch_size and batch_size must be positive")
    stride = stride or patch_size // 2
    if stride <= 0 or stride > patch_size:
        raise ValueError("stride must be between 1 and patch_size")

    _, height, width = X.shape
    rows = _start_positions(height, patch_size, stride)
    cols = _start_positions(width, patch_size, stride)
    total = len(rows) * len(cols)
    blend = _blend_window(patch_size)
    probability_sum = np.zeros((height, width), dtype="float32")
    weight_sum = np.zeros((height, width), dtype="float32")

    model.eval()
    patches = []
    locations = []
    completed = 0

    def process_batch():
        nonlocal completed
        if not patches:
            return
        inputs = torch.from_numpy(np.stack(patches)).to(
            device, non_blocking=True
        )
        with torch.inference_mode():
            probabilities = torch.sigmoid(model(inputs))[:, 0].cpu().numpy()
        for probability, (row, col) in zip(probabilities, locations):
            region = np.s_[row:row + patch_size, col:col + patch_size]
            probability_sum[region] += probability * blend
            weight_sum[region] += blend
        completed += len(patches)
        print(f"\rFull-raster inference: {completed}/{total} patches", end="", flush=True)
        patches.clear()
        locations.clear()

    for row in rows:
        for col in cols:
            patches.append(X[:, row:row + patch_size, col:col + patch_size])
            locations.append((row, col))
            if len(patches) == batch_size:
                process_batch()
    process_batch()
    print()

    return probability_sum / np.maximum(weight_sum, 1e-8)


def save_prediction_geotiffs(
    potential,
    ice_mask,
    reference_path,
    potential_path,
    threshold,
    binary_path=None,
):
    """Save continuous potential and optionally a thresholded prediction."""
    valid = ice_mask > 0
    with rasterio.open(reference_path) as reference:
        profile = reference.profile.copy()

    potential_nodata = -9999.0
    potential_output = np.where(valid, potential, potential_nodata).astype("float32")
    profile.update(
        count=1,
        dtype="float32",
        nodata=potential_nodata,
        compress="deflate",
        tiled=True,
        BIGTIFF="IF_SAFER",
    )
    with rasterio.open(potential_path, "w", **profile) as destination:
        destination.write(potential_output, 1)
    print(f"Saved continuous fracture potential to {potential_path}")

    if binary_path:
        binary_nodata = 255
        binary = np.full(potential.shape, binary_nodata, dtype="uint8")
        binary[valid] = (potential[valid] >= threshold).astype("uint8")
        binary_profile = profile.copy()
        binary_profile.update(dtype="uint8", nodata=binary_nodata)
        with rasterio.open(binary_path, "w", **binary_profile) as destination:
            destination.write(binary, 1)
        print(
            f"Saved thresholded prediction (threshold={threshold:.2f}) "
            f"to {binary_path}"
        )
