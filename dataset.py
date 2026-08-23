import numpy as np
import torch
from torch.utils.data import Dataset

from image_patching import (
    make_negative_coords,
    make_positive_coords,
    patch_positive_count,
)

class FracturePatchDataset(Dataset):
    def __init__(
        self,
        X_full,
        y_full,
        ice_mask,
        patch_size,
        positive_patches=5000,
        negative_patches=5000,
        max_attempts_per_positive_pixel=20,
        seed=None,
        allowed_mask=None,
        augment=False,
        normalization_stats=None,
        positive_density_upper_bounds=None,
        boundary_distance=None,
        boundary_upper_bounds=(8, 32, 128),
        min_positive_pixels=8,
    ):
        self.X_full = X_full
        self.y_full = y_full
        self.ice_mask = ice_mask
        self.patch_size = patch_size
        self.positive_patches = positive_patches
        self.negative_patches = negative_patches
        self.max_attempts_per_positive_pixel = max_attempts_per_positive_pixel
        self.allowed_mask = allowed_mask
        self.augment = augment
        self.normalization_stats = normalization_stats
        self.positive_density_upper_bounds = positive_density_upper_bounds
        self.boundary_distance = boundary_distance
        self.boundary_upper_bounds = boundary_upper_bounds
        self.min_positive_pixels = min_positive_pixels
        if augment and (
            normalization_stats is None
            or 1 not in normalization_stats
            or 2 not in normalization_stats
        ):
            raise ValueError(
                "Velocity normalization statistics are required for augmentation"
            )

        # Finding every eligible fracture pixel is expensive on a large raster,
        # so cache this pool and reuse it when patch locations are resampled.
        positive_pixels = y_full > 0
        if allowed_mask is not None:
            positive_pixels &= allowed_mask
        self.positive_pixel_pool = np.where(positive_pixels)
        self.resample(seed)

    def resample(self, seed=None):
        """Generate a new balanced set of patch coordinates."""
        rng = np.random.default_rng(seed)
        positive_coords = make_positive_coords(
            self.y_full,
            self.X_full,
            self.patch_size,
            self.positive_patches,
            max_attempts_per_pixel=self.max_attempts_per_positive_pixel,
            rng=rng,
            allowed_mask=self.allowed_mask,
            positive_pixel_pool=self.positive_pixel_pool,
            density_upper_bounds=self.positive_density_upper_bounds,
            min_positive_pixels=self.min_positive_pixels,
        )
        target_boundary_counts = None
        if self.boundary_distance is not None:
            target_boundary_counts = [0] * (len(self.boundary_upper_bounds) + 1)
            for row, col in positive_coords:
                center_row = row + self.patch_size // 2
                center_col = col + self.patch_size // 2
                boundary_bin = int(np.searchsorted(
                    self.boundary_upper_bounds,
                    self.boundary_distance[center_row, center_col],
                    side="right",
                ))
                target_boundary_counts[boundary_bin] += 1
        negative_coords = make_negative_coords(
            self.y_full,
            self.X_full,
            self.ice_mask,
            self.patch_size,
            self.negative_patches,
            rng=rng,
            allowed_mask=self.allowed_mask,
            boundary_distance=self.boundary_distance,
            boundary_upper_bounds=self.boundary_upper_bounds,
            target_boundary_counts=target_boundary_counts,
        )
        coords = positive_coords + negative_coords
        rng.shuffle(coords)
        self.patch_coords = coords
        if self.positive_density_upper_bounds is not None:
            density_counts = [0] * (len(self.positive_density_upper_bounds) + 1)
            for row, col in positive_coords:
                count = patch_positive_count(
                    self.y_full, self.ice_mask, row, col, self.patch_size
                )
                density_counts[int(np.searchsorted(
                    self.positive_density_upper_bounds, count, side="right"
                ))] += 1
            self.positive_density_counts = density_counts
        self.boundary_bin_counts = target_boundary_counts

    def __len__(self):
        return len(self.patch_coords)

    def __getitem__(self, idx):
        row, col = self.patch_coords[idx]
        p = self.patch_size

        X_patch = self.X_full[:, row:row+p, col:col+p]
        y_patch = self.y_full[row:row+p, col:col+p]
        ice_patch = self.ice_mask[row:row+p, col:col+p]

        X_patch = torch.from_numpy(X_patch).float()
        y_patch = torch.from_numpy(y_patch).float().unsqueeze(0)
        ice_patch = torch.from_numpy(ice_patch > 0).float().unsqueeze(0)

        if self.augment:
            # Apply the same random dihedral transform to predictors, target,
            # and validity mask. Velocity components are converted back to raw
            # units so the vector itself can be transformed consistently.
            vx_stats = self.normalization_stats[1]
            vy_stats = self.normalization_stats[2]
            vx = X_patch[1] * vx_stats["std"] + vx_stats["mean"]
            vy = X_patch[2] * vy_stats["std"] + vy_stats["mean"]

            k = int(torch.randint(0, 4, ()).item())
            if k:
                X_patch = torch.rot90(X_patch, k, dims=(-2, -1))
                y_patch = torch.rot90(y_patch, k, dims=(-2, -1))
                ice_patch = torch.rot90(ice_patch, k, dims=(-2, -1))
                vx = torch.rot90(vx, k, dims=(-2, -1))
                vy = torch.rot90(vy, k, dims=(-2, -1))
                if k == 1:
                    vx, vy = -vy, vx
                elif k == 2:
                    vx, vy = -vx, -vy
                else:
                    vx, vy = vy, -vx
            if bool(torch.randint(0, 2, ()).item()):
                X_patch = torch.flip(X_patch, dims=(-1,))
                y_patch = torch.flip(y_patch, dims=(-1,))
                ice_patch = torch.flip(ice_patch, dims=(-1,))
                vx = -torch.flip(vx, dims=(-1,))
                vy = torch.flip(vy, dims=(-1,))

            # rot90/flip can return views into the full raster. Clone before
            # overwriting the velocity channels to leave the source unchanged.
            X_patch = X_patch.clone()
            X_patch[1] = (vx - vx_stats["mean"]) / max(vx_stats["std"], 1e-6)
            X_patch[2] = (vy - vy_stats["mean"]) / max(vy_stats["std"], 1e-6)
            velocity_valid = X_patch[8] > 0
            X_patch[1] = torch.where(velocity_valid, X_patch[1], 0.0)
            X_patch[2] = torch.where(velocity_valid, X_patch[2], 0.0)

        return X_patch, y_patch, ice_patch
