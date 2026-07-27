import numpy as np
import torch
from torch.utils.data import Dataset

from image_patching import make_positive_coords, make_negative_coords

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
    ):
        self.X_full = X_full
        self.y_full = y_full
        self.ice_mask = ice_mask
        rng = np.random.default_rng(seed)
        positive_coords = make_positive_coords(
            y_full,
            X_full,
            patch_size,
            positive_patches,
            max_attempts_per_pixel=max_attempts_per_positive_pixel,
            rng=rng,
            allowed_mask=allowed_mask,
        )
        negative_coords = make_negative_coords(
            y_full,
            X_full,
            ice_mask,
            patch_size,
            negative_patches,
            rng=rng,
            allowed_mask=allowed_mask,
        )
        coords = positive_coords + negative_coords
        rng.shuffle(coords)
        self.patch_coords = coords        
        self.patch_size = patch_size

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

        return X_patch, y_patch, ice_patch
