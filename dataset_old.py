import numpy as np
import torch
from torch.utils.data import Dataset

from load_data import load_data
from image_patching import make_positive_coords, make_negative_coords

class FracturePatchDataset(Dataset):
    def __init__(self, X_full, y_full, ice_mask, patch_size):
        self.X_full = X_full
        self.y_full = y_full
        positive_coords = make_positive_coords(y_full,  X_full, patch_size, 5000)
        negative_coords = make_negative_coords(y_full,  X_full, ice_mask, patch_size, 5000)
        coords = positive_coords + negative_coords
        np.random.shuffle(coords)
        self.patch_coords = coords        
        self.patch_size = patch_size

    def __len__(self):
        return len(self.patch_coords)

    def __getitem__(self, idx):
        row, col = self.patch_coords[idx]
        p = self.patch_size

        X_patch = self.X_full[:, row:row+p, col:col+p]
        y_patch = self.y_full[row:row+p, col:col+p]

        X_patch = torch.from_numpy(X_patch).float()
        y_patch = torch.from_numpy(y_patch).float().unsqueeze(0)

        return X_patch, y_patch