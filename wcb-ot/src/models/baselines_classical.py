"""
Classical Baselines (B1, B2, B3, B7) for cervical cytology synthesis.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, List
from imblearn.over_sampling import SMOTE
from src.utils.io import ensure_dir

class NoAugmentation:
    """Baseline B1: Marker for raw training."""
    def generate(self, X_train: torch.Tensor, y_train: torch.Tensor, n_synthetic: int = 0) -> Dict[str, Any]:
        return {'Z': torch.empty(0), 'method': 'no_aug', 'n_synthetic': 0}

class RandomOversampling:
    """Baseline B2: Uniform duplication."""
    def __init__(self, seed: int = 42):
        self.seed = seed
    def generate(self, X_common: torch.Tensor, Y_rare: torch.Tensor, n_synthetic: int = 1600) -> Dict[str, Any]:
        torch.manual_seed(self.seed)
        m = Y_rare.size(0)
        indices = torch.randint(0, m, (n_synthetic,))
        return {'Z': Y_rare[indices], 'indices': indices.tolist(), 'method': 'rand_over'}

class SMOTEAugmenter:
    """Baseline B3: Linear k-NN interpolation."""
    def __init__(self, k_neighbors: int = 5, seed: int = 42):
        self.k = k_neighbors
        self.seed = seed
    def generate(self, X_common: torch.Tensor, Y_rare: torch.Tensor, n_synthetic: int = 1600) -> Dict[str, Any]:
        n_common, n_rare = X_common.shape[0], Y_rare.shape[0]
        X = torch.cat([X_common, Y_rare]).cpu().numpy()
        y = np.array([0] * n_common + [1] * n_rare)
        sm = SMOTE(sampling_strategy={1: n_rare + n_synthetic}, k_neighbors=self.k, random_state=self.seed)
        X_res, y_res = sm.fit_resample(X, y)
        Z = X_res[n_common + n_rare : ]
        return {'Z': torch.from_numpy(Z).float(), 'method': 'smote'}

import albumentations as A
import cv2
from pathlib import Path

class AugmentationMix:
    """
    Baseline B7: Traditional pixel-space augmentations.
    Preserves cell structure but adds diversity via stochastic transforms.
    """
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.transform = A.Compose([
            A.Rotate(limit=30, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomScale(scale_limit=0.1, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, p=0.5),
            A.GaussNoise(var_limit=(10, 50), p=0.5)
        ], p=1.0)

    def generate(self, image_paths: list[Path], output_dir: Path, n_per_image: int = 3) -> list[Path]:
        """Generates n_per_image augmented versions of each input image."""
        np.random.seed(self.seed)
        new_paths = []
        ensure_dir(output_dir)
        
        for path in image_paths:
            img = cv2.imread(str(path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            for i in range(n_per_image):
                augmented = self.transform(image=img)['image']
                out_path = output_dir / f"{path.stem}_aug{i}.png"
                cv2.imwrite(str(out_path), cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
                new_paths.append(out_path)
                
        return new_paths

if __name__ == "__main__":
    pass
