"""
Reinhard stain normalization for cervical cytology images.
Converts images to LAB space and aligns statistics to a reference.
"""

import cv2
import numpy as np
from typing import Tuple

class ReinhardNormalizer:
    """
    Implements Reinhard stain normalization.
    """
    def __init__(self):
        self.target_mean = None
        self.target_std = None

    def fit(self, reference_image: np.ndarray) -> None:
        """
        Learns the mean and std from a reference image in LAB space.

        Args:
            reference_image (np.ndarray): RGB image.
        """
        lab = cv2.cvtColor(reference_image, cv2.COLOR_RGB2LAB).astype(np.float32)
        self.target_mean = np.mean(lab, axis=(0, 1))
        self.target_std = np.std(lab, axis=(0, 1))

    def transform(self, image: np.ndarray) -> np.ndarray:
        """
        Applies Reinhard normalization to an image.

        Args:
            image (np.ndarray): RGB image.

        Returns:
            np.ndarray: Normalized RGB image.
        """
        if self.target_mean is None or self.target_std is None:
            # Default fallback to a generic neutral cervical cell statistic if fit() wasn't called
            # Based on standard Superficial_Intermediate averages
            self.target_mean = np.array([150.0, 130.0, 125.0])
            self.target_std = np.array([20.0, 5.0, 5.0])

        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        current_mean = np.mean(lab, axis=(0, 1))
        current_std = np.std(lab, axis=(0, 1))

        # Avoid division by zero
        current_std[current_std == 0] = 1.0

        # Aligining
        lab = (lab - current_mean) / current_std * self.target_std + self.target_mean
        
        # Clip to valid LAB range and convert back
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def normalize_stain(image: np.ndarray, reference: np.ndarray = None) -> np.ndarray:
    """Convenience function for stain normalization."""
    normalizer = ReinhardNormalizer()
    if reference is not None:
        normalizer.fit(reference)
    return normalizer.transform(image)
