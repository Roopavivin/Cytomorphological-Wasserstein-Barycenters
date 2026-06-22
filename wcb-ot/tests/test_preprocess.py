"""
Unit tests for the preprocessing pipeline.
"""

import cv2
import pytest
import numpy as np
from pathlib import Path
from src.data.stain import ReinhardNormalizer
from src.data.preprocess import segment_cell

def test_stain_normalization():
    """Test if ReinhardNormalizer changes stats correctly."""
    img = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
    ref = (np.zeros((100, 100, 3)) + 128).astype(np.uint8)
    
    norm = ReinhardNormalizer()
    norm.fit(ref)
    out = norm.transform(img)
    
    assert out.shape == img.shape
    # Output mean should be closer to 128 in LAB space (Reinhard property)
    # Simple check for change
    assert not np.array_equal(img, out)

def test_resize_dimensions():
    """Verify resizing logic."""
    img = np.zeros((500, 600, 3), dtype=np.uint8)
    resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    assert resized.shape == (256, 256, 3)

def test_segmentation_masks():
    """Verify nucleus and cytoplasm mask extraction."""
    # Create a synthetic cell: dark nucleus (center), gray cytoplasm, white background
    img = np.full((256, 256, 3), 240, dtype=np.uint8) # Background
    cv2.circle(img, (128, 128), 60, (180, 180, 180), -1) # Cytoplasm
    cv2.circle(img, (128, 128), 20, (50, 50, 50), -1)    # Nucleus
    
    nuc_mask, cyt_mask = segment_cell(img)
    
    assert nuc_mask.shape == (256, 256)
    assert cyt_mask.shape == (256, 256)
    assert np.sum(nuc_mask) > 100
    assert np.sum(cyt_mask) > 100
    # Nucleus and cytoplasm should be disjoint
    assert np.logical_and(nuc_mask, cyt_mask).sum() == 0

def test_clahe_contrast():
    """Test if CLAHE increases local contrast/variance."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(img, (50, 50), 30, (100, 100, 100), -1)
    
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    assert not np.array_equal(l, cl)

def test_denoise_effect():
    """Test if denoising changes a noisy image."""
    img = (np.random.rand(100, 100, 3) * 20).astype(np.uint8) + 128
    denoised = cv2.fastNlMeansDenoisingColored(img, None, h=8, templateWindowSize=5, searchWindowSize=21)
    
    assert not np.array_equal(img, denoised)
    # Variance of denoised should be lower
    assert np.var(denoised) < np.var(img)

if __name__ == "__main__":
    pass
