"""
Unit tests for classical baseline methods.
"""

import torch
import pytest
import numpy as np
from src.models.baselines_classical import RandomOversampling, NoAugmentation

def test_no_aug_marker():
    b1 = NoAugmentation()
    res = b1.generate(torch.randn(10), torch.randn(10))
    assert res['n_synthetic'] == 0
    assert res['Z'].numel() == 0

def test_aug_mix(tmp_path):
    from src.models.baselines_classical import AugmentationMix
    import cv2
    import numpy as np
    
    # Create dummy image
    img_path = tmp_path / "test.png"
    cv2.imwrite(str(img_path), np.zeros((100, 100, 3), dtype=np.uint8))
    
    b7 = AugmentationMix(seed=42)
    out_dir = tmp_path / "aug"
    out_paths = b7.generate([img_path], out_dir, n_per_image=2)
    
    assert len(out_paths) == 2
    assert out_paths[0].exists()
    assert out_paths[1].exists()

def test_rand_over_logic():
    X = torch.randn(100, 5)
    Y = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    b2 = RandomOversampling(seed=42)
    res = b2.generate(X, Y, n_synthetic=10)
    
    Z = res['Z']
    assert Z.shape == (10, 2)
    # Every row in Z must be one of the rows in Y
    for i in range(10):
        found = False
        for j in range(2):
            if torch.allclose(Z[i], Y[j]):
                found = True
                break
        assert found, f"Row {i} of Z not found in parents Y"

def test_smote_logic():
    from src.models.baselines_classical import SMOTEAugmenter
    X_c = torch.randn(100, 5) + 10.0
    Y_r = torch.randn(20, 5) - 10.0
    b3 = SMOTEAugmenter(k_neighbors=5, seed=42)
    res = b3.generate(X_c, Y_r, n_synthetic=30)
    
    assert res['Z'].shape == (30, 5)
    # Check if Z is in the general vicinity of Y_r
    assert torch.mean(res['Z']) < 0 # Should be closer to -10 than +10
