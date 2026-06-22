"""
Unit tests for Wasserstein Cellular Barycenter implementation.
"""

import torch
import pytest
import numpy as np
from src.ot.barycenter import WassersteinCellularBarycenter

def test_barycenter_shape():
    Y = torch.randn(20, 10)
    wcb = WassersteinCellularBarycenter(k_subset=5, device='cpu')
    res = wcb.generate(Y, n_synthetic=50)
    assert res['Z'].shape == (50, 10)
    assert len(res['parents']) == 50

def test_barycenter_k1_identity():
    """With k=1, barycenter must return the original point."""
    Y = torch.randn(10, 5)
    wcb = WassersteinCellularBarycenter(k_subset=1, device='cpu')
    res = wcb.generate(Y, n_synthetic=10, seed=42)
    Z = res['Z']
    parents = res['parents']
    
    for i in range(10):
        parent_idx = parents[i][0]
        assert torch.allclose(Z[i], Y[parent_idx], atol=1e-5)

def test_barycenter_convex_hull():
    """Verify barycenter lies within the bounding box of parents."""
    Y = torch.randn(50, 2)
    wcb = WassersteinCellularBarycenter(k_subset=5, device='cpu')
    res = wcb.generate(Y, n_synthetic=10)
    
    for i in range(10):
        parents_idx = res['parents'][i]
        parents = Y[parents_idx]
        z = res['Z'][i]
        
        # Check if z is within the min/max range of its parents per dimension
        assert torch.all(z >= parents.min(0)[0] - 1e-5)
        assert torch.all(z <= parents.max(0)[0] + 1e-5)

def test_barycenter_determinism():
    Y = torch.randn(20, 5)
    wcb = WassersteinCellularBarycenter(k_subset=3, device='cpu')
    res1 = wcb.generate(Y, n_synthetic=10, seed=123)
    res2 = wcb.generate(Y, n_synthetic=10, seed=123)
    assert torch.allclose(res1['Z'], res2['Z'])

def test_barycenter_convergence():
    Y = torch.randn(100, 10)
    wcb = WassersteinCellularBarycenter(k_subset=5, max_inner_iter=50, device='cpu')
    res = wcb.generate(Y, n_synthetic=20)
    avg_iters = sum(res['n_iters']) / len(res['n_iters'])
    assert avg_iters >= 1
    assert avg_iters <= 50

def test_barycenter_k2_midpoint():
    """With k=2 and high epsilon, it should be arithmetic mean."""
    Y = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    # Large epsilon makes softmax weights uniform
    wcb = WassersteinCellularBarycenter(k_subset=2, epsilon=10.0, device='cpu')
    res = wcb.generate(Y, n_synthetic=1, seed=42)
    assert torch.allclose(res['Z'][0], torch.tensor([0.5, 0.5]), atol=1e-2)

if __name__ == "__main__":
    pass
