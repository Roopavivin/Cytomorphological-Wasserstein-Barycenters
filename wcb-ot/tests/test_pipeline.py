"""
Partial pipeline tests for unified WCB_OT model.
"""

import torch
import pytest
from src.models.wcb_ot import WCB_OT

def test_wcb_ot_pipeline_tiny():
    # Tiny data
    n, m, d_pca, d_raw = 50, 30, 16, 32
    X = torch.randn(n, d_pca)
    Y = torch.randn(m, d_pca)
    X_raw = torch.randn(n, 618) # Must be 618 for cyto_cost indices
    Y_raw = torch.randn(m, 618)
    
    config = {
        'ot': {
            'sinkhorn': {'epsilon': 0.05, 'max_iter': 100, 'tol': 1e-4},
            'barycenter': {'k_subset': 3, 'n_synthetic': 100},
            'cost_weights': {'alpha': 0.5, 'beta': 0.35, 'gamma': 0.15},
            'dynamical': {'enabled': True}
        }
    }
    
    model = WCB_OT(config)
    
    # 1. Fit
    stats = model.fit_transport(X, Y, X_raw, Y_raw)
    assert stats['wasserstein_dist'] > 0
    assert stats['sparsity'] > 0 # At least some entries should be small
    
    # 2. Generate
    Z = model.generate_synthetic(Y, X_common=X)
    # 100 (bary) + 400 (dyn) = 500
    assert Z.shape == (500, 16)
    assert not torch.isnan(Z).any()

if __name__ == "__main__":
    pass
