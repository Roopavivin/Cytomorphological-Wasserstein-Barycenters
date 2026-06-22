"""
Unit tests for Sinkhorn OT implementation.
"""

import torch
import pytest
import numpy as np
from src.ot.sinkhorn import EntropicSinkhornOT

def test_sinkhorn_self_distance():
    """W(X, X) should be near zero for small epsilon."""
    X = torch.randn(20, 5)
    ot = EntropicSinkhornOT(epsilon=0.01, log_domain=True, device='cpu')
    res = ot.fit(X, X)
    # Entropic OT diagonal is not exactly zero but very small
    assert res['dist'] < 0.1
    assert res['dist'] >= 0

def test_sinkhorn_plan_marginals():
    """Transport plan should marginalize to weights a and b."""
    n, m = 15, 20
    X = torch.randn(n, 2)
    Y = torch.randn(m, 2)
    a = torch.ones(n) / n
    b = torch.ones(m) / m
    
    ot = EntropicSinkhornOT(epsilon=0.1, log_domain=True, device='cpu')
    res = ot.fit(X, Y, a=a, b=b)
    plan = res['plan']
    
    # Sum across columns should be a
    assert torch.allclose(plan.sum(1), a, atol=1e-3)
    # Sum across rows should be b
    assert torch.allclose(plan.sum(0), b, atol=1e-3)

def test_log_vs_standard():
    """Verify log-domain and standard implementations match for large epsilon."""
    X = torch.randn(10, 2)
    Y = torch.randn(10, 2)
    
    # High epsilon to avoid instability in standard version
    eps = 0.5
    ot_std = EntropicSinkhornOT(epsilon=eps, log_domain=False, device='cpu')
    ot_log = EntropicSinkhornOT(epsilon=eps, log_domain=True, device='cpu')
    
    res_std = ot_std.fit(X, Y)
    res_log = ot_log.fit(X, Y)
    
    assert torch.allclose(res_std['dist'], res_log['dist'], rtol=1e-2)

def test_custom_cost_fn():
    """Test injection of custom cost function (L1 norm)."""
    X = torch.randn(5, 2)
    Y = torch.randn(5, 2)
    
    def l1_cost(x, y):
        # (n, 1, d) - (1, m, d) -> (n, m, d) -> sum(d) -> (n, m)
        return torch.abs(x.unsqueeze(1) - y.unsqueeze(0)).sum(-1)
        
    ot = EntropicSinkhornOT(epsilon=0.1, device='cpu')
    res = ot.fit(X, Y, cost_fn=l1_cost)
    assert res['dist'] > 0

if __name__ == "__main__":
    pass
