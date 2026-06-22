"""
Unit tests for Dynamical OT interpolation.
"""

import torch
import numpy as np
import pytest
from src.ot.dynamical import DynamicalOT

def test_dynamical_endpoints():
    """At t=0, output mean should be close to X_common mean."""
    X = torch.randn(100, 5) + 10.0
    Y = torch.randn(100, 5) - 10.0
    dot = DynamicalOT(device='cpu')
    
    Z0 = dot.sample_intermediate(X, Y, time_point=0.0, n_samples=100)
    assert torch.allclose(Z0.mean(0), X.mean(0), atol=0.5)
    
    Z1 = dot.sample_intermediate(X, Y, time_point=1.0, n_samples=100)
    assert torch.allclose(Z1.mean(0), Y.mean(0), atol=0.5)

def test_dynamic_mean_interpolation():
    """Mean at t=0.5 should be between X and Y means."""
    X = torch.zeros(100, 2)
    Y = torch.ones(100, 2) * 10.0
    dot = DynamicalOT(device='cpu')
    
    Z05 = dot.sample_intermediate(X, Y, time_point=0.5, n_samples=100)
    expected_mean = torch.tensor([5.0, 5.0])
    assert torch.allclose(Z05.mean(0), expected_mean, atol=0.5)

def test_monotonicity():
    """W2 distance to target should decrease as t increases."""
    X = torch.randn(50, 2)
    Y = torch.randn(50, 2) + 5.0
    dot = DynamicalOT(device='cpu')
    
    Z_start = dot.sample_intermediate(X, Y, time_point=0.1, n_samples=50)
    Z_end = dot.sample_intermediate(X, Y, time_point=0.9, n_samples=50)
    
    # Distance to target Y
    dist_start = torch.norm(Z_start.mean(0) - Y.mean(0))
    dist_end = torch.norm(Z_end.mean(0) - Y.mean(0))
    
    assert dist_end < dist_start

def test_determinism():
    X = torch.randn(20, 2)
    Y = torch.randn(20, 2)
    dot = DynamicalOT(device='cpu')
    
    Z1 = dot.sample_intermediate(X, Y, time_point=0.3, seed=42)
    Z2 = dot.sample_intermediate(X, Y, time_point=0.3, seed=42)
    assert torch.allclose(Z1, Z2)

if __name__ == "__main__":
    pass
