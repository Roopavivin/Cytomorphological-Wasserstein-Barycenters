"""
Tests for theoretical validation module.
"""

import torch
import pytest
import numpy as np
from src.evaluation.theoretical import validate_moment_preservation, validate_kl_divergence, validate_barycenter_consistency

def test_kl_divergence_range():
    # Identical distributions should have small KL
    P = torch.randn(100, 10)
    Q = P.clone()
    kl = validate_kl_divergence(P, Q)
    assert kl < 0.1
    
    # Different distributions
    Q_shifted = P + 5.0
    kl_diff = validate_kl_divergence(P, Q_shifted)
    assert kl_diff > kl

def test_barycenter_consistency_format():
    Y = torch.randn(50, 10)
    res = validate_barycenter_consistency(Y, n_trials=3)
    assert 'mean' in res
    assert res['mean'] >= 0

def test_moment_preservation_df():
    P = torch.randn(100, 10)
    Q = torch.randn(100, 10)
    df = validate_moment_preservation(P, Q)
    assert 'mean' in df.columns
    assert len(df) == 1

if __name__ == "__main__":
    pass
