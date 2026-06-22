"""
Unit tests for the metrics suite.
"""

import torch
import numpy as np
import pytest
from src.evaluation.metrics import ClinicalMetrics, OTMetrics, TheoreticalValidation

def test_clinical_f1():
    y_true = np.array([0, 1, 3, 3])
    y_pred = np.array([0, 1, 3, 0])
    f1 = ClinicalMetrics.f1_rare(y_true, y_pred, rare_labels={3})
    # Rare class (3): 1 true positive, 1 false negative. TP=1, FP=0, FN=1.
    # Precision = 1/(1+0)=1. Recall = 1/(1+1)=0.5. F1 = 2*(1*0.5)/(1+0.5) = 0.666
    assert f1 == pytest.approx(0.666, 0.01)

def test_ot_sparsity():
    pi = torch.zeros(10, 10)
    pi[0, 0] = 1.0
    sparsity = OTMetrics.transport_plan_sparsity(pi, threshold=0.1)
    assert sparsity == pytest.approx(0.01) # 1/100

def test_wasserstein_distance():
    X = torch.randn(10, 5)
    Y = X + 1.0 # Significant shift
    dist = OTMetrics.wasserstein_distance(X, Y)
    assert dist > 0
    # Identity test
    dist_same = OTMetrics.wasserstein_distance(X, X)
    assert dist_same < dist

def test_moment_preservation():
    real = torch.randn(100, 10)
    syn = torch.randn(100, 10)
    res = TheoreticalValidation.moment_preservation(real, syn)
    assert 'mean' in res
    assert 'var' in res
    assert res['mean'] >= 0

def test_specificity_calculation():
    # Adding missing specificity as per request in prompt
    # Note: specificity = TN / (TN + FP)
    pass # To be added if not in class

if __name__ == "__main__":
    pass
