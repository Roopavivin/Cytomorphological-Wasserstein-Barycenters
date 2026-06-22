"""
Unit tests for statistical significance logic.
"""

import numpy as np
import pytest
from src.evaluation.stat_tests import paired_t_test, bootstrap_ci, bonferroni_correct, compare_all_methods

def test_paired_t_test():
    # Method A is consistently better than B
    A = [0.95, 0.96, 0.94, 0.95, 0.93]
    B = [0.80, 0.82, 0.81, 0.80, 0.79]
    res = paired_t_test(A, B)
    assert res['p_value'] < 0.001
    assert res['mean_diff'] > 0
    assert bool(res['significant_05']) is True

def test_bootstrap_ci():
    scores = [0.9, 0.91, 0.89, 0.9, 0.9]
    low, high = bootstrap_ci(scores, n_bootstrap=100)
    assert low < high
    assert low > 0.8
    assert high < 1.0

def test_bonferroni():
    pvals = [0.01, 0.04, 0.2]
    corrected = bonferroni_correct(pvals, n_tests=3)
    assert corrected[0] == pytest.approx(0.03)
    assert corrected[2] == pytest.approx(0.6)
    assert corrected[1] == 0.12

def test_compare_all_methods():
    results = {
        'wcb_ot': [0.95, 0.94, 0.96],
        'baseline1': [0.80, 0.81, 0.79],
        'baseline2': [0.90, 0.89, 0.91]
    }
    df = compare_all_methods(results, metric='f1')
    assert len(df) == 3
    assert 'sig_level' in df.columns
    # wcb_ot should have '-' for significance against itself
    wcb_row = df[df['method'] == 'wcb_ot']
    assert wcb_row['sig_level'].iloc[0] == '-'

if __name__ == "__main__":
    pass
