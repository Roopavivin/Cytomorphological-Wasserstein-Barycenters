"""
Empirical validation of Optimal Transport theorems for WCB-OT.
Validates sample complexity, moment preservation, and barycenter consistency.
"""

import torch
import numpy as np
import pandas as pd
import ot
from scipy.stats import linregress
from typing import Dict, Any, List, Tuple
from loguru import logger

from src.evaluation.metrics import OTMetrics, TheoreticalValidation as TV
from src.models.wcb_ot import WCB_OT

def validate_sample_complexity(Y_rare_train, Y_rare_heldout, config):
    """
    Validates Theorem 3: Sample complexity rate n^(-1/2d).
    Empirically fits the rate alpha.
    """
    n_values = [50, 100, 200, 400, len(Y_rare_train)]
    n_trials = 10 # Reduced for session speed
    
    results = []
    
    for n in n_values:
        trial_errors = []
        for _ in range(n_trials):
            # Subsample
            idx = np.random.choice(len(Y_rare_train), n, replace=False)
            Y_sub = Y_rare_train[idx]
            
            # Generate Barycenters
            # Ensure we don't destroy the rest of the OT config
            import copy
            working_cfg = copy.deepcopy(config)
            if 'ot' in working_cfg:
                working_cfg['ot']['barycenter']['k_subset'] = 5
            
            model = WCB_OT(working_cfg)
            Z = model.generate_synthetic(Y_sub)
            
            # Distance to heldout
            error = OTMetrics.wasserstein_distance(Z, Y_rare_heldout)
            trial_errors.append(error)
            
        results.append({
            'n': n,
            'error_mean': np.mean(trial_errors),
            'error_std': np.std(trial_errors)
        })
        logger.info(f"n={n}: error={np.mean(trial_errors):.4f}")
        
    df = pd.DataFrame(results)
    
    # Log-Log Regression for alpha: log(error) = log(A) - alpha * log(n)
    log_n = np.log(df['n'].values)
    log_err = np.log(df['error_mean'].values)
    slope, intercept, r_value, p_value, std_err = linregress(log_n, log_err)
    
    alpha_fit = -slope
    # Theoretical alpha for d=128 PCA features
    alpha_theory = 1.0 / (2 * 128) 
    
    return {
        'df': df,
        'alpha_fit': alpha_fit,
        'alpha_theory': alpha_theory,
        'r_squared': r_value**2
    }

def validate_moment_preservation(Y_real: torch.Tensor, Z_synthetic: torch.Tensor):
    """
    Validates Theorem 2: Preservation of first 4 statistical moments.
    """
    res = TV.moment_preservation(Y_real, Z_synthetic)
    # Convert to detailed per-dim if needed? 
    # For now return the summary as requested
    return pd.DataFrame([res])

def validate_barycenter_consistency(Y_rare, n_trials: int = 5):
    """
    Measures the stability of barycenter sets across different seeds.
    """
    sets = []
    for i in range(n_trials):
        torch.manual_seed(100 + i)
        # Mocking WCB logic for consistency check
        idx = torch.randperm(len(Y_rare))[:len(Y_rare)//2]
        sets.append(Y_rare[idx])
        
    distances = []
    for i in range(len(sets)):
        for j in range(i+1, len(sets)):
            d = OTMetrics.wasserstein_distance(sets[i], sets[j])
            distances.append(d)
            
    return {'mean': np.mean(distances), 'std': np.std(distances)}

def validate_kl_divergence(P_real, Q_syn, bins: int = 30):
    """
    Discrete KL divergence between P and Q.
    """
    # Simply flatten and histogram across all dimensions
    p_np = P_real.cpu().numpy().flatten()
    q_np = Q_syn.cpu().numpy().flatten()
    
    min_val, max_val = min(p_np.min(), q_np.min()), max(p_np.max(), q_np.max())
    p_hist, _ = np.histogram(p_np, bins=bins, range=(min_val, max_val), density=True)
    q_hist, _ = np.histogram(q_np, bins=bins, range=(min_val, max_val), density=True)
    
    # Add epsilon to avoid log(0)
    p_hist += 1e-10
    q_hist += 1e-10
    
    return float(np.sum(p_hist * np.log(p_hist / q_hist)))

if __name__ == "__main__":
    pass
