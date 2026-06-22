"""
Dynamical Optimal Transport for temporal cellular interpolation.
Implements McCann displacement interpolation to generate intermediate cell states.
"""

import torch
import numpy as np
import ot
from typing import List, Dict, Any, Tuple
from loguru import logger

class DynamicalOT:
    """
    Computes interpolants between two distributions using displacement interpolation.
    """
    def __init__(self, epsilon: float = 0.05, device: str = 'cuda'):
        self.epsilon = epsilon
        self.device = 'cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu'

    def sample_intermediate(
        self, 
        X_common: torch.Tensor, 
        Y_rare: torch.Tensor, 
        time_point: float = 0.3,
        n_samples: int = 200,
        seed: int = 42
    ) -> torch.Tensor:
        """
        Samples n_samples cells at a specific time point t in [0, 1].
        t=0: X_common (Benign), t=1: Y_rare (Malignant).
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Ensure input counts match or handle distributionally
        n, d = X_common.shape
        m, _ = Y_rare.shape
        
        # 1. Compute OT Plan between X and Y
        # We treat them as uniform distributions
        a = np.ones((n,)) / n
        b = np.ones((m,)) / m
        
        # Compute cost matrix (Euclidean squared)
        C = ot.dist(X_common.cpu().numpy(), Y_rare.cpu().numpy(), metric='sqeuclidean')
        C /= C.max() # Normalize for stability
        
        # Sinkhorn plan
        pi = ot.sinkhorn(a, b, C, self.epsilon)
        
        # 2. Sample pairs (i, j) proportional to plan weights
        # Flatten plan and sample indices
        pi_flat = pi.flatten()
        indices = np.random.choice(len(pi_flat), size=n_samples, p=pi_flat / pi_flat.sum())
        
        # Convert flat indices back to (i, j)
        rows = indices // m
        cols = indices % m
        
        # 3. McCann Displacement Interpolation
        # z(t) = (1-t)*x_i + t*y_j
        X_sampled = X_common[rows]
        Y_sampled = Y_rare[cols]
        
        Z_t = (1.0 - time_point) * X_sampled + time_point * Y_sampled
        
        return Z_t.to(self.device)

    def compute_path(
        self, 
        X_common: torch.Tensor, 
        Y_rare: torch.Tensor, 
        time_steps: List[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ) -> Dict[float, torch.Tensor]:
        """
        Computes interpolation path across multiple time steps.
        """
        path = {}
        for t in time_steps:
            path[t] = self.sample_intermediate(X_common, Y_rare, time_point=t, n_samples=min(len(X_common), 500))
        return path

if __name__ == "__main__":
    pass
