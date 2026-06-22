"""
Wasserstein Cellular Barycenter implementation.
Generates synthetic samples as entropic-regularized barycenters of real rare cells.
"""

import random
import time
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple, Callable
from loguru import logger

from src.ot.sinkhorn import EntropicSinkhornOT
from src.utils.io import save_tensor, ensure_dir

class WassersteinCellularBarycenter:
    """
    Implements WCB for synthetic data generation.
    Generates new points as the Wasserstein Barycenter of k-subsets of real data.
    """
    def __init__(
        self, 
        k_subset: int = 5, 
        max_inner_iter: int = 50,
        convergence_tol: float = 1e-4, 
        entropic_speedup: bool = True,
        epsilon: float = 0.05, 
        device: str = 'cuda'
    ):
        """
        Args:
            k_subset (int): Number of real cells to interpolate per synthetic cell.
            max_inner_iter (int): Iterations for the fixed-point barycenter update.
            convergence_tol (float): Stop if z moves less than this.
            entropic_speedup (bool): Use Sinkhorn weights for the update.
            epsilon (float): Regularization for Sinkhorn.
            device (str): Computation device.
        """
        self.k = k_subset
        self.max_iter = max_inner_iter
        self.tol = convergence_tol
        self.entropic_speedup = entropic_speedup
        self.epsilon = epsilon
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        self.ot = EntropicSinkhornOT(epsilon=self.epsilon, device=self.device)

    def _single_barycenter(self, subset: torch.Tensor, cost_fn: Optional[Callable] = None) -> Tuple[torch.Tensor, int]:
        """
        Computes the barycenter of a k-subset of vectors.
        """
        z = subset.mean(dim=0, keepdim=True) # [1, d]
        
        if not self.entropic_speedup:
            return z.squeeze(0), 1

        for i in range(self.max_iter):
            z_old = z.clone()
            
            C = self.ot.compute_cost_matrix(z, subset, cost_fn=cost_fn)
            weights = torch.softmax(-C / self.epsilon, dim=1)
            z = weights @ subset
            
            # Check convergence
            if torch.norm(z - z_old) < self.tol:
                return z.squeeze(0), i + 1
                
        return z.squeeze(0), self.max_iter

    def generate(
        self, 
        Y_rare: torch.Tensor, 
        n_synthetic: int = 1600,
        seed: int = 42,
        cost_fn: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Generates n_synthetic rare cells from real rare cells.
        """
        torch.manual_seed(seed)
        random.seed(seed)
        
        m, d = Y_rare.shape
        Y_rare = Y_rare.to(self.device)
        
        Z = torch.zeros((n_synthetic, d), device=self.device)
        parents_list = []
        iters_list = []
        
        start_time = time.time()
        
        for i in range(n_synthetic):
            indices = random.sample(range(m), self.k)
            subset = Y_rare[indices]
            
            z_i, n_iters = self._single_barycenter(subset, cost_fn=cost_fn)
            
            Z[i] = z_i
            parents_list.append(indices)
            iters_list.append(n_iters)
            
        total_time = time.time() - start_time
        
        return {
            'Z': Z,
            'parents': parents_list,
            'n_iters': iters_list,
            'time': total_time
        }

if __name__ == "__main__":
    pass
