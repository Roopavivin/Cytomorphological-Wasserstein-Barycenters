"""
Unified WCB-OT Generator Model.
Integrates Sinkhorn, Barycenters, Cytomorphological Cost, and Dynamical OT.
"""

import torch
import numpy as np
import time
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

from src.ot.sinkhorn import EntropicSinkhornOT
from src.ot.barycenter import WassersteinCellularBarycenter
from src.ot.cyto_cost import CytomorphologicalCost
from src.ot.dynamical import DynamicalOT
from src.utils.io import ensure_dir, save_tensor

class WCB_OT:
    """
    Master model for Wasserstein Cellular Barycenter - Optimal Transport synthesis.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ot_cfg = config.get('ot', config) # support nested or flat config
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize components
        self.sinkhorn = EntropicSinkhornOT(
            epsilon=self.ot_cfg['sinkhorn']['epsilon'],
            max_iter=self.ot_cfg['sinkhorn']['max_iter'],
            tol=self.ot_cfg['sinkhorn']['tol'],
            device=self.device
        )
        
        self.barycenter = WassersteinCellularBarycenter(
            k_subset=self.ot_cfg['barycenter']['k_subset'],
            epsilon=self.ot_cfg['sinkhorn']['epsilon'],
            device=self.device
        )
        
        self.dynamical = DynamicalOT(
            epsilon=self.ot_cfg['sinkhorn']['epsilon'],
            device=self.device
        )
        
        # State
        self.cost_fn = None
        self.transport_plan = None
        self.wasserstein_dist = None
        self.stats = {}

    def fit_transport(
        self, 
        X_common: torch.Tensor, 
        Y_rare: torch.Tensor,
        X_common_raw: Optional[torch.Tensor] = None,
        Y_rare_raw: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Computes the transport plan using Cytomorphological cost.
        """
        logger.info("Computing Cytomorphological Optimal Transport plan...")
        
        # 1. Initialize cost function with optimized weights
        weights = self.ot_cfg['cyto_cost']
        self.cost_fn = CytomorphologicalCost(
            alpha=weights['alpha'],
            beta=weights['beta'],
            gamma=weights['gamma']
        )
        
        # 2. Compute Sinkhorn π*
        # We pass raw features to the cost_fn call internally in sinkhorn?
        # My sinkhorn.fit takes X, Y and a cost_fn(X, Y).
        # We need to wrap cost_fn to use raw if available.
        
        def wrapped_cost_fn(X_pca, Y_pca):
            # In a real scenario, we map PCA rows back to Raw rows.
            # Here we assume X_common and X_common_raw are aligned.
            return self.cost_fn(X_common_raw, Y_rare_raw)
            
        res = self.sinkhorn.fit(X_common, Y_rare, cost_fn=wrapped_cost_fn)
        
        self.transport_plan = res['plan']
        self.wasserstein_dist = res['dist']
        
        # 3. Calculate Sparsity
        n, m = self.transport_plan.shape
        # Plan is sparse if many entries are near zero
        non_zero = (self.transport_plan > 1.0 / (n * m * 10)).sum().item()
        sparsity = 1.0 - (non_zero / (n * m))
        
        self.stats.update({
            'wasserstein_dist': self.wasserstein_dist.item(),
            'sparsity': sparsity,
            'plan_shape': (n, m)
        })
        
        return self.stats

    def generate_synthetic(
        self, 
        Y_rare: torch.Tensor, 
        X_common: Optional[torch.Tensor] = None,
        Y_rare_raw: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generates hybrid synthetic cohort (Barycenters + Dynamical).
        """
        logger.info("Generating synthetic samples...")
        start_time = time.time()
        
        # Phase A: Barycenter cells (provably distribution-preserving)
        # Use same cost_fn for barycenter generation
        def wrapped_cost_fn(z, subset_pca):
            # Since z is synthetic, we don't have its 'raw' form yet?
            # Actually, WCB logic usually operates on the feature space where k points are picked.
            # For simplicity, if cost_fn is provided to WCB, it uses it.
            # But the 'raw' vs 'pca' mismatch is tricky here.
            # For the synthesis phase, we'll use Euclidean in PCA space 
            # but allow the Sinkhorn weights to guide the geometry.
            return None # Fallback to Euclidean
            
        res_b = self.barycenter.generate(
            Y_rare, 
            n_synthetic=self.ot_cfg['barycenter']['n_synthetic'],
            seed=42
        )
        Z_bary = res_b['Z']
        
        # Phase B: Dynamical interpolation (boundary enrichment)
        Z_dyn = []
        if X_common is not None and self.ot_cfg.get('dynamical', {}).get('enabled', True):
            logger.info("Adding Dynamical OT boundary samples...")
            z03 = self.dynamical.sample_intermediate(X_common, Y_rare, time_point=0.3, n_samples=200)
            z06 = self.dynamical.sample_intermediate(X_common, Y_rare, time_point=0.6, n_samples=200)
            Z_dyn = [z03, z06]
            
        Z_all = torch.cat([Z_bary] + Z_dyn, dim=0)
        
        self.stats['generation_time'] = time.time() - start_time
        self.stats['total_synthetic'] = Z_all.shape[0]
        
        return Z_all

    def save(self, out_dir: Path):
        ensure_dir(out_dir)
        torch.save(self.transport_plan, out_dir / "transport_plan.pt")
        # Metadata
        import json
        with open(out_dir / "model_stats.json", "w") as f:
            json.dump(self.stats, f, indent=4)
        logger.info(f"Model and results saved to {out_dir}")

if __name__ == "__main__":
    pass
