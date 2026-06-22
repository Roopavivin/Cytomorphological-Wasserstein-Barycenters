"""
Entropic Optimal Transport implementation using Sinkhorn algorithm.
Supports log-domain stabilization for numerical robustness.
"""

import torch
import torch.nn as nn
from typing import Optional, Callable, Dict, Any, Tuple
from loguru import logger

class EntropicSinkhornOT:
    """
    Computes the Entropic-regularized Wasserstein distance (Sinkhorn Distance).
    """
    def __init__(
        self, 
        epsilon: float = 0.05, 
        max_iter: int = 1000, 
        tol: float = 1e-6,
        log_domain: bool = True, 
        cond: float = 1e-8, 
        device: str = 'cuda'
    ):
        """
        Args:
            epsilon (float): Regularization parameter.
            max_iter (int): Maximum number of Sinkhorn iterations.
            tol (float): Convergence threshold.
            log_domain (bool): Use log-domain stabilization.
            cond (float): Numerical conditioning constant.
            device (str): Device to run on ('cuda' or 'cpu').
        """
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.log_domain = log_domain
        self.cond = cond
        self.device = device if torch.cuda.is_available() else 'cpu'

    def compute_cost_matrix(
        self, 
        X: torch.Tensor, 
        Y: torch.Tensor,
        cost_fn: Optional[Callable] = None
    ) -> torch.Tensor:
        """
        Computes the cost matrix between X and Y.
        
        Args:
            X (torch.Tensor): Source samples [n, d].
            Y (torch.Tensor): Target samples [m, d].
            cost_fn (Callable): Custom cost function. If None, uses squared Euclidean.
            
        Returns:
            torch.Tensor: Cost matrix C [n, m].
        """
        if cost_fn is not None:
            return cost_fn(X, Y)
        
        # Squared Euclidean: ||x - y||^2 = ||x||^2 + ||y||^2 - 2xy'
        x_norm = (X**2).sum(1).view(-1, 1)
        y_norm = (Y**2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(X, Y.t())
        return torch.clamp(dist, min=0.0)

    def _softmin(self, C_over_eps: torch.Tensor, g_over_eps: torch.Tensor, dim: int) -> torch.Tensor:
        """Helper for LogSumExp stabilization."""
        return -self.epsilon * torch.logsumexp(-C_over_eps + g_over_eps, dim=dim)

    def fit(
        self, 
        X: torch.Tensor, 
        Y: torch.Tensor, 
        a: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None, 
        cost_fn: Optional[Callable] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the Optimal Transport plan and Sinkhorn distance.

        Args:
            X, Y (torch.Tensor): Source and target batches.
            a, b (torch.Tensor): Probability weights. Defaults to uniform.
            cost_fn (Callable): Optional custom cost function.

        Returns:
            Dict: {'dist': sinkhorn_distance, 'plan': transport_plan}
        """
        X, Y = X.to(self.device), Y.to(self.device)
        n, m = X.size(0), Y.size(0)
        
        if a is None: a = torch.ones(n, device=self.device) / n
        if b is None: b = torch.ones(m, device=self.device) / m
        
        C = self.compute_cost_matrix(X, Y, cost_fn)
        
        if not self.log_domain:
            # Standard Sinkhorn (can be unstable for small epsilon)
            K = torch.exp(-C / self.epsilon)
            u = torch.ones(n, device=self.device) / n
            v = torch.ones(m, device=self.device) / m
            
            for i in range(self.max_iter):
                u_old = u.clone()
                u = a / (torch.mv(K, v) + self.cond)
                v = b / (torch.mv(K.t(), u) + self.cond)
                
                if torch.allclose(u, u_old, atol=self.tol):
                    break
            
            plan = torch.diag(u) @ K @ torch.diag(v)
            dist = torch.sum(plan * C)
        else:
            # stabilized log-domain
            f = torch.zeros(n, device=self.device)
            g = torch.zeros(m, device=self.device)
            
            log_a = torch.log(a)
            log_b = torch.log(b)
            
            C_over_eps = C / self.epsilon
            
            for i in range(self.max_iter):
                f_old = f.clone()
                
                # f = eps * (log a - logsumexp(-C/eps + g/eps))
                f = self.epsilon * (log_a - torch.logsumexp(-C_over_eps + g.view(1, -1) / self.epsilon, dim=1))
                g = self.epsilon * (log_b - torch.logsumexp(-C_over_eps.t() + f.view(1, -1) / self.epsilon, dim=1))
                
                if torch.allclose(f, f_old, atol=self.tol):
                    break
            
            # Transport plan: exp((f + g - C) / eps)
            log_plan = (f.view(-1, 1) + g.view(1, -1) - C) / self.epsilon
            plan = torch.exp(log_plan)
            dist = torch.sum(plan * C)
            
        return {'dist': dist, 'plan': plan}

if __name__ == "__main__":
    X = torch.randn(10, 2)
    Y = torch.randn(10, 2)
    ot = EntropicSinkhornOT(epsilon=0.01)
    res = ot.fit(X, Y)
    print(f"Sinkhorn Distance: {res['dist'].item():.4f}")
