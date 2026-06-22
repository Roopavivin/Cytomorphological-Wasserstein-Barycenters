"""
Cytomorphological Cost Function for Optimal Transport.
Combines nuclear, chromatin, and membrane sub-costs with biological penalties.
"""

import torch
import numpy as np
from typing import Optional, List
from loguru import logger

class CytomorphologicalCost:
    """
    Implements c(x,y) = alpha*Cn + beta*Cc + gamma*Cm with normalization.
    """
    def __init__(self, alpha: float, beta: float, gamma: float, lambda_penalty: float = 10.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_penalty = lambda_penalty

    def _normalize(self, M: torch.Tensor) -> torch.Tensor:
        """Min-max normalization to [0, 1]."""
        m_min = M.min()
        m_max = M.max()
        if m_max > m_min:
            return (M - m_min) / (m_max - m_min)
        return M

    def c_nuclear(self, X_raw: torch.Tensor, Y_raw: torch.Tensor) -> torch.Tensor:
        """
        Nuclear morphology cost: [nc_ratio, area, eccentricity].
        Indices: nc_ratio=27, area=0, eccentricity=4.
        """
        indices = [27, 0, 4]
        x_nuc = X_raw[:, indices]
        y_nuc = Y_raw[:, indices]
        
        # Vectorized Euclidean squared
        # (n, 1, 3) - (1, m, 3) -> (n, m, 3) -> sum(dim=2)
        C = torch.cdist(x_nuc, y_nuc, p=2)**2
        
        # Reversed transport penalty: nc_x > 0.6 (malignant-like) to nc_y < 0.3 (benign)
        # Assuming nc_ratio is at col 27
        nc_x = X_raw[:, 27].view(-1, 1)
        nc_y = Y_raw[:, 27].view(1, -1)
        
        penalty_mask = (nc_x > 0.6) & (nc_y < 0.3)
        C[penalty_mask] += self.lambda_penalty * 10.0
        
        return self._normalize(C)

    def c_chromatin(self, X_raw: torch.Tensor, Y_raw: torch.Tensor) -> torch.Tensor:
        """
        Chromatin texture cost: contrast, correlation, entropy across 4 directions.
        In 618-D: texture starts at 42. Block of 16 per direction.
        Contrast: 42, 58, 74, 90
        Correlation: 47, 63, 79, 95
        Entropy: 48, 64, 80, 96
        """
        indices = [42, 58, 74, 90, 47, 63, 79, 95, 48, 64, 80, 96]
        x_text = X_raw[:, indices]
        y_text = Y_raw[:, indices]
        
        C = torch.cdist(x_text, y_text, p=2)**2
        return self._normalize(C)

    def c_membrane(self, X_raw: torch.Tensor, Y_raw: torch.Tensor) -> torch.Tensor:
        """Membrane cost: First 12 Fourier descriptors [30:42]."""
        indices = list(range(30, 42))
        x_mem = X_raw[:, indices]
        y_mem = Y_raw[:, indices]
        
        C = torch.cdist(x_mem, y_mem, p=2)**2
        return self._normalize(C)

    def __call__(self, X_raw: torch.Tensor, Y_raw: torch.Tensor) -> torch.Tensor:
        """Combines all sub-costs into a single cost matrix."""
        cn = self.c_nuclear(X_raw, Y_raw)
        cc = self.c_chromatin(X_raw, Y_raw)
        cm = self.c_membrane(X_raw, Y_raw)
        
        return self.alpha * cn + self.beta * cc + self.gamma * cm

if __name__ == "__main__":
    pass
