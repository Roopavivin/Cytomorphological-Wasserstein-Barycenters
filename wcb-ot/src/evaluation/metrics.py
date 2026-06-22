"""
Comprehensive metrics suite for WCB-OT evaluation.
Includes classification, generative, OT-specific, and theoretical metrics.
"""

import torch
import numpy as np
import ot
from sklearn.metrics import f1_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from typing import Dict, Any, List, Set, Tuple, Optional, Callable
from scipy.stats import skew, kurtosis

# Note: Image metrics used conditionally to avoid heavy imports in non-image paths
# import torchmetrics

class ClinicalMetrics:
    """Classification performance metrics for the manuscript."""
    
    @staticmethod
    def f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return f1_score(y_true, y_pred, average='macro')

    @staticmethod
    def f1_rare(y_true: np.ndarray, y_pred: np.ndarray, rare_labels: Set[int] = {3, 4}) -> float:
        """Binary F1 on Rare-vs-Other across the 5 classes."""
        y_true_bin = np.array([1 if y in rare_labels else 0 for y in y_true])
        y_pred_bin = np.array([1 if y in rare_labels else 0 for y in y_pred])
        return f1_score(y_true_bin, y_pred_bin)

    @staticmethod
    def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return balanced_accuracy_score(y_true, y_pred)

    @staticmethod
    def precision_rare(y_true: np.ndarray, y_pred: np.ndarray, rare_labels: Set[int] = {3, 4}) -> float:
        y_true_bin = np.array([1 if y in rare_labels else 0 for y in y_true])
        y_pred_bin = np.array([1 if y in rare_labels else 0 for y in y_pred])
        return precision_score(y_true_bin, y_pred_bin, zero_division=0)

    @staticmethod
    def auc_roc_rare(y_true: np.ndarray, y_score: np.ndarray, rare_labels: Set[int] = {3, 4}) -> float:
        # y_score should be probabilities for the rare classes (summed or max)
        y_true_bin = np.array([1 if y in rare_labels else 0 for y in y_true])
        # If multi-class scores, take max or mean over rare indices
        if len(y_score.shape) > 1:
            indices = list(rare_labels)
            y_score_bin = y_score[:, indices].max(axis=1)
        else:
            y_score_bin = y_score
        return roc_auc_score(y_true_bin, y_score_bin)

class OTMetrics:
    """Optimal Transport and Barycenter specific metrics."""

    @staticmethod
    def wasserstein_distance(X: torch.Tensor, Y: torch.Tensor, epsilon: float = 0.05) -> float:
        """W_2 Sinkhorn distance."""
        X_np, Y_np = X.cpu().numpy(), Y.cpu().numpy()
        a, b = np.ones(len(X_np)) / len(X_np), np.ones(len(Y_np)) / len(Y_np)
        M = ot.dist(X_np, Y_np)
        return float(ot.sinkhorn2(a, b, M, reg=epsilon))

    @staticmethod
    def transport_plan_sparsity(pi: torch.Tensor, threshold: float = 1e-5) -> float:
        """Fraction of entries in pi exceeding threshold."""
        return float((pi > threshold).float().mean())

    @staticmethod
    def marginal_violation(pi: torch.Tensor, mu: torch.Tensor, nu: torch.Tensor) -> Tuple[float, float]:
        """L1 distance between pi's marginals and expected mu, nu."""
        mu_pi = pi.sum(dim=1)
        nu_pi = pi.sum(dim=0)
        return float(torch.norm(mu_pi - mu, 1)), float(torch.norm(nu_pi - nu, 1))

class ImageMetrics:
    """Generative quality metrics using torchmetrics."""
    
    @staticmethod
    def fid(real_images: torch.Tensor, syn_images: torch.Tensor) -> float:
        from torchmetrics.image.fid import FrechetInceptionDistance
        fid_metric = FrechetInceptionDistance(feature=2048).to(real_images.device)
        # Images should be 8-bit [0, 255] for FID
        fid_metric.update(real_images, real=True)
        fid_metric.update(syn_images, real=False)
        return float(fid_metric.compute())

class TheoreticalValidation:
    """Metrics for distribution preservation and moment matching."""

    @staticmethod
    def moment_preservation(real: torch.Tensor, syn: torch.Tensor) -> Dict[str, float]:
        """L2 distance between the first 4 moments of real and syn features."""
        r_np, s_np = real.cpu().numpy(), syn.cpu().numpy()
        moments = {}
        moments['mean'] = float(np.linalg.norm(np.mean(r_np, 0) - np.mean(s_np, 0)))
        moments['var'] = float(np.linalg.norm(np.var(r_np, 0) - np.var(s_np, 0)))
        moments['skew'] = float(np.linalg.norm(skew(r_np, axis=0) - skew(s_np, axis=0)))
        return moments

    @staticmethod
    def morphological_validity_proxy(syn_features: torch.Tensor, real_features: torch.Tensor, 
                                   nc_ratio_idx: int = 27) -> float:
        """
        Conservative proxy: fraction of synthetic samples within 1st-99th percentile of real.
        Default nc_ratio_idx=27 (based on 618-D raw mapping).
        """
        # If input is PCA features, this metric is less meaningful; 
        # it should be applied to raw feature space.
        real_nc = real_features[:, nc_ratio_idx]
        syn_nc = syn_features[:, nc_ratio_idx]
        p1, p99 = np.percentile(real_nc.cpu().numpy(), [1, 99])
        valid = (syn_nc >= p1) & (syn_nc <= p99)
        return float(valid.float().mean())

if __name__ == "__main__":
    pass
