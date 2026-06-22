"""
Generate intermediate dynamical cells for t=0.3 and t=0.6.
"""

import torch
import pandas as pd
from pathlib import Path
from loguru import logger

from src.ot.dynamical import DynamicalOT
from src.utils.io import load_yaml, ensure_dir

def generate_dynamical():
    config = load_yaml(Path("configs/config.yaml"))
    ot_config = load_yaml(Path("configs/ot_config.yaml"))
    feat_dir = Path(config['paths']['features'])
    
    # Load (reduced) features for TRAIN split
    manifest = pd.read_csv(feat_dir / "feature_manifest.csv")
    
    # Common (Train)
    common_mask = (manifest['class'].isin(config['dataset']['common_classes'])) & (manifest['split'] == 'train')
    X_common_all = torch.load(feat_dir / "X_common.pt")
    # Need to filter the loaded tensor based on the mask within its own group
    group_mask_common = manifest[manifest['class'].isin(config['dataset']['common_classes'])]['split'] == 'train'
    X_common_train = X_common_all[group_mask_common.values]
    
    # Rare (Train)
    rare_mask = (manifest['class'].isin(config['dataset']['rare_classes'])) & (manifest['split'] == 'train')
    X_rare_all = torch.load(feat_dir / "X_rare.pt")
    group_mask_rare = manifest[manifest['class'].isin(config['dataset']['rare_classes'])]['split'] == 'train'
    X_rare_train = X_rare_all[group_mask_rare.values]
    
    logger.info(f"Dynamical OT between {len(X_common_train)} common and {len(X_rare_train)} rare cells.")
    
    dot = DynamicalOT(epsilon=ot_config['sinkhorn']['epsilon'])
    
    # t = 0.3 (Early transformation)
    Z_03 = dot.sample_intermediate(X_common_train, X_rare_train, time_point=0.3, n_samples=200, seed=config['project']['seed'])
    
    # t = 0.6 (Mid transformation)
    Z_06 = dot.sample_intermediate(X_common_train, X_rare_train, time_point=0.6, n_samples=200, seed=config['project']['seed'])
    
    # Save
    ensure_dir(feat_dir)
    torch.save(Z_03, feat_dir / "Z_dynamical_t03.pt")
    torch.save(Z_06, feat_dir / "Z_dynamical_t06.pt")
    
    logger.info(f"Generated 200 samples for t=0.3 and 200 for t=0.6.")
    logger.info(f"Stats t=0.3: Mean={Z_03.mean():.4f}, Std={Z_03.std():.4f}")
    logger.info(f"Stats t=0.6: Mean={Z_06.mean():.4f}, Std={Z_06.std():.4f}")

if __name__ == "__main__":
    generate_dynamical()
