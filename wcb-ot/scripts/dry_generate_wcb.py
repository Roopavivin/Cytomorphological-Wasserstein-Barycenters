"""
Dry run for WCB generation on the TRAIN split of rare cells.
"""

import torch
import json
import pandas as pd
from pathlib import Path
from loguru import logger

from src.ot.barycenter import WassersteinCellularBarycenter
from src.utils.io import load_yaml, ensure_dir

def dry_run():
    config = load_yaml(Path("configs/config.yaml"))
    ot_config = load_yaml(Path("configs/ot_config.yaml"))
    
    feat_dir = Path(config['paths']['features'])
    manifest = pd.read_csv(feat_dir / "feature_manifest.csv")
    
    # Load all rare features
    X_rare_all = torch.load(feat_dir / "X_rare.pt")
    
    # Find indices in X_rare that belong to 'train' split
    # Note: features.py saved X_rare based on manifest filter
    rare_mask = manifest['class'].isin(config['dataset']['rare_classes'])
    rare_df = manifest[rare_mask].reset_index(drop=True)
    
    train_mask = rare_df['split'] == 'train'
    X_rare_train = X_rare_all[train_mask.values]
    
    logger.info(f"Loaded {X_rare_train.shape[0]} real rare cells for synthesis (TRAIN split).")
    
    wcb = WassersteinCellularBarycenter(
        k_subset=ot_config['barycenter']['k_subset'],
        epsilon=ot_config['sinkhorn']['epsilon'],
        device='cuda'
    )
    
    n_gen = ot_config['barycenter']['n_synthetic']
    result = wcb.generate(X_rare_train, n_synthetic=n_gen, seed=config['project']['seed'])
    
    Z = result['Z']
    avg_iters = sum(result['n_iters']) / len(result['n_iters'])
    
    # Save
    ensure_dir(feat_dir)
    torch.save(Z, feat_dir / "Z_synthetic_wcb.pt")
    
    with open(feat_dir / "Z_synthetic_wcb_parents.json", "w") as f:
        json.dump(result['parents'], f)
        
    logger.info("="*30)
    logger.info(f"WCB Generation Complete.")
    logger.info(f"Generated samples: {Z.shape}")
    logger.info(f"Average inner iterations: {avg_iters:.2f}")
    logger.info(f"Total time: {result['time']:.2f}s")
    logger.info("="*30)

if __name__ == "__main__":
    dry_run()
