"""
CLI script to execute the full WCB-OT generation pipeline.
"""

import torch
import argparse
import pandas as pd
from pathlib import Path
from loguru import logger
import time

from src.models.wcb_ot import WCB_OT
from src.utils.io import load_yaml, ensure_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--ot_config", type=str, default="configs/ot_config.yaml")
    args = parser.parse_args()
    
    config = load_yaml(Path(args.config))
    ot_config = load_yaml(Path(args.ot_config))
    
    # 1. Load Train Split Data
    feat_dir = Path(config['paths']['features'])
    raw_dir = feat_dir / "raw_618"
    manifest = pd.read_csv(feat_dir / "feature_manifest.csv")
    
    # Identify indices for training
    common_classes = config['dataset']['common_classes']
    rare_classes = config['dataset']['rare_classes']
    
    # PCA Features
    X_common_all = torch.load(feat_dir / "X_common.pt")
    X_rare_all = torch.load(feat_dir / "X_rare.pt")
    
    # Raw Features
    X_common_raw_all = torch.load(raw_dir / "X_common.pt")
    Y_rare_raw_all = torch.load(raw_dir / "X_rare.pt")
    
    # Filter for 'train' split
    mask_c = manifest[manifest['class'].isin(common_classes)]['split'] == 'train'
    mask_r = manifest[manifest['class'].isin(rare_classes)]['split'] == 'train'
    
    X_common = X_common_all[mask_c.values]
    Y_rare = X_rare_all[mask_r.values]
    X_common_raw = X_common_raw_all[mask_c.values]
    Y_rare_raw = Y_rare_raw_all[mask_r.values]
    
    # 2. Initialize Model
    # Combine configs for the model
    full_cfg = {**config, 'ot': ot_config}
    model = WCB_OT(full_cfg)
    
    # 3. Fit Transport
    model.fit_transport(X_common, Y_rare, X_common_raw, Y_rare_raw)
    
    # 4. Generate Synthetic Cohort
    Z_synth = model.generate_synthetic(Y_rare, X_common=X_common)
    
    # 5. Save Results
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(config['paths']['results']) / "synthetic" / f"wcb_ot_{timestamp}"
    model.save(out_dir)
    torch.save(Z_synth, out_dir / "Z_synthetic_final.pt")
    
    # 6. Summary Table
    stats = model.stats
    print("\n" + "="*50)
    print("WCB-OT Final Generation Summary")
    print("-" * 50)
    print(f"Wasserstein distance  : {stats['wasserstein_dist']:.4f}")
    print(f"Transport sparsity %  : {stats['sparsity']*100:.2f}%")
    print(f"Total synthetic cells : {stats['total_synthetic']}")
    print(f"Generation time       : {stats['generation_time']:.2f}s")
    
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
    print(f"Peak GPU memory       : {peak_mem:.2f} MB")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
