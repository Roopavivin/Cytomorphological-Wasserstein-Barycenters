"""
Script to validate OT theorems on real data.
"""

import torch
import pandas as pd
from pathlib import Path
from src.evaluation.theoretical import validate_sample_complexity
from src.utils.io import load_yaml, ensure_dir

def main():
    cfg = load_yaml(Path('configs/config.yaml'))
    ot_cfg = load_yaml(Path('configs/ot_config.yaml'))
    feat_dir = Path(cfg['paths']['features'])
    
    # Load Rare Features
    Y_rare = torch.load(feat_dir / "X_rare.pt")
    # Use a small held-out set from the rare train for complexity validation
    Y_heldout = Y_rare[:50]
    Y_train = Y_rare[50:]
    
    # Run Complexity Analysis
    print("Running Sample Complexity Analysis (Theorem 3)...")
    full_cfg = {**cfg, 'ot': ot_cfg}
    res = validate_sample_complexity(Y_train, Y_heldout, full_cfg)
    
    out_dir = Path("results/tables")
    ensure_dir(out_dir)
    res['df'].to_csv(out_dir / "theorem3_sample_complexity.csv", index=False)
    
    print("\n" + "="*40)
    print("THEOREM 3 VALIDATION RESULTS")
    print("-" * 40)
    print(f"Empirical Convergence Rate (alpha): {res['alpha_fit']:.4f}")
    print(f"Theoretical Rate (1/2d):           {res['alpha_theory']:.4f}")
    print(f"R-squared of Power Law Fit:         {res['r_squared']:.4f}")
    print(f"Table saved to {out_dir / 'theorem3_sample_complexity.csv'}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
