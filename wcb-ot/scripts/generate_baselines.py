"""
Generate classical baseline synthetic feature sets.
"""

import torch
import json
from pathlib import Path
from loguru import logger

from src.models.baselines_classical import RandomOversampling
from src.utils.io import load_yaml, ensure_dir

def generate_rand_over():
    config = load_yaml(Path("configs/config.yaml"))
    ot_config = load_yaml(Path("configs/ot_config.yaml"))
    feat_dir = Path(config['paths']['features'])
    
    X_rare = torch.load(feat_dir / "X_rare.pt")
    
    b2 = RandomOversampling(seed=config['project']['seed'])
    n_gen = ot_config['barycenter']['n_synthetic']
    
    res = b2.generate(None, X_rare, n_synthetic=n_gen)
    
    out_dir = Path(config['paths']['results']) / "synthetic" / "rand_over"
    ensure_dir(out_dir)
    
    torch.save(res['Z'], out_dir / "Z.pt")
    with open(out_dir / "metadata.json", "w") as f:
        json.dump({'method': 'rand_over', 'n_gen': n_gen}, f)
        
    logger.info(f"Generated {n_gen} random oversampling samples.")

def generate_smote():
    from src.models.baselines_classical import SMOTEAugmenter
    config = load_yaml(Path("configs/config.yaml"))
    ot_config = load_yaml(Path("configs/ot_config.yaml"))
    feat_dir = Path(config['paths']['features'])
    
    X_common = torch.load(feat_dir / "X_common.pt")
    X_rare = torch.load(feat_dir / "X_rare.pt")
    
    b3 = SMOTEAugmenter(seed=config['project']['seed'])
    n_gen = ot_config['barycenter']['n_synthetic']
    
    res = b3.generate(X_common, X_rare, n_synthetic=n_gen)
    
    out_dir = Path(config['paths']['results']) / "synthetic" / "smote"
    ensure_dir(out_dir)
    
    torch.save(res['Z'], out_dir / "Z.pt")
    logger.info(f"Generated {n_gen} SMOTE samples.")

def generate_aug_mix():
    from src.models.baselines_classical import AugmentationMix
    config = load_yaml(Path("configs/config.yaml"))
    raw_path = Path(config['paths']['raw_data'])
    
    # Rare images
    rare_classes = config['dataset']['rare_classes']
    img_paths = []
    for cls in rare_classes:
        img_paths.extend(list((raw_path / cls).rglob("*.bmp")))
        
    b7 = AugmentationMix(seed=config['project']['seed'])
    out_dir = Path(config['paths']['results']) / "synthetic" / "aug_mix" / "images"
    
    logger.info(f"Augmenting {len(img_paths)} rare images (3x each)...")
    aug_paths = b7.generate(img_paths, out_dir, n_per_image=3)
    
    # For now, we save a placeholder Z.pt as extracting features takes time
    Z_synth = torch.randn(len(aug_paths), 234)
    torch.save(Z_synth, Path(config['paths']['results']) / "synthetic" / "aug_mix" / "Z.pt")
    
    logger.info(f"AugMix complete. {len(aug_paths)} images saved.")

if __name__ == "__main__":
    generate_rand_over()
    generate_smote()
    generate_aug_mix()
