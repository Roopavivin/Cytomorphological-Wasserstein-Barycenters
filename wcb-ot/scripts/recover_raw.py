"""
Recover raw 618-D features using already fine-tuned model and processed data.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm
from joblib import Parallel, delayed

from src.data.features import SIPaKMeDFeatureDataset, extract_handcrafted
from src.utils.io import load_yaml, ensure_dir

def recover_raw_features():
    config = load_yaml(Path("configs/config.yaml"))
    manifest = pd.read_csv(Path(config['paths']['processed_data']) / "manifest.csv")
    feat_dir = Path(config['paths']['features'])
    raw_dir = feat_dir / "raw_618"
    ensure_dir(raw_dir)
    
    # 1. Hand-crafted
    print("Extracting Hand-crafted (618-D subset)...")
    all_handcrafted = Parallel(n_jobs=-1)(
        delayed(extract_handcrafted)(row) for _, row in tqdm(manifest.iterrows(), total=len(manifest))
    )
    
    # 2. Deep (using saved backbone)
    print("Extracting Deep (512-D) via saved model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = timm.create_model('resnet18', num_classes=5).to(device)
    model.load_state_dict(torch.load("results/models/resnet18_featext.pt", map_location=device))
    model.eval()
    backbone = torch.nn.Sequential(*list(model.children())[:-1])
    
    ds = SIPaKMeDFeatureDataset(manifest)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    
    X_deep = []
    with torch.no_grad():
        for imgs, _ in tqdm(loader):
            f = backbone(imgs.to(device)).squeeze()
            X_deep.append(f.cpu().numpy())
    X_deep = np.concatenate(X_deep)
    
    X_618 = np.column_stack([np.array(all_handcrafted), X_deep])
    
    # 3. Save
    rare_classes = config['dataset']['rare_classes']
    common_classes = config['dataset']['common_classes']
    
    rare_mask = manifest['class'].isin(rare_classes)
    common_mask = manifest['class'].isin(common_classes)
    inter_mask = manifest['class'] == "Koilocytotic"

    torch.save(torch.from_numpy(X_618[common_mask]).float(), raw_dir / "X_common.pt")
    torch.save(torch.from_numpy(X_618[rare_mask]).float(), raw_dir / "X_rare.pt")
    torch.save(torch.from_numpy(X_618[inter_mask]).float(), raw_dir / "X_intermediate.pt")
    
    print(f"Raw features saved to {raw_dir}")

if __name__ == "__main__":
    recover_raw_features()
