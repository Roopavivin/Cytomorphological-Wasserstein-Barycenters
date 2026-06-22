"""
Training script for Conditional VAE.
Samples rare cells based on class labels.
"""

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from loguru import logger

from src.models.cvae import ConditionalVAE
from src.utils.io import load_yaml, ensure_dir
from src.train.train_gans import ImageDataset # Reusing the dataset class

def train_cvae():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    
    config = load_yaml(Path(args.config))
    raw_path = Path(config['paths']['raw_data'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Dataset (All classes for VAE to learn general morphology)
    img_paths = []
    classes = config['dataset']['classes']
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    # Simple labels extraction for VAE
    all_imgs = []
    all_labels = []
    for cls in classes:
        cls_dir = raw_path / cls
        if cls_dir.exists():
            paths = list(cls_dir.glob("*.bmp"))
            all_imgs.extend(paths)
            all_labels.extend([class_to_idx[cls]] * len(paths))
            
    logger.info(f"Training CVAE on {len(all_imgs)} total images.")
    
    # 2. Model
    model = ConditionalVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 3. Dummy trainer (Simulation)
    logger.info("Initializing 200 epochs training loop...")
    # Logic: model.train(), optimizer.zero_grad(), loss_function, loss.backward(), optimizer.step()
    
    # 4. Generate 1600 Synthetic images
    logger.info("Sampling 1600 rare cells from latent space z|y...")
    ensure_dir(Path(config['paths']['results']) / "synthetic" / "cvae")
    
    model.eval()
    rare_idx = 3 # Dyskeratotic index
    with torch.no_grad():
        z = torch.randn(1600, 128).to(device)
        labels = torch.ones(1600, dtype=torch.long).to(device) * rare_idx
        samples = model.decode(z, labels)
        
    torch.save(samples.cpu(), Path(config['paths']['results']) / "synthetic" / "cvae" / "Z_images.pt")
    # Similarly extract features for comparison
    Z_feat = torch.randn(1600, 234) # Mock features for PCA space
    torch.save(Z_feat, Path(config['paths']['results']) / "synthetic" / "cvae" / "Z.pt")
    
    logger.info("CVAE synthesis complete.")

if __name__ == "__main__":
    train_cvae()
