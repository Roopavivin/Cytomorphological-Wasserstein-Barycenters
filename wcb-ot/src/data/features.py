"""
Feature extraction pipeline for WCB-OT.
Extracts 618-D feature vectors (Morphological, Texture, Deep) and applies PCA.
"""

import os
import cv2
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm
from loguru import logger
import timm
from skimage.measure import regionprops
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

from src.utils.io import load_yaml, ensure_dir, save_pickle, load_pickle

class MorphologicalExtractor:
    """Extracts 42 morphological features."""
    def extract(self, image: np.ndarray, nuc_mask: np.ndarray, cyt_mask: np.ndarray) -> np.ndarray:
        # Nuclear Features (15)
        props = regionprops(nuc_mask)
        if not props: return np.zeros(42)
        p = props[0]
        
        # Chromatin density (mean L channel in nucleus)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:,:,0]
        chromatin_density = np.mean(l_channel[nuc_mask > 0]) if np.any(nuc_mask) else 0
        
        nuc_feats = [
            p.area, p.perimeter, p.axis_major_length, p.axis_minor_length,
            p.eccentricity, p.solidity, p.extent, p.area_convex,
            p.equivalent_diameter_area, p.orientation,
            (4 * np.pi * p.area) / (p.perimeter**2) if p.perimeter > 0 else 0, # compactness
            (4 * p.area) / (np.pi * p.axis_major_length**2) if p.axis_major_length > 0 else 0, # roundness
            p.feret_diameter_max if hasattr(p, 'feret_diameter_max') else 0,
            p.perimeter**2 / (4 * np.pi * p.area) if p.area > 0 else 0, # irregularity
            chromatin_density
        ]
        
        # Cytoplasmic Features (12)
        c_props = regionprops(cyt_mask)
        if c_props:
            cp = c_props[0]
            cyt_vals = image[cyt_mask > 0]
            cyt_feats = [
                cp.area, cp.perimeter,
                (4 * np.pi * cp.area) / (cp.perimeter**2) if cp.perimeter > 0 else 0, # circularity
                cp.axis_major_length / cp.axis_minor_length if cp.axis_minor_length > 0 else 0, # aspect ratio
                (4 * cp.area) / (np.pi * cp.axis_major_length**2) if cp.axis_major_length > 0 else 0, # roundness
                cp.solidity,
                np.var(cyt_vals) if len(cyt_vals) > 0 else 0,
                np.mean(cyt_vals) if len(cyt_vals) > 0 else 0,
                np.std(cyt_vals) if len(cyt_vals) > 0 else 0,
                skew(cyt_vals.flatten()) if len(cyt_vals) > 5 else 0,
                kurtosis(cyt_vals.flatten()) if len(cyt_vals) > 5 else 0,
                cp.perimeter / np.sqrt(cp.area) if cp.area > 0 else 0 # edge roughness proxy
            ]
        else:
            cyt_feats = [0] * 12
            
        # Ratios (3)
        n_area = nuc_feats[0]
        c_area = cyt_feats[0]
        ratios = [
            n_area / c_area if c_area > 0 else 0,
            n_area / (n_area + c_area) if (n_area + c_area) > 0 else 0,
            np.mean(image[nuc_mask > 0]) / np.mean(image[cyt_mask > 0]) if np.any(cyt_mask) and np.any(nuc_mask) else 1
        ]
        
        # Fourier Descriptors (12) of nuclear boundary
        fourier = np.zeros(12)
        contours, _ = cv2.findContours(nuc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            cnt = contours[0].reshape(-1, 2)
            if len(cnt) > 12:
                coeffs = np.fft.fft(cnt[:,0] + 1j*cnt[:,1])
                fourier = np.abs(coeffs[:12])
        
        return np.concatenate([nuc_feats, cyt_feats, ratios, fourier])

class TextureExtractor:
    """Extracts 64 texture features via GLCM."""
    def extract(self, image: np.ndarray, nuc_mask: np.ndarray, cyt_mask: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        nuc_gray = gray.copy()
        nuc_gray[nuc_mask == 0] = 0
        
        glcm = graycomatrix(nuc_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
        
        final_64 = []
        for angle in range(4):
            P = glcm[:, :, 0, angle]
            # Essential Haralick-style stats
            ent = -np.sum(P * np.log2(P + 1e-15))
            con = graycoprops(glcm, 'contrast')[0, angle]
            dis = graycoprops(glcm, 'dissimilarity')[0, angle]
            hom = graycoprops(glcm, 'homogeneity')[0, angle]
            asm = graycoprops(glcm, 'ASM')[0, angle]
            nrg = graycoprops(glcm, 'energy')[0, angle]
            cor = graycoprops(glcm, 'correlation')[0, angle]
            
            angle_vec = [con, dis, hom, asm, nrg, cor, ent]
            angle_vec += [0] * (16 - len(angle_vec)) 
            final_64.extend(angle_vec)
            
        return np.array(final_64)

class DeepExtractor:
    """Extracts 512-D deep features using fine-tuned ResNet-18."""
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.model = timm.create_model('resnet18', pretrained=True, num_classes=5).to(device)
        self.backbone = None

    def fine_tune(self, train_loader: DataLoader, epochs: int = 5):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.model.train()
        for epoch in range(epochs):
            pbar = tqdm(train_loader, desc=f"Fine-tuning Epoch {epoch+1}/{epochs}")
            for imgs, labels in pbar:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=loss.item())
        
        self.model.eval()
        self.backbone = nn.Sequential(*list(self.model.children())[:-1])

    def extract_batch(self, loader: DataLoader) -> np.ndarray:
        feats = []
        with torch.no_grad():
            for imgs, _ in tqdm(loader, desc="Deep Feature Extraction"):
                f = self.backbone(imgs.to(self.device))
                feats.append(f.squeeze().cpu().numpy())
        return np.concatenate(feats)

class SIPaKMeDFeatureDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame):
        self.manifest = manifest
        self.classes = ["Superficial_Intermediate", "Parabasal", "Koilocytotic", "Dyskeratotic", "Metaplastic"]
        
    def __len__(self): return len(self.manifest)
    
    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        img = np.load(row['processed_path']).transpose(2,0,1).astype(np.float32) / 255.0
        label = self.classes.index(row['class'])
        return torch.tensor(img), torch.tensor(label)

def extract_handcrafted(row: pd.Series) -> np.ndarray:
    """Helper for parallelization."""
    img = np.load(row['processed_path'])
    n_mask = np.load(row['nuclear_mask_path'])
    c_mask = np.load(row['cytoplasm_mask_path'])
    
    m_ext = MorphologicalExtractor()
    t_ext = TextureExtractor()
    
    m_f = m_ext.extract(img, n_mask, c_mask)
    t_f = t_ext.extract(img, n_mask, c_mask)
    return np.concatenate([m_f, t_f])

def run_feature_extraction(config_path: str):
    config = load_yaml(Path(config_path))
    manifest = pd.read_csv(Path(config['paths']['processed_data']) / "manifest.csv")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Fine-tune Deep Model
    logger.info(f"Fine-tuning deep model on {device}...")
    train_df = manifest[manifest['split'] == 'train']
    train_ds = SIPaKMeDFeatureDataset(train_df)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    
    deep_ext = DeepExtractor(device)
    deep_ext.fine_tune(train_loader, epochs=5)
    ensure_dir(Path("results/models/resnet18_featext.pt"))
    torch.save(deep_ext.model.state_dict(), "results/models/resnet18_featext.pt")
    
    # 2. Extract Hand-crafted features (Parallelized)
    logger.info("Extracting Hand-crafted features (Morphological + Texture)...")
    all_handcrafted = Parallel(n_jobs=-1)(
        delayed(extract_handcrafted)(row) for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Hand-crafted")
    )
    
    # 3. Extract Deep features
    all_ds = SIPaKMeDFeatureDataset(manifest)
    all_loader = DataLoader(all_ds, batch_size=32, shuffle=False)
    X_deep = deep_ext.extract_batch(all_loader)
    
    X_618 = np.hstack([np.array(all_handcrafted), X_deep])
    logger.info(f"Feature extraction complete. Shape: {X_618.shape}")
    
    # 4. PCA reduction
    logger.info("Applying PCA to reach 128-D...")
    train_indices = manifest[manifest['split'] == 'train'].index
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_618)
    
    pca = PCA(n_components=128)
    pca.fit(X_scaled[train_indices])
    
    var_sum = pca.explained_variance_ratio_.sum()
    logger.info(f"Explained variance (128D): {var_sum:.4f}")
    
    if var_sum < 0.95:
        logger.info("Variance < 0.95. Adjusting PCA n_components...")
        pca = PCA(n_components=0.95)
        pca.fit(X_scaled[train_indices])
        logger.info(f"New dimension for 95% variance: {pca.n_components_}")
        
    X_reduced = pca.transform(X_scaled)
    
    # 5. Save results
    feat_dir = Path(config['paths']['features'])
    raw_dir = feat_dir / "raw_618"
    ensure_dir(raw_dir)
    
    rare_classes = config['dataset']['rare_classes']
    common_classes = config['dataset']['common_classes']
    
    rare_mask = manifest['class'].isin(rare_classes)
    common_mask = manifest['class'].isin(common_classes)
    inter_mask = manifest['class'] == "Koilocytotic"

    # Save RAW 618-D
    torch.save(torch.from_numpy(X_618[common_mask]).float(), raw_dir / "X_common.pt")
    torch.save(torch.from_numpy(X_618[rare_mask]).float(), raw_dir / "X_rare.pt")
    torch.save(torch.from_numpy(X_618[inter_mask]).float(), raw_dir / "X_intermediate.pt")
    
    # Save PCA Reduced
    save_pickle(pca, feat_dir / "pca_model.pkl")
    
    torch.save(torch.from_numpy(X_reduced[common_mask]).float(), feat_dir / "X_common.pt")
    torch.save(torch.from_numpy(X_reduced[rare_mask]).float(), feat_dir / "X_rare.pt")
    torch.save(torch.from_numpy(X_reduced[inter_mask]).float(), feat_dir / "X_intermediate.pt")
    
    for split in ['train', 'val', 'test']:
        mask = manifest['split'] == split
        labels = manifest[mask]['class'].map({c: i for i, c in enumerate(all_ds.classes)})
        torch.save(torch.tensor(labels.values), feat_dir / f"labels_{split}.pt")

    manifest['feature_index'] = range(len(manifest))
    manifest[['image_id', 'split', 'class', 'feature_index']].to_csv(feat_dir / "feature_manifest.csv", index=False)
    
    logger.info("Feature extraction and reduction saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    run_feature_extraction(args.config)
