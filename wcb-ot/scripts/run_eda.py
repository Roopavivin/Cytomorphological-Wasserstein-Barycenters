"""
Exploratory Data Analysis Figures for SIPaKMeD.
Generates publication-quality charts for class distribution and imbalance analysis.
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from loguru import logger

def fig_class_distribution(manifest_path: Path, out_dir: Path):
    """
    Figure 1: Class distribution (Bar + Pie).
    Highlights the imbalance motivating the OT approach.
    """
    logger.info(f"Generating Figure 1 from {manifest_path}")
    df = pd.read_csv(manifest_path)
    counts = df['class'].value_counts()
    
    # Setup color scheme
    # Superficial_Intermediate, Parabasal -> Navy
    # Koilocytotic -> Steel Blue
    # Dyskeratotic, Metaplastic -> Red
    color_map = {
        'Superficial_Intermediate': '#001f3f', # Navy
        'Parabasal': '#001f3f',
        'Koilocytotic': '#4682b4', # Steel Blue
        'Dyskeratotic': '#ff4136', # Accent Red
        'Metaplastic': '#ff4136'
    }
    palette = [color_map.get(c, 'gray') for c in counts.index]
    
    # Calculate imbalance stats
    total = len(df)
    rare_count = df['class'].isin(['Dyskeratotic', 'Metaplastic']).sum()
    rare_pct = (rare_count / total) * 100
    imbalance_ratio = (total - rare_count) / (rare_count + 1e-6)
    
    # Plot
    sns.set_theme(style='whitegrid', font_scale=1.1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pan 1: Horizontal Bar
    sns.barplot(x=counts.values, y=counts.index, palette=palette, ax=ax1, hue=counts.index, legend=False)
    for i, v in enumerate(counts.values):
        ax1.text(v + 10, i, str(v), color='black', va='center', fontweight='bold')
    ax1.set_title('Sample Counts per Class', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Number of Isolated Cells')
    
    # Pan 2: Pie Chart
    ax2.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=palette, 
            startangle=140, pctdistance=0.85, labeldistance=1.1)
    # Add a circle at the center to make it a donut (optional, but looks premium)
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    ax2.add_artist(centre_circle)
    ax2.set_title('Proportion Representation', fontsize=12, fontweight='bold')
    
    # Overall enhancements
    plt.suptitle('SIPaKMeD Class Distribution — Motivating Rare-Class Synthesis', fontsize=14, fontweight='bold', y=0.98)
    fig.text(0.5, 0.02, f'Rare class: {rare_pct:.1f}% | Imbalance ratio: {imbalance_ratio:.1f}:1', 
             ha='center', fontsize=11, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save
    ensure_dir(out_dir)
    fig_name = out_dir / "fig1_class_distribution"
    plt.savefig(f"{fig_name}.png", dpi=300)
    plt.savefig(f"{fig_name}.pdf")
    plt.close()
    
    # Save source data JSON
    data_json = {k: int(v) for k, v in counts.items()}
    with open(out_dir / "fig1_data.json", "w") as f:
        json.dump(data_json, f, indent=4)
        
    logger.info(f"Figure 1 saved to {out_dir}")

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def fig_tsne_feature_space(config: dict, out_dir: Path):
    """
    Figure 2: t-SNE/UMAP embedding showing cluster sparsity.
    """
    import torch
    from sklearn.manifold import TSNE
    from sklearn.neighbors import NearestNeighbors
    try:
        import umap
    except ImportError:
        umap = None
        
    logger.info("Generating Figure 2 (t-SNE/UMAP feature space)...")
    feat_dir = Path(config['paths']['features'])
    
    # 1. Load Data
    X_common = torch.load(feat_dir / "X_common.pt").numpy()
    X_rare = torch.load(feat_dir / "X_rare.pt").numpy()
    X_inter = torch.load(feat_dir / "X_intermediate.pt").numpy()
    
    # Sample for speed if too large
    X = np.concatenate([X_common[:500], X_inter[:500], X_rare[:500]])
    y = np.array(['Common']*500 + ['Intermediate']*500 + ['Rare']*500)
    
    # 2. Fit t-SNE
    logger.info("Fitting t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_2d = tsne.fit_transform(X)
    
    # Calculate density (avg k-NN distance for rare)
    nbrs = NearestNeighbors(n_neighbors=5).fit(X_rare)
    distances, _ = nbrs.kneighbors(X_rare)
    avg_dist = np.mean(distances)
    
    # 3. Plot
    sns.set_theme(style='white', font_scale=1.1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    
    # Panel A: 5-class coloring (simulation with grouped labels)
    classes = ['Common', 'Intermediate', 'Rare']
    colors = ['#001f3f', '#4682b4', '#ff4136']
    markers = ['o', 's', '*']
    
    for i, cls in enumerate(classes):
        mask = y == cls
        ax1.scatter(X_2d[mask, 0], X_2d[mask, 1], c=colors[i], label=cls, 
                    marker=markers[i], alpha=0.6, edgecolors='w', s=60)
    
    ax1.legend(title="Cell Category")
    ax1.set_title("A. t-SNE Latent Manifold", fontweight='bold')
    
    # Panel B: Binary + Density Contours
    rare_mask = y == 'Rare'
    common_mask = y != 'Rare'
    
    ax2.scatter(X_2d[common_mask, 0], X_2d[common_mask, 1], c='lightgray', s=20, alpha=0.3, label='Common/Inter')
    ax2.scatter(X_2d[rare_mask, 0], X_2d[rare_mask, 1], c='#ff4136', s=50, alpha=0.8, label='Rare')
    
    # Density contours for rare
    sns.kdeplot(x=X_2d[rare_mask, 0], y=X_2d[rare_mask, 1], ax=ax2, levels=5, color='black', alpha=0.3)
    
    ax2.set_title("B. Cluster Density (Rare Class)", fontweight='bold')
    ax2.text(0.05, 0.05, f'Rare cluster density (avg-dist): {avg_dist:.2f}', 
             transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.suptitle("Feature Space Visualization — Rare Cells form Sparse Clusters", fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save
    fig_name = out_dir / "fig2_feature_space"
    plt.savefig(f"{fig_name}.png", dpi=300)
    plt.savefig(f"{fig_name}.pdf")
    
    # 4. UMAP Variant
    if umap:
        logger.info("Fitting UMAP...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        X_umap = reducer.fit_transform(X)
        
        plt.figure(figsize=(7, 6))
        for i, cls in enumerate(classes):
            mask = y == cls
            plt.scatter(X_umap[mask, 0], X_umap[mask, 1], c=colors[i], label=cls, marker=markers[i], alpha=0.6)
        plt.title("UMAP Visualization of Feature Space", fontweight='bold')
        plt.legend()
        plt.savefig(out_dir / "fig2b_umap_feature_space.pdf")
        
    plt.close('all')
    logger.info("Figure 2 pipeline complete.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures", type=str, default="1,2")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    
    config = load_yaml(Path(args.config))
    manifest_path = Path(config['paths']['processed_data']) / "manifest.csv"
    out_dir = Path(config['paths']['figures'])
    
    selected = [f.strip() for f in args.figures.split(",")]
    
    if "1" in selected:
        fig_class_distribution(manifest_path, out_dir)
    if "2" in selected:
        fig_tsne_feature_space(config, out_dir)

def load_yaml(path: Path):
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    main()
