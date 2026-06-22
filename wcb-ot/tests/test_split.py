"""
Unit tests for data splitting.
"""

import os
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.data.split import save_split_indices

def test_disjoint_splits(tmp_path: Path):
    """Verify train, val, and test indices are completely disjoint."""
    # Create dummy manifest
    manifest_path = tmp_path / "manifest.csv"
    data = {'image_id': range(100), 'class': ['A']*50 + ['B']*50}
    pd.DataFrame(data).to_csv(manifest_path, index=False)
    
    split_dir = tmp_path / "splits"
    ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
    
    df = save_split_indices(manifest_path, split_dir, 42, ratios)
    
    train = np.load(split_dir / "train_idx.npy")
    val = np.load(split_dir / "val_idx.npy")
    test = np.load(split_dir / "test_idx.npy")
    
    intersection = set(train) & set(val) | set(val) & set(test) | set(train) & set(test)
    assert len(intersection) == 0
    assert len(train) + len(val) + len(test) == 100

def test_ratio_preservation(tmp_path: Path):
    """Verify class ratios remain within 2% margin in train split."""
    manifest_path = tmp_path / "manifest.csv"
    # 80/20 split
    data = {'image_id': range(1000), 'class': ['A']*800 + ['B']*200}
    pd.DataFrame(data).to_csv(manifest_path, index=False)
    
    ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
    df = save_split_indices(manifest_path, tmp_path / "splits", 42, ratios)
    
    train_df = df[df['split'] == 'train']
    ratio_a = len(train_df[train_df['class'] == 'A']) / len(train_df)
    assert abs(ratio_a - 0.8) < 0.02

def test_reproducibility(tmp_path: Path):
    """Verify same seed produces identical numpy arrays."""
    manifest_path = tmp_path / "manifest.csv"
    data = {'image_id': range(100), 'class': ['A']*50 + ['B']*50}
    pd.DataFrame(data).to_csv(manifest_path, index=False)
    
    ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
    
    save_split_indices(manifest_path, tmp_path / "s1", 42, ratios)
    save_split_indices(manifest_path, tmp_path / "s2", 42, ratios)
    
    s1 = np.load(tmp_path / "s1" / "train_idx.npy")
    s2 = np.load(tmp_path / "s2" / "train_idx.npy")
    assert np.array_equal(s1, s2)

if __name__ == "__main__":
    pass
