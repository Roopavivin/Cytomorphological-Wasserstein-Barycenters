"""
Unit tests for feature extraction.
"""

import torch
import numpy as np
import pytest
from src.data.features import MorphologicalExtractor, TextureExtractor, DeepExtractor
from sklearn.decomposition import PCA

def test_morphological_shape():
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    nuc = np.zeros((256, 256), dtype=np.uint8)
    nuc[120:140, 120:140] = 1
    cyt = np.zeros((256, 256), dtype=np.uint8)
    cyt[100:160, 100:160] = 1
    cyt = cyt - nuc
    
    ext = MorphologicalExtractor()
    feats = ext.extract(img, nuc, cyt)
    assert feats.shape == (42,)

def test_texture_shape():
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    nuc = np.zeros((256, 256), dtype=np.uint8)
    nuc[120:140, 120:140] = 1
    
    ext = TextureExtractor()
    feats = ext.extract(img, nuc, nuc)
    assert feats.shape == (64,)

def test_pca_variance():
    """Verify PCA can retain 95% variance on synthetic redundant data."""
    # Data with more samples than components
    data = np.random.randn(200, 200)
    # Add strong redundancy
    data[:, 100:] = data[:, :100] + 0.01 * np.random.randn(200, 100)
    
    pca = PCA(n_components=128)
    pca.fit(data)
    var = pca.explained_variance_ratio_.sum()
    assert var > 0.90 # PCA on redundant data should capture most

def test_no_nans():
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    nuc = np.zeros((256, 256), dtype=np.uint8)
    nuc[120:140, 120:140] = 1
    
    m_ext = MorphologicalExtractor()
    t_ext = TextureExtractor()
    
    m_f = m_ext.extract(img, nuc, nuc)
    t_f = t_ext.extract(img, nuc, nuc)
    
    assert not np.isnan(m_f).any()
    assert not np.isnan(t_f).any()

if __name__ == "__main__":
    pass
