"""
Unit tests for the classification trainer.
"""

import torch
import pytest
import numpy as np
from src.models.resnet18 import Resnet18Classifier
from src.train.train_classifier import UnifiedDataset, get_class_weights

def test_classifier_modes():
    # Image mode
    model_img = Resnet18Classifier(num_classes=5, mode='image')
    img = torch.randn(1, 3, 256, 256)
    out = model_img(img)
    assert out.shape == (1, 5)
    
    # Feature mode
    model_feat = Resnet18Classifier(num_classes=5, feature_dim=128, mode='feature')
    feat = torch.randn(1, 128)
    out = model_feat(feat)
    assert out.shape == (1, 5)

def test_dataset_loading():
    X = torch.randn(10, 128)
    y = torch.randint(0, 5, (10,))
    ds = UnifiedDataset(X, y, mode='feature')
    assert len(ds) == 10
    xi, yi = ds[0]
    assert xi.shape == (128,)
    assert isinstance(yi, torch.Tensor)

def test_class_weighting():
    labels = torch.tensor([0, 0, 1, 2])
    weights = get_class_weights(labels)
    # Class 0 has 2 samples, higher weight for 1 and 2
    assert weights[0] < weights[1]
    assert torch.allclose(weights.sum(), torch.tensor(1.0))

if __name__ == "__main__":
    pass
