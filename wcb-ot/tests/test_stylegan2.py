"""
Smoke tests for StyleGAN2 implementation.
"""

import torch
import pytest
from src.models.stylegan2 import StyleGAN2Generator

def test_stylegan2_forward():
    model = StyleGAN2Generator(style_dim=64, latent_dim=64, channels=[64, 64, 32, 16])
    z = torch.randn(2, 64)
    out = model(z)
    # Since resolution decreases by layers, we need to check output size
    # Const: 4x4. 3 blocks -> 8x8 -> 16x16 -> 32x32
    assert out.shape == (2, 3, 32, 32)
    assert torch.all(out >= -1.05) and torch.all(out <= 1.05)

def test_stylegan2_mapping():
    from src.models.stylegan2 import MappingNetwork
    net = MappingNetwork(32, 32, num_layers=4)
    z = torch.randn(5, 32)
    s = net(z)
    assert s.shape == (5, 32)

if __name__ == "__main__":
    pass
