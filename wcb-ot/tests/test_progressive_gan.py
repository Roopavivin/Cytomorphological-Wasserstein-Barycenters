"""
Smoke tests for Progressive GAN implementation.
"""

import torch
import pytest
from src.models.progressive_gan import ProGANGenerator

def test_progan_forward():
    # Latent dim 64, small channels for test
    model = ProGANGenerator(latent_dim=64, channels=[64, 64, 32, 16])
    z = torch.randn(2, 64)
    out = model(z)
    # Architecture: Initial(4x4) + 3 blocks(upsample x2)
    # 4 -> 8 -> 16 -> 32
    assert out.shape == (2, 3, 32, 32)

def test_pixel_norm():
    from src.models.progressive_gan import PixelNorm
    pn = PixelNorm()
    x = torch.randn(2, 64, 4, 4) * 10
    out = pn(x)
    # Mean of squares should be ~1
    ms = torch.mean(out**2, dim=1)
    assert torch.allclose(ms, torch.ones_like(ms), atol=1e-3)

if __name__ == "__main__":
    pass
