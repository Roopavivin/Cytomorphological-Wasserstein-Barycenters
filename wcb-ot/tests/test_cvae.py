"""
Smoke tests for Conditional VAE.
"""

import torch
import pytest
from src.models.cvae import ConditionalVAE

def test_cvae_forward():
    # Use smaller scale for test
    model = ConditionalVAE(latent_dim=32)
    # Mock image 128x128
    x = torch.randn(2, 3, 256, 256)
    labels = torch.randint(0, 5, (2,))
    
    recon, mu, logvar = model(x, labels)
    assert recon.shape == (2, 3, 256, 256)
    assert mu.shape == (2, 32)
    assert logvar.shape == (2, 32)

def test_cvae_sampling():
    model = ConditionalVAE(latent_dim=32)
    z = torch.randn(5, 32)
    labels = torch.ones(5, dtype=torch.long)
    samples = model.decode(z, labels)
    assert samples.shape == (5, 3, 256, 256)
    assert torch.all(samples >= 0.0) and torch.all(samples <= 1.0) # Sigmoid

def test_cvae_loss():
    model = ConditionalVAE(latent_dim=16)
    x = torch.randn(2, 3, 256, 256)
    labels = torch.zeros(2, dtype=torch.long)
    recon, mu, logvar = model(x, labels)
    loss = model.loss_function(recon, x, mu, logvar)
    assert loss.item() > 0

if __name__ == "__main__":
    pass
