"""
Simplified StyleGAN2 Implementation in PyTorch.
Used as a pixel-space baseline for rare-cell synthesis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MappingNetwork(nn.Module):
    def __init__(self, latent_dim, style_dim, num_layers=8):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(latent_dim if i == 0 else style_dim, style_dim))
            layers.append(nn.LeakyReLU(0.2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ModulatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, style_dim, demodulate=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.style_proj = nn.Linear(style_dim, in_channels)
        self.demodulate = demodulate

    def forward(self, x, style):
        # x: [b, c, h, w], style: [b, s]
        s = self.style_proj(style).view(-1, x.size(1), 1, 1) + 1.0
        x = x * s
        if self.demodulate:
            # Simplified demodulation
            pass 
        return self.conv(x)

class GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.conv1 = ModulatedConv2d(in_channels, out_channels, 3, style_dim)
        self.conv2 = ModulatedConv2d(out_channels, out_channels, 3, style_dim)
        
    def forward(self, x, style):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv1(x, style)
        x = F.leaky_relu(x, 0.2)
        x = self.conv2(x, style)
        x = F.leaky_relu(x, 0.2)
        return x

class StyleGAN2Generator(nn.Module):
    def __init__(self, style_dim=512, latent_dim=512, channels=[512, 512, 512, 256, 128, 64, 32]):
        super().__init__()
        self.mapping = MappingNetwork(latent_dim, style_dim)
        self.const = nn.Parameter(torch.randn(1, channels[0], 4, 4))
        
        self.blocks = nn.ModuleList()
        in_c = channels[0]
        for out_c in channels[1:]:
            self.blocks.append(GeneratorBlock(in_c, out_c, style_dim, upsample=True))
            in_c = out_c
            
        self.to_rgb = nn.Conv2d(channels[-1], 3, 1)

    def forward(self, z):
        style = self.mapping(z)
        x = self.const.repeat(z.size(0), 1, 1, 1)
        for block in self.blocks:
            x = block(x, style)
        return torch.tanh(self.to_rgb(x))

class StyleGAN2Baseline:
    """Wrapper for training and sampling StyleGAN2."""
    def __init__(self, config):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gen = StyleGAN2Generator().to(self.device)
        self.opt = torch.optim.Adam(self.gen.parameters(), lr=2e-4)

    def train_step(self, images):
        # Simplified train step (just generator side for demo/infra)
        z = torch.randn(images.size(0), 512).to(self.device)
        fake = self.gen(z)
        # In actual training, would involve discriminator + losses
        return fake

    def generate(self, n_samples=1600):
        self.gen.eval()
        samples = []
        with torch.no_grad():
            for _ in range(n_samples // 8):
                z = torch.randn(8, 512).to(self.device)
                fake = self.gen(z)
                samples.append(fake.cpu())
        return torch.cat(samples)

if __name__ == "__main__":
    pass
