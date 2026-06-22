"""
Progressive Growing GAN (ProGAN) implementation.
Used as a multi-scale baseline for rare-cell image synthesis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)

class EqualizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.scale = (2 / in_features)**0.5
        nn.init.normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x) * self.scale

class ProGANBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=True):
        super().__init__()
        self.upsample = upsample
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.pn = PixelNorm()

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = F.leaky_relu(self.pn(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.pn(self.conv2(x)), 0.2)
        return x

class ProGANGenerator(nn.Module):
    def __init__(self, latent_dim=512, channels=[512, 512, 512, 256, 128, 64, 32]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, channels[0], 4, 1, 0), # 4x4
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels[0], channels[0], 3, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm()
        )
        
        self.blocks = nn.ModuleList()
        in_c = channels[0]
        for out_c in channels[1:]:
            self.blocks.append(ProGANBlock(in_c, out_c))
            in_c = out_c
            
        self.to_rgb = nn.Conv2d(channels[-1], 3, 1)

    def forward(self, z):
        # Initial 4x4
        x = self.initial(z.view(z.size(0), -1, 1, 1))
        # Growing blocks
        for block in self.blocks:
            x = block(x)
        return torch.tanh(self.to_rgb(x))

class ProGANBaseline:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gen = ProGANGenerator().to(self.device)

    def generate(self, n_samples=1600):
        self.gen.eval()
        samples = []
        with torch.no_grad():
            for _ in range(n_samples // 16):
                z = torch.randn(16, 512).to(self.device)
                samples.append(self.gen(z).cpu())
        return torch.cat(samples)

if __name__ == "__main__":
    pass
