import torch
import torch.nn as nn
import torch.nn.functional as F

class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=256, num_layers=4):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.extend([nn.Linear(latent_dim if _ == 0 else hidden_dim, hidden_dim), nn.LeakyReLU(0.2)])
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)

class StyledConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, style_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.style_fc = nn.Linear(style_dim, in_channels)

    def forward(self, x, style):
        s = self.style_fc(style).unsqueeze(-1).unsqueeze(-1)
        x = x * s
        return self.conv(x)

class WCBGenerator(nn.Module):
    """
    Generator influenced by StyleGAN architecture, adapted for Wasserstein Barycenter augmentation.
    """
    def __init__(self, latent_dim=256, base_channels=256, img_size=128):
        super().__init__()
        self.img_size = img_size
        self.base_channels = base_channels
        self.latent_dim = latent_dim

        self.mapping = MappingNetwork(latent_dim, latent_dim, 4)
        
        self.init_const = nn.Parameter(torch.randn(1, base_channels, 4, 4))
        
        self.blocks = nn.ModuleList([
            StyledConv2d(base_channels, base_channels, 3, latent_dim),
            StyledConv2d(base_channels, base_channels//2, 3, latent_dim),
            StyledConv2d(base_channels//2, base_channels//4, 3, latent_dim),
            StyledConv2d(base_channels//4, base_channels//8, 3, latent_dim),
            StyledConv2d(base_channels//8, 3, 3, latent_dim)
        ])
        
    def forward(self, z):
        w = self.mapping(z)
        x = self.init_const.expand(w.shape[0], -1, -1, -1)
        
        for i, block in enumerate(self.blocks):
            if i > 0 and i < len(self.blocks) - 1:
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = block(x, w)
            if i < len(self.blocks) - 1:
                x = F.leaky_relu(x, 0.2)
            else:
                x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear')
                x = torch.tanh(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, img_size=128, base_channels=64):
        super().__init__()
        # Ensure divisible by 16 as quick check
        self.net = nn.Sequential(
            nn.Conv2d(3, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels, base_channels*2, 4, 2, 1),
            nn.BatchNorm2d(base_channels*2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels*2, base_channels*4, 4, 2, 1),
            nn.BatchNorm2d(base_channels*4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels*4, base_channels*8, 4, 2, 1),
            nn.BatchNorm2d(base_channels*8),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels*8, 1)
        )
        
    def forward(self, x):
        return self.net(x)

class ResNet18Evaluator(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        from torchvision.models import resnet18
        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
