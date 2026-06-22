"""
Conditional Variational Autoencoder (CVAE) for rare-cell synthesis.
Conditions on class labels to sample specific morphologies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim=128, num_classes=5, img_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.label_emb = nn.Embedding(num_classes, 64)
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, 2, 1), # 128
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # 64
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), # 32
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1), # 16
            nn.ReLU(),
            nn.Flatten()
        )
        # 512 * 16 * 16 = 131072
        # We add label embedding to the input of mu/var? 
        # Or concat after flattening.
        self.fc_mu = nn.Linear(131072 + 64, latent_dim)
        self.fc_var = nn.Linear(131072 + 64, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim + 64, 131072)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (512, 16, 16)),
            nn.ConvTranspose2d(512, 256, 4, 2, 1), # 32
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 64
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), # 128
            nn.ReLU(),
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1), # 256
            nn.Sigmoid()
        )

    def encode(self, x, labels):
        h = self.encoder(x)
        l = self.label_emb(labels)
        h_l = torch.cat([h, l], dim=1)
        return self.fc_mu(h_l), self.fc_var(h_l)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, labels):
        l = self.label_emb(labels)
        z_l = torch.cat([z, l], dim=1)
        h = self.decoder_input(z_l)
        return self.decoder(h)

    def forward(self, x, labels):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, labels), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        # Recon loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        # KL Divergence: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kld

if __name__ == "__main__":
    pass
