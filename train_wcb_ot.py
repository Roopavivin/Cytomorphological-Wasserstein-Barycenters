import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.networks import WCBGenerator, Discriminator
from datasets.sipakmed import get_dataloaders
from utils.ot_math import entropic_wasserstein_loss

class WCB_OT_Trainer:
    def __init__(self, root_dir, batch_size=16, latent_dim=256, img_size=128, device='cuda', epochs=100):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.epochs = epochs
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        
        self.generator = WCBGenerator(latent_dim=latent_dim, img_size=img_size).to(self.device)
        self.discriminator = Discriminator(img_size=img_size).to(self.device)
        
        self.opt_g = optim.Adam(self.generator.parameters(), lr=1e-4, betas=(0.0, 0.99))
        self.opt_d = optim.Adam(self.discriminator.parameters(), lr=1e-4, betas=(0.0, 0.99))
        
        # We only train generator on Rare Classes to implicitly model their distribution
        self.train_loader, _ = get_dataloaders(root_dir, batch_size, img_size, rare_only=True)

    def train(self):
        print(f"Starting WCB-OT Training on {self.device}...")
        self.generator.train()
        self.discriminator.train()
        
        for epoch in range(self.epochs):
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            for real_imgs, _ in pbar:
                real_imgs = real_imgs.to(self.device)
                b_size = real_imgs.size(0)
                
                # Setup labels based on whether it's RaGAN or simple LSGAN
                # Using simple LSGAN for stability
                
                # ---------- Train Discriminator ----------
                self.opt_d.zero_grad()
                z = torch.randn(b_size, self.latent_dim, device=self.device)
                fake_imgs = self.generator(z)
                
                real_pred = self.discriminator(real_imgs)
                fake_pred = self.discriminator(fake_imgs.detach())
                
                d_loss_real = F.mse_loss(real_pred, torch.ones_like(real_pred))
                d_loss_fake = F.mse_loss(fake_pred, torch.zeros_like(fake_pred))
                d_loss = (d_loss_real + d_loss_fake) * 0.5
                
                d_loss.backward()
                self.opt_d.step()
                
                # ---------- Train Generator ----------
                self.opt_g.zero_grad()
                fake_imgs = self.generator(z)
                fake_pred = self.discriminator(fake_imgs)
                
                # Adversarial Loss
                g_adv_loss = F.mse_loss(fake_pred, torch.ones_like(fake_pred))
                
                # Entropic Optimal Transport Loss (Wasserstein Sinkhorn)
                # Provably bounds the distribution divergence
                sinkhorn_loss = entropic_wasserstein_loss(fake_imgs, real_imgs, blur=0.05)
                
                g_loss = g_adv_loss + 10.0 * sinkhorn_loss
                
                g_loss.backward()
                self.opt_g.step()
                
                pbar.set_postfix({'D_loss': f"{d_loss.item():.4f}", 'G_adv': f"{g_adv_loss.item():.4f}", 'OT_Loss': f"{sinkhorn_loss.item():.4f}"})
                
        # Save theoretical guarantee checkpoint model
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(self.generator.state_dict(), 'checkpoints/wcb_generator.pth')
        print("Training Complete. Model saved to checkpoints/wcb_generator.pth")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='d:/C_Research works/Rupa/WCB-OT/Datasets')
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    
    trainer = WCB_OT_Trainer(args.root_dir, epochs=args.epochs)
    trainer.train()
