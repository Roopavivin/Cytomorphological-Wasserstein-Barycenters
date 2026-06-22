"""
Lightweight decoder to map feature space back for qualitative visualization.
"""

import torch
import torch.nn as nn

class FeatureDecoder(nn.Module):
    def __init__(self, feature_dim=128, output_size=256):
        super().__init__()
        self.output_size = output_size
        
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 8 * 8 * 256),
            nn.ReLU(True)
        )
        
        self.decoder = nn.Sequential(
            # 8x8
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), # 128x128
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 4, 2, 1), # 256x256
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, 8, 8)
        x = self.decoder(x)
        return x

def train_decoder(model, features, images, epochs=10, batch_size=32):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    for epoch in range(epochs):
        for i in range(0, len(features), batch_size):
            batch_f = features[i:i+batch_size]
            batch_img = images[i:i+batch_size]
            
            optimizer.zero_grad()
            out = model(batch_f)
            loss = criterion(out, batch_img)
            loss.backward()
            optimizer.step()
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    return model
