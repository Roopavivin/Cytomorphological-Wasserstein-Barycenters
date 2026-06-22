"""
Dual-mode ResNet-18 Classifier for WCB-OT.
Supports training on raw images or PCA feature vectors.
"""

import torch
import torch.nn as nn
import timm
from typing import Optional, Literal

class Resnet18Classifier(nn.Module):
    """
    Unified classifier for image-based and feature-based training.
    """
    def __init__(
        self, 
        num_classes: int = 5, 
        feature_dim: int = 128, 
        mode: Literal['image', 'feature'] = 'image'
    ):
        super().__init__()
        self.mode = mode
        
        if mode == 'image':
            # Pretrained ResNet-18 for image mode
            self.net = timm.create_model('resnet18', pretrained=True, num_classes=num_classes)
        else:
            # 3-layer MLP for feature-space classification
            # Structure: Input -> 512 -> 256 -> 128 -> Output
            self.net = nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def save_checkpoint(self, path: str):
        torch.save({
            'state_dict': self.state_dict(),
            'mode': self.mode,
            'config': {
                'num_classes': 5,
                'feature_dim': self.net[0].in_features if self.mode == 'feature' else None
            }
        }, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: str = 'cpu'):
        ckpt = torch.load(path, map_location=device)
        model = cls(
            num_classes=ckpt['config']['num_classes'],
            feature_dim=ckpt['config']['feature_dim'] or 128,
            mode=ckpt['mode']
        )
        model.load_state_dict(ckpt['state_dict'])
        return model.to(device)

if __name__ == "__main__":
    pass
