"""
Seed utility for WCB-OT project.
Ensures deterministic behavior across random, numpy, and torch.
"""

import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed for all necessary libraries to ensure reproducibility.

    Args:
        seed (int): The random seed to set. Defaults to 42.

    Returns:
        None
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
if __name__ == "__main__":
    set_seed(42)
    print(f"Seeds set to 42. Torch random: {torch.rand(1).item()}")
