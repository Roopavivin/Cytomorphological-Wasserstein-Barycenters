import random
import os
import yaml
import torch
import numpy as np
from loguru import logger
from typing import Dict, Any

def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed for all necessary libraries to ensure reproducibility.

    Args:
        seed (int): The random seed to set. Defaults to 42.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seeds fully initialized to {seed}.")

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        Dict[str, Any]: The loaded configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded configuration from {config_path}")
    return config
