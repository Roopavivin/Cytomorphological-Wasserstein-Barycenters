"""
I/O utility for WCB-OT project.
Provides helper functions for saving/loading various data formats.
"""

import yaml
import torch
import joblib
from pathlib import Path
from typing import Any, Dict

def ensure_dir(path: Path) -> Path:
    """
    Creates parent directories for a path if they do not exist.

    Args:
        path (Path): Path to a file or directory.

    Returns:
        Path: The original path.
    """
    if path.suffix:  # If it looks like a file path
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)
    return path

def save_pickle(obj: Any, path: Path) -> None:
    """
    Saves an object to a pickle file using joblib.

    Args:
        obj (Any): Object to save.
        path (Path): Destination path.
    """
    ensure_dir(path)
    joblib.dump(obj, path)

def load_pickle(path: Path) -> Any:
    """
    Loads an object from a pickle file using joblib.

    Args:
        path (Path): Path to the pickle file.

    Returns:
        Any: The loaded object.
    """
    return joblib.load(path)

def save_yaml(obj: Dict[str, Any], path: Path) -> None:
    """
    Saves a dictionary to a YAML file.

    Args:
        obj (Dict[str, Any]): Dictionary to save.
        path (Path): Destination path.
    """
    ensure_dir(path)
    with open(path, 'w') as f:
        yaml.safe_dump(obj, f)

def load_yaml(path: Path) -> Dict[str, Any]:
    """
    Loads a dictionary from a YAML file.

    Args:
        path (Path): Path to the YAML file.

    Returns:
        Dict[str, Any]: The loaded dictionary.
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_tensor(t: torch.Tensor, path: Path) -> None:
    """
    Saves a PyTorch tensor to a file.

    Args:
        t (torch.Tensor): Tensor to save.
        path (Path): Destination path (usually .pt).
    """
    ensure_dir(path)
    torch.save(t, path)

def load_tensor(path: Path) -> torch.Tensor:
    """
    Loads a PyTorch tensor from a file.

    Args:
        path (Path): Path to the saved tensor.

    Returns:
        torch.Tensor: The loaded tensor.
    """
    return torch.load(path, weights_only=True)

if __name__ == "__main__":
    pass
