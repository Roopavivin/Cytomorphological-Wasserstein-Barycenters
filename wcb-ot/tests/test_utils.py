"""
Unit tests for utility modules.
"""

import os
import torch
import random
import numpy as np
import pytest
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.utils.io import (
    ensure_dir, save_pickle, load_pickle, 
    save_yaml, load_yaml, save_tensor, load_tensor
)

def test_get_logger(tmp_path: Path):
    """Test logger initialization and file creation."""
    log_file = tmp_path / "test.log"
    logger = get_logger("test_proc", log_file=log_file)
    logger.info("Test message")
    assert log_file.exists()
    content = log_file.read_text()
    assert "Test message" in content

def test_set_seed():
    """Test if seed sets deterministic values for all libraries."""
    set_seed(42)
    val_rand = random.random()
    val_np = np.random.rand()
    val_pt = torch.rand(1).item()
    
    set_seed(42)
    assert val_rand == random.random()
    assert val_np == np.random.rand()
    assert val_pt == torch.rand(1).item()

def test_set_seed_hash():
    """Test if PYTHONHASHSEED is set."""
    set_seed(100)
    assert os.environ['PYTHONHASHSEED'] == '100'

def test_ensure_dir_file(tmp_path: Path):
    """Test ensure_dir creating parents for a file path."""
    target_file = tmp_path / "level1" / "level2" / "test.txt"
    ensure_dir(target_file)
    assert (tmp_path / "level1" / "level2").exists()

def test_ensure_dir_folder(tmp_path: Path):
    """Test ensure_dir creating a directory path."""
    target_dir = tmp_path / "new_folder"
    ensure_dir(target_dir)
    assert target_dir.exists()

def test_save_load_pickle(tmp_path: Path):
    """Test joblib save/load consistency."""
    path = tmp_path / "data.joblib"
    data = {"array": np.array([1, 2, 3]), "text": "hello"}
    save_pickle(data, path)
    assert path.exists()
    loaded = load_pickle(path)
    np.testing.assert_array_equal(data["array"], loaded["array"])
    assert data["text"] == loaded["text"]

def test_save_load_yaml(tmp_path: Path):
    """Test YAML save/load consistency."""
    path = tmp_path / "config.yaml"
    config = {"params": {"lr": 0.01, "epochs": 10}, "tags": ["rare", "ot"]}
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    assert path.exists()
    loaded = load_yaml(path)
    assert loaded["params"]["lr"] == 0.01
    assert loaded["tags"][0] == "rare"

def test_save_load_tensor(tmp_path: Path):
    """Test tensor save/load consistency."""
    path = tmp_path / "model.pt"
    tensor = torch.randn(3, 3)
    save_tensor(tensor, path)
    assert path.exists()
    loaded = load_tensor(path)
    assert torch.equal(tensor, loaded)

def test_save_pickle_creates_dir(tmp_path: Path):
    """Test save_pickle recursively creates missing directories."""
    path = tmp_path / "nested" / "output.pkl"
    save_pickle(123, path)
    assert path.exists()

def test_save_yaml_creates_dir(tmp_path: Path):
    """Test save_yaml recursively creates missing directories."""
    path = tmp_path / "deep" / "config.yml"
    save_yaml({"a": 1}, path)
    assert path.exists()

def test_save_tensor_creates_dir(tmp_path: Path):
    """Test save_tensor recursively creates missing directories."""
    path = tmp_path / "models" / "weights.pt"
    lab = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB).astype(np.float32)
    save_tensor(torch.zeros(1), path)
    assert path.exists()

def test_logger_default_path():
    """Test if logger creates default log directory."""
    # This might create a results/logs folder in the project root
    logger = get_logger("cleanup_test")
    logger.info("Checking default log creation")
    # Clean up is hard without knowing exact timestamp, but we can check if dir exists
    assert Path("results/logs").exists()

if __name__ == "__main__":
    pass
