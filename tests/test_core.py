import os
import torch
import numpy as np
import pytest
from src.utils.core import set_seed, load_config

def test_seed_reproducibility() -> None:
    """
    Tests if setting the seed results in deterministic outputs 
    for NumPy and PyTorch.
    """
    set_seed(42)
    val1_np = np.random.rand(5)
    val1_pt = torch.rand(5)
    
    set_seed(42)
    val2_np = np.random.rand(5)
    val2_pt = torch.rand(5)
    
    np.testing.assert_array_almost_equal(val1_np, val2_np)
    assert torch.allclose(val1_pt, val2_pt)

def test_load_config(tmp_path: str) -> None:
    """
    Tests if a YAML config is loaded correctly.
    """
    import yaml
    config_file = os.path.join(tmp_path, "dummy.yaml")
    dummy_data = {"test_key": "test_value"}
    with open(config_file, 'w') as f:
        yaml.dump(dummy_data, f)
        
    loaded = load_config(config_file)
    assert loaded["test_key"] == "test_value"
