"""
Tests for figure generation.
"""

import os
from pathlib import Path

def test_fig1_exists():
    out_dir = Path("results/figures")
    png = out_dir / "fig1_class_distribution.png"
    pdf = out_dir / "fig1_class_distribution.pdf"
    json_data = out_dir / "fig1_data.json"
    
    assert png.exists()
    assert pdf.exists()
    assert json_data.exists()
    assert png.stat().st_size > 1000
    assert pdf.stat().st_size > 1000

if __name__ == "__main__":
    pass
