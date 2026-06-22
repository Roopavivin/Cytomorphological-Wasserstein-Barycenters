"""
Integration tests for dataset download/extraction.
"""

import os
import json
import pytest
import py7zr
from pathlib import Path
from src.data.download import compute_sha256, extract_archive

def test_checksum_consistency(tmp_path: Path):
    """Test SHA-256 calculation."""
    dummy_file = tmp_path / "test.txt"
    dummy_file.write_text("WCB-OT test content")
    
    sha1 = compute_sha256(dummy_file)
    sha2 = compute_sha256(dummy_file)
    assert sha1 == sha2
    assert len(sha1) == 64

def test_extract_7z(tmp_path: Path):
    """Test 7z extraction integration."""
    # Create a dummy 7z archive
    archive_path = tmp_path / "test.7z"
    content_dir = tmp_path / "content"
    content_dir.mkdir()
    (content_dir / "image.bmp").write_text("fake binary")
    
    with py7zr.SevenZipFile(archive_path, 'w') as archive:
        archive.writeall(content_dir, "content")
        
    # Extract
    dest_dir = tmp_path / "extracted"
    dest_dir.mkdir()
    extract_archive(archive_path, dest_dir)
    
    assert (dest_dir / "content" / "image.bmp").exists()

if __name__ == "__main__":
    pass
