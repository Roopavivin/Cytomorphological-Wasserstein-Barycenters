"""
Download and extract the SIPaKMeD dataset.
Supports RAR (official) and 7z (user provided) formats.
"""

import os
import sys
import json
import hashlib
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import requests
from tqdm import tqdm
from loguru import logger

# Try to import extraction libraries
try:
    import rarfile
except ImportError:
    rarfile = None
try:
    import py7zr
except ImportError:
    py7zr = None

from src.utils.io import load_yaml, ensure_dir

# Expected image counts per class
EXPECTED_COUNTS = {
    "Superficial_Intermediate": 1200,
    "Parabasal": 1150,
    "Koilocytotic": 950,
    "Dyskeratotic": 450,
    "Metaplastic": 299
}

CLASS_MAP = {
    "im_Superficial-Intermediate": "Superficial_Intermediate",
    "im_Parabasal": "Parabasal",
    "im_Koilocytotic": "Koilocytotic",
    "im_Dyskeratotic": "Dyskeratotic",
    "im_Metaplastic": "Metaplastic"
}

BASE_URL = "https://www.cs.uoi.gr/~marina/sipakmed"

def compute_sha256(file_path: Path) -> str:
    """Computes SHA-256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def download_file(url: str, dest_path: Path) -> bool:
    """Downloads a file with a progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as f, tqdm(
            desc=dest_path.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
        return True
    except Exception as e:
        logger.error(f"Download failed for {url}: {e}")
        return False

def extract_archive(archive_path: Path, dest_dir: Path) -> None:
    """Extracts .rar or .7z archives."""
    if archive_path.suffix == ".rar":
        if rarfile is None:
            raise ImportError("rarfile package not installed. Required for .rar extraction.")
        with rarfile.RarFile(archive_path) as rf:
            rf.extractall(dest_dir)
    elif archive_path.suffix == ".7z":
        if py7zr is None:
            raise ImportError("py7zr package not installed. Required for .7z extraction.")
        with py7zr.SevenZipFile(archive_path, mode='r') as sz:
            sz.extractall(dest_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.suffix}")

def main(config_path: str = 'configs/config.yaml') -> None:
    """Main download and extraction logic."""
    # Ensure logs directory exists for the custom log path
    ensure_dir(Path("results/logs/download.log"))
    logger.add("results/logs/download.log", rotation="10 MB")
    
    logger.info(f"Starting SIPaKMeD dataset process using config: {config_path}")
    config = load_yaml(Path(config_path))
    raw_dir = Path(config['paths']['raw_data'])
    ensure_dir(raw_dir)
    
    # Path where the user mentioned they have the dataset
    user_dataset_dir = Path("D:/C_Research works/Rupa/WCB-OT/Datasets")
    
    checksums = {}
    summary = []

    for internal_name, class_name in CLASS_MAP.items():
        dest_class_dir = raw_dir / class_name
        ensure_dir(dest_class_dir)
        
        # Check for local files first (User provided .7z)
        local_7z = user_dataset_dir / f"{internal_name}.7z"
        target_archive = raw_dir / f"{internal_name}.rar" # Default target if downloading
        
        if local_7z.exists():
            logger.info(f"Found local archive for {class_name}: {local_7z}")
            target_archive = raw_dir / f"{internal_name}.7z"
            if not target_archive.exists():
                shutil.copy(local_7z, target_archive)
        
        # If no local or target archive, try to download RAR
        if not target_archive.exists():
            url = f"{BASE_URL}/{internal_name}.rar"
            logger.info(f"Downloading {url} ...")
            if not download_file(url, target_archive):
                print(f"\n[ERROR] Official URL {url} is unavailable.")
                print("[FALLBACK] Please manual download from Kaggle: https://www.kaggle.com/datasets/paultimothymooney/cervical-cytology-black-and-white")
                print(f"Place the archive files in {raw_dir}")
                sys.exit(0)
        
        # Checksum
        logger.info(f"Computing checksum for {target_archive.name}...")
        checksums[target_archive.name] = compute_sha256(target_archive)
        
        # Extraction
        logger.info(f"Extracting {target_archive.name} to {dest_class_dir}...")
        extract_archive(target_archive, dest_class_dir)
        
        # Count images (recursively find bmp/jpg)
        img_count = len(list(dest_class_dir.rglob("*.[bB][mM][pP]"))) + \
                    len(list(dest_class_dir.rglob("*.[jJ][pP][gG]")))
        
        expected = EXPECTED_COUNTS[class_name]
        match = "Yes" if img_count == expected else "No"
        summary.append({"Class": class_name, "Count": img_count, "Expected": expected, "Match": match})
        
        if img_count != expected:
            logger.warning(f"Count mismatch for {class_name}: Found {img_count}, expected {expected}")

    # Save checksums
    with open(raw_dir / "checksums.json", "w") as f:
        json.dump(checksums, f, indent=4)
        
    # Summary Table
    print("\n" + "="*60)
    print(f"{'Class':<30} | {'Count':<8} | {'Expected':<8} | {'Match?':<6}")
    print("-" * 60)
    for row in summary:
        print(f"{row['Class']:<30} | {row['Count']:<8} | {row['Expected']:<8} | {row['Match']:<6}")
    print("="*60 + "\n")
    
    logger.info("Dataset download and extraction complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SIPaKMeD dataset.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
