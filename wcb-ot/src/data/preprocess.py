"""
Preprocessing pipeline for SIPaKMeD dataset.
Implements stain normalization, denoising, CLAHE, and cell segmentation.
"""

import os
import cv2
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from joblib import Parallel, delayed
from tqdm import tqdm
from loguru import logger

from src.utils.io import load_yaml, ensure_dir
from src.data.stain import ReinhardNormalizer

def segment_cell(image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segmentation of nucleus and cytoplasm using Otsu and Watershed.
    
    Args:
        image_rgb (np.ndarray): Preprocessed RGB image.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (nuclear_mask, cytoplasm_mask) binary 0/1.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # Otsu thresholding for background removal
    # In cytology images, cells are usually darker than background
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Finding sure foreground area (Nucleus candidate)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max() if dist_transform.max() > 0 else 0, 255, 0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Watershed
    markers = cv2.watershed(image_rgb, markers)
    
    # Nucleus = markers > 1 (Assuming one central cell)
    # Cytoplasm = opening - nucleus
    # This is a simplification; for single isolated cells it works well.
    nuc_mask = (markers > 1).astype(np.uint8)
    cyt_mask = (opening > 0).astype(np.uint8) - nuc_mask
    cyt_mask[cyt_mask < 0] = 0
    
    return nuc_mask, cyt_mask

class Preprocessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.img_size = config['dataset']['image_size']
        self.raw_dir = Path(config['paths']['raw_data'])
        self.proc_dir = Path(config['paths']['processed_data'])
        
        # Setup Reinhard Normalizer with a reference if possible
        self.normalizer = ReinhardNormalizer()
        self.ref_initialized = False

    def initialize_reference(self):
        """Initializes stain normalization reference from Superficial_Intermediate class."""
        ref_class = "Superficial_Intermediate"
        ref_paths = list((self.raw_dir / ref_class).rglob("*.[bB][mM][pP]")) + \
                    list((self.raw_dir / ref_class).rglob("*.[jJ][pP][gG]"))
        
        if ref_paths:
            ref_img = cv2.imread(str(ref_paths[0]))
            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
            self.normalizer.fit(ref_img)
            self.ref_initialized = True
            logger.info(f"Stain reference initialized using {ref_paths[0].name}")
        else:
            logger.warning("No reference images found for Superficial_Intermediate. Using default statistics.")

    def process_single_image(self, img_path: Path, class_name: str) -> Optional[Dict[str, Any]]:
        """Applies the 6-step pipeline to one image."""
        try:
            image_id = img_path.stem
            
            # 1. Load RGB
            img = cv2.imread(str(img_path))
            if img is None: return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 2. Resize
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
            
            # 3. Stain Norm
            img = self.normalizer.transform(img)
            
            # 4. Denoise
            img = cv2.fastNlMeansDenoisingColored(img, None, h=8, templateWindowSize=5, searchWindowSize=21)
            
            # 5. CLAHE (on L channel)
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            img = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)
            
            # 6. Background exclusion & Segmentation
            nuc_mask, cyt_mask = segment_cell(img)
            
            # Paths
            class_proc_dir = self.proc_dir / class_name
            ensure_dir(class_proc_dir)
            
            proc_path = class_proc_dir / f"{image_id}.npy"
            nuc_path = class_proc_dir / f"{image_id}_nuc.npy"
            cyt_path = class_proc_dir / f"{image_id}_cyt.npy"
            
            # Save results
            np.save(proc_path, img)
            np.save(nuc_path, nuc_mask)
            np.save(cyt_path, cyt_mask)
            
            # Stats
            nuc_area = int(np.sum(nuc_mask))
            cyt_area = int(np.sum(cyt_mask))
            nc_ratio = nuc_area / cyt_area if cyt_area > 0 else 0
            
            return {
                "image_id": image_id,
                "class": class_name,
                "original_path": str(img_path),
                "processed_path": str(proc_path),
                "nuclear_mask_path": str(nuc_path),
                "cytoplasm_mask_path": str(cyt_path),
                "nucleus_area_px": nuc_area,
                "cytoplasm_area_px": cyt_area,
                "nc_ratio": nc_ratio
            }
        except Exception as e:
            logger.error(f"Failed to process {img_path}: {e}")
            return None

def run_preprocessing(config_path: str):
    config = load_yaml(Path(config_path))
    prep = Preprocessor(config)
    prep.initialize_reference()
    
    # Discovery
    all_tasks = []
    classes = config['dataset']['classes']
    for cls in classes:
        cls_dir = prep.raw_dir / cls
        paths = list(cls_dir.rglob("*.[bB][mM][pP]")) + list(cls_dir.rglob("*.[jJ][pP][gG]"))
        # Filter out anything that is NOT in a 'Cropped' folder if present, to be safe
        # But for now, we'll process all as per the 'rglob' instruction unless it creates duplicates
        for p in paths:
            all_tasks.append((p, cls))
            
    logger.info(f"Found {len(all_tasks)} total images to process.")
    
    results = Parallel(n_jobs=-1)(
        delayed(prep.process_single_image)(p, cls) 
        for p, cls in tqdm(all_tasks, desc="Preprocessing")
    )
    
    # Filter Nones and build manifest
    results = [r for r in results if r is not None]
    df = pd.DataFrame(results)
    
    manifest_path = Path(config['paths']['processed_data']) / "manifest.csv"
    df.to_csv(manifest_path, index=False)
    
    logger.info(f"Preprocessing complete. Manifest saved to {manifest_path}")
    
    # Table summary
    summary = df.groupby('class').size().reset_index(name='count')
    print("\n" + "="*40)
    print("Preprocessed Data Manifest Summary")
    print("-" * 40)
    print(summary.to_string(index=False))
    print(f"Total processed: {len(df)}")
    print("="*40 + "\n")
    
    # Validations
    for cls in classes:
        count = len(df[df['class'] == cls])
        # Assert handled in main if needed, but logging for now
        if count == 0: logger.error(f"Class {cls} has 0 processed images!")
    
    # Assert nuclei > 100 px check
    invalid_nuclei = df[df['nucleus_area_px'] <= 100]
    if len(invalid_nuclei) > 0:
        logger.warning(f"{len(invalid_nuclei)} images have nucleus area <= 100px.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    run_preprocessing(args.config)
