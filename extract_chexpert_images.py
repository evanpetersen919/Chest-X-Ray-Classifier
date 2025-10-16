#!/usr/bin/env python3
"""
Extract only the filtered CheXpert images for rare diseases
Run this AFTER downloading the full CheXpert dataset
"""

import pandas as pd
import os
import shutil
from pathlib import Path
from tqdm import tqdm

# Paths
CHEXPERT_BASE = "C:/xray_data/chexpert/CheXpert-v1.0-small"  # Where you downloaded CheXpert
FILTERED_CSV = "C:/xray_data/chexpert/filtered_rare_diseases.csv"
OUTPUT_DIR = "C:/xray_data/chexpert/rare_disease_images"

def extract_filtered_images():
    """Copy only the images we need for rare diseases"""
    
    # Load filtered list
    print(f"Loading filtered image list from: {FILTERED_CSV}")
    filtered_df = pd.read_csv(FILTERED_CSV)
    print(f"Total images to copy: {len(filtered_df):,}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Track statistics
    copied = 0
    missing = 0
    errors = []
    
    print(f"\nCopying images to: {OUTPUT_DIR}")
    print("This may take 30-60 minutes for 143K images...\n")
    
    # Copy each image
    for idx, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Copying images"):
        # Get image path from CSV (format: "CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg")
        relative_path = row['Path']
        
        # Source path
        source_path = os.path.join("C:/xray_data/chexpert", relative_path)
        
        # Destination path (flatten structure, use patient+study+view as filename)
        # Example: patient00001_study1_view1_frontal.jpg
        filename = relative_path.replace('/', '_').replace('\\', '_')
        filename = filename.replace('CheXpert-v1.0-small_train_', '')
        dest_path = os.path.join(OUTPUT_DIR, filename)
        
        try:
            if os.path.exists(source_path):
                # Copy file
                shutil.copy2(source_path, dest_path)
                copied += 1
            else:
                missing += 1
                if missing <= 10:  # Only log first 10 missing files
                    errors.append(f"Missing: {source_path}")
        except Exception as e:
            errors.append(f"Error copying {source_path}: {str(e)}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"Images copied: {copied:,}")
    print(f"Images missing: {missing:,}")
    print(f"Success rate: {(copied / len(filtered_df) * 100):.1f}%")
    
    if errors:
        print(f"\nFirst few errors:")
        for error in errors[:10]:
            print(f"  {error}")
    
    print(f"\n✓ Filtered images saved to: {OUTPUT_DIR}")
    print(f"  Space used: ~{copied * 0.1:.1f} MB (estimated)")

if __name__ == "__main__":
    # Check if CheXpert dataset exists
    if not os.path.exists(CHEXPERT_BASE):
        print(f"⚠ CheXpert dataset not found at: {CHEXPERT_BASE}")
        print(f"\nPlease download CheXpert first:")
        print(f"1. Go to: https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7aea2")
        print(f"2. Request access and download CheXpert-v1.0-small.zip")
        print(f"3. Extract to: C:/xray_data/chexpert/")
        print(f"4. Run this script again")
    elif not os.path.exists(FILTERED_CSV):
        print(f"⚠ Filtered CSV not found at: {FILTERED_CSV}")
        print(f"Please run the CheXpert analysis cell in your notebook first!")
    else:
        extract_filtered_images()
