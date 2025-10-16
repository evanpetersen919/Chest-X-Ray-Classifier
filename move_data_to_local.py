"""
Move NIH Chest X-Ray data from OneDrive to local SSD for faster training
"""

import shutil
import os
from pathlib import Path

# Source (current OneDrive location)
SOURCE_BASE = Path(r'c:\Users\evanp\OneDrive\Documents\vs code\xray classifier')
SOURCE_ARCHIVE = SOURCE_BASE / 'archive (1)'
SOURCE_DATA = SOURCE_BASE / 'data'

# Destination (local C: drive, outside OneDrive)
DEST_BASE = Path(r'C:\xray_data')

print("=" * 80)
print("Moving NIH Chest X-Ray Data to Local SSD")
print("=" * 80)
print(f"\nSource: {SOURCE_BASE}")
print(f"Destination: {DEST_BASE}")

# Create destination directory
DEST_BASE.mkdir(parents=True, exist_ok=True)
print(f"\n✓ Created destination: {DEST_BASE}")

# Move archive folder (raw data)
if SOURCE_ARCHIVE.exists():
    print(f"\nMoving archive folder...")
    dest_archive = DEST_BASE / 'archive (1)'
    if dest_archive.exists():
        print(f"  Archive already exists at destination, skipping...")
    else:
        shutil.move(str(SOURCE_ARCHIVE), str(dest_archive))
        print(f"  ✓ Moved archive to: {dest_archive}")
else:
    print(f"\n⚠ Archive folder not found at: {SOURCE_ARCHIVE}")

# Move processed data folder (train/val/test splits)
if SOURCE_DATA.exists():
    print(f"\nMoving processed data folder...")
    dest_data = DEST_BASE / 'data'
    if dest_data.exists():
        print(f"  Data folder already exists at destination")
        # Check what's inside
        train_count = len(list((dest_data / 'train').glob('*.png'))) if (dest_data / 'train').exists() else 0
        val_count = len(list((dest_data / 'val').glob('*.png'))) if (dest_data / 'val').exists() else 0
        test_count = len(list((dest_data / 'test').glob('*.png'))) if (dest_data / 'test').exists() else 0
        print(f"  Found: {train_count:,} train, {val_count:,} val, {test_count:,} test images")
    else:
        shutil.move(str(SOURCE_DATA), str(dest_data))
        print(f"  ✓ Moved data to: {dest_data}")
else:
    print(f"\n⚠ Data folder not found at: {SOURCE_DATA}")
    print(f"  You may need to run data.ipynb first to create processed data")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

if DEST_BASE.exists():
    # Count files
    archive_exists = (DEST_BASE / 'archive (1)').exists()
    data_exists = (DEST_BASE / 'data').exists()
    
    print(f"\nDestination: {DEST_BASE}")
    print(f"  Archive folder: {'✓ Present' if archive_exists else '✗ Missing'}")
    print(f"  Data folder: {'✓ Present' if data_exists else '✗ Missing'}")
    
    if data_exists:
        dest_data = DEST_BASE / 'data'
        train_count = len(list((dest_data / 'train').glob('*.png'))) if (dest_data / 'train').exists() else 0
        val_count = len(list((dest_data / 'val').glob('*.png'))) if (dest_data / 'val').exists() else 0
        test_count = len(list((dest_data / 'test').glob('*.png'))) if (dest_data / 'test').exists() else 0
        
        print(f"\n  Processed Images:")
        print(f"    Train: {train_count:,}")
        print(f"    Val:   {val_count:,}")
        print(f"    Test:  {test_count:,}")
        print(f"    Total: {train_count + val_count + test_count:,}")

print("\n" + "=" * 80)
print("✓ Data migration complete!")
print("\nNext steps:")
print("  1. Update data.ipynb paths to use C:\\xray_data")
print("  2. classifier.ipynb DATA_DIR is already set to 'C:/xray_data'")
print("=" * 80)
