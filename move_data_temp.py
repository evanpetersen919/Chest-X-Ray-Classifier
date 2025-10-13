"""
Move data folder to C:\temp for faster training
Run this BEFORE training to fix GPU utilization
"""
import shutil
import os

source = r"c:\Users\evanp\OneDrive\Documents\vs code\xray classifier\data"
destination = r"C:\temp\xray_data"

print(f"Moving data from:\n  {source}")
print(f"To:\n  {destination}")
print("\nThis will take ~2-3 minutes...")

# Create temp directory
os.makedirs(destination, exist_ok=True)

# Move the data
shutil.copytree(source, destination, dirs_exist_ok=True)

print(f"\n✅ Data moved successfully!")
print(f"\nNow update your notebook Cell 3 to use:")
print(f"  train_dataset = ChestXrayDataset(root_dir='C:/temp/xray_data/train', transform=transform)")
print(f"  val_dataset = ChestXrayDataset(root_dir='C:/temp/xray_data/val', transform=transform)")
print(f"  test_dataset = ChestXrayDataset(root_dir='C:/temp/xray_data/test', transform=transform)")
