# Chest X-Ray Disease Classifier

Multi-label classification of chest diseases from X-ray images using deep learning.

## Overview

This project implements an EfficientNet-B3 model for multi-label classification of 7 chest diseases from the NIH Chest X-Ray dataset. The model achieves **47.80% exact match accuracy** on patient-aware test splits.

## Dataset

- **Source**: NIH Clinical Center Chest X-Ray Dataset
- **Size**: ~28,000 images (patient-aware splits)
- **Split**: 70% train / 15% validation / 15% test
- **Classification Type**: Multi-label (each image can have multiple diseases)

### Disease Classes (7 Total)

1. Atelectasis
2. Effusion
3. Infiltration
4. Mass
5. No Finding
6. Nodule
7. Pneumothorax

**Excluded Diseases** (insufficient training data):
- Cardiomegaly (664 samples)
- Consolidation (1,204 samples)
- Pleural_Thickening (160 samples)

## Project Structure

```
xray classifier/
├── data.ipynb                  # Dataset preparation and processing
├── classifier.ipynb            # Model training and evaluation
├── README.md                   # Project documentation
└── best_efficientnet_b3.pth    # Trained model weights
```

## Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)

### Installation

1. Clone the repository
```bash
git clone https://github.com/evanpetersen919/Chest-X-Ray-Classifier.git
cd Chest-X-Ray-Classifier
```

2. Install dependencies
```bash
pip install torch torchvision timm pandas numpy scikit-learn matplotlib seaborn pillow tqdm
```

3. Prepare the dataset
   - Run `data.ipynb` to prepare and organize the NIH dataset
   - Creates patient-aware train/val/test splits
   - Saves to `C:/xray_data/data/`

## Usage

### Training

Run `classifier.ipynb` to train the model:

1. Load data with augmentation transforms
2. Initialize EfficientNet-B3 with class weights
3. Train with Mixup augmentation and early stopping
4. Evaluate on test set
5. Generate per-class performance visualizations
6. Test inference on sample images

### Model Configuration

- **Architecture**: EfficientNet-B3 (pretrained on ImageNet)
  - Parameters: ~10.7M
  - Input Size: 300×300
- **Loss Function**: BCEWithLogitsLoss with class weights [1.88-4.59]
- **Training**:
  - Batch Size: 32
  - Epochs: 20 (with early stopping, patience=5)
  - Optimizer: AdamW (lr=0.001, weight_decay=0.01)
  - Scheduler: CosineAnnealingWarmRestarts (T_0=10, T_mult=2)
  - Mixed Precision: Enabled (AMP)
- **Data Augmentation**:
  - Mixup (alpha=0.2)
  - Horizontal flip
  - Rotation (±10°)
  - Affine transforms
  - Color jitter
  - Random erasing

## Performance

### Overall Metrics
- **Exact Match Accuracy**: 47.80%
- **Hamming Accuracy**: 87.35%
- **Average F1-Score**: 0.347
- **AUC-ROC**: 0.7847

### Per-Class Performance

| Disease | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Atelectasis | 0.366 | 0.340 | 0.353 | 653 |
| Effusion | 0.598 | 0.314 | 0.412 | 668 |
| Infiltration | 0.450 | 0.087 | 0.145 | 1049 |
| Mass | 0.388 | 0.277 | 0.324 | 339 |
| No Finding | 0.706 | 0.789 | 0.745 | 3225 |
| Nodule | 0.265 | 0.043 | 0.074 | 302 |
| Pneumothorax | 0.392 | 0.362 | 0.376 | 282 |

### Key Features

- **Patient-Aware Splits**: Prevents data leakage by ensuring no patient appears in multiple splits
- **Class Weighting**: Handles imbalance (21.4:1 ratio) with moderate weights instead of oversampling
- **Multi-Label Classification**: Predicts multiple diseases per image
- **Mixed Precision Training**: 30% faster training with AMP
- **Early Stopping**: Prevents overfitting (patience=5)

## Design Decisions

### Why 7 Classes?

The model focuses on 7 diseases with adequate training data (≥1,500 samples each). This "quality over quantity" approach ensures robust learning for each class.

**Three diseases were excluded** due to insufficient training samples:
- Cardiomegaly: 664 samples (too few for reliable training)
- Consolidation: 1,204 samples (borderline insufficient)
- Pleural_Thickening: 160 samples (severely underrepresented)

Training models on these rare classes without sufficient data would lead to overfitting and poor generalization.

### Why Class Weights Instead of Oversampling?

The model uses moderate class weights (1.88-4.59) rather than oversampling to handle the 21.4:1 class imbalance (No Finding vs. rarest diseases). 

**Why not oversample?**
- Duplicating multi-label images corrupts natural disease co-occurrence patterns
- Rare diseases often co-occur with common ones (e.g., Effusion + Atelectasis)
- Oversampling creates artificial bias in these correlations
- Class weights preserve the original distribution while improving rare disease detection

## Future Improvements

- Optimize per-class prediction thresholds
- Experiment with ensemble methods
- Add external datasets (CheXpert) for rare diseases
- Implement attention visualization for interpretability
- Deploy as REST API or web service

## License

This project is for educational purposes. The NIH Chest X-Ray dataset is publicly available.

## Acknowledgments

- NIH Clinical Center for the Chest X-Ray dataset
- PyTorch team for the deep learning framework
