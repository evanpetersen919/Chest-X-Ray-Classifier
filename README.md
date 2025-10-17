# Chest X-Ray Disease Classifier

Multi-label classification of chest diseases from X-ray images using deep learning.

## Overview

This project implements a ResNet50 model for multi-label classification of 7 chest diseases from the NIH Chest X-Ray dataset. The model achieves **F1 score of 0.442** with optimized per-class thresholds on patient-aware test splits.

## Dataset

- **Source**: NIH Clinical Center Chest X-Ray Dataset
- **Size**: ~28,000 images (patient-aware splits)
- **Split**: 70% train / 15% validation / 15% test
- **Classification Type**: Multi-label (each image can have multiple diseases)

### Disease Classes (7 of 15 Original)

The NIH dataset contains 15 disease findings. This project focuses on 7 diseases with sufficient training data (≥1,500 samples):

1. Atelectasis
2. Effusion
3. Infiltration
4. Mass
5. No Finding
6. Nodule
7. Pneumothorax

**8 Diseases Excluded** (insufficient training data):
- Cardiomegaly (664 samples)
- Consolidation (1,204 samples)
- Pleural_Thickening (160 samples)
- Edema, Emphysema, Fibrosis, Hernia, Pneumonia (all <1,500 samples)

## Project Structure

```
Chest-X-Ray-Classifier/
├── app.py               # Streamlit web application
├── data.ipynb           # Dataset preparation and processing
├── classifier.ipynb     # Model training and evaluation
├── samples/             # Sample X-ray images for testing
├── requirements.txt     # Python dependencies
└── README.md           # Project documentation
```

**Note**: Data and model weights are stored locally and not tracked in git. Sample images are provided in the `samples/` folder for testing the application.

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
pip install -r requirements.txt
```

3. Download the NIH Chest X-Ray dataset
   - [NIH Clinical Center Dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC)
   - Extract to a local directory (e.g., `C:/xray_data/`)

4. Prepare the dataset
   - Run `data.ipynb` to prepare and organize the dataset
   - Creates patient-aware train/val/test splits (no data leakage)
   - Generates metadata CSV files with labels

## Usage

### Web Application Demo

Run the Streamlit web app for interactive predictions:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`. Upload a chest X-ray image to get instant disease predictions with probability scores and visualizations.

**Quick Test:** Sample X-ray images are provided in the `samples/` folder. Try uploading these to see the model in action.

### Training

Run `classifier.ipynb` to train the model:

1. Load data with augmentation transforms
2. Initialize ResNet50 with class weights
3. Train with Mixup augmentation and early stopping
4. Optimize per-class prediction thresholds
5. Evaluate on test set with optimal thresholds
6. Visualize performance and test inference

### Model Configuration

- **Architecture**: ResNet50 (pretrained on ImageNet)
  - Parameters: 23.5M
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
- **Threshold Optimization**:
  - Per-class thresholds tuned on validation set
  - Optimizes F1-score for each disease independently

## Performance

### Overall Metrics (with Optimal Thresholds)
- **Average F1-Score**: 0.442
- **Exact Match Accuracy**: 40.57%
- **Hamming Accuracy**: 83.36%
- **AUC-ROC**: 0.7934
- **Improvement**: +27.5% over baseline (0.5 threshold)

### Per-Class F1 Scores

| Disease | Default (0.5) | Optimal Threshold | Optimized F1 |
|---------|---------------|-------------------|--------------|
| Atelectasis | 0.311 | 0.35 | 0.407 |
| Effusion | 0.456 | 0.35 | 0.536 |
| Infiltration | 0.205 | 0.30 | 0.291 |
| Mass | 0.323 | 0.35 | 0.393 |
| No Finding | 0.736 | 0.45 | 0.739 |
| Nodule | 0.078 | 0.20 | 0.201 |
| Pneumothorax | 0.318 | 0.50 | 0.527 |

### Key Features

- **Patient-Aware Splits**: Prevents data leakage by ensuring no patient appears in multiple splits
- **Class Weighting**: Handles imbalance (21.4:1 ratio) with moderate weights instead of oversampling
- **Multi-Label Classification**: Predicts multiple diseases per image
- **Mixed Precision Training**: 30% faster training with AMP
- **Early Stopping**: Prevents overfitting (patience=5)

## Design Decisions

### Why 7 of 15 Classes?

The NIH dataset contains 15 disease findings, but this project focuses on the 7 diseases with adequate training data (≥1,500 samples each). This "quality over quantity" approach ensures robust learning for each class.

**8 diseases were excluded** due to insufficient training samples (<1,500 each):
- Cardiomegaly: 664 samples
- Consolidation: 1,204 samples  
- Pleural_Thickening: 160 samples
- Edema, Emphysema, Fibrosis, Hernia, Pneumonia: <1,500 samples each

Training models on these rare classes without sufficient data would lead to overfitting and poor generalization. The 7 selected diseases represent the most common and well-represented conditions in the dataset.

### Why Class Weights Instead of Oversampling?

The model uses moderate class weights (1.88-4.59) rather than oversampling to handle the 21.4:1 class imbalance (No Finding vs. rarest diseases). 

**Why not oversample?**
- Duplicating multi-label images corrupts natural disease co-occurrence patterns
- Rare diseases often co-occur with common ones (e.g., Effusion + Atelectasis)
- Oversampling creates artificial bias in these correlations
- Class weights preserve the original distribution while improving rare disease detection

## Future Improvements

- Use full dataset (~112k images) instead of 40k subset
- Experiment with larger models (ResNet101, EfficientNet-B4)
- Implement Segmentation
- Add external datasets (CheXpert) for improved generalization
- Implement attention visualization for interpretability
- Deploy as REST API or web service

## License

This project is for educational purposes. The NIH Chest X-Ray dataset is publicly available.

## Acknowledgments

- NIH Clinical Center for the Chest X-Ray dataset
- PyTorch team for the deep learning framework
