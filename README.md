# NIH Chest X-Ray Disease Classification

A deep learning project for multi-label classification of chest X-ray images using PyTorch and transfer learning.

## Project Overview

This project implements an EfficientNet-B3 classifier to identify 15 different chest pathologies from the NIH Chest X-Ray dataset. Using multi-label classification with mixed precision training, Mixup augmentation, and cosine annealing scheduling, the model targets 75-80% accuracy with 0.85-0.88 AUC-ROC.

## Dataset

- **Source**: NIH Chest X-Ray Dataset
- **Total Images**: 40,000 chest X-rays (sampled from 112,120)
- **Classification Type**: Multi-label (images can have multiple diseases)
- **Classes**: 15 disease categories
- **Split**: 70% train (28,000), 15% validation (6,000), 15% test (6,000)

### Disease Classes
Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, No Finding, Nodule, Pleural Thickening, Pneumonia, Pneumothorax

## Project Structure

```
.
├── data.ipynb              # Data preprocessing and organization
├── classifier.ipynb        # Model training and evaluation
├── data/                   # Dataset directory (not included)
│   ├── train/
│   ├── val/
│   └── test/
├── requirements.txt        # Python dependencies
└── README.md
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
pip install -r requirements.txt
```

3. Prepare the dataset
   - Run `data.ipynb` to preprocess and organize the NIH dataset
   - Ensure data is organized in `data/train/`, `data/val/`, `data/test/`

## Usage

### Training

Open and run `classifier.ipynb` to train the model:

1. Load data and apply transformations
2. Initialize ResNet50 model
3. Train for 20 epochs with validation
4. Evaluate on test set
5. Generate confusion matrix
6. Test inference on sample images

### Model Configuration

- **Architecture**: EfficientNet-B3 (pretrained on ImageNet, 12M parameters)
- **Loss Function**: BCEWithLogitsLoss (for multi-label classification)
- **Batch Size**: 192
- **Epochs**: 50
- **Optimizer**: AdamW (lr=0.001, weight_decay=0.01)
- **Scheduler**: CosineAnnealingWarmRestarts (T_0=10, T_mult=2)
- **Mixed Precision**: Enabled (AMP with GradScaler)
- **Mixup Augmentation**: alpha=0.2
- **Data Augmentation**: Random horizontal flip, rotation, affine, color jitter, random erasing
- **Image Size**: 300x300 (optimal for EfficientNet-B3)

## Results

- **Target Accuracy**: 75-80% (multi-label)
- **Target AUC-ROC**: 0.85-0.88 (mean across 15 classes)
- **Training Time**: ~8.5 hours (50 epochs with mixed precision)
- **Model File**: `best_efficientnet_b3.pth`

### Model Performance

Multi-label classification allows the model to predict multiple diseases per image, which is more realistic for chest X-ray diagnosis where patients often have co-occurring conditions.

**Key Features:**
- Per-label accuracy tracking for each of 15 disease categories
- AUC-ROC metric for robust multi-label evaluation
- Mixed precision training for 30% faster training and higher batch sizes
- Mixup augmentation to improve generalization
- Cosine annealing with warm restarts for better convergence

### Metrics

The model tracks:
- **Per-Label Accuracy**: Binary accuracy for each disease label
- **Mean AUC-ROC**: Area under ROC curve averaged across all 15 classes
- **Training Loss**: BCEWithLogitsLoss with Mixup augmentation
- **Learning Rate**: Tracked per epoch with cosine annealing schedule

## Performance Notes

For optimal training speed:
- Use an SSD for data storage
- Avoid OneDrive or network drives
- Enable GPU acceleration
- Consider increasing batch size if GPU memory allows

## Future Improvements

- Test-time augmentation (+1-3% accuracy improvement)
- Gradual unfreezing of backbone layers (+2-3% accuracy)
- Dropout regularization for better generalization (+1-2% accuracy)
- Ensemble multiple models with different architectures
- Experiment with focal loss for class imbalance
- Implement attention mechanisms for interpretability

## License

This project is for educational purposes. The NIH Chest X-Ray dataset is publicly available.

## Acknowledgments

- NIH Clinical Center for the Chest X-Ray dataset
- PyTorch team for the deep learning framework
