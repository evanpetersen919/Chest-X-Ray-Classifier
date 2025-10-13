# NIH Chest X-Ray Disease Classification

A deep learning project for multi-class classification of chest X-ray images using PyTorch and transfer learning.

## Project Overview

This project implements a ResNet50-based classifier to identify 15 different chest pathologies from the NIH Chest X-Ray dataset. The model achieves 37% validation accuracy, demonstrating the challenges of fine-grained medical image classification.

## Dataset

- **Source**: NIH Chest X-Ray Dataset
- **Total Images**: 31,416 chest X-rays
- **Classes**: 15 disease categories
- **Split**: 70% train, 15% validation, 15% test

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
git clone <your-repo-url>
cd xray-classifier
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

- **Architecture**: ResNet50 (pretrained on ImageNet)
- **Batch Size**: 128
- **Epochs**: 20
- **Optimizer**: Adam (lr=0.001)
- **Scheduler**: ReduceLROnPlateau
- **Data Augmentation**: Random horizontal flip, rotation, color jitter

## Results

- **Validation Accuracy**: 37%
- **Test Accuracy**: ~37%
- **Model File**: `best_resnet50.pth`

The confusion matrix shows the model performs best on common findings like "No Finding" and "Effusion", while struggling with rarer conditions.

## Performance Notes

For optimal training speed:
- Use an SSD for data storage
- Avoid OneDrive or network drives
- Enable GPU acceleration
- Consider increasing batch size if GPU memory allows

## Future Improvements

- Train for more epochs (50-100) for better convergence
- Implement class balancing techniques
- Try other architectures (EfficientNet, DenseNet)
- Add more aggressive data augmentation
- Ensemble multiple models

## License

This project is for educational purposes. The NIH Chest X-Ray dataset is publicly available.

## Acknowledgments

- NIH Clinical Center for the Chest X-Ray dataset
- PyTorch team for the deep learning framework
